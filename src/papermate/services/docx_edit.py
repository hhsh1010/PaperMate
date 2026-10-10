"""PaperMate · 改稿直写 docx（C2·P1）：段落可寻址 + track-changes 写入 + quote 校验阀。

设计要点（docs/改稿直写docx_子方案.md · docs/工作间与改稿增强_落地计划.md C2）：
- **段落视图**：重开 ``Document(path)`` 取真实段落序列线性编号 ``P{0..n}``（不复用 RAG chunk——
  ``DocxParser.parse`` 会合并段落进 Section，无段落索引）。
- **track-changes**：只插入 ``<w:ins>`` / ``<w:del>`` 修订标记，原 run 转为 ``<w:delText>`` **保留在原位**，
  保格式（样式/字体/表格/图片原样）、可逆（Word 中可拒绝修订）、零破坏。
- **quote 校验阀**：edit op 携带的 ``quote`` 若不在目标段落文本中 → 该 op 记入 ``rejected``；
  **宁可拒不可错**，杜绝改错段落。
- **round-trip 保真**：先落临时文件 → ``zipfile.testzip`` + 重开验证 → ``os.replace`` 原子替换。
- P1 边界：正文段落 replace / append / insert_after；表格/列表/delete/numPr 复制属 C3；
  含域代码（TOC/题注）或媒体（图/表对象）的段落一律 reject，避免破坏结构。

本模块为纯文件操作（无 DB 依赖），存储路径由 DocumentService 提供。
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

logger = logging.getLogger(__name__)

_EDIT_OPS = ("replace", "append", "insert_after", "delete", "purge")
_AUTHOR = "PaperMate"
#: 每个文档保留的 .bak 备份上限（超出删最旧）
_MAX_BACKUPS = 5
#: 表格段落地址：T{table}.R{row}.C{cell}.P{para}
_T_ADDR = re.compile(r"^T(\d+)\.R(\d+)\.C(\d+)\.P(\d+)$")
#: 落盘重试参数：Windows 下目标文件可能被瞬时占用（WinError 32 共享冲突——
#: Word/WPS 开着文档、预览/下载流、杀软/索引服务），等待重试通常可自愈
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_S = 0.15


def _locked_file_error(target: Path) -> DocxEditError:
    """构造「文件被占用」的友好业务错误（WinError 32/5 → 400）。"""
    return DocxEditError(
        f"文件正被其他程序占用，无法写入：{target.name}。"
        "最常见原因：文档正在 Word/WPS 中打开，或刚触发的预览/下载还未结束。"
        "请关闭占用它的程序后重试。"
    )


def _replace_with_retry(tmp: Path, target: Path) -> None:
    """原子替换 + 占用重试：``os.replace`` 遇共享冲突（PermissionError）按退避重试。

    瞬时锁（预览流/杀软扫描/索引）在数百毫秒内自愈；若持续被锁（如 Word 一直开着），
    重试耗尽后抛带明确指引的业务错误，绝不静默丢改动（tmp 由调用方清理）。
    """
    last_exc: PermissionError | None = None
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, target)
            return
        except PermissionError as e:  # WinError 32/5 均归 PermissionError
            last_exc = e
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
    raise _locked_file_error(target) from last_exc


def _copy_with_retry(src: Path, dst: Path) -> None:
    """向（可能被占用的）目标路径写入的重试版 copy，语义同 :func:`_replace_with_retry`。"""
    last_exc: PermissionError | None = None
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError as e:
            last_exc = e
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
    raise _locked_file_error(dst) from last_exc


class DocxEditError(Exception):
    """改稿直写业务错误基类；路由层按子类映射状态码。"""


class DocxNotFound(DocxEditError):
    """文档不存在或文件丢失（→ 404）。"""


class DocxNotEditable(DocxEditError):
    """非 .docx 文件或文件损坏（→ 400）。"""


# ---------- 段落视图（元素级：lxml element 代理均可） ----------

def _as_el(obj: Any) -> Any:
    """兼容入参：python-docx 对象（Paragraph/Table 等）取 ``._p``，lxml 元素原样返回。"""
    return obj._p if hasattr(obj, "_p") else obj


def _has_field(p: Any) -> bool:
    """段落是否含域代码（TOC/页码/题注等）——此类段落跳过编辑，避免破坏域。"""
    el = _as_el(p)
    return bool(el.findall(".//" + qn("w:fldChar")) or el.findall(".//" + qn("w:instrText")))


def _has_media(p: Any) -> bool:
    """段落是否含图片/嵌入对象——replace 此类段落会破坏内容，一律 reject。"""
    el = _as_el(p)
    return bool(
        el.findall(".//" + qn("w:drawing"))
        or el.findall(".//" + qn("w:object"))
        or el.findall(".//" + qn("w:pict"))
    )


def _para_text(p: Any) -> str:
    """段落「有效文本」：收集全部 ``<w:t>``（含 ``<w:ins>`` 内新增文字）、排除 ``<w:delText>``（已删旧文字）。

    语义 = 「接受全部修订后」的段落文本。注意 python-docx 自带 ``p.text`` 不读
    ``<w:ins>`` 内容，直接用会导致应用修改后视图看不到新文字。
    """
    el = _as_el(p)
    parts: list[str] = []
    for node in el.iter():
        if node.tag == qn("w:t"):
            parts.append(node.text or "")
    return "".join(parts)


def _entry(pid: str, el: Any, style: str | None = None) -> dict:
    text = _para_text(el)
    field, media = _has_field(el), _has_media(el)
    reason = "field" if field else ("media" if media else None)
    # 段落标记是否被标记删除（pPr/rPr/w:del）——标记删除后的「空壳段」识别依据，
    # 前端据此展示「已删除」徽标与「彻底删除」入口（跨刷新稳定，不依赖前端状态）。
    deleted = el.find(f"{qn('w:pPr')}/{qn('w:rPr')}/{qn('w:del')}") is not None
    return {
        "pid": pid,
        "style": style,
        "text": text,
        "empty": not text.strip(),
        "editable": bool(text.strip()) and field is False and media is False,
        "reason": reason,
        "deleted": deleted,
    }


def _table_paragraphs(doc: Document) -> list[tuple[str, Any]]:
    """表格内段落（C3）：``T{ti}.R{ri}.C{ci}.P{pi}`` 地址 → 段落对象。

    合并单元格会在 ``row.cells`` 重复出现同一 ``_tc``——按 tc 去重，避免同段多地址。
    不递归嵌套表格（嵌套表格段落不编址，属后续增强）。
    """
    out: list[tuple[str, Any]] = []
    seen_tc: set[int] = set()
    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                if id(cell._tc) in seen_tc:
                    continue
                seen_tc.add(id(cell._tc))
                for pi, p in enumerate(cell.paragraphs):
                    out.append((f"T{ti}.R{ri}.C{ci}.P{pi}", p))
    return out


def list_paragraphs(path: Path) -> list[dict]:
    """段落视图：正文 ``P{i}`` + 表格 ``T..R..C..P{pi}``（C3），附可编辑判定。"""
    doc = Document(str(path))
    out: list[dict] = []
    for i, p in enumerate(doc.paragraphs):
        out.append(_entry(f"P{i}", p._p, p.style.name if p.style is not None else None))
    for pid, p in _table_paragraphs(doc):
        out.append(_entry(pid, p._p))
    return out


# ---------- 文档流视图（C4·P1：高保真渲染模式的编址） ----------

def _stream_elements(doc: Document) -> list[Any]:
    """body 文档流元素（**排除 sectPr** —— docx-preview 不渲染它，编址必须与渲染顺序一致）。"""
    return [el for el in doc.element.body if el.tag != qn("w:sectPr")]


def list_stream(path: Path) -> list[dict]:
    """文档流视图：按 body 实际顺序输出段落与表格，供高保真渲染层按序对齐。

    - 段落 → ``{"kind": "para", "pid": "B{i}", ...}``；
    - 表格 → ``{"kind": "table", "pid": "B{i}", "cells": [{"pid": "B{i}.R{ri}.C{ci}.P{pi}", ...}]}``。
    表格单元格在 XML 层遍历（每个 ``w:tc`` 天然唯一，合并单元格无重复地址）。
    """
    doc = Document(str(path))
    out: list[dict] = []
    for i, el in enumerate(_stream_elements(doc)):
        if el.tag == qn("w:p"):
            out.append({"kind": "para", "pid": f"B{i}", **_entry(f"B{i}", el)})
        elif el.tag == qn("w:tbl"):
            cells: list[dict] = []
            for ri, tr in enumerate(el.findall(qn("w:tr"))):
                for ci, tc in enumerate(tr.findall(qn("w:tc"))):
                    for pi, p in enumerate(tc.findall(qn("w:p"))):
                        cells.append(_entry(f"B{i}.R{ri}.C{ci}.P{pi}", p))
            out.append({"kind": "table", "pid": f"B{i}", "cells": cells})
    return out


# ---------- track-changes 构造 ----------

def _max_revision_id(doc: Document) -> int:
    """扫描现有 <w:ins>/<w:del> 的最大 w:id，避免与既有修订冲突。"""
    mx = 0
    for el in doc.element.body.iter():
        if el.tag in (qn("w:ins"), qn("w:del")):
            v = el.get(qn("w:id"))
            if v and v.lstrip("-").isdigit():
                mx = max(mx, int(v))
    return mx


def _run_visible_text(r: Any) -> str:
    """单个 ``w:r`` 元素的可见文本（``w:t`` 与 ``w:delText`` 拼接，含全部后代）。"""
    return "".join((t.text or "") for t in r.iter() if t.tag in (qn("w:t"), qn("w:delText")))


def _new_ins_run(text: str, author: str, ts: str, rev_id: int, rpr_src: Any | None = None) -> Any:
    """构造 <w:ins> 修订元素，内含一个新 run（插入文字）。

    N15（2026-09-12）：新增 ``rpr_src``——传入一个原 ``w:r`` 元素时，**深拷贝其 <w:rPr>**
    到新 run，使插入文字继承原字符格式（字体/字号/加粗等）。此前不带格式 → 整段替换后
    新文字退化为默认字体，用户实测「改一段文字整段都变格式」。
    ★ 2026-10-10：``rpr_src`` 也可直接传 ``w:rPr`` 元素（如段落的 ``w:pPr/w:rPr`` 段落标记
    格式，供"段尾追加"继承）——按 tag 判别，两种约定并存。
    """
    ins = OxmlElement("w:ins")
    ins.set(qn("w:id"), str(rev_id))
    ins.set(qn("w:author"), author)
    ins.set(qn("w:date"), ts)
    r = OxmlElement("w:r")
    if rpr_src is not None:
        src_rpr = rpr_src if rpr_src.tag == qn("w:rPr") else rpr_src.find(qn("w:rPr"))
        if src_rpr is not None:
            import copy

            r.append(copy.deepcopy(src_rpr))
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    ins.append(r)
    return ins


def _wrap_del_run(r: Any, author: str, ts: str, rev_id: int) -> Any:
    """把已存在的 <w:r> 包进 <w:del>：其 <w:t> 就地转为 <w:delText>（保留删除痕迹）。"""
    for t in r.findall(qn("w:t")):
        t.tag = qn("w:delText")
    wdel = OxmlElement("w:del")
    wdel.set(qn("w:id"), str(rev_id))
    wdel.set(qn("w:author"), author)
    wdel.set(qn("w:date"), ts)
    parent = r.getparent()
    if parent is None:  # 悬空 run（异常结构）：不包，原样返回
        return r
    parent.replace(r, wdel)
    wdel.append(r)
    return wdel


# ---------- 应用编辑 ----------

_B_ADDR = re.compile(r"^B(\d+)(?:\.R(\d+)\.C(\d+)\.P(\d+))?$")


def _locate(doc: Document, pid: str) -> Any | None:
    """按地址定位段落元素；非法/越界 → None。

    - ``P{i}``：正文段落；``T{ti}.R{ri}.C{ci}.P{pi}``：表格内段落（C3）；
    - ``B{i}`` / ``B{i}.R{ri}.C{ci}.P{pi}``：文档流地址（C4·P1，高保真模式）。
      ``B{i}}`` 指向 body 第 i 个非 sectPr 子元素（须为 w:p）；
      表格段落在 XML 层直接寻址（每个 w:tc 唯一，与 list_stream 编址一致）。

    返回轻量代理（仅保证 ``._p`` 可用）——所有写操作均经 ``._p``，与对象类型无关。
    """
    from types import SimpleNamespace

    if pid.startswith("P") and pid[1:].isdigit():
        idx = int(pid[1:])
        paras = doc.paragraphs
        return paras[idx] if 0 <= idx < len(paras) else None
    m = _T_ADDR.match(pid)
    if m:
        ti, ri, ci, pi = (int(g) for g in m.groups())
        try:
            return doc.tables[ti].rows[ri].cells[ci].paragraphs[pi]
        except IndexError:
            return None
    mb = _B_ADDR.match(pid)
    if mb:
        idx = int(mb.group(1))
        els = _stream_elements(doc)
        if idx >= len(els):
            return None
        el = els[idx]
        if mb.group(2) is None:
            if el.tag != qn("w:p"):
                return None
            return SimpleNamespace(_p=el)
        ri, ci, pi = int(mb.group(2)), int(mb.group(3)), int(mb.group(4))
        try:
            trs = el.findall(qn("w:tr"))
            tcs = trs[ri].findall(qn("w:tc"))
            ps = tcs[ci].findall(qn("w:p"))
            return SimpleNamespace(_p=ps[pi])
        except IndexError:
            return None
    return None


def _mark_para_deleted(p: Any, author: str, ts: str, rev_id: int) -> None:
    """段落标记删除（C3）：``pPr/rPr/<w:del>`` —— Word 中整段显示删除线，
    接受修订后段落随段落标记一起消失；拒绝修订则整段恢复。"""
    pPr = p._p.get_or_add_pPr()
    rPr = pPr.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        pPr.append(rPr)
    del_el = OxmlElement("w:del")
    del_el.set(qn("w:id"), str(rev_id))
    del_el.set(qn("w:author"), author)
    del_el.set(qn("w:date"), ts)
    rPr.append(del_el)


def _inside_table_cell(el: Any) -> bool:
    """元素是否位于表格单元格（``w:tc``）内——purge 物理移除仅限正文段落，防止破坏单元格结构。"""
    anc = el.getparent()
    while anc is not None:
        if anc.tag == qn("w:tc"):
            return True
        anc = anc.getparent()
    return False


def _runs_pending_del(p_el: Any) -> list[Any]:
    """段内需要标记删除的 run：``w:r`` 的**全部后代**，排除已处于 ``w:del`` 之下的（旧文字已是 delText）。

    不能用 ``findall(qn("w:r"))``——它只取直接子级 run，会漏掉三类常见文字：
    AI 改写/追加留下的 ``w:ins`` 嵌套 run、真实文档的 ``w:hyperlink``（链接/引用）、
    内联 ``w:sdt``/``w:smartTag``。漏标的表现正是「徽标已删除，文字却还在」。
    """
    out: list[Any] = []
    for r in p_el.iter(qn("w:r")):
        anc = r.getparent()
        inside_del = False
        while anc is not None:
            if anc.tag == qn("w:del"):
                inside_del = True
                break
            anc = anc.getparent()
        if not inside_del:
            out.append(r)
    return out


def _replace_preserving_runs(p_el: Any, text: str, author: str, ts: str, base_id: int) -> bool:
    """N15 增强（2026-09-12）：**逐 run 精确保留格式**的 replace。

    仅当段内所有有效 run 都是 ``p_el`` 的**直接子级**（无 hyperlink / 既有 w:ins 嵌套）时启用；
    否则返回 ``False``，由调用方回退到整段替换（del 全部 + 单 ins 继承主格式）——安全阀，
    避免拆分克隆破坏复杂结构（嵌套 run 脱离容器会丢失链接等语义）。

    算法：
    1. 收集各 run 的可见文本与字符边界；
    2. ``SequenceMatcher(old, new)`` 得 opcodes；
    3. equal 段**保留原 run 原样**（部分覆盖时用裁剪克隆），replace/delete 段把旧文字
       以裁剪克隆包 ``<w:del>``（保留痕迹），replace/insert 的新文字以 ``<w:ins>`` 写入
       （继承该块第一个被覆盖 run / 前一字符所在 run 的格式）；
    4. 原位替换（★ 2026-10-10）：每个原 run **就地**替换为它的生成块（addprevious + 移除原 run），
       书签/域等非 run 节点位置一概不动 ⇒ 编辑点之外的文档结构零改变
       （旧实现把整条序列插回第一个 run 的位置 ⇒ 其后的 run 跨书签边界挪动）。
    """
    from difflib import SequenceMatcher

    runs = _runs_pending_del(p_el)
    if not runs:
        return False
    for r in runs:
        if r.getparent() is not p_el:
            return False  # 嵌套结构（hyperlink / 既有 w:ins 等）→ 回退整段替换
    texts = [_run_visible_text(r) for r in runs]
    old = "".join(texts)
    if not old:
        return False
    bnds: list[tuple[int, int]] = []
    acc = 0
    for t in texts:
        bnds.append((acc, acc + len(t)))
        acc += len(t)

    import copy as _copy

    def _slice_run(k: int, a: int, b: int) -> Any | None:
        """克隆 runs[k]，把其可见文本裁剪为 old 全局区间 [a,b)。空片段返回 None。"""
        piece = old[a:b]
        if not piece:
            return None
        clone = _copy.deepcopy(runs[k])
        for t in list(clone.findall(qn("w:t"))) + list(clone.findall(qn("w:delText"))):
            clone.remove(t)
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = piece
        rpr = clone.find(qn("w:rPr"))
        if rpr is not None:
            rpr.addnext(t)
        else:
            clone.insert(0, t)
        return clone

    def _runs_overlapping(a: int, b: int) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        for k, (s, e) in enumerate(bnds):
            if s < b and e > a:
                out.append((k, max(s, a), min(e, b)))
        return out

    # ★ 2026-10-10（用户实测：改稿后书签区内容跨边界挪动 + 插入文字继承上标格式）：
    #   生成序列改为**按 run 分块（chunks）**：每个原 run 对应一个替换块，重建时**就地**替换
    #   —— 段内非 run 节点（w:bookmarkStart/End、批注引用等）一概不动，编辑点之外零改变。
    #   （旧实现把整条 seq 插回"第一个 run 的位置" ⇒ 其后的 run 全部被拉到 bookmarkEnd 之前
    #     ⇒ 书签覆盖范围被改变——实测"中文作者单位"书签区被掏空。）
    chunks: list[list[Any]] = [[] for _ in runs]
    rev = base_id
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, text).get_opcodes():
        if tag == "equal":
            for k, s, e in _runs_overlapping(i1, i2):
                clone = _slice_run(k, s, e)
                if clone is not None:
                    chunks[k].append(clone)
        elif tag == "replace":
            ovl = _runs_overlapping(i1, i2)
            for k, s, e in ovl:  # 旧文字（该块）裁剪克隆包 <w:del>，保留痕迹
                dclone = _slice_run(k, s, e)
                if dclone is not None:
                    for t in dclone.findall(qn("w:t")):
                        t.tag = qn("w:delText")
                    wdel = OxmlElement("w:del")
                    wdel.set(qn("w:id"), str(rev))
                    wdel.set(qn("w:author"), author)
                    wdel.set(qn("w:date"), ts)
                    wdel.append(dclone)
                    chunks[k].append(wdel)
                    rev += 1
            new_piece = text[j1:j2]
            rpr_src = runs[ovl[-1][0]] if ovl else None
            chunks[ovl[-1][0]].append(_new_ins_run(new_piece, author, ts, rev, rpr_src=rpr_src))
            rev += 1
        elif tag == "insert":
            # rPr 继承规则（★ 2026-10-10 修后半条）：
            #   · 段中插入（i1 < len(old)）→ 继承**前一字符所在 run**（延续语义：拆分上标 run
            #     时新文字同为上标是 Word 的正确行为）；
            #   · **段落末尾追加**（i1 == len(old)）→ 继承**段落标记**格式（w:pPr/w:rPr）——
            #     旧实现继承最后一个 run，实测在上标序号"2"后追加作者名被写成上标（用户报告）。
            if i1 >= len(old):
                src = p_el.find(f"{qn('w:pPr')}/{qn('w:rPr')}")
                if src is None and runs:
                    src = runs[-1]
            else:
                src = None
                if i1 > 0:
                    for k, (s, e) in enumerate(bnds):
                        if s <= i1 - 1 < e:
                            src = runs[k]
                            break
                if src is None and runs:
                    src = runs[-1]
            k = 0 if i1 == 0 else next(kk for kk, (s, e) in enumerate(bnds) if s <= i1 - 1 < e)
            chunks[k].append(_new_ins_run(text[j1:j2], author, ts, rev, rpr_src=src))
            rev += 1
        elif tag == "delete":
            for k, s, e in _runs_overlapping(i1, i2):
                dclone = _slice_run(k, s, e)
                if dclone is not None:
                    for t in dclone.findall(qn("w:t")):
                        t.tag = qn("w:delText")
                    wdel = OxmlElement("w:del")
                    wdel.set(qn("w:id"), str(rev))
                    wdel.set(qn("w:author"), author)
                    wdel.set(qn("w:date"), ts)
                    wdel.append(dclone)
                    chunks[k].append(wdel)
                    rev += 1

    if not any(chunks):
        return False
    # ★ 原位替换（2026-10-10）：每个原 run 就地换成它的生成块（addprevious 后移除原 run）——
    #   书签/域等非 run 节点的位置一概不动 ⇒ 编辑点之外的文档结构零改变。
    for k, r in enumerate(runs):
        for el in chunks[k]:
            r.addprevious(el)
        r.getparent().remove(r)
    return True


def _apply_one(doc: Document, p: Any, op: str, text: str, author: str, ts: str, base_id: int) -> None:
    """对已通过校验的段落执行单个编辑操作（P2：正文 + 表格段落，四操作）。"""
    if op == "replace":
        # N15 增强（2026-09-12）：优先走**逐 run 精确保留**——未变化部分的格式与内容零改动，
        # 变化部分继承对应原 run 的格式；仅当段内含嵌套结构（hyperlink/既有 w:ins 等）
        # 时回退到下面的整段替换（del 全部 + 单 ins 继承主格式，安全阀）。
        if _replace_preserving_runs(p._p, text, author, ts, base_id):
            return
        # 整段替换（回退路径）：原 run（含 ins/hyperlink 等嵌套）就地包 <w:del>（保留删除线痕迹）；
        # 新文字以 <w:ins> 追加到段尾，并**继承主 run（文本最长者）的字符格式**。
        # N15：原先 `_new_ins_run` 不带 rPr → 整段替换后新文字退化为默认字体/字号，
        # 用户实测「改一段文字整段都变格式」。取主 run 的 <w:rPr> 深拷贝继承。
        rev = base_id
        runs = _runs_pending_del(p._p)
        main_rpr_src = None
        if runs:
            main_rpr_src = max(runs, key=lambda r: len(_run_visible_text(r)))
        for r in runs:
            _wrap_del_run(r, author, ts, rev)
            rev += 1
        p._p.append(_new_ins_run(text, author, ts, rev, rpr_src=main_rpr_src))
    elif op == "append":
        # ★ 2026-10-10：段尾追加继承**段落标记**格式（w:pPr/w:rPr）——旧实现不带 rPr ⇒
        #   格式化段落（如黑体作者行）里追加的文字退化为默认字体（N15 同族问题）。
        ppr_rpr = p._p.find(f"{qn('w:pPr')}/{qn('w:rPr')}")
        p._p.append(_new_ins_run(text, author, ts, base_id, rpr_src=ppr_rpr))
    elif op == "delete":
        # 整段删除：段内全部有效文字（含嵌套 run）逐个标记删除（保留痕迹），段落标记一并标记删除
        rev = base_id
        for r in _runs_pending_del(p._p):
            _wrap_del_run(r, author, ts, rev)
            rev += 1
        _mark_para_deleted(p, author, ts, rev)
    elif op == "insert_after":
        new_p = OxmlElement("w:p")
        # 样式继承：浅拷贝原段 pPr —— C3 起含 numPr 也复制（新段延续列表编号）
        src_ppr = p._p.find(qn("w:pPr"))
        if src_ppr is not None:
            import copy

            new_p.append(copy.deepcopy(src_ppr))
        new_p.append(_new_ins_run(text, author, ts, base_id))
        p._p.addnext(new_p)
    elif op == "purge":
        # 彻底删除（物理移除段落元素）：仅用于标记删除后的空壳段（校验层保证有效文本已清空）。
        # 物理删除不可逆于修订标记，但落盘前的 .bak 备份 + 「撤销上次应用」仍可整体还原。
        parent = p._p.getparent()
        if parent is not None:
            parent.remove(p._p)


def apply_edits(path: Path, edits: list[dict], author: str = _AUTHOR) -> dict:
    """应用编辑操作到 docx（track-changes），返回 applied / rejected 明细。

    流程：pid 定位 → quote 校验阀 → 域/媒体段拒绝 → 写入 → round-trip 保真验证 → 原子替换。
    任一环节失败均不落盘（校验全部前置，写入阶段只在内存副本上进行）。
    """
    if not path.exists():
        raise DocxNotFound(str(path))
    if path.suffix.lower() != ".docx":
        raise DocxNotEditable(f"仅支持 .docx 文件，收到：{path.suffix!r}")
    if not edits:
        raise DocxEditError("未提供任何编辑操作")

    try:
        doc = Document(str(path))
    except Exception as e:  # noqa: BLE001  损坏文件统一按不可编辑处理
        raise DocxNotEditable(f"docx 打开失败（文件可能已损坏）：{e}") from e

    # 可编辑性视图：段落视图（P/T）+ 文档流视图（B，含表格 cells）合并
    view: dict[str, dict] = {v["pid"]: v for v in list_paragraphs(path)}
    for item in list_stream(path):
        view.setdefault(item["pid"], item)
        for cell in item.get("cells", []):
            view.setdefault(cell["pid"], cell)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    base_id = _max_revision_id(doc) + 1

    applied: list[dict] = []
    rejected: list[dict] = []
    # (段落对象, op, text) —— 校验通过待执行
    pending: list[tuple[Any, str, str]] = []

    for e in edits:
        pid = str(e.get("pid") or "")
        op = str(e.get("op") or "")
        text = str(e.get("text") or "")
        quote = e.get("quote")
        item = {"pid": pid, "op": op}

        if op not in _EDIT_OPS:
            rejected.append({**item, "reason": f"不支持的操作类型（支持 {_EDIT_OPS}）"})
            continue
        # delete / purge 不写入文字，text 允许为空；
        # 仅 replace / append / insert_after 需要实际内容。前端 delete/purge op 恒传 text=""。
        if op not in ("delete", "purge") and not text.strip():
            rejected.append({**item, "reason": "text 为空"})
            continue
        target = _locate(doc, pid)
        if target is None:
            rejected.append({**item, "reason": "pid 无法定位（越界或格式非法）"})
            continue
        if op == "purge":
            # 安全阀（宁拒不错）：purge 物理移除段落、不可逆于修订标记，仅允许
            # ① 有效文本已清空（= 标记删除后的空壳段）；② 正文段落（表格内段落会破坏单元格结构）。
            if _para_text(target._p).strip():
                rejected.append({**item, "reason": "仅允许彻底删除已清空的段落（请先执行标记删除）"})
                continue
            if _inside_table_cell(target._p):
                rejected.append({**item, "reason": "表格内段落不支持彻底删除"})
                continue
            pending.append((target, op, text))
            applied.append(item)
            continue
        v = view.get(pid)
        if not v["editable"]:
            rejected.append({**item, "reason": f"段落不可编辑（{v['reason'] or 'empty'}）"})
            continue
        # quote 校验阀：宁拒不错
        if quote and str(quote) not in _para_text(target._p):
            rejected.append({**item, "reason": "quote 与段落原文不匹配（安全拒绝）"})
            continue
        pending.append((target, op, text))
        applied.append(item)

    # 统一执行（replace/delete 会重排同段 run，故按申请顺序逐个执行；id 连续递增）
    rev = base_id
    for target, op, text in pending:
        _apply_one(doc, target, op, text, author, ts, rev)
        rev += 1

    # C3：有实际写入前先快照原文件并入 undo 栈（.baks.json 管栈，保留最近 _MAX_BACKUPS 级）
    backup_path: str | None = None
    if applied:
        backup_path = f"{path.name}.bak-{snapshot_state(path)}"

    # round-trip 保真：临时文件 → zip 完整性 → 重开验证 → 原子替换
    tmp = path.with_name(path.name + ".tmp")
    try:
        doc.save(str(tmp))
        with zipfile.ZipFile(tmp) as z:
            bad = z.testzip()
            if bad is not None:
                raise DocxEditError(f"写入后 zip 校验失败（损坏条目：{bad}），已放弃落盘")
        Document(str(tmp))  # 重开验证
        _replace_with_retry(tmp, path)  # 占用重试（WinError 32）：瞬时锁自愈，持续锁给明确指引
    except DocxEditError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise DocxEditError(f"落盘失败：{e}") from e

    return {
        "applied": applied,
        "rejected": rejected,
        "total": len(edits),
        "paragraph_count_after": len(Document(str(path)).paragraphs),
        "backup": backup_path,
    }


# ---------- 备份与撤销（C3） ----------

def _bak_glob(path: Path) -> str:
    return f"{path.name}.bak-*"


def _baks_meta_path(path: Path) -> Path:
    """undo/redo 双栈的元数据文件（与快照同目录）。"""
    return path.with_name(path.name + ".baks.json")


def _load_baks_meta(path: Path) -> dict:
    """读取 undo/redo 栈；文件缺失（栈机制启用前的旧备份）返回空栈。

    ★ O-06（2026-10-01）：**损坏不再静默** —— 原实现 `except: pass` 无日志、无痕迹，两个后果：
      ① `restore_from_backup` 报「没有可撤销的操作(400)」，而磁盘上 `.bak-*` **仍真实存在**
        （用户无法区分"真的没有"与"读坏了"）；
      ② 随后 `_save_baks_meta` 会把**损坏的栈覆盖** ⇒ 旧历史**永久失联**（不可逆）。

    现在的行为：
      - 文件**缺失**（`FileNotFoundError`）= 正常路径（机制启用前的旧备份），**不打日志**；
      - 内容**损坏**（JSON 坏 / 结构不对 / 读取失败）⇒ **warning + 原文件改名保留**
        （`.baks.json.corrupt-<ts>`）⇒ ① 后续 `_save_baks_meta` **不再覆盖**它；
        ② `restore_from_backup` 的「栈文件不存在 ⇒ 收编磁盘最新 `.bak-*` 作栈底」兜底
        **顺势生效** ⇒ 撤销历史可恢复（而不是误报"没有可撤销的操作"）。
    """
    meta_path = _baks_meta_path(path)
    try:
        raw = meta_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"undo": [], "redo": []}
    except OSError as e:  # 读失败（权限/占用）：没法改名保留，只能按空栈 + 显式告警
        logger.warning("撤销栈读取失败（%s）⇒ 按空栈处理；原文件保留待人工排查", e)
        return {"undo": [], "redo": []}
    try:
        meta = json.loads(raw)
        undo, redo = meta.get("undo"), meta.get("redo")
        if isinstance(meta, dict) and isinstance(undo, list) and isinstance(redo, list):
            return {"undo": list(undo), "redo": list(redo)}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # 走到这里 = **内容损坏**（JSON 坏 / 结构不对）⇒ 保留现场 + 显式告警
    logger.warning(
        "撤销栈损坏（%s）⇒ 已改名保留为 .corrupt 并按空栈处理；"
        "磁盘上的 .bak-* 快照不受影响，下次撤销会收编最新一份",
        meta_path.name,
    )
    try:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        meta_path.rename(meta_path.with_name(meta_path.name + f".corrupt-{stamp}"))
    except OSError:
        # 改名失败（被占用等）：主流程照常，但下次写入会覆盖损坏内容 —— 必须留痕
        logger.warning("损坏的撤销栈改名保留失败（下次写入将覆盖它）", exc_info=True)
    return {"undo": [], "redo": []}


def _save_baks_meta(path: Path, meta: dict) -> None:
    """**原子**写入 undo/redo 栈（O-06，2026-10-01）。

    原实现 `write_text` 直接写 —— 既违反项目铁律（**截断后写失败即全丢**），
    也正是"损坏栈被覆盖 ⇒ 历史永久失联"的帮凶。改为 tmp + `os.replace`（复用既有
    `_replace_with_retry`，带占用重试），失败时清理 tmp、原文件保持完好。
    """
    payload = json.dumps(meta, ensure_ascii=False, indent=2)
    target = _baks_meta_path(path)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        _replace_with_retry(Path(tmp), target)
    except BaseException:  # noqa: BLE002  任何失败都要清 tmp、保原文件
        Path(tmp).unlink(missing_ok=True)
        raise


def _bak_name(path: Path, ts: str) -> Path:
    return path.with_name(f"{path.name}.bak-{ts}")


def snapshot_state(path: Path) -> str:
    """应用前快照：当前状态入 **undo 栈**（多级撤销的回退点）。

    新操作会使「重做」失效：redo 栈清空、其快照文件一并删除（标准 undo/redo 语义）。
    undo 栈超 _MAX_BACKUPS 级时淘汰最旧（文件+栈同步删）。返回快照 ts。
    """
    meta = _load_baks_meta(path)
    undo_stack: list[str] = meta["undo"]
    redo_stack: list[str] = meta["redo"]
    for ts in redo_stack:  # 新操作使 redo 失效
        _bak_name(path, ts).unlink(missing_ok=True)
    redo_stack = []
    while len(undo_stack) >= _MAX_BACKUPS:  # 淘汰最旧的 undo 快照
        old_ts = undo_stack.pop(0)
        _bak_name(path, old_ts).unlink(missing_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    shutil.copy2(path, _bak_name(path, ts))
    undo_stack.append(ts)
    meta["undo"], meta["redo"] = undo_stack, redo_stack
    _save_baks_meta(path, meta)
    return ts


def restore_from_backup(path: Path) -> dict:
    """一键撤销（**多级**）：回退到上一个用户操作前的快照。

    2026-09-06 修复横跳缺陷：旧版还原前把当前（改后）状态再备份成新的"最新备份"——
    撤销动作自己进入可撤销历史，再次撤销就把「撤销」回退掉，状态在改前/改后之间反复横跳。
    现改为显式双栈：撤销把当前状态转入 **redo 栈**（不进 undo 栈 → 不会自引用），
    从 undo 栈弹出上一级快照还原；undo 栈为空时明确报"没有可撤销的操作"。
    兼容：栈文件不存在（机制启用前的旧备份）时收编磁盘上最新一份作为栈底，行为向后兼容。
    返回 {restored_from}。
    """
    if not path.exists():
        raise DocxNotFound(str(path))
    meta_path = _baks_meta_path(path)
    meta = _load_baks_meta(path)
    undo_stack: list[str] = meta["undo"]
    if not undo_stack:
        # 旧格式兼容：栈文件不存在（此前从未走过新栈）→ 收编磁盘上最新备份作为栈底
        if not meta_path.exists():
            baks = sorted(path.parent.glob(_bak_glob(path)))
            if baks:
                undo_stack = [baks[-1].name.rsplit(".bak-", 1)[-1]]
        if not undo_stack:
            raise DocxEditError("没有可撤销的操作")
    # 撤销前的当前状态 → redo 栈（未来重做可用；新操作会清空 redo，见 snapshot_state）
    now_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    shutil.copy2(path, _bak_name(path, now_ts))
    redo_stack: list[str] = meta["redo"]
    redo_stack.append(now_ts)
    while len(redo_stack) > _MAX_BACKUPS:  # redo 超限淘汰最旧
        old_ts = redo_stack.pop(0)
        _bak_name(path, old_ts).unlink(missing_ok=True)
    # 弹出 undo 栈顶 → 还原该快照
    top_ts = undo_stack.pop()
    bak = _bak_name(path, top_ts)
    if not bak.exists():
        _save_baks_meta(path, {"undo": undo_stack, "redo": redo_stack})
        raise DocxEditError("撤销目标快照已不存在（可能被清理）")
    meta["undo"], meta["redo"] = undo_stack, redo_stack
    _save_baks_meta(path, meta)
    _copy_with_retry(bak, path)
    return {"restored_from": bak.name}


# ---------- 导出（C4 增补：修改后保存导出） ----------

def _unwrap(el: Any) -> None:
    """把元素从父级解包：子节点提升到原位置。"""
    parent = el.getparent()
    if parent is None:
        return
    idx = parent.index(el)
    for child in list(el):
        parent.insert(idx, child)
        idx += 1
    parent.remove(el)


def accept_all_revisions(path: Path) -> bytes:
    """导出「接受全部修订」后的干净 docx 字节（不改动原文件）。

    处理语义 = Word 中「接受所有修订」的 run 级近似：
    1. 所有 ``<w:ins>`` 解包 —— 新文字变为普通 run；
    2. 所有 ``<w:del>`` 移除 —— 旧文字（delText）与段落标记删除标记一并消失。
    被整段删除的段落在接受后会留下空壳段（无文字），与 Word 的「段落合并」行为有差，
    但视觉上仅为空行；保持实现简单可靠，不做段级删除（避免误伤）。
    """
    if path.suffix.lower() != ".docx":
        raise DocxNotEditable(f"仅支持 .docx 文件，收到：{path.suffix!r}")
    doc = Document(str(path))
    body = doc.element.body

    # 1) ins 解包（先收集再改树）
    for ins in list(body.iter(qn("w:ins"))):
        _unwrap(ins)
    # 2) del 移除（含段落标记 del）
    for del_el in list(body.iter(qn("w:del"))):
        parent = del_el.getparent()
        if parent is not None:
            parent.remove(del_el)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
