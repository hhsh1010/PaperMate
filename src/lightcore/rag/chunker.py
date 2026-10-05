"""LightCore · 语义分块（docs/05 §RAG / 分块）

目标块大小 300–500 token（docs/05；`_TARGET_TOKENS = 400` / `_OVERLAP_TOKENS = 60`）。
以句子为单位累积，达到目标上限即切分；块间留 overlap（按句子回退）以保留上下文。

★ 2026-10-04（O9）修正过时描述：此处原写
「中文按 **~1.5 字/token**、英文 **~4 字符/token** 近似」，而 `_estimate_tokens`
（`:30-33`）实际是**统一** `max(1, int(len(text) / 2.5))`，**没有**按语言分档。
⇒ 实际口径：中文 ~2.5 字/token、英文 ~2.5 字符/token。
⚠️ 旧描述会让调参者高估中文块的实际 token 密度（以为 1.5 字就 1 token）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .parser import ParsedBlock, merge_split_numbers

_TARGET_TOKENS = 400
_OVERLAP_TOKENS = 60
_SENT_SPLIT = re.compile(
    r"(?<=[。！？!?；;\n])"  # ① 中文句末标点 / 换行
    r"|(?<=\.)(?=\s|$)"  # ② 英文句点 + 空白或行尾（典型英文句末）
    r"|(?<=[a-z]\.)(?=[A-Z])"  # ③ 小写 + 句点 + 大写（缩写/紧贴的英文句末）
)
# ⚠️ 2026-09-19 修复：旧正则 `(?<=[。！？!?；;\.\n])` 把**英文句点无条件**当句子边界，
# 于是 `77.4` → `77.` + `4`，再被 chunk 内的 `" ".join(buf)` 拼成 `77. 4`
# —— 实测全库 631 块中 314 块（49.8%）中招（`mAP@0. 5`、`3. 4. 4`、`v1. 0`、DOI 全被劈开），
# 检索返回的数值/章节号无法判读（见 `docs/检索质量诊断-分块碎片化.md` §6）。
# 新规则只在「句点后是空白/行尾」或「小写→大写」时才切，小数点/版本号/DOI/章节号保持完整。
# 三个分支**全部零宽** → 句点一律留在前一段末尾，切分后拼接行为可预测。


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 2.5))


def _iter_sentences(text: str) -> list[str]:
    """切句并丢弃空白片段。

    `_SENT_SPLIT` 把换行 `\\n` 也当句子边界 —— **保留**它：跨 block 累积之后，
    换行不再制造碎片（句子会被 `" ".join` 拼回同一块），反而让 overlap 的回退粒度更细
    （跨行英文 `... field of the` + `feature map ...` 能拼回完整语义单元）。

    ⚠️ 英文句点 `.` **不是**无条件边界（见 `_SENT_SPLIT` 的注释）：否则小数点、
    版本号、DOI、章节号都会被劈开（`77.4` → `77. 4`）。判据是「句点后必须跟空白/行尾」
    或「小写→大写」。
    """
    return [s for s in (x.strip() for x in _SENT_SPLIT.split(text)) if s]


@dataclass
class Chunk:
    id: str
    text: str
    page: int | None = None  # None=无物理页码；PDF 为 1 起算
    token_count: int = 0


class Chunker:
    def __init__(self, target_tokens: int = _TARGET_TOKENS, overlap_tokens: int = _OVERLAP_TOKENS) -> None:
        self.target = target_tokens
        self.overlap = overlap_tokens

    def chunk_blocks(self, blocks: list[ParsedBlock], doc_id: str = "doc") -> list[Chunk]:
        """按语义累积分块（**跨 block 累积**）。

        ⚠️ 2026-09-17 修复「碎片化」：`buf` 原先建在 **block 循环内部**，于是每个 block
        处理完立刻 flush、**永不跨 block 累积**。而 PDF 解析（`page.get_text("blocks")`）
        对期刊双栏排版常返回"每行一块"，两者相乘 → **一个 chunk ≈ 一行**。
        实测后果（3 篇论文）：chunk 平均仅 **22~39 字符**，doc16 有 **93.6% 短于 50 字符**
        （最短 1 字符），相邻 chunk 还能拼回同一句 → 检索出的"片段"零散不完整，
        注入模型后无法支撑准确判断。而设计目标是 300~500 **token**。

        现在 `buf` 跨 block 累积，只在【达到 target / **换页** / 文档结束】时成块。

        ⚠️ **页边界必须保留**（换页无条件收块）：`pdf_extract(page=N)` 按页取原文、
        引用溯源显示"第 N 页"，都依赖「页 → 块」的对应关系。实测回归
        `test_page_indexing.py::test_pipeline_from_db_chunks_preserves_pages` 正是守这条 ——
        若让块跨页合并，第 2 页会取不到内容（那是功能损失，不只是测试问题）。
        真实论文每页正文通常 800~1500 字符，页内累积后**不会**因此产生碎片；
        只有封面/尾页这类本就内容极少的页面会出现短块，属实反映该页内容。
        """
        chunks: list[Chunk] = []
        idx = 0
        buf: list[str] = []
        buf_tokens = 0
        buf_page: int | None = None
        last_page: int | None = None

        def flush() -> None:
            """当前缓冲成块；尾部若干句子留作下一块的 overlap。"""
            nonlocal idx, buf, buf_tokens
            if not buf:
                return
            chunks.append(self._make_chunk(doc_id, idx, " ".join(buf), buf_page))
            idx += 1
            buf = self._overlap_sentences(buf)
            buf_tokens = sum(_estimate_tokens(s) for s in buf)

        for b in blocks:
            # 换页 → 先收块（见 docstring：页号是硬语义）。page=None 的非分页文档不触发。
            if buf and last_page is not None and b.page != last_page:
                flush()
                buf_page = b.page
            for sent in _iter_sentences(b.text):
                st = _estimate_tokens(sent)
                if buf and buf_tokens + st > self.target:
                    flush()
                    buf_page = b.page
                elif not buf:
                    buf_page = b.page  # 新块的第一句 → 记录起始页
                buf.append(sent)
                buf_tokens += st
            last_page = b.page
        flush()
        return chunks

    def _overlap_sentences(self, buf: list[str]) -> list[str]:
        kept: list[str] = []
        toks = 0
        for s in reversed(buf):
            t = _estimate_tokens(s)
            if toks + t > self.overlap:
                break
            kept.insert(0, s)
            toks += t
        return kept

    def _make_chunk(self, doc_id: str, idx: int, text: str, page: int) -> Chunk:
        cid = f"{doc_id}-c{idx}"
        # 再跑一次合并（2026-09-19）：`ParsedBlock` 只在**单个 block 内**归一化，
        # 而 `" ".join(buf)` 会跨 block 拼接 —— block 末尾的 `77.` 接上下一 block 开头的 `4`
        # 同样会生成 `77. 4`。这里对最终产物再补一道（幂等，规则与 parser 完全一致）。
        text = merge_split_numbers(text)
        return Chunk(id=cid, text=text, page=page, token_count=_estimate_tokens(text))
