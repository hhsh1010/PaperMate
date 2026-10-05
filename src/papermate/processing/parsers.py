"""PaperMate · 文档处理子包 · 解析器（parsers）

按来源格式解析为结构化章节（Section）：
- TextParser：纯文本，按空行分段；标题用启发式识别（短行 + 非句末）
- MarkdownParser：`#`~`######` 标题解析，正文归并到最近标题下
- PdfParser：pymupdf 惰性导入（避免无 rag extra 时 import 失败）；按页取文本块，
  保留真实页码（A-04：PDF page ≥ 1 填充，修复引用溯源 p.0 根因）；
  标题识别复用通用启发式。

ParserFactory.for_file(filename) 按后缀路由；未知后缀回退 TextParser。
所有解析器纯逻辑、无 DB 依赖；PdfParser 的页面读取抽象为 _read_pages，
便于测试注入伪页面块而无需真实 PDF（pymupdf 不在基础依赖里）。
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import ParsedContent, Section

#: 标题启发式：行首带编号/短句且不以句号结尾
_HEADING_RE = re.compile(r"^(?:\d+(?:[.\-、]\d+)*[.\-、]?\s*|第[一二三四五六七八九十\d]+[章节部分]\s*)?")
_SENT_END = re.compile(r"[。！？!?；;]$")
_MAX_HEADING_CHARS = 80


class DocParseError(Exception):
    """文档解析失败（格式不支持 / 文件损坏 / 依赖缺失）。消息面向用户可读。"""


def _looks_like_heading(line: str) -> bool:
    """通用标题启发式：非空、较短、无句末标点、不以小写字母/逗号开头。"""
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_CHARS:
        return False
    if _SENT_END.search(line):
        return False
    if line[0] in ",，。；;：:：":
        return False
    # 包含句子分隔符（多于一个标点簇）视为正文
    if len(re.findall(r"[，。；;！？!?]", line)) > 1:
        return False
    return True


def _extract_stats(text: str) -> dict:
    """轻量文本统计：字符数/段落数/句子数（与 metadata 独立，仅解析层使用）。"""
    chars = len(text)
    paragraphs = sum(1 for p in text.split("\n\n") if p.strip())
    sentences = max(1, len(re.findall(r"[。！？!?；;\.]", text)))
    return {"chars": chars, "paragraphs": paragraphs, "sentences": sentences}


#: 全角小写字母（断词重连检测用，见 _normalize_pdf_text）
_FW_LOWER = tuple("ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ")
#: 全角小写字母 + 全角数字（断词重连的下行首字符允许集）
_FW_LOWER_OR_DIGIT = _FW_LOWER + tuple("０１２３４５６７８９")


def _rejoin_broken_words(text: str) -> str:
    """行级断词重连（2026-09-09，U1 残留项 C：双栏/窄格 PDF 的「混乱断句」）。

    两类断词、两种精确启发式（均在逐字符全角映射**之前**、全角状态下检测行边界）：
    1. **半角连字符断词**（英文排版标准行为）：行尾 `-`/`－` + 下行 ASCII 字母开头
       → 去连字符拼接（improve-\\nment → improvement）。经典 dehyphenation，零风险。
    2. **全角窄格强制断行**（中文期刊表格单元格列宽极窄，无连字符）：
       行尾与下行首**均为全角小写字母** → 无缝拼接（Ｍｏ\\nｄｅｌ → Ｍｏｄｅｌ、
       ａ\\nｎｄ → ａｎｄ）。该形态只出现在破碎排版中——正常全角行不会以全角小写
       字母结尾又以下一全角小写字母开头——合并零误伤。
    只在**块内部**处理：块边界通常是段落/表格单元边界，跨块重连不安全。
    """
    merged: list[str] = []
    for ln in text.split("\n"):
        if merged:
            prev = merged[-1]
            if prev.endswith(("-", "－")) and ln[:1].isascii() and ln[:1].isalpha():
                merged[-1] = prev[:-1] + ln  # dehyphenation
                continue
            if prev.endswith(_FW_LOWER) and ln.startswith(_FW_LOWER_OR_DIGIT):
                merged[-1] = prev + ln  # 全角窄格断行重连
                continue
        merged.append(ln)
    return "\n".join(merged)


def _normalize_pdf_text(text: str) -> str:
    """PDF 提取文本归一化（2026-09-09 修复「回复乱码/混乱断句」的源头）。

    中文期刊 PDF 普遍用**全角**字母/数字排版（如 Ｆｉｇ．９、４８、Ｍａｓｋ Ｒ－ＣＮＮ），
    pymupdf 原样提取后随 chunk 进入引用与回复——视觉上呈「乱码」；双栏排版还会把
    单词从中间断开（断句错乱观感）。

    流程：
    1. `_rejoin_broken_words`：行级断词重连（连字符断词 + 全角窄格断行）；
    2. 逐字符映射：**仅**全角字母/数字/全角连字符/全角句点 → 半角，并清除零宽字符、
       压缩行内空白。**有意不用 NFKC**——它会连带把中文全角标点（，：；！？（））转成
       半角，对中文文档是排版质量劣化。换行结构保持不变，不改变分块结构与页码归属。
    注意：仅对新解析的文档生效，存量 chunk 需重新解析/上传才会改善。
    表格**结构**感知（find_tables 单元格重排）不在本层——对扫描件/无线表格不鲁棒，另行立项。
    """
    text = _rejoin_broken_words(text)
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if 0xFF10 <= o <= 0xFF19 or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
            out.append(chr(o - 0xFEE0))  # 全角 ０-９ / Ａ-Ｚ / ａ-ｚ → 半角
        elif ch == "\u3000":
            out.append(" ")  # 全角空格
        elif ch == "\uff0d":
            out.append("-")  # 全角连字符 －（Ｒ－ＣＮＮ → R-CNN）
        elif ch == "\uff0e":
            out.append(".")  # 全角句点 ．（Ｆｉｇ．９ → Fig.9）
        elif ch in "\u200b\u200c\u200d\u2060\ufeff\xad":
            continue  # 零宽字符 / 软连字符（PDF 提取常见噪声）
        else:
            out.append(ch)  # 中文标点等一律保留
    text = "".join(out)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return text


class TextParser:
    """纯文本解析：按空行分段，段内首行为短句时提升为标题（level=1）。"""

    def parse(self, text: str, title: str = "") -> ParsedContent:
        sections: list[Section] = []
        order = 0
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            lines = para.splitlines()
            heading: str | None = None
            body = para
            if len(lines) >= 2 and _looks_like_heading(lines[0]) and not lines[0].endswith(":"):
                heading = lines[0].strip()
                body = "\n".join(lines[1:]).strip()
            order += 1
            sections.append(Section(level=1 if heading else 0, heading=heading, page=0, text=body or heading or "", order=order))
        return ParsedContent(
            source_format="text",
            title=title or (sections[0].heading if sections and sections[0].heading else Path(title).stem or "未命名文档"),
            sections=sections,
            raw_text=text,
            stats=_extract_stats(text),
        )


class MarkdownParser:
    """Markdown 解析：`#`~`######` 为标题；正文归并到最近标题的节；代码块/引用原样保留在节文本内。"""

    _HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def parse(self, text: str, title: str = "") -> ParsedContent:
        sections: list[Section] = []
        order = 0
        current_level = 0
        current_heading: str | None = None
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf
            body = "\n".join(buf).strip()
            if body or current_heading is not None:
                order_local = order
                sections.append(Section(level=current_level, heading=current_heading, page=0, text=body, order=order_local))
            buf = []

        for line in text.splitlines():
            m = self._HEADING.match(line)
            if m:
                flush()
                current_level = len(m.group(1))
                current_heading = m.group(2).strip()
                order += 1
            else:
                buf.append(line)
        flush()

        first_heading = next((s.heading for s in sections if s.heading), None)
        return ParsedContent(
            source_format="markdown",
            title=title or first_heading or Path(title).stem or "未命名文档",
            sections=sections,
            raw_text=text,
            stats=_extract_stats(text),
        )


class PdfParser:
    """PDF 解析：按页取文本块并保留页码（A-04）。

    _read_pages 返回 [(page_no, [block_text, ...])]，真实实现惰性 import pymupdf；
    测试可注入伪页面块（不依赖 pymupdf）。
    """

    def _read_pages(self, path: str | Path) -> list[tuple[int, list[str]]]:
        import pymupdf  # 惰性：仅 PDF 场景才需要，基础依赖不强制

        pages: list[tuple[int, list[str]]] = []
        with pymupdf.open(str(path)) as doc:
            for page_no, page in enumerate(doc, start=1):
                texts = [b[4].strip() for b in page.get_text("blocks") if b[4].strip()]
                if texts:
                    pages.append((page_no, texts))
        return pages

    def parse(self, path: str | Path, title: str = "") -> ParsedContent:
        pages = self._read_pages(path)
        # 2026-09-09 修复「回复乱码/断句错乱」：归一化放在 parse 层（而非 _read_pages 内），
        # 保证测试注入的伪页面块与真实 pymupdf 提取走同一条归一化路径。
        pages = [(page_no, [_normalize_pdf_text(t) for t in texts]) for page_no, texts in pages]
        sections: list[Section] = []
        order = 0
        raw_parts: list[str] = []
        for page_no, texts in pages:
            for t in texts:
                raw_parts.append(t)
                # 块级标题启发式：单块、短、非句末 → 提升为标题；否则并入上一节正文
                if _looks_like_heading(t) and not _HEADING_RE.sub("", t).endswith("."):
                    order += 1
                    sections.append(Section(level=1, heading=t, page=page_no, text="", order=order))
                else:
                    if sections:
                        sections[-1].text = (sections[-1].text + "\n" + t).strip()
                    else:
                        order += 1
                        sections.append(Section(level=0, heading=None, page=page_no, text=t, order=order))
        raw_text = "\n".join(raw_parts)
        first_heading = next((s.heading for s in sections if s.heading), None)
        return ParsedContent(
            source_format="pdf",
            title=title or first_heading or Path(title).stem or "未命名文档",
            sections=sections,
            raw_text=raw_text,
            num_pages=pages[-1][0] if pages else 0,
            stats=_extract_stats(raw_text),
        )


class DocxParser:
    """Word .docx 解析（python-docx 惰性导入）：段落 + 表格 + Heading 样式级别。

    - 样式名以 Heading 开头 → 对应级别标题；其余短句走通用标题启发式。
    - 表格行以「单元格 | 单元格」文本并入所在章节。
    - .doc（OLE 二进制）python-docx 不支持 → 抛 DocParseError 提示转存 .docx（见 ParserFactory）。
    """

    def parse(self, path: str | Path, title: str = "") -> ParsedContent:
        try:
            from docx import Document  # 惰性：仅 .docx 场景才需要
        except ImportError:
            raise DocParseError("缺少 python-docx 依赖（uv add python-docx）") from None
        try:
            doc = Document(str(path))
        except Exception as e:  # noqa: BLE001 OLE .doc / 损坏文件 / 非 Word
            raise DocParseError(
                f"无法解析该 Word 文档（{type(e).__name__}）。"
                f"老版 .doc 二进制格式不受支持，请先用 Word/WPS 转存为 .docx 后重新上传。"
            ) from e
        sections: list[Section] = []
        order = 0

        def _emit(text: str) -> None:
            nonlocal order
            text = text.strip()
            if not text:
                return
            if sections:
                sections[-1].text = (sections[-1].text + "\n" + text).strip()
            else:
                order += 1
                sections.append(Section(level=0, heading=None, page=0, text=text, order=order))

        from docx.oxml.ns import qn

        def _accepted_text(el) -> str:
            """「接受全部修订」语义的文本：取元素内**全部** `w:t`（含 `w:ins` 嵌套 run）、
            忽略 `w:delText`（track-changes 删除的旧文字）。

            ★ 为什么不能用 `.text`：python-docx 的 `paragraph.text` / `cell.text` 只遍历
            **直接子级** `w:r`，改稿（track-changes replace）后的新文字包在 `w:ins` 里
            ⇒ 会被整个漏掉，索引出碎片文本。实测（2026-09-30，
            `scripts/probe_edit_consistency.py`）：改一段后 chunks = `'引言。 e icrow s.'`。
            与 `docx_edit._para_accepted_text` 同口径（= 导出"干净版"的语义）。
            """
            return "".join(node.text or "" for node in el.iter() if node.tag == qn("w:t"))

        for p in doc.paragraphs:
            text = _accepted_text(p._p).strip()
            if not text:
                continue
            style = (p.style.name or "") if p.style is not None else ""
            sname = style.lower()
            if sname.startswith("heading"):
                parts = style.split()
                lvl = int(parts[-1]) if parts and parts[-1].isdigit() else 1
                order += 1
                sections.append(Section(level=min(6, lvl), heading=text, page=0, text="", order=order))
            elif _looks_like_heading(text):
                order += 1
                sections.append(Section(level=1, heading=text, page=0, text="", order=order))
            else:
                _emit(text)
        # 表格：行文本并入章节（Word 表格在检索中同样需要可搜索）
        for tbl in doc.tables:
            for row in tbl.rows:
                cells = [_accepted_text(c._tc).strip() for c in row.cells]
                if any(cells):
                    _emit(" | ".join(cells))
        raw_text = "\n".join(
            t for t in (_accepted_text(p._p) for p in doc.paragraphs) if t.strip()
        )
        first_heading = next((s.heading for s in sections if s.heading), None)
        return ParsedContent(
            source_format="docx",
            title=title or first_heading or Path(title).stem or "未命名文档",
            sections=sections,
            raw_text=raw_text,
            num_pages=0,
            stats=_extract_stats(raw_text),
        )


class ParserFactory:
    """按文件名后缀返回解析器；未知后缀回退 TextParser。

    - .pdf → PdfParser；.md/.markdown/.mdown/.mkd → MarkdownParser
    - .docx → DocxParser（.doc 老格式同样路由到 DocxParser，parse 时抛 DocParseError 提示转存）
    """

    _MARKDOWN_EXT = {".md", ".markdown", ".mdown", ".mkd"}

    @classmethod
    def for_file(cls, filename: str):
        ext = Path(filename or "").suffix.lower()
        if ext == ".pdf":
            return PdfParser()
        if ext in cls._MARKDOWN_EXT:
            return MarkdownParser()
        if ext in (".docx", ".doc"):  # .doc 老格式 → DocxParser 抛 DocParseError 提示转存
            return DocxParser()
        return TextParser()
