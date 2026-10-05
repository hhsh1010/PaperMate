"""PaperMate · 文档处理子包 · 格式转换（converter）

把解析后的结构化章节转换为多种输出格式：
- to_markdown：`# 标题` + 层级标题 + 正文（可反向读回，保证转换不丢内容）
- to_structured：JSON 可序列化字典（sections + metadata 快照）
- to_plain_text：纯文本（标题行 + 正文）
- to_bibtex：由元信息生成最小 BibTeX 条目（供导出模块复用；元信息不足时降级为 @misc）

设计约定：转换是纯函数，输入 ParsedContent/DocumentMetadata，输出 str/dict；
与落盘、DB 无关，便于单测。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .models import DocumentMetadata, ParsedContent

#: 用于 BibTeX 键的清理（保留字母数字、连字符与**中文**——见 to_bibtex 的 M12 修复）
_KEY_CLEAN = re.compile(r"[^A-Za-z0-9\-_:\u4e00-\u9fff]+")
#: 从文件名猜测年份（如 2024_xxx / xxx2024）
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def to_markdown(content: ParsedContent) -> str:
    """章节 → Markdown（含标题）。首个与文档标题相同的 H1 章节视为标题本身，不再重复输出。"""
    lines: list[str] = [f"# {content.title}", ""]
    first_title_skipped = False
    for s in content.sections:
        if not first_title_skipped and s.level == 1 and s.heading == content.title:
            first_title_skipped = True
            if s.text:
                lines.append(s.text)
                lines.append("")
            continue
        if s.heading:
            lines.append(f"{'#' * min(max(s.level, 1), 6)} {s.heading}")
        if s.text:
            lines.append(s.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_plain_text(content: ParsedContent) -> str:
    """章节 → 纯文本（标题行 + 正文）。"""
    lines: list[str] = [content.title, ""]
    for s in content.sections:
        if s.heading:
            lines.append(s.heading)
        if s.text:
            lines.append(s.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_structured(content: ParsedContent, metadata: DocumentMetadata | None = None) -> dict[str, Any]:
    """章节 + 元信息 → JSON 可序列化字典（结构化存储的载体）。"""
    meta: dict[str, Any]
    if metadata is not None:
        meta = {
            "language": metadata.language,
            "chars": metadata.chars,
            "words": metadata.words,
            "paragraphs": metadata.paragraphs,
            "sentences": metadata.sentences,
            "headings": metadata.headings,
            "has_code_block": metadata.has_code_block,
            "keywords": metadata.keywords,
        }
    else:
        meta = {
            "chars": len(content.raw_text),
            "paragraphs": content.stats.get("paragraphs", 0),
        }
    return {
        "title": content.title,
        "source_format": content.source_format,
        "num_pages": content.num_pages,
        "sections": [
            {
                "order": s.order,
                "level": s.level,
                "heading": s.heading,
                "page": s.page,
                "text": s.text,
            }
            for s in content.sections
        ],
        "metadata": meta,
    }


def guess_year(filename: str) -> str | None:
    """从文件名猜测年份（如 2024_xxx / xxx2024），无则 None。"""
    m = _YEAR_RE.search(filename or "")
    return m.group(0) if m else None


def to_bibtex(
    title: str,
    filename: str = "",
    metadata: DocumentMetadata | None = None,
    year: str | None = None,
) -> str:
    """由元信息生成最小 BibTeX 条目；无作者/年份时降级为 @misc。"""
    source = filename or title
    # M12 修复（2026-09-10）：原 `_KEY_CLEAN` 只留 `[A-Za-z0-9\-_:]`，中文标题被整段剔除
    # → 多篇中文文献导出 key 恒为 "paper"（BibTeX 键必须唯一，条目互相覆盖）。
    # 现保留 CJK；若清洗后仍为空（纯符号标题），用标题哈希兜底保证唯一。
    base = _KEY_CLEAN.sub("", (title or Path(source).stem or "paper"))[:40]
    key = base or f"paper{hashlib.sha1((title or source).encode('utf-8')).hexdigest()[:6]}"
    fields = [f"  title = {{{title or 'Untitled'}}}"]
    year = year or guess_year(filename)
    if year:
        fields.append(f"  year = {{{year}}}")
    fields.append(f"  note = {{Processed by PaperMate from {filename or 'unknown source'}}}")
    return "@misc{{{key},\n{fields}\n}}".format(key=key, fields=",\n".join(fields))
