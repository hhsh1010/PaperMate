"""PaperMate · 文档处理子包 · 元信息自动识别与校验（metadata）

自动识别：
- 标题（文件名去扩展名 → 解析结果标题 → 首段启发式）
- 语言（中/英/其他，字符分布启发式）
- 基础统计（字符/英文词数/段落/句子/标题层级）
- 关键词（英文高频词 + 中文二元组，去停用词）
- 是否含代码块（markdown）

校验（识别潜在异常文档，供容错/重试判定）：
- error：empty_document（空内容）、binary_like（乱码/二进制）、too_short（过短）
- warning：title_fallback（标题未识别，用了文件名兜底）、few_headings（无章节结构）

错误码映射（pipeline / queue 依赖）：
- empty_document / binary_like / unsupported_format → fatal（重试无意义）
- 其余 → 可继续处理（warning 级）
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import DocumentMetadata, ParsedContent, ValidationIssue

#: 中文/英文/其他语言判定阈值（中文字符占比）
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")
#: 英文停用词（高频无信息量词）
_EN_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "with", "is", "are",
    "was", "were", "be", "been", "by", "as", "at", "from", "that", "this", "these",
    "those", "it", "its", "we", "our", "you", "your", "they", "their", "he", "she",
    "not", "but", "if", "then", "than", "so", "which", "who", "whom", "have", "has",
    "had", "do", "does", "did", "will", "would", "can", "could", "may", "might", "shall",
    "should", "about", "into", "over", "after", "before", "between", "under", "against",
}
#: 中文停用词（单字高频虚词）
_ZH_STOP = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "我们",
    "你们", "他们", "这", "那", "也", "很", "与", "及", "或", "被", "把", "让", "对", "从",
    "到", "于", "而", "但", "并", "又", "再", "等", "之", "其", "该", "所",
}

#: 最小字符数（低于视为过短）
_MIN_CHARS = 50
#: 乱码/二进制判定：不可打印字符比例
_BINARY_THRESHOLD = 0.05
#: 关键词数量
_TOP_KEYWORDS = 5


def detect_language(text: str) -> str:
    """按字符分布粗判语言：zh（中文字符占比高）/ en / other。"""
    if not text:
        return "other"
    zh = len(_ZH_RE.findall(text))
    en = len(_EN_WORD_RE.findall(text))
    if zh > 0 and zh >= en * 2:
        return "zh"
    if en > 0:
        return "en"
    return "other"


def _extract_keywords(text: str, language: str) -> list[str]:
    """英文取 top-N 高频实词；中文取 top-N 高频二元组。"""
    if language == "zh":
        chars = [c for c in re.sub(r"\s", "", text) if _ZH_RE.match(c)]
        bigrams: list[str] = []
        for i in range(len(chars) - 1):
            bg = chars[i] + chars[i + 1]
            if not (bg[0] in _ZH_STOP or bg[1] in _ZH_STOP):
                bigrams.append(bg)
        from collections import Counter

        return [w for w, _ in Counter(bigrams).most_common(_TOP_KEYWORDS)]
    words = [w.lower() for w in _EN_WORD_RE.findall(text) if w.lower() not in _EN_STOP and len(w) > 1]
    from collections import Counter

    return [w for w, _ in Counter(words).most_common(_TOP_KEYWORDS)]


def _count_sentences(text: str) -> int:
    return max(1, len(re.findall(r"[。！？!?；;\.]", text)))


def extract_metadata(content: ParsedContent, filename: str) -> DocumentMetadata:
    """从解析结果 + 文件名自动识别元信息。"""
    text = content.raw_text
    language = detect_language(text)
    words = len(_EN_WORD_RE.findall(text)) if language != "zh" else len(re.sub(r"\s", "", text))
    headings = [s.heading for s in content.sections if s.heading]
    issues: list[ValidationIssue] = []

    # 标题识别链：解析结果标题 → 文件名（去扩展名）→ 兜底
    title = content.title
    fallback_name = Path(filename).stem.strip() or "未命名文档"
    if not title or title == "未命名文档" or title == Path(filename).stem:
        if fallback_name != "未命名文档":
            title = fallback_name
            issues.append(ValidationIssue("warning", "title_fallback", "未能从内容识别标题，已用文件名兜底"))

    # 校验
    if not text.strip():
        issues.append(ValidationIssue("error", "empty_document", "文档内容为空，无法处理"))
    elif _binary_ratio(text) > _BINARY_THRESHOLD:
        issues.append(ValidationIssue("error", "binary_like", "内容疑似二进制/乱码（不可打印字符占比过高）"))
    elif len(text.strip()) < _MIN_CHARS:
        issues.append(ValidationIssue("error", "too_short", f"内容过短（{len(text.strip())} 字符 < {_MIN_CHARS}），可能不是有效文档"))
    if not headings:
        issues.append(ValidationIssue("warning", "few_headings", "未识别到章节标题，文档可能缺少结构"))

    return DocumentMetadata(
        title=title,
        language=language,
        chars=len(text),
        words=words,
        paragraphs=content.stats.get("paragraphs", 0),
        sentences=_count_sentences(text),
        headings=len(headings),
        max_heading_level=max((s.level for s in content.sections), default=0),
        has_code_block="```" in text,
        keywords=_extract_keywords(text, language),
        issues=issues,
    )


def _binary_ratio(text: str) -> float:
    """不可打印字符占比（用于乱码/二进制检测）。"""
    if not text:
        return 0.0
    sample = text[:2000]
    bad = sum(1 for ch in sample if ord(ch) < 9 or (13 < ord(ch) < 32) or ord(ch) == 0xFFFD)
    return bad / len(sample)


#: 判定是否为「重试无意义」的致命错误码
FATAL_ERROR_CODES = {"empty_document", "binary_like", "unsupported_format", "decode_error", "too_short"}


def is_fatal(code: str | None) -> bool:
    """错误码 → 是否致命（致命错误重试无意义，直接 failed）。"""
    return code in FATAL_ERROR_CODES
