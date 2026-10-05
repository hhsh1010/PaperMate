"""PaperMate · 文档处理子包 · 纯数据模型（dataclass，零外部依赖）

文档处理模块的核心表示：
- ParsedContent：一次解析的结果（来源格式、标题、结构化章节、原始文本、页数）
- ProcessingResult：一次完整处理（解析→提取→元信息→转换→存储）的结果
- Job / JobItem：批量异步任务队列的条目
- DiffResult / VersionInfo：差异对比与版本管理
- ValidationIssue：元信息校验问题（error / warning）

设计约定：本模块为纯逻辑层，不依赖 SQLAlchemy / FastAPI / pymupdf，
便于离线单测与独立复用；与 DB / 落盘的联动由上层 processing_service 完成。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------- 解析 / 结构化表示 ----------

#: 支持的来源格式
SOURCE_FORMATS = ("text", "markdown", "pdf")

#: 结构化章节
@dataclass(slots=True)
class Section:
    level: int  # 1..6（markdown 标题层级）；0 = 无标题段落
    heading: str | None  # 标题文本；level=0 时为 None
    page: int  # 页码（PDF 从 1 起；文本类为 0）
    text: str  # 该节正文（不含标题）
    order: int  # 出现顺序（1 起）


@dataclass(slots=True)
class ParsedContent:
    source_format: str  # text | markdown | pdf
    title: str
    sections: list[Section]
    raw_text: str
    num_pages: int = 0
    stats: dict[str, Any] = field(default_factory=dict)  # 见 extract_stats()

    @property
    def headings(self) -> list[str]:
        return [s.heading for s in self.sections if s.heading]


# ---------- 元信息识别与校验 ----------

@dataclass(slots=True)
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str  # 机器可读错误码，如 empty_document
    message: str  # 人类可读说明


@dataclass(slots=True)
class DocumentMetadata:
    title: str
    language: str  # zh | en | other
    chars: int
    words: int  # 英文词数（中文按字计）
    paragraphs: int
    sentences: int
    headings: int
    max_heading_level: int
    has_code_block: bool
    keywords: list[str]  # top-N 高频词（英文词/中文二元组）
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


# ---------- 处理结果 ----------

@dataclass(slots=True)
class ProcessingResult:
    filename: str
    source_format: str
    status: str  # ok | failed
    title: str
    metadata: DocumentMetadata | None
    sections: list[Section]
    num_chunks: int  # 按 300-500 token 语义分块后的块数（供 RAG 对接）
    structured: dict[str, Any]  # 结构化产物（sections + metadata 的 JSON 可序列化视图）
    markdown: str  # 转换出的 Markdown
    duration_ms: float
    error: str | None = None
    error_code: str | None = None
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)  # 落盘产物路径（json/md/txt）


# ---------- 批量任务队列 ----------

@dataclass(slots=True)
class JobItem:
    filename: str
    content: bytes


@dataclass(slots=True)
class Job:
    id: str
    filename: str
    status: str = "pending"  # pending | running | retrying | done | failed
    attempts: int = 0
    max_retries: int = 2
    error: str | None = None
    error_code: str | None = None
    result: ProcessingResult | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ---------- 差异对比 / 版本管理 ----------

@dataclass(slots=True)
class DiffResult:
    left_title: str
    right_title: str
    similarity: float  # 0..1（SequenceMatcher）
    added_lines: int
    removed_lines: int
    changed_ratio: float  # (added + removed) / total
    unified: str  # unified diff 文本（截断）
    summary: str  # 一句话变更摘要


@dataclass(slots=True)
class VersionInfo:
    version: int  # 1 起递增
    content_hash: str  # sha256（原文）
    fingerprint: str  # 归一化指纹
    summary: str  # 相对上一版本的变更摘要
    created_at: float = field(default_factory=time.time)


# ---------- 统计摘要 ----------

@dataclass(slots=True)
class StatsSummary:
    total: int
    ok: int
    failed: int
    retried: int
    by_format: dict[str, int]
    by_error: dict[str, int]
    avg_chars: float
    avg_sections: float
    avg_duration_ms: float
    warnings: list[str] = field(default_factory=list)
