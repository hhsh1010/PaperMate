"""PaperMate · 文档处理子包 · Pydantic API 响应模型"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SectionOut(BaseModel):
    order: int
    level: int
    heading: str | None = None
    page: int = 0
    text: str = ""


class MetadataOut(BaseModel):
    title: str
    language: str
    chars: int = 0
    words: int = 0
    paragraphs: int = 0
    sentences: int = 0
    headings: int = 0
    max_heading_level: int = 0
    has_code_block: bool = False
    keywords: list[str] = Field(default_factory=list)
    issues: list[dict[str, str]] = Field(default_factory=list)


class ProcessingResultOut(BaseModel):
    filename: str
    source_format: str
    status: str
    title: str = ""
    num_chunks: int = 0
    duration_ms: float = 0.0
    metadata: MetadataOut | None = None
    sections: list[SectionOut] = Field(default_factory=list)
    markdown: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


class JobOut(BaseModel):
    id: str
    filename: str
    status: str
    attempts: int = 0
    max_retries: int = 0
    error: str | None = None
    error_code: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    result: ProcessingResultOut | None = None


class BatchSubmitOut(BaseModel):
    batch_id: str
    total: int
    jobs: list[JobOut]


class StatsOut(BaseModel):
    total: int = 0
    ok: int = 0
    failed: int = 0
    retried: int = 0
    by_format: dict[str, int] = Field(default_factory=dict)
    by_error: dict[str, int] = Field(default_factory=dict)
    avg_chars: float = 0.0
    avg_sections: float = 0.0
    avg_duration_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class VersionOut(BaseModel):
    version: int
    content_hash: str
    fingerprint: str
    summary: str
    created_at: float


class DiffOut(BaseModel):
    left_title: str
    right_title: str
    similarity: float
    added_lines: int
    removed_lines: int
    changed_ratio: float
    unified: str
    summary: str


class ProcessedDocumentOut(BaseModel):
    """单文档处理 + 落库后的对外视图（含结构化产物与版本历史引用）。"""

    doc_id: int
    filename: str
    status: str
    processing: ProcessingResultOut | None = None
    versions: list[VersionOut] = Field(default_factory=list)
