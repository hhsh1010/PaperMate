"""PaperMate · 文档处理子包

覆盖 docs/功能增强规格说明.md 第 2 章（文档处理模块）核心能力 + 用户补充增强：
- 解析（parsers）：Text / Markdown / PDF（页码 A-04）
- 内容提取（metadata）：元信息自动识别 + 校验
- 格式转换（converter）：Markdown / 结构化 JSON / 纯文本 / BibTeX
- 结构化存储（pipeline.artifacts / service 落库）
- 批量异步队列 + 容错重试（queue）
- 差异对比与版本管理（versioning）
- 处理结果与统计摘要（stats）

子包为纯逻辑层（无 FastAPI/SQLAlchemy 依赖），
DB 联动与 REST 挂载由 papermate.services.processing_service 与 api.routes_processing 完成。
"""

from __future__ import annotations

from .models import (
    DiffResult,
    DocumentMetadata,
    Job,
    JobItem,
    ParsedContent,
    ProcessingResult,
    Section,
    StatsSummary,
    ValidationIssue,
    VersionInfo,
)
from .pipeline import DocumentProcessingPipeline
from .queue import InMemoryJobContentStore, JobQueue
from .schemas import (
    BatchSubmitOut,
    DiffOut,
    JobOut,
    MetadataOut,
    ProcessedDocumentOut,
    ProcessingResultOut,
    SectionOut,
    StatsOut,
    VersionOut,
)

__all__ = [
    "DiffResult",
    "DocumentMetadata",
    "Job",
    "JobItem",
    "ParsedContent",
    "ProcessingResult",
    "Section",
    "StatsSummary",
    "ValidationIssue",
    "VersionInfo",
    "DocumentProcessingPipeline",
    "InMemoryJobContentStore",
    "JobQueue",
    "BatchSubmitOut",
    "DiffOut",
    "JobOut",
    "MetadataOut",
    "ProcessedDocumentOut",
    "ProcessingResultOut",
    "SectionOut",
    "StatsOut",
    "VersionOut",
]
