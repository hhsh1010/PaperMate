"""PaperMate · 文档处理子包 · 处理结果与统计摘要（stats）

- result_summary(result)：单文档处理的统计卡片（字符/章节/标题/耗时/状态）
- summarize(results)：批量聚合（成功/失败/重试计数、格式分布、错误分布、均值）
- queue_stats_to_summary(jobs)：把 Job 列表转成 StatsSummary（含 retried 计数）

设计约定：纯函数，输入 ProcessingResult / Job，输出可序列化 dict 或 StatsSummary。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import Job, ProcessingResult, StatsSummary


def result_summary(result: ProcessingResult) -> dict[str, Any]:
    """单文档处理统计卡片（JSON 可序列化）。"""
    meta = result.metadata
    return {
        "filename": result.filename,
        "status": result.status,
        "source_format": result.source_format,
        "title": result.title,
        "duration_ms": result.duration_ms,
        "chars": meta.chars if meta else 0,
        "words": meta.words if meta else 0,
        "paragraphs": meta.paragraphs if meta else 0,
        "sentences": meta.sentences if meta else 0,
        "headings": meta.headings if meta else 0,
        "num_sections": len(result.sections),
        "num_chunks": result.num_chunks,
        "keywords": meta.keywords if meta else [],
        "warnings": result.warnings,
        "error": result.error,
        "error_code": result.error_code,
    }


def summarize(results: Iterable[ProcessingResult]) -> StatsSummary:
    """聚合多个处理结果：计数、均值、格式/错误分布。"""
    results = list(results)
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status == "failed"]
    by_format: dict[str, int] = {}
    by_error: dict[str, int] = {}
    for r in results:
        by_format[r.source_format] = by_format.get(r.source_format, 0) + 1
        if r.status == "failed":
            by_error[r.error_code or "unknown"] = by_error.get(r.error_code or "unknown", 0) + 1
    return StatsSummary(
        total=len(results),
        ok=len(ok),
        failed=len(failed),
        retried=0,  # 由队列层补记（见 queue_stats_to_summary）
        by_format=by_format,
        by_error=by_error,
        avg_chars=round(sum(r.metadata.chars for r in ok if r.metadata) / len(ok), 1) if ok else 0.0,
        avg_sections=round(sum(len(r.sections) for r in ok) / len(ok), 1) if ok else 0.0,
        avg_duration_ms=round(sum(r.duration_ms for r in ok) / len(ok), 1) if ok else 0.0,
        warnings=[w for r in ok for w in r.warnings][:10],
    )


def queue_stats_to_summary(jobs: Iterable[Job]) -> StatsSummary:
    """把 Job 列表转成统计摘要（retried = attempts>1 的 done/failed 数）。"""
    jobs = list(jobs)
    results = [j.result for j in jobs if j.result is not None]
    summary = summarize(results)
    summary.retried = sum(1 for j in jobs if j.attempts > 1 and j.status in ("done", "failed"))
    return summary


def summary_to_dict(summary: StatsSummary) -> dict[str, Any]:
    return {
        "total": summary.total,
        "ok": summary.ok,
        "failed": summary.failed,
        "retried": summary.retried,
        "by_format": summary.by_format,
        "by_error": summary.by_error,
        "avg_chars": summary.avg_chars,
        "avg_sections": summary.avg_sections,
        "avg_duration_ms": summary.avg_duration_ms,
        "warnings": summary.warnings,
    }
