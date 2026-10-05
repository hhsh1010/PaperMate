"""PaperMate · 文档处理门面服务（组合 processing 子包 + DB 持久化）

职责：
- 单文档处理（process_bytes：同步路径，可落产物文件）
- 批量异步队列（submit_batch / list_jobs / job_stats，懒启动 worker）
- 持久化：处理结果 → documents.meta_json["processing"]；产物 → processed_dir/{doc_id}/
- 版本管理：documents.meta_json["versions"]（零迁移）；与上一版本自动 diff
- 差异对比：两文档 parsed 文本 diff
- 导出：md/json/txt 从落盘产物读取（缺失时从 meta 重建）

纯逻辑在 papermate.processing 子包；本服务仅做组合 + DB 读写（write_with_retry 防沙箱锁）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from sqlalchemy.orm import sessionmaker

from ..config import Settings
from ..models.db import Document
from ..processing.models import DiffResult, Job, JobItem, ProcessingResult, VersionInfo
from ..processing.pipeline import DocumentProcessingPipeline
from ..processing.queue import JobQueue
from ..processing.versioning import DocumentVersioner, diff_texts
from .db_retry import write_with_retry
from .document_service import DocumentService

#: 导出格式 → 产物文件名
_ARTIFACT_FILES = {"md": "document.md", "json": "document.json", "txt": "document.txt"}
#: 支持的导出格式
EXPORT_FORMATS = ("md", "json", "txt")

#: 文档处理的**中间态**（进程重启后必然已中断，C5 启动恢复的对象）。
#: 与 `DocumentService` 写的状态字符串保持一致。
IN_FLIGHT_STATUSES = ("uploaded", "parsing", "chunking", "embedding")

#: 终态：这些状态不会再自行变化（`ready` 可用 / `failed` 可重试 / `cancelled` 已取消）
TERMINAL_STATUSES = frozenset({"ready", "failed", "cancelled"})

logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self, doc_svc: DocumentService, settings: Settings, SessionLocal: sessionmaker) -> None:
        self._doc_svc = doc_svc
        self._settings = settings
        self._Session = SessionLocal
        self._pipeline = DocumentProcessingPipeline()
        self._queue = JobQueue(
            processor=self._pipeline.process,
            # H3（2026-09-10）：终态 job 裁剪前归档，避免批量已完成任务 404 / 结果丢失
            archive_dir=settings.resolved_data_dir / "job_archive",
        )
        self._queue_started = False

    # ---------- 单文档处理（同步，可在线程中执行） ----------

    def process_bytes(self, filename: str, content: bytes, doc_id: int | None = None) -> ProcessingResult:
        """处理单个文档；doc_id 给定则把产物落盘到 processed_dir/{doc_id}/。"""
        out_dir = self._settings.processed_dir / str(doc_id) if doc_id is not None else None
        return self._pipeline.process(filename, content, output_dir=out_dir)

    # ---------- 批量异步队列 ----------

    async def submit_batch(self, files: list[tuple[str, bytes]]) -> list[Job]:
        """提交批量任务；worker 懒启动（首次提交时）。返回 Job 列表。"""
        if not self._queue_started:
            await self._queue.start()
            self._queue_started = True
        items = [JobItem(filename=name, content=content) for name, content in files]
        return self._queue.submit(items)

    async def wait_idle(self, timeout: float = 30.0) -> None:
        """等待队列排空（供测试与同步调用方）。"""
        if self._queue_started:
            await self._queue.wait_idle(timeout)

    async def shutdown(self) -> None:
        """停止 worker（FastAPI shutdown 事件调用）。"""
        if self._queue_started:
            await self._queue.stop()
            self._queue_started = False

    # ---------- C5（2026-09-21）：中断任务的启动恢复 ----------

    def recover_interrupted(self) -> dict[str, Any]:
        """启动时把「非终态」文档标记为 `failed`。

        **背景**：上传流水线是 `uploaded → parsing → chunking → embedding → ready/failed/cancelled`
        （状态由 `DocumentService` 写库）。这些状态**只存在于进程内存中驱动** ——
        进程一旦重启（崩溃 / 手动重启 / 部署），正在处理的文档会**永久停在中间态**：
        - 用户界面上它一直显示"处理中"，没有任何原因说明；
        - 也没有重试入口（既不是 failed 也不是 ready）→ 只能删了重传。

        **为什么标 `failed` 而不是重新入队**：
        1. 重启后内存里的任务已丢失，**无法续跑**（只能从头再来）；
        2. embedding 可能很慢（实测单文档重建 742s），静默重跑会让启动变慢且用户不知情；
        3. 标记 failed 后用户**看得见原因、点得动重试** —— 这比"静默卡住"好。

        ⚠️ **前提：单实例**。本项目定位本地单用户（同时只有一个后端进程）。
        多实例部署下本方法会误标其它实例正在处理的任务 —— 那时必须改成带 owner/心跳的方案。

        返回：`{"recovered": n, "documents": [(id, filename, from_status), ...]}`
        """
        rows: list[tuple[int, str, str]] = []
        try:
            with self._Session() as sess:
                stuck = (
                    sess.query(Document)
                    .filter(Document.status.in_(IN_FLIGHT_STATUSES))
                    .all()
                )
                rows = [(d.id, d.filename or "", d.status) for d in stuck]
                if rows:
                    # 只改 status —— 与既有 failed 路径一致：`documents` 表**没有 error 列**，
                    # 失败原因此前只存在于内存进度里（`_set_upload_progress`，重启即丢）。
                    # ⚠️ 因此中间态恢复后用户**看不到具体原因**（只能看到"失败、可重试"）。
                    # 这是既有的可追溯性缺口，已记入 docs/实施记录.md 遗留，不在本项内扩大改动面。
                    for d in stuck:
                        d.status = "failed"
                    sess.commit()
        except Exception:  # noqa: BLE001  恢复失败不得阻断启动（有迁移前快照兜底）
            logger.warning("中断任务恢复失败（服务继续启动，中间态文档需手工处理）", exc_info=True)

        if rows:
            logger.warning(
                "C5 启动恢复：%d 个文档停留在中间态，已标记为 failed（原因：服务重启）→ %s",
                len(rows),
                [(i, f, st) for i, f, st in rows[:5]],
            )
        return {"recovered": len(rows), "documents": rows}

    def get_job(self, job_id: str) -> Job | None:
        return self._queue.get(job_id)

    def list_jobs(self, status: str | None = None, limit: int = 50) -> list[Job]:
        return self._queue.list(status=status, limit=limit)

    def job_stats(self) -> dict[str, Any]:
        return self._queue.stats()

    # ---------- 持久化（处理结果 → meta_json + 产物落盘） ----------

    def persist_processing(self, doc_id: int, result: ProcessingResult) -> None:
        """把处理结果写入 documents.meta_json["processing"]；产物文件在 process_bytes 时已落盘。"""

        def _persist() -> None:
            with self._Session() as db:
                d = db.get(Document, doc_id)
                if d is None:
                    return
                meta = _load_meta(d)
                meta["processing"] = _result_to_dict(result)
                d.meta_json = json.dumps(meta, ensure_ascii=False)
                db.commit()

        write_with_retry(_persist)

    def get_processing(self, doc_id: int) -> dict[str, Any] | None:
        """读取已持久化的处理结果（meta_json["processing"]）。"""
        with self._Session() as db:
            d = db.get(Document, doc_id)
            if d is None:
                return None
            meta = _load_meta(d)
        return meta.get("processing")

    # ---------- 版本管理 ----------

    def register_version(
        self, doc_id: int, text: str, raw_content: bytes, prev_text: str | None = None
    ) -> VersionInfo:
        """注册新版本（与上一版本自动 diff 摘要）；重复指纹自动跳过。

        prev_text 应传**本次处理之前**的解析文本作为 diff 基线（2026-09-08 修复 H4）：
        未显式传入时退化为读取文档当前文本，而该文本往往已被新内容覆盖，
        导致 `prev_text == text`、similarity 恒为 1.0 的错误摘要。
        """
        versions, meta = self._load_versions(doc_id)
        if prev_text is None and versions:
            prev_text = self._doc_svc.get_parsed_text(doc_id)
        info = DocumentVersioner.register(versions, text, raw_content, prev_text=prev_text)
        self._save_versions(doc_id, versions, meta)
        return info

    def list_versions(self, doc_id: int) -> list[VersionInfo]:
        versions, _ = self._load_versions(doc_id)
        return DocumentVersioner.list_versions(versions)

    def _load_versions(self, doc_id: int) -> tuple[list[VersionInfo], dict]:
        with self._Session() as db:
            d = db.get(Document, doc_id)
            if d is None:
                return [], {}
            meta = _load_meta(d)
        versions = [VersionInfo(**v) for v in meta.get("versions", [])]
        return versions, meta

    def _save_versions(self, doc_id: int, versions: list[VersionInfo], meta: dict) -> None:
        def _save() -> None:
            with self._Session() as db:
                d = db.get(Document, doc_id)
                if d is None:
                    return
                m = _load_meta(d)
                m["versions"] = [_version_to_dict(v) for v in versions]
                d.meta_json = json.dumps(m, ensure_ascii=False)
                db.commit()

        write_with_retry(_save)

    # ---------- 差异对比 / 导出 ----------

    def compare_documents(self, left_id: int, right_id: int) -> DiffResult:
        """两个文档的 parsed 文本差异对比。"""
        lt = self._doc_svc.get_parsed_text(left_id) or ""
        rt = self._doc_svc.get_parsed_text(right_id) or ""
        left = self._doc_svc.get_document(left_id)
        right = self._doc_svc.get_document(right_id)
        lname = left.filename if left else f"doc-{left_id}"
        rname = right.filename if right else f"doc-{right_id}"
        return diff_texts(lt, rt, left_title=f"{lname}(#{left_id})", right_title=f"{rname}(#{right_id})")

    def export(self, doc_id: int, fmt: str) -> tuple[str, str] | None:
        """导出转换结果：返回 (content_type, body)；不支持/缺失返回 None。"""
        if fmt not in EXPORT_FORMATS:
            return None
        artifact = self._settings.processed_dir / str(doc_id) / _ARTIFACT_FILES[fmt]
        if artifact.exists():
            return ("application/json; charset=utf-8" if fmt == "json" else "text/plain; charset=utf-8", artifact.read_text(encoding="utf-8"))
        # 回退 1：从 meta_json["processing"] 重建（同步 process 流程写入）
        proc = self.get_processing(doc_id)
        if proc is not None:
            if fmt == "json":
                return ("application/json; charset=utf-8", json.dumps(proc.get("structured", {}), ensure_ascii=False, indent=2))
            if fmt == "md":
                return ("text/plain; charset=utf-8", proc.get("markdown", ""))
            return ("text/plain; charset=utf-8", proc.get("markdown", "").replace("# ", "").replace("## ", ""))
        # 回退 2（2026-09-05）：异步上传管线不写 meta_json["processing"]，导致所有新上传文档导出
        # 一律 404。但只要文档已就绪，就必然有解析文本（parsed_dir 落盘 → chunk 表回退，见
        # DocumentService.get_parsed_text，/text 预览与版本对比同源）——直接以解析文本作 Markdown
        # 导出（json 含标题+正文；txt 去标记），保证导出对所有就绪文档可用。
        text = (self._doc_svc.get_parsed_text(doc_id) or "").strip()
        if not text:
            return None
        doc = self._doc_svc.get_document(doc_id)
        title = doc.filename.rsplit(".", 1)[0] if doc and doc.filename else f"doc-{doc_id}"
        md = f"# {title}\n\n{text}\n"
        if fmt == "json":
            return ("application/json; charset=utf-8", json.dumps({"title": title, "markdown": md}, ensure_ascii=False, indent=2))
        if fmt == "txt":
            return ("text/plain; charset=utf-8", md.replace("# ", ""))
        return ("text/plain; charset=utf-8", md)

    # ---------- 批处理结果统计 ----------

    def batch_summary(self) -> dict[str, Any]:
        """队列任务统计摘要（含重试计数）。"""
        from ..processing.stats import queue_stats_to_summary, summary_to_dict

        return summary_to_dict(queue_stats_to_summary(self._queue.list(limit=1000)))


def _load_meta(d: Document) -> dict:
    try:
        return json.loads(d.meta_json or "{}")
    except json.JSONDecodeError:
        return {}


def _result_to_dict(result: ProcessingResult) -> dict[str, Any]:
    """ProcessingResult → 可序列化字典（meta_json 存储视图，不存 sections 大字段文本冗余）。"""
    meta = result.metadata
    return {
        "filename": result.filename,
        "source_format": result.source_format,
        "status": result.status,
        "title": result.title,
        "num_chunks": result.num_chunks,
        "duration_ms": result.duration_ms,
        "processed_at": time.time(),
        "metadata": {
            "language": meta.language if meta else None,
            "chars": meta.chars if meta else 0,
            "words": meta.words if meta else 0,
            "paragraphs": meta.paragraphs if meta else 0,
            "sentences": meta.sentences if meta else 0,
            "headings": meta.headings if meta else 0,
            "has_code_block": meta.has_code_block if meta else False,
            "keywords": meta.keywords if meta else [],
        }
        if meta
        else None,
        "warnings": result.warnings,
        "error": result.error,
        "error_code": result.error_code,
        "markdown": result.markdown,
        "structured": result.structured,
    }


def _version_to_dict(v: VersionInfo) -> dict[str, Any]:
    return {
        "version": v.version,
        "content_hash": v.content_hash,
        "fingerprint": v.fingerprint,
        "summary": v.summary,
        "created_at": v.created_at,
    }
