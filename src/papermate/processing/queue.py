"""PaperMate · 文档处理子包 · 批量异步任务队列（queue）

基于 asyncio 的内存任务队列，零外部依赖：
- submit(items) 批量提交，每个文件一个 Job（pending → running → done/failed）
- 并发 worker（max_workers），处理走 asyncio.to_thread 不阻塞事件循环
- 容错：单 Job 失败不影响其他 Job（失败隔离）
- 重试：可重试错误（非致命，见 metadata.is_fatal）按指数退避重试，
  达到 max_retries 后标记 failed（带 error/error_code）
- start()/stop() 管理 worker 生命周期；wait_idle() 供测试与调用方等待排空

内容存储：Job 与原始字节解耦，通过 InMemoryJobContentStore 维护（可替换为落盘实现）。
设计约定：queue 只依赖 pipeline 与 models；结果保存在内存（个人使用规模），
如需跨进程持久化可把 Job.result 序列化落库（上层 service 负责）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .metadata import is_fatal
from .models import Job, JobItem, ProcessingResult

logger = logging.getLogger(__name__)


def _restore_result(d: dict[str, Any]) -> SimpleNamespace:
    """归档 JSON → 支持属性访问的结果对象（H3 修复 2026-09-10）。

    用 SimpleNamespace 而非重建 ProcessingResult dataclass：归档保存的是「API 视图」，
    只需满足 `_to_result_out` 的属性访问契约（`meta.issues[i].severity`、
    `sections[i].order` 等）。**注意** `structured`/`warnings`/`artifacts` 必须保持原生
    dict/list —— 它们直接喂给 pydantic 字段，包成 namespace 会导致校验失败。
    """
    meta = d.get("metadata")
    return SimpleNamespace(
        filename=d.get("filename"),
        source_format=d.get("source_format"),
        status=d.get("status"),
        title=d.get("title"),
        num_chunks=d.get("num_chunks"),
        duration_ms=d.get("duration_ms"),
        metadata=(
            SimpleNamespace(
                **{k: v for k, v in meta.items() if k != "issues"},
                issues=[SimpleNamespace(**i) for i in (meta.get("issues") or [])],
            )
            if isinstance(meta, dict)
            else None
        ),
        sections=[SimpleNamespace(**s) for s in (d.get("sections") or [])],
        markdown=d.get("markdown"),
        structured=d.get("structured"),
        warnings=d.get("warnings") or [],
        error=d.get("error"),
        error_code=d.get("error_code"),
        artifacts=d.get("artifacts") or {},
    )


class InMemoryJobContentStore:
    """提交内容的内存存储：job_id → bytes（与 queue 解耦，便于替换为落盘实现）。"""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, job_id: str, content: bytes) -> None:
        self._store[job_id] = content

    def get(self, job_id: str) -> bytes:
        return self._store[job_id]

    def drop(self, job_id: str) -> None:
        self._store.pop(job_id, None)


class JobQueue:
    def __init__(
        self,
        processor: Callable[[str, bytes], ProcessingResult],
        max_workers: int = 2,
        max_retries: int = 2,
        retry_delay: float = 0.05,
        content_store: InMemoryJobContentStore | None = None,
        archive_dir: Path | None = None,
    ) -> None:
        self._processor = processor
        self._max_workers = max(1, max_workers)
        self._max_retries = max(0, max_retries)
        self._retry_delay = max(0.0, retry_delay)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._contents = content_store or InMemoryJobContentStore()
        # H3（2026-09-10）：终态裁剪前的归档目录——批量 job 只存内存，裁剪后 get() 会
        # 丢历史（路由 404）。None = 不归档（保持既有行为，便于测试/嵌入式使用）。
        self._archive_dir = archive_dir
        self._pending = 0
        self._workers: list[asyncio.Task] = []
        # 等待退避重投的 job_id → TimerHandle（保存句柄，stop() 才能取消已排定的定时器）
        self._retrying: dict[str, asyncio.TimerHandle] = {}
        self._started = False

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        """启动 worker 协程（幂等）。"""
        if self._started:
            return
        self._started = True
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"doc-worker-{i}") for i in range(self._max_workers)
        ]

    async def stop(self) -> None:
        """停止 worker 并等待结束（幂等）。

        2026-09-08 修复两处「停止后状态永不收敛」：
        1. 未触发的重试定时器（call_later）必须先 cancel——否则回调在队列停止后触发，
           `_requeue` 因 `not _started` 直接 return，job 永久停留 retrying 且字节不释放。
        2. 兜底把仍未到终态的 job 标记为 cancelled——worker 被 cancel 时若 job 正卡在
           running/pending/retrying，wait_idle() 会永久等待。
        """
        if not self._started:
            return
        self._started = False
        for h in list(self._retrying.values()):
            h.cancel()
        self._retrying.clear()
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            try:
                await w
            except asyncio.CancelledError:
                pass
        self._workers = []
        for j in self._jobs.values():
            if j.status not in ("done", "failed"):
                self._mark_failed(j, "cancelled", "队列停止，任务已取消")

    async def wait_idle(self, timeout: float = 10.0) -> None:
        """等待所有已提交任务到达终态（done/failed）。测试与同步调用方使用。

        2026-09-06 修复竞态：旧实现判 `_pending == 0 and queue.empty() and not _retrying`，
        但 worker 从队列取出最后一项（pending→0）与任务真正跑完之间存在窗口——wait_idle
        在任务仍 running 时误判空闲，调用方（如测试的 _drain）随即 stop() 把 worker 连同
        运行中的任务一起取消 → 任务永久卡在 running。现改为直接校验 Job 状态：
        终态由 worker 单点推进，无快照窗口。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(j.status in ("done", "failed") for j in self._jobs.values()):
                return
            await asyncio.sleep(0.01)
        raise TimeoutError("任务队列等待超时")

    # ---------- 提交与查询 ----------

    def submit(self, items: list[JobItem]) -> list[Job]:
        """批量提交任务，返回按提交顺序的 Job 列表。"""
        jobs: list[Job] = []
        for item in items:
            job = Job(id=uuid.uuid4().hex[:12], filename=item.filename, max_retries=self._max_retries)
            self._jobs[job.id] = job
            self._contents.put(job.id, item.content)
            self._queue.put_nowait(job.id)
            self._pending += 1
            jobs.append(job)
        return jobs

    def get(self, job_id: str) -> Job | None:
        """按 id 取任务；内存未命中时回落磁盘归档（H3 修复 2026-09-10）。

        返回对象满足与 `Job` 相同的**序列化契约**（`_to_job_out` / `_to_result_out`
        的属性访问），但它可能是归档还原的 SimpleNamespace——调用方不应依赖
        `isinstance(x, Job)`。
        """
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        return self._load_archived(job_id)

    def _archive_job(self, job: Job) -> None:
        """终态裁剪前把 job 快照写入磁盘（H3）。失败不阻断裁剪（记 warning）。"""
        if self._archive_dir is None:
            return
        try:
            self._archive_dir.mkdir(parents=True, exist_ok=True)
            path = self._archive_dir / f"{job.id}.json"
            path.write_text(json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001  归档是尽力而为的兜底，不应影响队列主流程
            logger.warning("任务 %s 归档失败（裁剪后将不可恢复）", job.id, exc_info=True)

    def _load_archived(self, job_id: str):
        """从归档 JSON 还原 job 视图；无归档/损坏 → None（H3）。"""
        if self._archive_dir is None:
            return None
        # job_id 由 uuid4 生成，仍做文件名净化防目录穿越（防御性）
        if not job_id or Path(job_id).name != job_id:
            return None
        path = self._archive_dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001  归档损坏等同不存在
            logger.warning("任务 %s 归档文件损坏，忽略", job_id, exc_info=True)
            return None
        raw_result = d.get("result")
        return SimpleNamespace(
            id=d.get("id"),
            filename=d.get("filename"),
            status=d.get("status"),
            attempts=d.get("attempts", 0),
            max_retries=d.get("max_retries", 0),
            error=d.get("error"),
            error_code=d.get("error_code"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            result=_restore_result(raw_result) if isinstance(raw_result, dict) else None,
        )

    def list(self, status: str | None = None, limit: int = 50) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs[:limit]

    def stats(self) -> dict:
        """队列概览：按状态计数。"""
        counts = {"pending": 0, "running": 0, "retrying": 0, "done": 0, "failed": 0}
        for j in self._jobs.values():
            counts[j.status] = counts.get(j.status, 0) + 1
        counts["total"] = len(self._jobs)
        return counts

    # ---------- 内部 ----------

    async def _worker(self, _index: int) -> None:
        while True:
            job_id = await self._queue.get()
            self._pending -= 1
            try:
                await self._run_job(job_id)
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        job.status = "running"
        job.attempts += 1
        job.updated_at = time.time()
        try:
            result: ProcessingResult = await asyncio.to_thread(
                self._processor, job.filename, self._contents.get(job_id)
            )
        except asyncio.CancelledError:
            # 队列停止（stop()）时运行中的任务被取消：必须显式落终态再重抛。
            # `except Exception` 捕不到 CancelledError（它继承自 BaseException），
            # 否则 job 会永久停留 running，wait_idle() 后续调用永久超时。
            self._mark_failed(job, "cancelled", "队列停止，任务已取消")
            raise
        except Exception as exc:  # noqa: BLE001  处理层未捕获的异常（防御性兜底）
            self._mark_failed(job, "internal_error", f"内部错误：{exc}")
            return
        job.result = result
        job.updated_at = time.time()
        if result.status == "failed":
            if not is_fatal(result.error_code) and job.attempts < job.max_retries:
                self._schedule_retry(job_id, job.attempts)
                return  # 注意：retrying 的内容字节必须保留（重投后还要用），不能在此 drop
            self._mark_failed(job, result.error_code, result.error)
        else:
            job.status = "done"
            job.error = None
            self._contents.drop(job_id)  # 终态后内容字节不再需要（2026-09-06：此前滞留到进程结束）
            self._prune_terminal_jobs()

    def _prune_terminal_jobs(self, keep: int = 200) -> None:
        """终态 Job 只保留最近 keep 条（2026-09-06：此前 _jobs 只增不减）。"""
        if len(self._jobs) <= keep:
            return
        excess = len(self._jobs) - keep
        terminal = sorted(
            (j for j in self._jobs.values() if j.status in ("done", "failed")),
            key=lambda j: j.updated_at,
        )
        for j in terminal[:excess]:
            # H3（2026-09-10）：淘汰前先归档——批量路径的 job 只存内存，直接 pop 会让
            # 已完成任务在 get() 处 404 且结果不可恢复。
            self._archive_job(j)
            self._jobs.pop(j.id, None)
            self._contents.drop(j.id)

    def _schedule_retry(self, job_id: str, attempt: int) -> None:
        job = self._jobs[job_id]
        job.status = "retrying"
        job.updated_at = time.time()
        delay = self._retry_delay * (2 ** (attempt - 1))
        # 保存 TimerHandle：stop() 必须能取消尚未触发的重投，否则见 _requeue 说明
        self._retrying[job_id] = asyncio.get_running_loop().call_later(
            delay, lambda: self._requeue(job_id)
        )

    def _requeue(self, job_id: str) -> None:
        self._retrying.pop(job_id, None)
        # 定时器已触发但队列已停止（stop() 未来得及 cancel 的竞态窗口）：
        # 必须在此落终态并释放字节，否则 job 永久停留 retrying。
        if not self._started:
            job = self._jobs.get(job_id)
            if job is not None and job.status == "retrying":
                self._mark_failed(job, "cancelled", "队列停止，重试已取消")
            return
        job = self._jobs.get(job_id)
        if job is None:  # 已被终态裁剪（_prune_terminal_jobs）移除
            return
        if job.status == "retrying":
            job.status = "pending"
            self._queue.put_nowait(job_id)
            self._pending += 1

    def _mark_failed(self, job: Job, code: str | None, message: str | None) -> None:
        job.status = "failed"
        job.error = message or "处理失败"
        job.error_code = code
        job.updated_at = time.time()
        self._contents.drop(job.id)  # 终态后内容字节不再需要
        self._prune_terminal_jobs()
