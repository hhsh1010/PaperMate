"""JobQueue 停止路径回归测试（2026-09-08 补审查 H1 / H2）

背景：`stop()` 路径存在两处「状态永不收敛」缺陷——

* **H1**：`_run_job` 的 `except Exception` **捕不到 `CancelledError`**（它继承自 BaseException），
  worker 被取消后 job 永久停留 `running`，后续 `wait_idle()` 永久超时。
* **H2**：`_retrying` 原为 `set[str]`，未保存 `call_later` 的 `TimerHandle`，
  `stop()` 无法取消已排定的重投；而 `_requeue` 遇 `not _started` 直接 return，
  job 永久停留 `retrying` 且内容字节不释放。

## 为什么前两个用例要「绕开 stop() 兜底」

首次编写的用例直接调 `stop()`，结果**即使把 H1 修复回退，测试依然通过**——
因为 `stop()` 末尾的兜底（把非终态 job 统一标 cancelled）会独立修复症状，
把 `_run_job` 的缺陷完全掩盖，测试退化为空跑。

因此这里改为**精确断言**：
* `test_cancelled_worker_marks_job_terminal` 只取消 worker 任务本身，**不调用 `stop()`**，
  故观察到的终态只能来自 `_run_job` 自己的 `CancelledError` 处理；
* `test_requeue_after_stop_marks_job_cancelled` 直接手动触发 `_requeue`，
  不经过 `stop()` 的兜底遍历。

> 教义：**修复 + 兜底同时引入时，若只测端到端结果，兜底会掩盖被修复的缺陷本身。**
> 回归测试必须断言到「具体哪段代码产生了这个结果」，否则等于没测。
"""

from __future__ import annotations

import asyncio
import threading

from papermate.processing.models import JobItem, ProcessingResult
from papermate.processing.queue import JobQueue


def _result(
    filename: str,
    *,
    status: str = "ok",
    error: str | None = None,
    error_code: str | None = None,
) -> ProcessingResult:
    return ProcessingResult(
        filename=filename,
        source_format="txt",
        status=status,
        title="t",
        metadata=None,
        sections=[],
        num_chunks=1,
        structured={},
        markdown="",
        duration_ms=1.0,
        error=error,
        error_code=error_code,
    )


async def _wait_status(job, want: str, timeout: float = 3.0) -> bool:
    """轮询等待 job 到达指定状态（避免依赖固定 sleep 时长的脆弱时序）。"""
    step = 0.01
    for _ in range(int(timeout / step)):
        if job.status == want:
            return True
        await asyncio.sleep(step)
    return job.status == want


def test_cancelled_worker_marks_job_terminal() -> None:
    """H1（精确）：单独取消 worker 时，`_run_job` 必须自己把 job 落到终态。

    刻意**不调用 `stop()`**，以排除 stop() 末尾兜底标记的干扰——
    修复前 job 会永久停留 `running`，断言失败。
    """
    gate = threading.Event()

    def slow_processor(filename: str, content: bytes) -> ProcessingResult:
        gate.wait(timeout=5)  # 卡在 worker 线程里，模拟长任务
        return _result(filename)

    async def main() -> None:
        q = JobQueue(processor=slow_processor, max_workers=1)
        await q.start()
        jobs = q.submit([JobItem(filename="a.txt", content=b"x")])
        assert await _wait_status(jobs[0], "running"), f"未进入 running：{jobs[0].status}"
        worker = q._workers[0]
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        # 修复前：status 仍是 "running"（CancelledError 未被 except Exception 捕获）
        assert jobs[0].status == "failed", f"取消后状态未收敛：{jobs[0].status}"
        assert jobs[0].error_code == "cancelled"
        gate.set()  # 放后台线程跑完，避免悬挂

    asyncio.run(main())


def test_requeue_after_stop_marks_job_cancelled() -> None:
    """H2（精确）：队列已停止时触发 `_requeue`，必须落终态并清理句柄。

    模拟「定时器已排定但 stop() 未来得及取消」的竞态窗口——
    修复前 `_requeue` 直接 return，job 永久停留 `retrying`。
    """
    def fail_processor(filename: str, content: bytes) -> ProcessingResult:
        return _result(filename, status="failed", error="boom", error_code="transient_error")

    async def main() -> None:
        q = JobQueue(processor=fail_processor, max_workers=1, retry_delay=5.0)
        await q.start()
        jobs = q.submit([JobItem(filename="a.txt", content=b"x")])
        assert await _wait_status(jobs[0], "retrying"), f"未进入 retrying：{jobs[0].status}"
        q._started = False  # 模拟 stop() 已置位，但定时器回调仍将触发
        q._requeue(jobs[0].id)  # 手动触发定时器回调（不经过 stop() 兜底遍历）
        assert jobs[0].status == "failed", f"停止后状态未收敛：{jobs[0].status}"
        assert jobs[0].error_code == "cancelled"
        assert not q._retrying, "重投句柄未被清理"

    asyncio.run(main())


def test_stop_cancels_pending_retry_timers() -> None:
    """H2（端到端）：`stop()` 必须取消已排定的重投定时器并收敛所有 job 状态。"""
    def fail_processor(filename: str, content: bytes) -> ProcessingResult:
        return _result(filename, status="failed", error="boom", error_code="transient_error")

    async def main() -> None:
        q = JobQueue(processor=fail_processor, max_workers=1, retry_delay=5.0)
        await q.start()
        jobs = q.submit([JobItem(filename="a.txt", content=b"x")])
        assert await _wait_status(jobs[0], "retrying"), f"未进入 retrying：{jobs[0].status}"
        assert q._retrying, "retrying 期间应登记定时器句柄"
        await q.stop()
        assert not q._retrying, "stop() 后不应残留未取消的重投定时器"
        assert jobs[0].status == "failed", f"停止后状态未收敛：{jobs[0].status}"

    asyncio.run(main())
