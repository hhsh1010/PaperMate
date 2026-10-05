"""N9 回归（2026-09-11）：检索不得阻塞事件循环。

真实故障：`MultiQueryRAGPipeline.query` 在 async 函数里**同步调用**同步的 base 管线
（BGE-M3 编码 + 混合检索 + CrossEncoder 重排，GPU 密集）→ **事件循环被阻塞**：
- 用户实测一次工具调用耗时 **262.82s**，期间每 0.3s 轮询断连的 `_watch_disconnect`
  **完全没有机会运行** → 点「停止」无效、其它请求无响应、GPU 持续满载。

修法：同步检索一律 `asyncio.to_thread(...)` 下放线程池。
本文件用「心跳协程能否在检索期间推进」来判定是否阻塞——这是事件循环可用性的直接探针。
"""

from __future__ import annotations

import asyncio
import time

from lightcore.rag.multi_query import MultiQueryRAGPipeline
from lightcore.rag.pipeline import RetrievedChunk
from lightcore.tools.builtin.basic import _make_retrieve  # noqa: PLC2701

_SLEEP = 0.3


class _SlowSyncPipeline:
    """同步阻塞式检索（模拟真实 base：编码 + 检索 + 重排）。"""

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        time.sleep(_SLEEP)
        return [RetrievedChunk(id="doc-18-c1", text="段落内容", page=1, score=0.8)][:top_k]


class _StubExpander:
    async def expand(self, q: str, n: int | None = None) -> list[str]:  # noqa: ARG002
        return [q]


async def _run_with_heartbeat(coro_factory) -> int:
    """返回**检索执行期间**心跳推进的次数（而非结束后补跑的）。

    注意：若只统计总次数会失效——事件循环被阻塞 _SLEEP 秒后，心跳任务仍会在检索结束后
    把剩余次数跑满（实测：阻塞版也能统计到 20 次）。必须按时间戳界定"检索窗口内"。
    """
    stamps: list[float] = []
    t0 = time.perf_counter()

    async def _heartbeat() -> None:
        while True:
            stamps.append(time.perf_counter() - t0)
            await asyncio.sleep(_SLEEP / 20)

    hb = asyncio.create_task(_heartbeat())
    await coro_factory()
    t_end = time.perf_counter() - t0
    hb.cancel()
    # 只数「发生在检索结束之前」的心跳（即检索窗口内事件循环是否仍可调度）
    return len([s for s in stamps if s < t_end])


def test_multi_query_does_not_block_event_loop():
    """N9 回归：多查询管线必须把同步 base.query 下放线程池。"""
    pipeline = MultiQueryRAGPipeline(_SlowSyncPipeline(), _StubExpander())

    async def _call():
        res = await pipeline.query("文档中的对比模型", top_k=3)
        assert res and res[0].id == "doc-18-c1"

    ticks = asyncio.run(_run_with_heartbeat(_call))
    # 修复前：同步调用阻塞整个 _SLEEP 窗口 → 窗口内心跳 0~1 次
    # 修复后：线程池执行 → 窗口内心跳 ~20 次
    assert ticks >= 5, f"事件循环在检索窗口内被阻塞（窗口内心跳仅 {ticks} 次）"


def test_retrieve_does_not_block_event_loop_for_sync_pipeline():
    """N9 回归：retrieve 工具对**同步**管线也必须走线程池。"""
    handler = _make_retrieve(_SlowSyncPipeline())

    async def _call():
        res = await handler("对比模型")
        assert res.ok, res.error

    ticks = asyncio.run(_run_with_heartbeat(_call))
    assert ticks >= 5, f"retrieve 阻塞了事件循环（窗口内心跳仅 {ticks} 次）"


def test_retrieve_still_awaiting_async_pipeline():
    """兼容：async 管线仍被正确 await（N4 修复不得回退）。

    注：本用例**不**做心跳断言——stub 的 async query 立即返回，检索窗口过短，
    "窗口内心跳数"没有判别力（阻塞型实现也会 0 次）。心跳断言只用带 sleep 的同步 stub。
    """
    class _AsyncPipeline:
        async def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
            return [RetrievedChunk(id="doc-9-c75", text="段落", page=5, score=0.9)][:top_k]

    handler = _make_retrieve(_AsyncPipeline())

    async def _call():
        res = await handler("对比模型")
        assert res.ok, res.error
        assert res.output["results"][0]["id"] == "doc-9-c75"
        assert res.output["results"][0]["page"] == 5

    asyncio.run(_call())
