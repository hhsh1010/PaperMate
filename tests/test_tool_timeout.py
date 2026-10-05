"""单次工具执行超时回归（2026-09-16，P2）。

缺口：循环层**此前完全没有工具超时**（`asyncio.wait_for` 只用在历史摘要）。
某个工具挂死（网络连接不返回、外部服务僵住）会让整轮无限等待，用户只能点「停止」脱身。

修法：`LoopConfig.tool_timeout_s`（默认 120s，刻意宽松 —— 目的是兜住"挂死"，
不是给正常慢调用设卡；LLM 驱动的工具本身可能跑十几秒以上，卡太紧反而误伤）。
超时在 `_run_one` 里落为 `ToolResult(ok=False, error="tool_timeout: …")`：
**与既有"系统故障 ≠ 空结果"口径一致**（超时是系统侧异常，伪装成空结果会让模型以为"文档里没有"），
并计入 `max_tool_failures`（连续 3 次 → 中止整轮）。

⚠️ 本文件同时守住一条容易写错的地方：**只捕 `TimeoutError`，不能让 `CancelledError` 变成"工具失败"**
—— 否则用户点「停止」会被记成工具出错。
"""

from __future__ import annotations

import asyncio

from lightcore.core.loop import AgentLoop, LoopConfig, LoopState
from lightcore.llm.base import ToolCall
from lightcore.llm.fake import FakeLLM
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec

SLOW_TOOL = "slow"


def _llm_always_calls(tool: str = SLOW_TOOL) -> FakeLLM:
    """永远请求调用指定工具的替身（逼真触发工具执行路径）。"""

    class _Caller(FakeLLM):
        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
            r = await super().chat(messages, tools, temperature=temperature,
                                   max_tokens=max_tokens, tool_choice=tool_choice)
            if tools:  # THINK 轮 → 请求工具；收尾轮（tools=None）→ 普通回答
                r.message.tool_calls = [ToolCall(id="c", name=tool, arguments={})]
                r.message.content = "calling"
            return r

    return _Caller()


def _executor(handler) -> ToolExecutor:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name=SLOW_TOOL, description="slow", parameters={"type": "object", "properties": {}},
        permission=Permission.SAFE, handler=handler,
    ))
    return ToolExecutor(reg)


def _make_loop(handler, *, timeout_s: float, max_iterations: int = 2) -> AgentLoop:
    return AgentLoop(
        _llm_always_calls(), _executor(handler),
        config=LoopConfig(max_iterations=max_iterations, tool_timeout_s=timeout_s),
    )


def _run(loop: AgentLoop) -> list:
    return asyncio.run(_collect(loop))


async def _collect(loop: AgentLoop) -> list:
    return [ev async for ev in loop.run_stream("跑一下工具")]


def _tool_results(events: list) -> list[dict]:
    return [e.data for e in events if e.type == "tool_result"]


def test_timeout_is_recorded_as_a_failure_with_actionable_error():
    """超时 → `ok=False` + 错误文案含 `tool_timeout`、工具名与上限（模型/用户据即可操作）。"""
    async def slow():
        await asyncio.sleep(0.3)
        return ToolResult(ok=True, output="never")

    events = _run(_make_loop(slow, timeout_s=0.05))
    results = _tool_results(events)
    assert results, "应产生 tool_result 事件"
    first = results[0]
    assert first["ok"] is False, "超时必须算失败（否则会伪装成「空结果」）"
    err = first["error"]
    assert "tool_timeout" in err, "错误码需机器可读（供上层分支/埋点）"
    assert SLOW_TOOL in err, "必须带工具名 —— 否则不知道卡在哪"
    assert "执行超过" in err and "未返回" in err, "必须带上限说明，模型据此换参数/换工具"


def test_fast_tool_is_unaffected():
    """回归：正常快工具不受影响（超时只在真的超时后生效）。"""
    def fast():
        return ToolResult(ok=True, output="ok-value")

    events = _run(_make_loop(fast, timeout_s=5.0))
    results = _tool_results(events)
    assert results and results[0]["ok"] is True
    assert results[0]["output"] == "ok-value"


def test_consecutive_timeouts_trip_the_failure_breaker():
    """连续超时与其它工具失败**同一阈值**：3 次 → 中止整轮（而不是死等）。"""
    async def slow():
        await asyncio.sleep(0.3)
        return ToolResult(ok=True, output="never")

    events = _run(_make_loop(slow, timeout_s=0.05, max_iterations=5))
    aborts = [e for e in events if e.type == "error" and e.state == LoopState.ABORT]
    assert aborts, "连续工具失败/超时必须中止整轮"
    assert "工具连续失败" in aborts[0].data["reason"]


def test_cancellation_is_not_misreported_as_tool_failure():
    """**关键**：工具被取消（用户点停止）时 `CancelledError` 必须原样传播，不得记成"工具失败"。"""
    async def cancelled():
        raise asyncio.CancelledError

    loop = _make_loop(cancelled, timeout_s=5.0)

    async def run():
        tc = ToolCall(id="c", name=SLOW_TOOL, arguments={})
        await loop._run_one(tc, "thought")  # noqa: SLF001  直接验证最内层语义

    try:
        asyncio.run(run())
    except asyncio.CancelledError:
        return  # ✅ 正确：取消语义没被吞
    raise AssertionError("CancelledError 被吞掉了 —— 用户点「停止」会被误记成工具失败")


def test_timeout_is_client_configurable():
    """超时值是 `LoopConfig` 的可配项（便于按部署/工具特性调整）。"""
    assert LoopConfig().tool_timeout_s == 120.0
    assert LoopConfig(tool_timeout_s=1.5).tool_timeout_s == 1.5
