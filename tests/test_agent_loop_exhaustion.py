"""迭代耗尽/中止时的「必有回答」回归（2026-09-10，用户实测「改稿工作间无法生成有效回答」）。

根因：模型在 doc-bound 场景持续调用 retrieve 而不主动作答 → `max_iterations` 耗尽 →
原实现直接 ABORT 且**不产出任何回答** → 上层 `final_text` 为空 → 用户只看到
「抱歉，本次没有生成有效回答」（功能不可用表现）。

修复：①接近上限时注入**收敛提醒**（软着陆）；②中止/耗尽前做一次
`_finalize_without_tools` **强制收尾生成**（不提供 tools → 机制上保证收敛）。
"""

from __future__ import annotations

import asyncio

from lightcore.core.loop import AgentLoop, LoopConfig, LoopState
from lightcore.llm.base import ToolCall
from lightcore.llm.fake import FakeLLM
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec

CONVERGENCE_HINT = "请立即基于现有证据给出最终回答"


def _loop_factory(max_iterations: int):
    """构造一个「永远调用 noop 工具」的 LLM（复现不收敛场景），并记录每次收到的 messages。"""
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="noop", description="noop", parameters={"type": "object", "properties": {}},
        permission=Permission.SAFE,
        handler=lambda: ToolResult(ok=True, output="empty"),
    ))
    calls: list[list] = []

    class LoopFake(FakeLLM):
        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
            calls.append(list(messages))
            r = await super().chat(messages, tools, temperature=temperature,
                                   max_tokens=max_tokens, tool_choice=tool_choice)
            r.message.tool_calls = [ToolCall(id="c", name="noop", arguments={})]
            r.message.content = "calling"
            return r

    return AgentLoop(LoopFake(), ToolExecutor(reg), config=LoopConfig(max_iterations=max_iterations)), calls


def test_exhaustion_yields_fallback_answer():
    """回归：迭代耗尽时必须产出**实质回答**，而不是只发 error 让上层空手兜底。"""
    async def run():
        loop, _calls = _loop_factory(max_iterations=3)
        events = [ev async for ev in loop.run_stream("loop me")]

        assert any(e.type == "error" and e.state == LoopState.ABORT for e in events), \
            "中止信号必须保留（上层据此提示未收敛）"
        answers = [e for e in events if e.type == "message" and (e.data or {}).get("content")]
        assert answers, "耗尽时必须产出兜底回答——否则用户只会看到「本次没有生成有效回答」"

    asyncio.run(run())


def test_convergence_hint_injected_near_limit():
    """回归：接近迭代上限时注入收敛提醒（软着陆），且只注入一次。"""
    async def run():
        loop, calls = _loop_factory(max_iterations=5)
        _ = [ev async for ev in loop.run_stream("loop me")]

        # max_iterations=5 → 在第 3 轮（0-based index 3 = max_iterations-2）请求前注入
        assert len(calls) >= 4
        remind_at = [i for i, msgs in enumerate(calls) if any(CONVERGENCE_HINT in m.content for m in msgs)]
        assert remind_at, "接近上限时应注入收敛提醒"
        assert remind_at[0] == 5 - 2, f"提醒应在 max_iterations-2 轮注入：{remind_at}"
        # 注入后自然保留在上下文中（后续请求都含它），但**不得重复追加**
        counts = [sum(1 for m in msgs if CONVERGENCE_HINT in m.content) for msgs in calls]
        assert max(counts) == 1, f"提醒不得重复追加（否则污染上下文）：{counts}"

    asyncio.run(run())


def test_convergence_hint_skipped_for_tiny_limits():
    """边界：max_iterations < 4 时不注入提醒（没有「接近上限」的余量，避免占满上下文）。"""
    async def run():
        loop, calls = _loop_factory(max_iterations=2)
        _ = [ev async for ev in loop.run_stream("loop me")]
        assert not any(CONVERGENCE_HINT in m.content for msgs in calls for m in msgs)

    asyncio.run(run())
