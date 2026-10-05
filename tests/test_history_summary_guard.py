"""N11 回归（2026-09-11）：多轮历史压缩的摘要调用必须**限 token + 有超时**。

真实故障：「多轮上下文无法生成有效回答」——
多轮对话使历史超过压缩阈值后，`_compact_history` → `_summarize_dropped` 会发起一次
**完整 LLM 调用**，而它**未限制 max_tokens**；在 thinking 模型
（deepseek-v4-flash-vision-exp）下思考链吃满数千 token、耗时 **70+ 秒**，
且它在 `run()` 开头执行、期间**零日志零输出** → 用户以为卡死 → 点「停止」→ 无回答。

修复：`max_tokens=SUMMARY_MAX_TOKENS`(600) + `asyncio.wait_for(..., SUMMARY_TIMEOUT_S)`，
超时回退滚动截断（返回空摘要），绝不阻断主流程。
"""

from __future__ import annotations

import asyncio

from lightcore.core import loop as loop_module
from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import FakeLLM
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry


def _make_loop(llm, *, max_tokens_budget: int = 8000) -> AgentLoop:  # noqa: ANN001
    """显式传预算（默认 8000 = 旧值）：本文件锁定的是**压缩/兜底行为**，不是预算默认值。

    2026-09-16 全局默认 8000 → 30000 后，压缩阈值（预算 // 2）同步从 4000 变 15000，
    若用默认值，构造的 30 轮历史将不再触发压缩 —— 测试不应依赖默认值的巧合。
    """
    reg = ToolRegistry()
    register_builtin_tools(reg)
    return AgentLoop(llm, ToolExecutor(reg), config=LoopConfig(max_tokens_budget=max_tokens_budget))


def test_summarize_dropped_limits_max_tokens(monkeypatch):
    """回归：摘要调用必须带 max_tokens 上限（原先为 None → thinking 模型可无限长思考）。"""
    captured: dict = {}

    class _RecordingLLM(FakeLLM):
        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None):  # noqa: ANN001
            captured["max_tokens"] = max_tokens
            return await super().chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )

    loop = _make_loop(_RecordingLLM())
    dropped = [
        ChatMessage(role=Role.USER, content="历史问题"),
        ChatMessage(role=Role.ASSISTANT, content="历史回答"),
    ]

    asyncio.run(loop._summarize_dropped(dropped))  # noqa: SLF001
    assert captured["max_tokens"] == loop_module.SUMMARY_MAX_TOKENS, \
        "摘要必须限制 max_tokens（thinking 模型会吃满 token 并耗时数十秒）"


def test_summarize_dropped_times_out_gracefully(monkeypatch):
    """回归：摘要超时必须**安全回退**（返回空摘要供滚动截断），不得无限等待或抛异常。"""
    monkeypatch.setattr(loop_module, "SUMMARY_TIMEOUT_S", 0.2)

    class _HangingLLM(FakeLLM):
        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None):  # noqa: ANN001
            await asyncio.sleep(30)  # 模拟 thinking 模型长思考
            return await super().chat(messages, tools=tools, temperature=temperature)

    loop = _make_loop(_HangingLLM())
    dropped = [ChatMessage(role=Role.USER, content="很长的一段历史对话内容" * 20)]

    out = asyncio.run(loop._summarize_dropped(dropped))  # noqa: SLF001
    assert out == "", "超时应返回空摘要（调用方回退滚动截断）"


def test_compact_history_still_compacts_when_summary_times_out(monkeypatch):
    """端到端：摘要超时后，历史仍被压缩（滚动截断兜底），主流程不阻断。"""
    monkeypatch.setattr(loop_module, "SUMMARY_TIMEOUT_S", 0.2)

    class _HangingLLM(FakeLLM):
        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None):  # noqa: ANN001
            await asyncio.sleep(30)
            return await super().chat(messages, tools=tools, temperature=temperature)

    loop = _make_loop(_HangingLLM(), max_tokens_budget=8000)
    long_history = []
    for i in range(30):
        long_history.append(ChatMessage(role=Role.USER, content=f"第 {i} 轮问题，" + "详细内容" * 40))
        long_history.append(ChatMessage(role=Role.ASSISTANT, content=f"第 {i} 轮回答，" + "详细回答" * 40))

    compacted = asyncio.run(loop._compact_history(long_history))  # noqa: SLF001
    assert len(compacted) < len(long_history), "超长历史必须被压缩"
