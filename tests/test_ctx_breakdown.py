"""上下文**分项构成**（2026-10-03，P2）—— 行为测试。

## 为什么需要
前端「上下文用量」面板（对齐 WorkBuddy）要回答"**这轮 prompt 里哪一块最大**"。
原先 `_log_context_composition` 的 buckets **只按角色分**（system/user/assistant/tool），
三个问题：
1. **`system` 混算**：`SYSTEM_PROMPT` 与「长期记忆注记」都是 `role=system`
   ⇒ 无法区分"提示词占多少"与"记忆占多少"；
2. ★ **工具定义完全不在统计里**：它在 API 的 `tools` 参数里、**不在 messages 里**
   ⇒ 用户会看到"工具占 0%"，而它其实是**每轮都付的固定成本**；
3. 分类名（`user`/`assistant`）对用户无意义 ⇒ 用户看不懂"哪块该优化"。

⇒ 改为**按来源**分 5 类，与前端面板一一对应：
`system` / `notes` / `messages` / `tool_results` / `tool_defs`。
"""
from __future__ import annotations

import asyncio

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import FakeLLM
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry

SYS_PROMPT = "你是论文阅读助手。" * 20


def _loop() -> AgentLoop:
    reg = ToolRegistry()
    register_builtin_tools(reg)
    return AgentLoop(
        FakeLLM(),
        ToolExecutor(reg),
        config=LoopConfig(max_tokens_budget=100_000),
        system_prompt=SYS_PROMPT,
    )


class TestBreakdownBuckets:
    def test_system_prompt_and_notes_are_separated(self):
        """★★ `SYSTEM_PROMPT` 与「记忆注记」必须**分开**（原先都是 `system`，混算）。"""
        loop = _loop()
        msgs = [
            ChatMessage(role=Role.SYSTEM, content="【长期记忆】用户偏好中文。"),
            ChatMessage(role=Role.USER, content="问题"),
            ChatMessage(role=Role.SYSTEM, content=SYS_PROMPT),  # ← 与 self.system_prompt 精确相同
        ]
        loop._log_context_composition(msgs)
        b = loop._ctx_breakdown
        assert "system" in b and "notes" in b, f"未分开：{b}"
        assert b["system"] > 0 and b["notes"] > 0
        # 记忆注记很短 ⇒ 必然远小于系统提示词（否则说明分错了）
        assert b["notes"] < b["system"], f"注记比提示词还大 ⇒ 可能归错类：{b}"

    def test_conversation_and_tool_results_are_separated(self):
        loop = _loop()
        msgs = [
            ChatMessage(role=Role.USER, content="问题" * 50),
            ChatMessage(role=Role.ASSISTANT, content="回答" * 50),
            ChatMessage(role=Role.TOOL, content="工具结果" * 50, tool_call_id="t1"),
        ]
        loop._log_context_composition(msgs)
        b = loop._ctx_breakdown
        assert b.get("messages", 0) > 0, f"对话消息未归类：{b}"
        assert b.get("tool_results", 0) > 0, f"工具结果未归类：{b}"
        # 旧口径的 `tool` / `user` / `assistant` 键不应再出现（避免两套口径并存）
        for stale in ("tool", "user", "assistant"):
            assert stale not in b, f"残留旧分类键 {stale}：{b}"

    def test_tool_defs_are_counted(self):
        """★★ 工具定义**不在 messages 里**，必须单独算 —— 否则显示"工具占 0%"。

        这是本项最容易漏的地方：它不来自任何一条消息，而来自 API 的 `tools` 参数。
        """
        loop = _loop()
        loop._log_context_composition([ChatMessage(role=Role.USER, content="问题")])
        assert loop._ctx_breakdown.get("tool_defs", 0) > 0, (
            "工具定义没被计入 ⇒ 面板会显示「工具 0%」，而它是每轮固定成本"
        )

    def test_breakdown_is_available_even_for_small_turns(self):
        """★ 分项**无条件**留存：面板要显示小轮次；只有**日志**才按阈值过滤。

        原因：阈值（3000）是给日志信噪比用的，而面板是**用户主动查看**的
        ⇒ 不能因为"轮次小"就让面板空白。
        """
        loop = _loop()
        loop._log_context_composition([ChatMessage(role=Role.USER, content="短")])
        assert loop._ctx_breakdown, "小轮次下分项为空 ⇒ 面板会没有数据"

    def test_empty_messages_do_not_crash(self):
        loop = _loop()
        assert loop._ctx_breakdown == {}
        loop._log_context_composition([])
        # 空 messages 时仍可能只有 tool_defs（每轮固定成本），但**不得抛错**
        assert isinstance(loop._ctx_breakdown, dict)

    def test_total_matches_sum_of_parts(self):
        loop = _loop()
        msgs = [
            ChatMessage(role=Role.SYSTEM, content=SYS_PROMPT),
            ChatMessage(role=Role.USER, content="问题" * 30),
            ChatMessage(role=Role.TOOL, content="结果" * 30, tool_call_id="t"),
        ]
        loop._log_context_composition(msgs)
        b = loop._ctx_breakdown
        assert sum(b.values()) == loop._last_ctx_estimate, (
            "分项之和必须等于总量（否则面板的百分比加起来不是 100%）"
        )


class TestDoneEventCarriesBreakdown:
    def test_done_event_has_context_breakdown(self):
        """★ 跨端契约：done 事件必须带 `context_breakdown`（前端据此渲染面板）。"""
        loop = _loop()

        async def run():
            return [e async for e in loop.run_stream("问题")]

        events = asyncio.run(run())
        done = [e for e in events if e.type == "done"]
        assert done, "必须收到 done 事件"
        assert "context_breakdown" in done[0].data, (
            "done 未带分项 ⇒ 前端面板拿不到数据（跨端功能两端都要断言）"
        )
        assert isinstance(done[0].data["context_breakdown"], dict)
