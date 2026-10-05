"""上下文估算**单一来源**（2026-10-02）：`message_tokens` 必须把三样都算进去。

## 事故背景（实测，见 `docs/诊断-离题回答与熔断预警-2026-10-02.md` §6）

`_gate_context` 的增量原先只算 `sum(_rough_tokens(m.content))`。实测某一步：
- 闸门增量 **1436**（= 两次工具结果的估算 982 + 454，精确吻合）；
- 该步真实增量 **9809**（23243 → 33052）；
- 差额含 assistant 那条消息的 **`tool_calls[].arguments`**（协议要求回传）与
  **`reasoning`**（带工具调用的轮次**必须**回传，`llm/router.py:40`）——两者都被漏算。

⇒ 闸门这把尺子偏小 ⇒ 判"安全"⇒ 下一步直接越过单次上限 30000 ⇒ 用户侧 = **"突然熔断"**。
而 `[上下文构成]` 日志**早已**计入 tool_calls ⇒ 同一项目**两把尺子** ⇒ 本文件把它们钉成一把。

## 本文件在守什么

1. `message_tokens` 计入 `content` + `tool_calls.arguments` + `reasoning`（三者缺一即红）；
2. `_gate_context` 的投影**确实**用到了它们（不是只改了辅助函数）；
3. `_log_context_composition` 与闸门**同源**（同一函数），防再次分叉。
"""

from __future__ import annotations

import asyncio
import logging
import re

from lightcore.core.loop import (
    _TOOL_CALL_OVERHEAD,
    AgentLoop,
    LoopConfig,
    message_tokens,
)
from lightcore.llm.base import ChatMessage, Role, ToolCall
from lightcore.llm.fake import FakeLLM

_LOGGER = "lightcore.core.loop"
_PROJ = re.compile(r"投影 (\d+)（真实基线 (\d+) \+ 增量估 (\d+)）|投影 (\d+) ≥ 触发线")


def _tool_msg(content: str = "工具结果") -> ChatMessage:
    return ChatMessage(role=Role.TOOL, content=content, tool_call_id="c1")


# ============ ① 三样都要算 ============

class TestMessageTokens:
    def test_counts_tool_call_arguments(self):
        """★ 带工具调用的 assistant 消息：content 常为空，但参数占上下文（协议要求回传）。"""
        m = ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
            ToolCall(id="c1", name="retrieve", arguments={"query": "水利设施 数据集 实验对象 样本区"}),
        ])
        assert message_tokens(m) > _TOOL_CALL_OVERHEAD, (
            "tool_calls.arguments 未被计入 ⇒ 闸门投影会偏小（本次事故的成因之一）"
        )

    def test_counts_reasoning(self):
        """★ `reasoning_content` 在带工具调用的轮次必须回传（router.py:40）⇒ 必须计入。"""
        plain = ChatMessage(role=Role.ASSISTANT, content="答案")
        with_reasoning = ChatMessage(role=Role.ASSISTANT, content="答案",
                                     reasoning="我先分析一下问题的构成……" * 20)
        assert message_tokens(with_reasoning) > message_tokens(plain) + 50, (
            "reasoning 未被计入 ⇒ 工具轮次的增量会被系统性低估"
        )

    def test_still_counts_plain_content(self):
        m = ChatMessage(role=Role.USER, content="你好")
        assert message_tokens(m) > 0

    def test_robust_to_missing_attrs(self):
        """★ 测试替身可能没有 tool_calls / reasoning ⇒ 必须 getattr 兜底，不得抛异常。"""
        class _Bare:
            content = "只有 content"
            role = Role.USER

        assert message_tokens(_Bare()) > 0


# ============ ② 闸门确实用上了（不是只改辅助函数） ============

class TestGateUsesSingleSource:
    def _loop_with_baseline(self, baseline: int) -> AgentLoop:
        loop = object.__new__(AgentLoop)  # 只塞闸门最小字段（与 test_token_gate 同款）
        loop.config = LoopConfig(max_tokens_budget=1000)
        loop._last_real_prompt_tokens = baseline
        loop._steps = 0
        return loop

    def test_gate_projection_includes_tool_calls_and_reasoning(self, caplog):
        """★ 事故复现：assistant 只带 tool_calls + reasoning（content 为空）时，
        闸门投影必须**显著高于**基线 —— 若退回"只算 content"，投影 ≈ 基线，本用例变红。"""
        caplog.set_level(logging.INFO, logger=_LOGGER)
        loop = self._loop_with_baseline(900)  # 线 = 0.7 × 1000 = 700 ⇒ 必进"超线"分支
        messages = [
            _tool_msg("旧工具结果" * 50),
            ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id="c1", name="retrieve",
                         arguments={"query": "水利设施 数据集 构建 实验对象 样本区 高分二号"}),
            ], reasoning="推理链" * 100),
        ]
        loop._gate_mark = 0
        loop._gate_context(messages)  # noqa: SLF001

        lines = [r.getMessage() for r in caplog.records if "[上下文闸门]" in r.getMessage()]
        assert lines, "闸门超线时必须留痕（本轮判据依赖这行日志）"
        m = _PROJ.search(lines[0])
        assert m, f"日志格式变了，请同步本测试：{lines[0]}"
        projected = int(m.group(1) or m.group(4))
        delta_est = int(m.group(3)) if m.group(3) else None
        # 核心断言：tool_calls + reasoning 必须计入增量（老实现下 delta ≈ 0 ⇒ 投影 ≈ 900）
        assert projected > 1100, (
            f"闸门投影 {projected} 未包含 tool_calls/reasoning ⇒ 退回「只算 content」的老缺陷"
            f"（增量估={delta_est}）"
        )

    def test_single_source_contract(self):
        """源码契约：两个消费方都必须走 `message_tokens`，不得再出现"只算 content"的求和。"""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "src" / "lightcore" / "core" / "loop.py").read_text(
            encoding="utf-8")
        assert "delta = sum(message_tokens(m) for m in messages[self._gate_mark:])" in src, (
            "闸门的增量估算必须走唯一入口 message_tokens"
        )
        assert "sum(_rough_tokens(m.content) for m in messages[self._gate_mark:])" not in src, (
            "出现了只算 content 的旧写法 —— 这正是本次事故的成因"
        )
        assert "buckets.get(key, 0) + message_tokens(m)" in src, (
            "[上下文构成] 也必须同源，否则又分叉成两把尺子"
        )


# ============ ③ 压缩侧同样走唯一入口（2026-10-03 审查补盲） ============


class TestCompactUsesSingleSource:
    """★★ **覆盖盲区**（2026-10-03 审查发现）：本文件原先覆盖 `message_tokens` 与
    `_gate_context`，**唯独没测 `_compact_history`** ⇒ 2026-10-02 那次"统一入口"改造
    **漏改了压缩侧**而测试全绿（漏改持续到 2026-10-03 才被发现）。

    实测症状：闸门裁剪 **30** 次 vs 压缩触发 **16** 次，而 `[历史压缩]` 只有 1 条
    （还是"触发了但未丢弃"= 空转）⇒ 有损的闸门被当成主力。
    """

    def _loop(self, baseline: int) -> AgentLoop:
        loop = object.__new__(AgentLoop)  # 只塞压缩路径最小字段
        loop.llm = FakeLLM()
        # ⚠️ `compact_keep_capacity` 是**只读 property**（= trigger×0.7，且有 max(1000, …) 下限）
        #   ⇒ 不能直接赋值压小。改用 `prompt_baseline` 换算路径：
        #   `need_cut = baseline − keep` ⇒ baseline 给得大 ⇒ `hist_cap` 变小 ⇒ 强制丢弃。
        #   （顺带覆盖 P0 的换算逻辑本身。）
        loop.config = LoopConfig(max_tokens_budget=1000)
        loop._is_pinned_note = lambda content: False  # 不排除任何注记，简化判读
        return loop

    def test_compact_counts_tool_calls_and_reasoning(self, caplog):
        """★ 事故复现：assistant **content 为空**、只带 tool_calls + reasoning 时，
        压缩判据必须认为它"很大"⇒ 会因超出 cap 而丢弃。
        老实现（只算 content）下这条估值为 0 ⇒ 永不丢弃 ⇒ 本用例变红。"""
        caplog.set_level(logging.INFO, logger=_LOGGER)
        loop = self._loop(baseline=100_000)
        # 前两条"占位"小消息，最后一条是"只带 tool_calls+reasoning"的 assistant
        history = [
            ChatMessage(role=Role.USER, content="问题一"),
            ChatMessage(role=Role.ASSISTANT, content="回答一"),
            ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id="c1", name="retrieve",
                         arguments={"query": "水利设施 数据集 构建 实验对象 样本区 高分二号"}),
            ], reasoning="推理链" * 100),
        ]
        loop._compacted_dropped = 0
        # ⚠️ 必须传**非 0** 的 baseline：`_history_cap_for` 在 `baseline=0` 时会退回
        #   "hist_cap = 整 prompt 目标"（= max(1000, …)），本例 est≈200 < 1000 ⇒ 不丢
        #   ⇒ 那样测的就不是换算路径，而是旧兜底（会误判为"修复无效"）。
        out = asyncio.run(
            loop._compact_history(history, triggered=True, prompt_baseline=100_000)
        )

        assert len(out) < len(history), (
            "含 tool_calls/reasoning 的 assistant 没被丢弃 ⇒ 压缩判据退回「只算 content」"
            "（content 为空 ⇒ 老实现估 0 ⇒ 永不丢）"
        )

    def test_compact_source_contract(self):
        """源码契约：压缩路径**不得**再出现"只算 content"的求和。"""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "src" / "lightcore" / "core" / "loop.py").read_text(
            encoding="utf-8")
        assert "sum(_rough_tokens(m.content) for m in history)" not in src, (
            "压缩判据又出现「只算 content」的旧写法 —— 这正是 2026-10-03 审查发现的 P0-1"
        )
        assert "est = sum(message_tokens(m) for m in history)" in src, (
            "压缩判据必须走唯一入口 message_tokens"
        )
