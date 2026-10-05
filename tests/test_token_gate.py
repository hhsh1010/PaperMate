"""T-01 上下文闸门与真实值回写：守护测试。

守三组契约（对应卡片验收门 ①）：
1. **`_gate_context` 的行为**（循环内闸门）：无真实基线不动手、未超线不动手、
   超线时**替换内容而不删消息**（保证 `assistant.tool_calls` 与 `tool` 结果配对不破）、
   **最近 N 条不动**。配对被拆散会让厂商 API 直接 400 —— 这是本设计最关键的不变量。
2. **口径一致**：`compact_keep_capacity` 必须由 `compact_trigger_tokens` 派生且**小于**它。
   旧实现写死 `max_tokens_budget // 2` 并拿**估算值**去比，而触发线用**真实值** →
   两把尺子 ⇒ 触发后一条都不丢（压缩空转）。这条断言防它回归。
3. **真实值回写覆盖三条路径**：agent / 直答 / 技能。技能路径此前完全缺失
   （`skill.run(ctx)` 不经 AgentLoop），用 `_UsageRecorder` 补上，且**估算不得回写**。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import FakeLLM

# ---------------- 1. 循环内闸门 ----------------


def _loop(budget: int = 1000, *, baseline: int = 0, mark: int = 0) -> AgentLoop:
    """造一个只带闸门所需字段的 AgentLoop（不碰 LLM/工具依赖）。"""
    lp = object.__new__(AgentLoop)
    lp.config = LoopConfig(max_tokens_budget=budget)
    lp._last_real_prompt_tokens = baseline
    lp._gate_mark = mark
    lp._steps = 0
    return lp


def _tool_msg(chars: int, call_id: str = "c1") -> ChatMessage:
    return ChatMessage(role=Role.TOOL, content="x" * chars, tool_call_id=call_id)


def test_gate_noop_without_real_baseline() -> None:
    """首轮没有真实值 → 不在此动手（交给 `_compact_history` 的估算路径）。"""
    lp = _loop(baseline=0, mark=1)
    msgs = [_tool_msg(5000), ChatMessage(role=Role.USER, content="问题")]
    before = list(msgs)
    lp._gate_context(msgs)
    assert msgs == before, "无真实基线时闸门不得改动上下文"


def test_gate_noop_below_trigger() -> None:
    """投影未达触发线 → 一个字都不动。"""
    lp = _loop(baseline=100, mark=2)   # 触发线 = 700
    msgs = [_tool_msg(100), ChatMessage(role=Role.USER, content="问题")]
    before = list(msgs)
    lp._gate_context(msgs)
    assert msgs == before


def test_gate_elides_old_tool_result_and_preserves_pairing() -> None:
    """★ 超线 → 裁剪早期工具结果，且**消息条数与 tool_call_id 都不变**（配对不破）。"""
    lp = _loop(baseline=700, mark=3)   # 已达触发线 700
    msgs = [_tool_msg(3000, "call-A")] + [ChatMessage(role=Role.USER, content="填充") for _ in range(9)]
    n_before = len(msgs)
    lp._gate_context(msgs)

    assert len(msgs) == n_before, "不得删消息 —— 删了会拆散工具调用配对"
    assert msgs[0].role == Role.TOOL
    assert msgs[0].tool_call_id == "call-A", "tool_call_id 必须保留"
    assert msgs[0].content.startswith("[早期工具结果已裁剪"), "内容应被替换为裁剪标记"
    assert len(msgs[0].content) < 3000


def test_gate_keeps_tail_untouched() -> None:
    """最近 N 条（含当前 assistant/tool 配对）不裁剪 —— 它们正是本步要用的证据。"""
    lp = _loop(baseline=700, mark=99)   # 增量起点在末尾 → 头部无可裁项
    msgs = [ChatMessage(role=Role.USER, content="填充") for _ in range(9)] + [_tool_msg(3000, "tail-1")]
    before = [m.content for m in msgs]
    lp._gate_context(msgs)
    assert [m.content for m in msgs] == before, "尾部工具结果不得被裁剪"


def test_gate_ignores_short_tool_results() -> None:
    """太短的工具有结果裁了没收益还丢信息 → 不动（且不应报错）。"""
    lp = _loop(baseline=900, mark=3)
    msgs = [_tool_msg(50, "short")] + [ChatMessage(role=Role.USER, content="填充") for _ in range(9)]
    before = [m.content for m in msgs]
    lp._gate_context(msgs)
    assert [m.content for m in msgs] == before


# ---------------- 2. 口径一致（防"两把尺子"回归） ----------------


def test_keep_capacity_derived_from_trigger_and_smaller() -> None:
    """★ cap 必须由触发线派生、且严格小于它 —— 否则压缩会"触发但一条都不丢"。"""
    lp = _loop(budget=30000)
    trigger = lp.config.compact_trigger_tokens
    cap = lp.config.compact_keep_capacity
    assert cap < trigger, "cap 必须小于触发线，否则触发后不会真的压缩（空转）"
    assert cap == int(trigger * 0.7)
    # 旧实现是 max_tokens_budget // 2 = 15000（**估算口径**）→ 用真实基线折算会大于触发线
    assert cap != 30000 // 2 or cap < trigger


def test_gate_uses_trigger_for_both_paths() -> None:
    """触发线与 cap 必须是同一配置派生（改一个另一个跟着变）。"""
    a, b = _loop(budget=10000), _loop(budget=50000)
    assert a.config.compact_trigger_tokens < b.config.compact_trigger_tokens
    assert a.config.compact_keep_capacity < b.config.compact_keep_capacity


# ---------------- 3. 真实值回写：三条路径 + 估算不回写 ----------------


def test_usage_recorder_tracks_peak_from_chat_and_stream() -> None:
    """`_UsageRecorder` 必须取**峰值**（不是最后一次），且对其他属性透传。

    ⚠️ 2026-09-28：本壳同时记录 `first_prompt_tokens`（回写用它，不用峰值）——
    技能会把检索到的 chunk 拼进 prompt，那部分**不进下一轮历史** ⇒ 峰值会系统性高估。
    """
    from papermate.api.routes_search import _UsageRecorder

    class _Resp:
        def __init__(self, pt): self.prompt_tokens = pt

    class _Chunk:
        def __init__(self, pt):
            self.finished = True
            self.prompt_tokens = pt

    class _Inner:
        sentinel = "keep-me"
        async def chat(self, *a, **kw): return _Resp(120)
        async def stream(self, *a, **kw):
            yield _Chunk(50)
            yield _Chunk(900)
            yield _Chunk(300)   # 最后一块更小 → 峰值仍应是 900

    rec = _UsageRecorder(_Inner())
    assert rec.sentinel == "keep-me", "非 LLM 属性/方法必须透传"

    import asyncio

    async def _run():
        await rec.chat([], max_tokens=1)

        async for _ in rec.stream([]):
            pass

    asyncio.run(_run())
    assert rec.peak_prompt_tokens == 900, "必须取 max 而非 last"
    assert rec.first_prompt_tokens == 120, "首步必须取**第一次**调用的值（回写基线用）"


def test_usage_recorder_is_still_seen_as_real_llm() -> None:
    """包装后 `is_real_llm` 必须仍为 True —— 否则技能会误判"没有真实模型"而改变行为。"""
    from papermate.api.routes_search import _UsageRecorder
    from papermate.skills._common import is_real_llm

    class _Real:
        pass

    assert is_real_llm(_Real()) is True
    assert is_real_llm(_UsageRecorder(_Real())) is True
    # 反向：FakeLLM 及其包装都应为 False（Fake 本就不该被包装）
    assert is_real_llm(FakeLLM()) is False


def test_all_three_paths_write_back_real_baseline() -> None:
    """★ 跨轮链必须覆盖三条路径。用**源码级**断言守住"技能路径又被漏掉"这一历史缺口。

    为什么用源码扫描而不是集成测试：三条路径分属不同的 SSE 分支，
    集成测试要么很重、要么只能覆盖其中一条；而"少了一条"正是本次修复的缺陷类型，
    用调用点计数能直接钉住它。

    ⚠️ 2026-09-28：入口从 `_write_back_prompt_peak` 更名为 `_write_back_prompt_baseline` ——
    回写的不再是"峰值"而是**首步真实 prompt**（峰值含本轮工具结果、不进下一轮历史 ⇒ 高估且棘轮）。
    """
    src = (Path(__file__).resolve().parent.parent
           / "src" / "papermate" / "api" / "routes_search.py").read_text(encoding="utf-8")
    sites = re.findall(r"_write_back_prompt_baseline\(", src)
    # 定义 1 处 + 调用 ≥3 处（agent / 直答 / 技能）
    assert len(sites) >= 4, f"回写调用点疑似不足三条路径：找到 {len(sites)} 处"
    # 旧名不得复活（它的语义就是"写峰值"，正是被修掉的缺陷）
    assert "_write_back_prompt_peak(" not in src, "旧入口（写峰值）不得复活"
    # 技能路径必须用录制壳（否则又会拿不到真实值）
    assert "_UsageRecorder(agent.llm) if is_real_llm(agent.llm)" in src, \
        "技能路径缺少真实 prompt 录制"
    # ★ agent 路径必须回写**首步**，不得退回峰值
    assert '"first_prompt_tokens"' in src and "peak_prompt_tokens\") or 0" not in src, (
        "agent 路径必须回写 first_prompt_tokens（峰值会让跨轮基线变成只增不减的棘轮）"
    )
    # 且只写真实值：唯一入口处必须带 baseline<=0 的守卫
    assert "if baseline <= 0:" in src, "缺少『估算（0）不得回写』的守卫"


@pytest.mark.parametrize("bad", [0, -1])
def test_write_back_skips_non_positive(bad: int) -> None:
    """`_write_back_prompt_baseline(<=0)` 必须直接跳过（估算值就是靠"不传真实值"表达的）。"""
    src = (Path(__file__).resolve().parent.parent
           / "src" / "papermate" / "api" / "routes_search.py").read_text(encoding="utf-8")
    assert "if baseline <= 0:" in src and "return" in src[src.index("if baseline <= 0:"):][:40]
