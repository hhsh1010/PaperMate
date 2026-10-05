"""方案 D′（2026-09-17）：历史压缩的**触发判据改用真实 token**，估算仅作首轮兜底。

背景：压缩触发原先靠 `_rough_tokens` 的字符估算，而它实测偏低约 2 倍**且会漂移**
（消息/工具调用越多偏差越大）→ 触发时机失真，且**压缩时机错过就补不回来**
（上下文已经发出去了）。

改法：`AgentLoop` 每轮新建，所以"上一轮**首步**真实 prompt"由调用方跨轮注入
（存 session config，零 DDL）；有真实值时**完全不用估算**做这个判断。

⚠️ **2026-09-28 口径修正**：注入的必须是**上一轮首步**的 prompt，**不是峰值**。
峰值 = max(各步 prompt)，而本轮的工具结果**不会进入下一轮历史** ⇒ 峰值系统性高估；
且 `max()` 会让基线**只增不减（棘轮）** —— 实测事故：某轮峰值 29,357 回写后，
之后每轮首步实际仅 15,411（**高估 87%**）⇒ 压缩每轮误触发、
闸门每轮都去裁历史里的长工具结果（**由过期数字驱动的静态证据损失**）。

覆盖：
- 触发线 = 单次上限 × 70%；
- 真实值低于触发线 → **不压缩、不调摘要**（省开销，且这是"估算会说该压"的常见场景）；
- 真实值达到触发线 → 压缩；
- 无真实值（首轮/进程重启后首次）→ 退回估算兜底（不能因为没真实值就完全不设防）；
- 真实 prompt 的记录（**首步值** + 本轮峰值，均**按轮**结算、**不跨轮继承**）与 done 事件回传。
"""

from __future__ import annotations

import asyncio
from unittest import mock

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, LLMResponse, Role, ToolCall
from lightcore.llm.fake import FakeLLM
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry


def _loop(*, max_tokens_budget: int = 10000, last_prompt_tokens: int = 0,
          llm=None) -> AgentLoop:  # noqa: ANN001
    reg = ToolRegistry()
    register_builtin_tools(reg)
    return AgentLoop(
        llm or FakeLLM(), ToolExecutor(reg),
        config=LoopConfig(max_tokens_budget=max_tokens_budget),
        last_prompt_tokens=last_prompt_tokens,
    )


def _history(n: int = 6) -> list[ChatMessage]:
    """构造明显超过压缩阈值的_历史（每条 ≈1360 估算 token，阈值 = 10000//2 = 5000）。"""
    big = "这是一段用于撑大上下文的测试文本。" * 200
    return [ChatMessage(role=Role.USER, content=big) for _ in range(n)]


# --------------------------------------------------------------- 触发线

def test_compact_trigger_is_70_percent_of_single_request_limit() -> None:
    """触发线派生自单次上限（同一件事的两个刻度，不单独配置以免调一个忘一个）。"""
    assert LoopConfig(max_tokens_budget=10000).compact_trigger_tokens == 7000
    assert LoopConfig(max_tokens_budget=30000).compact_trigger_tokens == 21000


# --------------------------------------------------------------- 真实值路径

def test_no_compaction_when_last_real_prompt_is_below_trigger() -> None:
    """真实值低于触发线 → **原样返回**，连估算都不算、更不调摘要。

    这正是方案 D′ 的核心收益：不再让"偏低 2 倍的估算"决定要不要压缩。
    实际语义是"上一轮 prompt 才 2000 token，本轮再加也远没到 7000 上限"。
    """
    loop = _loop(max_tokens_budget=10000, last_prompt_tokens=2000)
    history = _history(6)

    with mock.patch.object(loop, "_compact_history", side_effect=AssertionError("不应触发压缩")):
        out = asyncio.run(loop._prepare_history(list(history)))

    assert len(out) == len(history), "低于触发线时必须原样返回"
    assert out is not history, "应返回副本，避免调用方原地改动共享列表"


def test_compaction_when_last_real_prompt_reaches_trigger() -> None:
    """真实值达到触发线 → 压缩（此时"该压"是真实数据说的，不是估算猜的）。"""
    loop = _loop(max_tokens_budget=10000, last_prompt_tokens=7000)
    history = _history(6)

    out = asyncio.run(loop._prepare_history(list(history)))

    assert len(out) < len(history), f"达到触发线必须压缩，实际 {len(out)}/{len(history)}"


def test_compaction_trigger_logged_with_real_numbers(caplog) -> None:
    """触发时打一条含真实数字的日志 —— 便于线上核对"这次为什么压了"。"""
    loop = _loop(max_tokens_budget=10000, last_prompt_tokens=9000)

    with caplog.at_level("INFO", logger="lightcore.core.loop"):
        asyncio.run(loop._prepare_history(_history(6)))

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "上下文压缩" in text and "9000" in text, f"应记录真实值与触发线：{text}"


# --------------------------------------------------------------- 首轮兜底

def test_falls_back_to_estimate_when_no_real_value() -> None:
    """无真实值（首轮）→ 退回估算兜底：不能因为"没有真实值"就完全不设防。"""
    loop = _loop(max_tokens_budget=10000, last_prompt_tokens=0)
    history = _history(6)

    out = asyncio.run(loop._prepare_history(list(history)))

    assert len(out) < len(history), "首轮无真实值时必须用估算兜底压缩"


# --------------------------------------------------------------- 真实值记录

class _UsageLLM(FakeLLM):
    """按给定 prompt/completion 数返回，用于验证真实 prompt 的记录口径。"""

    def __init__(self, prompt_tokens: int, completion_tokens: int = 5) -> None:
        super().__init__()
        self._pt = prompt_tokens
        self._ct = completion_tokens

    async def chat(self, messages, tools=None, **kw):  # noqa: ANN001, ANN003
        self._calls.append(messages)
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="好"),
            prompt_tokens=self._pt, completion_tokens=self._ct,
            model="stub", latency_ms=1, cost=0.0,
        )

    async def stream(self, messages, tools=None, **kw):  # noqa: ANN001, ANN003
        self._calls.append(messages)
        yield LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="好"),
            prompt_tokens=self._pt, completion_tokens=self._ct,
            model="stub", latency_ms=1, cost=0.0,
        )


def _drain(loop: AgentLoop, user_input: str = "问题") -> list:
    async def run():
        return [e async for e in loop.run_stream(user_input)]

    return asyncio.run(run())


def test_real_prompt_recorded_on_done_event() -> None:
    """done 事件必须同时回传**首步**与**本轮峰值**，且都是**单次**口径。

    - `first_prompt_tokens`：跨轮基线的唯一来源（落 session config）
    - `peak_prompt_tokens`：仅供展示/落库，**不参与**跨轮判据
    """
    loop = _loop(max_tokens_budget=100000)
    loop.llm = _UsageLLM(prompt_tokens=4321, completion_tokens=7)

    events = _drain(loop)

    done = [e for e in events if e.type == "done"]
    assert done, "必须收到 done 事件"
    data = done[0].data
    assert data["peak_prompt_tokens"] == 4321, "峰值应为本次调用的真实 prompt"
    assert data["first_prompt_tokens"] == 4321, "单次调用下首步 == 峰值（跨轮基线取它）"
    assert data["prompt_tokens"] == 4321, "单次调用下累计值等于单次值"
    assert loop._last_real_prompt_tokens == 4321, "内部字段也要刷新（供下一次请求判据）"

    # ★★ P1-a（2026-10-03）：**分母与警示线必须真的下发**。
    # ⚠️ 本条补的是一个**测试盲区**：P1-a 当时只写了前端 `ctxUsage()` 的纯函数测试，
    #   而后端若漏加（或加错位置），**前端测试照样全绿、功能却完全不工作**
    #   —— 典型的"假通过"。前端只负责**算**，"数"得后端给。
    assert data["max_tokens_budget"] == 100_000, "done 必须带分母（前端算占用率要用）"
    assert data["compact_trigger_tokens"] == 70_000, (
        "警示线须由后端下发（前端不乘 0.7，否则=再造一把尺子）"
    )


class _TwoStepUsageLLM(FakeLLM):
    """首轮调 `retrieve`、次轮作答；两步返回**不同**的 prompt_tokens。

    用于验证"峰值取 max、首步取第一次"这条**同一轮内**的区分 —— 单步替身做不到。
    """

    def __init__(self, first_pt: int, second_pt: int) -> None:
        super().__init__()
        self._first_pt = first_pt
        self._second_pt = second_pt
        self.n = 0

    async def chat(self, messages, tools=None, **kw):  # noqa: ANN001, ANN003
        self._calls.append(messages)
        self.n += 1
        if self.n == 1:
            return LLMResponse(
                message=ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                    ToolCall(id="c1", name="retrieve", arguments={"query": "q"}),
                ]),
                prompt_tokens=self._first_pt, completion_tokens=5,
                model="stub", latency_ms=1, cost=0.0,
            )
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="答案"),
            prompt_tokens=self._second_pt, completion_tokens=5,
            model="stub", latency_ms=1, cost=0.0,
        )


def test_peak_takes_max_within_turn_but_first_stays_first() -> None:
    """★ 同一轮内：峰值取 **max**，首步**固定为第一次**（后续调用不覆盖）。

    为什么必须区分：首步 = "进入本轮时的上下文规模"（= 下一轮会继承的部分），
    而峰值含本轮工具结果 —— 那部分**不进下一轮历史**，拿它当基线会系统性高估。
    """
    loop = _loop(max_tokens_budget=100000)
    loop.llm = _TwoStepUsageLLM(first_pt=3000, second_pt=9000)

    events = _drain(loop)
    data = [e for e in events if e.type == "done"][0].data

    assert data["peak_prompt_tokens"] == 9000, "峰值取本轮 max（工具结果撑大了第二轮）"
    assert data["first_prompt_tokens"] == 3000, "首步必须是第一次调用的值，不得被后续调用抬高"
    assert data["prompt_tokens"] == 12000, "累计值 = 两次之和（与单次口径严格不同）"


def test_first_and_peak_are_per_turn_no_ratchet() -> None:
    """★★ 跨轮基线**不得继承**：某轮的峰值不能成为下一轮的起点（只增不减的棘轮）。

    实测事故（2026-09-28）：会话里某轮峰值 29,357 被回写后，之后每轮首步实际仅 15,411
    （高估 87%），却仍以 29,357 为基线 ⇒ 压缩每轮误触发、闸门每轮都试图裁历史工具结果。
    """
    loop = _loop(max_tokens_budget=100000)

    async def simulate(values: list[int]) -> None:
        for v in values:
            loop.llm = _UsageLLM(prompt_tokens=v)
            async for _ in loop.run_stream("问"):
                pass

    asyncio.run(simulate([9000, 3000]))

    assert loop._peak_prompt_tokens == 3000, "第二轮峰值必须是它自己的，不得继承上一轮的 9000"
    assert loop._first_prompt_tokens == 3000, "首步同样按轮结算"
    assert loop._last_real_prompt_tokens == 3000, "单次值应是最新一次"


# ============================================================
# P0（2026-10-03）：压缩触发/执行的**口径统一**
# ============================================================
class TestHistoryCapFor:
    """`LoopConfig._history_cap_for`：把「整 prompt 目标」换算成「历史保留量」。

    ## 为什么需要这一层换算（P0 的根因）
    触发线的比较对象是「**整 prompt** 真实值」（21000），执行线的比较对象却是
    「**仅历史**估算值」。数值都从触发线派生（21000×0.7=14700）⇒ 看似同源，
    **量纲却是两回事**：`整 prompt = 固定开销 + 历史 + 本轮新增`。
    实测（2026-10-02 日志）：触发时整 prompt 21000+，**历史只有 11427**
    ⇒ `11427 < 14700` ⇒ 一条不丢（**空转**）。

    ⇒ 换算后：`hist_cap = hist_est − max(0, 整prompt基线 − 整prompt目标)`。
    """

    def test_default_case_matches_measured_incident(self):
        """★ 用 2026-10-02 线上实测数字验算，修复必须让该场景**真的丢**。"""
        cfg = LoopConfig()                     # 触发线 21000 · 整 prompt 目标 14699
        hist_est, baseline = 11427, 21000      # 实测：历史 11427、整 prompt 触发 21000
        cap = cfg._history_cap_for(hist_est, baseline)
        assert cap == 11427 - (21000 - cfg.compact_keep_capacity)
        assert hist_est > cap, (
            f"★ 回归：实测场景又变成空转（hist={hist_est} ≤ cap={cap}）—— P0 被绕过"
        )
        # 换算后保留量应明显小于历史总量（真的削掉了一块）
        assert cap < hist_est * 0.6, f"只削了一点点，收益不足：cap={cap}"

    def test_no_baseline_falls_back_to_legacy_rule(self):
        """★ 首轮（无真实基线）⇒ 退回旧判据（`hist_cap = 目标`），**行为逐字不变**。

        若这里也套公式（baseline=0 ⇒ need_cut=0 ⇒ hist_cap=hist_est）
        会**永不压缩** = 把首轮兜底整个删掉 ⇒ 回归。
        """
        cfg = LoopConfig()
        assert cfg._history_cap_for(999_999, 0) == cfg.compact_keep_capacity

    def test_baseline_below_target_means_no_cut(self):
        """整 prompt 基线**没超**目标 ⇒ 无需削（need_cut=0）。"""
        cfg = LoopConfig()
        target = cfg.compact_keep_capacity
        assert cfg._history_cap_for(5000, target - 1) == 5000

    def test_baseline_far_over_target_caps_at_zero_not_negative(self):
        """★★ 严重超目标 ⇒ 削光历史（下限 0），**不得出现负数**。

        ⚠️ 削光仍可能超限 —— 那由**既有熔断**兜住，不是本层职责
        （本层只负责"能削多少削多少"）。
        """
        cfg = LoopConfig()
        assert cfg._history_cap_for(50, 500_000) == 0

    def test_exact_boundary(self):
        """边界：基线恰好等于目标 ⇒ need_cut=0 ⇒ 原样保留（不削）。"""
        cfg = LoopConfig()
        target = cfg.compact_keep_capacity
        assert cfg._history_cap_for(1234, target) == 1234
        # 目标 +1 ⇒ 削 1
        assert cfg._history_cap_for(1234, target + 1) == 1233

    def test_target_and_trigger_are_derived_from_budget(self):
        """两条线都随预算派生（避免"调了一个、另一个没调"）。"""
        a, b = LoopConfig(max_tokens_budget=10000), LoopConfig(max_tokens_budget=30000)
        assert a.compact_trigger_tokens == 7000 and b.compact_trigger_tokens == 21000
        assert a.compact_keep_capacity < b.compact_keep_capacity
