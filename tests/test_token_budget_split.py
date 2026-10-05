"""token 上限「拆两半」回归（2026-09-17）

**背景（真实数据）**：`max_tokens_budget` 原先一个数当两个用 ——
既做「整轮累计熔断」（`_acc_tokens` 累加每次调用），又是「历史压缩阈值」的来源（`// 2`）。

实测（真实库 / `messages.prompt_tokens`）：

| 会话 | 单轮 prompt 实测 |
|---|---|
| 工作间 s15（qwen3-max） | 4818 / 10063 / 10129 |
| s18（qwen3-vl-flash） | **24883** / 16574 / 16938 |

→ 阈值 30000 只够 **2~3 轮** LLM 调用，而 `max_iterations=12`（允许 12 轮）。
两次真实失败正是如此：`s15` 累计 33420、`s18` 累计 36495 → "token 预算超支"熔断，
**而模型本身没有任何问题**（s18 那轮还正常调用了 3 次检索）。

拆分后两个口径各自清晰：
- `max_tokens_budget`   = **单次请求上下文上限**（单轮 prompt 超限即中止；兼作压缩阈值来源）
- `max_tokens_per_turn` = **整轮累计上限**（含所有迭代；与 max_cost_per_turn 互补）

本文件锁住：① 单轮超限会熔断；② **单轮 24883 这类正常规模不再误伤**（核心回归）；
③ 累计超限会熔断；④ 两个阈值互相独立（改一个不影响另一个）。
"""

from __future__ import annotations

import asyncio

from lightcore.core import failure
from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, Role, ToolCall
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry


def _registry() -> ToolExecutor:
    reg = ToolRegistry()
    register_builtin_tools(reg)
    return ToolExecutor(reg)


class _TokLLM(BaseLLM):
    """可控 prompt/completion tokens 的桩；不返回工具调用 → 一轮即 FINISH。"""

    def __init__(self, prompt_tokens: int, completion_tokens: int = 10) -> None:
        self._pt = prompt_tokens
        self._ct = completion_tokens
        self.calls = 0
        self.model_name = "stub"

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        self.calls += 1
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="最终答复"),
            prompt_tokens=self._pt, completion_tokens=self._ct,
            model="stub", latency_ms=1, cost=0.0,
        )


class _ToolThenTextLLM(BaseLLM):
    """第 1 轮返回工具调用（制造第 2 轮），第 2 轮返回文本 —— 用于把「整轮累计」推过上限。"""

    def __init__(self, prompt_tokens: int, completion_tokens: int = 10) -> None:
        self._pt = prompt_tokens
        self._ct = completion_tokens
        self.calls = 0
        self.model_name = "stub"

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                message=ChatMessage(role=Role.ASSISTANT, content="",
                                    tool_calls=[ToolCall(id="c1", name="calculator",
                                                         arguments={"expression": "1+1"})]),
                prompt_tokens=self._pt, completion_tokens=self._ct,
                model="stub", latency_ms=1, cost=0.0,
            )
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="最终答复"),
            prompt_tokens=self._pt, completion_tokens=self._ct,
            model="stub", latency_ms=1, cost=0.0,
        )


def _run(loop: AgentLoop, query: str = "开始") -> list:
    async def run():
        return [ev async for ev in loop.run_stream(query)]

    return asyncio.run(run())


def _errors(events: list) -> list[str]:
    return [str(e.data.get("reason", "")) for e in events if e.type == "error"]


# --------------------------------------------------------------- ① 单轮上下文超限


def test_single_request_over_context_limit_aborts() -> None:
    """单轮 prompt 超过上限 → 中止（这是它的新语义：防上下文爆炸）。"""
    llm = _TokLLM(prompt_tokens=40000)
    loop = AgentLoop(llm, _registry(), config=LoopConfig(max_tokens_budget=30000))

    events = _run(loop)

    reasons = _errors(events)
    assert reasons, f"必须熔断：{reasons}"
    assert "单次请求上下文超限" in reasons[0]
    assert "40000" in reasons[0], "错误信息要带上实际值，便于用户判断是否需要调高"
    # ★ C2（2026-09-28）**契约变更**：中止路径**必须**发 done（本轮真实用量）。
    # 旧契约在此断言"没有 done"，而那正是缺陷本身 —— 缺 done 会让路由层改用
    # `count_usage()`（**整段会话累计**）展示与落库，与 `[本轮]` 日志的真实值不一致。
    done = next(e for e in events if e.type == "done")
    assert done.data.get("aborted") is True, "中止轮必须标记 aborted，供路由层写结构化 notice"
    assert done.data.get("failure_kind") == failure.BUDGET_EXCEEDED
    assert done.data["prompt_tokens"] == 40000, "用量必须是**本轮真实值**（不得回落成会话累计）"
    assert llm.calls == 1, "超限应立即中止，不做第二次调用"


# ------------------------------------------------- ② 核心回归：正常规模不再误伤


def test_real_world_single_turn_prompt_no_longer_aborts() -> None:
    """**核心回归**：实测过的真实规模（单轮 prompt 24883、绑 3 篇文档）必须能正常跑完。

    修复前：`_acc_tokens`（累计口径）下 30000 只够 2-3 轮 → 这类正常会话必然被熔断。
    """
    llm = _TokLLM(prompt_tokens=24883, completion_tokens=585)
    loop = AgentLoop(llm, _registry(),
                     config=LoopConfig(max_tokens_budget=30000, max_tokens_per_turn=150000))

    events = _run(loop)

    assert _errors(events) == [], f"真实规模的单轮 prompt 不应熔断：{_errors(events)}"
    assert any(e.type == "done" for e in events)


def test_multi_iteration_within_cumulative_limit_completes() -> None:
    """多轮迭代只要累计没超上限，就应跑完（旧口径下这里会被误杀）。"""
    llm = _ToolThenTextLLM(prompt_tokens=20000, completion_tokens=500)
    loop = AgentLoop(llm, _registry(),
                     config=LoopConfig(max_iterations=4, max_tokens_budget=30000,
                                       max_tokens_per_turn=150000))

    events = _run(loop)

    assert _errors(events) == [], f"两轮 × 20500 = 41000 < 150000 不应熔断：{_errors(events)}"
    assert any(e.type == "done" for e in events)


# --------------------------------------------------------------- ③ 整轮累计超限


def test_cumulative_tokens_over_turn_limit_aborts() -> None:
    """累计口径仍有效：多轮叠加超过 `max_tokens_per_turn` → 中止（防"跑飞"）。"""
    llm = _ToolThenTextLLM(prompt_tokens=200, completion_tokens=50)
    loop = AgentLoop(llm, _registry(),
                     # 单轮 250 远低于上下文上限；但累计上限只给 300 → 第 2 轮就该停
                     config=LoopConfig(max_iterations=4, max_tokens_budget=30000,
                                       max_tokens_per_turn=300))

    events = _run(loop)

    reasons = _errors(events)
    assert reasons, f"累计超限必须熔断：{reasons}"
    assert "本轮累计 token 超限" in reasons[0]
    # C2（2026-09-28）：同测试 ① —— 熔断轮也要发 done（本轮口径），否则用量退化成会话累计。
    done = next(e for e in events if e.type == "done")
    assert done.data.get("aborted") is True
    assert done.data.get("failure_kind") == failure.BUDGET_EXCEEDED


# --------------------------------------------------------------- ④ 两个阈值互相独立


def test_two_limits_are_independent() -> None:
    """单轮超限被拦、但累计额度充足时，错误原因必须是"上下文超限"而非"累计超限"（不串味）。"""
    llm = _TokLLM(prompt_tokens=50000)
    loop = AgentLoop(llm, _registry(),
                     config=LoopConfig(max_tokens_budget=30000, max_tokens_per_turn=10_000_000))

    reasons = _errors(_run(loop))

    assert reasons and "单次请求上下文超限" in reasons[0]
    assert "累计" not in reasons[0], "两个口径不得互相冒充"


def test_defaults_match_documented_values() -> None:
    """默认值即文档承诺：单轮 30000（真实单轮可达 2.5 万）+ 整轮 150000。"""
    cfg = LoopConfig()

    assert cfg.max_tokens_budget == 30000
    assert cfg.max_tokens_per_turn == 150000
