"""C1（2026-09-28）：无进展检测 —— 判据从「工具消息**条数**」改为「**是否获得新证据**」。

## 被修的缺陷

原实现（`loop.py`）：

```python
new_tool_count = len([m for m in messages if m.role == Role.TOOL])   # ← 数的是**条数**
if new_tool_count == last_tool_count:
    no_progress_steps += 1
if no_progress_steps >= self.config.max_steps_no_progress:  # ← 6，**永不可达**
    ...中止...
```

该判据位于 ACT **之后**，而能走到这里的每一步都至少追加了一条 TOOL 消息
⇒ 计数**单调递增** ⇒ `>= 6` 不可达 ⇒ **唯一的"无进展"预警是死代码**。

已付代价（实测）：2026-09-27 一次 **9 轮 / 23 秒**反复读 page 8/9/10
（同一批内容读 2~3 遍），**全程零预警**，最终靠用户点「停止」脱身。

## 新判据与新边界

- 判据 = **本步是否产出新的"证据键"**（`chunk:<id>` / 页内容指纹），
  对齐语义（"有没有拿到新信息"）而不是计数（"追加了几条消息"）。
- **产不出证据键的工具**（答案结构里既无 `results` 也无 `text`+`page`）⇒
  `_is_key_trackable()` 返回 False ⇒ **判不了**，必须保守视为"有进展"。
  否则会把"换用非检索类工具"误报成原地打转（**假阳性**在他处会被当成"卡住"）。
- ⚠️ **动作 = 换策略（补救），不是中止（放弃）**（2026-10-12 观察期收口）：
  跨越阈值时**注入一条系统注记**（要求换关键词 / 换工具 / 换参数），**仍然不得中止**。
  故本文件的两条核心断言是：①跨越阈值**必须**注入注记，且**只注入一次**；
  ②连续无进展**超过**阈值也**不得**以 `NO_PROGRESS` 中止。
  取舍理由：中止是放弃、换策略是补救；未证伪误报前不得引入中止能力（质量优先）。
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from lightcore.core import failure
from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, LLMResponse, Role, ToolCall
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec

LOGGER_NAME = "lightcore.core.loop"


# --------------------------------------------------------------------- 替身


class _AlwaysToolLLM:
    """永远返回一次 `probe` 工具调用 → 一直不收敛（迭代耗尽由外层收尾）。"""

    def __init__(self) -> None:
        self.model_name = "test-model"
        self.n = 0
        #: 记录每次调用收到的 messages —— 用于断言「换策略注记」**确实被注入**（而非只写日志）。
        self.seen: list[list] = []

    async def chat(self, messages, *args, **kw):  # noqa: ANN001, ANN002
        self.n += 1
        self.seen.append(list(messages))
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id=f"c{self.n}", name="probe", arguments={"query": "q"}),
            ]),
            prompt_tokens=50, completion_tokens=5, latency_ms=3, model=self.model_name,
        )


def _registry(output_factory) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="probe", description="测试用工具",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission=Permission.SAFE, handler=lambda **_kw: ToolResult(ok=True, output=output_factory()),
    ))
    return reg


def _same_chunk() -> dict:
    """**每步都命中同一个 chunk** ⇒ 第一次之后不再产出新键（= 原地打转）。"""
    return {"results": [{"id": "doc-1-c0", "page": 1, "score": 0.9, "text": "同一段正文"}]}


_counter = {"i": 0}


def _fresh_chunk() -> dict:
    """**每步命中新 chunk** ⇒ 每步都有新证据（= 正常推进）。"""
    _counter["i"] += 1
    return {"results": [{"id": f"doc-1-c{_counter['i']}", "page": 1, "score": 0.9,
                         "text": f"第 {_counter['i']} 段正文"}]}


def _no_keys() -> dict:
    """答案结构里既无 `results` 也无 `text`+`page` ⇒ **判不了**（非检索类工具）。"""
    return {"value": 42, "unit": "mm"}


def _run(loop: AgentLoop) -> list:
    async def run():
        return [ev async for ev in loop.run_stream("开始")]
    return asyncio.run(run())


def _progress_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[无进展]" in r.getMessage()]


def _hint_notes(msgs) -> list:  # noqa: ANN001
    """本次 LLM 调用收到的 messages 里的「换策略注记」（按内容识别，不依赖位置）。"""
    return [
        m for m in msgs
        if m.role == Role.SYSTEM and "无进展提醒" in (m.content or "")
    ]


# ------------------------------------------------------------------- 断言


def test_repeated_same_chunk_is_logged_as_no_progress(caplog) -> None:
    """★ 每步命中同一 chunk ⇒ 必须出现 `[无进展]` 日志（旧的死判据下**一行都没有**）。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    loop = AgentLoop(_AlwaysToolLLM(), ToolExecutor(_registry(_same_chunk)),
                     LoopConfig(max_iterations=4, max_steps_no_progress=2))

    _run(loop)

    lines = _progress_lines(caplog)
    assert lines, "重复命中同一 chunk 却没有任何 [无进展] 日志 —— 判据又退化成数条数了"
    assert any("连续 1/2 步" in ln for ln in lines), lines


def test_threshold_crossing_emits_warning_once(caplog) -> None:
    """跨过阈值那一刻发**一条** warning；不是每步都刷（否则日志不可用）。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    loop = AgentLoop(_AlwaysToolLLM(), ToolExecutor(_registry(_same_chunk)),
                     LoopConfig(max_iterations=6, max_steps_no_progress=2))

    _run(loop)

    warns = [r for r in caplog.records
             if "[无进展]" in r.getMessage() and r.levelno >= logging.WARNING]
    assert len(warns) == 1, f"跨阈值应只报一次，实际 {len(warns)} 次"


def test_no_progress_injects_hint_and_never_aborts(caplog) -> None:
    """★★ 契约（2026-10-12 收口版）：跨阈值 ⇒ **注入换策略注记**，但**绝不因 NO_PROGRESS 中止**。

    终止原因必须是"达到最大迭代"（既有路径），且 `failure_kind` 不得是 `no_progress`
    —— 一旦这条红了，说明有人给新判据接上了**中止**能力（那必须先证明误报率）。
    """
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    llm = _AlwaysToolLLM()
    loop = AgentLoop(llm, ToolExecutor(_registry(_same_chunk)),
                     # 上限 2，却跑满 5 轮 ⇒ 远超阈值
                     LoopConfig(max_iterations=5, max_steps_no_progress=2))

    events = _run(loop)

    err = next(e for e in events if e.type == "error")
    assert "达到最大迭代" in err.data["reason"], err.data["reason"]
    assert err.data.get("failure_kind") == failure.MAX_ITERATIONS
    assert err.data.get("failure_kind") != failure.NO_PROGRESS
    # 反向证据：确实跨过了阈值（否则"没中止"是空谈）
    assert any("连续 2/2 步" in ln for ln in _progress_lines(caplog))
    # ★ 动作确实发生：注记被**注入**（不是只写日志）
    notes = _hint_notes(llm.seen[-1])
    assert notes, "跨阈值却没有任何换策略注记 —— 动作出口又变空了"
    assert len(notes) == 1, f"注记应只注入一次，实际 {len(notes)} 条"


def test_strategy_hint_injected_only_once(caplog) -> None:
    """注记只在**跨越阈值那一刻**注入一次；后续步不得重复注入（否则等于每步都在改 prompt）。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    llm = _AlwaysToolLLM()
    loop = AgentLoop(llm, ToolExecutor(_registry(_same_chunk)),
                     LoopConfig(max_iterations=7, max_steps_no_progress=2))

    _run(loop)

    for i, msgs in enumerate(llm.seen):
        n = len(_hint_notes(msgs))
        assert n <= 1, f"第 {i} 次调用里出现了 {n} 条换策略注记（应至多 1 条）"
    assert len(_hint_notes(llm.seen[-1])) == 1, "跑满多轮后注记应恰好 1 条"


def test_fresh_chunks_are_never_flagged(caplog) -> None:
    """反向自检：每步都拿到**新** chunk ⇒ 一行 `[无进展]` 都不该有（防"永远报"）。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    _counter["i"] = 0
    loop = AgentLoop(_AlwaysToolLLM(), ToolExecutor(_registry(_fresh_chunk)),
                     LoopConfig(max_iterations=4, max_steps_no_progress=2))

    _run(loop)

    assert _progress_lines(caplog) == [], "拿到新证据却被报成无进展 —— 判据方向反了"


def test_untrackable_tool_is_not_flagged(caplog) -> None:
    """★ 假阳性防线：产不出证据键的工具 ⇒ 判不了 ⇒ 保守视为有进展（不得报警）。

    否则任何"用非检索类工具"的正常轮次都会被报成原地打转。
    """
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    loop = AgentLoop(_AlwaysToolLLM(), ToolExecutor(_registry(_no_keys)),
                     LoopConfig(max_iterations=5, max_steps_no_progress=2))

    _run(loop)

    assert _progress_lines(caplog) == [], "无法判定的工具被报成无进展（假阳性）"


@pytest.mark.parametrize("output,expected", [
    ({"results": []}, True),                                   # retrieve 空命中：仍可判定
    ({"results": [{"id": "x", "text": "y"}]}, True),
    ({"text": "整页正文", "page": 3}, True),                    # pdf_extract(page=)
    ({"value": 42}, False),                                    # 非检索类：判不了
    ("纯字符串输出", False),
])
def test_is_key_trackable_contract(output, expected) -> None:
    """`_is_key_trackable` 的口径表（它是"判得了 / 判不了"的唯一来源）。"""
    from lightcore.core.loop import _is_key_trackable

    assert _is_key_trackable(output) is expected
