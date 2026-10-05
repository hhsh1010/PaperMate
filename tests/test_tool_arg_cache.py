"""同 args 结果缓存（D3，2026-10-02）的守护测试。

## 它解决什么

实测（`docs/去重换策略-影响面评估.md`）：**6% 的工具调用是完全重复**（同工具同 args，
如 `pdf_extract(page=8)` 在一轮里被取两次）。重复执行要再付一次检索编码 / PDF 解析的**时间**；
而内容既然逐字相同，直接复用上一次的结果即可。

⚠️ 与 D2（结果级去重）的分工：D2 管"**注入**的分布"，D3 管"**执行**的次数"。
两者互不替代：D3 命中后，注入侧仍会走 D2 的去重（同长短 ⇒ 降级为占位），行为一致。

## 本文件在守什么（四条硬约束）

1. ★★ **只缓存只读检索类工具** —— 对写记忆 / 改稿 / plan_executor 等**有副作用**的工具缓存，
   会让"第二次调用"**静默不生效**（用户以为写了，其实没有）。
2. ★★ **只缓存成功结果** —— 失败/超时是瞬时的，缓存它等于把一次故障永久钉死。
3. **args 必须规范化**（键序无关）—— 否则模型换个键序就漏命中，缓存形同虚设。
4. **只在本轮内缓存** —— 跨轮历史可能被压缩、且"换角度重取"本身是有效行为。
"""

from __future__ import annotations

import asyncio
import logging

from lightcore.core.loop import _CACHEABLE_TOOLS, AgentLoop, LoopConfig, _tool_cache_key
from lightcore.llm.base import ChatMessage, LLMResponse, Role, ToolCall
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec

LOGGER_NAME = "lightcore.core.loop"

#: 有副作用的工具（**绝不能**进缓存白名单）——名字取自真实注册表，防止有人顺手扩白名单。
_SIDE_EFFECT_TOOLS = ("memory_write", "plan_executor", "docx_edit", "reindex", "arxiv_search")


class _TwoSameCallsLLM:
    """前两轮用**完全相同的 args** 调同一个工具，第三轮出最终答案。"""

    model_name = "stub"

    def __init__(self, tool: str = "retrieve", args: dict | None = None) -> None:
        self.n = 0
        self.tool = tool
        self.args = args or {"query": "q"}

    async def chat(self, messages, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        self.n += 1
        if self.n <= 2:
            return LLMResponse(
                message=ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                    ToolCall(id=f"c{self.n}", name=self.tool, arguments=dict(self.args)),
                ]),
                prompt_tokens=100, completion_tokens=10, latency_ms=5, model=self.model_name,
            )
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="最终答案"),
            prompt_tokens=500, completion_tokens=20, latency_ms=5, model=self.model_name,
        )


class _CountingHandler:
    """记录被真正执行了几次；可配置前 N 次失败（用于验"失败不入缓存"）。"""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail_first = fail_first

    def __call__(self, **kw):  # noqa: ANN003
        self.calls.append(kw)
        if self.fail_first and len(self.calls) == 1:
            return ToolResult(ok=False, output=None, error="boom")
        return ToolResult(ok=True, output={"results": [{"id": "d-c0", "page": 1, "score": 0.9,
                                                       "text": "正文" * 20}]})


def _loop(llm, tool: str, handler, **cfg) -> AgentLoop:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name=tool, description="测试用工具",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission=Permission.SAFE, handler=handler,
    ))
    return AgentLoop(llm, ToolExecutor(reg), LoopConfig(**cfg))


def _one_turn(loop: AgentLoop) -> None:
    async def _go():
        return [ev async for ev in loop.run_stream("问题", stream_tokens=False)]

    asyncio.run(_go())


def _tool_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[工具]" in r.getMessage()]


# ============ ① 命中：同 args 第二次不再执行 ============

def test_same_args_second_call_reuses_cached_result(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    handler = _CountingHandler()
    loop = _loop(_TwoSameCallsLLM(), "retrieve", handler)

    _one_turn(loop)

    assert len(handler.calls) == 1, f"同 args 第二次调用必须复用缓存，实际执行 {len(handler.calls)} 次"
    lines = _tool_lines(caplog)
    assert len(lines) == 2, lines
    assert "缓存=命中" not in lines[0], f"首次调用不可能是命中：{lines[0]}"
    assert "缓存=命中" in lines[1], f"第二次必须标注命中（否则收益无法验收）：{lines[1]}"
    # ★ 缓存对象必须是**未被污染的原始输出**：命中后仍要照常走 D2 去重（同长短 ⇒ 降级为占位）。
    #   若缓存被 D2 原地改写过，第二次会把"占位符"当成原文 → 去重计数变 0、占位符还会嵌套。
    assert "去重=1" in lines[1], f"命中路径仍必须走结果级去重（缓存对象被污染？）：{lines[1]}"


def test_cached_output_is_not_mutated_by_dedup():
    """单元层：同一条原始 output 反复过 D2 去重，结果必须一致（不得累积/嵌套占位）。"""
    from lightcore.core.loop import _dedup_tool_output, _prepare_tool_output

    raw = {"results": [{"id": "d-c0", "page": 1, "score": 0.9, "text": "正文" * 20}]}
    first_prepared = _prepare_tool_output("retrieve", raw)
    _, keys, dropped1 = _dedup_tool_output("retrieve", first_prepared, {})
    seen = dict(keys)
    assert dropped1 == 0

    second_prepared = _prepare_tool_output("retrieve", raw)  # 模拟"从缓存再取一次"
    out2, _, dropped2 = _dedup_tool_output("retrieve", second_prepared, seen)
    assert dropped2 == 1, "第二次应去重"
    placeholder = out2["results"][0]["text"]
    assert placeholder.count("重复命中") == 1, f"占位符不得嵌套：{placeholder[:60]}"
    assert raw["results"][0]["text"].startswith("正文"), "原始 output 不得被就地改写（缓存污染）"


def test_cache_key_is_order_insensitive():
    """args 键序不同必须命中同一缓存项（模型每次生成的键序可能不同）。"""
    assert _tool_cache_key("retrieve", {"a": 1, "b": 2}) == _tool_cache_key("retrieve", {"b": 2, "a": 1})
    assert _tool_cache_key("retrieve", {"a": 1}) != _tool_cache_key("retrieve", {"a": 2})
    assert _tool_cache_key("retrieve", {"a": 1}) != _tool_cache_key("pdf_extract", {"a": 1})


def test_different_args_are_not_cached(caplog):
    """args 不同 ⇒ 各自执行（缓存不得误命中）。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    handler = _CountingHandler()
    llm = _TwoSameCallsLLM(args={"query": "q1"})
    # 第二轮换成不同 args：直接改剧本
    orig_chat = llm.chat

    async def _chat(messages, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        resp = await orig_chat(messages, *a, **kw)
        if llm.n == 2 and resp.message.tool_calls:
            resp.message.tool_calls[0].arguments = {"query": "q2"}
        return resp

    llm.chat = _chat  # type: ignore[method-assign]
    loop = _loop(llm, "retrieve", handler)
    _one_turn(loop)

    assert len(handler.calls) == 2, "不同 args 必须各自执行"
    assert all("缓存=命中" not in x for x in _tool_lines(caplog))


# ============ ② 白名单：副作用工具绝不缓存 ============

def test_side_effect_tool_is_never_cached(caplog):
    """★ 硬约束①：非白名单工具（此处用 `memory_write`）第二次必须**真的再执行**。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    handler = _CountingHandler()
    loop = _loop(_TwoSameCallsLLM(tool="memory_write"), "memory_write", handler)

    _one_turn(loop)

    assert len(handler.calls) == 2, (
        "有副作用的工具被缓存 ⇒ 第二次调用静默不生效（用户以为写了其实没写）"
    )


def test_whitelist_excludes_side_effect_tools():
    """源码契约：白名单里不得出现任何副作用工具（防有人顺手扩）。"""
    for name in _SIDE_EFFECT_TOOLS:
        assert name not in _CACHEABLE_TOOLS, f"{name} 是副作用工具，绝不能进缓存白名单"
    assert "retrieve" in _CACHEABLE_TOOLS and "pdf_extract" in _CACHEABLE_TOOLS


# ============ ③ 失败不入缓存 ============

def test_failed_result_is_not_cached(caplog):
    """★ 硬约束②：第一次失败不得被缓存 —— 第二次必须真的重试。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    handler = _CountingHandler(fail_first=True)
    loop = _loop(_TwoSameCallsLLM(), "retrieve", handler)

    _one_turn(loop)

    assert len(handler.calls) == 2, "失败结果被缓存 ⇒ 瞬时故障被永久钉死"
    assert any("缓存=命中" not in x for x in _tool_lines(caplog))


# ============ ④ 只在本轮内 ============

def test_cache_does_not_leak_across_turns(caplog):
    """★ 硬约束④：同一个 loop 跑第二轮时缓存必须已清空（跨轮复用会掩盖"换角度重取"）。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    handler = _CountingHandler()
    llm = _TwoSameCallsLLM()
    loop = _loop(llm, "retrieve", handler)

    _one_turn(loop)
    assert len(handler.calls) == 1, "第一轮内应命中一次"

    llm.n = 0  # 第二轮用同一剧本（LLM 策略不属于本轮状态）
    caplog.clear()
    _one_turn(loop)

    lines2 = _tool_lines(caplog)
    # ★ 泄漏判据（关键）：若缓存跨轮残留，第二轮的**第一次**调用就会命中 ⇒
    #   执行次数停在 1、且 lines2[0] 带 `缓存=命中`。
    assert len(handler.calls) == 2, (
        f"每轮内应各执行一次（轮内第二次命中）—— 实际累计 {len(handler.calls)} 次"
        "；若停在 1 次说明缓存跨轮泄漏"
    )
    assert "缓存=命中" not in lines2[0], f"第二轮首次调用不得命中（缓存必须已随轮起点清空）：{lines2[0]}"
    assert "缓存=命中" in lines2[1], "第二轮**轮内**的第二次仍应命中"
