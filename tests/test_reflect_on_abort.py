"""`Reflector` 首次接线回归（2026-09-16，P1）。

背景：`lightcore/planning/reflector.py` 的 `Reflector`（Reflexion 式自评）**写了从没被调用过**
——全库无引用，属"死代码 / 能力预留"。这里给它接上唯一划算的触发面：
**只在循环中止路径**（迭代耗尽 / 连续无进展）自评一次；判 `NEEDS_IMPROVEMENT` 则带建议重生成一次。

为什么只挂在中止路径：那一轮**本来就要失败**，多花 1~2 次调用换一个更好的收尾是值得的；
**正常收敛的轮次零额外调用**（本文件第 1 个用例就是守这条）。

三重护栏也在本文件里逐一守住：
- 限 `max_tokens`（用例断言自评调用收到 400）；
- 超时 → 沿用原草稿（用例用 monkeypatch 把超时压到 10ms）；
- 异常 → 沿用原草稿。
"""

from __future__ import annotations

import asyncio

from lightcore.core import loop as loop_mod
from lightcore.core.loop import REFLECT_MAX_TOKENS, AgentLoop, LoopConfig, LoopState
from lightcore.llm.base import ChatMessage, LLMResponse, Role, ToolCall
from lightcore.llm.fake import FakeLLM

# C7-lightcore（2026-09-23）：这两个标记**改从单一来源导入**，不再本地硬编码。
#
# 为什么这是真问题（不是洁净癖）：本文件的 `ScriptedLLM` 用它们**给假模型路由返回** ——
# `text.startswith(REFLECT_PROMPT_HEAD)` 决定"这一轮当成反思调用"，`GUIDANCE_MARK in text`
# 决定"返回改进稿还是初稿"。提示词一改、而这两行字面串没跟着改 →
# **假模型再也匹配不上，测试静默走错分支**：仍会通过，但判的已不是原来那件事。
from lightcore.planning.reflector import REFLECT_PROMPT
from lightcore.prompts import GUIDANCE_MARK
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec

#: 反思提示词的**首行**（取真实提示词的前缀，而非复制一份字面串）。
#: ⚠️ 这里**刻意不带末尾的「。」**：真实提示词是「请评价…满足目标。」，
#: 用非空前缀做 `startswith` 判定即可，同时避免与提示词末尾标点耦合。
REFLECT_PROMPT_HEAD = REFLECT_PROMPT.split("。", 1)[0]


class ScriptedLLM(FakeLLM):
    """按调用形态分派的替身：

    - **带 tools**（THINK 轮）→ 永远返回 `noop` 工具调用 ⇒ 循环永不收敛，必然走到中止路径；
    - **不带 tools 且 prompt 是自评** → 返回 `reflect_reply`（可令其抛错 / 变慢）；
    - **不带 tools 且 prompt 含改进建议** → 返回 `improved`（= 带建议重生成）；
    - 其余不带 tools → 返回 `draft`（首次兜底收尾）。
    """

    def __init__(self, *, reflect_reply: str = "SATISFIED", draft: str = "草稿回答",
                 improved: str = "改进后的回答", reflect_delay: float = 0.0,
                 reflect_raises: bool = False) -> None:
        super().__init__()
        self.reflect_reply = reflect_reply
        self.draft = draft
        self.improved = improved
        self.reflect_delay = reflect_delay
        self.reflect_raises = reflect_raises
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        text = messages[-1].content if messages else ""
        self.calls.append({"tools": tools, "text": text, "max_tokens": max_tokens})
        if tools:
            r = await super().chat(messages, tools, temperature=temperature,
                                   max_tokens=max_tokens, tool_choice=tool_choice)
            r.message.tool_calls = [ToolCall(id="c", name="noop", arguments={})]
            r.message.content = "calling"
            return r
        if isinstance(text, str) and text.startswith(REFLECT_PROMPT_HEAD):
            if self.reflect_delay:
                await asyncio.sleep(self.reflect_delay)
            if self.reflect_raises:
                raise RuntimeError("reflect boom")
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=self.reflect_reply))
        body = self.improved if (isinstance(text, str) and GUIDANCE_MARK in text) else self.draft
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=body))

    # 便于用例筛选
    @property
    def reflect_calls(self) -> list[dict]:
        return [c for c in self.calls if isinstance(c["text"], str)
                and c["text"].startswith(REFLECT_PROMPT_HEAD)]

    @property
    def regenerate_calls(self) -> list[dict]:
        return [c for c in self.calls if isinstance(c["text"], str) and GUIDANCE_MARK in c["text"]]


def _make_loop(llm: ScriptedLLM, *, reflect_on_abort: bool, max_iterations: int = 3) -> AgentLoop:
    """构造「永远调用 noop 工具」的工具集 —— 复现不收敛场景，必然走到中止路径。"""
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="noop", description="noop", parameters={"type": "object", "properties": {}},
        permission=Permission.SAFE,
        handler=lambda: ToolResult(ok=True, output="empty"),
    ))
    return AgentLoop(
        llm, ToolExecutor(reg),
        config=LoopConfig(max_iterations=max_iterations, reflect_on_abort=reflect_on_abort),
    )


async def _run(loop: AgentLoop) -> list:
    return [ev async for ev in loop.run_stream("帮我总结一下")]


def _final_text(events: list) -> str:
    answers = [e.data["content"] for e in events
               if e.type == "message" and (e.data or {}).get("content")]
    assert answers, "中止路径必须产出实质回答（否则用户只看到「本次没有生成有效回答」）"
    return answers[-1]


def test_disabled_keeps_original_behavior_and_costs_nothing_extra():
    """开关关闭（默认）：**零额外调用**，行为与 2026-09-10 版完全一致。"""
    llm = ScriptedLLM()
    events = asyncio.run(_run(_make_loop(llm, reflect_on_abort=False)))

    assert llm.reflect_calls == [], "未开启时不得调用自评（正常收敛的轮次更不该有）"
    assert llm.regenerate_calls == []
    assert _final_text(events) == "草稿回答"
    assert any(e.type == "error" and e.state == LoopState.ABORT for e in events), "中止信号须保留"


def test_needs_improvement_triggers_exactly_one_regeneration():
    """开启 + 判「需改进」→ 带建议重生成**一次**，最终回答取改进版。"""
    llm = ScriptedLLM(
        reflect_reply="NEEDS_IMPROVEMENT\n反思: 只罗列了检索片段\n改进: 补上一句明确结论，并标出依据页码",
        draft="草稿回答", improved="改进后的回答",
    )
    events = asyncio.run(_run(_make_loop(llm, reflect_on_abort=True)))

    assert len(llm.reflect_calls) == 1, "自评只做一次（避免因自评再引入循环）"
    assert len(llm.regenerate_calls) == 1, "只需一次带建议重生成"
    guidance = llm.regenerate_calls[0]["text"]
    assert "补上一句明确结论" in guidance, "改进建议必须真的进入重生成指令（否则自评白做）"
    assert _final_text(events) == "改进后的回答"
    assert any(e.type == "error" and e.state == LoopState.ABORT for e in events)


def test_reflect_call_is_token_capped():
    """护栏①：自评属「元任务」，必须限 max_tokens（N11 教训：否则 thinking 模型吃满思考链）。"""
    llm = ScriptedLLM(reflect_reply="NEEDS_IMPROVEMENT\n改进: 补结论")
    asyncio.run(_run(_make_loop(llm, reflect_on_abort=True)))
    assert llm.reflect_calls, "应已调用自评"
    assert llm.reflect_calls[0]["max_tokens"] == REFLECT_MAX_TOKENS


def test_satisfied_skips_regeneration():
    """自评判「已满足」→ 不做无谓的重生成（省一次调用）。"""
    llm = ScriptedLLM(reflect_reply="SATISFIED", draft="草稿回答", improved="不该出现")
    events = asyncio.run(_run(_make_loop(llm, reflect_on_abort=True)))

    assert len(llm.reflect_calls) == 1
    assert llm.regenerate_calls == [], "SATISFIED 时不得重生成"
    assert _final_text(events) == "草稿回答"


def test_reflect_exception_falls_back_to_draft():
    """护栏③：自评抛异常 → 沿用原草稿，且中止信号照常下发（绝不阻断收尾）。"""
    llm = ScriptedLLM(reflect_raises=True, draft="草稿回答")
    events = asyncio.run(_run(_make_loop(llm, reflect_on_abort=True)))

    assert _final_text(events) == "草稿回答"
    assert any(e.type == "error" and e.state == LoopState.ABORT for e in events)


def test_reflect_timeout_falls_back_to_draft(monkeypatch):
    """护栏②：自评超时 → 沿用原草稿（不把失败轮次拖得更慢）。"""
    monkeypatch.setattr(loop_mod, "REFLECT_TIMEOUT_S", 0.01)
    llm = ScriptedLLM(reflect_reply="NEEDS_IMPROVEMENT\n改进: 补结论",
                      draft="草稿回答", reflect_delay=0.2)
    events = asyncio.run(_run(_make_loop(llm, reflect_on_abort=True)))

    assert _final_text(events) == "草稿回答"
    assert llm.regenerate_calls == [], "自评超时后不得再走重生成"


def test_reflect_skipped_when_no_draft():
    """边界：兜底收尾没产出内容（草稿为空）→ 不启动自评（没有可评价的对象）。"""
    llm = ScriptedLLM(draft="", reflect_reply="NEEDS_IMPROVEMENT\n改进: 补结论")
    asyncio.run(_run(_make_loop(llm, reflect_on_abort=True)))
    assert llm.reflect_calls == []
