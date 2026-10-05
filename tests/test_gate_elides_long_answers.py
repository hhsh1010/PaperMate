"""上下文闸门 ③（2026-10-02）：工具结果不够时，裁剪早期**长回答 / 技能产物**。

## 为什么需要（实测，`docs/诊断-离题回答与熔断预警-2026-10-02.md` §3.3）

真实会话（session 18，4 篇文档）里历史的大头是 `/精读`·`/综述`·`/术语表` 的**长回答**，
而闸门原先**只认工具结果** ⇒ 实测两次日志分别是「没有「值得裁剪」的早期工具结果」
与「只省 1269 tokens」⇒ 对本类会话近乎无效 ⇒ 最后表现为「**突然熔断**」。

## 本文件在守什么（三条硬约束，缺一即"用质量换 token"）

1. ★★ **只替换内容，绝不删消息** —— 删一条会拆散 `assistant.tool_calls ↔ tool` 配对（厂商 400）；
2. ★★ **保留 `tool_calls` 与 `reasoning`** —— 带工具调用的轮次必须回传 `reasoning_content`
   （`llm/router.py:40`），丢掉它同样 400；
3. ★ **占位里保留开头片段** —— 模型要能认出"这是哪一轮的产物"，而不是面对一句无来由的省略。

另守一条**顺序契约**：工具结果优先（第一阶段），不够才动长回答（第二阶段）
⇒ 工具结果够用时行为与改动前**逐字一致**（既有 `test_token_gate.py` 即为此的回归网）。
"""

from __future__ import annotations

import logging
import re

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, Role, ToolCall

_LOGGER = "lightcore.core.loop"
_PROJ = re.compile(r"投影 (\d+)")


def _loop(*, budget: int = 1000, baseline: int = 900) -> AgentLoop:
    """最小替身：只塞闸门所需字段（与 `test_token_gate.py` 同款，不跑 `__init__`）。"""
    loop = object.__new__(AgentLoop)
    loop.config = LoopConfig(max_tokens_budget=budget)  # 触发线 = 0.7 × budget
    loop._last_real_prompt_tokens = baseline
    loop._steps = 0
    return loop


def _long_answer(n: int = 2000, *, head: str = "精读报告开头", tool_calls=None, reasoning=None):
    return ChatMessage(role=Role.ASSISTANT, content=head + "内容" * n,
                       tool_calls=tool_calls, reasoning=reasoning)


def _stage(*window_head: ChatMessage):
    """搭出**可裁窗口非空且够宽**的会话（本文件最容易搭错的点）。

    闸门只在前段 `[0 : min(_gate_mark, len-6))` 里裁 ⇒ 两个必要条件：
    ① 目标必须落在窗口内；② **总条数要足够大**（否则 `keep_from = len-6` 会把窗口压扁
    —— 我第一次就踩了：7 条 ⇒ `keep_from=1` ⇒ 窗口只剩第 0 条，工具结果根本没机会被裁）。
    布局：`window_head + 填充至 7 条`（`_gate_mark=7`）+ `[问答×2]` ⇒ 总 9 条 ⇒
    `keep_from=3` ⇒ 窗口 = `[0,3)`，头两条正好是传入的目标。
    """
    window = list(window_head)
    window += [ChatMessage(role=Role.USER, content="填充") for _ in range(7 - len(window))]
    mark = len(window)
    tail = [ChatMessage(role=Role.USER, content="后续提问"),
            ChatMessage(role=Role.ASSISTANT, content="短答")]
    return window + tail, mark


# ============ ① 工具结果优先（零回归） ============

class TestToolsFirst:
    def test_tools_suffice_means_answers_untouched(self, caplog):
        """★ 顺序契约：工具结果够用 ⇒ 长回答**一条都不许动**（既有行为逐字不变）。"""
        caplog.set_level(logging.INFO, logger=_LOGGER)
        loop = _loop()
        tool_msg = ChatMessage(role=Role.TOOL, content="工具结果" * 800, tool_call_id="c1")
        messages, loop._gate_mark = _stage(tool_msg, _long_answer())
        loop._gate_context(messages)  # noqa: SLF001

        # 窗口布局 = [工具结果, 长回答, 填充…] ⇒ 工具（索引 0）应被裁、长回答（索引 1）原样
        assert "早期工具结果已裁剪" in messages[0].content, f"工具结果应先被裁：{messages[0].content[:40]}"
        assert messages[1].content.startswith("精读报告开头"), "工具结果够用时不得动长回答"
        lines = [r.getMessage() for r in caplog.records if "[上下文闸门]" in r.getMessage()]
        assert lines and "省 ≈" in lines[0]


# ============ ② 长回答被裁（内容替换 + 硬约束） ============

class TestElidesLongAnswers:
    def _run(self) -> list[ChatMessage]:
        loop = _loop(budget=1000, baseline=900)  # 触发线 700；窗口内**没有**工具结果
        messages, loop._gate_mark = _stage(_long_answer(2000, head="精读报告开头"))
        loop._gate_context(messages)  # noqa: SLF001
        return messages

    def test_long_answer_is_replaced_with_stub(self):
        msgs = self._run()
        assert msgs[0].content.startswith("[早期长回答已裁剪："), f"未被裁剪：{msgs[0].content[:40]}"
        assert "早期工具结果已裁剪" not in msgs[0].content, "工具阶段不该误伤回答文案"

    def test_stub_keeps_head_for_recognizability(self):
        """★ 硬约束③：占位必须带开头片段，模型才能认出这是哪一轮的产物。"""
        msgs = self._run()
        stub = msgs[0].content
        assert "开头为「精读报告开头" in stub, f"占位丢了开头片段：{stub[:80]}"

    def test_message_count_unchanged(self):
        """★★ 硬约束①：**绝不删消息**（替换内容），否则会拆散 tool_calls↔tool 配对（400）。"""
        loop = _loop()
        messages, loop._gate_mark = _stage(_long_answer())
        before = len(messages)
        loop._gate_context(messages)  # noqa: SLF001
        assert len(messages) == before

    def test_short_answers_untouched(self):
        """阈值（800 字符）以下不动 —— 普通问答的中等长度回答不该被裁。"""
        loop = _loop()
        short = ChatMessage(role=Role.ASSISTANT, content="中等长度的回答" * 30)  # 210 字符
        messages, loop._gate_mark = _stage(short)
        loop._gate_context(messages)  # noqa: SLF001
        assert messages[0].content == short.content, "短回答被裁了（阈值失守）"


# ============ ③ tool_calls / reasoning 必须保留（否则 400） ============

class TestPreservesProtocolFields:
    def test_tool_calls_and_reasoning_kept(self):
        """★★ 硬约束②：被裁的长回答若带 `tool_calls`，两者都必须原样保留。"""
        loop = _loop(budget=1000, baseline=900)
        calls = [ToolCall(id="c1", name="retrieve", arguments={"query": "水利设施"})]
        msg = ChatMessage(role=Role.ASSISTANT, content="长回答" * 900,
                          tool_calls=calls, reasoning="推理链" * 50)
        messages, loop._gate_mark = _stage(msg)
        loop._gate_context(messages)  # noqa: SLF001

        assert messages[0].content.startswith("[早期长回答已裁剪："), "前置条件：该条应被裁"
        assert messages[0].tool_calls == calls, "裁掉了 tool_calls ⇒ 配对断裂（厂商 400）"
        assert messages[0].reasoning == msg.reasoning, (
            "裁掉了 reasoning ⇒ 带工具调用的轮次缺 reasoning_content（llm/router.py:40 ⇒ 400）"
        )

    def test_source_contract_two_stages(self):
        """源码契约：必须是"工具结果优先、长回答随后"，且都只替换内容。"""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "src" / "lightcore" / "core" / "loop.py").read_text(
            encoding="utf-8")
        tool_stage = src.index('stub = f"[早期工具结果已裁剪')
        ans_stage = src.index('stub = (f"[早期长回答已裁剪')
        assert tool_stage < ans_stage, "长回答阶段必须排在工具结果阶段之后"
        assert "if saved < need:" in src[tool_stage:ans_stage], (
            "长回答阶段必须以「工具结果不够」为前提（否则会误伤，且既有行为回归）"
        )
