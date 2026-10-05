"""LightCore · FakeLLM 确定性替身（docs/05 / docs/08）

用途：
- 无网络、无 API key 即可在本地跑通 Agent 循环、SSE、单元测试（docs/08 明确要求 FakeLLM 替身）。
- 通过一组"剧本(script)"驱动：每条 user/tool 消息后，FakeLLM 按剧本返回固定响应，
  从而可复现地测试 THINK→ACT→OBSERVE→FINISH 状态机，不依赖真实模型。

设计：剧本是 (matcher, response_factory) 列表。matcher 为 str 时做子串匹配；
为 callable 时传入最后一条消息内容返回 bool。response 可为 LLMResponse 或 ChatMessage 或字符串。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from .base import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    Role,
    ToolCall,
    ToolSpecForLLM,
    _rough_tokens,
)


class FakeLLM(BaseLLM):
    def __init__(self, model_name: str = "fake-llm-1") -> None:
        self.model_name = model_name
        self._calls: list[list[ChatMessage]] = []  # 记录每次 chat 的历史，便于断言

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpecForLLM] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,  # noqa: ARG002  兼容 OAICompatibleLLM 的 tool_choice 形参（FakeLLM 默认 auto）
    ) -> LLMResponse:
        self._calls.append(list(messages))
        last = messages[-1] if messages else None
        last_content = last.content if last else ""
        resp_msg = self._decide(last_content, messages)
        prompt_tok = sum(_rough_tokens(m.content) + sum(_rough_tokens(json_args(m)) for m in m.tool_calls)
                         for m in messages)
        comp_tok = _rough_tokens(resp_msg.content) + sum(_rough_tokens(json_args(m)) for m in resp_msg.tool_calls)
        return LLMResponse(
            message=resp_msg,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            model=self.model_name,
            latency_ms=1,
        )

    # ---- 剧本驱动 ----
    def _decide(self, last_content: str, messages: list[ChatMessage]) -> ChatMessage:
        """默认剧本：若看到 OBSERVE 回传（含 'RESULT:'），则给出最终答案；否则调用一个示例工具。

        子类可覆盖 _decide，或在测试里直接替换 self.script。
        """
        # 如果上一条是工具结果，产出最终总结
        if last_content.startswith("RESULT:"):
            return ChatMessage(
                role=Role.ASSISTANT,
                content="根据检索结果，论文的核心贡献是：提出了一种轻量级 Agent 内核 LightCore。"
                "引用：[chunk-1]、[chunk-3]。",
            )
        # 否则：演示一次 ReAct 的 ACT —— 调用 retrieve 工具
        return ChatMessage(
            role=Role.ASSISTANT,
            content="我需要先检索相关段落。",
            tool_calls=[ToolCall(id="call_1", name="retrieve", arguments={"query": "LightCore 的核心贡献", "top_k": 3})],
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpecForLLM] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,  # noqa: ARG002  兼容透传
    ) -> AsyncIterator[LLMStreamChunk]:
        """离线增量：把 chat 结果按句/定长切分，逐 chunk 产出 delta（演示逐字渲染）。"""
        resp = await self.chat(messages, tools, temperature=temperature, max_tokens=max_tokens)
        msg = resp.message
        if msg.tool_calls:
            yield LLMStreamChunk(tool_calls=msg.tool_calls)
            yield LLMStreamChunk(finished=True, prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens)
            return
        text = msg.content
        parts = self._chunk_text(text)
        for p in parts:
            yield LLMStreamChunk(delta=p)
        # M6 修复（2026-09-10）：与 chat() 口径统一——原先把逐 chunk 重算的 token 数累加
        # 作为 completion_tokens，与 chat() 对整段文本的计算差若干 token（SSE 落库用量
        # 与非流式路径不一致）。直接复用 chat() 的统计结果。
        yield LLMStreamChunk(finished=True, prompt_tokens=resp.prompt_tokens,
                             completion_tokens=resp.completion_tokens)

    @staticmethod
    def _chunk_text(text: str, size: int = 8) -> list[str]:
        if not text:
            return []
        # 优先按句末标点切，保证逐字观感自然；退化为定长切
        segs = re.split(r"(?<=[\u3002\uff01\uff1f!?.\n])", text)
        segs = [s for s in segs if s]
        if len(segs) <= 1:
            return [text[i : i + size] for i in range(0, len(text), size)] or [text]
        return segs


def json_args(tc: ToolCall) -> str:
    import json

    return json.dumps(tc.arguments, ensure_ascii=False)


class TextFakeLLM(BaseLLM):
    """始终返回文本（不触发工具调用），用于记忆压缩、反思、规划等只需文本的场景测试。"""

    def __init__(self, model_name: str = "fake-text-llm", reply: str = "（模拟摘要）这是一段压缩后的历史摘要。") -> None:
        self.model_name = model_name
        self.reply = reply

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None) -> LLMResponse:  # noqa: ARG002
        last = messages[-1].content if messages else ""
        # 反思场景：返回带重要性的事实行，便于测试 write_via_reflection
        if "salient" in last or "长期记住" in last:
            content = "0.9|用户偏好用中文回答\n0.85|PaperMate 是论文研读 Agent\nNONE"
        else:
            content = self.reply
        tok = _rough_tokens(content)
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content=content),
            prompt_tokens=10, completion_tokens=tok, model=self.model_name, latency_ms=1,
        )
