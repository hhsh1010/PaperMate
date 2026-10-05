"""LightCore · 短期记忆（docs/05 §记忆系统）

滚动窗口：保留最近 N 条消息（默认 20）。超出时，把最旧的若干条交给 LLM 压缩为一条摘要，
维持"窗口内明细 + 一条历史摘要"的结构，控制 token 预算（NFR: max_tokens_budget 配合）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import BaseLLM, ChatMessage, Role

# C7-lightcore（2026-09-23）：提示词已集中到 `lightcore/prompts`，此处**只导入不再定义**。
# 保留本模块级名字是为了不破坏既有引用方（`from ..memory.short_term import SUMMARY_PROMPT`）。
from ..prompts import SUMMARY_PROMPT


@dataclass
class ShortTermMemory:
    window_size: int = 20
    _summary: str | None = None
    _messages: list[ChatMessage] = field(default_factory=list)

    def add(self, message: ChatMessage) -> None:
        # 仅追加；超出窗口的压缩发生在 compress()（需 LLM，异步），由 manager 周期性调用。
        self._messages.append(message)

    async def compress(self, llm: BaseLLM) -> None:
        """把当前窗口外的历史（已由 add 裁剪前的旧消息）压缩——这里压缩 _messages 最早的批次。"""
        if len(self._messages) <= self.window_size:
            return
        batch = self._messages[: len(self._messages) - self.window_size]
        text = "\n".join(f"{m.role.value}: {m.content}" for m in batch)
        resp = await llm.chat([ChatMessage(role=Role.USER, content=f"{SUMMARY_PROMPT}\n\n{text}")])
        self._summary = (self._summary + "\n" + resp.message.content) if self._summary else resp.message.content
        self._messages = self._messages[len(self._messages) - self.window_size:]

    def as_messages(self) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        if self._summary:
            out.append(ChatMessage(role=Role.SYSTEM, content=f"[历史摘要]\n{self._summary}"))
        out.extend(self._messages)
        return out

    def __len__(self) -> int:
        return len(self._messages)
