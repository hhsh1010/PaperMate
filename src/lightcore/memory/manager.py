"""LightCore · 记忆管理器（docs/05 §记忆系统）

协调短期（滚动窗口）+ 长期（向量）记忆：
- add(msg)              ：写入短期；对话轮次结束时由 reflect() 触发长期反思写入。
- get_context(query)    ：返回给 LLM 的上下文 = 短期窗口消息 + 召回的长期记忆（作为 system 注记）。
- recall_long(query, k) ：显式召回长期记忆（供 retrieve/引用增强）。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..llm.base import BaseLLM, ChatMessage, Role
from ..rag.embedder import Embedder
from .long_term import LongTermMemory
from .short_term import ShortTermMemory


class MemoryManager:
    """短期（滚动窗口）+ 长期（向量）记忆协调器。

    2026-08-29 C 档：反思写入加触发条件降本——此前每轮对话都打一次 LLM，
    而多数轮次并无值得长期记住的偏好（模型常返回 NONE），白白付出成本与延迟。
    策略（reflect_policy）：
    - "auto"（默认）：命中偏好信号词 → 立即反思；否则每 every_n_turns 轮兜底一次；
      对话文本过短（< min_chars）一律跳过。every_n_turns 默认 5（2026-08-29 优化：
      由 3 放宽到 5，进一步压低自动反思频率，减少每轮对话产生的记忆条数）。
    - "always"：每轮都反思（旧行为，便于对比/回归）。
    - "off"：完全不反思写入（召回不受影响）。
    """

    #: 出现这些词说明本轮可能含用户偏好/长期结论，值得立即反思
    REFLECT_SIGNALS: tuple[str, ...] = (
        "偏好", "喜欢", "不喜欢", "习惯", "必须", "不要", "禁止", "记住", "以后",
        "我一直", "我通常", "我是", "我的方向", "默认",
        "always", "never", "prefer", "must", "from now on", "remember",
    )

    def __init__(self, embedder: Embedder, window_size: int = 20, store=None,
                 reflect_policy: str = "auto", every_n_turns: int = 5,
                 min_chars: int = 40) -> None:
        self.short = ShortTermMemory(window_size=window_size)
        self.long = LongTermMemory(embedder, store=store)
        self.reflect_policy = reflect_policy
        self.every_n_turns = max(1, int(every_n_turns))
        self.min_chars = max(0, int(min_chars))
        self._turn_counts: dict[str, int] = {}

    def add(self, message: ChatMessage) -> None:
        self.short.add(message)

    async def compress_short(self, llm: BaseLLM) -> None:
        await self.short.compress(llm)

    def should_reflect(self, conversation: Sequence[ChatMessage], session_key: str = "") -> bool:
        """判断本轮是否需要触发反思（不产生 LLM 调用）。"""
        if self.reflect_policy == "off":
            return False
        text = "\n".join(m.text_only() for m in conversation if m.content).strip()
        if self.reflect_policy == "always":
            return True
        # 命中偏好信号词 → 立即反思（不看长度与轮次）
        if any(sig in text for sig in self.REFLECT_SIGNALS):
            return True
        # 无信号：内容过短不值得反思；否则每 every_n_turns 轮兜底一次
        if len(text) < self.min_chars:
            return False
        n = self._turn_counts.get(session_key, 0) + 1
        return n % self.every_n_turns == 0

    async def reflect(self, llm: BaseLLM, conversation: Sequence[ChatMessage] | None = None,
                      session_key: str = "",
                      source_session_id: int | None = None) -> bool:
        """对话轮次结束：满足条件时从短期窗口内容反思写入长期记忆。

        返回是否实际执行了反思（便于测试与埋点）。

        2026-09-07：source_session_id 改为**显式入参**——MemoryManager 是进程级共享单例，
        调用方若先写 `manager.source_session_id = sid` 再 await reflect，两个会话并发反思时
        会相互覆盖，记忆被错误归属到另一个会话（配合 scope=session 分层会放大事实串台）。
        显式传参消除该竞态；省略时仍回退到实例属性（向后兼容既有调用方）。
        """
        conv = list(conversation) if conversation is not None else self.short.as_messages()
        if len(conv) < 2:
            return False
        # 先判定再计数：should_reflect 用「本轮序号 = 已计数 + 1」，避免重复计数导致错开一轮
        should = self.should_reflect(conv, session_key)
        self._turn_counts[session_key] = self._turn_counts.get(session_key, 0) + 1
        if not should:
            return False
        sid = source_session_id if source_session_id is not None else self.source_session_id
        await self.long.write_via_reflection(llm, conv, source_session_id=sid)
        return True

    #: 反思归属会话 id 的**兜底默认值**；推荐由 reflect(source_session_id=...) 显式传入。
    #: （2026-09-07：改为入参以消除并发下的归属串台；此属性保留仅为向后兼容）
    source_session_id: int | None = None

    def get_context(self, query: str, k: int = 5) -> list[ChatMessage]:
        msgs = self.short.as_messages()
        recalled = self.long.recall(query, k=k)
        if recalled:
            note = "【长期记忆】\n" + "\n".join(f"- {it.text}" for it in recalled)
            msgs = [ChatMessage(role=Role.SYSTEM, content=note)] + msgs
        return msgs

    def recall_long(self, query: str, k: int = 5, min_rel: float = 0.0,
                   min_score_ratio: float = 0.0, session_id: int | None = None):
        """显式召回长期记忆；阈值与会话作用域参数透传给 LongTermMemory.recall（见其文档）。

        传入 session_id 时启用分层 scope 过滤：global 全量 + session 仅来源会话。
        """
        return self.long.recall(query, k=k, min_rel=min_rel,
                                min_score_ratio=min_score_ratio, session_id=session_id)
