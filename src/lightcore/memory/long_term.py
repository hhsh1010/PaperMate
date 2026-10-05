"""LightCore · 长期记忆（docs/05 §记忆系统 / Generative Agents 反思思想）

- 存储 salient 事实（`MemoryItem`：text + embedding + importance + created_at + last_accessed）。
- 召回打分（docs/05 口径）：score = 0.6·相关 + 0.3·重要 + 0.1·时效
    - 相关：query 与 item 的余弦相似度（经 Embedder）
    - 重要：item.importance（0~1，反思写入时由 LLM 评估）
    - 时效：越新分越高，recency = exp(-age_hours / TAU)
- 反思写入（write_via_reflection）：给 LLM 一段对话，提炼出若干 salient 事实并评估重要性后入库。
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..llm.base import BaseLLM, ChatMessage, Role
from ..rag.embedder import Embedder, cosine
from .poison import POISON_PATTERNS, is_poisoned

logger = logging.getLogger(__name__)

RECENCY_TAU_HOURS = 24.0  # 时效半衰期

# ---- 作用域（2026-08-29 分层 scope）----
# global ：用户偏好/身份/规则类事实，跨会话注入（跨会话记忆的核心价值）；
# session：会话事实（某篇论文的实验结果等），仅在其来源会话召回，避免跨会话串台。
SCOPE_GLOBAL = "global"
SCOPE_SESSION = "session"
_VALID_SCOPES = (SCOPE_GLOBAL, SCOPE_SESSION)


@dataclass
class MemoryItem:
    id: str
    text: str
    embedding: list[float]
    importance: float = 0.5  # 0~1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    source_session_id: int | None = None  # 来源会话（2026-08-29 C 档：面板可溯源）
    scope: str = SCOPE_GLOBAL  # 作用域（旧数据缺省 → global，行为向后兼容）

    def recency(self, now: float | None = None) -> float:
        """时效分（0~1，越新越高）。

        2026-08-29 B 档：基准由 created_at 改为 last_accessed——「最近被召回/写入过」的
        记忆才算新，避免长期有用但创建较早的记忆时效分永久归零；新建未召回时
        last_accessed == created_at，行为自然退化到旧语义。
        """
        now = now or time.time()
        age_h = (now - self.last_accessed) / 3600.0
        return math.exp(-age_h / RECENCY_TAU_HOURS)


class LongTermMemory:
    def __init__(self, embedder: Embedder, store=None) -> None:
        self._embedder = embedder
        self._store = store
        self._items: dict[str, MemoryItem] = {}
        # O-09（2026-10-01）：_norm 归一化结果随条目缓存。此前 add() 写入路径的
        # _find_duplicate 对全部已有条目逐条重算 _norm(it.text)（纯函数、结果只随 text 变）
        # —— 写入路径每轮 O(n) 次正则替换。缓存随条目生命周期维护（见 _norm_cached /
        # _set_item_text / remove / clear），文本变更处必须同步刷新。
        self._norms: dict[str, str] = {}
        # 最近访问时间落盘节流：默认 60s 一次（B 档）
        self._ACCESS_FLUSH_SECONDS = 60.0
        self._last_access_flush = 0.0
        if store is not None:
            for it in store.load():
                self._items[it.id] = it

    def ensure_embeddings(self) -> int:
        """若现有向量的维度与当前 embedder 不一致（切换了 embedder），重算全部向量。

        返回重算条数。切换 hash(64) ↔ BGE-M3(1024) 后必须调用，否则余弦比较维度不匹配。
        """
        if not self._items:
            return 0
        probe = self._embedder.encode(["维度探测"])[0]
        sample = next(iter(self._items.values()))
        if len(sample.embedding) == len(probe):
            return 0
        for it in self._items.values():
            it.embedding = self._embedder.encode([it.text])[0]
        self._persist()
        return len(self._items)

    def _persist(self) -> None:
        if self._store is not None:
            self._store.save(list(self._items.values()))

    #: 合并阈值（2026-08-29 A 档去重 → 本轮优化扩展为「合并带」）：
    #: 余弦相似度 ≥ 该值视为同一/近义事实，刷新（或归并）而非新增。
    #: 背景：反思每轮都跑，同一偏好被反复写入，top-k 召回被同质条目占满；
    #: 阈值由 0.92 放宽到 0.8——0.8~0.92 的近义表述（如「中文回答」vs「用中文作答」）
    #: 也归并，不再各自成条，从根上压住概念冗余导致的条数膨胀。
    MERGE_THRESHOLD = 0.8

    #: 写入门槛（2026-08-29 优化·闸A）：反思自动提炼的事实，重要度 ≤ 该值直接丢弃，不入库。
    #: 只拦「自动反思」路径（write_via_reflection）；显式写记忆工具（domain._memory_write）
    #: 走 add() 不经此门槛——Agent 主动保存的事实由它自定重要性，不被自动门槛误伤。
    WRITE_GATE_IMPORTANCE = 0.8

    #: 作用域判定启发词（2026-08-29 分层 scope）：命中 → 用户偏好/身份/规则（global），
    #: 否则视为会话事实（session）。规则判定零成本且行为确定；面板可手动调整归属兜底。
    GLOBAL_HINTS: tuple[str, ...] = (
        "偏好", "喜欢", "不喜欢", "习惯", "必须", "禁止", "不要", "记住", "以后", "默认",
        "我是", "我叫", "我的方向", "研究方向", "身份", "一律", "所有回答", "每次回答",
        "always", "never", "prefer", "must", "remember", "from now on",
    )

    @classmethod
    def classify_scope(cls, text: str) -> str:
        """按启发词判定事实作用域：偏好/身份类 → global，其余 → session。"""
        t = (text or "").lower()
        return SCOPE_GLOBAL if any(h in t for h in cls.GLOBAL_HINTS) else SCOPE_SESSION

    def _norm(self, text: str) -> str:
        """归一化用于精确去重：去空白与常见标点、统一大小写。"""
        return re.sub(r"[\s，。！？；、,.!?;:：\"'`~·]+", "", (text or "")).lower()

    def _norm_cached(self, it: MemoryItem) -> str:
        """条目文本的归一化结果（带缓存，O-09）。文本经 _set_item_text 变更时同步刷新。"""
        n = self._norms.get(it.id)
        if n is None:
            n = self._norm(it.text)
            self._norms[it.id] = n
        return n

    def _set_item_text(self, it: MemoryItem, text: str) -> None:
        """变更条目文本的唯一入口：同步刷新归一化缓存（漏刷 = 去重口径与文本脱节）。"""
        it.text = text
        self._norms[it.id] = self._norm(text)

    def _find_duplicate(self, text: str, vec: list[float]) -> MemoryItem | None:
        """找同义条目：先精确归一化匹配，再向量近邻匹配（取最相似的一条）。"""
        key = self._norm(text)
        if not key:
            return None
        best: MemoryItem | None = None
        best_rel = 0.0
        for it in self._items.values():
            if self._norm_cached(it) == key:
                return it
            rel = max(0.0, cosine(vec, it.embedding))
            if rel >= self.MERGE_THRESHOLD and rel > best_rel:
                best, best_rel = it, rel
        return best

    def add(self, text: str, importance: float = 0.5,
            source_session_id: int | None = None, scope: str | None = None) -> MemoryItem:
        """写入一条记忆；命中同义条目时刷新（重要性取高、保留更完整的表述），不新增。

        scope 缺省时按启发词自动分类（classify_scope）；session 作用域但无来源会话 id
        的条目会在召回时永远无法命中，故强制回落 global（安全兜底）。
        """
        text = (text or "").strip()
        # 2026-09-09 修复 I5：与同文件 `update()` 的 `if not text: return None` 口径保持一致。
        # 空文本入库后会被 `recall` 注入 SYSTEM 注记——既无信息量又占位；
        # 且 `_find_duplicate` 对空 key 直接 return None（不去重也不拒绝），会持续累积空条目。
        if not text:
            logger.warning("拒绝写入空记忆文本（add）")
            raise ValueError("记忆文本不能为空")
        vec = self._embedder.encode([text])[0]
        if scope not in _VALID_SCOPES:
            scope = self.classify_scope(text)
        if scope == SCOPE_SESSION and source_session_id is None:
            scope = SCOPE_GLOBAL
        dup = self._find_duplicate(text, vec)
        if dup is not None:
            dup.importance = max(dup.importance, importance)
            dup.last_accessed = time.time()
            if source_session_id is not None and dup.source_session_id is None:
                dup.source_session_id = source_session_id
            if scope == SCOPE_GLOBAL and dup.scope == SCOPE_SESSION:
                dup.scope = SCOPE_GLOBAL  # 同一事实被确认为偏好 → 提升为全局
            if len(text) > len(dup.text):  # 保留信息量更大的表述
                self._set_item_text(dup, text)
                dup.embedding = vec
            self._persist()
            return dup
        item = MemoryItem(id=uuid.uuid4().hex[:12], text=text, embedding=vec, importance=importance,
                          source_session_id=source_session_id, scope=scope)
        self._items[item.id] = item
        self._persist()
        return item

    def all_items(self) -> list[MemoryItem]:
        """全部记忆（按创建时间倒序），供管理面板列表展示。"""
        return sorted(self._items.values(), key=lambda it: it.created_at, reverse=True)

    def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    def update(self, item_id: str, text: str, importance: float | None = None,
               scope: str | None = None) -> MemoryItem | None:
        """编辑记忆文本：必须重算 embedding，否则召回仍按旧文本的向量匹配（语义错位）。

        importance / scope 缺省时保留原值（原实现用关键词启发式覆盖 LLM 评估结果，会丢信息）；
        显式传入才更新。scope 改为 session 但条目无来源会话时回落 global（避免不可召回的死条目）。
        """
        it = self._items.get(item_id)
        if it is None:
            return None
        text = (text or "").strip()
        if not text:
            return None
        self._set_item_text(it, text)
        it.embedding = self._embedder.encode([text])[0]
        if importance is not None:
            it.importance = max(0.0, min(1.0, float(importance)))
        if scope in _VALID_SCOPES and scope != it.scope:
            it.scope = scope
            if it.scope == SCOPE_SESSION and it.source_session_id is None:
                it.scope = SCOPE_GLOBAL
        it.last_accessed = time.time()
        self._persist()
        return it

    def remove(self, item_id: str) -> bool:
        """删除一条记忆；返回是否命中。"""
        if item_id not in self._items:
            return False
        del self._items[item_id]
        self._norms.pop(item_id, None)  # 缓存随条目销毁
        self._persist()
        return True

    def clear(self) -> int:
        """清空全部记忆；返回被清除条数。"""
        n = len(self._items)
        self._items.clear()
        self._norms.clear()  # 缓存随条目销毁
        self._persist()
        return n

    def recall(self, query: str, k: int = 5, alpha_rel: float = 0.6, alpha_imp: float = 0.3,
               alpha_rec: float = 0.1, min_rel: float = 0.0,
               min_score_ratio: float = 0.0, session_id: int | None = None) -> list[MemoryItem]:
        """按 score = 0.6·相关 + 0.3·重要 + 0.1·时效 召回。

        2026-08-29 A 档新增两级过滤（默认关闭，行为向后兼容）：
        - min_rel：绝对相关度下限（与 embedder 强相关，谨慎使用）；
        - min_score_ratio：相对下限，丢弃得分低于「最高分 × 该比例」的条目。
          相对阈值与 embedder 无关，用于挡掉「完全不沾边但仍被 top-k 捞进来」的记忆。
        2026-08-29 分层 scope：传入 session_id 时按作用域过滤——
        global 条目全量参与 + session 条目仅其来源会话参与；session_id=None 不过滤（旧语义）。
        """
        if not self._items:
            return []
        qv = self._embedder.encode([query])[0]
        now = time.time()
        scored: list[tuple[float, float, MemoryItem]] = []
        for it in self._items.values():
            if (session_id is not None and it.scope == SCOPE_SESSION
                    and it.source_session_id != session_id):
                continue  # 其它会话的事实不跨会话注入
            rel = max(0.0, cosine(qv, it.embedding))  # 相关（0~1）
            score = alpha_rel * rel + alpha_imp * it.importance + alpha_rec * it.recency(now)
            scored.append((score, rel, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        if min_rel > 0:
            scored = [s for s in scored if s[1] >= min_rel]
        if min_score_ratio > 0 and scored:
            floor_score = scored[0][0] * min_score_ratio
            scored = [s for s in scored if s[0] >= floor_score]
        for _, _, it in scored[:k]:
            it.last_accessed = now
        # 2026-08-29 B 档：把「最近访问时间」落盘（此前只改内存，重启即丢）。
        # 节流：距上次刷盘不足 _ACCESS_FLUSH_SECONDS 则跳过，避免每轮对话都全量重写 JSON。
        self._maybe_flush_access(now)
        return [it for _, _, it in scored[:k]]

    def _maybe_flush_access(self, now: float) -> None:
        if self._store is None:
            return
        if now - self._last_access_flush < self._ACCESS_FLUSH_SECONDS:
            return
        self._last_access_flush = now
        self._persist()

    # 毒化模式已收敛到 lightcore.memory.poison（唯一来源），此处保留兼容入口：
    # 2026-08-29 C 档前本类自带 _WRITE_BLOCKLIST，与 ChatService 那份并存且已漂移。
    _WRITE_BLOCKLIST = POISON_PATTERNS

    @classmethod
    def _is_poisoned(cls, text: str) -> bool:
        return is_poisoned(text)

    async def write_via_reflection(self, llm: BaseLLM, conversation: Sequence[ChatMessage], k: int = 2,
                                   source_session_id: int | None = None) -> list[MemoryItem]:
        """从一段对话中提炼 salient 事实并写入（Generative Agents 反思思想）。

        拦截『状态观察型/拒接型』结论：这些是某次拒接/守门输出，不是用户偏好，
        写入会被 recall 反复注入 SYSTEM 注记，让 LLM 把『无文档』『静默跳过』当成
        硬规则，跳过 retrieve 真实检索（见 2026-08-19（三续））。

        2026-08-29 优化·闸A（压住『每轮对话大量记忆』）：
        - 单次提炼上限 k 由 3 收到 2，减少每次反思的落盘条数；
        - 重要度 ≤ WRITE_GATE_IMPORTANCE(0.8) 的闲聊/临时结论直接丢弃，不入库；
        - 无来源会话的「会话级事实」（非偏好/身份）是临时表述，若强写会回落 global
          永久污染全局记忆，故直接丢弃（防泄漏）；显式写记忆工具不经此门槛。
        """
        text = "\n".join(f"{m.role.value}: {m.text_only()}" for m in conversation if m.content)
        prompt = (
            "请从下列对话中提炼最多 2 条**用户偏好或长期结论**——即『值得长期记住(salient)的事实』"
            "（用户身份/习惯/调研方向/方法论偏好等）。\n"
            "注意：**只保留用户层面的事实**，不写『当前系统中无文档』『向量库为空』之类的**临时状态观察**，"
            "这类陈述是状态而非偏好，下次就会变化，写入会误导未来的对话。\n"
            "每行一条，格式：<重要性0-1>|<事实>。例如：0.9|用户偏好用中文回答。\n"
            "若无重要事实，仅回复 NONE。"
        )
        # 2026-09-08 修复：反思是「尽力而为」的旁路，原先此处**无任何异常保护**——
        # LLM 调用失败（key 失效 / 网络异常）或 resp.message.content 为 None 都会抛异常，
        # 经 chat_service → routes_search 一路冒泡，在已 yield 消息之后冲垮整条 SSE 流，
        # 导致 auto_title 与 done 事件**永不发出**（前端卡在无终止状态，看不到结束）。
        try:
            resp = await llm.chat([ChatMessage(role=Role.USER, content=f"{prompt}\n\n{text}")])
            content = resp.message.content or ""
        except Exception as e:  # noqa: BLE001  反思失败绝不能影响主对话：降级为「本次不写入」
            logger.warning("长期记忆反思失败（跳过本次写入，不影响对话）：%s", e)
            return []
        created: list[MemoryItem] = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            if "|" not in line:
                continue
            imp_s, fact = line.split("|", 1)
            try:
                imp = max(0.0, min(1.0, float(imp_s.strip())))
            except ValueError:
                imp = 0.5
            fact = fact.strip()
            if not fact:
                continue
            if self._is_poisoned(fact):
                # 不写入：状态观察型/拒接型结论
                continue
            # 闸A：自动反思事实需重要度 > 门槛才入库；≤门槛的闲聊/临时结论丢弃
            if imp <= self.WRITE_GATE_IMPORTANCE:
                continue
            # 闸A 配套·防泄漏：无来源会话的会话级事实（非偏好）是临时表述，
            # 若强写会回落 global 永久污染全局记忆 → 直接丢弃
            if source_session_id is None and self.classify_scope(fact) == SCOPE_SESSION:
                continue
            created.append(self.add(fact, importance=imp, source_session_id=source_session_id))
        return created

    def __len__(self) -> int:
        return len(self._items)
