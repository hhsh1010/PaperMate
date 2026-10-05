"""LightCore · 记忆系统单元测试（docs/08 单元层 / M2）

新增：M2.1 跨会话记忆持久化（store=JSONFileMemoryStore）——验证记忆落盘后，
全新会话/进程加载同一文件即可召回，满足【跨会话偏好/结论共享】验收点。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import TextFakeLLM
from lightcore.memory.long_term import SCOPE_GLOBAL, SCOPE_SESSION, LongTermMemory
from lightcore.memory.manager import MemoryManager
from lightcore.memory.short_term import ShortTermMemory
from lightcore.memory.store import JSONFileMemoryStore
from lightcore.rag.embedder import FakeHashEmbedder, cosine


def _mgr(store_path: Path | None = None) -> MemoryManager:
    store = JSONFileMemoryStore(store_path) if store_path is not None else None
    return MemoryManager(FakeHashEmbedder(dim=32), store=store)


def test_short_term_window_and_summary():
    async def run():
        st = ShortTermMemory(window_size=3)
        for i in range(5):
            st.add(ChatMessage(role=Role.USER, content=f"msg-{i}"))
        # add 仅追加，未压缩前明细为 5 条
        assert len(st) == 5, len(st)
        # 压缩：调用 LLM 生成摘要，明细裁剪到窗口大小
        await st.compress(TextFakeLLM())
        assert st._summary is not None
        assert len(st) == 3  # 压缩后裁剪到窗口大小
        msgs = st.as_messages()
        # 第一条应为历史摘要（SYSTEM）
        assert msgs[0].role == Role.SYSTEM and "摘要" in msgs[0].content

    asyncio.run(run())


def test_long_term_recall_scoring():
    async def run():
        from lightcore.memory.long_term import LongTermMemory

        lt = LongTermMemory(FakeHashEmbedder(dim=32))
        lt.add("用户偏好用中文回答", importance=0.9)
        lt.add("今天天气晴", importance=0.1)
        lt.add("LightCore 是论文研读 Agent 内核", importance=0.7)
        recalled = lt.recall("用户偏好", k=2)
        assert recalled, "应召回至少 1 条"
        # 高重要性且相关项应排前
        assert "中文" in recalled[0].text

    asyncio.run(run())


def test_reflection_writes_facts():
    async def run():
        from lightcore.memory.long_term import LongTermMemory

        lt = LongTermMemory(FakeHashEmbedder(dim=32))
        conv = [
            ChatMessage(role=Role.USER, content="我喜欢用中文。"),
            ChatMessage(role=Role.ASSISTANT, content="好的，我会用中文。"),
        ]
        added = await lt.write_via_reflection(TextFakeLLM(), conv, source_session_id=7)
        assert len(added) == 2, f"应提炼 2 条事实，实际 {len(added)}"
        texts = [a.text for a in added]
        assert any("中文" in t for t in texts)

    asyncio.run(run())


def test_manager_context_merges_long_term():
    async def run():
        mgr = _mgr()
        for i in range(2):
            mgr.add(ChatMessage(role=Role.USER, content=f"q{i}"))
        mgr.long.add("用户偏好用中文回答", importance=0.9)
        ctx = mgr.get_context("用户偏好", k=3)
        # 含长期记忆 system 注记
        assert any("长期记忆" in m.content for m in ctx)
        # recall_long 直接可用
        rec = mgr.recall_long("用户偏好", k=1)
        assert rec and "中文" in rec[0].text

    asyncio.run(run())


# ---------------- M2.1 跨会话记忆持久化 ----------------
def test_reflect_persists_to_store(tmp_path: Path) -> None:
    async def run():
        mem_path = tmp_path / "memories.json"
        mgr = _mgr(mem_path)
        conv = [
            ChatMessage(role=Role.USER, content="我喜欢用中文回答。"),
            ChatMessage(role=Role.ASSISTANT, content="好的，我会用中文。"),
        ]
        await mgr.reflect(TextFakeLLM(), conversation=conv)
        # 记忆已落盘 JSON
        assert mem_path.exists(), "记忆应持久化到 memories.json"
        loaded = JSONFileMemoryStore(mem_path).load()
        assert len(loaded) >= 1
        assert any("中文" in it.text for it in loaded)

    asyncio.run(run())


def test_recall_across_sessions(tmp_path: Path) -> None:
    async def run():
        mem_path = tmp_path / "memories.json"
        # 会话 1：写入偏好并落盘
        mgr1 = _mgr(mem_path)
        await mgr1.reflect(TextFakeLLM(), conversation=[
            ChatMessage(role=Role.USER, content="我喜欢用中文回答。"),
            ChatMessage(role=Role.ASSISTANT, content="好的。"),
        ])
        # 会话 2：全新 MemoryManager 读同一文件（模拟跨会话/跨进程）
        mgr2 = _mgr(mem_path)
        recalled = mgr2.recall_long("用户希望用什么语言交流？", k=5)
        assert len(recalled) >= 1
        assert any("中文" in it.text for it in recalled)

    asyncio.run(run())


def test_reflection_skips_state_observations():
    """回归：write_via_reflection 必须拦截『状态观察型』毒化结论（见 2026-08-19 三续）。"""

    class RefuseLLM(TextFakeLLM):
        def __init__(self):
            super().__init__()
            self.reply = (
                "0.95|系统当前无任何可检索文档，所有文档相关操作均因前提不成立而无法执行\n"
                "0.90|重复发送「总结文档内容」等无效指令将被系统静默忽略\n"
                "0.85|用户偏好用中文回答\n"
                "0.70|PaperMate 是论文研读 Agent"
            )

    async def run():
        lt = LongTermMemory(FakeHashEmbedder(dim=32))
        added = await lt.write_via_reflection(
            RefuseLLM(),
            [ChatMessage(Role.USER, "总结文档内容"),
             ChatMessage(Role.ASSISTANT, "当前系统中无任何文档。")],
        )
        texts = [a.text for a in added]
        # 前两条是毒化拦截项（不写）；0.85 中文是合法偏好（写）；0.70 PaperMate 重要度≤门槛（不写）
        assert not any("无任何可检索" in t for t in texts), texts
        assert not any("静默忽略" in t or "无效指令" in t for t in texts), texts
        assert any("中文" in t for t in texts), texts
        assert not any("PaperMate" in t for t in texts), "重要度≤0.8 的事实应被写入门槛丢弃"

    asyncio.run(run())


def test_long_term_persist_and_reload_roundtrip(tmp_path: Path) -> None:
    async def run():
        from lightcore.memory.long_term import LongTermMemory

        mem_path = tmp_path / "memories.json"
        lt = LongTermMemory(FakeHashEmbedder(dim=32), store=JSONFileMemoryStore(mem_path))
        lt.add("用户偏好用中文回答", importance=0.9)
        lt.add("LightCore 是论文研读 Agent 内核", importance=0.7)
        # 全新实例重载
        lt2 = LongTermMemory(FakeHashEmbedder(dim=32), store=JSONFileMemoryStore(mem_path))
        assert len(lt2) == 2
        assert any("中文" in it.text for it in lt2.recall("用户偏好", k=2))

    asyncio.run(run())


# ---------------- 2026-08-29 A 档：去重 / 编辑重算向量 / 共享单例 ----------------

def test_add_dedupes_near_identical_facts():
    """反思每轮都跑：同义事实不应重复入库（命中则刷新重要性与更完整表述）。"""
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    a = lt.add("用户偏好用中文回答", importance=0.5)
    b = lt.add("用户偏好用中文回答。", importance=0.9)   # 归一化后完全相同
    c = lt.add("用户偏好使用中文作答", importance=0.7)    # 近义（需向量阈值命中）
    assert a.id == b.id, "归一化相同的文本必须命中去重"
    assert len(lt) <= 2, f"不应重复入库，当前 {len(lt)} 条"
    assert a.importance == 0.9, "重复写入应取更高的重要性"
    assert c.id in (a.id, c.id)


def test_add_keeps_distinct_facts():
    """去重不能误杀：语义无关的事实必须各自入库。"""
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    lt.add("用户偏好用中文回答", importance=0.9)
    lt.add("PaperMate 是论文研读 Agent 内核", importance=0.7)
    lt.add("实验在 COCO 数据集上评估", importance=0.6)
    assert len(lt) == 3, f"无关事实被误去重：{len(lt)} 条"


# ---------------- 2026-08-29 优化：闸A 写入门槛 + 0.8 合并带 ----------------

def test_reflection_write_gate_drops_low_importance():
    """闸A：反思提炼的事实重要度 ≤0.8 不入库，只留 >0.8 的偏好/结论。"""
    from lightcore.llm.base import LLMResponse

    class GateLLM(TextFakeLLM):
        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None) -> LLMResponse:
            return LLMResponse(message=ChatMessage(
                role=Role.ASSISTANT,
                content="0.9|用户偏好用中文回答\n0.6|今天午饭吃了牛肉面\n0.75|实验在 COCO 上跑通",
            ))

    async def run():
        lt = LongTermMemory(FakeHashEmbedder(dim=32))
        added = await lt.write_via_reflection(GateLLM(), [ChatMessage(Role.USER, "记住我用中文")])
        texts = [a.text for a in added]
        assert any("中文" in t for t in texts)
        assert not any("牛肉面" in t for t in texts), "闲聊(0.6)应被写入门槛丢弃"
        assert not any("COCO" in t for t in texts), "重要度 0.75≤0.8 应被写入门槛丢弃"
        assert len(lt) == 1, "仅 >0.8 的 1 条应入库"

    asyncio.run(run())


def test_reflection_skips_session_fact_without_source():
    """闸A 防泄漏：无来源会话的会话级事实（非偏好）即便 >0.8 也丢弃，避免回落 global 污染。"""
    from lightcore.llm.base import LLMResponse

    class LeakLLM(TextFakeLLM):
        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None) -> LLMResponse:
            return LLMResponse(message=ChatMessage(
                role=Role.ASSISTANT,
                content="0.9|用户偏好用中文回答\n0.9|本论文在 COCO 上 mAP 为 50.2",
            ))

    async def run():
        lt = LongTermMemory(FakeHashEmbedder(dim=32))
        added = await lt.write_via_reflection(
            LeakLLM(), [ChatMessage(Role.USER, "我们论文在 COCO 测了")], source_session_id=None)
        texts = [a.text for a in added]
        assert any("中文" in t for t in texts)
        assert not any("COCO" in t for t in texts), "无来源的会话事实应被丢弃，不得污染全局记忆"
        assert len(lt) == 1

    asyncio.run(run())


def test_merge_band_merges_near_synonyms():
    """合并带：余弦 ∈[0.8,0.92) 的近义表述归并而非新增（阈值由 0.92 放宽到 0.8）。"""
    from lightcore.rag.embedder import Embedder

    class MockEmb(Embedder):
        dim = 4

        def encode(self, texts):
            table = {
                "用户偏好用中文回答": [1.0, 0.0, 0.0, 0.0],
                "用户偏好使用中文作答": [0.85, 0.5268, 0.0, 0.0],  # 与上式余弦 0.85 ∈[0.8,0.92)
                "实验在 COCO 上评估": [0.0, 1.0, 0.0, 0.0],        # 与首条余弦 0
            }
            return [table.get(t, [0.0, 0.0, 0.0, 1.0]) for t in texts]

    lt = LongTermMemory(MockEmb())
    a = lt.add("用户偏好用中文回答", importance=0.9)
    b = lt.add("用户偏好使用中文作答", importance=0.85)  # 0.85 ≥0.8 → 归并
    c = lt.add("实验在 COCO 上评估", importance=0.85)    # 余弦 0 → 新增
    assert len(lt) == 2, f"近义应归并，当前 {len(lt)} 条"
    assert a.id == b.id, "近义表述必须归并到同一记忆"
    assert c.id != a.id


def test_update_recomputes_embedding():
    """编辑记忆必须重算 embedding，否则召回仍按旧文本匹配（语义错位缺陷回归）。

    注：FakeHashEmbedder 是字符位置哈希，个别短文本会向量碰撞（实测「…中文回答」与
    「…英文回答」余弦为 1.0），故这里用向量差异明确的一组文本来验证。
    """
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    it = lt.add("用户偏好用中文回答", importance=0.9)
    old_vec = list(it.embedding)
    new_text = "实验在 COCO 数据集上评估，mAP 为 50.2"
    updated = lt.update(it.id, new_text)
    assert updated is not None and updated.text == new_text
    assert updated.embedding != old_vec, "编辑后必须重算向量"
    assert cosine(updated.embedding, FakeHashEmbedder(dim=32).encode([new_text])[0]) > 0.999
    recalled = lt.recall("COCO 数据集 mAP", k=1)
    assert recalled and "COCO" in recalled[0].text


def test_update_preserves_importance_unless_explicit():
    """importance 缺省时保留 LLM 评估值，不被引擎启发式覆盖。"""
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    it = lt.add("用户偏好用中文回答", importance=0.9)
    assert lt.update(it.id, "用户偏好中文交流").importance == 0.9
    assert lt.update(it.id, "用户偏好中文交流", importance=0.4).importance == 0.4


def test_remove_and_clear():
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    a = lt.add("苹果是一种常见的水果", importance=0.5)
    lt.add("量子计算基于量子叠加态", importance=0.5)
    assert lt.remove(a.id) is True
    assert lt.remove("not-exist") is False
    assert len(lt) == 1
    assert lt.clear() == 1
    assert len(lt) == 0


def test_recall_min_score_ratio_drops_irrelevant():
    """相对阈值：挡掉得分远低于首位的记忆（与 embedder 分布无关）。"""
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    lt.add("用户偏好用中文回答", importance=0.9)
    lt.add("实验在 COCO 数据集上评估，mAP 为 50.2", importance=0.1)
    lt.add("论文第三章介绍了网络结构", importance=0.1)
    full = lt.recall("用户偏好", k=5)
    filtered = lt.recall("用户偏好", k=5, min_score_ratio=0.9)
    assert len(filtered) <= len(full)
    assert filtered and "中文" in filtered[0].text


# ---------------- 2026-08-29 B 档：时效基准 / 访问落盘 / 向量引擎可切换 ----------------

def test_recency_uses_last_accessed(tmp_path: Path):
    """时效分以 last_accessed 为基准：被召回过的老记忆不再因 created_at 归零。"""
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    old = lt.add("用户偏好用中文回答", importance=0.5)
    old.created_at -= 72 * 3600   # 创建于 3 天前
    old.last_accessed -= 72 * 3600
    assert old.recency() < 0.1, "未访问的老记忆时效分应接近 0"
    lt.recall("用户偏好", k=1)      # 召回 → 刷新 last_accessed
    assert old.recency() > 0.9, "被召回后时效分应回到接近 1"


def test_last_accessed_persisted_with_throttle(tmp_path: Path):
    """最近访问时间要落盘（此前只改内存，重启即丢），且带节流避免每轮全量重写。"""
    mem_path = tmp_path / "memories.json"
    lt = LongTermMemory(FakeHashEmbedder(dim=32), store=JSONFileMemoryStore(mem_path))
    it = lt.add("用户偏好用中文回答", importance=0.8)
    time.sleep(0.01)  # 保证时间戳可区分（Windows 时钟精度较粗，同 tick 内会相等）
    lt.recall("用户偏好", k=1)
    reloaded = JSONFileMemoryStore(mem_path).load()
    assert len(reloaded) == 1
    assert reloaded[0].last_accessed > reloaded[0].created_at, "访问时间应已落盘"
    # 节流窗口内不重复写盘
    before = mem_path.stat().st_mtime_ns
    it.last_accessed = time.time()
    lt._maybe_flush_access(time.time())
    assert mem_path.stat().st_mtime_ns == before, "节流窗口内不应重复写盘"


def test_ensure_embeddings_rebuilds_on_embedder_switch(tmp_path: Path):
    """切换 embedder（维度变化）后必须重算向量，否则余弦比较维度不匹配。"""
    mem_path = tmp_path / "memories.json"
    lt = LongTermMemory(FakeHashEmbedder(dim=32), store=JSONFileMemoryStore(mem_path))
    lt.add("用户偏好用中文回答", importance=0.8)
    assert len(next(iter(lt._items.values())).embedding) == 32
    # 模拟切换到 128 维引擎
    lt._embedder = FakeHashEmbedder(dim=128)
    assert lt.ensure_embeddings() == 1
    assert len(next(iter(lt._items.values())).embedding) == 128
    # 维度一致时不重算
    assert lt.ensure_embeddings() == 0
    # 重载后仍是新维度
    lt2 = LongTermMemory(FakeHashEmbedder(dim=128), store=JSONFileMemoryStore(mem_path))
    assert len(next(iter(lt2._items.values())).embedding) == 128


# ---------------- 2026-08-29 C 档：毒化黑名单唯一来源 / 反思降本 / 来源溯源 ----------------

def test_poison_patterns_single_source():
    """C 档：三处使用方必须引用同一份毒化模式，不再各自维护拷贝。"""
    from lightcore.memory.long_term import LongTermMemory
    from lightcore.memory.poison import POISON_PATTERNS, is_poisoned
    from papermate.services.chat_service import ChatService

    assert LongTermMemory._WRITE_BLOCKLIST == POISON_PATTERNS
    assert ChatService._POISON_PATTERNS == POISON_PATTERNS
    # 行为一致：写入侧与召回侧对同一文本的判定必须相同
    for text in ("系统中无任何文档", "对重复指令静默跳过", "用户偏好用中文回答"):
        assert LongTermMemory._is_poisoned(text) == is_poisoned(text) == ChatService._is_poisoned(text)
    assert is_poisoned("我无法回答这个问题")
    assert not is_poisoned("用户偏好中文回答，引用必须标页码")


def test_reflect_skips_when_no_signal_and_not_nth_turn():
    """C 档降本：无偏好信号且未到兜底轮次 → 不发起 LLM 调用。"""
    class CountingLLM(TextFakeLLM):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def chat(self, messages, tools=None, temperature=0.7, **kw):
            self.calls += 1
            return await super().chat(messages, tools=tools, temperature=temperature, **kw)

    async def run():
        llm = CountingLLM()
        mgr = _mgr()
        # 无偏好信号词、但长度足够（>min_chars）的日常问答
        conv = [ChatMessage(Role.USER, "请帮我看看这篇论文第三章节讲了什么内容？"),
                ChatMessage(Role.ASSISTANT, "第三章主要介绍网络结构与训练细节，共两小节。")]
        # every_n_turns 默认 5：前 4 轮兜底都不触发，第 5 轮才触发
        for n in range(1, 5):
            await mgr.reflect(llm, conversation=conv, session_key="s1")
            assert llm.calls == 0, f"无信号词且未到第 5 轮（第 {n} 轮），不应调用 LLM"
        await mgr.reflect(llm, conversation=conv, session_key="s1")   # 第 5 轮 → 兜底触发
        assert llm.calls == 1
        # 其它会话的轮次互不影响
        await mgr.reflect(llm, conversation=conv, session_key="s2")
        assert llm.calls == 1

    asyncio.run(run())


def test_reflect_runs_immediately_on_preference_signal():
    """C 档：命中偏好信号词 → 立即反思（不看轮次）。"""
    async def run():
        mgr = _mgr()
        conv = [
            ChatMessage(Role.USER, "我偏好所有回答都用中文，记住这一点。"),
            ChatMessage(Role.ASSISTANT, "好的，之后一律用中文回答。"),
        ]
        assert await mgr.reflect(TextFakeLLM(), conversation=conv, session_key="s3") is True
        assert len(mgr.long) >= 1

    asyncio.run(run())


def test_reflect_skips_short_conversation_and_off_policy():
    """C 档：过短对话跳过；policy=off 时完全不反思。"""
    async def run():
        mgr = _mgr()
        short = [ChatMessage(Role.USER, "好"), ChatMessage(Role.ASSISTANT, "嗯")]
        assert await mgr.reflect(TextFakeLLM(), conversation=short, session_key="s4") is False
        mgr.reflect_policy = "off"
        conv = [
            ChatMessage(Role.USER, "我偏好用中文回答，请记住。"),
            ChatMessage(Role.ASSISTANT, "好的。"),
        ]
        assert await mgr.reflect(TextFakeLLM(), conversation=conv, session_key="s5") is False
        assert len(mgr.long) == 0
        mgr.reflect_policy = "always"
        assert await mgr.reflect(TextFakeLLM(), conversation=short, session_key="s5") is True

    asyncio.run(run())


def test_memory_source_session_persisted(tmp_path: Path):
    """C 档：反思写入的记忆带来源会话，落盘后可溯源。"""
    async def run():
        mem_path = tmp_path / "memories.json"
        mgr = MemoryManager(FakeHashEmbedder(dim=32), store=JSONFileMemoryStore(mem_path))
        mgr.reflect_policy = "always"
        mgr.source_session_id = 42
        await mgr.reflect(TextFakeLLM(), conversation=[
            ChatMessage(Role.USER, "我偏好用中文回答。"),
            ChatMessage(Role.ASSISTANT, "好的。"),
        ], session_key="42")
        loaded = JSONFileMemoryStore(mem_path).load()
        assert loaded, "应写入记忆"
        assert all(it.source_session_id == 42 for it in loaded)

    asyncio.run(run())


# ---------------- 2026-08-29 分层 scope：分类 / 召回隔离 / 旧数据兼容 ----------------

def test_classify_scope_rule():
    """启发词命中 → 用户偏好/身份（global）；否则会话事实（session）。"""
    assert LongTermMemory.classify_scope("用户偏好用中文回答") == SCOPE_GLOBAL
    assert LongTermMemory.classify_scope("我是一名博士研究生，方向是多模态") == SCOPE_GLOBAL
    assert LongTermMemory.classify_scope("必须始终标注引用来源") == SCOPE_GLOBAL
    assert LongTermMemory.classify_scope("实验在 COCO 数据集上评估 mAP 为 50.2") == SCOPE_SESSION
    assert LongTermMemory.classify_scope("论文第三章介绍了网络结构") == SCOPE_SESSION


def test_add_session_scope_requires_source_id():
    """session 作用域但无来源会话 → 回落 global（否则永远无法召回，成死条目）。"""
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    it = lt.add("实验在 COCO 上评估 mAP 50.2", importance=0.5, scope=SCOPE_SESSION,
                source_session_id=None)
    assert it.scope == SCOPE_GLOBAL
    it2 = lt.add("用户偏好中文回答", importance=0.5, scope=SCOPE_SESSION, source_session_id=7)
    assert it2.scope == SCOPE_SESSION  # 有来源会话 → 保留 session


def test_recall_respects_session_scope():
    """全局偏好跨会话注入；会话事实仅来源会话可见；session_id=None 不过滤（旧语义）。"""
    lt = LongTermMemory(FakeHashEmbedder(dim=32))
    g = lt.add("用户偏好用中文回答", importance=0.9, scope=SCOPE_GLOBAL)
    a = lt.add("A 论文实验在 COCO 上评估 mAP 50.2", importance=0.6, source_session_id=7, scope=SCOPE_SESSION)
    b = lt.add("B 论文在 ImageNet 上评测 top1 88.1", importance=0.6, source_session_id=9, scope=SCOPE_SESSION)

    in_session7 = {it.id for it in lt.recall("评估指标", k=10, session_id=7)}
    assert g.id in in_session7 and a.id in in_session7 and b.id not in in_session7

    in_session9 = {it.id for it in lt.recall("评估指标", k=10, session_id=9)}
    assert g.id in in_session9 and b.id in in_session9 and a.id not in in_session9

    # session_id=None：不启用作用域过滤，旧语义（全部候选参与）
    no_filter = {it.id for it in lt.recall("评估指标", k=10, session_id=None)}
    assert {g.id, a.id, b.id} <= no_filter


def test_legacy_store_defaults_global_and_roundtrip(tmp_path: Path):
    """旧数据缺 scope → global；合法 scope 持久化往返保持。"""
    import json

    mem_path = tmp_path / "memories.json"
    lt = LongTermMemory(FakeHashEmbedder(dim=32), store=JSONFileMemoryStore(mem_path))
    lt.add("用户偏好中文回答", importance=0.9)  # 缺省 → global
    lt.add("A 论文实验在 COCO 上评估", importance=0.6, source_session_id=7, scope=SCOPE_SESSION)
    loaded = JSONFileMemoryStore(mem_path).load()
    by_text = {it.text: it for it in loaded}
    assert by_text["用户偏好中文回答"].scope == SCOPE_GLOBAL
    assert by_text["A 论文实验在 COCO 上评估"].scope == SCOPE_SESSION

    # 模拟旧数据（无 scope 键）落盘，重载后回落 global
    raw = [{
        "id": "x1", "text": "旧记忆无 scope 字段", "embedding": [0.1] * 32,
        "importance": 0.5, "created_at": 1.0, "last_accessed": 1.0, "source_session_id": None,
    }]
    mem_path.write_text(json.dumps(raw), encoding="utf-8")
    legacy = JSONFileMemoryStore(mem_path).load()
    assert legacy[0].scope == SCOPE_GLOBAL


def test_reflection_writes_session_by_default_for_facts():
    """C 档反思写道：无偏好信号的会话事实 → 默认 session 作用域（按来源会话隔离）。"""
    from lightcore.llm.base import LLMResponse

    async def run():
        class StubLLM(TextFakeLLM):
            async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None) -> LLMResponse:
                return LLMResponse(message=ChatMessage(
                    role=Role.ASSISTANT,
                    content="0.9|用户偏好用中文回答\n0.85|实验在 COCO 上评估 mAP 50.2",
                ))

        mgr = _mgr()
        mgr.reflect_policy = "always"
        mgr.source_session_id = 7
        await mgr.reflect(StubLLM(), conversation=[
            ChatMessage(Role.USER, "我们这篇论文在 COCO 上做了实验，另外请记住我偏好中文回答。"),
            ChatMessage(Role.ASSISTANT, "好的，已记录。"),
        ], session_key="7")
        scopes = {it.text: it.scope for it in mgr.long.all_items()}
        assert scopes.get("用户偏好用中文回答") == SCOPE_GLOBAL
        assert scopes.get("实验在 COCO 上评估 mAP 50.2") == SCOPE_SESSION

    asyncio.run(run())
