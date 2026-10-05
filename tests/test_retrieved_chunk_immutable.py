"""RetrievedChunk 不可变性 + 查询缓存副本保护（2026-09-15 加固回归）。

背景：`RAGPipeline.query` 命中查询缓存时返回 `list(cached)` —— 这是**浅拷贝**，
只复制列表容器，元素仍指向同一批 RetrievedChunk 对象。由此产生两条不同强度的保证：

- 「调用方原地改返回的列表」（append/pop/sort/clear）**不会**污染缓存 ✅
  → 由 `list(cached)` 提供，用例 2 守护。
- 「调用方原地改某个 chunk 的字段」（`c.score = x`）**会**污染缓存 ❌
  → 因此把 RetrievedChunk 改为 `frozen=True`，从类型层面禁止，用例 1 守护。

用例 3 守护另一条相关契约：索引变更（add_chunks）时查询缓存必须整体失效，
否则新增的分块永远检索不到（缓存返回的是旧索引算出的结果）。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lightcore.rag.pipeline import RAGPipeline, RetrievedChunk


def test_retrieved_chunk_is_immutable() -> None:
    """字段赋值必须抛 FrozenInstanceError —— 防止静默污染查询缓存。"""
    c = RetrievedChunk(id="d-c0", text="原文段落", page=1, score=0.9)

    with pytest.raises(FrozenInstanceError):
        c.score = 0.1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        c.page = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        c.text = "篡改"  # type: ignore[misc]

    # 原值不受影响
    assert (c.score, c.page, c.text) == (0.9, 1, "原文段落")


def test_query_cache_immune_to_caller_mutating_returned_list() -> None:
    """调用方清空/篡改返回的列表，不得影响缓存（list(cached) 副本保护）。"""
    pipeline = RAGPipeline()  # 默认 FakeHashEmbedder + ScoreReranker：零重依赖
    pipeline.add_chunks([("c0", "轻量化方法显著降低了模型参数量", 1), ("c1", "另一段无关内容", 2)])

    first = pipeline.query("轻量化", top_k=2)
    assert first, "假嵌入器下也应能检索到内容"

    # 破坏性操作：清空 + 塞入伪造条目
    first.clear()
    first.append(RetrievedChunk(id="fake", text="伪造内容", page=None, score=1.0))

    second = pipeline.query("轻量化", top_k=2)
    assert second, "缓存应仍然可用"
    assert all(r.id != "fake" for r in second), "缓存被调用方的列表操作污染了"


def test_query_cache_cleared_on_index_change() -> None:
    """索引变更（add_chunks）后缓存必须失效 —— 否则新增分块永远检索不到。"""
    pipeline = RAGPipeline()
    pipeline.add_chunks([("c0", "苹果 苹果 苹果", 1)])
    first = pipeline.query("苹果", top_k=3)
    assert {r.id for r in first} == {"c0"}

    # 新增分块后，同一 query 必须能看到新块（若缓存未清，仍只会返回 c0）
    pipeline.add_chunks([("c1", "苹果 苹果 苹果 苹果", 2)])
    second = pipeline.query("苹果", top_k=3)
    assert "c1" in {r.id for r in second}, "索引变更后查询缓存未失效"
