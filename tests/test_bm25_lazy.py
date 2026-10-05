"""O-11（2026-10-01）：BM25 惰性重建回归。

背景：add() 原每批对全部累积文档重建 BM25Okapi（O(n²) 冗余）；
现 add() 只置 _dirty，search() 前一次性重建。探针（scripts/probe_bm25_rebuild.py）
实测真实 658 chunks 冗余 152.7ms / 15.3%，且 Top-K 逐位一致。
"""

from __future__ import annotations

from lightcore.rag.bm25 import BM25Retriever, jieba_tokenize

DOCS = [
    "注意力机制是 Transformer 的核心组件，用于捕获长距离依赖。",
    "实验结果表明，所提方法在三个基准数据集上均优于基线模型。",
    "我们在模型训练中采用了余弦学习率调度与梯度裁剪。",
    "图 3 展示了不同方法在测试集上的对比数据。",
    "消融实验验证了各模块的有效性。",
]


def test_lazy_add_defers_build() -> None:
    r = BM25Retriever(tokenize=jieba_tokenize)
    r.add(DOCS)
    # 惰性：add 后 BM25Okapi 尚未构建，仅标记 dirty
    assert r._bm25 is None
    assert r._dirty is True
    assert len(r._docs) == len(DOCS)


def test_search_builds_exactly_once() -> None:
    r = BM25Retriever(tokenize=jieba_tokenize)
    r.add(DOCS)
    hits1 = r.search("注意力 机制", k=3)
    assert hits1, "search 前必然重建 ⇒ 首次 search 不得返回空"
    assert r._dirty is False and r._bm25 is not None
    built_after_first = r._bm25
    r.search("实验 结果", k=3)  # 第二次 search 不得重复重建
    assert r._bm25 is built_after_first, "非 dirty 状态下不得重复构建 BM25Okapi"


def test_batched_add_equivalent_to_single_add() -> None:
    """O-11 核心等价性：分批 add（现行路径）与一次性 add 检索结果逐位一致。"""
    batched = BM25Retriever(tokenize=jieba_tokenize)
    for s in range(0, len(DOCS), 2):
        batched.add(DOCS[s : s + 2])
    single = BM25Retriever(tokenize=jieba_tokenize)
    single.add(DOCS)
    for q in ("注意力 机制", "实验 结果", "训练 方法"):
        assert batched.search(q, k=5) == single.search(q, k=5)


def test_empty_add_and_empty_retriever() -> None:
    r = BM25Retriever(tokenize=jieba_tokenize)
    assert r.search("任意", k=3) == []
    r.add([])  # 空 add 不得置 dirty（避免无谓重建）
    assert r._dirty is False
    r.add(DOCS)
    r.search("x", k=1)
    r.add([])  # 有内容后空 add 也不得触发重建
    assert r._dirty is False


def test_docs_added_but_never_searched_then_dropped() -> None:
    """边界：add 后未 search 即销毁 —— 不构建、不报错。"""
    r = BM25Retriever(tokenize=jieba_tokenize)
    r.add(DOCS)
    assert r._bm25 is None  # 销毁时无半成品状态
    del r
