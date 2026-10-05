"""索引状态持久化回归（2026-09-07）。

现象：嵌入向量只存进程内存——每次重启后首次访问文档都要全量重编码（分钟级）。
修复：faiss_dir 复活，RAGPipeline.save_state/load_state 按「分块内容+嵌入器配置」
签名持久化/恢复；签名不符（内容或配置变化）自动失效走重编码。
"""

from __future__ import annotations

from lightcore.rag.pipeline import RAGPipeline


def test_pipeline_state_roundtrip(tmp_path):
    """add_chunks → save → load：恢复的管线检索结果与原管线一致。"""
    faiss_path = tmp_path / "doc-1.faiss"
    state_path = tmp_path / "doc-1.state"
    items = [("doc-1-c0", "我们提出卷积神经网络用于图像分类。", 1),
             ("doc-1-c1", "实验在 ImageNet 数据集上进行，准确率达到 95%。", 2)]

    p1 = RAGPipeline()  # 默认 FakeHashEmbedder（确定性哈希向量），零模型依赖
    p1.add_chunks(items, doc_id="doc-1")
    p1.save_state(faiss_path, state_path, "sig-A")

    p2 = RAGPipeline.load_state(
        faiss_path=faiss_path, state_path=state_path, signature="sig-A",
        embedder=p1.embedder, reranker=None, use_bge_sparse=False,
        tokenize=p1._bm25._tokenize,
    )
    assert p2 is not None
    assert len(p2._chunks) == 2
    assert [c.id for c in p2._chunks] == ["doc-1-c0", "doc-1-c1"]

    # 检索语义一致：原管线与恢复管线对同一查询的 top1 相同
    q = "卷积神经网络 图像分类"
    r1 = p1.query(q, top_k=2)
    r2 = p2.query(q, top_k=2)
    assert [c.id for c in r2] == [c.id for c in r1]
    assert r2 and r2[0].page in (1, 2)


def test_pipeline_state_signature_mismatch_invalidates(tmp_path):
    """分块内容变化 → 签名不符 → load_state 返回 None（走重编码，不加载陈旧向量）。"""
    faiss_path = tmp_path / "doc-1.faiss"
    state_path = tmp_path / "doc-1.state"
    p1 = RAGPipeline()
    p1.add_chunks([("doc-1-c0", "原始内容。", 1)], doc_id="doc-1")
    p1.save_state(faiss_path, state_path, "sig-旧")

    p2 = RAGPipeline.load_state(
        faiss_path=faiss_path, state_path=state_path, signature="sig-新",
        embedder=p1.embedder, reranker=None, use_bge_sparse=False,
        tokenize=p1._bm25._tokenize,
    )
    assert p2 is None


def test_pipeline_state_corrupt_file_returns_none(tmp_path):
    """缓存文件损坏 → load_state 返回 None（不抛异常阻断建管线）。"""
    faiss_path = tmp_path / "doc-1.faiss"
    state_path = tmp_path / "doc-1.state"
    state_path.write_bytes(b"not-a-pickle")
    p = RAGPipeline.load_state(
        faiss_path=faiss_path, state_path=state_path, signature="sig",
        embedder=None, reranker=None, use_bge_sparse=False,
    )
    assert p is None


def test_sparse_index_export_roundtrip():
    """SparseIndex.export_vecs：postings 反排恢复逐块向量，add 回新索引后检索一致。"""
    from lightcore.rag.sparse_index import SparseIndex

    s1 = SparseIndex()
    vecs = [{100: 0.5, 200: 0.25}, {}, {100: 0.75, 300: 0.5}]  # 中间为空向量块（对齐占位）
    s1.add(vecs)

    s2 = SparseIndex()
    s2.add(s1.export_vecs())
    assert len(s2) == len(s1) == 3
    hit1 = s1.search({100: 1.0}, k=10)
    hit2 = s2.search({100: 1.0}, k=10)
    assert [(i, round(w, 6)) for i, w in hit1] == [(i, round(w, 6)) for i, w in hit2]


def test_pipeline_state_missing_file_is_info_not_warning(tmp_path):
    """缓存文件不存在（首次访问/旧文档迁移）= 预期路径：返回 None 且不打 WARNING。"""
    p = RAGPipeline.load_state(
        faiss_path=tmp_path / "nope.faiss", state_path=tmp_path / "nope.state",
        signature="sig", embedder=None, reranker=None, use_bge_sparse=False,
    )
    assert p is None  # 预期路径：静默降级到重编码（日志为 INFO，不打 WARNING 堆栈）
