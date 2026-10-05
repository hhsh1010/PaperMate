"""重排器契约回归测试（2026-09-03：GPU 加速后暴露的潜伏 bug）

背景：CrossEncoderReranker 此前几乎加载不成功（HF 超时）→ 降级哨兵 → 走 ScoreReranker，
      该 bug 一直休眠。GPU 加速后重排器能正常加载，真实 CrossEncoder 路径首次生效即崩溃：
      rerank 用 candidates[0]（全局 chunk 下标）去取 texts，而 RAGPipeline.query 传的是
      「按候选位置一一对应」的文本列表 → chunk 下标（137/289…）远超候选数 → IndexError
      → 整条检索抛错 → /search 返回 []、会话被判定「未关联任何文档」。

契约（现以本文件锁定）：texts 与 candidates 按位置一一对应，不是按 chunk 下标索引的全量文本。
"""

from __future__ import annotations

import sys
import types

import pytest

from lightcore.rag.embedder import FakeHashEmbedder
from lightcore.rag.pipeline import RAGPipeline


class _FakeCrossEncoder:
    """记录并回传打分的 CrossEncoder 替身：predict 返回与 pairs 等长的递增分数。"""

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.last_pairs: list[tuple[str, str]] = []

    def predict(self, pairs, show_progress_bar=None):  # noqa: ANN001
        # 签名与真实 `sentence_transformers.CrossEncoder.predict` 对齐
        # （2026-09-11：rerank 显式传 show_progress_bar=False 关闭 tqdm 噪音）
        self.last_pairs = list(pairs)
        return [float(i) for i in range(len(pairs))]


@pytest.fixture()
def fake_st(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    holder: dict = {}

    def _factory(*args, **kwargs):  # noqa: ANN002, ANN003
        enc = _FakeCrossEncoder(*args, **kwargs)
        holder["inst"] = enc
        return enc

    mod.CrossEncoder = _factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return holder


class TestCrossEncoderRerankContract:
    def test_positional_texts_not_chunk_index(self, fake_st):
        """候选下标很大（全局 chunk 下标）时，必须按位置取文本，不得 IndexError。"""
        from lightcore.rag.rerank import CrossEncoderReranker

        r = CrossEncoderReranker(model_name="/fake/path", device="cpu")
        candidates = [(137, 0.9), (289, 0.8)]   # 全局 chunk 下标，远超 texts 长度
        texts = ["候选A文本", "候选B文本"]        # 按候选位置一一对应
        out = r.rerank("查询", candidates, top_k=2, texts=texts)
        # 不抛异常，且保留原始 chunk 下标；最高分（位置1）排前
        assert [idx for idx, _ in out] == [289, 137]
        pairs = fake_st["inst"].last_pairs
        assert [t for _, t in pairs] == ["候选A文本", "候选B文本"]

    def test_length_mismatch_raises_value_error(self, fake_st):
        from lightcore.rag.rerank import CrossEncoderReranker

        r = CrossEncoderReranker(model_name="/fake/path", device="cpu")
        with pytest.raises(ValueError, match="必须等于候选数"):
            r.rerank("查询", [(0, 1.0), (1, 0.5)], top_k=2, texts=["只有一条"])

    def test_missing_texts_raises(self, fake_st):
        from lightcore.rag.rerank import CrossEncoderReranker

        r = CrossEncoderReranker(model_name="/fake/path", device="cpu")
        with pytest.raises(ValueError):
            r.rerank("查询", [(0, 1.0)], top_k=1, texts=None)


class _BoomReranker:
    """任何调用都抛错的重排器：用于验证 query 的降级兜底。"""

    def rerank(self, query, candidates, top_k=5, texts=None):  # noqa: ANN001
        raise RuntimeError("boom")


class TestPipelineQueryResilience:
    def _pipeline(self, reranker) -> RAGPipeline:
        p = RAGPipeline(
            embedder=FakeHashEmbedder(dim=64),
            reranker=reranker,
            use_cross_encoder=False,
            use_bge_sparse=False,
        )
        items = [(f"c{i}", f"这是第{i}段内容，包含检索关键词。", i + 1) for i in range(20)]
        p.add_chunks(items, doc_id="doc-test")
        return p

    def test_query_baseline_with_score_reranker(self):
        from lightcore.rag.rerank import ScoreReranker

        p = self._pipeline(ScoreReranker())
        res = p.query("检索关键词", top_k=3)
        assert len(res) == 3
        assert all(r.text for r in res)

    def test_reranker_failure_degrades_not_crash(self):
        """重排器异常不得摧毁整条检索：退化为融合分数排序并仍有结果。"""
        p = self._pipeline(_BoomReranker())
        res = p.query("检索关键词", top_k=3)
        assert len(res) == 3
