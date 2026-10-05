"""LightCore · 混合检索（docs/05 §RAG / 混合检索 RRF）

Reciprocal Rank Fusion (RRF)：融合向量检索与 BM25 的排名，
score = Σ 1/(k + rank)，k=60（docs/05）。不依赖具体分数尺度，鲁棒。
"""

from __future__ import annotations

from collections.abc import Callable


class HybridRetriever:
    def __init__(self, vector_search: Callable[[str, int], list[tuple[int, float]]],
                 bm25_search: Callable[[str, int], list[tuple[int, float]]], k: int = 60,
                 sparse_search: Callable[[str, int], list[tuple[int, float]]] | None = None) -> None:
        self._vector = vector_search
        self._bm25 = bm25_search
        self._sparse = sparse_search
        self.k = k

    def retrieve(self, query: str, top_n: int = 50) -> list[tuple[int, float]]:
        vec = self._vector(query, top_n)
        bm = self._bm25(query, top_n)
        fused: dict[int, float] = {}
        for rank, (idx, _) in enumerate(vec):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.k + rank + 1)
        for rank, (idx, _) in enumerate(bm):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.k + rank + 1)
        if self._sparse is not None:
            sp = self._sparse(query, top_n)
            for rank, (idx, _) in enumerate(sp):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.k + rank + 1)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        # 2026-09-06：截断到 top_n——三路融合的规模上界是 3×top_n（150 条），
        # 此前全量返回会把 CrossEncoder 重排的输入固定放大 3 倍（耗时/显存）。
        return ordered[:top_n]
