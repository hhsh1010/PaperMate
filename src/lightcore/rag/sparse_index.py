"""LightCore · 稀疏向量索引（docs/05 §RAG / BGE-M3 sparse 增强）

BGE-M3 除稠密向量外还输出稀疏（lexical）表示（token_id -> weight），与 BM25 同源词法信号，
但同处 BGE-M3 向量空间，融合更一致。本索引做倒排 + 点积累加，接口对齐 VectorIndex/BM25Retriever
（返回 (chunk_idx, score)，idx 与管线 self._chunks 顺序一致）。
"""

from __future__ import annotations


class SparseIndex:
    def __init__(self) -> None:
        # token_id -> [(chunk_idx, weight), ...]
        self._postings: dict[int, list[tuple[int, float]]] = {}
        # 已 add 的块总数（**含空向量块**）。作为下次 add 的下标偏移量：
        # 原实现用 enumerate 从 0 计数，多次 add（如增量追加）会让新块下标与
        # 已有块重叠，导致检索分数累加到错误块上。
        self._n = 0

    def add(self, sparse_vecs: list[dict[int, float]]) -> None:
        """按 chunk 顺序追加稀疏向量（idx 与管线 self._chunks 对齐，跨批次连续）。"""
        base = self._n
        for ci, vec in enumerate(sparse_vecs):
            if not vec:
                continue
            idx = base + ci
            for tok, w in vec.items():
                self._postings.setdefault(int(tok), []).append((idx, float(w)))
        self._n = base + len(sparse_vecs)

    def search(self, query_sparse: dict[int, float], k: int = 50) -> list[tuple[int, float]]:
        if not query_sparse:
            return []
        scores: dict[int, float] = {}
        for tok, qw in query_sparse.items():
            for ci, dw in self._postings.get(int(tok), ()):
                scores[ci] = scores.get(ci, 0.0) + float(qw) * float(dw)
        order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(ci, s) for ci, s in order[:k] if s > 0]

    def export_vecs(self) -> list[dict[int, float]]:
        """导出逐块稀疏向量（postings 反排），供索引状态持久化后 load 时按序 add 回。"""
        out: list[dict[int, float]] = [dict() for _ in range(self._n)]
        for tok, lst in self._postings.items():
            for idx, w in lst:
                if 0 <= idx < self._n:
                    out[idx][int(tok)] = float(w)
        return out

    def __len__(self) -> int:
        return self._n
