"""LightCore · 向量索引（docs/05 §RAG / 索引 · FAISS）

用 FAISS IndexFlatIP（内积）做向量检索；由于 Embedder 输出已 L2 归一化，内积等价于余弦相似度。
提供 add / search，并在 add 时确保归一化，保证分数可解释为余弦。
"""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._texts: list[str] = []
        self._meta: list[dict] = []

    def add(self, vectors: list[list[float]], texts: list[str], meta: list[dict] | None = None) -> None:
        if not vectors:
            return
        arr = np.array(vectors, dtype="float32")
        # 归一化（防御性，确保内积=余弦）
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self._index.add(arr)
        self._texts.extend(texts)
        self._meta.extend(meta or [{} for _ in texts])

    def search(self, query_vec: list[float], k: int = 5) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []
        q = np.array([query_vec], dtype="float32")
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        k = min(k, self._index.ntotal)
        scores, ids = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=False) if i != -1]

    def __len__(self) -> int:
        return self._index.ntotal
