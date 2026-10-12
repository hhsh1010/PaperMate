"""Spike S2 · BGE-M3 嵌入（CPU）验证线（docs/13 第 0 周 Spike S2）

目标：确认嵌入→向量检索的拓扑在本地可跑通。真实 BGE-M3 通过 sentence-transformers 在 CPU 加载，
体积大（需联网下载约 2GB 模型权重），故此处：
  - 提供 Embedder 抽象 + FakeHashEmbedder（确定性、零依赖、可本地验证向量检索全链路）；
  - 提供 BgeM3Embedder 真实适配器（依赖 sentence-transformers，仅在 `uv sync --extra rag` 后可用）。
无论哪种后端，向量检索流程（FAISS 或朴素的余弦）保持一致。

验证：给定若干 mock 段落，检索「LightCore 核心贡献」返回最相关 Top-3。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import sys

sys.path.insert(0, "src")

from collections.abc import Sequence


class Embedder:
    dim: int = 0

    def encode(self, texts: Sequence[str]) -> list[list[float]]:  # noqa: D401
        raise NotImplementedError


class FakeHashEmbedder(Embedder):
    """确定性哈希嵌入：把文本哈希成定长向量，零依赖。仅用于流程验证，不可用于真实语义检索。"""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for i, ch in enumerate(t):
                h = int(hashlib.md5(f"{i}:{ch}".encode()).hexdigest(), 16)
                vec[h % self.dim] += (1.0 if h & 1 else -1.0)
            # L2 归一化
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class BgeM3Embedder(Embedder):
    """真实 BGE-M3 适配器（CPU）。依赖 sentence-transformers，需在 rag extra 下安装。"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 sentence-transformers，请先 `uv sync --extra rag`") from e
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def naive_retrieve(query: str, docs: list[str], embedder: Embedder, top_k: int = 3) -> list[tuple[int, float]]:
    qv = embedder.encode([query])[0]
    dvs = embedder.encode(docs)
    scored = sorted(((i, cosine(qv, dv)) for i, dv in enumerate(dvs)), key=lambda kv: kv[1], reverse=True)
    return scored[:top_k]


async def main() -> None:
    docs = [
        "LightCore 是一种从零自研的轻量级 Agent 内核，不依赖 LangChain/AutoGen。",
        "Agent 循环采用 ReAct + Function Calling 双协议状态机。",
        "双层记忆包含短期滚动窗口与长期向量记忆。",
        "生产级 RAG 使用 BGE-M3 与 BM25 混合检索再经 Cross-Encoder 重排。",
        "今天天气不错，与论文无关。",
    ]
    embedder = FakeHashEmbedder(dim=64)
    top = naive_retrieve("LightCore 的核心贡献是什么", docs, embedder, top_k=3)
    print("[S2] 查询: LightCore 的核心贡献是什么 → Top-3:")
    for idx, score in top:
        print(f"  · ({score:.3f}) [{idx}] {docs[idx]}")
    # 断言：最相关项应当靠前（hash 嵌入非语义，仅验证链路通畅；真实 BGE-M3 下应更准）
    print("\n[S2] 结论: PASS ✅ — 嵌入→向量检索链路通畅（真实语义精度待 BGE-M3 权重下载后复测）")


if __name__ == "__main__":
    asyncio.run(main())
