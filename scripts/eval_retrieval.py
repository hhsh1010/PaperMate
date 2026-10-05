"""LightCore · 最小检索评测集（量化 Cross-Encoder / BGE-M3 sparse 增益）

背景：RAG 管线默认用混合检索(RRF) + ScoreReranker（按融合分直接排），
CrossEncoderReranker 已可装配（--reranker cross）但默认未启用；BGE-M3 sparse
信号已可选接入（--sparse）但默认关闭。本脚本用内置小型学术问答集量化对比，
为「是否默认开启」提供数据依据。

用法：
  # 1) sandbox / 无重型依赖：fake 模式跑通流程（无语义，仅验证脚本可用）
  uv run python scripts/eval_retrieval.py --embedder fake
  # 2) 真实量化（需 rag extra + BGE-M3 权重，在本地终端跑）
  uv run python scripts/eval_retrieval.py --embedder real --reranker cross
  # 3) 同时对比 BGE-M3 sparse 信号（仅 embedder real 生效）
  uv run python scripts/eval_retrieval.py --embedder real --reranker cross --sparse

指标：hit@1/3/5（至少一个标注相关块进入 Top-k）+ MRR@5。
"""

from __future__ import annotations

import argparse
import sys

# 内置小型学术问答集：query -> 相关 chunk 索引（0-based，对应 CORPUS）
CORPUS: list[str] = [
    "BGE-M3 是智源研究院开源的多语言嵌入模型，支持稠密、稀疏与多向量三种检索表示，广泛用于语义检索。",
    "BM25 是经典词法稀疏检索算法，依赖关键词精确匹配，对同义改写不敏感。",
    "混合检索用 RRF 融合向量检索与 BM25 的排名，score = Σ 1/(k + rank)，k 常取 60。",
    "Cross-Encoder 重排把 (query, chunk) 拼接输入模型打分，精度高于向量相似度，常用于精排阶段。",
    "FAISS 是高效向量索引库，IndexFlatIP 用内积做最近邻搜索，配合归一化向量等价于余弦相似度。",
    "语义分块按句子累积到目标 token 数（300–500）切分，块间留 overlap 以保留上下文。",
    "BGE-reranker-v2-m3 是重排模型，对候选段落与查询做相关性打分，通常只对 Top50 候选重排。",
    "检索增强生成 RAG 把检索到的文档片段注入提示，提升回答的事实性与可溯源。",
]
QUERIES: list[tuple[str, list[int]]] = [
    ("什么是 BGE-M3 嵌入模型？", [0]),
    ("BM25 有什么缺点？", [1]),
    ("RRF 如何融合多个检索结果？", [2]),
    ("Cross-Encoder 重排为什么精度更高？", [3]),
    ("FAISS 用什么索引做相似检索？", [4]),
    ("语义分块如何切分文档？", [5]),
    ("bge-reranker 的作用是什么？", [6]),
    ("RAG 如何提升回答可靠性？", [7]),
]


def _hit_mrr(pred: list[int], gold: list[int]) -> tuple[list[bool], float]:
    """pred 为预测 chunk 索引序列（已按 score 降序），gold 为相关索引集。
    返回 ([hit@1, hit@3, hit@5], mrr@5)。"""
    hit = [any(i in gold for i in pred[:k]) for k in (1, 3, 5)]
    mrr = 0.0
    for rank, i in enumerate(pred[:5], start=1):
        if i in gold:
            mrr = 1.0 / rank
            break
    return hit, mrr


def _build_pipeline(embedder: str, reranker: str, sparse: bool):
    from lightcore.rag.embedder import BgeM3Embedder, FakeHashEmbedder
    from lightcore.rag.pipeline import RAGPipeline
    from lightcore.rag.rerank import CrossEncoderReranker, ScoreReranker

    emb = BgeM3Embedder() if embedder == "real" else FakeHashEmbedder()
    rk = CrossEncoderReranker() if reranker == "cross" else ScoreReranker()
    return RAGPipeline(embedder=emb, reranker=rk, use_bge_sparse=sparse, top_n_hybrid=50, top_k=5)


def main() -> int:
    ap = argparse.ArgumentParser(description="最小检索评测集（量化 Cross-Encoder / BGE-M3 sparse 增益）")
    ap.add_argument("--embedder", choices=["fake", "real"], default="fake",
                    help="fake=哈希占位(无语义，仅验证流程) / real=BGE-M3(需 rag extra + 权重)")
    ap.add_argument("--reranker", choices=["score", "cross"], default="score",
                    help="score=融合分直接排(基线) / cross=CrossEncoderReranker(需权重)")
    ap.add_argument("--sparse", action="store_true", help="启用 BGE-M3 sparse 信号（仅 real 生效）")
    args = ap.parse_args()

    print(f"== 检索评测：embedder={args.embedder} reranker={args.reranker} sparse={args.sparse}")
    pipe = _build_pipeline(args.embedder, args.reranker, args.sparse)
    pipe.add_document(CORPUS, doc_id="eval")

    agg = {"hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0}
    n = len(QUERIES)
    print(f"{'query':<26}{'hit@1':<7}{'hit@3':<7}{'hit@5':<7}{'MRR@5'}")
    for q, gold in QUERIES:
        pred_ids = [r.id for r in pipe.query(q, top_k=5)]
        pred_idx = [int(p.split("-c")[1]) for p in pred_ids]
        hit, mrr = _hit_mrr(pred_idx, gold)
        agg["hit1"] += hit[0]
        agg["hit3"] += hit[1]
        agg["hit5"] += hit[2]
        agg["mrr"] += mrr
        print(f"{q[:24]:<26}{str(hit[0]):<7}{str(hit[1]):<7}{str(hit[2]):<7}{mrr:.3f}")

    print("-" * 52)
    print(f"平均 hit@1={agg['hit1'] / n:.3f}  hit@3={agg['hit3'] / n:.3f}  "
          f"hit@5={agg['hit5'] / n:.3f}  MRR@5={agg['mrr'] / n:.3f}")
    print("提示：对比 score(基线) vs cross 两列即可量化 Cross-Encoder 增益；"
          "真实量化请用 --embedder real（本地终端）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
