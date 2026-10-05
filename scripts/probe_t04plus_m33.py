"""T-04+ m33 用例取证脚本（固化版）。

目的：把 `_t04plus_scores4.log` 的取证数据复跑成可重现脚本。

为什么单独写（不直接复用 `eval/run_eval.py`）：
  - run_eval.py 跑全量 221 例，单脚本启动 + embedder 构建 ≈ 30s+；
    只想看 m33 一条时太重。
  - 验证一个**修复方案**（"每篇配额 N"）需要改 CompositeRAGPipeline 的合并策略——
    这是行为变更，应该先在脚本里"模拟"修复，验证有效后再改源码。

输出（默认写到 eval/_probe_m33.log）：
  - single 模式 k=3/4
  - composite 模式 k=3/4（每篇取 top_k=3 + 合并按分数截断，**与线上同源**）
  - 修复模拟 k=4（每篇取 4 = 与 metrics k 对齐）
  - 修复模拟 k=5（每篇取 5）

跑法（需 GPU/CPU 装载 BGE-M3 + bge-reranker-v2-m3）：
  .venv/Scripts/python.exe scripts/probe_t04plus_m33.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 路径：把 eval/ 和 src/ 加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from lightcore.rag.bm25 import jieba_tokenize  # noqa: E402
from lightcore.rag.embedder import BgeM3Embedder  # noqa: E402
from lightcore.rag.pipeline import RAGPipeline  # noqa: E402
from lightcore.rag.query_rewrite import rewrite_query  # noqa: E402
from lightcore.rag.rerank import CrossEncoderReranker  # noqa: E402
from papermate.runtime_settings import get_state  # noqa: E402

CASE = {
    "id": "m33",
    "query": "取消应该被记成什么状态",
    "corpus": ["async-pipeline", "hot-reload-cfg"],
    "gold": ["async-pipeline-c0"],
}
CORPUS_DIR = ROOT / "eval" / "corpus"


def _load_pipelines() -> list[RAGPipeline]:
    """加载 m33 corpus 里两个文档的独立 RAGPipeline（与线上 CompositeRAGPipeline 同口径）。"""
    state = get_state()
    embedder = BgeM3Embedder()
    reranker = CrossEncoderReranker(
        max_length=state["rag_reranker_max_length"],
        fp16=state["rag_reranker_fp16"],
    )
    pipes: list[RAGPipeline] = []
    for doc_id in CASE["corpus"]:
        pdf = CORPUS_DIR / f"{doc_id}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(f"语料缺失：{pdf}")
        pipe = RAGPipeline(
            embedder=embedder,
            reranker=reranker,
            use_bge_sparse=state["use_bge_sparse"],
            tokenize=jieba_tokenize,
        )
        pipe.add_document(pdf, doc_id=doc_id)
        pipes.append(pipe)
    return pipes


def _hit(retrieved_ids: list[str], gold: list[str]) -> bool:
    return any(g in retrieved_ids for g in gold)


def _print(label: str, retrieved: list, gold: list[str], elapsed_ms: float) -> None:
    ids = [(r.id, round(r.score, 4)) for r in retrieved]
    hit = _hit([c for c, _ in ids], gold)
    flag = "✅" if hit else "❌"
    print(f"  {flag} {label}  ({elapsed_ms:.0f}ms)  top={ids}")
    print(f"      gold={gold}")


def probe_single(merged: RAGPipeline, query: str, gold: list[str]) -> None:
    """单管线 = 两篇进同一 RAGPipeline，全局排序（= 改动前评测口径 / 旧 multi_doc 行为）。
    注意：评测里的 single 模式**不调 rewrite_query**（原 query 直接喂子管线），
    这与 composite 模式的 rewrite_query 路径差异是 m33 漏掉的根因之一。"""
    print(f"\n原 query = {query!r}")
    for k in (3, 4):
        t0 = time.perf_counter()
        retrieved = merged.query(query, top_k=k)
        _print(f"single k={k}", retrieved, gold, (time.perf_counter() - t0) * 1000)


def probe_composite(pipes: list[RAGPipeline], query: str, gold: list[str]) -> None:
    """Composite 模式：
    1) 与线上同源（每篇 top_k + 合并按分数截断）
    2) 修复模拟：每篇配额 N（不修改源码，脚本里直接调各子管线）"""
    rewritten = rewrite_query(query) or query

    print("\n=== Composite（与线上同源：每篇各取 top_k → 整体按分数截断）===")
    for k in (3, 4):
        t0 = time.perf_counter()
        merged: list = []
        for p in pipes:
            merged.extend(p.query(rewritten, top_k=k))
        merged.sort(key=lambda r: r.score, reverse=True)
        retrieved = merged[:k]
        _print(f"composite k={k} (rewrite_query)", retrieved, gold, (time.perf_counter() - t0) * 1000)

    # ★ 修复模拟 C：composite **用原 query**（与 single 对齐，验证 rewrite_query 是否是根因）
    print("\n=== 修复模拟 C：composite 用原 query（不 rewrite）===")
    for k in (3, 4):
        t0 = time.perf_counter()
        merged: list = []
        for p in pipes:
            merged.extend(p.query(query, top_k=k))
        merged.sort(key=lambda r: r.score, reverse=True)
        retrieved = merged[:k]
        _print(f"composite-orig k={k}", retrieved, gold, (time.perf_counter() - t0) * 1000)

    print("\n=== 各篇内部（同口径，composite 实际用的 rewrite query）取 4 ===")
    for i, p in enumerate(pipes):
        print(f"  --- 第 {i} 篇 ---")
        for r in p.query(rewritten, top_k=4):
            print(f"      {r.id:<30}  {r.score:.4f}")

    print("\n=== 各篇内部取 6（看 gold 在篇内的真实位置）===")
    for i, p in enumerate(pipes):
        for r in p.query(rewritten, top_k=6):
            print(f"  [{i}] {r.id:<30}  {r.score:.4f}")

    print("\n=== 修复模拟（每篇配额 = max(top_k, per_doc_top_k)）===")
    print("  假设：每篇至少 top_k 个进榜后再合并整体排序，模拟配额修复")
    for k in (3, 4, 5):
        per_doc_top_k = k  # ★ 每篇保底 top_k 个进榜
        t0 = time.perf_counter()
        merged = []
        for p in pipes:
            merged.extend(p.query(rewritten, top_k=per_doc_top_k))
        merged.sort(key=lambda r: r.score, reverse=True)
        # 与现行不同：不再整体截到 k，而是把"每篇至少 1 个"保留
        # 修复方案 A：保留每篇前 (top_k / num_docs + 1) 个
        per_doc_min = k // len(pipes) + 1
        seen_per_doc: dict[str, int] = {}
        result = []
        for r in merged:
            doc = r.id.rsplit("-c", 1)[0]
            seen_per_doc.setdefault(doc, 0)
            if seen_per_doc[doc] < per_doc_min or len(result) < k:
                result.append(r)
                seen_per_doc[doc] += 1
            if len(result) >= k:
                break
        _print(
            f"repair-A k={k} per_doc_min={per_doc_min}",
            result,
            gold,
            (time.perf_counter() - t0) * 1000,
        )

        # 修复方案 B：每篇保底 k 个进榜（不考虑合并后 top_k 限制）
        t0 = time.perf_counter()
        merged = []
        for p in pipes:
            merged.extend(p.query(rewritten, top_k=k))
        merged.sort(key=lambda r: r.score, reverse=True)
        # 每篇前 k 个都进榜
        result = []
        seen: dict[str, int] = {}
        for r in merged:
            doc = r.id.rsplit("-c", 1)[0]
            if seen.get(doc, 0) < k:
                result.append(r)
                seen[doc] = seen.get(doc, 0) + 1
        _print(
            f"repair-B k={k} per_doc_topk={k}",
            result,
            gold,
            (time.perf_counter() - t0) * 1000,
        )

        # ★ 修复模拟 D：composite 用原 query + 每篇前 k 个都进榜
        t0 = time.perf_counter()
        merged = []
        for p in pipes:
            merged.extend(p.query(query, top_k=k))
        merged.sort(key=lambda r: r.score, reverse=True)
        result = []
        seen: dict[str, int] = {}
        for r in merged:
            doc = r.id.rsplit("-c", 1)[0]
            if seen.get(doc, 0) < k:
                result.append(r)
                seen[doc] = seen.get(doc, 0) + 1
        _print(
            f"repair-D k={k} (orig_query+per_doc_topk)",
            result,
            gold,
            (time.perf_counter() - t0) * 1000,
        )


def main() -> None:
    print(f"=== T-04+ m33 取证（{CASE['id']}: {CASE['query']}）===")
    print(f"  corpus = {CASE['corpus']}")
    print(f"  gold   = {CASE['gold']}")

    print("\n加载 BGE-M3 + bge-reranker（prod profile，与线上同源）...")
    t0 = time.perf_counter()
    pipes = _load_pipelines()
    print(f"加载完成（{(time.perf_counter() - t0) / 60:.1f}min）")

    # 构造单管线（两篇进同一个池子 = 改动前评测口径）
    state = get_state()
    base = pipes[0]
    merged = RAGPipeline(
        embedder=base.embedder,
        reranker=base.reranker,
        use_bge_sparse=state["use_bge_sparse"],
        tokenize=jieba_tokenize,
    )
    for doc_id in CASE["corpus"]:
        merged.add_document(CORPUS_DIR / f"{doc_id}.pdf", doc_id=doc_id)

    probe_single(merged, CASE["query"], CASE["gold"])
    probe_composite(pipes, CASE["query"], CASE["gold"])

    print("\nDONE")


if __name__ == "__main__":
    main()