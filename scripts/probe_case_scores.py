"""单用例检索取证：任意 case 在「不同查询预处理」下的篇内排序对照。

用途：当某个用例在两种预处理（`sections` / `full`）下结论不同时，
定位**是系统性劣化还是第 N 位之争**——只看报告里的 hit 无法区分这两者。

用法：
    .venv/Scripts/python.exe scripts/probe_case_scores.py r69
    .venv/Scripts/python.exe scripts/probe_case_scores.py m33 --top 6

输出：每种预处理策略下，各篇内部的完整排序（含分数）+ gold 落位。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from lightcore.rag.bm25 import jieba_tokenize  # noqa: E402
from lightcore.rag.embedder import BgeM3Embedder  # noqa: E402
from lightcore.rag.pipeline import RAGPipeline  # noqa: E402
from lightcore.rag.query_rewrite import expand_sections, rewrite_query  # noqa: E402
from lightcore.rag.rerank import CrossEncoderReranker  # noqa: E402
from papermate.runtime_settings import get_state  # noqa: E402

CORPUS_DIR = ROOT / "eval" / "corpus"
CASES = ROOT / "eval" / "cases" / "golden_real.jsonl"

PREPS = {"sections(①-b)": expand_sections, "full(旧)": rewrite_query}


def load_case(case_id: str) -> dict:
    with open(CASES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["id"] == case_id:
                return c
    raise SystemExit(f"未找到用例 {case_id!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--top", type=int, default=6, help="每篇取前 N 个（默认 6）")
    args = ap.parse_args()

    case = load_case(args.case_id)
    corpus = case.get("corpus") or [case.get("doc_id")]
    gold = list(case.get("gold") or [])
    print(f"=== 用例 {case['id']} ===")
    print(f"  query  = {case['query']!r}")
    print(f"  kind   = {case.get('kind')}")
    print(f"  corpus = {corpus}")
    print(f"  gold   = {gold}")
    print(f"  gains  = {case.get('gains')}")

    state = get_state()
    print(f"\n口径：sparse={state['use_bge_sparse']} · rerank fp16={state['rag_reranker_fp16']} "
          f"max_len={state['rag_reranker_max_length']}（均来自 runtime_settings，与线上同源）")
    print("\n加载 BGE-M3 + bge-reranker ...")
    t0 = time.perf_counter()
    embedder = BgeM3Embedder()
    reranker = CrossEncoderReranker(
        max_length=state["rag_reranker_max_length"], fp16=state["rag_reranker_fp16"]
    )
    pipes: list[RAGPipeline] = []
    for doc_id in corpus:
        pdf = CORPUS_DIR / f"{doc_id}.pdf"
        if not pdf.exists():
            raise SystemExit(f"语料缺失：{pdf}")
        p = RAGPipeline(embedder=embedder, reranker=reranker,
                        use_bge_sparse=state["use_bge_sparse"], tokenize=jieba_tokenize)
        p.add_document(pdf, doc_id=doc_id)
        pipes.append(p)
    print(f"加载完成（{(time.perf_counter() - t0) / 60:.1f}min）\n")

    for label, prep in PREPS.items():
        q = prep(case["query"]) or case["query"]
        print(f"--- {label} ---")
        print(f"    实际检索 query = {q!r}  （变化：{'否' if q == case['query'] else '是'}）")
        merged = []
        for i, p in enumerate(pipes):
            hits = p.query(q, top_k=args.top)
            merged.extend(hits)
            print(f"    第 {i} 篇（{corpus[i]}）:")
            for rank, r in enumerate(hits, 1):
                mark = "  ★ gold" if r.id in gold else ""
                print(f"        #{rank}  {r.id:<28} {r.score:>8.4f}{mark}")
        merged.sort(key=lambda r: r.score, reverse=True)
        top4 = [r.id for r in merged[:4]]
        hit = any(g in top4 for g in gold)
        rank = next((i + 1 for i, r in enumerate(merged) if r.id in gold), None)
        print(f"    ⇒ 合并后 gold 排名 = {rank}  ·  top4={top4}  ·  hit@4={'✅' if hit else '❌'}\n")


if __name__ == "__main__":
    main()
