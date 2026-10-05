"""`retrieve` 阶段耗时剖析探针（2026-10-05）。

## 为什么需要（一次实测定全局）
`retrieve` p50 = 900ms，但我们**不知道这 900ms 花在哪**：
- query 编码（`encode_dense` + `encode_sparse`，各 2N 次前向）
- ANN 搜索 / BM25
- **rerank**（T-03 曾记录 670ms @ top_n=50，怀疑才是大头）

而 `pipeline.py` **早就有阶段埋点**（`PAPERMATE_RAG_TRACE=1` → `last_stage_ms`），
只是**从未落库、也没打日志** ⇒ 导致反复靠估算争论优化优先级。

## 安全性（为什么可以放心跑）
**全程只读**，不写任何文件：
- 用 `RAGPipeline.load_state` **直接读** `data/faiss/doc-N.{faiss,state}`；
- 签名**从 state 文件里读**（`bundle["signature"]`）⇒ 不自己算、不接触可能写盘的 `get_pipeline`；
- 只用 `DocumentService` 之外的方式构造 embedder/reranker（不触发索引重建）。

## 用法
    .venv/Scripts/python.exe scripts/probe_retrieve_stages.py
    # 可选：PAPERMATE_PROBE_QUERIES=8 指定查询次数
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

# ⚠️ 必须在 import pipeline 之前设置：`_TRACE` 是模块级常量，import 时求值一次。
os.environ["PAPERMATE_RAG_TRACE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightcore.rag.bm25 import jieba_tokenize  # noqa: E402
from lightcore.rag.embedder import BgeM3Embedder  # noqa: E402
from lightcore.rag.pipeline import RAGPipeline  # noqa: E402
from lightcore.rag.rerank import CrossEncoderReranker  # noqa: E402


def _real_queries(limit: int) -> list[str]:
    """取真实用户提问（贴近线上分布，而不是编造 query）。"""
    import sqlite3

    db = ROOT / "data" / "papermate.db"
    if not db.exists():
        return []
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(
            "SELECT content FROM messages WHERE role='user' AND LENGTH(content) BETWEEN 6 AND 120 "
            "ORDER BY id DESC LIMIT ?",
            (limit * 3,),
        ).fetchall()
    finally:
        con.close()
    out, seen = [], set()
    for (c,) in rows:
        c = (c or "").strip()
        if c and c not in seen:          # 去重：`query()` 有缓存，重复 query 会走 cache_hit
            seen.add(c)
            out.append(c)
    return out


def _pick_doc_with_index() -> tuple[int, Path, Path, str] | None:
    """找一个**已有落盘索引**的文档，返回 (doc_id, faiss_path, state_path, signature)。"""
    faiss_dir = ROOT / "data" / "faiss"
    for state_path in sorted(faiss_dir.glob("doc-*.state")):
        try:
            bundle = pickle.loads(state_path.read_bytes())
        except Exception:
            continue
        if not isinstance(bundle, dict) or not bundle.get("signature"):
            continue
        doc_id = int(state_path.stem.split("-")[1])
        faiss_path = faiss_dir / f"doc-{doc_id}.faiss"
        if faiss_path.exists():
            return doc_id, faiss_path, state_path, bundle["signature"]
    return None


def main() -> int:
    from papermate import runtime_settings as rs

    rt = rs.get_state()
    n_q = int(os.environ.get("PAPERMATE_PROBE_QUERIES", "6"))
    print("=" * 68)
    print("retrieve 阶段耗时剖析")
    print("=" * 68)
    print(f"  配置: rag_embedder={rt.get('rag_embedder')} use_bge_sparse={rt.get('use_bge_sparse')} "
          f"use_jieba={rt.get('use_jieba')}")
    print(f"        reranker fp16={rt.get('rag_reranker_fp16')} max_length={rt.get('rag_reranker_max_length')}")

    picked = _pick_doc_with_index()
    if picked is None:
        print("  ✗ 没找到带落盘索引的文档 ⇒ 无法只读剖析")
        return 1
    doc_id, faiss_path, state_path, sig = picked
    print(f"  文档: doc-{doc_id}（索引缓存命中，直接 load_state）")

    queries = _real_queries(n_q)
    if not queries:
        print("  ✗ 没取到真实 query")
        return 1
    print(f"  查询: {len(queries)} 条真实用户提问")

    print("\n  加载模型（BGE-M3 + reranker，冷启动约 30~60s）…")
    t0 = time.perf_counter()
    embedder = BgeM3Embedder()
    reranker = CrossEncoderReranker(
        max_length=int(rt.get("rag_reranker_max_length") or 384),
        fp16=bool(rt.get("rag_reranker_fp16", True)),
    )
    print(f"  模型就绪，耗时 {time.perf_counter() - t0:.1f}s")

    pipeline = RAGPipeline.load_state(
        faiss_path=faiss_path, state_path=state_path, signature=sig,
        embedder=embedder, reranker=reranker,
        use_bge_sparse=bool(rt.get("use_bge_sparse")),
        tokenize=jieba_tokenize,
    )
    if pipeline is None:
        print("  ✗ load_state 返回 None（签名不符/损坏）—— 探针不做重建，直接退出")
        return 1
    print(f"  管线就绪：{len(pipeline._chunks)} 个分块，top_n={pipeline.top_n}")

    # 预热一次（首次调用含 CUDA 上下文/懒加载，会污染统计）
    pipeline.query(queries[0])
    print("  （已预热 1 次，下面为正式测量）\n")

    stages: dict[str, list[float]] = {}
    totals: list[float] = []
    for i, q in enumerate(queries[:n_q]):
        t = time.perf_counter()
        pipeline.query(q)
        dt = (time.perf_counter() - t) * 1000.0
        totals.append(dt)
        row = dict(pipeline.last_stage_ms)
        for k, v in row.items():
            stages.setdefault(k, []).append(v)
        detail = "  ".join(f"{k}={v:.0f}" for k, v in sorted(row.items()) if k != "cache_hit")
        tag = "（缓存命中！）" if row.get("cache_hit") else ""
        print(f"  [{i + 1}] 总 {dt:7.1f} ms | {detail} {tag}")

    print("\n" + "=" * 68)
    print("汇总（中位数，ms）")
    print("=" * 68)
    tot_med = sorted(totals)[len(totals) // 2]
    # ⚠️ `total` 是**总和**、`cache_hit` 是**短路标记**，两者都不是"阶段" ⇒ 不计入阶段合计。
    #    首版把它们加进去 ⇒ 合计显示 197.5%，看着像"打点重复"，实为脚本口径错。
    NON_STAGES = {"total", "cache_hit"}
    real = {k: v for k, v in stages.items() if k not in NON_STAGES}
    print(f"  {'阶段':22s} {'中位':>9s} {'占总量':>8s}")
    acc = 0.0
    for k in sorted(real, key=lambda x: -sorted(real[x])[len(real[x]) // 2]):
        v = sorted(real[k])[len(real[k]) // 2]
        acc += v
        print(f"  {k:22s} {v:9.1f} {v / tot_med * 100:7.1f}%")
    print(f"  {'（阶段合计）':22s} {acc:9.1f} {acc / tot_med * 100:7.1f}%")
    print(f"  {'实测总耗时中位':22s} {tot_med:9.1f}")
    print("\n  注：阶段合计与总耗时的差额 = 跨文档合并/去重/后处理等未打点部分。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
