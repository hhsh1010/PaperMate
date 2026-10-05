"""rerank 旋钮 · 质量对照探针（2026-10-05）

回答两个问题（用户"先评估"→批准跑对照）：

  Q1 质量可判定：`max_length` 384→256 是否损伤召回？
     语料：`eval/cases/golden_real.jsonl`（221 case / 12 篇 / 带 gold + gains）
     指标：hit@5 / nDCG@5 / MRR@10 —— **有 gold，可判定好与坏**。
     臂：384（线上）/ 256（拟下调）/ 512（反向，验证"质量优先下该不该上调"）

  Q2 无损证明：`top_n` 20→10 是否改变最终交给 LLM 的证据？
     语料：`data/uploads` 真实论文（23~75 块/篇）
     ⚠️ **为什么不用 golden_real**：其合成语料每篇仅 3~6 块 ⇒ 候选数 < top_n=10
        ⇒ 两臂候选池完全相同 ⇒ **指标恒饱和、测不出差异**（项目已记录该教条）。
     判据：**最终 top-5 输出是否完全一致** —— 若一致，则"减少候选"对用户可见结果零影响，
           质量必然不变；这是演绎推理，**不需要 gold**。

**同源性（关键）**
- Q1 全程复用 `eval/run_eval.py` 的 `PipelineFactory` / `run_case` / `summarize_results`，
  `profile="prod"`（CrossEncoder 重排 + jieba 分词 + BGE-M3 稀疏）、`merge="composite"`
  （与线上 `CompositeRAGPipeline` 同源）—— 与 T-03/T-04 报告同一口径。
- Q2 复用线上同一个 `CompositeRAGPipeline` 类。
- `top_n` 在 `RAGPipeline.__init__` 里只落到 `self.top_n`，`query()` 读 `self.top_n`
  ⇒ 改实例属性与"构造时传不同值"**完全等价**，故可切臂而不重建（省一次重编码）。

**资源**：单进程；GPU（BGE-M3 ≈2.2GB + CrossEncoder ≈2.2GB，臂间释放）。
预计 5~15 分钟。不写 DB、不改索引、不改配置。

用法：
    .venv/Scripts/python.exe -u scripts/probe_rerank_arms_quality.py
    .venv/Scripts/python.exe -u scripts/probe_rerank_arms_quality.py --only q1
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

try:  # Windows 控制台默认 cp936，中文/符号会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

_GOLDEN = _ROOT / "eval" / "cases" / "golden_real.jsonl"
_UPLOADS = _ROOT / "data" / "uploads"
_OUT = _ROOT / "eval" / "rerank_arms_quality_20261005.json"
_LOCK_PATH = str(_ROOT / "scripts" / ".rerank_arms.lock")

#: Q2 的对照查询（论文通用问题，覆盖方法/数据/结论/公式等不同语义面）
QUERIES: list[str] = [
    "本文提出的方法是什么？",
    "系统的整体架构如何设计？",
    "实验中使用了哪些数据集？",
    "与基线方法相比效果提升多少？",
    "该方法存在哪些局限性？",
    "训练时用了什么优化器和学习率？",
    "消融实验验证了哪些模块？",
    "论文的核心贡献有哪几点？",
    "损失函数是怎么定义的？",
    "网络结构的骨干是什么？",
    "评价指标有哪些？",
    "模型参数量和计算量是多少？",
    "注意力机制是如何设计的？",
    "特征融合采用了什么策略？",
    "实验的硬件环境是什么？",
    "与已有方法的对比结论是什么？",
    "数据预处理流程是怎样的？",
    "超参数是如何选取的？",
    "方法的创新点在哪里？",
    "未来工作有哪些方向？",
]


def _try_lock(path: str) -> int | None:
    """跨平台非阻塞独占锁；已被占用返回 None。

    为什么必须有：本环境会**重放整个 turn**（同一任务两个实例并发）。
    基准测量必须单实例，否则两个进程同时抢 GPU/内存 ⇒ 数字不可信（09-24 已踩）。
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


# ------------------------------------------------------------------ Q1


def run_q1() -> dict:
    """max_length 三臂对照（golden_real，有 gold，可判定好坏）。"""
    import metrics as metrics_mod  # noqa: E402  eval/metrics.py
    import run_eval  # noqa: E402
    import torch

    from lightcore.rag.query_rewrite import expand_sections  # noqa: E402

    cases = run_eval.load_cases(_GOLDEN)
    print(f"[Q1] 用例 {len(cases)} 条（含 answerable=False 的负样本，聚合时自动排除）", flush=True)

    arms = [384, 256, 512]
    keys = ("hit@5", "recall@5", "precision@5", "ndcg@5", "mrr@10", "map")
    out: dict = {"cases": len(cases), "arms": {}}

    for ml in arms:
        print(f"\n[Q1] === 臂 max_length={ml} ===", flush=True)
        t0 = time.perf_counter()
        factory = run_eval.PipelineFactory(
            embedder_kind="real", profile="prod", use_bge_sparse=True,
            reranker_fp16=True, reranker_max_length=ml,
            rerank_source=f"probe ml={ml}", merge="composite",
        )
        factory.retrieve_top_k = 0          # 0 = 沿用 max(k,_MRR_K)=10（改动前行为）
        factory.query_rewrite_mode = "sections"
        factory.query_prep = expand_sections  # 与线上同源
        rows = []
        for i, c in enumerate(cases):
            rows.append(run_eval.run_case(c, 5, factory))
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(cases)} …", flush=True)
        m = metrics_mod.summarize_results(rows)
        lat = statistics.median([r["latency_ms"] for r in rows])
        out["arms"][str(ml)] = {
            "metrics": {k: m.get(k) for k in keys},
            "latency_p50_ms": round(lat, 1),
            "latency_mean_ms": round(statistics.mean([r["latency_ms"] for r in rows]), 1),
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        print("    -> " + "  ".join(f"{k}={m.get(k)}" for k in keys)
              + f"  p50={lat:.0f}ms", flush=True)
        del factory, rows
        gc.collect()
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ Q2


def _pick_real_docs(limit: int) -> list[tuple[str, Path]]:
    """挑块数最多的若干篇真实论文（块数越多，top_n 20↔10 越可能有差异）。"""
    import pickle

    cand: list[tuple[int, str, Path]] = []
    for state in sorted((_ROOT / "data" / "faiss").glob("doc-*.state")):
        try:
            b = pickle.loads(state.read_bytes())
            n = len(b.get("texts") or [])
        except Exception:  # noqa: BLE001
            continue
        doc_id = state.stem  # 形如 doc-10
        num = doc_id.split("-", 1)[1]
        pdfs = sorted((_UPLOADS / num).glob("*.pdf"))
        if pdfs:
            cand.append((n, num, pdfs[0]))
    cand.sort(reverse=True)
    chosen = [(num, pdf, n) for n, num, pdf in cand[:limit]]
    print(f"[Q2] 选用 {len(chosen)} 篇真实论文：" +
          ", ".join(f"doc-{num}({n}块)" for num, _, n in chosen), flush=True)
    return [(num, pdf) for num, pdf, _ in chosen]


def run_q2(limit_docs: int = 6) -> dict:
    """top_n 20↔10 对照（真实文档；判据 = 最终 top-5 输出一致性）。"""
    import run_eval  # noqa: E402  仅为复用同源构造（embedder/reranker 档位）
    import torch

    from lightcore.rag.bm25 import jieba_tokenize  # noqa: E402
    from lightcore.rag.embedder import BgeM3Embedder  # noqa: E402
    from lightcore.rag.pipeline import RAGPipeline  # noqa: E402
    from lightcore.rag.query_rewrite import expand_sections  # noqa: E402
    from lightcore.rag.rerank import CrossEncoderReranker  # noqa: E402
    from papermate.services.chat_service import CompositeRAGPipeline  # noqa: E402

    docs = _pick_real_docs(limit_docs)
    emb = BgeM3Embedder()
    rk = CrossEncoderReranker(max_length=384, fp16=True)   # 与线上同档
    pipes: list = []
    for num, pdf in docs:
        p = RAGPipeline(embedder=emb, reranker=rk, use_bge_sparse=True, tokenize=jieba_tokenize)
        n = p.add_document(pdf, doc_id=f"doc-{num}")
        print(f"    编码 doc-{num}: {n} 块", flush=True)
        pipes.append(p)
    comp = CompositeRAGPipeline(pipes, doc_ids=[f"doc-{n}" for n, _ in docs],
                               query_prep=expand_sections)

    def probe(top_n: int) -> dict[str, list[str]]:
        # ⚠️ 必须清查询缓存：`query()` 命中缓存会**跳过编码+检索+重排**，
        #    不清则第二臂全是缓存命中 ⇒ 必然"完全一致"（假结论）。
        for p in pipes:
            p.top_n = top_n
            p._query_cache.clear()          # noqa: SLF001  探针，允许触内部
            p._query_cache_order.clear()    # noqa: SLF001
        got: dict[str, list[str]] = {}
        for q in QUERIES:
            got[q] = [r.id for r in comp.query(q, top_k=5)]
        return got

    t0 = time.perf_counter()
    a = probe(20)                          # 线上基线
    t_a = time.perf_counter() - t0
    t0 = time.perf_counter()
    b = probe(10)                          # 拟下调
    t_b = time.perf_counter() - t0

    same_seq = [q for q in QUERIES if a[q] == b[q]]
    same_set = [q for q in QUERIES if a[q] != b[q] and set(a[q]) == set(b[q])]
    diff = [q for q in QUERIES if set(a[q]) != set(b[q])]

    rc = run_eval.PipelineFactory(embedder_kind="real", profile="prod")  # 仅取口径常量，不做检索
    del rc
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "docs": [f"doc-{n}" for n, _ in docs],
        "queries": len(QUERIES),
        "top_n20_wall_s": round(t_a, 1),
        "top_n10_wall_s": round(t_b, 1),
        "identical_seq": len(same_seq),
        "identical_set_only": len(same_set),
        "different": len(diff),
        "identical_seq_rate": round(len(same_seq) / len(QUERIES), 4),
        "diff_detail": [
            {"q": q, "top_n20": a[q], "top_n10": b[q]} for q in diff
        ][:10],
        "per_query": {
            q: {"top_n20": a[q], "top_n10": b[q], "same": a[q] == b[q]} for q in QUERIES
        },
    }


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description="rerank 旋钮质量对照探针")
    ap.add_argument("--only", choices=["q1", "q2"], default=None, help="只跑其中一臂")
    ap.add_argument("--docs", type=int, default=6, help="Q2 使用的真实文档篇数")
    args = ap.parse_args()

    fd = _try_lock(_LOCK_PATH)
    if fd is None:
        print("[跳过] 另一个实例正在运行（单实例锁被占用）—— 本环境会重放 turn，"
              "并发跑会让 GPU/内存争用，数字不可信。", flush=True)
        return 0

    report: dict = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    t_all = time.perf_counter()
    try:
        if args.only in (None, "q1"):
            report["q1_max_length"] = run_q1()
        if args.only in (None, "q2"):
            print("\n[Q2] === top_n 20 vs 10（真实文档，判据=输出一致性）===", flush=True)
            report["q2_top_n"] = run_q2(args.docs)
        report["wall_s"] = round(time.perf_counter() - t_all, 1)
        _OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[完成] 报告 → {_OUT}（总耗时 {report['wall_s']}s）", flush=True)
    finally:
        os.close(fd)
        try:
            os.remove(_LOCK_PATH)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
