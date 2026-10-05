"""T-07 探针 · 真实臂（H1 + H2）：需要 GPU + BGE-M3（约 1~3 分钟，单进程单线程，零 API 费用）。

## H1 —— `encode_dense` 的真实延迟（决定"命中 ≤20ms"是否可达）
语义缓存命中**必须先算查询向量**才能比余弦 ⇒ 其延迟**下界 = 一次查询编码**。
本臂用 `PAPERMATE_RAG_TRACE=1` 打开阶段打点（`pipeline.last_stage_ms`），
逐次打印 `encode_dense / encode_sparse / ann_search / sparse_search / bm25_search / rerank / total`。

**线上同源**：`use_bge_sparse=True`（线上实配）+ `jieba_tokenize` + `CrossEncoderReranker(max_length=384, fp16=True)`
（T-03 落地档位）。首次调用含懒加载 ⇒ 先做一次 warm-up，再取 6 个**互不相同**的 query
（不同 query 才不会命中精确缓存）。

## H2 —— 0.95 阈值的通过率（决定"命中率 ≥35%"是否可达）
**不靠我手写同义句**（那是我构造的、可能有偏），而是**从真实评测集里客观抽取**：
`eval/cases/golden_real.jsonl` 中 **`gold` 有交集**的两个 query ⇒ 指向**同一段证据** ⇒ 记作"同目标对"；
`gold` **不相交**的随机对 ⇒ 记作"对照对"。
⇒ 看两组的余弦分布**是否可分**，以及 0.85/0.90/0.95 三档各自的通过率。

⚠️ 口径诚实说明：`gold` 相同 ⇒ "**同一段证据**"，**不完全等价于"同一意图"**
（同一段话可以被问成"是多少"或"为什么"）。故此臂给的是**上界性质的参考**，
真命中率仍须真实流量观察（报告里会写明）。

## 跑法
    .venv/Scripts/python.exe scripts/probe_t07_real.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ★ 必须在 import pipeline **之前**设置（`_TRACE` 是模块级常量）
os.environ["PAPERMATE_RAG_TRACE"] = "1"

from lightcore.rag.bm25 import jieba_tokenize  # noqa: E402
from lightcore.rag.embedder import BgeM3Embedder  # noqa: E402
from lightcore.rag.pipeline import RAGPipeline  # noqa: E402

CASES = ROOT / "eval" / "cases" / "golden_real.jsonl"
CHUNK = ("论文指出，系统的检索链路由稠密向量与稀疏词法两路并行构成，"
         "再经交叉编码器重排后取前若干条进入提示词。分块大小取三百至五百字，"
         "既保证语义完整，又避免单块过长稀释相关性。索引变更时查询缓存整体失效。")

QUERIES_H1 = [
    "分块大小是怎么定的",
    "BM25 中文怎么分词",
    "重排器最多处理多少 token",
    "查询缓存什么时候失效",
    "稀疏向量是哪来的",
    "检索链路分几路",
]


def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False)) / (na * nb)


def h1_pipeline_stages() -> None:
    print("=" * 78)
    print("H1 · 真实检索的阶段耗时（线上同源：bge_sparse + jieba + CrossEncoder(fp16,384)）")
    print("=" * 78)

    t0 = time.perf_counter()
    emb = BgeM3Embedder()
    print(f"[load] BgeM3Embedder 就绪 device={emb.device} dim={emb.dim}  "
          f"耗时 {time.perf_counter() - t0:.1f}s")

    rr = None
    try:
        from lightcore.rag.rerank import CrossEncoderReranker

        t1 = time.perf_counter()
        rr = CrossEncoderReranker(max_length=384, fp16=True)
        print(f"[load] CrossEncoderReranker 就绪 耗时 {time.perf_counter() - t1:.1f}s")
    except Exception as e:  # noqa: BLE001
        print(f"[load] ⚠️ CrossEncoder 加载失败，退化为 ScoreReranker：{e}")

    p = RAGPipeline(embedder=emb, reranker=rr, use_cross_encoder=False,
                    use_bge_sparse=True, tokenize=jieba_tokenize)
    items = [(f"doc-probe-c{i}", CHUNK, i + 1) for i in range(12)]
    p.add_chunks(items, doc_id="doc-probe")
    print(f"[index] {len(items)} chunk（每块 {len(CHUNK)} 字）")

    p.query("warm-up：让懒加载与模型预热完成", top_k=6)   # warm-up（不计入统计）

    print(f"\n{'query':<22}{'encode_dense':>13}{'encode_sparse':>14}{'ann':>8}{'sparse':>8}{'bm25':>8}{'rerank':>9}{'total':>9}")
    rows = []
    for q in QUERIES_H1:
        p.query(q, top_k=6)
        ms = dict(p.last_stage_ms)
        rows.append(ms)
        print(f"{q:<22}{ms.get('encode_dense', 0):>13.1f}{ms.get('encode_sparse', 0):>14.1f}"
              f"{ms.get('ann_search', 0):>8.1f}{ms.get('sparse_search', 0):>8.1f}"
              f"{ms.get('bm25_search', 0):>8.1f}{ms.get('rerank', 0):>9.1f}{ms.get('total', 0):>9.1f}")

    enc = [r.get("encode_dense", 0.0) for r in rows]
    tot = [r.get("total", 0.0) for r in rows]
    print(f"\n[H1 汇总] encode_dense: min={min(enc):.1f} p50={statistics.median(enc):.1f} "
          f"max={max(enc):.1f} ms   （n={len(enc)}，warm-up 已剔除）")
    print(f"          retrieve 端到端 total: p50={statistics.median(tot):.1f} max={max(tot):.1f} ms")
    print(f"          ★ 语义缓存命中延迟的下界 = encode_dense（p50 {statistics.median(enc):.1f} ms）")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"[gpu] cuda 峰值显存 = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    except Exception:  # noqa: BLE001
        pass
    return p


def h2_threshold_study(emb: BgeM3Embedder) -> None:
    print("\n" + "=" * 78)
    print("H2 · 阈值通过率（同目标对 = gold 有交集；对照对 = gold 不相交）")
    print("=" * 78)

    rows = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
    uniq = sorted({r["query"] for r in rows})
    print(f"[data] 用例 {len(rows)} 条 · 去重 query {len(uniq)} 条")

    t0 = time.perf_counter()
    vecs: dict[str, list[float]] = {}
    B = 32
    for i in range(0, len(uniq), B):
        batch = uniq[i:i + B]
        for t, v in zip(batch, emb.encode(batch), strict=False):
            vecs[t] = v
    print(f"[encode] {len(uniq)} 条 query 编码耗时 {time.perf_counter() - t0:.1f}s")

    gold = {r["query"]: set(r.get("gold") or []) for r in rows}
    same, ctrl = [], []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            ga, gb = gold.get(a, set()), gold.get(b, set())
            if not ga or not gb:
                continue
            c = _cos(vecs[a], vecs[b])
            (same if ga & gb else ctrl).append(c)
    ctrl.sort()
    step = max(1, len(ctrl) // max(1, len(same)))
    ctrl = ctrl[::step][:len(same)]

    def stats(xs: list[float]) -> str:
        if not xs:
            return "（无）"
        return (f"n={len(xs)} min={min(xs):.3f} p50={statistics.median(xs):.3f} "
                f"p90={sorted(xs)[int(len(xs) * 0.9)]:.3f} max={max(xs):.3f}")

    print(f"\n[同目标对] {stats(same)}")
    print(f"[对照  对] {stats(ctrl)}")
    print(f"\n{'阈值':<8}{'同目标通过率':>14}{'对照通过率':>13}")
    for t in (0.85, 0.90, 0.95):
        ps = sum(1 for c in same if c >= t) / max(1, len(same))
        pc = sum(1 for c in ctrl if c >= t) / max(1, len(ctrl))
        print(f"{t:<8.2f}{ps:>13.1%}{pc:>13.1%}")
    print("\n★ 判读：'对照通过率' 就是**错命中风险**的直接量化（越高越危险）；"
          "\n  '同目标通过率' 是命中率的上界参考（同 gold ≠ 同意图）。")
    print("\n[gpu] 真实编码路径跑通，max_length=384 fp16；如需复算请重跑本脚本。")
    print(json.dumps({"same_p50": statistics.median(same) if same else None,
                      "same_n": len(same), "ctrl_n": len(ctrl),
                      "ctrl_max": max(ctrl) if ctrl else None}, ensure_ascii=False))


def main() -> None:
    emb_holder = h1_pipeline_stages()
    h2_threshold_study(emb_holder.embedder)


if __name__ == "__main__":
    main()
