"""检索阶段剖析探针（T-00 步 3~4）。

**要回答的问题**：项目内检索延迟存在 **100× 跨度** ——
`eval/report_real.json` 中位 **10541ms**（逐例 8073~14505，分布很紧、非冷启），
而本机补测小语料单篇仅 **79ms**（sparse 关）/ **125ms**（sparse 开）。
既有观测只有一个"总数"，无法判断时间花在查询编码、三路召回还是重排上。

**做法**：复用 `eval.PipelineFactory`（**与评测完全相同的建管线路径**，含 prod 口径与单例缓存），
按语料规模（1 / 2 / 3 篇）逐档测量 `RAGPipeline.last_stage_ms` 的各阶段耗时。

**三条纪律（否则数据无效）**

1. **必须绕开 `RAGPipeline._query_cache`（FIFO 32）**：同 query 二次调用会命中缓存返回 ~0ms。
   故每轮用**不同 query**，并**断言** `last_stage_ms` 不含 `cache_hit` —— 一旦命中就判该轮无效。
   （2026-09-23 踩过：预热调用污染后续测量，全部读成 0ms。）
2. **口径与线上同源**：用 `PipelineFactory("real", "prod")` + `--sparse`（默认 auto 读运行时设置）。
3. **报告分位数而非单点**：单次测量易受抖动影响，故每档测多轮，报 p50 / min / max。

用法::

    .venv/Scripts/python.exe scripts/probe_rag_stages.py                 # 默认 3 篇 eval 语料
    .venv/Scripts/python.exe scripts/probe_rag_stages.py --sparse off    # 双臂对照的关臂
    .venv/Scripts/python.exe scripts/probe_rag_stages.py --sizes 1 3
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

#: ★ 必须在 import lightcore **之前**设置：`_TRACE` 是模块级常量（import 时求值）。
#:   若放在 import 之后，打点不会开启，且 `last_stage_ms` 恒为空 —— 而脚本会"正常"跑完。
os.environ.setdefault("PAPERMATE_RAG_TRACE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402

from lightcore.rag.pipeline import _TRACE  # noqa: E402

#: 每档用的查询（**互不相同**，避免命中缓存）；语义覆盖"方法/实验/结论"等常见问法
QUERIES = [
    "本文提出的方法是什么？",
    "实验设置和对比基线有哪些？",
    "主要贡献有哪几点？",
    "训练用了什么数据集？",
    "结论部分说了什么？",
    "与现有方法相比优在哪里？",
    "有哪些局限性？",
    "评价指标是怎么定义的？",
    "模型结构包含哪些模块？",
]

#: 规模分桶（按 **chunk 数**）—— 检索成本取决于检索单元数，而非文档数
BUCKETS = ((0, 100, "<100"), (100, 300, "100~300"), (300, 10**9, ">300"))


def _bucket(n: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= n < hi:
            return name
    return "?"


def measure(pipe, queries: list[str], top_k: int = 10) -> dict[str, list[float]]:
    """逐 query 测量，返回 `{阶段: [耗时 ms, ...]}`。

    ⚠️ 命中缓存的轮次**直接判无效并报错**，而不是静默丢弃 ——
    静默丢弃会让人以为"样本就是少"，而真实原因可能是 query 池被复用。
    """
    out: dict[str, list[float]] = {}
    for q in queries:
        pipe.query(q, top_k=top_k)
        ms = dict(pipe.last_stage_ms)
        if "cache_hit" in ms:
            raise AssertionError(
                f"查询命中缓存，数据无效（q={q!r}）—— 检查 QUERIES 是否重复或该 query 已提交过"
            )
        if not ms:
            raise AssertionError(
                "last_stage_ms 为空 → 打点未开启。确认 PAPERMATE_RAG_TRACE 在 import 前已设置"
            )
        for k, v in ms.items():
            out.setdefault(k, []).append(v)
    return out


def _p50(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def _unattributed(r: dict) -> float:
    """未单独打点的部分（含 RRF 融合）= **逐次**算 `total − Σ阶段` 后取中位数。

    ⚠️ **不能**写成"各阶段 p50 之和"再与 total 的 p50 相减 ——
    中位数不满足可加性，那样算会得出**虚假的负值**（第一版即如此：
    标签写着"负数说明存在阶段重叠"，其实只是统计口径错，本次实测 −8.3ms ≈ 0.5%）。
    逐次相减再取中位数才是正确口径。
    """
    st = r["stages"]
    totals = st.get("total", [])
    if not totals:
        return 0.0
    per_query = []
    for i, total in enumerate(totals):
        known = sum(v[i] for k, v in st.items() if k != "total" and len(v) > i)
        per_query.append(total - known)
    return statistics.median(per_query)


def main() -> int:
    ap = argparse.ArgumentParser(description="检索阶段剖析探针（T-00）")
    ap.add_argument("--sparse", choices=list(run_eval.SPARSE_CHOICES), default="auto",
                    help="稀疏向量开关（同 run_eval 语义；auto=读运行时设置）")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3],
                    help="按前 N 篇语料逐档测量（默认 1 2 3）")
    ap.add_argument("--top-k", type=int, default=10,
                    help="与评测一致取 max(k, _MRR_K)=10（见 run_eval.run_case）")
    ap.add_argument("--synth", type=int, default=0,
                    help="额外跑一档**合成语料**：把 eval 语料的真实文本重复拼接至约 N 个 chunk。"
                         "用于把规模拉到评测语料达不到的量级 —— eval/corpus 实测只有 2~4 个 chunk，"
                         "仅凭真实语料无法刻画「成本随规模增长」。0=不跑（默认）")
    args = ap.parse_args()

    if not _TRACE:
        print("✗ 打点未开启（_TRACE=False）—— 请确认环境变量在 import 前设置")
        return 2

    # 静音第三方噪声与本模块的逐次 trace 行；阶段数据从 `last_stage_ms` 读取（与日志同源）
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("lightcore.rag.pipeline").setLevel(logging.WARNING)

    sparse_on, sparse_src = run_eval._resolve_sparse(args.sparse)
    # 设备必须显式打印：本轮要回答"100× 跨度是否来自设备差异"，
    # 若日志里没有设备信息，两组数据就**无法互相归因**（这是最容易被忽略的对照条件）。
    try:
        from lightcore.rag.hf_local import auto_device

        device = auto_device()
    except Exception as e:  # noqa: BLE001  设备探测失败不影响测量，但要说明
        device = f"<探测失败: {type(e).__name__}>"
    print("=" * 78)
    print("检索阶段剖析（T-00）")
    print(f"  设备：{device}（PAPERMATE_DEVICE={os.environ.get('PAPERMATE_DEVICE', '(未设置)')}）")
    print(f"  口径：profile=prod（CrossEncoder + jieba） · use_bge_sparse={sparse_on}（{sparse_src}）")
    print(f"  top_k={args.top_k}（与评测一致） · 每档测 {len(QUERIES)} 个不同 query")
    print("=" * 78)

    pdfs = sorted(run_eval.CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print("✗ eval/corpus 下无 PDF 语料")
        return 2
    names = [p.stem for p in pdfs]
    print(f"语料：{names}")

    print("\n① 建管线（含 PDF 解析 + 分块 + 编码）")
    factory = run_eval.PipelineFactory("real", "prod",
                                       use_bge_sparse=sparse_on, sparse_source=sparse_src)
    rows: list[dict] = []
    for n in args.sizes:
        if n > len(names):
            print(f"  跳过 size={n}（语料只有 {len(names)} 篇）")
            continue
        corpus = names[:n]
        t0 = time.perf_counter()
        pipe = factory.for_case({"corpus": corpus})
        build_s = time.perf_counter() - t0
        n_chunks = len(pipe.get_chunks())
        print(f"  {n} 篇 {corpus} → chunks={n_chunks} · 建管线 {build_s:.1f}s"
              f"（含解析/分块/编码；**不计入** query 耗时）")
        rows.append({"n": n, "corpus": corpus, "chunks": n_chunks, "build_s": build_s, "pipe": pipe})

    if args.synth:
        # 合成语料：把 eval 语料的**真实文本**重复拼接 ——
        # 目的是把 chunk 数拉到评测语料达不到的量级（实测 eval/corpus 仅 2~4 chunk），
        # 从而观察「编码 / 召回 / 重排」各自如何随规模变化。
        # 走 `for_case({"docs": ...})` 内联用例路径（复用 factory 的单例模型，省显存），
        # 且该路径**每次新建管线**、不缓存 —— 正合"要一份干净的规模档"之意。
        base_texts = [c.text for c in rows[-1]["pipe"].get_chunks()] if rows else []
        if not base_texts:
            print("  ⚠️ 无基础文本，跳过 --synth")
        else:
            k = max(1, args.synth // len(base_texts))
            synth_texts = base_texts * k
            t0 = time.perf_counter()
            pipe = factory.for_case({"docs": {"synth": synth_texts}})
            build_s = time.perf_counter() - t0
            n_chunks = len(pipe.get_chunks())
            print(f"  合成语料（真实文本 ×{k}）→ chunks={n_chunks} · 建管线 {build_s:.1f}s")
            rows.append({"n": f"synth×{k}", "corpus": ["synth"], "chunks": n_chunks,
                         "build_s": build_s, "pipe": pipe})

    print("\n② 逐档测量 query 阶段耗时")
    all_stages: set[str] = set()
    for r in rows:
        stages = measure(r["pipe"], QUERIES, top_k=args.top_k)
        r["stages"] = stages
        all_stages |= set(stages.keys())

    #: 各阶段按稳定顺序展示（total 最后）
    order = [k for k in ("encode_dense", "encode_sparse", "ann_search", "sparse_search",
                         "bm25_search", "rerank", "total") if k in all_stages]
    others = sorted(all_stages - set(order))

    print("\n" + "=" * 78)
    print("结果（p50，单位 ms）")
    print("=" * 78)
    head = f"{'篇数':>4} {'chunks':>7} {'桶':>9} " + " ".join(f"{k:>14}" for k in order)
    print(head)
    for r in rows:
        cells = " ".join(f"{_p50(r['stages'].get(k, [])):>14.1f}" for k in order)
        print(f"{r['n']:>4} {r['chunks']:>7} {_bucket(r['chunks']):>9} {cells}")

    print("\n③ 阶段占比（占 total 的百分比，p50 口径）")
    for r in rows:
        total = _p50(r["stages"].get("total", []))
        if total <= 0:
            continue
        parts = " · ".join(
            f"{k}={100 * _p50(r['stages'].get(k, [])) / total:.1f}%"
            for k in order if k != "total"
        )
        print(f"  {r['n']} 篇（chunks={r['chunks']}，total={total:.1f}ms）：{parts}")

    print("\n④ 未单独打点部分（含 RRF 融合；逐次算 total−Σ阶段 后取中位数）")
    for r in rows:
        v = _unattributed(r)
        flag = ""
        if v < -1:
            flag = "  ← 负值：若绝对值很小属计时噪声；显著为负才说明打点位置有重叠"
        print(f"  {r['n']} 篇：{v:+.1f}ms{flag}")

    if others:
        print(f"\n  附带阶段：{others}")

    print("\n⑤ 离散度（同档内 min/max，判断是否受抖动主导）")
    for r in rows:
        t = r["stages"].get("total", [])
        if len(t) > 1:
            print(f"  {r['n']} 篇 total：min={min(t):.1f} · p50={statistics.median(t):.1f} · "
                  f"max={max(t):.1f} · 极差/中位={((max(t) - min(t)) / statistics.median(t)):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
