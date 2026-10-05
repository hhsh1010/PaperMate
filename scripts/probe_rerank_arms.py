"""重排推理加速 · 对照探针（T-03）

**目的**：在**不改生产代码**的前提下，量化重排的三个低成本杠杆各值多少毫秒，
并给出质量侧的一致性证据（打分差 + Top-K 顺序是否变化）。

**为什么单独写这个探针，而不是直接改 `rerank.py`**：
`rerank.py` 在 `provenance.SOURCES` 指纹内 —— 一改就让 `eval/report_real.json` 失效、
门禁转红。而本探针**只读**（自建 embedder + ScoreReranker 取候选，再用独立构造的
CrossEncoder 测各臂），故可**先测量、后决定改什么**（项目既定教义：测量先于优化）。

**方法（与线上同源的部分）**
- 候选来自真实检索：`RAGPipeline._hybrid.retrieve(query, top_n=20)` —— 与 `query()` 完全一致；
- 候选文本来自 `pipe._chunks[idx].text` —— 与 `query()` 的 `texts = [...]` 完全一致；
- 重排在检索之后 ⇒ **换掉重排器不影响候选**，所以本探针的候选与线上逐字节相同。

**arm 设计（v2，2026-09-24 修订）**
| 臂 | max_length | 精度 | 说明 |
|---|---|---|---|
| A0 baseline | None（=tokenizer 的 8192，**今天的行为**） | fp32 | 对照基准 |
| A1 ml512 | 512 | fp32 | 只截断 |
| A2 ml256 | 256 | fp32 | 更激进截断 |
| A3 ml512_fp16 | 512 | fp16（`model.half()`） | 截断 + 半精度 |

**v1 的两个教训（已修，勿重犯）**
1. **长跑必须落盘**：v1 只往 stdout 打印，后台任务记录被清后 47 分钟 GPU 白跑。
   现在同时 `tee` 到 `eval/t03_arms_result.txt`，且 `print(..., flush=True)`。
2. **臂之间必须释放模型**：v1 保留 baseline 模型 + 不 `empty_cache()`，
   实测显存爬到 **7833/8188 MiB**（96%）→ 大概率第三次加载 CUDA OOM 崩掉。
   现在每臂结束 `del + gc.collect() + torch.cuda.empty_cache()`，
   且长度统计改用**独立加载的 AutoTokenizer**（16MB 级），不再为统计而扣着一个大模型。
3. 采样量下调：**臂间对照统一在 batch=32**（= 线上默认），
   批次扫描只对**胜出臂**做（v1 是 288 次 predict，跑到 20 分钟还没完）。

用法：
    .venv/Scripts/python.exe -u scripts/probe_rerank_arms.py                 # 默认：5 query / 20 候选 / 3 轮
    .venv/Scripts/python.exe -u scripts/probe_rerank_arms.py --reps 5
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

import run_eval  # noqa: E402

DEFAULT_TOP_N = 20
DEFAULT_QUERIES = 5

#: 互不相同的查询（避免命中 `_query_cache`）
QUERIES = [
    "本文提出的方法是什么？",
    "系统的整体架构如何设计？",
    "实验中使用了哪些数据集？",
    "与基线方法相比效果提升多少？",
    "该方法的主要局限性是什么？",
    "训练时用了什么优化器与学习率？",
    "消融实验验证了哪些模块？",
    "论文的核心贡献有哪几点？",
]

#: 臂定义：(id, 名称, max_length, fp16)
ARMS: list[tuple[str, str, int | None, bool]] = [
    ("A0", "A0 baseline(8192/fp32)", None, False),
    ("A1", "A1 ml512/fp32", 512, False),
    ("A2", "A2 ml256/fp32", 256, False),
    ("A3", "A3 ml512+fp16", 512, True),
    # 2026-09-24 补测：确认「截断」与「半精度」两个杠杆是否叠加
    ("A4", "A4 ml384+fp16", 384, True),
    ("A5", "A5 ml256+fp16", 256, True),
]

#: 臂间对照固定用线上默认 batch；批次扫描只对胜出臂做
BASE_BATCH = 32
BATCH_SWEEP = (8, 16, 32)

OUT_FILE = _ROOT / "eval" / "t03_arms_result.txt"


def _say(msg: str) -> None:
    """同时打印与写盘 —— 长跑必须落盘（v1 的教训）。"""
    print(msg, flush=True)
    try:
        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:  # noqa: BLE001  落盘失败不应中断测量
        pass


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _release_gpu() -> None:
    """清 CUDA 缓存池（v1 漏了这步，显存爬到 96% 后疑似 OOM 崩）。

    ⚠️ 调用方**必须先自己断引用**（`model = None`）再调本函数 ——
    `del` 只删函数内的局部名，**删不掉调用方持有的那个引用**，
    否则 gc 看到的仍是"有引用"，缓存池也就清不干净（这是 v2 自查发现的自欺点）。
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _build_corpus_pipeline(sparse_on: bool, sizes: int):
    """建一条**不加载 CrossEncoder** 的真实管线（重排在检索之后，故候选不受影响）。"""
    from lightcore.rag.bm25 import _char_tokenize, jieba_tokenize
    from lightcore.rag.embedder import BgeM3Embedder
    from lightcore.rag.pipeline import RAGPipeline
    from lightcore.rag.rerank import ScoreReranker

    try:
        tokenize = jieba_tokenize
    except Exception:  # noqa: BLE001
        tokenize = _char_tokenize
    pipe = RAGPipeline(embedder=BgeM3Embedder(), reranker=ScoreReranker(),
                       use_bge_sparse=sparse_on, tokenize=tokenize)
    pdfs = sorted(run_eval.CORPUS_DIR.glob("*.pdf"))[:sizes]
    for p in pdfs:
        pipe.add_document(p, doc_id=p.stem)
    return pipe, [p.stem for p in pdfs]


def _collect_pairs(pipe, queries: list[str], top_n: int):
    """按线上口径收集 (query, candidate_text) 对。"""
    per_query: list[list[tuple[str, str]]] = []
    counts: list[int] = []
    for q in queries:
        cands = pipe._hybrid.retrieve(q, top_n=top_n)          # noqa: SLF001  与 query() 同源
        texts = [pipe._chunks[i].text for i, _ in cands]        # noqa: SLF001
        per_query.append([(q, t) for t in texts])
        counts.append(len(texts))
    return per_query, counts


def _load_arm(model_name: str, max_length: int | None, fp16: bool, device: str):
    """按臂构造一个 CrossEncoder（max_length/精度是构造参数，故每臂独立加载）。"""
    from sentence_transformers import CrossEncoder

    from lightcore.rag.hf_local import resolve_local_hf_snapshot

    local = resolve_local_hf_snapshot(model_name, env_var="PAPERMATE_RERANKER_PATH")
    t0 = time.perf_counter()
    model = (CrossEncoder(local, device=device, local_files_only=True, max_length=max_length)
             if local else CrossEncoder(model_name, device=device, max_length=max_length))
    if fp16:
        model.model.half()
    _sync()
    load_s = time.perf_counter() - t0
    try:
        eff_ml = model.max_length
    except Exception:  # noqa: BLE001
        eff_ml = max_length
    dtype = str(next(model.model.parameters()).dtype).replace("torch.", "")
    return model, load_s, eff_ml, dtype


def _score(model, pairs: list[tuple[str, str]], batch_size: int) -> tuple[list[float], float]:
    """跑一次 predict，返回 (每个 pair 的分数, 耗时 ms)。

    计时用 `torch.cuda.synchronize()` 夹住：CUDA kernel 是**异步发射**的，
    不同步就计时会把 GPU 耗时系统性低估成"发射耗时"（这会直接毁掉对照实验）。
    """
    import numpy as np

    _sync()
    t0 = time.perf_counter()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0
    return [float(x) for x in np.asarray(scores).reshape(-1)], ms


def _top5(scores: list[float]) -> set[int]:
    return set(sorted(range(len(scores)), key=lambda j: -scores[j])[:5])


def _compare(base_scores: list[list[float]], scores: list[list[float]]) -> tuple[str, str]:
    """与基线逐 query 比：打分 max-abs-diff、Top-5 是否换位。"""
    diffs, swaps = [], 0
    for b, s in zip(base_scores, scores, strict=False):
        if not b or len(b) != len(s):
            continue
        diffs.append(max(abs(x - y) for x, y in zip(b, s, strict=False)))
        if _top5(s) != _top5(b):
            swaps += 1
    diff_s = f"{max(diffs):.2e}" if diffs else "n/a"
    return diff_s, f"{swaps}/{len(scores)}"


def main() -> int:
    ap = argparse.ArgumentParser(description="重排推理加速对照探针（T-03 / v2）")
    ap.add_argument("--sizes", type=int, default=3, help="用前 N 篇 eval 语料（默认 3）")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="候选数（默认 20，与线上 top_n 对齐）")
    ap.add_argument("--queries", type=int, default=DEFAULT_QUERIES, help="用几个不同 query（默认 5）")
    ap.add_argument("--reps", type=int, default=3, help="每臂每 query 重复轮数（默认 3）")
    ap.add_argument("--sparse", choices=list(run_eval.SPARSE_CHOICES), default="auto")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--arms", default="", help="只跑这些臂 id（逗号分隔，如 A4,A5）；A0 始终跑（作对照基准）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("lightcore.rag.pipeline").setLevel(logging.WARNING)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ **追加**而非覆盖：早期版本在这里 write_text("")，
    # 会把上一次运行的结果整份抹掉（A0~A3 的结果差点因此丢失）。改为打印运行横幅分隔。
    sel = [s.strip().upper() for s in args.arms.split(",") if s.strip()]

    from lightcore.rag.hf_local import auto_device

    device = auto_device()
    sparse_on, sparse_src = run_eval._resolve_sparse(args.sparse)
    queries = QUERIES[: args.queries]

    _say("")
    _say("#" * 88)
    _say(f"# 新一次运行 {time.strftime('%Y-%m-%d %H:%M:%S')}"
         f"（picked arms: {','.join(sel) if sel else 'ALL'}）")
    _say("#" * 88)
    _say("=" * 88)
    _say("重排推理加速对照（T-03 / v2）")
    _say(f"  设备：{device}（PAPERMATE_DEVICE={os.environ.get('PAPERMATE_DEVICE', '(未设置)')}）")
    _say(f"  口径：候选 ≤{args.top_n} · 语料前 {args.sizes} 篇 · {len(queries)} 个 query · 每 query {args.reps} 轮")
    _say(f"  use_bge_sparse={sparse_on}（{sparse_src}）")
    _say("=" * 88)

    _say("\n① 建管线（ScoreReranker，**不加载 CrossEncoder**）+ 收集真实候选")
    t0 = time.perf_counter()
    pipe, names = _build_corpus_pipeline(sparse_on, args.sizes)
    _say(f"  语料 {names} → chunks={len(pipe.get_chunks())} · 建管线 {time.perf_counter() - t0:.1f}s")
    per_query, counts = _collect_pairs(pipe, queries, args.top_n)
    _say(f"  候选数分布：{counts}（合计 {sum(counts)} 对）")
    # 候选已收齐（纯字符串）→ 立刻放掉 embedder（≈2.2GB），把显存全留给重排臂。
    # v1 就是"embedder + 两个 reranker 实例"堆到 96% 才崩的。
    pipe = None
    _release_gpu()

    # ---- 长度分布：用独立加载的 tokenizer 算（不再为统计扣着一个 2.2GB 模型）----
    from transformers import AutoTokenizer

    from lightcore.rag.hf_local import resolve_local_hf_snapshot

    snap = resolve_local_hf_snapshot(args.model, env_var="PAPERMATE_RERANKER_PATH")
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True) if snap else AutoTokenizer.from_pretrained(args.model)
    _say("\n② 候选长度分布（决定截断能省多少算力）")
    all_pairs = [p for qs in per_query for p in qs]
    char_lens = [len(t) for _, t in all_pairs]
    tok_lens = [len(tok(t, truncation=False)["input_ids"]) for _, t in all_pairs]
    p = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(len(xs) * q))]  # noqa: E731
    _say(f"  字符长度  p50={statistics.median(char_lens):.0f}  p95={p(char_lens, 0.95)}  max={max(char_lens)}")
    _say(f"  token长度 p50={statistics.median(tok_lens):.0f}  p95={p(tok_lens, 0.95)}  max={max(tok_lens)}  ← 真正进模型的长度")
    for ml in (512, 256):
        over = sum(1 for n in tok_lens if n > ml)
        _say(f"  max_length={ml}：{over}/{len(tok_lens)} 个候选会被截断"
             f"（中位省 {max(0, int(statistics.median(tok_lens)) - ml)}/{int(statistics.median(tok_lens))} token）")

    # ---- ③ 臂间对照（统一 batch=32）----
    _say(f"\n③ 臂间对照（batch={BASE_BATCH} = 线上默认）")
    _say("=" * 88)
    _say(f"{'臂':<24}{'加载s':>7}{'max_len':>9}{'dtype':>7}{'p50 ms':>10}{'p95 ms':>10}{'vs A0':>9}{'打分差max':>12}{'Top5换位':>10}")
    _say("-" * 88)

    base_scores: list[list[float]] = []
    base_p50: float | None = None
    rows: list[dict] = []

    for aid, arm_name, ml, fp16 in ARMS:
        # A0 永远跑（它是本轮的对照基准）；其余臂按 --arms 过滤。
        # —— 同一进程内测出的基准与实验臂才叫"环境对等"，跨进程比会掺入机器状态差。
        if aid != "A0" and sel and aid.upper() not in sel:
            continue
        model = None
        try:
            model, load_s, eff_ml, dtype = _load_arm(args.model, ml, fp16, device)
            lat: list[float] = []
            scores_all: list[list[float]] = []
            for qi, pairs in enumerate(per_query):
                for _ in range(args.reps):
                    sc, ms = _score(model, pairs, BASE_BATCH)
                    lat.append(ms)
                    if len(scores_all) <= qi:
                        scores_all.append(sc)
            lat.sort()
            p50 = statistics.median(lat)
            p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
            if not base_scores:
                base_scores = scores_all
                base_p50 = p50
                diff_s, swap = "—", "—"
            else:
                diff_s, swap = _compare(base_scores, scores_all)
            ratio = "—" if base_p50 is None else f"×{base_p50 / p50:.2f}"
            rows.append({"aid": aid, "arm": arm_name, "p50": p50, "p95": p95, "ratio": ratio,
                         "diff": diff_s, "swap": swap, "lat": lat, "ml": eff_ml,
                         "dtype": dtype, "fp16": fp16})
            _say(f"{arm_name:<24}{load_s:>7.1f}{str(eff_ml):>9}{dtype:>7}{p50:>10.1f}{p95:>10.1f}{ratio:>9}{diff_s:>12}{swap:>10}")
        except Exception as e:  # noqa: BLE001  单臂失败（如 OOM）不应毁掉整轮
            _say(f"{arm_name:<24}  ✗ 失败：{type(e).__name__}: {str(e)[:80]}")
        finally:
            model = None          # 先断引用，否则 _release_gpu 清不掉（见其 docstring）
            _release_gpu()

    # ---- ④ 批次扫描：只对胜出臂（且只有全量跑时才做——补测单臂时跳过，省时间）----
    candidates = [r for r in rows if r["aid"] != "A0"] if rows else []
    winner = min(candidates, key=lambda r: r["p50"]) if candidates else (rows[0] if rows else None)
    if winner and not sel:
        ml = winner["ml"]
        fp16 = winner["fp16"]
        _say(f"\n④ 批次扫描（仅胜出臂：{winner['arm']}）")
        _say(f"{'batch':>7}{'p50 ms':>10}{'p95 ms':>10}{'vs A0':>9}")
        model = None
        try:
            model, _, _, _ = _load_arm(args.model, ml, fp16, device)
            for bs in BATCH_SWEEP:
                lat = []
                for pairs in per_query:
                    for _ in range(args.reps):
                        _, ms = _score(model, pairs, bs)
                        lat.append(ms)
                lat.sort()
                p50 = statistics.median(lat)
                p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
                ratio = "—" if base_p50 is None else f"×{base_p50 / p50:.2f}"
                _say(f"{bs:>7}{p50:>10.1f}{p95:>10.1f}{ratio:>9}")
        except Exception as e:  # noqa: BLE001
            _say(f"  ✗ 批次扫描失败：{type(e).__name__}: {str(e)[:80]}")
        finally:
            model = None
            _release_gpu()

    # ---- ⑤ 结论 ----
    _say("\n⑤ 结论")
    if base_p50 is not None:
        _say(f"  A0 基线 p50 = {base_p50:.1f} ms / 平均候选 {statistics.fmean(counts):.1f} 个"
             f" → 单候选 ≈ {base_p50 / max(1, statistics.fmean(counts)):.1f} ms")
    if winner:
        _say(f"  ★ 最快臂：{winner['arm']} p50={winner['p50']:.1f} ms"
             f"（vs A0 {winner['ratio']}）· 打分差 max={winner['diff']} · Top5 换位 {winner['swap']}")
    _say(f"\n结果已落盘：{OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
