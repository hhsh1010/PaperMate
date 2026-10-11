"""消融/分辨力诊断：读若干 `eval/report_*.json`，打印**对比表 + 分辨力诊断**。

## 为什么需要它
本项目踩过的坑：**指标饱和会让消融"测不出差异"**（平均 gold 仅 1.2 篇而 k=5 ⇒ recall@5 贴天花板）。
所以跑消融之前/之后都必须回答一句：**这些指标有分辨力吗？**
本脚本把这件事变成一条命令 —— 输出每个指标的**跨配置极差**与**跨用例类型极差**，
极差 ≈ 0 的指标直接标 `[饱和]`，避免拿它当判据。

## 用法
    python scripts/ablation_report.py eval/report_ablation_k1.json eval/report_ablation_k3.json ...
    python scripts/ablation_report.py --kind report_ablation_k5.json     # 只打按 kind 的分组
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: 越大越好的指标（latency_ms 越小越好，单独处理）
METRICS = ("recall@5", "precision@5", "ndcg@5", "hit@5", "map", "mrr@10")


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metrics_of(rep: dict) -> dict:
    """兼容两种报告形状：`metrics` 字典，或平铺在顶层。"""
    m = dict(rep.get("metrics") or {})
    for k in ("recall@5", "precision@5", "ndcg@5", "hit@5", "map", "mrr@10", "latency_ms"):
        if k in rep and k not in m:
            m[k] = rep[k]
    return m


def _label(rep: dict) -> str:
    """一行的名字：优先用命令行锚点（k / 旋钮），便于读表。"""
    bits = []
    if rep.get("k") is not None:
        bits.append(f"k={rep['k']}")
    if rep.get("use_bge_sparse") is not None:
        bits.append("sparse=" + ("on" if rep["use_bge_sparse"] else "off"))
    if rep.get("rag_reranker_max_length"):
        bits.append(f"rerank={rep['rag_reranker_max_length']}")
    if rep.get("merge_strategy"):
        bits.append(rep["merge_strategy"])
    if rep.get("query_rewrite_mode"):
        bits.append(f"qr={rep['query_rewrite_mode']}")
    return " · ".join(bits) or Path(rep.get("__path__", "?")).name


def print_spread(rep: dict) -> None:
    """分辨力诊断 —— **直接采用框架自带的判据**（不另造阈值）。

    ★ 2026-10-11 核实时发现：报告里 `metrics_by_kind[kind]` 已经带了
    `discriminative.has_headroom`（哪些指标还有空间）与 `headroom`（还剩多少），
    以及 `precision_cap.saturation`（精度是否被 k 与 gold 数夹死）。
    ⇒ 评测框架早就把"这个用例集能测出什么"算好了；消融只需要**读它的结论**，
      而不是自己写一个 `spread < eps` 的土判据（那是"两把尺子"）。
    """
    by_kind = rep.get("metrics_by_kind") or {}
    if len(by_kind) < 2:
        print("\n（报告里分组不足，跳过分辨力诊断）")
        return

    print(f"\n### 分辨力诊断（框架自带判据 · n={rep.get('num_cases')}）")
    print("| 用例类型 | n | 有无空间（has_headroom）| 剩余空间 headroom | 精度饱和 |")
    print("|---|---|---|---|---|")
    for kind, blk in by_kind.items():
        disc = blk.get("discriminative") or {}
        head = disc.get("has_headroom") or {}
        room = disc.get("headroom") or {}
        has = "、".join(k for k, v in head.items() if v) or "（全部无空间）"
        room_s = "、".join(f"{k} {v:.4f}" for k, v in room.items() if (head.get(k) and isinstance(v, (int, float)))) or "—"
        sat = (blk.get("precision_cap") or {}).get("saturation")
        print(f"| {kind} | {blk.get('n', '—')} | {has} | {room_s} | {'—' if sat is None else f'{sat:.2f}'} |")
    print("\n> 判定口径：**只有 `has_headroom=true` 的指标才能当消融判据**；`false` 的指标在本用例集上恒满分，")
    print("> 拿它对比只会得到「没差异」的假结论。")

    # 汇总：哪些指标在**所有**用例类型上都无空间 ⇒ 整体不可用
    unusable: list[str] = []
    all_metrics: set[str] = set()
    for blk in by_kind.values():
        all_metrics |= set((blk.get("discriminative") or {}).get("has_headroom") or {})
    for metric in sorted(all_metrics):
        if all(not ((blk.get("discriminative") or {}).get("has_headroom") or {}).get(metric) for blk in by_kind.values()):
            unusable.append(metric)
    if unusable:
        print(f"\n**本用例集上不可作为判据的指标**：`{'`、`'.join(unusable)}` —— 全类型无空间，建议改用 mrr@10 / map / ndcg@5 或收紧 `--k`。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="消融报告对比 + 分辨力诊断")
    ap.add_argument("reports", nargs="+", help="一个或多个 eval/report_*.json")
    ap.add_argument("--kind", action="store_true", help="只打按用例类型的分组表")
    args = ap.parse_args(argv)

    reps = []
    for p in args.reports:
        rep = load(p)
        rep["__path__"] = p
        reps.append(rep)

    if not args.kind:
        print("## 配置对比（整体）\n")
        print("| 配置 | " + " | ".join(METRICS) + " | latency_ms |")
        print("|---" * (len(METRICS) + 2) + "|")
        for rep in reps:
            m = _metrics_of(rep)
            cells = [("—" if m.get(kk) is None else f"{m[kk]:.4f}") for kk in METRICS]
            lat = m.get("latency_ms")
            print(f"| {_label(rep)} | " + " | ".join(cells) + f" | {'—' if lat is None else f'{lat:.1f}'} |")

        # ★ 相对基线的 Δ 表 —— 这才是"成本-收益权衡表"的本体：
        #   绝对指标只能说明"现在多好"，**差值**才能回答"这个组件值不值它的代价"。
        if len(reps) > 1:
            base = _metrics_of(reps[0])
            print(f"\n## 相对基线（第一行：{_label(reps[0])}）的边际收益\n")
            cols = ("map", "mrr@10", "ndcg@5", "recall@5")
            print("| 配置 | " + " | ".join(f"Δ{c}" for c in cols) + " | Δlatency_ms |")
            print("|---" * (len(cols) + 2) + "|")
            for rep in reps[1:]:
                m = _metrics_of(rep)
                cells = []
                for c in cols:
                    b, v = base.get(c), m.get(c)
                    if not isinstance(b, (int, float)) or not isinstance(v, (int, float)):
                        cells.append("—")
                    else:
                        d = v - b
                        cells.append(f"{d:+.4f}" + (" ✅" if d > 0.0005 else (" ❌" if d < -0.0005 else " ＝")))
                bl, vl = base.get("latency_ms"), m.get("latency_ms")
                lat_d = "—" if not (isinstance(bl, (int, float)) and isinstance(vl, (int, float))) else f"{vl - bl:+.1f}"
                print(f"| {_label(rep)} | " + " | ".join(cells) + f" | {lat_d} |")
            print("\n> 读法：`✅` 优于基线、`❌` 劣于基线、`＝` 无实质差异（阈值 ±0.0005）。")
            print("> ⚠️ 延迟的 run 间噪声约 ±10%（实测），**跨 run 比较延迟要看趋势而非单点**。")

    for rep in reps:
        print(f"\n### 按用例类型：{_label(rep)}")
        by_kind = rep.get("metrics_by_kind") or {}
        if not by_kind:
            print("（无分组数据）")
            continue
        keys = list(by_kind.keys())
        print("| 指标 | " + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for metric in METRICS + ("latency_ms",):
            row = []
            for kk in keys:
                # ⚠️ 分组里的指标**嵌在 `metrics` 下**（顶层却是平铺的 —— 结构不对称，踩过一次）
                v = (by_kind[kk].get("metrics") or {}).get(metric)
                row.append("—" if v is None else (f"{v:.1f}" if metric == "latency_ms" else f"{v:.4f}"))
            if any(c != "—" for c in row):
                print(f"| {metric} | " + " | ".join(row) + " |")

    print_spread(reps[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
