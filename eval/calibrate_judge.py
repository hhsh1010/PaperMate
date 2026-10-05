"""judge 与**人工标注**的一致性校准（Cohen's κ）—— T-02-D。

## 为什么必须有人工标注（不能全自动）

`run_eval --judge` 跑出 `faithfulness = 1.0000` 时，有两种解释：**答案真的干净**，或**判官宽松**。
`scripts/probe_judge_discrimination.py` 能排除"判官对什么都放行"（受控负样本），
但它只用 3 条合成样本 —— **判官在真实分布上的准确率仍然未知**。
要知道判官能不能当"自动裁判"，就必须拿**人**的判定跟它比：这正是 κ 的作用。

⚠️ 本项目约束（卡片原文）：**不允许全自动生成标注后直接当 gold** —— 那样 "判官 vs AI 标注"
的一致性会因为两边同源而虚高。若用 AI 预标注，必须在结论里显式标注"非人工，仅为下界"。

## 两个模式

    # ① 生成标注表（从 export_judge_annotation.py 的 dump 或 run_eval 的报告）
    .venv/Scripts/python.exe eval/calibrate_judge.py --dump eval/judge_annotation.jsonl \
        --make-template eval/judge_annotation.csv

    # ② 填完 human 列后算一致性
    .venv/Scripts/python.exe eval/calibrate_judge.py --annotations eval/judge_annotation.csv

## human 列怎么填

- `true`  = 判官的结论正确（该忠实的判忠实 / 该拒答的判拒答）
- `false` = 判官结论错误
- 留空   = 未标注（**不计入 κ**，但会在覆盖率里显示）

⚠️ 这里比的是「**判官对不对**」，不是「答案好不好」—— 若你想改判的是答案，那就是另一件事
（那需要重建 gold，不属于本脚本范围）。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

_TRUE = {"true", "1", "t", "yes", "y", "是", "对", "通过"}
_FALSE = {"false", "0", "f", "no", "n", "否", "错", "不通过"}

#: κ 的常用解读区间（Landis & Koch 1977）。**只作参考**：本任务 n 很小，区间会很宽。
_KAPPA_BANDS = [
    (0.81, 1.01, "almost perfect 几乎完全一致"),
    (0.61, 0.81, "substantial 高度一致"),
    (0.41, 0.61, "moderate 中等一致"),
    (0.21, 0.41, "fair 一般"),
    (-0.01, 0.21, "slight 轻微"),
    (-1.01, -0.01, "poor 差于随机"),
]


def _parse_bool(raw: str | None) -> bool | None:
    """宽容解析标注值；**无法识别一律返回 None（=未标注）**，绝不猜成 True。"""
    if raw is None:
        return None
    t = raw.strip().lower()
    if not t:
        return None
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    return None


def load_annotations(path: Path) -> list[dict]:
    """读标注表。**编码自动兼容**：UTF-8(BOM) → GB18030 → UTF-8(容错)。

    ★ 为什么必须容错（2026-09-25 实测）：标注表多半用 **Excel 打开填写**，而中文 Windows 的 Excel
    默认按本地 ANSI 码页另存 —— 用户第一次交回的 `judge_annotation.csv` 就是 **GB18030**，
    原实现只按 utf-8-sig 读，直接 `UnicodeDecodeError` 崩掉，把"用户已交作业"误报成"文件坏了"。
    """
    raw = path.read_bytes()
    text, used = None, ""
    for enc in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            text, used = raw.decode(enc), enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:                       # utf-8 只对非法字节容错（保底不崩）
        text, used = raw.decode("utf-8", errors="replace"), "utf-8(容错)"
    if used not in ("utf-8-sig", "utf-8"):
        print(f"ℹ️ 标注表编码 = {used}（Excel 另存常见），已自动兼容")
    rows = list(csv.DictReader(io.StringIO(text)))
    out: list[dict] = []
    for r in rows:
        judge = _parse_bool(r.get("judge"))
        human = _parse_bool(r.get("human"))
        out.append({
            "id": r.get("id", ""), "kind": (r.get("kind") or "").strip(),
            "judge": judge, "human": human,
            "note": (r.get("human_note") or "").strip(),
        })
    return out


def binom_upper(k: int, n: int, conf: float = 0.95) -> float | None:
    """观察 `k/n` 次错误时，**真实错误率的单侧置信上界**（Clopper–Pearson，二分求根）。

    为什么需要它：当判官与人工**全一致且都只取一种取值**时，κ 数学上无定义（`p_e = 1`），
    但「**0/40 观察到判官错误**」仍然是一条**有信息**的结论 —— 它给出错误率的上界（≈7.2%），
    比"κ 算不出来"有用得多。**没有 κ 不等于没有结论。**
    """
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    alpha = 1.0 - conf
    lo = k / n
    hi = 1.0
    for _ in range(200):                   # 二分：P(X≤k|p) 关于 p 单调递减
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(k + 1))
        if cdf > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> dict:
    """两分类 Cohen's κ。`pairs` 已经是 (judge, human) 且两者都非 None。

    κ = (p_o − p_e) / (1 − p_e)；`p_e` 用两侧的边缘分布估计。
    ⚠️ 若 `p_e == 1`（两边都只有一种取值）κ **无定义** → 返回 None 并说明，
    不许拿 0 或 1 冒充（这正是"数字好看但没有信息"的典型）。
    """
    n = len(pairs)
    if n == 0:
        return {"n": 0, "kappa": None, "note": "无可比对样本"}
    a = sum(1 for j, h in pairs if j and h)          # 都判 true
    b = sum(1 for j, h in pairs if j and not h)      # 判官 true、人工 false
    c = sum(1 for j, h in pairs if not j and h)      # 判官 false、人工 true
    d = sum(1 for j, h in pairs if not j and not h)  # 都判 false
    po = (a + d) / n
    p_j_true, p_h_true = (a + b) / n, (a + c) / n
    pe = p_j_true * p_h_true + (1 - p_j_true) * (1 - p_h_true)
    if abs(1 - pe) < 1e-12:
        return {"n": n, "a": a, "b": b, "c": c, "d": d, "po": round(po, 4), "pe": round(pe, 4),
                "kappa": None, "note": "两侧取值都只有一种 → κ 无定义（无法区分一致与巧合）"}
    kappa = (po - pe) / (1 - pe)
    band = next((name for lo, hi, name in _KAPPA_BANDS if lo <= kappa < hi), "未定义")
    return {"n": n, "a": a, "b": b, "c": c, "d": d,
            "po": round(po, 4), "pe": round(pe, 4), "kappa": round(kappa, 4), "band": band}


def _dump_rows(dump: Path) -> list[dict]:
    rows: list[dict] = []
    for line in open(dump, encoding="utf-8"):
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def make_template_from_dump(dump: Path, out: Path) -> int:
    """从 `export_judge_annotation.py` 的 dump 生成标注表（含**完整上下文**与答案）。

    ★ 为什么 CSV 里必须带 `context` 正文（2026-09-25 修）：
    第一版只放了 `context_chars`（长度），而人工要判的正是"**答案是否忠于这些证据**" ——
    **没有上下文就没法判**，表等于不可用。CSV 里带长文本确实不好读，所以同时提供
    `make_annotation_sheet()` 输出 Markdown 阅读版（同一份数据、给人看的排版）。
    """
    rows: list[dict] = []
    for d in _dump_rows(dump):
        rows.append({
            "id": d.get("id", ""),
            "kind": d.get("kind", ""),
            "query": d.get("query", ""),
            "context": d.get("context") or "",          # ← 人工判定必需
            "answer": (d.get("answer") or ""),
            "judge": "" if d.get("judge_value") is None else str(bool(d["judge_value"])).lower(),
            "judge_reason": (d.get("judge_reason") or "").replace("\n", " "),
            "context_chars": len(d.get("context") or ""),
            "human": "",              # ← 留空，由人填（true/false）
            "human_note": "",
        })
    _write_csv(out, rows)
    return len(rows)


def make_annotation_sheet(dump: Path, out: Path) -> int:
    """输出**给人读**的 Markdown 标注单（逐条：问题 / 上下文 / 答案 / 判官结论 / 待勾选）。

    CSV 适合程序汇总，读长上下文很痛苦；这份 sheet 才是实际标注时用的界面。
    标注完成后把结论填回 CSV 的 `human` 列（或直接在 sheet 上写，再抄进 CSV）。
    """
    rows = _dump_rows(dump)
    lines = [
        "# judge 人工标注单（T-02-D）",
        "",
        "> **判什么**：判官的结论**对不对** —— 不是判「答案好不好」。",
        "> - 判官判 `true`（认为答案忠于证据）→ 你若认为**确实忠于**，填 `true`；若发现**编造/超出证据**，填 `false`。",
        "> - 判官判 `false` → 你若认为**判官没错**（答案确实不忠实），填 `true`；若认为答案其实没问题，填 `false`。",
        "> - 拿不准就**留空**（不计入 κ，但会显示在覆盖率里）—— **别硬猜**。",
        ">",
        "> 填回 `judge_annotation.csv` 的 `human` 列后，运行：",
        "> `python eval/calibrate_judge.py --annotations eval/judge_annotation.csv`",
        "",
        f"共 **{len(rows)}** 条。",
        "",
        "---",
        "",
    ]
    for i, d in enumerate(rows, 1):
        ctx = (d.get("context") or "").strip() or "（检索无结果）"
        lines += [
            f"## {i}. `{d.get('id')}` · {d.get('kind', '')}"
            f"{'  · **负样本（应拒答）**' if not d.get('answerable', True) else ''}",
            "",
            f"**问题**：{d.get('query', '')}",
            "",
            f"**判官结论**：`{d.get('judge_value')}` —— {d.get('judge_reason') or d.get('judge_error') or ''}",
            "",
            "**检索到的上下文**（判定依据）：",
            "",
            "```text",
            ctx,
            "```",
            "",
            "**模型答案**：",
            "",
            "```text",
            (d.get("answer") or "").strip() or "（空）",
            "```",
            "",
            "**人工判定**：`human = ` ______  （判官正确 → `true`；判官错误 → `false`；拿不准 → 留空）",
            "",
            "备注：",
            "",
            "---",
            "",
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


def make_template_from_report(report: Path, out: Path) -> int:
    """从 `run_eval` 报告生成**降级版**标注表（报告只存 200 字预览，上下文未存）。

    ⚠️ 降级点必须说清楚：`answer_preview` 被截到 200 字符、且**没有检索上下文** ——
    据此判"答案是否忠于证据"是不可靠的。仅用于先看格式/先标一部分。
    """
    rep = json.load(open(report, encoding="utf-8"))
    det = (rep.get("judge") or {}).get("details") or []
    truncated = sum(1 for d in det if len(d.get("answer_preview") or "") >= 200)
    rows = [{
        "id": d.get("id", ""), "kind": d.get("kind", ""), "query": "",
        "context": "",
        "answer": (d.get("answer_preview") or "").replace("\n", " "),
        "judge": "" if d.get("value") is None else str(bool(d["value"])).lower(),
        "judge_reason": (d.get("reason") or d.get("error") or "").replace("\n", " "),
        "context_chars": 0,
        "human": "", "human_note": "答案预览被截断(>=200字符)" if len(d.get("answer_preview") or "") >= 200 else "",
    } for d in det]
    _write_csv(out, rows)
    print(f"⚠️ 降级表：无检索上下文；{truncated}/{len(rows)} 条答案预览被截断到 200 字符 "
          f"→ 不能据此可靠判定忠实性。要完整表请先跑 export_judge_annotation.py。", file=sys.stderr)
    return len(rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)


def report_kappa(rows: list[dict], min_annotated: int) -> int:
    pairs = [(r["judge"], r["human"]) for r in rows
             if r["judge"] is not None and r["human"] is not None]
    unparsed = sum(1 for r in rows if r["judge"] is None or r["human"] is None)
    print(f"总条数 = {len(rows)} · 可比对 = {len(pairs)} · 未标注/无法解析 = {unparsed}")
    if len(pairs) < min_annotated:
        print(f"\n✗ 可比对样本 {len(pairs)} < {min_annotated} → **拒绝给出 κ**。")
        print("  理由：n 太小时 κ 的置信区间会宽到没有意义，数字会被误当成结论。")
        print("  （这与项目的取向一致：**没有结论好过假结论**。）")
        return 3

    # ★★ 语义换算（2026-09-25 修正，**算错过一次**）：
    # `human` 列的语义是「**判官结论是否正确**」，而 κ 比的是两侧对**答案**的判定 ——
    # 两者不是同一个变量，直接配对会把 κ 算到完全错误的方向
    # （本次实测：直接配对 κ≈−0.03，换算后 κ≈0.93，结论天壤之别）。
    # 换算规则：**人工对答案的判定 = (judge == human)**
    #   judge=T & human=T → 判官说忠实、人认可   → 人工答案判定 = **忠实**
    #   judge=F & human=T → 判官说不忠实、人认可 → 人工答案判定 = **不忠实**
    #   judge=T & human=F → 判官说忠实、人认为是编造 → 人工答案判定 = **不忠实**
    #   judge=F & human=F → 判官说不忠实、人认为其实忠实 → 人工答案判定 = **忠实**
    derived = [(j, (j == h)) for (j, h) in pairs]
    # ★ 判官错误 = 人工标 `human=false`（"判官结论错误"）—— 与 judge 取值无关，
    #   别用 `judge != human` 数（那是原始标签不一致数，混入了 8 条"人认可判官判不忠实"的**一致**样本）
    errors = sum(1 for r in rows if r["judge"] is not None and r["human"] is not None
                 and r["human"] is False)
    res = cohen_kappa(derived)
    a, b, c, d = res["a"], res["b"], res["c"], res["d"]
    print(f"\n判官正确率 = {1 - errors / res['n']:.4f}（{res['n'] - errors}/{res['n']}）"
          f" · 人工判定「判官错误」= {errors} 条")
    print("（human 列语义=「判官是否正确」，已换算为人工对**答案**的判定后再算 κ）")
    print("\n=== 混淆矩阵（行=judge，列=人工对**答案**的判定）===")
    print("                人工-忠实  人工-不忠实")
    print(f"  judge-忠实      {a:^10}{b:^13}")
    print(f"  judge-不忠实    {c:^10}{d:^13}")
    print(f"\n一致率 p_o = {res['po']} · 期望一致率 p_e = {res['pe']}")
    if res["kappa"] is None:
        # κ 无定义 ≠ 无结论：此时该报的是**错误率及其置信上界**（人工判"判官错"的占比）。
        ub = binom_upper(errors, res["n"])
        ub_s = f"{ub * 100:.1f}%" if ub is not None else "无法估计"
        print(f"κ 无定义：{res['note']}")
        print("\n=== 但这份数据仍有一条**有信息**的结论 ===")
        print(f"  人工判定「判官错误」的条数 = {errors}/{res['n']}")
        print(f"  ⇒ **判官错误率 95% 置信上界 ≈ {ub_s}**（Clopper–Pearson）")
        print("  （即：在本次抽样的口径下，即使按最坏情况估计，判官也只在约 "
              f"{ub_s} 的用例上可能错。）")
        print("\n=== 为什么算不出 κ / 怎么才能算 ===")
        print("  κ 度量的是『**超出随机的一致**』，前提是两侧**都有区分**。")
        print("  本批样本里判官 40 条全判 `true`、人工也 40 条全判 `true` → 两侧都无方差，")
        print("  p_e = 1 ⇒ 任何一致率都可能与巧合无法区分。**这不是标注质量问题，是样本设计问题**：")
        print("  要得到 κ，样本里必须混入**可能被判官判错**的案例（压力样本），例如：")
        print("    ① 刻意不忠实的答案（让模型脱离资料自由发挥）② 只引用部分证据的答案")
        print("    ③ 负样本但模型硬答的答案 ④ 边界情况（改写后的证据）")
        print("  详见 docs/实施记录.md 的 T-02-D 节。")
        return 4
    print(f"**Cohen's κ = {res['kappa']}**（{res['band']}）")
    print(f"  95% 近似区间（Wald，n={res['n']}）："
          f"{res['kappa'] - 1.96 * _se(a, b, c, d):.3f} ~ {res['kappa'] + 1.96 * _se(a, b, c, d):.3f}")

    print("\n=== 按标注批次 ===")
    for kind in sorted({r["kind"] for r in rows if r["kind"]}):
        sub = [(j, (j == h)) for r in rows
               if r["kind"] == kind and r["judge"] is not None and r["human"] is not None
               for j, h in [(r["judge"], r["human"])]]
        if not sub:
            continue
        k = cohen_kappa(sub)
        kk = "无定义" if k["kappa"] is None else k["kappa"]
        n_err = sum(1 for j, h in sub if j != h)
        print(f"  {kind:<12} n={k['n']:<3} κ={kk}  一致率={k['po']}  判官错误={n_err}")

    print("\n=== 分歧清单（人工判「判官错误」的条目）===")
    bad = [r for r in rows if r["judge"] is not None and r["human"] is not None and r["human"] is False]
    if not bad:
        print("  无（0 条分歧）—— ⚠️ 若样本里没有压力案例，这通常只说明**样本太易**，"
              "别直接当成「判官很准」（见 κ 无定义分支的解释）")
    for r in bad:
        print(f"  {r['id']:<6} judge={r['judge']} human={r['human']}  备注：{(r.get('note') or '')[:90]}")
    return 0


def _se(a: int, b: int, c: int, d: int) -> float:
    """κ 的 Wald 标准误（近似）。目的只是提醒"n 小的时候区间很宽"。"""
    n = a + b + c + d
    if n == 0:
        return 0.0
    po = (a + d) / n
    p_j, p_h = (a + b) / n, (a + c) / n
    pe = p_j * p_h + (1 - p_j) * (1 - p_h)
    if abs(1 - pe) < 1e-12:
        return 0.0
    # 简化式：Var(κ) ≈ p_o(1-p_o) / (n (1-p_e)^2)
    return ((po * (1 - po)) / (n * (1 - pe) ** 2)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description="judge × 人工标注一致性（Cohen's κ）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--annotations", type=Path, help="已填好的标注 CSV（含 human 列）")
    src.add_argument("--dump", type=Path, help="export_judge_annotation.py 的 JSONL dump")
    src.add_argument("--report", type=Path, help="run_eval 报告（降级表，无上下文）")
    ap.add_argument("--make-template", type=Path, help="把标注表模板（CSV）写到该路径")
    ap.add_argument("--make-sheet", type=Path,
                    help="把**给人读**的 Markdown 标注单写到该路径（只在 --dump 模式下可用）")
    ap.add_argument("--min-annotated", type=int, default=30,
                    help="少于该数量则拒绝给 κ（默认 30，与卡片要求的下沿一致）")
    args = ap.parse_args()

    if args.annotations:
        return report_kappa(load_annotations(args.annotations), args.min_annotated)
    if args.make_sheet and not args.dump:
        print("✗ --make-sheet 只在 --dump 模式下可用（报告里的上下文是缺的，读了会误判）",
              file=sys.stderr)
        return 2
    if not args.make_template and not args.make_sheet:
        print("✗ 给 --dump/--report 时必须同时给 --make-template <路径>（或 --dump + --make-sheet）",
              file=sys.stderr)
        return 2
    if args.dump:
        if args.make_template:
            n = make_template_from_dump(args.dump, args.make_template)
            print(f"已写出标注表（CSV）：{args.make_template}（{n} 条）")
        if args.make_sheet:
            n = make_annotation_sheet(args.dump, args.make_sheet)
            print(f"已写出标注单（Markdown，给人读）：{args.make_sheet}（{n} 条）")
    else:
        n = make_template_from_report(args.report, args.make_template)
        print(f"已写出标注表模板：{args.make_template}（{n} 条）")
    print("→ 请填 human 列（true/false，拿不准留空）后回跑本脚本 --annotations <csv>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
