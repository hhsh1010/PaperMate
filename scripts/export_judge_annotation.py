"""生成判官人工标注材料（T-02-D 的输入）——**把"人工标注 κ"所需的一切落到文件里**。

## ★ 两条路径，只用第一条（2026-09-25 事故后重构）

**主路径（零成本、唯一正确）**：`run_eval --judge --judge-dump <path>` 在判分时**顺手落盘**
`(query, context, answer, 结论)`，本脚本只把它翻译成 CSV / 给人读的 Markdown：

    .venv/Scripts/python.exe scripts/export_judge_annotation.py --dump eval/judge_dump.jsonl \
        --csv eval/judge_annotation.csv --sheet eval/judge_annotation_sheet.md

**旧路径（已降级为排查用）**：重新跑检索 + 重新生成答案，再与报告里的判官结论按 id 配对。
**实测证明它不成立**：重新生成的 40 条里有 **22 条与判分时用的答案实质不同**
（`qwen3.6-flash` 在 `temperature=0` 下**并不确定**）⇒ 表里「判官结论」与「答案」不是同一条
⇒ 人工据此标 κ 是**无效**的。根因是"结论存报告、输入没存"这一设计缺口。

## 旧路径的成本（仅当你明确要用时）

| 项 | 量 |
|---|---|
| API | 40 例生成 ≈ 7.6 万 tokens（Qwen 免费额度） |
| GPU | 检索 40 例 ≈ 20~30s |
| 风险 | ★ **与判分时的答案不一致 → κ 无效**（实测 22/40 不同） |
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "eval"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calibrate_judge as cal  # noqa: E402
import judge as judge_mod  # noqa: E402
import run_eval  # noqa: E402


async def _export(args) -> int:
    # ★★ 主路径（零成本、且唯一正确）：直接从 `--judge-dump` 的表生成标注材料。
    # 2026-09-25 事故：原先"重新生成答案"的路径，40 条里 22 条与判官当时评的答案**实质不同**
    # （temperature=0 也不确定）→ 表里「判官结论」与「答案」不是同一条 → κ 不成立。
    # ⇒ **谁产生结论，谁落盘原始输入**；本脚本只做"翻译成给人读的表"，不再自己生成。
    if args.dump:
        print(f"从 dump 读取（**不调任何 API、不用 GPU**）：{args.dump}")
        if args.csv:
            n = cal.make_template_from_dump(Path(args.dump), Path(args.csv))
            print(f"已写出标注表（CSV，机器汇总用）：{args.csv}（{n} 条）")
        if args.sheet:
            n = cal.make_annotation_sheet(Path(args.dump), Path(args.sheet))
            print(f"已写出标注单（Markdown，**给人读**）：{args.sheet}（{n} 条）")
        print(f"→ 填完 human 列后跑：`python eval/calibrate_judge.py --annotations {args.csv}`")
        return 0

    print("⚠️ 你用的是**旧的「重新生成答案」路径** —— 实测它与判分时用的答案不一致（40 条里 22 条不同），\n"
          "   据此标注的 κ 不成立。正确做法：用 `run_eval --judge --judge-dump <path>` 产出 dump，\n"
          "   再用 `--dump <path>` 走零成本主路径。现按旧路径执行，产物仅供排查用。",
          file=sys.stderr)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    jblock = report.get("judge") or {}
    sample = jblock.get("sample") or {}
    if not jblock:
        print("✗ 报告里没有 judge 块 —— 先跑 `run_eval --judge`（本脚本不产生判官结论）")
        return 2

    verdicts = {d["id"]: d for d in jblock.get("details") or []}
    ids = sample.get("ids") or list(verdicts)
    if not ids:
        print("✗ 报告里既无 sample.ids 也无 details —— 无法确定要标注哪些用例")
        return 2

    all_cases = run_eval.load_cases(args.cases)
    picked = [c for c in all_cases if c["id"] in set(ids)]
    print(f"待标注用例 {len(picked)}/{len(all_cases)}（来自报告抽样口径：{sample.get('strategy', '?')}）")

    # ① 取检索上下文：只在**抽到的这些用例**上跑，不重复跑全量（省 GPU）
    sparse_on, sparse_src = run_eval._resolve_sparse(args.sparse)
    rk = run_eval._resolve_reranker_opts(args.rerank)
    factory = run_eval.PipelineFactory(args.embedder, args.profile,
                                       use_bge_sparse=sparse_on, sparse_source=sparse_src,
                                       reranker_fp16=rk[0], reranker_max_length=rk[1],
                                       rerank_source=rk[2])
    print(f"口径：profile={args.profile} · sparse={sparse_on}（{sparse_src}）· rerank={rk[2]}")
    contexts: dict[str, list[str]] = {}
    for c in picked:
        run_eval.run_case(c, args.k, factory, context_sink=contexts)
    print(f"已取回 {len(contexts)} 个用例的检索上下文")

    # ② 生成答案：**复用报告里的厂商/模型**，保证与判官当时评的是同一个配置
    answer_p = (jblock.get("answer") or {}).get("provider") or run_eval._auto_answer_provider()
    answer_model = (jblock.get("answer") or {}).get("model") or run_eval._default_model_for(answer_p)
    llm = judge_mod.build_llm(answer_p, model=answer_model, timeout=args.timeout)
    judge_mod.ensure_available(llm)
    print(f"生成侧：{llm.endpoint.name}/{llm.model_name}（密钥来源 {getattr(llm, 'key_source', '?')}）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_mismatch = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for c in picked:
            cid = c["id"]
            context = "\n\n".join(contexts.get(cid) or []) or "（检索无结果）"
            answer, err = await judge_mod.generate_answer(
                llm, question=c["query"], context=context, timeout_s=args.timeout)
            v = verdicts.get(cid) or {}
            if err:
                print(f"  {cid:<6} ✗ 生成失败：{err[:80]}")
            # 一致性抽查：报告里的 200 字预览应当是本答案的前缀
            prev = (v.get("answer_preview") or "").strip()
            if prev and not answer.strip().startswith(prev[:80]):
                n_mismatch += 1
            f.write(json.dumps({
                "id": cid, "kind": c.get("kind", ""), "answerable": bool(c.get("answerable", True)),
                "query": c["query"], "context": context, "answer": answer,
                "judge_value": v.get("value"), "judge_ok": v.get("ok"),
                "judge_reason": v.get("reason"), "judge_error": v.get("error"),
                "gen_error": err,
            }, ensure_ascii=False) + "\n")
            mark = "?" if v.get("value") is None else ("True " if v["value"] else "False")
            print(f"  {cid:<6} {c.get('kind', ''):<12} 判官={mark} 答案 {len(answer)} 字 · 上下文 {len(context)} 字")

    print(f"\n已写出 {out}")
    if n_mismatch:
        print(f"⚠️ 有 {n_mismatch} 条答案与报告里的预览不一致 —— 生成不是确定性的？"
              f"（temperature=0 本应可复现；请核对后再用这张表）")
    else:
        print("✓ 答案与报告预览一致（温度 0 复现成功）")

    if args.csv:
        n = cal.make_template_from_dump(out, Path(args.csv))
        print(f"已写出标注表（CSV，机器汇总用）：{args.csv}（{n} 条）")
    if args.sheet:
        n = cal.make_annotation_sheet(out, Path(args.sheet))
        print(f"已写出标注单（Markdown，**给人读**）：{args.sheet}（{n} 条）")
    print(f"→ 填完 human 列后跑：`python eval/calibrate_judge.py --annotations {args.csv}`")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="判官人工标注材料生成（主路径＝从 dump 翻译，零成本）")
    ap.add_argument("--dump", default=None,
                    help="★ **推荐**：`run_eval --judge-dump` 产出的 JSONL → 零 API、零 GPU 生成标注材料")
    ap.add_argument("--report", default="eval/report_real.json")
    ap.add_argument("--cases", default="cases/golden_real.jsonl")
    ap.add_argument("--out", default="eval/judge_annotation.jsonl")
    ap.add_argument("--csv", default="eval/judge_annotation.csv")
    ap.add_argument("--sheet", default="eval/judge_annotation_sheet.md",
                    help="Markdown 标注单（含完整上下文与答案，实际标注时看这份）")
    ap.add_argument("--embedder", choices=["fake", "real"], default="real")
    ap.add_argument("--profile", choices=["fast", "prod"], default="prod")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sparse", choices=list(run_eval.SPARSE_CHOICES), default="auto")
    ap.add_argument("--rerank", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--timeout", type=float, default=judge_mod.DEFAULT_TIMEOUT_S)
    args = ap.parse_args()
    return asyncio.run(_export(args))


if __name__ == "__main__":
    raise SystemExit(main())
