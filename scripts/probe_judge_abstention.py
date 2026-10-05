"""判官 abstention 加固版验证脚本（限流解除后跑）。

目的：验证 2026-09-26 加固的 abstention prompt 能否修复 s13 这类已知误判。

用例：
  s13: 负样本硬答"8 GPUs"（应判 abstained=False）
  s01: 变异-编造（部分事实对，末尾硬编"已开源上千 star"，应判 False）
  还要跑两条**正样本**（应判 True）：
  - 真实拒答样本（资源里没有时模型说"资料不足"）—— 从负样本中找 answer 中含"资料中没有"等表述的

为什么单独写（不直接复用 `run_eval.py --judge`）：
  - run_eval.py 跑全量 221 例需要数分钟 + DeepSeek 调用配额
  - 只想做 8~10 次抽样验证（≈ ¥0.01）时太重
  - 这个脚本专门做"加固前 vs 加固后"对比，便于审计

资源：1 个 python × 8~10 次 DeepSeek API 调用 × 每条 ≈ 2 秒
      总时长 ≈ 30 秒，总费用 ≈ ¥0.01（DeepSeek chat 输入 ¥1/M，输出 ¥2/M）
"""
from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from judge import Judge, build_llm  # noqa: E402


def load_stress_samples(csv_path: Path, ids: list[str]) -> list[dict]:
    """从 stress 标注 CSV 读指定 id 的样本（GB18030 兼容，与 load_annotations 同源）。"""
    out: list[dict] = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["id"] in ids:
                out.append({
                    "id": r["id"],
                    "query": r["query"],
                    "context": r["context"],
                    "answer": r["answer"],
                    "expected": r["human"].strip().upper() == "TRUE",
                    "pre_judge": r["judge"].strip().upper() == "TRUE",
                    "stress_kind": r["kind"],
                    "human_note": r.get("human_note", ""),
                })
    return out


async def main() -> None:
    csv_path = ROOT / "eval" / "judge_annotation_stress.csv"

    # 已知误判样本 + 加几条对照（True/False 各覆盖）
    # s13：负样本硬答"8 GPUs"，加固前 judge=True（错），加固后期望 judge=False
    # s01：变异-编造，加固前/后期望 judge=False
    # 真实拒答样本：手动构造（从 stress 集里找 answer 明确说"资料中没有..."的）
    target_ids = ["s13", "s01"]
    samples = load_stress_samples(csv_path, target_ids)

    # ★ 构造两条"应该判 True" 的正样本（人工标过 TRUE 的负样本）
    # 从 full 集（40 条负样本里）中选 answer 明确拒答的
    full_path = ROOT / "eval" / "judge_annotation_full.csv"
    with open(full_path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if (r["human"].strip().upper() == "TRUE"
                    and "资料中没有" in r["answer"]
                    and r["id"] not in {s["id"] for s in samples}):
                samples.append({
                    "id": r["id"],
                    "query": r["query"],
                    "context": r["context"],
                    "answer": r["answer"],
                    "expected": True,
                    "pre_judge": r["judge"].strip().upper() == "TRUE",
                    "stress_kind": r.get("kind", ""),
                    "human_note": "真实拒答（应判 abstained=True）",
                })
                if sum(1 for s in samples if s["expected"]) >= 2:
                    break

    print(f"=== 判官 abstention 加固版验证（{len(samples)} 条）===")
    print(f"  用例：{[s['id'] for s in samples]}")

    # 构造判官
    llm = build_llm("deepseek", model="deepseek-chat")
    judge = Judge(llm)

    correct = 0
    fixed = 0  # 加固后修对的样本数（加固前错，加固后对）
    for s in samples:
        v = await judge.abstention(context=s["context"], question=s["query"], answer=s["answer"])
        post = v.value
        ok = "✅" if post == s["expected"] else "❌"
        was = "fixed" if (post == s["expected"] and s["pre_judge"] != s["expected"]) else (
               "still-wrong" if post != s["expected"] else "still-right")
        if post == s["expected"]:
            correct += 1
            if s["pre_judge"] != s["expected"]:
                fixed += 1
        print(f"\n  {ok} {s['id']} ({s['stress_kind']}): "
              f"expected={s['expected']} · pre={s['pre_judge']} → post={post} · {was}")
        print(f"      query: {s['query']}")
        print(f"      answer[:80]: {s['answer'][:80]!r}")
        if v.error:
            print(f"      ⚠️ 解析失败：{v.error}")
        else:
            print(f"      reason: {v.reason[:120]!r}")

    print("\n=== 汇总 ===")
    print(f"  正确 {correct}/{len(samples)} = {correct / len(samples):.1%}")
    print(f"  加固前错、加固后对：{fixed} 条")
    if correct == len(samples):
        print("  ✅ 加固版 100% 通过；可写报告")
    elif correct >= len(samples) * 0.8:
        print(f"  ⚠️ 80% 通过，需复核 {len(samples) - correct} 条")
    else:
        print("  ❌ 加固版未达标，需进一步加固")


if __name__ == "__main__":
    asyncio.run(main())