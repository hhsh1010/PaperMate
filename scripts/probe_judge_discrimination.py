"""判官分辨力反向自检（T-02-C 配套）——**判官说"全都忠实"时，先怀疑判官会不会说"不"**。

## 为什么需要它（真实动机，2026-09-25）

`run_eval --judge --judge-limit 40` 首次跑出 **faithfulness = 1.0000（33/33）· abstention = 1.0000（7/7）**。
这个"满分"有**两种完全不同的解释**：

1. 答案确实干净（生成侧提示词强制"只依据资料"，且样本多为逐字命中型 → 抽取式作答）；
2. 判官**宽松**（对什么都判通过） —— 此时 1.0 是"没有分辨力"，不是"质量好"。

**光看汇总数字区分不了这两者**（0 个失败样本 ⇒ 无法估计判官漏判率）。
本脚本用**受控负样本**把第 2 种可能性排除掉：给判官一条**忠于证据**的答案与两条**故意编造**的答案，
期望结果是 `[True, False, False]`。只要判官能拒掉编造项，`1.0` 才是**有信息的**。

## 判据（先定后测）

| 用例 | 期望 | 若不符 |
|---|---|---|
| 对照·忠于证据（复述原文） | `True` | 判官过敏（把忠实当不忠实） |
| 反向·编造数字（原文没有的数字） | `False` | ★ 判官宽松 → 汇总 1.0 不可信 |
| 反向·凭空断言（原文没有的内容） | `False` | ★ 判官宽松 → 汇总 1.0 不可信 |

## 成本

**3 次真实判官调用**（约 1.5k~3k tokens，¥0.01 量级），零 GPU、零检索、秒级。
比"跑一遍 40 例再人工核对"便宜两个数量级，适合**每次改判官提示词后**都跑一次。

用法：

    .venv/Scripts/python.exe scripts/probe_judge_discrimination.py
    .venv/Scripts/python.exe scripts/probe_judge_discrimination.py --judge-provider qwen
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "eval"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import judge as judge_mod  # noqa: E402
import run_eval  # noqa: E402

OUT_FILE = _ROOT / "eval" / "t02_judge_discrimination.txt"

#: 受控上下文：内容自洽、事实明确，便于构造"明显不在其中"的编造项
CONTEXT = (
    "第三章 调优顺序。混合检索的调优顺序应当是：先确认通道互补，再调融合常数，"
    "最后在重排之前裁剪候选。反过来做会得到调不动、说不清、也复现不了的结果。"
)

#: (标签, 答案, 期望的判官结论)
CASES: list[tuple[str, str, bool]] = [
    ("对照-忠于证据", "混合检索的调优顺序是先确认通道互补，再调融合常数，最后在重排之前裁剪候选。", True),
    ("反向-编造数字",
     "论文要求必须先把候选数降到 10 以下再调融合常数；并指出混合检索在中文场景下召回率必然低于 60%。", False),
    ("反向-凭空断言", "该文给出了完整的实现代码，并证明重排阶段可以完全省略以节省一半延迟。", False),
]


def _say(line: str = "") -> None:
    print(line, flush=True)
    with open(OUT_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


async def main() -> int:
    ap = argparse.ArgumentParser(description="判官分辨力反向自检")
    ap.add_argument("--judge-provider", choices=["auto", "deepseek", "qwen"], default="auto",
                    help="判官厂商（auto=按答案厂商取另一家，保证跨厂商）")
    ap.add_argument("--answer-provider", choices=["auto", "deepseek", "qwen"], default="auto")
    ap.add_argument("--timeout", type=float, default=judge_mod.DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _say("")
    _say("=" * 92)
    _say("判官分辨力反向自检（3 次真实调用；判据：对照应为 True、两条编造应为 False）")
    _say("=" * 92)

    answer_p = args.answer_provider
    if answer_p == "auto":
        answer_p = run_eval._auto_answer_provider()
    judge_p = args.judge_provider
    if judge_p == "auto":
        judge_p = judge_mod.pick_judge_provider(answer_p)
    if judge_p == answer_p:      # 与 run_eval._resolve_providers 同一硬约束
        _say(f"✗ 判官厂商与答案厂商相同（{judge_p}）—— 同源自评会系统性偏乐观，拒绝执行")
        return 2

    llm = judge_mod.build_llm(judge_p)
    judge_mod.ensure_available(llm)
    _say(f"判官：{llm.endpoint.name} / {llm.model_name}（答案侧 {answer_p} · 跨厂商 ✓）")
    _say(f"密钥来源：{getattr(llm, 'key_source', '?')}")
    _say("")
    judge = judge_mod.Judge(llm, timeout_s=args.timeout)

    got: dict[str, object] = {}
    failed = 0
    for tag, answer, _expect in CASES:
        try:
            # 用**生产同一入口**判分（`Judge.faithfulness`），不另造一套调用，避免"探针测的不是线上那条路"
            v = await asyncio.wait_for(
                judge.faithfulness(context=CONTEXT, answer=answer), timeout=args.timeout * 3)
        except Exception as e:  # noqa: BLE001  单条异常不毁整轮
            v = judge_mod.JudgeVerdict(ok=False, value=None, error=f"{type(e).__name__}: {e}")
        got[tag] = v.value
        mark = "?" if not v.ok else ("PASS" if v.value else "FAIL")
        _say(f"  {tag:<14} ok={v.ok!s:<5} 判为={mark:<5} | {(v.reason or v.error or '')[:130]}")
        if not v.ok:
            failed += 1

    _say("")
    _say("=== 判定 ===")
    bad: list[str] = []
    for tag, _answer, expect in CASES:
        actual = got.get(tag)
        ok = (actual is expect)
        if not ok:
            bad.append(tag)
        _say(f"  {tag:<14} 期望={expect!s:<5} 实际={actual!s:<5} {'✓' if ok else '✗ 不符'}")

    if failed:
        _say(f"\n⚠️ 有 {failed} 条判分失败（ok=False）→ 结论不成立，先查判官可用性/提示词。")
        return 5
    if bad:
        _say(f"\n★ 判官**分辨力不足**：{', '.join(bad)} 与期望不符。")
        _say("  ⇒ 此时 `run_eval --judge` 的 faithfulness/abstention 若为 1.0，**不能当作质量证据**")
        _say("     （判官可能对什么都判通过）——先修提示词，再重跑正式评测。")
        return 4

    _say("\n✅ 判官有分辨力：能通过忠实答案、能拒绝编造答案。")
    _say("  ⇒ 汇总里的 1.0 是**有信息**的（不是'永远通过'），可作为本轮质量结论的依据。")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PAPERMATE_DEVICE", "cpu")   # 本脚本不需要 GPU（纯文本判分）
    raise SystemExit(asyncio.run(main()))
