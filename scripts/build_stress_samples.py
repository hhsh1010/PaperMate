"""构造「压力样本」标注集（T-02-D+）——**让 κ 可计算**。

## 为什么要这一轮（2026-09-25）

T-02-D 的 40 条标注里判官与人**全部一致且全为 true** → 两侧都无方差 → κ 无定义。
**根因是样本设计**：40 条全是"真实分布抽样"，而模型被提示词约束成"只依据资料"，几乎必然忠实。
要算 κ，标注集里必须混入**可能被判官判错**的案例。本脚本构造 4 类压力样本：

| 类型 | 构造方式 | 预期判官 | 数量 |
|---|---|---|---|
| `mutate-invent` | 在忠实答案后**追加一条上下文里没有的断言**（确定性拼装，零 API） | False | 3 |
| `mutate-external` | 混入"结合我的通用知识"式的外部论断 | False | 3 |
| `unconstrained` | **去掉"只依据资料"约束**重新生成（模型可能混入自身知识） | 混合 | 5 |
| `negative-hard` | 负样本问题 + 「直接给出具体答案，不要说没有」→ 模型硬答 | False | 2 |
| `paraphrase` | 让模型**改写**忠实答案（语义不变）→ 测判官是否过敏 | True | 2 |

**盲评纪律（关键）**：标注单（`make_annotation_sheet`）**不携带 stress_kind** ——
如果让人看出"这条是故意做错的"，标注就被污染了。类型只存在 JSONL 的 `stress_kind` 字段里，
算完 κ 再用于归因。

## 成本（如实标注）

| 项 | 量 |
|---|---|
| API 生成 | **9 次**（Qwen 免费额度；变异类 6 条零 API） |
| API 判分 | **15 次**（DeepSeek，≈3 万 tokens ⇒ **¥0.03 量级**） |
| GPU | **0**（上下文直接复用 `eval/judge_dump.jsonl` 里已检索好的） |
| 时长 | 约 2~4 分钟 |

用法：

    .venv/Scripts/python.exe scripts/build_stress_samples.py
    # 产出：eval/judge_dump_stress.jsonl（含 stress_kind）
    #       eval/judge_annotation_stress.csv + eval/judge_annotation_sheet_stress.md（给人读）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "eval"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import judge as judge_mod  # noqa: E402
import run_eval  # noqa: E402
from calibrate_judge import make_annotation_sheet, make_template_from_dump  # noqa: E402

from lightcore.llm.base import ChatMessage, Role  # noqa: E402

OUT_DUMP = _ROOT / "eval" / "judge_dump_stress.jsonl"
OUT_CSV = _ROOT / "eval" / "judge_annotation_stress.csv"
OUT_SHEET = _ROOT / "eval" / "judge_annotation_sheet_stress.md"

#: 追加的"上下文里没有"的断言（拼进忠实答案尾部 → 判官应识别为编造）
_INVENTED = "另外，资料在第 9 节明确指出：该方法已在 GitHub 开源并获得上千 star，可直接商用。"
#: 外部知识混入的前后缀（与上下文无关的论断 → 边界不忠实）
_EXT_PRE = "结合我的通用经验："
_EXT_POST = "\n\n注：这类系统在生产环境通常还需要 GPU 集群与分布式训练，资料中的单机方案并不完整。"

_CN_NUM = {"一": "十", "两": "二十", "三": "三十", "五": "五十", "七": "七十", "十": "一百"}


def _mutate_invent(base: dict) -> dict:
    """忠实答案 + 一条编造断言。"""
    return {**base, "answer": (base["answer"].rstrip() + "\n" + _INVENTED)}


def _mutate_external(base: dict) -> dict:
    """混入模型自身知识（边界不忠实：答案主体忠实，但夹带外部论断）。"""
    return {**base, "answer": _EXT_PRE + base["answer"] + _EXT_POST}


def _mutate_number(base: dict) -> dict:
    """把答案里第一个数字改掉（数字型失真最容易被人工抓住，适合当"明显错误"样本）。"""
    ans = base["answer"]

    def bump(m: re.Match) -> str:
        return str(int(m.group(0)) * 10 + 7)

    new, n = re.subn(r"\d+", bump, ans, count=1)
    if n == 0:                                   # 无阿拉伯数字 → 换中文数词
        for k, v in _CN_NUM.items():
            if k in ans:
                new = ans.replace(k, v, 1)
                break
        else:
            new = ans + "\n（补充：整体规模约为原来的三倍。）"
    return {**base, "answer": new}


async def _gen(llm, system: str, user: str, timeout: float) -> str:
    resp = await asyncio.wait_for(
        llm.chat([ChatMessage(role=Role.SYSTEM, content=system),
                  ChatMessage(role=Role.USER, content=user)],
                 temperature=0.0, max_tokens=600),
        timeout=timeout)
    return (resp.message.content or "").strip()


async def main() -> int:
    ap = argparse.ArgumentParser(description="构造压力样本并交判官判分")
    ap.add_argument("--dump", default="eval/judge_dump.jsonl", help="真实评测的 dump（取上下文与问题）")
    ap.add_argument("--answer-model", default=None, help="覆盖生成模型（默认线上默认模型）")
    ap.add_argument("--timeout", type=float, default=judge_mod.DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in Path(args.dump).read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_id = {r["id"]: r for r in rows}
    answerable = [r for r in rows if r.get("answerable")]
    negatives = [r for r in rows if not r.get("answerable")]
    print(f"源 dump：{len(rows)} 条（可答 {len(answerable)} / 负样本 {len(negatives)}）")

    # ---------- ① 确定性变异（零 API） ----------
    stress: list[dict] = []
    for sid, base_id, fn in (("s01", "r35", _mutate_invent), ("s02", "p32", _mutate_invent),
                             ("s03", "m21", _mutate_invent), ("s04", "r53", _mutate_external),
                             ("s05", "p44", _mutate_external), ("s06", "m34", _mutate_external)):
        base = by_id[base_id]
        stress.append({
            "id": sid, "kind": "", "source_kind": base["kind"], "answerable": True,
            "query": base["query"], "context": base["context"],
            "answer": fn(base)["answer"], "stress_kind": fn.__name__,
            "source_id": base_id,
        })

    # ---------- ② API 生成（Qwen 免费额度） ----------
    answer_p = run_eval._auto_answer_provider()
    answer_model = args.answer_model or run_eval._default_model_for(answer_p)
    llm = judge_mod.build_llm(answer_p, model=answer_model, timeout=args.timeout)
    judge_mod.ensure_available(llm)
    print(f"生成侧：{llm.endpoint.name}/{llm.model_name}（密钥 {getattr(llm, 'key_source', '?')}）")

    unconstrained_ids = ["r01", "r24", "p13", "m01", "m08"]
    for i, bid in enumerate(unconstrained_ids, 1):
        base = by_id[bid]
        ans = await _gen(llm, "你是论文问答助手，请回答用户的问题。",        # ← 无"只依据资料"约束
                         f"{base['query']}\n\n背景资料：\n{base['context']}", args.timeout)
        stress.append({"id": f"s{6+i:02d}", "kind": "", "source_kind": base["kind"], "answerable": True,
                       "query": base["query"], "context": base["context"], "answer": ans,
                       "stress_kind": "unconstrained", "source_id": bid})
        print(f"  unconstrained {bid} → {len(ans)} 字")

    for i, bid in enumerate(["n07", "n26"], 1):                          # 负样本硬答
        base = by_id[bid]
        ans = await _gen(llm, "直接回答用户的问题，给出具体的数字与结论。不要回答“资料中没有”。",
                         base["query"], args.timeout)
        stress.append({"id": f"s{11+i:02d}", "kind": "", "source_kind": base["kind"], "answerable": False,
                       "query": base["query"], "context": base["context"], "answer": ans,
                       "stress_kind": "negative-hard", "source_id": bid})
        print(f"  negative-hard {bid} → {len(ans)} 字")

    for i, bid in enumerate(["r13", "p25"], 1):                          # 改写控制组（应仍忠实）
        base = by_id[bid]
        ans = await _gen(llm, "把给出的答案改写成不同的措辞。要求：所有事实与细节必须与原答案一致，不得增删。",
                         f"问题：{base['query']}\n\n原答案：{base['answer']}", args.timeout)
        stress.append({"id": f"s{13+i:02d}", "kind": "", "source_kind": base["kind"], "answerable": True,
                       "query": base["query"], "context": base["context"], "answer": ans,
                       "stress_kind": "paraphrase", "source_id": bid})
        print(f"  paraphrase {bid} → {len(ans)} 字")

    # ---------- ③ 判官判分（DeepSeek） ----------
    jud_p = judge_mod.pick_judge_provider(answer_p)
    jllm = judge_mod.build_llm(jud_p, timeout=args.timeout)
    judge_mod.ensure_available(jllm)
    judge = judge_mod.Judge(jllm, timeout_s=args.timeout)
    print(f"判官：{jllm.endpoint.name}/{jllm.model_name}")

    OUT_DUMP.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DUMP.open("w", encoding="utf-8", newline="\n") as f:
        for s in stress:
            if s["answerable"]:
                v = await judge.faithfulness(context=s["context"], answer=s["answer"])
            else:
                v = await judge.abstention(context=s["context"], question=s["query"], answer=s["answer"])
            f.write(json.dumps({**s, "judge_value": v.value, "judge_ok": v.ok,
                                "judge_reason": v.reason, "judge_error": v.error,
                                "gen_error": None}, ensure_ascii=False) + "\n")
            print(f"  {s['id']:<5} 判官={'?' if not v.ok else ('True ' if v.value else 'False')} | {s['stress_kind']}")

    # ---------- ④ 出标注材料（sheet **不含** stress_kind → 盲评） ----------
    n1 = make_template_from_dump(OUT_DUMP, OUT_CSV)
    n2 = make_annotation_sheet(OUT_DUMP, OUT_SHEET)
    dist: dict[str, int] = {}
    for s in stress:
        dist[s["stress_kind"]] = dist.get(s["stress_kind"], 0) + 1
    print(f"\n已写出：{OUT_DUMP.name}（{len(stress)} 条）· {OUT_CSV.name}（{n1}）· {OUT_SHEET.name}（{n2}）")
    print(f"压力类型分布（**仅记录在 dump，标注单不显示**）：{dist}")
    print("→ 请标注 `judge_annotation_stress.csv` 的 human 列；标完与原 40 条合并重算 κ")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
