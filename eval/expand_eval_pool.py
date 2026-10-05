"""O-14 方案 A（2026-10-01）：把单文档用例的**候选池**扩大到多文档 —— **只改 `corpus` 字段**。

为什么这么做（诊断见 `docs/O-14-评测语料分辨力-诊断与方案.md`）：
评测语料 11 篇 × 5~6 块 = **57 个候选**，而真实库 **658 块**（小 11.5×）。142 条单文档用例的
候选池就是那**一篇文档的 ~6 个块**，`k=5` ⇒ 几乎等于把整篇原样返回 ⇒ `recall@5` 恒为 1.0、
`precision@5` 恰好等于理论上限 ⇒ **两个指标分辨力为 0**（构造 ≥ 返回结果 ⇒ 改不出差异）。

★ 为什么**不必重新标注**：gold 是目标文档的 chunk id（如 `pm-rag-c0`）；加干扰文档只是把
候选池变大，**不改目标文档的解析与分块** ⇒ gold / gains / answer_sentence / reference
全部仍然成立。`kind` 描述的是"query 与原文的表述关系"（verbatim/paraphrase/…），
与候选池大小正交 ⇒ 语义不变。

★ 为什么**保留一部分单文档用例**：
`eval/test_golden_real.py::test_golden_real_exists_with_three_kinds` 要求"单文档 ≥40 条"
（防止评测集悄悄退化）。全扩会让该守卫立刻变红。因此按 `--keep-single` 保留前 N 条
（按 case id 排序，确定性规则）作为**历史锚点**，其余才扩 ⇒ 老门槛不破，历史可比性也在。

用法：
    python eval/expand_eval_pool.py --dry-run          # 只看改动面，不落盘
    python eval/expand_eval_pool.py                    # 落盘（自动备份原文件）
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases" / "golden_real.jsonl"

_TARGET_KEY = "corpus"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plan_expansion(
    cases: list[dict], all_docs: list[str], *, keep_single: int, distractors: int,
    multi_gold_min_docs: int = 6,
) -> tuple[list[dict], list[str]]:
    """生成扩充计划：返回 (新 cases, 被改动 case 的 id 列表)。

    确定性规则（可复现，不依赖遍历顺序）:
    1. 目标 Case = `answerable` 且 `corpus` 只有 1 篇；按 **id 升序**排序后，
       先跳过前 `keep_single` 条（历史锚点），其余全扩；
    2. 干扰文档 = `all_docs` 排序后**排除目标篇**取前 `distractors` 篇（含已有的其它篇）；
    3. 只改 `corpus` 字段，其它字段**原样搬运**（gold/gains/reference 一字不动）。
    """
    # ★ 锚点选择改为**按 kind 轮转**（2026-10-01 实测修正）：
    # 曾用"按 id 取前 N 条"作锚点，结果锚点集中在排序靠前的 verbatim/paraphrase 上
    # ⇒ `paraphrase` 的 recall@5 又回到 1.0（该类几乎没被加难），而 multi_gold 却过度加难。
    # 轮转取锚点 ⇒ **每一类都既保留历史锚点（可比性）又保留加难用例（分辨力）**。
    singles_by_kind: dict[str, list[dict]] = {}
    for x in cases:
        if x.get("answerable", True) and len(x.get(_TARGET_KEY) or []) == 1:
            singles_by_kind.setdefault(x.get("kind", "unknown"), []).append(x)
    for lst in singles_by_kind.values():
        lst.sort(key=lambda c: c["id"])

    anchors: set[str] = set()
    i = 0
    while len(anchors) < keep_single:
        added = False
        for kind in sorted(singles_by_kind):
            lst = singles_by_kind[kind]
            if i < len(lst):
                anchors.add(lst[i]["id"])
                added = True
                if len(anchors) >= keep_single:
                    break
        if not added:  # 全部取完仍未达标 ⇒ 全集都不够，直接退出
            break
        i += 1

    targets = [x for lst in singles_by_kind.values() for x in lst if x["id"] not in anchors]

    changed: list[str] = []
    for c in targets:
        target_doc = c[_TARGET_KEY][0]
        pool = [d for d in all_docs if d != target_doc][:distractors]
        new_corpus = sorted({*c[_TARGET_KEY], *pool})
        if new_corpus != c[_TARGET_KEY]:
            c[_TARGET_KEY] = new_corpus
            changed.append(c["id"])

    # 第二轮（2026-10-01 实测追加）：`multi_gold` 用例此前**未被本脚本覆盖**（它们本来就是
    # 多文档），第一轮跑完后唯独它的 `recall@5` 仍为 1.0、`precision` saturation 仍 1.0
    # ⇒ 该类**依然没有分辨力**。这里把它补到至少 `multi_gold_min_docs` 篇。
    # 为什么值得再动一次：O-14 的目标就是"每个 kind 都能测出差异"，留一类贴顶等于该类的
    # 优化**依然无法被证明**（本项目最忌的"测不出差异 ⇒ 误判为没问题"）。
    if multi_gold_min_docs > 0:
        for c in sorted(
            [x for x in cases
             if x.get("kind") == "multi_gold" and x["id"] not in anchors
             and len(x.get(_TARGET_KEY) or []) < multi_gold_min_docs],
            key=lambda c: c["id"],
        ):
            pool = [d for d in all_docs if d not in c[_TARGET_KEY]]
            need = multi_gold_min_docs - len(c[_TARGET_KEY])
            new_corpus = sorted({*c[_TARGET_KEY], *pool[:need]})
            if new_corpus != c[_TARGET_KEY]:
                c[_TARGET_KEY] = new_corpus
                changed.append(c["id"])
    return cases, changed


def main() -> None:
    ap = argparse.ArgumentParser(description="O-14：扩大评测用例的候选池（只改 corpus 字段）")
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--keep-single", type=int, default=50,
                    help="保留多少条单文档用例不改（历史锚点，默认 50，须 ≥40 以满足既有守卫下界）")
    ap.add_argument("--distractors", type=int, default=3, help="每条用例补几篇干扰文档（默认 3）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases_path = Path(args.cases)
    cases = _load(cases_path)
    all_docs = sorted(p.stem for p in (HERE / "corpus").glob("*.pdf"))
    # ★ 必须先快照：`plan_expansion` 会**原地改** cases；若等改完再取 before / 再做对拍，
    #   比较的是"改过的自己" ⇒ 反向自检空转（本次差点犯这个错）。
    original = {c["id"]: json.loads(json.dumps(c, ensure_ascii=False)) for c in cases}
    before = Counter(len(c.get(_TARGET_KEY) or []) for c in original.values())
    new_cases, changed = plan_expansion(
        cases, all_docs, keep_single=args.keep_single, distractors=args.distractors,
    )

    after = Counter(len(c.get(_TARGET_KEY) or []) for c in new_cases)
    print(f"[plan] 总用例 {len(cases)} → 改 {len(changed)} 条（保留 {args.keep_single} 条单文档锚点）")
    print("       候选池大小分布 变更前后：")
    for size in sorted(set(before) | set(after)):
        if before.get(size, 0) != after.get(size, 0):
            print(f"         {size} 篇: {before.get(size, 0)} → {after.get(size, 0)}")

    # ★ 反向对拍：确认除 corpus 外**没有任何字段**被动过（防"顺手改了 gold"）
    for c in new_cases:
        o = original[c["id"]]
        for k, v in c.items():
            if k == _TARGET_KEY:
                continue
            assert v == o.get(k), f"{c['id']} 的 {k} 字段被改动（只允许改 corpus）"
        assert set(c) == set(o), f"{c['id']} 的字段集合发生变化"
        # gold 必须仍是目标文档自己的块（加干扰文档不得污染 gold）
        gold_docs = {g.rsplit("-c", 1)[0] for g in (c.get("gold") or [])}
        assert gold_docs <= set(o.get("corpus") or []), f"{c['id']} 的 gold 越出原目标文档：{gold_docs}"

    if args.dry_run:
        print("[dry-run] 未落盘")
        return

    backup = cases_path.with_name(f"{cases_path.name}.bak-pool{time.strftime('%Y%m%d%H%M')}")
    tmp = cases_path.with_suffix(".tmp")
    backup.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in original.values()) + "\n",
                      encoding="utf-8")
    tmp.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in new_cases) + "\n", encoding="utf-8")
    tmp.replace(cases_path)  # 原子替换
    print(f"[done] 已备份原文件 → {backup.name}；新 cases → {cases_path.name}")


if __name__ == "__main__":
    main()
