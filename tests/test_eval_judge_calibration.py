"""κ 校准的回归测试（T-02-D，`eval/calibrate_judge.py`）。

锁定的**设计取向**（每条对应一个真实风险）：

1. **κ 无定义时必须说"无定义"** —— 两侧都只取一种取值时 `p_e = 1`，公式除零。
   此时若返回 1.0 或 0.0，都会被读成"结果"，而实际上**这个样本根本没有分辨力**
   （判官全判通过 + 人工全填 true ⇒ 看着完美，其实什么也没证明）。
2. **标注值无法识别一律当"未标注"** —— 绝不猜成 True：把看不清的标注猜成"判官对"就是造假。
3. **样本太少拒绝给 κ**（默认 <30）—— n 小时置信区间宽到没有意义，数字会被误当结论。
   与项目取向一致：**没有结论好过假结论**。
4. **κ 的算术必须对**（用手算样例锁死），否则"一致性 0.17"这种数会被当成 0.7 来讲。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import calibrate_judge as cal  # noqa: E402

# ---------------------------------------------------------------- κ 算术


def test_kappa_matches_hand_computed_example() -> None:
    """手算样例：judge=[T,T,F,F,T] · human=[T,T,F,T,F] → κ = 1/6 ≈ 0.1667。

    a=2(TT) b=1(TF) c=1(FT) d=1(FF)；p_o=0.6；p_e=0.52 ⇒ κ=(0.6-0.52)/0.48。
    """
    pairs = [(True, True), (True, True), (False, False), (False, True), (True, False)]
    res = cal.cohen_kappa(pairs)
    assert res["n"] == 5
    assert (res["a"], res["b"], res["c"], res["d"]) == (2, 1, 1, 1)
    assert res["po"] == 0.6
    assert res["pe"] == 0.52
    assert res["kappa"] == pytest.approx(1 / 6, abs=1e-4)


def test_kappa_perfect_agreement() -> None:
    res = cal.cohen_kappa([(True, True), (False, False), (True, True)])
    assert res["kappa"] == 1.0


def test_kappa_is_none_when_no_variance() -> None:
    """★ 关键：判官全通过 + 人工全通过 → κ **无定义**，不许返回 1.0。

    这正是"汇总数字提示异常"的场景：40 条全一致、κ 却是"无定义"，
    说明这批样本**没有区分力**，还得靠受控负样本（probe_judge_discrimination）兜底。
    """
    res = cal.cohen_kappa([(True, True)] * 10)
    assert res["kappa"] is None
    assert "无定义" in res["note"]
    assert res["po"] == 1.0, "一致率仍应如实给出（1.0），只是 κ 无定义"


def test_kappa_none_for_empty() -> None:
    res = cal.cohen_kappa([])
    assert res["kappa"] is None and res["n"] == 0


# ---------------------------------------------------------------- 解析与防伪


def test_unparsable_label_counts_as_unannotated() -> None:
    """`?` / 空 / 乱码 都算未标注 —— 不能猜成 True（猜成 True 就是"判官对了"）。"""
    assert cal._parse_bool("?") is None
    assert cal._parse_bool("") is None
    assert cal._parse_bool(None) is None
    assert cal._parse_bool("maybe") is None
    assert cal._parse_bool("TRUE") is True
    assert cal._parse_bool(" false ") is False
    assert cal._parse_bool("是") is True
    assert cal._parse_bool("错") is False


def test_load_annotations_reads_csv_with_bom(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    open(p, "w", encoding="utf-8-sig", newline="").write(
        "id,kind,judge,human\nr01,faithfulness,true,true\nn01,abstention,false,\n")
    rows = cal.load_annotations(p)
    # 2026-09-25 起每行多带 `note` 列（人工备注），随标注表一起回流
    assert rows[0] == {"id": "r01", "kind": "faithfulness", "judge": True, "human": True, "note": ""}
    assert rows[1]["human"] is None


# ---------------------------------------------------------------- 门禁


def test_refuses_kappa_below_min_annotated(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """样本不足必须**拒绝给结论**（返回码 3），而不是给一个宽到没用的数。"""
    rows = [{"id": f"x{i}", "kind": "verbatim", "judge": bool(i % 2), "human": bool(i % 2)}
            for i in range(5)]
    code = cal.report_kappa(rows, min_annotated=30)
    assert code == 3
    out = capsys.readouterr().out
    assert "拒绝给出" in out


def test_reports_kappa_when_enough_samples(capsys: pytest.CaptureFixture) -> None:
    rows = [{"id": f"x{i}", "kind": "verbatim", "judge": bool(i % 2), "human": bool(i % 2)}
            for i in range(40)]
    assert cal.report_kappa(rows, min_annotated=30) == 0
    out = capsys.readouterr().out
    assert "Cohen's κ" in out


def test_template_from_report_warns_about_truncation(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """降级表必须**自己声明降级**（无上下文 + 预览截断），不能让人误以为可以据此判定。"""
    rep = tmp_path / "r.json"
    rep.write_text(
        '{"judge": {"details": [{"id": "r01", "kind": "faithfulness", "value": true, '
        '"reason": "ok", "answer_preview": "' + "字" * 200 + '"}]}}', encoding="utf-8")
    out_csv = tmp_path / "t.csv"
    n = cal.make_template_from_report(rep, out_csv)
    assert n == 1
    err = capsys.readouterr().err
    assert "降级表" in err and "截断" in err
    body = open(out_csv, encoding="utf-8-sig").read()
    assert "human" in body.splitlines()[0]
    assert "截断" in body, "被截断的行必须逐行标注出来，否则人会照着一个残缺答案打分"


def test_template_from_dump_keeps_full_context(tmp_path: Path) -> None:
    dump = tmp_path / "d.jsonl"
    # ⚠️ 乘法必须在**引号外**做：写成 '"上文" * 100' 会把 `* 100` 当成 JSON 内容
    # （我第一版就这么错了，报错是 JSONDecodeError: Expecting ',' delimiter）。
    dump.write_text(
        '{"id": "r01", "kind": "faithfulness", "query": "问", "context": "'
        + "上文" * 100
        + '", "answer": "答", "judge_value": true, "judge_reason": "ok"}\n',
        encoding="utf-8")
    out_csv = tmp_path / "t.csv"
    assert cal.make_template_from_dump(dump, out_csv) == 1
    body = open(out_csv, encoding="utf-8-sig").read()
    assert "200" in body and "human" in body.splitlines()[0]
    assert "context_chars" in body.splitlines()[0]


# ---------------------------------------------------------------- 标注材料必须"可判"


def test_template_carries_context_text_not_only_length(tmp_path: Path) -> None:
    """★ 回归：CSV 必须带**上下文正文**，不能只给长度。

    第一版只放了 `context_chars` → 人工要判"答案是否忠于证据"却**看不到证据**，
    整张表等于不可用（这正是"输出格式满足了自己、却满足不了使用场景"的典型）。
    """
    dump = tmp_path / "d.jsonl"
    dump.write_text(json.dumps({
        "id": "r01", "kind": "faithfulness", "query": "混合检索怎么调优",
        "context": "第三章 调优顺序：先确认通道互补，再调融合常数。",
        "answer": "先确认通道互补，再调融合常数。",
        "judge_value": True, "judge_reason": "直接引用", "answerable": True,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    out_csv = tmp_path / "t.csv"
    cal.make_template_from_dump(dump, out_csv)
    body = open(out_csv, encoding="utf-8-sig").read()
    assert "先确认通道互补" in body, "上下文正文必须出现在表里"
    assert "第三章 调优顺序" in body


def test_annotation_sheet_is_readable_and_has_all_verdict_inputs(tmp_path: Path) -> None:
    """给人读的 sheet 必须同时含：问题、上下文、答案、判官结论 + 待填位置。"""
    dump = tmp_path / "d.jsonl"
    dump.write_text(json.dumps({
        "id": "n01", "kind": "abstention", "query": "训练用了多少卡时",
        "context": "本文讨论检索增强生成。", "answer": "资料中没有相关信息。",
        "judge_value": True, "judge_reason": "上下文未提及", "answerable": False,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    sheet = tmp_path / "s.md"
    assert cal.make_annotation_sheet(dump, sheet) == 1
    body = sheet.read_text(encoding="utf-8")
    for key in ("训练用了多少卡时", "本文讨论检索增强生成", "资料中没有相关信息",
                "判官结论", "human =", "负样本"):
        assert key in body, f"标注单缺内容：{key}"


def test_make_sheet_refuses_without_dump(capsys: pytest.CaptureFixture) -> None:
    """`--make-sheet` 只允许配 `--dump`：报告里没有上下文，据此出 sheet 会让人误判。"""
    sys.argv = ["calibrate_judge.py", "--report", "x.json", "--make-sheet", "s.md"]
    assert cal.main() == 2          # main() 返回非零码（由 __main__ 转 SystemExit）
    assert "只在 --dump 模式下可用" in capsys.readouterr().err



# ---------------------------------------------------------------- 2026-09-25 实测补充
#
# 用户交回的 40 条标注：judge 与 human **全部为 true** → κ 无定义（两侧都无方差）。
# 这不是"算不出就完了"：数据里仍有一条有信息的结论（判官错误率上界），必须报出来而不是吞掉。


def test_binom_upper_zero_errors_matches_closed_form() -> None:
    """0/40 观察到错误 → 95% 上界 = 1 − 0.05^(1/40) ≈ 7.2%（闭式解对照）。"""
    closed = 1 - 0.05 ** (1 / 40)
    assert cal.binom_upper(0, 40) == pytest.approx(closed, abs=1e-6)
    assert 0.06 < cal.binom_upper(0, 40) < 0.09


def test_binom_upper_edge_cases() -> None:
    assert cal.binom_upper(0, 0) is None
    assert cal.binom_upper(10, 10) == 1.0          # 全错 → 上界 1（结论：毫无证据）
    assert cal.binom_upper(0, 10) == pytest.approx(1 - 0.05 ** (1 / 10), abs=1e-6)
    # n 越大，同样 0 错误的上界越紧
    assert cal.binom_upper(0, 100) < cal.binom_upper(0, 40)


def test_binom_upper_with_errors_is_monotone() -> None:
    """错误越多，上界越松；观测值本身应落在上界之下。"""
    u5 = cal.binom_upper(5, 40)
    assert u5 is not None and 5 / 40 < u5 < 1.0
    assert cal.binom_upper(3, 40) < u5


def test_load_annotations_accepts_gbk_from_excel(tmp_path: Path) -> None:
    """★ 回归：用户用 Excel 填完后另存 → 文件变 **GB18030**，原实现直接崩。

    把"用户已交作业"误报成"文件坏了"，是最伤人的那种失败。
    """
    body = "id,kind,judge,human\nr01,verbatim,true,true\nn01,negative,true,true\n"
    p = tmp_path / "a.csv"
    p.write_bytes(body.encode("gb18030"))          # 模拟 Excel 的本地码页另存
    rows = cal.load_annotations(p)
    assert len(rows) == 2
    assert rows[0]["judge"] is True and rows[0]["human"] is True


def test_load_annotations_keeps_note_column(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    open(p, "w", encoding="utf-8-sig", newline="").write(
        "id,kind,judge,human,human_note\nr01,verbatim,true,true,第3步可疑\n")
    rows = cal.load_annotations(p)
    assert rows[0]["note"] == "第3步可疑"


# ---------------------------------------------------------------- 2026-09-25 实测数据回归
#
# 真实数据：原批 40（judge=T, human=T ×40）+ 压力批 15（judge=F,human=T ×8；judge=T,human=F ×1）
# 人工语义 = 「判官是否正确」→ 换算成人工对**答案**的判定 = (judge == human)。
# ★ 直接配对（不换算）会得到 κ ≈ −0.03，结论天壤之别 —— 本组测试就是把这次错误钉死。


def _real_like_rows() -> list[dict]:
    """真实合并数据构成：40 (T,T) + 6 (T,T)压力 + 8 (F,T) + 1 (T,F) = **55**。

    ⚠️ 我第一版漏了压力批里 6 条 `judge=True` 的行（少 6 个 (T,T)），fixture 49 行 → 期望值全错。
    """
    rows = [{"id": f"r{i:02d}", "kind": "verbatim", "judge": True, "human": True} for i in range(40)]
    rows += [{"id": f"t{i:02d}", "kind": "", "judge": True, "human": True} for i in range(6)]
    rows += [{"id": f"s{i:02d}", "kind": "", "judge": False, "human": True} for i in range(1, 9)]
    rows.append({"id": "s13", "kind": "", "judge": True, "human": False})
    return rows


def test_kappa_on_real_like_data_matches_hand_computed() -> None:
    """换算后：a=46 b=1 c=0 d=8 → κ = 0.9305（almost perfect）。"""
    pairs = [(r["judge"], r["human"]) for r in _real_like_rows()]
    derived = [(j, (j == h)) for j, h in pairs]
    res = cal.cohen_kappa(derived)
    assert (res["a"], res["b"], res["c"], res["d"]) == (46, 1, 0, 8)
    assert res["po"] == pytest.approx(54 / 55, abs=1e-4)
    assert res["kappa"] == pytest.approx(0.9305, abs=1e-3)


def test_naive_pairing_gives_absurd_kappa_documenting_the_bug() -> None:
    """**反向自检**：不换算直接配对 → κ ≈ −0.03（与 0.9305 天壤之别）。

    把这个差异钉在测试里，防止将来有人"简化"掉换算逻辑而不自知。
    """
    naive = cal.cohen_kappa([(r["judge"], r["human"]) for r in _real_like_rows()])
    assert naive["kappa"] == pytest.approx(-0.033, abs=0.02), "若此断言失败，说明实现已变，需重新评估"


def test_report_kappa_counts_only_human_false_as_judge_error(capsys: pytest.CaptureFixture) -> None:
    """判官错误 = `human=false` 的条数（**不是** `judge != human` 的条数）。"""
    assert cal.report_kappa(_real_like_rows(), min_annotated=30) == 0
    out = capsys.readouterr().out
    assert "判官正确率 = 0.9818（54/55）" in out
    assert "κ = 0.9305" in out
    assert out.count("s13") >= 1, "分歧清单必须列出唯一的人工判错条目"


def test_report_kappa_lists_only_s13_not_the_eight_agreements(capsys: pytest.CaptureFixture) -> None:
    """回归：分歧清单曾把 8 条『人认可判官判不忠实』的**一致**样本也列进去（旧语义残留）。"""
    cal.report_kappa(_real_like_rows(), min_annotated=30)
    out = capsys.readouterr().out
    seg = out.split("分歧清单")[1]
    for i in range(1, 9):
        assert f"s{i:02d}" not in seg, f"s{i:02d} 是一致样本，不该出现在分歧清单里"
    assert "s13" in seg
