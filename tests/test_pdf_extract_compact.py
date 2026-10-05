"""P1-b（2026-10-03）：`pdf_extract` 注入量压缩 —— 行为测试。

## 为什么需要
`pdf_extract` 此前是**唯一完全不压缩**的注入源：`retrieve` 每段压到 200 字，
而它 `text` **原样直灌**（实测日志「原始=N 字 → 注入=N+8 字」，1:1 无压缩）。
前序实测它占注入 token 的 **46%**，单次 13,498 字（≈4.9k tokens）能吃掉单轮预算的 16%。

## 取值依据（实测，非拍脑袋）
`data/logs/app.log` 实测两工具注入量：
- `pdf_extract`：1836 / 2659 / 2859 / 3014 / 3089 / 3129 字，**极端 11057 与 13498 字**；
- `retrieve`（对照）：1112~3105 字。
⇒ 上限取 **4000**：保住正常 page 查询（实测上限 3129 < 4000），压住极端值。

## ★ 与 retrieve 的**刻意不同**
按**总量**设卡（不是逐段 200 字）——`pdf_extract` 的语义是"取本页/本节全文"，
逐段砍会让"整节翻译"失效。
"""
from __future__ import annotations

import pytest

from lightcore.core.loop import (
    _PDF_EXTRACT_MAX_TOTAL,
    _PDF_EXTRACT_TRUNCATED_FLAG,
    _compact_pdf_extract_output,
    _prepare_tool_output,
)

#: 实测正常值（必须**逐字不变**）
MEASURED_NORMAL = [1836, 2659, 2859, 3014, 3089, 3129]
#: 实测极端值（必须被截断）
MEASURED_HUGE = [11057, 13498]


class TestNormalSizesUntouched:
    """★ 最高优先：**正常用量必须逐字不变**（否则会砸掉"整节翻译"等既有能力）。"""

    @pytest.mark.parametrize("n", MEASURED_NORMAL)
    def test_measured_normal_pages_pass_through(self, n):
        out = {"page": 5, "text": "x" * n, "num_chunks": 3, "source": "pdf"}
        got = _compact_pdf_extract_output(out)
        assert got["text"] == out["text"], f"{n} 字被改动 —— 正常用量不该受影响"
        assert "note" not in got, "正常用量不该出现截断提示"

    def test_boundary_exactly_at_limit(self):
        """恰好等于上限 ⇒ 不截断（边界含等号，不许多截）。"""
        out = {"page": 1, "text": "x" * _PDF_EXTRACT_MAX_TOTAL, "source": "pdf"}
        got = _compact_pdf_extract_output(out)
        assert got["text"] == out["text"] and "note" not in got

    def test_limit_covers_all_measured_normal_values(self):
        """★ 上限必须**严格大于**所有实测正常值（回归网：有人把 4000 调小就会红）。"""
        assert _PDF_EXTRACT_MAX_TOTAL > max(MEASURED_NORMAL), (
            f"上限 {_PDF_EXTRACT_MAX_TOTAL} 已不大于实测正常上限 {max(MEASURED_NORMAL)}"
            " ⇒ 会开始削正常用量的内容"
        )


class TestHugeSizesTruncated:
    @pytest.mark.parametrize("n", MEASURED_HUGE)
    def test_measured_huge_pages_are_cut(self, n):
        out = {"page": 3, "text": "y" * n, "num_chunks": 5, "source": "pdf"}
        got = _compact_pdf_extract_output(out)
        assert len(got["text"]) <= _PDF_EXTRACT_MAX_TOTAL + 1  # +1 = 省略号
        assert len(got["text"]) < n, f"{n} 字未被截断 ⇒ P1-b 被绕过"

    def test_truncation_adds_rescue_hint(self):
        """★★ 截断**必须**告诉模型"还能怎么取"。

        只截断不给指引 = **静默降准**：模型会以为"文档就这么多"，
        进而回答"文档中未提及"。这正是本项目一贯警惕的失效模式。
        """
        out = {"page": 3, "text": "y" * 9999, "source": "pdf"}
        note = _compact_pdf_extract_output(out).get("note") or ""
        assert _PDF_EXTRACT_TRUNCATED_FLAG in note
        assert "pdf_extract(page=" in note and "retrieve(" in note, "必须给出两条自救路径"
        assert "不要" in note, "必须显式阻止模型得出『文档没有』的结论"

    def test_hint_survives_existing_note(self):
        """已有 note（如页码越界提示）不得被覆盖，只能追加。"""
        out = {"page": 3, "text": "y" * 9999, "source": "pdf", "note": "原有提示"}
        note = _compact_pdf_extract_output(out)["note"]
        assert "原有提示" in note and _PDF_EXTRACT_TRUNCATED_FLAG in note


class TestIdempotence:
    """★ 硬要求：`_prepare_tool_output` 会被调用**两次**（调用点 + `_format_tool_result`）。"""

    def test_second_pass_changes_nothing(self):
        out = {"page": 3, "text": "y" * 20000, "source": "pdf"}
        once = _compact_pdf_extract_output(out)
        twice = _compact_pdf_extract_output(dict(once))
        assert twice["text"] == once["text"], "二次截断改变了正文"
        assert twice["note"] == once["note"], "二次调用重复拼接了 note"

    def test_flag_not_duplicated(self):
        out = {"page": 3, "text": "y" * 20000, "source": "pdf"}
        once = _compact_pdf_extract_output(out)
        twice = _compact_pdf_extract_output(dict(once))
        assert twice["note"].count(_PDF_EXTRACT_TRUNCATED_FLAG) == 1


class TestQueryMode:
    def test_total_budget_across_results(self):
        """query 模式：**总量**卡（不是每条各给 4000）。

        ⚠️ 判据只算**正文**：额度用尽后的条目会被换成占位符
        （「（因注入上限未展开）」），那些占位符本身也占字符，
        把它们算进预算会让断言变成"看占位符多长"而不是"正文受不受控"。
        """
        out = {
            "query": "q", "source": "pdf",
            "results": [{"id": f"d-{i}", "page": i, "text": "z" * 2000} for i in range(5)],
        }
        got = _compact_pdf_extract_output(out)
        body = sum(len(r["text"]) for r in got["results"] if not r["text"].startswith("（因注入"))
        assert body <= _PDF_EXTRACT_MAX_TOTAL + 1, f"正文总量未受控：{body}"

    def test_all_items_keep_identity(self):
        """★ 额度用尽后**保留条目结构**（id/page），不得直接丢条。

        丢掉整条 ⇒ 模型看不到"这条存在" ⇒ 无法判断是否漏了内容、也无法再精确取。
        """
        out = {
            "query": "q", "source": "pdf",
            "results": [{"id": f"d-{i}", "page": i, "text": "z" * 3000} for i in range(4)],
        }
        got = _compact_pdf_extract_output(out)
        assert len(got["results"]) == 4, "条目被丢弃 ⇒ 模型无法感知遗漏"
        assert [r["id"] for r in got["results"]] == ["d-0", "d-1", "d-2", "d-3"]
        assert all(isinstance(r.get("page"), int) for r in got["results"])

    def test_small_query_results_untouched(self):
        out = {
            "query": "q", "source": "pdf",
            "results": [{"id": "d-0", "page": 1, "text": "z" * 300}],
        }
        got = _compact_pdf_extract_output(out)
        assert got["results"][0]["text"] == "z" * 300
        assert "note" not in got


class TestRobustness:
    @pytest.mark.parametrize("bad", [None, {}, [], "text", 42, {"text": None}, {"results": None}])
    def test_non_dict_or_missing_fields_do_not_raise(self, bad):
        assert _compact_pdf_extract_output(bad) == bad

    def test_text_not_str_is_ignored(self):
        out = {"page": 1, "text": 12345, "source": "pdf"}
        assert _compact_pdf_extract_output(out) == out


class TestWiring:
    """★★ 走**真实入口** `_prepare_tool_output` —— 防止"函数写了但没接上"。"""

    def test_prepare_tool_output_routes_pdf_extract(self):
        out = {"page": 1, "text": "y" * 20000, "source": "pdf"}
        got = _prepare_tool_output("pdf_extract", out)
        assert len(got["text"]) <= _PDF_EXTRACT_MAX_TOTAL + 1, (
            "`_prepare_tool_output` 没把 pdf_extract 路由到压缩函数 ⇒ P1-b 静默失效"
        )

    def test_prepare_tool_output_still_routes_retrieve(self):
        """回归网：retrieve 的既有压缩不得被改动。"""
        out = {"results": [{"id": "c", "page": 1, "text": "t" * 500, "score": 0.5}]}
        got = _prepare_tool_output("retrieve", out)
        assert len(got["results"][0]["text"]) <= 201

    def test_unknown_tool_passes_through(self):
        out = {"anything": "x" * 99999}
        assert _prepare_tool_output("calculator", out) == out
