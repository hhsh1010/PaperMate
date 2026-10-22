"""一键总结「注入上下文」的压缩判据（★ 2026-10-22 实测：摘要比原文还长）。

线上真实故障：5 篇文档的 summarize 工具输出 **78323 字 ≈ 35k tokens**（原始内容 78315 字），
把单次 prompt 从 21k 顶到 60k、那轮成本 ¥0.104（平时 ¥0.02~0.07）。
根因：`summarize()` 返回的 sections（确定性抽取，逐段接近原文）+ citations（每篇 240 字）
被工具 handler **原样**塞进 ToolResult ⇒ "摘要"退化成"整篇直灌"。

本文件只测**纯函数** `_compact_summary_for_agent`（完整结果仍由 `/summary` 接口提供）。
"""

from __future__ import annotations

import json

from lightcore.tools.builtin.domain import (
    _SUMMARY_INJECT_MAX,
    _clip,
    _compact_summary_for_agent,
)


def _big(**over) -> dict:
    base = {
        "doc_ids": [1, 2, 3, 4, 5],
        "sections": [{"key": k, "title": k, "content": "原文" * 4000} for k in "abcde"],
        "narrative": "综述" * 5000,
        "citations": [{"filename": "f.pdf", "page": 1, "text": "x" * 240}] * 20,
        "model": "llm",
    }
    base.update(over)
    return base


def test_clip_short_unchanged() -> None:
    assert _clip("短文本", 100) == "短文本"


def test_clip_long_truncated_with_ellipsis() -> None:
    out = _clip("啊" * 1000, 400)
    assert len(out) == 401 and out.endswith("…")


def test_compaction_shrinks_below_budget() -> None:
    """★ 核心：56000 字的输入必须压到上限以内。"""
    out = _compact_summary_for_agent(_big())
    assert len(json.dumps(out, ensure_ascii=False)) <= _SUMMARY_INJECT_MAX


def test_sections_each_clipped() -> None:
    out = _compact_summary_for_agent(_big())
    assert len(out["sections"]) == 5
    assert all(len(s["content"]) <= 401 for s in out["sections"])


def test_citations_limited_to_three() -> None:
    out = _compact_summary_for_agent(_big())
    assert len(out["citations"]) == 3
    assert "text" not in out["citations"][0]        # 只留 filename/page，不带正文


def test_narrative_clipped_and_marked_truncated() -> None:
    out = _compact_summary_for_agent(_big())
    assert len(out["narrative"]) <= 801
    assert out["truncated"] is True
    assert "右侧面板" in out["note"]                 # 告诉模型"完整版在哪"


def test_non_dict_passthrough() -> None:
    assert _compact_summary_for_agent("原始字符串") == "原始字符串"
    assert _compact_summary_for_agent(None) is None


def test_related_work_draft_kept() -> None:
    """★ related_work 的主产物是 `draft` —— 压缩必须保留它（否则 KeyError: 'draft'）。"""
    raw = {**_big(), "draft": "相关工作总结" * 1000}
    out = _compact_summary_for_agent(raw)
    assert "draft" in out
    assert len(out["draft"]) <= 1201
    assert out["doc_ids"] == raw["doc_ids"]        # 其它键也保留
