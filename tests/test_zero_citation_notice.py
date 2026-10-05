"""doc-bound 会话「零引用」提示（2026-09-23）。

**守的是什么**：先验知识幻觉的特征是「回答里没有引用」—— 而 L1 引用校验**只在有引用时才跑**，
于是最该拦的场景反而静默通过。`zero_citation_notice` 补的是这一刀。

边界（长短 / 绑没绑文档）是这段逻辑最容易写错的地方，所以逐条锁住。
"""

from __future__ import annotations

from papermate.services.verifier import (
    ZERO_CITE_MIN_LEN,
    zero_citation_notice,
)

LONG = "这是一段足够长的回答。" * 20  # 远超阈值
SHORT = "好的。"


def test_doc_bound_long_answer_without_cites_triggers_notice():
    """★ 核心场景：绑了文档、回答不短、却一个引用都没有 → 必须提示。"""
    out = zero_citation_notice(doc_bound=True, final_text=LONG, has_cites=False)
    assert out is not None
    issue = out["issues"][0]
    assert issue["severity"] == "warn", "不能用 suspect —— 零引用不等于引用错误，只是提示"
    assert issue["scope"] == "citations"
    assert "未引用任何文档片段" in issue["reason"]


def test_with_cites_never_triggers():
    """有引用时把判断权交给 L1 本身，不由本函数代劳（否则会重复提示）。"""
    assert zero_citation_notice(doc_bound=True, final_text=LONG, has_cites=True) is None


def test_not_doc_bound_never_triggers():
    """★ 未绑文档时本就不该要求引用 —— 否则普通聊天每轮都弹，用户会学会无视它。"""
    assert zero_citation_notice(doc_bound=False, final_text=LONG, has_cites=False) is None


def test_short_answer_does_not_trigger():
    """★ 短应答（"好的"/"收到"）不提示 —— 这是防「警示疲劳」的关键限定。"""
    assert zero_citation_notice(doc_bound=True, final_text=SHORT, has_cites=False) is None


def test_length_boundary_is_strictly_greater_than_threshold():
    """边界：正好等于阈值**不**触发（实现用的是 `<=`），差一字符才触发。

    锁边界是因为这类 off-by-one 最容易在后续重构里被改掉，而改了没人发现。
    """
    at_limit = "字" * ZERO_CITE_MIN_LEN
    just_over = "字" * (ZERO_CITE_MIN_LEN + 1)
    assert zero_citation_notice(True, at_limit, False) is None
    assert zero_citation_notice(True, just_over, False) is not None


def test_blank_and_none_text_are_safe():
    """空 / None 文本不得抛异常，也不得触发（模型偶发空回答时不能把流程带崩）。"""
    assert zero_citation_notice(True, "", False) is None
    assert zero_citation_notice(True, "   \n  ", False) is None
    assert zero_citation_notice(True, None, False) is None  # type: ignore[arg-type]


def test_whitespace_is_stripped_before_measuring():
    """测量长度前要 strip —— 否则"用空白凑长度"就能绕过提示。"""
    padded = "字" * ZERO_CITE_MIN_LEN + " " * 500
    assert zero_citation_notice(True, padded, False) is None
