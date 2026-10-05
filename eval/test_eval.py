"""A-01 评测指标单元测试（纯函数，离线可跑，无嵌入/LLM 依赖）。

覆盖两层指标：
- 检索层（IR）：recall / precision / mrr / hit / average_precision / ndcg
- Agent 层：citation_rate / tool_success_rate / aggregate_events / summarize_results

断言全部用**可手算的常量**（不写"约等于"），保证指标定义被真正锁住 ——
若哪天有人改了公式（如 nDCG 的对数底、AP 的分母），这里必须红。
"""

import pytest
from metrics import (
    aggregate_events,
    average_precision,
    citation_rate,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarize_results,
    tool_success_rate,
)

# ---------- 检索层：Recall / Precision / MRR ----------


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d", "e"]
    assert recall_at_k(retrieved, ["b"], 5) == 1.0
    assert recall_at_k(retrieved, ["x"], 5) == 0.0
    assert recall_at_k(retrieved, ["c", "x"], 5) == 0.5
    assert recall_at_k(retrieved, [], 5) == 0.0  # gold 为空 → 0.0（不抛错）


def test_mrr_at_k():
    assert mrr_at_k(["x", "a", "b"], ["a"], 5) == 0.5  # 第二个命中 → 1/2
    assert mrr_at_k(["a", "b"], ["a"], 5) == 1.0
    assert mrr_at_k(["x", "y"], ["a"], 5) == 0.0
    assert mrr_at_k(["x", "a"], ["a"], 1) == 0.0  # 截断到 k=1 后未命中


def test_precision_at_k():
    retrieved = ["a", "b", "x", "y", "z"]
    assert precision_at_k(retrieved, ["a", "b"], 3) == 2 / 3
    assert precision_at_k(retrieved, ["a"], 2) == 0.5
    assert precision_at_k([], ["a"], 5) == 0.0  # 空结果 → 0.0（分母不取 k，避免除零）


# ---------- 检索层：Hit@k / AP / nDCG ----------


def test_hit_at_k():
    assert hit_at_k(["x", "a"], ["a"], 5) == 1.0
    assert hit_at_k(["x", "y"], ["a"], 5) == 0.0
    assert hit_at_k(["x", "a"], ["a"], 1) == 0.0  # k 截断后看不到
    assert hit_at_k(["a"], [], 5) == 0.0


def test_average_precision_hand_computed():
    # retrieved = [a, x, b]，gold = {a, b}
    #   rank1 a 命中 → precision@1 = 1/1
    #   rank3 b 命中 → precision@3 = 2/3
    #   AP = (1 + 2/3) / 2 = 0.8333...
    ap = average_precision(["a", "x", "b"], ["a", "b"])
    assert ap == pytest.approx(5 / 6, abs=1e-6)


def test_average_precision_edge_cases():
    assert average_precision(["x", "y"], ["a"]) == 0.0
    assert average_precision([], ["a"]) == 0.0
    assert average_precision(["a", "b"], ["a", "b"]) == 1.0  # 全部命中且都在最前


def test_ndcg_perfect_and_degraded_ranking():
    # 完美排序：命中都在最前 → nDCG = 1.0
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 5) == pytest.approx(1.0, abs=1e-9)
    # 命中被推到第 2 位：DCG = 1/log2(3)，IDCG = 1 → ≈0.6309
    assert ndcg_at_k(["x", "a"], ["a"], 5) == pytest.approx(1 / 1.5849625007211563, abs=1e-6)
    # 完全未命中 → 0.0（且不因除零抛错）
    assert ndcg_at_k(["x", "y"], ["a"], 5) == 0.0
    assert ndcg_at_k(["x"], [], 5) == 0.0


def test_ndcg_graded_gains():
    """分级相关性：高相关文档排在前面应优于低相关文档排在前面。"""
    gains = {"a": 3, "b": 1}
    good = ndcg_at_k(["a", "b"], ["a", "b"], 5, gains=gains)   # 高相关在前
    bad = ndcg_at_k(["b", "a"], ["a", "b"], 5, gains=gains)    # 低相关在前
    assert good == pytest.approx(1.0, abs=1e-9)
    assert bad < good


# ---------- 兼容性：结果可以是 RetrievedChunk / dict / 裸字符串 ----------


def test_metrics_accept_chunk_like_and_dict():
    class _Chunk:
        def __init__(self, cid: str) -> None:
            self.id = cid

    assert recall_at_k([_Chunk("a")], ["a"], 5) == 1.0
    assert recall_at_k([{"id": "a"}], ["a"], 5) == 1.0
    assert recall_at_k(["a"], ["a"], 5) == 1.0


# ---------- Agent 层与聚合 ----------


def test_citation_rate():
    assert citation_rate([{"citations": [1]}, {"citations": []}]) == 0.5
    assert citation_rate([]) == 0.0


def test_tool_success_rate():
    events = [
        {"type": "tool_call", "data": {}},
        {"type": "tool_result", "data": {"ok": True}},
        {"type": "tool_call", "data": {}},
        {"type": "tool_result", "data": {"ok": False}},
    ]
    assert tool_success_rate(events) == 0.5
    assert tool_success_rate([{"type": "message"}]) == 1.0


def test_aggregate_events():
    events = [
        {"type": "thinking"},
        {"type": "thinking"},
        {"type": "done", "data": {"steps": 2, "total_tokens": 100, "cost": 0.01, "latency_ms": 800}},
    ]
    agg = aggregate_events(events)
    assert agg["steps"] == 2
    assert agg["cost"] == 0.01


def test_summarize_results_only_includes_present_metrics():
    """只汇总**实际存在**的指标键 —— 没测的项不出现，避免报告里出现误导性的 0.0。"""
    rows = [
        {"id": "q1", "recall@5": 1.0, "ndcg@5": 0.8, "latency_ms": 10.0},
        {"id": "q2", "recall@5": 0.0, "ndcg@5": 0.6, "latency_ms": 30.0},
    ]
    summary = summarize_results(rows)
    assert summary["recall@5"] == 0.5
    assert summary["ndcg@5"] == 0.7
    assert summary["latency_ms"] == 20.0
    assert "citation_rate" not in summary  # 未测的项不出现
    assert summarize_results([]) == {}


# ---------- 分级相关性（graded relevance）的边界 ----------


def test_ndcg_gains_beyond_gold_stays_within_one():
    """`gains` 含「非 gold」的相关项时，IDCG 必须把它们计入 —— 否则 nDCG 会 > 1（荒谬值）。

    这是 2026-09-15 修掉的实现缺陷：原实现只用 gold 集合算 IDCG，
    当分级标注里存在 gold 之外的次级相关段落时，IDCG 偏小 → nDCG 突破 1.0。
    """
    gains = {"a": 3, "b": 2, "c": 1}  # gold 只有 a，但 b/c 也相关
    # 完美排序 → 恰好 1.0
    assert ndcg_at_k(["a", "b", "c"], ["a"], 5, gains=gains) == pytest.approx(1.0, abs=1e-9)
    # 任意排序都必须落在 [0,1]
    for order in (["a", "c", "b"], ["b", "a", "c"], ["c", "b", "a"], ["x", "a"], ["a"]):
        val = ndcg_at_k(order, ["a"], 5, gains=gains)
        assert 0.0 <= val <= 1.0 + 1e-9, f"{order} → {val}"


def test_ndcg_graded_distinguishes_secondary_hits():
    """分级 nDCG 的**独有价值**：区分「前面的位置被次级相关占据」与「被无关内容占据」。

    二值 nDCG 看不到这个差别（grade 1 与「无关」都算 0）—— 这是启用分级标注的理由。
    """
    gains = {"a": 3, "b": 1}
    secondary_first = ndcg_at_k(["b", "a"], ["a"], 5, gains=gains)  # 次级相关排在前
    unrelated_first = ndcg_at_k(["x", "a"], ["a"], 5, gains=gains)  # 无关内容排在前
    assert secondary_first > unrelated_first
    # 对照：二值版本对这两种情况**给出相同分数**（所以它看不出差别）
    binary_secondary = ndcg_at_k(["b", "a"], ["a"], 5)
    binary_unrelated = ndcg_at_k(["x", "a"], ["a"], 5)
    assert binary_secondary == pytest.approx(binary_unrelated, abs=1e-12)
