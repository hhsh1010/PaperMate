"""LightCore · 规划与反思单元测试（docs/08 单元层 / M2）"""

from __future__ import annotations

import asyncio

from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, Role
from lightcore.planning.planner import Plan, Planner
from lightcore.planning.reflector import Reflection, Reflector


class PlanFakeLLM(BaseLLM):
    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None) -> LLMResponse:
        last = messages[-1].content if messages else ""
        if "规划器" in last or "子任务" in last:
            content = "1. 解析论文 PDF\n2. 检索相关段落 [1]\n3. 生成摘要 [2]"
        elif "评价" in last or "执行结果" in last:
            content = "NEEDS_IMPROVEMENT\n反思: 摘要缺少引用\n改进: 补充引用标注"
        else:
            content = "ok"
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=content),
                           prompt_tokens=5, completion_tokens=5, model="fake", latency_ms=1)


def test_planner_returns_steps_within_limit():
    async def run():
        p = Planner(PlanFakeLLM(), max_steps=5)
        plan = await p.plan("总结这篇论文并给出引用")
        assert isinstance(plan, Plan)
        assert 1 <= len(plan) <= 5
        assert plan.steps[0].description == "解析论文 PDF"
        # 依赖解析：步骤2 依赖步骤1
        assert plan.steps[1].depends_on == [1]

    asyncio.run(run())


def test_reflector_parses_needs_improvement():
    async def run():
        r = Reflector(PlanFakeLLM(), max_rounds=2)
        ref = await r.reflect("总结论文", "摘要文本")
        assert isinstance(ref, Reflection)
        assert ref.satisfied is False
        assert "引用" in ref.critique
        assert "引用" in ref.improvement

    asyncio.run(run())


def test_reflector_satisfied_on_good_result():
    async def run():
        class OkFake(PlanFakeLLM):
            async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None):
                return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="SATISFIED"),
                                   prompt_tokens=1, completion_tokens=1, model="fake")

        r = Reflector(OkFake())
        ref = await r.reflect("x", "y")
        assert ref.satisfied is True

    asyncio.run(run())


# ---------------------------------------------- C6（2026-09-21）：depends_on 真正执行


def _steps(*specs):
    """按 (index, depends_on) 造步骤；描述用 index 字母便于断言。"""
    return [
        {"index": i, "description": f"step{chr(96 + i)}", "depends_on": list(deps)}
        for i, deps in specs
    ]


def test_topo_order_respects_depends_on():
    """★ 乱序给出的依赖图必须被排成可执行顺序（此前完全按原顺序跑）。"""
    from lightcore.tools.builtin.domain import _resolve_order

    ordered, warnings = _resolve_order(_steps((3, [2]), (2, [1]), (1, [])))
    assert [s["index"] for s in ordered] == [1, 2, 3]
    assert any("重排" in w for w in warnings)


def test_topo_order_keeps_original_order_when_no_deps():
    """无依赖时不得重排（保持输出稳定，避免无谓变动）。"""
    from lightcore.tools.builtin.domain import _resolve_order

    steps = _steps((1, []), (2, []), (3, []))
    ordered, warnings = _resolve_order(steps)
    assert [s["index"] for s in ordered] == [1, 2, 3]
    assert warnings == []


def test_topo_order_cycle_falls_back_instead_of_hanging():
    """★ 成环必须降级为原顺序并告警 —— 不能因为 LLM 写了环状依赖就卡死/丢步骤。"""
    from lightcore.tools.builtin.domain import _resolve_order

    ordered, warnings = _resolve_order(_steps((1, [2]), (2, [1])))
    assert [s["index"] for s in ordered] == [1, 2], "成环时保留全部步骤且维持原序"
    assert any("环" in w for w in warnings)


def test_topo_order_ignores_invalid_and_self_deps():
    """无效引用与自依赖要被忽略并告警（LLM 生成的依赖图不可信）。"""
    from lightcore.tools.builtin.domain import _resolve_order

    ordered, warnings = _resolve_order(_steps((1, [99, 1]), (2, [])))
    assert [s["index"] for s in ordered] == [1, 2]
    assert any("不存在" in w for w in warnings)
    assert any("自身" in w for w in warnings)


def _run_with_rag(monkeypatch, steps, rag):
    import asyncio

    from lightcore.llm.fake import FakeLLM
    from lightcore.tools.builtin import domain

    monkeypatch.setattr(domain, "_rule_decompose", lambda *a, **k: steps)
    return asyncio.run(domain.run_planning("目标", FakeLLM(), rag, max_steps=5))


def test_dependency_failure_skips_downstream(monkeypatch):
    """★ 前置步骤**服务故障** → 依赖它的步骤跳过，不依赖的照常执行。"""
    from lightcore.rag.pipeline import RetrievalUnavailable

    class _FailOnA:
        def query(self, q, top_k=3):
            if q == "stepa":
                raise RetrievalUnavailable("检索服务故障")
            return []

    out = _run_with_rag(
        monkeypatch, _steps((1, []), (2, [1]), (3, [])), _FailOnA()
    )
    by = {e["step"]: e for e in out["execution"]}
    assert by[1]["status"] == "failed"
    assert by[2]["status"] == "skipped"
    assert "前置步骤" in by[2]["reason"]
    assert by[3]["status"] == "ok", "与失败步骤无依赖关系的步骤不得受影响"
    assert [s["step"] for s in out["skipped_steps"]] == [2]
    assert "未能完成" in out["note"], "note 要提示模型说明未完成部分"


def test_empty_evidence_is_not_a_failure(monkeypatch):
    """★★ 最关键的一条：「无证据」不是失败。

    沿用既有 K4 口径 —— "文档里没有"（evidence 为空）是**有效结论**，
    不是执行失败。若把它当失败，凡是文档没写的计划都会被判为"执行失败"并传染下游。
    """
    class _EmptyRAG:
        def query(self, q, top_k=3):
            return []

    out = _run_with_rag(monkeypatch, _steps((1, []), (2, [1])), _EmptyRAG())
    by = {e["step"]: e for e in out["execution"]}
    assert by[1]["evidence"] == [] and by[1]["status"] == "ok"
    assert by[2]["status"] == "ok", "上游无证据 ≠ 上游失败，下游不该被跳过"
    assert "skipped_steps" not in out


def test_skipped_steps_carry_reason(monkeypatch):
    """跳过必须带原因 —— 否则"这步为什么没跑"无从解释。"""
    from lightcore.rag.pipeline import RetrievalUnavailable

    class _FailOnA:
        def query(self, q, top_k=3):
            if q == "stepa":
                raise RetrievalUnavailable("boom")
            return []

    out = _run_with_rag(monkeypatch, _steps((1, []), (2, [1])), _FailOnA())
    rec = out["skipped_steps"][0]
    assert rec["step"] == 2 and rec["reason"]


def test_planning_still_reports_retrieval_unavailable(monkeypatch):
    """既有 K4 行为不得回归：服务故障要显式标记，避免被解读成"文档没有"。"""
    from lightcore.rag.pipeline import RetrievalUnavailable

    class _FailAll:
        def query(self, q, top_k=3):
            raise RetrievalUnavailable("boom")

    out = _run_with_rag(monkeypatch, _steps((1, []), (2, [])), _FailAll())
    assert out["retrieval_unavailable"] is True
    assert "检索服务暂时不可用" in out["note"]
