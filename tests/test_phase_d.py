"""阶段 D 测试（docs/统一方案 阶段D）：S9 复杂规划 / S10 开发评测 / 成本熔断 + trace 精确化。

覆盖：
- plan_executor 工具：fake 规则分解（≤5 步 + 逐子任务取证）、空目标报错、门控（关闭不注册）；
- planning 命令技能：SkillRouter per-skill 门控（features.planning / features.dev_eval）、/规划 运行；
- dev_eval 技能：读 eval_report 出指标卡 + 门禁判定、无报告降级；
- 成本熔断：LoopConfig.max_cost_per_turn 超限 → ABORT error；
- done 事件精确聚合：cost/total_tokens/steps 由 LLMResponse 累加回传；
- runtime_settings：7 键默认 + cost_limit_per_turn + update 合并语义。

使用替身，无需网络与 API key。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from importlib import import_module
from pathlib import Path
from unittest import mock

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, Role
from lightcore.llm.fake import FakeLLM
from lightcore.rag.pipeline import RetrievedChunk
from lightcore.skills.spec import SkillContext
from lightcore.tools.builtin.domain import register_planning_tools, run_planning
from lightcore.tools.spec import ToolExecutor, ToolRegistry
from papermate import runtime_settings
from papermate.skills import default_router

# 注意：papermate.skills.__init__ 把 dev_eval 名字绑定为 SkillSpec 对象，
# 因此必须经 import_module 取真实模块（sys.modules）才能 mock 其模块级常量。
dev_eval_mod = import_module("papermate.skills.dev_eval")


# ---------------- S9 复杂规划：fake 规则分解 ----------------
def test_planning_fake_rule_decompose():
    async def run():
        out = await run_planning("调研注意力机制，然后复现其长上下文实验", FakeLLM(), None, max_steps=5)
        assert out["ok"] is True
        assert 1 <= out["num_steps"] <= 5
        # 至少分出两个子任务（按连词切分）
        assert out["num_steps"] >= 2
        assert all(s["index"] >= 1 for s in out["plan"])
        # 无 RAG → 每步取证为空数组而非报错
        assert all(ex["evidence"] == [] for ex in out["execution"])

    asyncio.run(run())


def test_planning_rule_decompose_caps_steps():
    async def run():
        out = await run_planning("A和B和C和D和E和F和G", FakeLLM(), None, max_steps=3)
        assert out["num_steps"] <= 3
        assert len(out["plan"]) <= 3

    asyncio.run(run())


def test_planning_empty_goal_error():
    async def run():
        out = await run_planning("   ", FakeLLM(), None)
        assert out["ok"] is False
        assert "目标" in out["error"]

    asyncio.run(run())


def test_planning_real_llm_fallback_to_rules():
    """真实模型路径：Planner 解析失败（替身返回非 JSON）→ 回退规则分解，仍产出结构化计划。"""

    class _JunkLLM(BaseLLM):
        model_name = "stub"

        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="不是JSON"), model="stub")

    async def run():
        out = await run_planning("先做A再完成B", _JunkLLM(), None, max_steps=5)
        assert out["ok"] is True
        assert out["num_steps"] >= 1
        assert all(s["description"] for s in out["plan"])

    asyncio.run(run())


# ---------------- plan_executor 工具 + 门控 ----------------
def test_plan_executor_registered_and_gated():
    reg = ToolRegistry()
    register_planning_tools(reg, FakeLLM(), None, enabled=True)
    spec = reg.get("plan_executor")
    assert spec is not None
    assert spec.permission.value == "safe"

    reg2 = ToolRegistry()
    register_planning_tools(reg2, FakeLLM(), None, enabled=False)
    try:
        reg2.get("plan_executor")
        raise AssertionError("features.planning=False 时不应注册 plan_executor")
    except KeyError:
        pass


def test_plan_executor_executes():
    async def run():
        reg = ToolRegistry()
        register_planning_tools(reg, FakeLLM(), None, enabled=True)
        ex = ToolExecutor(reg)
        r = await ex.execute("plan_executor", {"goal": "调研 Transformer"})
        assert r.ok is True
        assert isinstance(r.output, dict) and r.output["num_steps"] >= 1
        assert "note" in r.output

    asyncio.run(run())


# ---------------- SkillRouter per-skill 门控 ----------------
def test_skill_router_per_skill_gating():
    r = default_router()
    # planning 开启 → 命中
    assert r.route("/规划 调研X", {"skills": True, "planning": True}).name == "planning"
    # planning 关闭 → 回落到普通对话（None）
    assert r.route("/规划 调研X", {"skills": True, "planning": False}) is None
    # master skills 关闭 → 全部回落
    assert r.route("/规划 调研X", {"skills": False, "planning": True}) is None
    # dev_eval 默认关闭 → 不可路由
    assert r.route("/eval", {"skills": True, "dev_eval": False}) is None
    assert r.route("/eval", {"skills": True, "dev_eval": True}).name == "dev_eval"
    # 无 feature 依赖的技能不受 planning/dev_eval 开关影响
    assert r.route("/综述 注意力", {"skills": True, "planning": False, "dev_eval": False}).name == "literature_review"


# ---------------- /规划 技能运行 ----------------
def test_planning_skill_requires_goal():
    async def run():
        r = default_router()
        spec = r.route("/规划", {"skills": True, "planning": True})
        assert spec is not None
        ctx = SkillContext(query="/规划", args="", doc_ids=[], llm=FakeLLM())
        res = await spec.run(ctx)
        assert res.ok is False
        assert "目标" in (res.error or "")

    asyncio.run(run())


def test_planning_skill_runs_rule_decompose():
    async def run():
        r = default_router()
        spec = r.route("/规划 调研注意力机制并复现实验", {"skills": True, "planning": True})
        ctx = SkillContext(query="/规划 调研注意力机制并复现实验", args="调研注意力机制并复现实验",
                           doc_ids=[], llm=FakeLLM())
        res = await spec.run(ctx)
        assert res.ok is True
        assert "任务规划" in res.content
        assert "子任务" in res.content

    asyncio.run(run())


def test_planning_evidence_truncated_at_sentence():
    """run_planning 取证文本应使用句子级截断，避免硬字符截断产生半截证据。"""

    class _LongEvidenceRAG:
        def query(self, q: str, top_k: int = 3):  # noqa: ARG002
            long_sent = (
                "注意力机制通过 query、key、value 三个投影计算相似度，"
                "再经过 softmax 和加权求和得到输出，在多头设置下可以并行关注不同子空间的信息，"
                "因此在长上下文扩展方面具有显著优势。"
            )
            return [RetrievedChunk(id="c1", text=long_sent, page=3, score=0.9)]

    async def run():
        out = await run_planning("调研注意力机制", FakeLLM(), _LongEvidenceRAG(), max_steps=1)
        assert out["ok"] is True
        assert out["execution"]
        ev_text = out["execution"][0]["evidence"][0]["text"]
        # 证据应结束在句末标点或省略号，不应在句中字符处截断
        assert ev_text[-1] in "。！？!?；;.…"
        # 命中句子较短，应完整保留
        assert "在长上下文扩展方面具有显著优势" in ev_text

    asyncio.run(run())


# ---------------- /eval 技能（dev_eval） ----------------
def _write_trusted_report(dir_: Path, *, recall: float, mrr: float = 0.72, n: int = 25) -> Path:
    """写一份**口径可信**的报告（real + prod + 用例集指纹一致）。

    2026-09-16 起 `dev_eval` 的取数口径变了：报告按可信度优先挑选，且**只有口径可信**
    （真实嵌入 + 线上 profile + 指纹新鲜）才会给出「达标/未达基线」结论。
    因此渲染层测试也必须造出可信报告，否则测的就不是门禁分支而是"不可信"分支。
    """
    cases_dir = dir_ / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    cases = cases_dir / "golden_real.jsonl"
    cases.write_text('{"id":"q01"}\n', encoding="utf-8")
    sha = hashlib.sha256(cases.read_bytes()).hexdigest()[:12]
    path = dir_ / "report_real.json"
    path.write_text(json.dumps({
        "embedder": "real",
        "profile": "prod",
        "cases_file": "golden_real.jsonl",
        "cases_sha256": sha,
        "num_cases": n,
        "metrics": {"recall@5": recall, "mrr@10": mrr},
        "avg_recall@5": recall,
        "avg_mrr@10": mrr,
    }, ensure_ascii=False), encoding="utf-8")
    return path


def test_dev_eval_no_report_falls_back(tmp_path):
    async def run():
        ctx = SkillContext(query="/eval", args="", doc_ids=[], llm=FakeLLM(), db=None)
        # 临时目录里没有任何报告
        with mock.patch.object(dev_eval_mod, "_EVAL_DIR", tmp_path):
            res = await dev_eval_mod.run(ctx)
        assert res.ok is True
        assert "无检索评测报告" in res.content
        assert "开发评测指标卡" in res.content

    asyncio.run(run())


def test_dev_eval_gate_with_report(tmp_path):
    _write_trusted_report(tmp_path, recall=0.91)

    async def run():
        ctx = SkillContext(query="/eval", args="", doc_ids=[], llm=FakeLLM(), db=None)
        with mock.patch.object(dev_eval_mod, "_EVAL_DIR", tmp_path):
            res = await dev_eval_mod.run(ctx)
        assert res.ok is True
        assert "达标" in res.content
        assert "avg Recall@5：0.91" in res.content
        assert "avg MRR@10：0.72" in res.content
        # 取数口径必须出现在指标卡上（"这个数字哪来的"要能回答）
        assert "report_real.json" in res.content
        assert "嵌入=real" in res.content

    asyncio.run(run())


def test_dev_eval_gate_below_threshold(tmp_path):
    _write_trusted_report(tmp_path, recall=0.55, mrr=0.3)

    async def run():
        ctx = SkillContext(query="/eval", args="", doc_ids=[], llm=FakeLLM(), db=None)
        with mock.patch.object(dev_eval_mod, "_EVAL_DIR", tmp_path):
            res = await dev_eval_mod.run(ctx)
        assert "未达基线" in res.content

    asyncio.run(run())


def test_dev_eval_gate_refuses_untrusted_report(tmp_path):
    """口径不可信（fake 嵌入 / 合成集）时**不得**渲染成"达标"。

    修复前：门禁只看 `avg_recall@5 >= 0.8`，于是合成集的 1.0 被显示为 ✅ 达标。
    """
    (tmp_path / "eval_report.json").write_text(json.dumps({
        "embedder": "fake", "num_cases": 20, "avg_recall@5": 1.0, "avg_mrr@10": 0.92,
    }, ensure_ascii=False), encoding="utf-8")

    async def run():
        ctx = SkillContext(query="/eval", args="", doc_ids=[], llm=FakeLLM(), db=None)
        with mock.patch.object(dev_eval_mod, "_EVAL_DIR", tmp_path):
            res = await dev_eval_mod.run(ctx)
        assert "口径不可信" in res.content
        assert "✅ 达标" not in res.content

    asyncio.run(run())


# ---------------- 成本熔断 + done 精确聚合 ----------------
class _CostLLM(BaseLLM):
    """返回带成本的响应；cost 可配置以触发/避开熔断。"""

    def __init__(self, cost: float, tokens: int = 100) -> None:
        self._cost = cost
        self._tokens = tokens
        self.calls = 0
        self.model_name = "stub"

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        self.calls += 1
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="最终答复"),
            prompt_tokens=self._tokens, completion_tokens=self._tokens,
            model="stub", latency_ms=42, cost=self._cost,
        )


def test_done_event_aggregates_cost_tokens():
    async def run():
        reg = ToolRegistry()
        loop = AgentLoop(_CostLLM(cost=0.5, tokens=100), ToolExecutor(reg),
                         config=LoopConfig(max_iterations=3, max_cost_per_turn=1.0))
        events = [ev async for ev in loop.run_stream("你好")]
        types = [e.type for e in events]
        assert "message" in types  # 无工具调用 → 直接回答
        done = next(e for e in events if e.type == "done")
        assert done.data["cost"] == 0.5
        assert done.data["total_tokens"] == 200  # prompt + completion
        assert done.data["latency_ms"] == 42
        assert done.data["steps"] == 1

    asyncio.run(run())


def test_cost_circuit_breaker_aborts():
    async def run():
        reg = ToolRegistry()
        # 每轮成本 0.6 > 上限 0.5 → 第 2 轮即熔断
        loop = AgentLoop(_CostLLM(cost=0.6, tokens=10), ToolExecutor(reg),
                         config=LoopConfig(max_iterations=5, max_cost_per_turn=0.5))
        events = [ev async for ev in loop.run_stream("开始")]
        err = next(e for e in events if e.type == "error")
        assert "成本" in err.data["reason"]
        assert "熔断" in err.data["reason"]
        # ★ C2（2026-09-28）契约变更：熔断轮也要发 done（**本轮**口径的真实成本）。
        # 旧契约断言"没有 done"，那会让路由层用 count_usage()（整段会话累计）展示与落库。
        done = next(e for e in events if e.type == "done")
        assert done.data.get("aborted") is True
        assert done.data["cost"] == 0.6, "必须是本轮累计成本（第 1 次调用 0.6 > 上限 0.5 熔断）"

    asyncio.run(run())


# ---------------- runtime_settings：9 键 + cost_limit ----------------
def test_runtime_settings_default_7keys_and_cost_limit():
    base = runtime_settings._default_state()
    assert base["features"] == {
        "memory": True, "summary": True, "multi_query": False,
        "code_interpreter": False, "skills": True,
        "planning": True, "dev_eval": False, "thinking": False,
        "verify_reply": False,  # P1-2.1 回复逐条核验 L2（L1 恒开；默认关）
        "reflect_on_abort": False,  # 失败轮次自评（默认关，2026-09-16 接线）
        "diagram_enrich": False,  # P1 结构图识别（默认关）
        "deep_read": True, "glossary": True,  # P0 单篇精读 / 术语表（默认开）
        "find_code": True, "research_gap": True,  # P1 找代码 / 研究缺口（默认开）
        "experiment_design": True, "review": True,  # P2 实验设计 / 审稿（默认开）
        "reproduction": True,  # S8 复现检索（默认开）
        "revision": True,  # 改稿（/改稿，对照参考文献逐节给修改建议，默认开）
        "prompt_enhance": True,  # 提示词增强（输入框改写为更明确任务描述，默认开）
        "para_transform": True,  # 段落 AI 改写（工作台扩写/缩写/润色/去重，默认开）
        "paper_search": False,  # 论文在线检索（阶段 3，默认关：主动联网）
    }
    assert base["cost_limit_per_turn"] == 1.0
    # token 上限（2026-09-17 拆成两项，口径不同）：
    #   max_tokens_budget   = 单次请求上下文上限（实测绑多篇文档时单轮 prompt 可达 24883）
    #   max_tokens_per_turn = 整轮累计上限（含所有迭代；防跑飞）
    assert base["max_tokens_budget"] == 30000
    assert base["max_tokens_per_turn"] == 150000


def test_runtime_settings_update_merges_known_keys(tmp_path):
    with mock.patch.object(runtime_settings, "_path", lambda: tmp_path / "provider_settings.json"):
        runtime_settings._state = None
        st = runtime_settings.get_state()
        # 更新 planning/dev_eval/cost_limit，其余键保持
        runtime_settings.update({"features": {"planning": False, "dev_eval": True},
                                 "cost_limit_per_turn": 2.5,
                                 "max_tokens_budget": 45000,
                                 "max_tokens_per_turn": 250000})
        assert st["features"]["planning"] is False
        assert st["features"]["dev_eval"] is True
        assert st["features"]["memory"] is True  # 未传键保持
        assert st["cost_limit_per_turn"] == 2.5
        assert st["max_tokens_budget"] == 45000
        assert st["max_tokens_per_turn"] == 250000
        # 落盘后可重读
        runtime_settings._state = None
        st2 = runtime_settings.get_state()
        assert st2["features"]["planning"] is False
        assert st2["cost_limit_per_turn"] == 2.5
        assert st2["max_tokens_budget"] == 45000
        assert st2["max_tokens_per_turn"] == 250000
        # 防脏数据：bool / <1000 一律拒绝（熔断不能被手滑关掉）
        runtime_settings.update({"max_tokens_budget": True})
        assert runtime_settings.get_state()["max_tokens_budget"] == 45000
        runtime_settings.update({"max_tokens_budget": 5})
        assert runtime_settings.get_state()["max_tokens_budget"] == 45000
        runtime_settings.update({"max_tokens_per_turn": False})
        assert runtime_settings.get_state()["max_tokens_per_turn"] == 250000
    # 还原进程内缓存，避免污染后续用例：_path 已退出 patch，下次 get_state() 重载走真实路径（默认 dev_eval=False）
    runtime_settings._state = None
