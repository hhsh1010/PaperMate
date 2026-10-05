"""F11 · 端到端忠实度看板接口测试（P2 体验优化）

覆盖：
- GET /api/v1/eval/summary 结构与指标计算（复用 dev_eval.collect_metrics，单一口径）
- 检索基线读取 + 门禁判定
- 运行期 Agent 指标：无 Trace / 有 Trace（工具成功率、溯源率从事件流实算）
- 看板不受 features.dev_eval 门控（只读聚合、无 LLM 调用，与 /eval 技能语义不同）

⚠️ 本文件的检索基线测试**注入临时报告**（不读仓库里的 eval/*.json）：
   报告内容会随评测重跑而变，写进断言等于"锁死数据"而不是"锁死行为"。
   仓库自带报告的诚实性（口径可信 + 指纹新鲜）由
   `tests/test_eval_baseline.py::test_repo_report_is_honest_or_absent` 单独守护。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_eval_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def trusted_report(tmp_path, monkeypatch):
    """注入一份口径可信的报告（real + prod），供门禁断言使用。"""
    import importlib

    # ⚠️ 必须用 importlib 取模块本身：包 __init__ 里 `from .dev_eval import SPEC as dev_eval`
    # 把包属性 `dev_eval` 覆盖成了 SkillSpec，`import papermate.skills.dev_eval as m` 会拿到 SPEC。
    dev_eval = importlib.import_module("papermate.skills.dev_eval")

    d = tmp_path / "eval"
    (d / "cases").mkdir(parents=True)
    monkeypatch.setattr(dev_eval, "_EVAL_DIR", d)
    (d / "report_real.json").write_text(json.dumps({
        "embedder": "real",
        "profile": "prod",
        "cases_file": "golden_real.jsonl",
        "num_cases": 25,
        "metrics": {"recall@5": 0.975, "mrr@10": 1.0},
        "avg_recall@5": 0.975,
        "avg_mrr@10": 1.0,
        "metrics_by_kind": {},
    }), encoding="utf-8")
    return d


def test_eval_summary_structure(client, trusted_report):
    r = client.get("/api/v1/eval/summary")
    assert r.status_code == 200
    d = r.json()
    # 检索基线：取数口径与来源必须对调用方可见（否则"这个 0.975 是哪来的"无从回答）
    assert d["retrieval"]["available"] is True
    assert d["retrieval"]["file"] == "report_real.json"
    assert d["retrieval"]["trusted"] is True
    assert d["retrieval"]["embedder"] == "real"
    assert d["retrieval"]["profile"] == "prod"
    assert d["retrieval"]["num_cases"] == 25
    assert d["retrieval"]["avg_recall@5"] == 0.975
    # 门禁：0.975 ≥ 0.8 → 达标（且仅因口径可信才给结论）
    assert d["gate"]["threshold"] == 0.8
    assert d["gate"]["passed"] is True
    # 运行期指标（空库：可读但无样本，指标为 None 不编造数字）
    assert d["agent"]["available"] is True
    assert d["agent"]["count"] == 0
    assert d["agent"]["tool_success_rate"] is None
    assert d["agent"]["citation_rate"] is None
    assert d["recent_traces"] == []


def test_eval_summary_not_gated_by_dev_eval(client):
    """看板是只读聚合（无 LLM 调用），不受 /eval 技能的 dev_eval 开关（默认 False）影响。"""
    from papermate import runtime_settings

    runtime_settings.update({"features": {**runtime_settings.DEFAULT_FEATURES, "dev_eval": False}})
    try:
        assert client.get("/api/v1/eval/summary").status_code == 200
    finally:
        # runtime_settings 为进程级单例，必须在 finally 中重置，避免污染其他测试
        runtime_settings._state = None


def test_eval_summary_with_traces(client):
    """有 Trace 后：工具成功率 / 溯源率从事件流实算（与 /eval 技能同口径）。"""
    from papermate.models.db import Trace

    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    SL = client.app.state.session_svc._Session
    events = [
        {"idx": 0, "type": "tool_call", "data": {"name": "retrieve"}},
        {"idx": 1, "type": "tool_result", "data": {"ok": True}},
        # 2026-09-16 修正：字段名必须是 `content`（真实生产者写入的键）。
        # 此处原为 `"text"` —— 现实中不存在该键，纯属夹具失真：
        # 旧消费者"任何 message 事件都当回答样本"，所以夹具写错也一直绿；
        # 修正后的消费者只认"带 content 的完整回答"，夹具失真会立刻暴露（这正是本次回归）。
        {"idx": 2, "type": "message", "data": {"content": "答案", "citations": [{"doc_id": 1, "page": 1}]}},
    ]
    with SL() as db:
        db.add(
            Trace(
                session_id=s["id"],
                trace_id="tr-1",
                events_json=json.dumps(events, ensure_ascii=False),
                total_steps=3,
                total_cost=0.01,
                total_latency_ms=100,
            )
        )
        db.commit()

    d = client.get("/api/v1/eval/summary").json()
    assert d["agent"]["count"] == 1
    assert d["agent"]["tool_success_rate"] == 1.0
    assert d["agent"]["citation_rate"] == 1.0
    assert d["agent"]["avg_cost"] == pytest.approx(0.01)
    assert d["agent"]["avg_steps"] == 3.0
    assert d["recent_traces"][0]["trace_id"] == "tr-1"


def test_eval_metrics_single_source():
    """溯源率 / 工具成功率的实现必须唯一（lightcore.rag.metrics），eval/metrics.py 仅 re-export。"""
    from lightcore.rag.metrics import citation_rate, tool_success_rate

    assert citation_rate([{"citations": [1]}, {}]) == 0.5
    assert citation_rate([]) == 0.0
    assert tool_success_rate([]) == 1.0
    evs = [
        {"type": "tool_call", "data": {}},
        {"type": "tool_result", "data": {"ok": False}},
        {"type": "tool_call", "data": {}},
        {"type": "tool_result", "data": {"ok": True}},
    ]
    assert tool_success_rate(evs) == 0.5
