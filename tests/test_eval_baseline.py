"""检索基线取数口径与轨迹采样口径（体检问题 4/5/6，2026-09-16）

背景（`docs/用量与评测功能体检报告.md`）：
- 问题 5：`dev_eval` **硬编码**读 `eval/eval_report.json` —— 那是「20 条合成集 + fake 哈希嵌入 +
  未开重排」的报告，门禁据此显示"✅ 达标"；它不会被任何脚本自动重跑（数字可无限期陈旧），
  真实口径的 `report_real.json` 反而**零消费者**。
- 问题 6：`traces` 里 42/179 条为空轨迹（steps=0、events=[]），混入分母系统性拉低平均步数/成本。
- 问题 4：平均成本恒 ¥0 无法区分"没采集"与"都是免费模型"。

本文件锁定的**设计取向**：
  ① 门禁**只在口径可信时给结论** —— 不可信宁可不给结论，也不给假通过；
  ② 报告**过时**（用例集指纹不符）同样不出结论；
  ③ 空轨迹**排除出分母，但不虚构数字**（排除后无样本仍返回 None）。

⚠️ 测试一律通过 monkeypatch 注入临时报告目录，**不依赖仓库里报告的当前内容** ——
   否则重跑一次评测就会改测试结果（那等于测试在锁死数据，而不是锁死行为）。
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


def _dev_eval():
    """取 `papermate.skills.dev_eval` **模块本身**（用于 monkeypatch `_EVAL_DIR`）。

    ⚠️ 必须用 `importlib.import_module`，不能用 `import papermate.skills.dev_eval as m`：
    包 `papermate/skills/__init__.py` 里 `from .dev_eval import SPEC as dev_eval` 把**包属性**
    `dev_eval` 覆盖成了 `SkillSpec` 对象；而 `import a.b.c as x` 的语义等价于
    `x = getattr(a.b, "c")`（**属性优先于 sys.modules**）→ 拿到的是 SPEC 而不是模块。
    """
    return importlib.import_module("papermate.skills.dev_eval")


#: 真·模块对象（后续代码照常写 `dev_eval.xxx`；monkeypatch 改属性同样生效）
dev_eval = _dev_eval()


@pytest.fixture
def eval_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """临时评测目录（接管 `dev_eval._EVAL_DIR`）。"""
    d = tmp_path / "eval"
    (d / "cases").mkdir(parents=True)
    monkeypatch.setattr(_dev_eval(), "_EVAL_DIR", d)
    return d


def _write_cases(d: Path, name: str = "golden_real.jsonl", text: str = '{"id":"q01"}\n') -> str:
    """写用例集并返回其指纹（sha256[:12]，与 run_eval._cases_digest 同口径）。"""
    p = d / "cases" / name
    p.write_text(text, encoding="utf-8")
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _write_report(
    d: Path,
    name: str = "report_real.json",
    *,
    embedder: str | None = "real",
    profile: str | None = "prod",
    recall: float = 0.975,
    cases_sha256: str | None = None,
    cases_file: str = "golden_real.jsonl",
    num_cases: int = 25,
    by_kind: dict | None = None,
    pipeline_digest: str | None = None,
) -> Path:
    payload = {
        "embedder": embedder,
        "profile": profile,
        "k": 5,
        "cases_file": cases_file,
        "cases_sha256": cases_sha256,
        # 缺省不写：模拟"旧报告没有该字段"的向后兼容分支（老报告不该被误判过时）
        "pipeline_digest": pipeline_digest,
        "generated_at": "2026-09-16T17:17:00",
        "num_cases": num_cases,
        "metrics": {"recall@5": recall, "mrr@10": 1.0},
        "metrics_by_kind": by_kind or {},
        "avg_recall@5": recall,
        "avg_mrr@10": 1.0,
    }
    p = d / name
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 取数优先级


def test_prefers_real_report_over_synthetic(eval_dir: Path) -> None:
    """两份报告并存时必须取真实文档集那份（此前是反的：硬编码读合成集）。"""
    _write_report(eval_dir, "eval_report.json", embedder="fake", profile=None, recall=1.0, num_cases=20)
    _write_report(eval_dir, "report_real.json", recall=0.975, num_cases=25)

    ret = dev_eval._load_retrieval_report()

    assert ret["file"] == "report_real.json"
    assert ret["num_cases"] == 25
    assert ret["avg_recall@5"] == pytest.approx(0.975)


def test_corrupt_preferred_report_falls_back(eval_dir: Path) -> None:
    """首选报告损坏时降级到下一个候选，而不是整体失去基线。"""
    (eval_dir / "report_real.json").write_text("{ not json", encoding="utf-8")
    _write_report(eval_dir, "eval_report.json", embedder="fake", profile=None, recall=1.0, num_cases=20)

    ret = dev_eval._load_retrieval_report()

    assert ret["available"] is True
    assert ret["file"] == "eval_report.json"


def test_by_kind_exposed(eval_dir: Path) -> None:
    """分组指标要透传 —— 单 gold 下 recall 饱和，分辨力全在 multi_gold / nDCG。"""
    _write_report(eval_dir, by_kind={
        "multi_gold": {"n": 2, "metrics": {"recall@5": 0.75, "ndcg@5": 0.7701}},
        "verbatim": {"n": 12, "metrics": {"recall@5": 1.0, "ndcg@5": 0.8758}},
    })

    ret = dev_eval._load_retrieval_report()

    assert ret["by_kind"]["multi_gold"] == {"n": 2, "recall@5": 0.75, "ndcg@5": 0.7701}
    assert ret["by_kind"]["verbatim"]["recall@5"] == 1.0


# ---------------------------------------------------------------- 门禁诚实性


def test_gate_refuses_fake_embedded_report(eval_dir: Path) -> None:
    """核心：口径不可信的报告**不得**判为达标 —— 宁可不给结论，也不给假通过。

    修复前：门禁只看 `avg_recall@5 >= 0.8`，于是"20 条合成集 + fake 嵌入"的
    1.0 被显示为 ✅ 达标。
    """
    _write_report(eval_dir, "eval_report.json", embedder="fake", profile=None, recall=1.0)

    ret = dev_eval._load_retrieval_report()
    gate = dev_eval._gate(ret)

    assert ret["trusted"] is False
    assert "嵌入=fake" in ret["reason"]
    assert gate["passed"] is None, "口径不可信时绝不能给 passed=True/False 的结论"
    assert "不可信" in gate["label"]


def test_gate_refuses_missing_profile(eval_dir: Path) -> None:
    """真实嵌入但未跑线上口径（profile 缺失）同样不可信。"""
    _write_report(eval_dir, "report_real.json", embedder="real", profile=None, recall=0.99)

    ret = dev_eval._load_retrieval_report()

    assert ret["trusted"] is False
    assert "检索口径" in ret["reason"]


def test_gate_refuses_stale_report(eval_dir: Path) -> None:
    """用例集改了但没重跑评测 → 报告过时，门禁必须能发现。"""
    real_digest = _write_cases(eval_dir)  # 当前用例集指纹
    _write_report(eval_dir, cases_sha256="000000000000")  # 报告记录的是旧的

    ret = dev_eval._load_retrieval_report()
    gate = dev_eval._gate(ret)

    assert real_digest != "000000000000"
    assert ret["trusted"] is False
    assert "过时" in ret["reason"]
    assert gate["passed"] is None


def test_gate_outcome_when_trusted(eval_dir: Path) -> None:
    """口径可信时才给出真结论：≥ 阈值达标、< 阈值未达。"""
    sha = _write_cases(eval_dir)

    _write_report(eval_dir, cases_sha256=sha, recall=0.975)
    assert dev_eval._gate(dev_eval._load_retrieval_report())["passed"] is True

    _write_report(eval_dir, cases_sha256=sha, recall=0.62)
    gate = dev_eval._gate(dev_eval._load_retrieval_report())
    assert gate["passed"] is False
    assert "未达基线" in gate["label"]


def test_no_report_at_all(eval_dir: Path) -> None:
    """没有任何报告 → 不编造数字，明确提示怎么生成。"""
    ret = dev_eval._load_retrieval_report()
    gate = dev_eval._gate(ret)

    assert ret["available"] is False
    # 键恒存在且为 None（不是缺键）—— 前端取 undefined 与"无数据"是两种语义，不应混同
    assert ret["avg_recall@5"] is None
    assert ret["num_cases"] is None
    assert gate["passed"] is None
    assert "run_eval.py" in gate["label"]


def test_repo_report_is_honest_or_absent() -> None:
    """**纳入一键门禁**：仓库自带报告若存在，必须口径可信且指纹新鲜。

    这条用例让 `pytest` 本身成为门禁 —— 报告被删/口径退回 fake/用例集改了没重跑，
    都会在这里变红（而不是等到有人打开看板才发觉数字不可信）。
    报告不存在时跳过：无 GPU/未下权重的环境不应被这条阻塞。
    """
    real = dev_eval._EVAL_DIR / "report_real.json"
    if not real.exists():
        pytest.skip("未找到 eval/report_real.json（先跑 python eval/run_eval.py --embedder real "
                    "--profile prod --cases cases/golden_real.jsonl --out eval/report_real.json）")

    ret = dev_eval._load_retrieval_report()

    assert ret["file"] == "report_real.json", "发现更优先的报告，测试需同步更新"
    assert ret["trusted"] is True, f"仓库自带报告口径/新鲜度不达标：{ret['reason']}"


# ---------------------------------------------------------------- 轨迹采样口径


@pytest.fixture
def client(tmp_path, monkeypatch):
    """独立 TestClient（临时 data_dir），用于轨迹采样口径验证。"""
    import os
    import tempfile

    from fastapi.testclient import TestClient

    import papermate.config as cfg_mod
    from papermate.api.app import create_app

    tmp = tempfile.mkdtemp(prefix="papermate_eval_baseline_")
    monkeypatch.setenv("PAPERMATE_DATA_DIR", tmp)
    cfg_mod._settings = None
    app = create_app()
    with TestClient(app) as c:
        yield c
    os.environ.pop("PAPERMATE_DATA_DIR", None)
    cfg_mod._settings = None


def _add_trace(client, session_id: int, trace_id: str, *, events: list | None, steps: int,
               cost: float, latency: int) -> None:
    from papermate.models.db import Trace

    SL = client.app.state.session_svc._Session
    with SL() as db:
        db.add(Trace(
            session_id=session_id,
            trace_id=trace_id,
            events_json=json.dumps(events or [], ensure_ascii=False),
            total_steps=steps,
            total_cost=cost,
            total_latency_ms=latency,
        ))
        db.commit()


def test_empty_traces_excluded_from_denominator(client, eval_dir: Path) -> None:
    """空轨迹（steps=0 / 无事件）不得计入分母 —— 它们不是"跑得差"，是"没跑"。

    真实库实测 179 条里 42 条为空（23%），且集中在最近记录 → 必然污染采样窗口。
    """
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    evs = [{"idx": 0, "type": "tool_call", "data": {"name": "retrieve"}},
           {"idx": 1, "type": "tool_result", "data": {"ok": True}}]
    # 两条空轨迹（模拟被中断 / 技能直答）+ 一条正常
    _add_trace(client, s["id"], "empty-1", events=[], steps=0, cost=0.0, latency=0)
    _add_trace(client, s["id"], "empty-2", events=[], steps=0, cost=0.0, latency=0)
    _add_trace(client, s["id"], "real-1", events=evs, steps=6, cost=0.03, latency=1200)

    _write_report(eval_dir, cases_sha256=_write_cases(eval_dir))
    d = client.get("/api/v1/eval/summary").json()
    agent = d["agent"]

    assert agent["count"] == 1, "分母只能是 1 条非空轨迹"
    assert agent["avg_steps"] == 6.0, "步数只按非空轨迹平均（修复前会被空轨迹拉低）"
    assert agent["avg_cost"] == pytest.approx(0.03)
    assert agent["avg_latency_ms"] == 1200, "延迟指标从无到有"
    assert agent["costed_count"] == 1
    assert agent["window"]["excluded_empty"] == 2
    assert agent["window"]["table_total"] == 3


def test_all_empty_traces_yield_no_fabricated_metrics(client, eval_dir: Path) -> None:
    """全是空轨迹时：count=0 且指标为 None —— 排除分母不等于编造一个"0 步 0 成本"。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    _add_trace(client, s["id"], "empty-1", events=[], steps=0, cost=0.0, latency=0)

    _write_report(eval_dir, cases_sha256=_write_cases(eval_dir))
    agent = client.get("/api/v1/eval/summary").json()["agent"]

    assert agent["count"] == 0
    assert agent["tool_success_rate"] is None
    assert agent["citation_rate"] is None
    assert agent["window"]["excluded_empty"] == 1


def test_cost_samples_counted_for_zero_cost(client, eval_dir: Path) -> None:
    """成本 0 可能是真的（未定价 / fake 模型）→ 不剔除，但必须能看出样本里有几条计过价。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    evs = [{"idx": 0, "type": "tool_call", "data": {"name": "retrieve"}},
           {"idx": 1, "type": "tool_result", "data": {"ok": True}}]
    _add_trace(client, s["id"], "free-1", events=evs, steps=2, cost=0.0, latency=300)
    _add_trace(client, s["id"], "paid-1", events=evs, steps=4, cost=0.02, latency=700)

    _write_report(eval_dir, cases_sha256=_write_cases(eval_dir))
    agent = client.get("/api/v1/eval/summary").json()["agent"]

    assert agent["count"] == 2
    assert agent["costed_count"] == 1, "只有 1 条真的计过价 → 看板应能解释「平均成本 ¥0.01」的构成"
    assert agent["avg_cost"] == pytest.approx(0.01)
    assert agent["avg_latency_ms"] == 500


# ------------------------------------------------- 检索实现指纹（#2：报告绑源码）


def test_stale_reason_detects_pipeline_change(eval_dir: Path) -> None:
    """用例集没变但**检索实现变了** → 报告同样过时。

    这是 A5"优化无回归守护"的后半段：原报告只绑用例集指纹，
    "改了分块/重排参数但没重跑评测"发现不了 —— 门禁守的是过去。
    """
    from lightcore.rag import provenance

    sha = _write_cases(eval_dir)
    _write_report(eval_dir, cases_sha256=sha, pipeline_digest="000000000000")

    ret = dev_eval._load_retrieval_report()

    assert ret["trusted"] is False
    assert "检索实现已变" in ret["reason"]
    assert provenance.digest() != "000000000000"


def test_pipeline_digest_matched_report_is_trusted(eval_dir: Path) -> None:
    """指纹一致时不得误报过时（避免"永远被判过时"→ 检查被忽略）。"""
    from lightcore.rag import provenance

    sha = _write_cases(eval_dir)
    _write_report(eval_dir, cases_sha256=sha, pipeline_digest=provenance.digest())

    assert dev_eval._load_retrieval_report()["trusted"] is True


def test_provenance_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    """指纹：同内容稳定；内容变则变；缺文件也能给出可比值（不抛）。"""
    from lightcore.rag import provenance

    for rel in provenance.SOURCES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n", encoding="utf-8")

    first = provenance.digest(tmp_path)
    assert first == provenance.digest(tmp_path), "同内容必须稳定"

    (tmp_path / provenance.SOURCES[0]).write_text("x = 2\n", encoding="utf-8")
    assert provenance.digest(tmp_path) != first, "内容变了指纹必须变"

    (tmp_path / provenance.SOURCES[0]).unlink()
    assert isinstance(provenance.digest(tmp_path), str), "缺文件时给可比值而非抛异常"
