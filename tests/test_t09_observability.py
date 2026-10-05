"""T-09 可观测性闭环（2026-09-30）：未收尾**四分类** + 延迟分位（含缺口）。

覆盖三层，重点在**口径**而不是"接口能返回什么"：

| 层 | 守什么 |
|---|---|
| ① 原语单测 | 四分类（done/interrupted/unknown/failed）与分位的边界 |
| ② ★ 与 numpy 对齐 | 分位插值口径**不自证** —— 与 `numpy.percentile` 比 |
| ③ ★★ 同源断言 | `/metrics`（全量）与 `dev_eval`（最近 N 条）对**同一批 events** 必须给出**同一个计数** |

③ 是重点：本项目「评测同源化」已踩**四次**（sparse / rerank 档位 / 合并层 / `rewrite_query`），
而 T-09 开工时 `/metrics` 与 `dev_eval._aggregate_agent` **各写了一份聚合**（同名字段两个口径）
—— 本用例把"它们必须一致"钉死。

★ 两个**实测踩出来的**口径，各有专项用例：
- `unknown`（`failure_kind` 缺失）**既不能算成功也不能算失败**；
- `total_latency_ms == 0` 是**未采集**，不是"很快"（实测占 80.1%，不剔除会把 p50 压成 0）。
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from papermate.api.routes_metrics import router
from papermate.models.db import ChatSession, Message, Trace, init_db, make_engine


# ---- 事件构造器（口径与生产方一致：done / error / 其它）---------------------------
def _ev_done() -> dict:
    return {"idx": 0, "type": "done", "data": {}}


def _ev_err(kind: str | None = "llm_error") -> dict:
    data = {} if kind is None else {"failure_kind": kind}
    return {"idx": 0, "type": "error", "data": data}


def _ev_think() -> dict:
    return {"idx": 0, "type": "thinking", "data": {}}


# ---- ① 原语 -------------------------------------------------------------------
def test_classify_events_four_way() -> None:
    from lightcore.core.trace_metrics import classify_events

    assert classify_events([_ev_think(), _ev_done()]) == "done"
    assert classify_events([_ev_think(), _ev_err()]) == "failed"
    assert classify_events([_ev_think()]) == "interrupted"
    # `done` 优先：出过错但最终收尾 ⇒ 算收尾成功（否则会高估失败）
    assert classify_events([_ev_err(), _ev_done()]) == "done"


def test_missing_kind_is_unknown_not_failed() -> None:
    """★ `failure_kind` 缺失 ⇒ `unknown`（打点缺口）。

    既**不能**当 failed（那是栽赃：不知道发生了什么就说它失败了），
    也**不能**当 done（那是假通过）。
    """
    from lightcore.core.trace_metrics import classify_events

    assert classify_events([_ev_err(None)]) == "unknown"


def test_cancelled_kind_maps_to_interrupted() -> None:
    """★ `failure_kind == cancelled` ⇒ `interrupted`（不是 failed）。

    这是**为将来补打点预留的通路**：`lightcore/core/failure.py` 早就定义了 `CANCELLED`
    并刻意不放进 `REAL_FAILURES`；等取消路径真的打上该标记时，这里**无需改动**。
    """
    from lightcore.core import failure as mf
    from lightcore.core.trace_metrics import classify_events

    assert classify_events([_ev_err(mf.CANCELLED)]) == "interrupted"


def test_classify_events_tolerates_broken_payload() -> None:
    """★ 坏数据归 `interrupted` 而**不是** `failed` —— 数据损坏 ≠ 系统失败。"""
    from lightcore.core.trace_metrics import classify_events

    assert classify_events(None) == "interrupted"
    assert classify_events("not-a-list") == "interrupted"
    assert classify_events([None, 42, "x"]) == "interrupted"


def test_summarize_traces_counts() -> None:
    from lightcore.core.trace_metrics import summarize_traces

    out = summarize_traces([
        [_ev_done()], [_ev_think()], [_ev_err()], [_ev_err(None)], [_ev_err("tool_error")],
    ])
    # `stopped` = 有**明确**用户停止标记的条数（这批里没有 cancelled 事件 ⇒ 0）
    assert out == {
        "done": 1, "interrupted": 1, "unknown": 1, "failed": 2, "stopped": 0, "total": 5,
    }


def test_percentile_boundaries() -> None:
    from lightcore.core.trace_metrics import percentile

    assert percentile([], 0.5) == 0.0
    assert percentile([7], 0.99) == 7.0
    v = [1, 2, 3, 4]
    assert percentile(v, 0.0) == 1.0
    assert percentile(v, 0.5) == 2.5
    assert percentile(v, 1.0) == 4.0


def test_percentile_matches_numpy() -> None:
    """★ 分位口径**不自证**：与 numpy 的默认 linear 插值逐位对齐。"""
    np = pytest.importorskip("numpy")
    from lightcore.core.trace_metrics import percentile

    vals = [12, 5, 88, 3, 47, 91, 20, 66, 30, 55]
    for q in (0.5, 0.9, 0.95, 0.99):
        assert percentile(vals, q) == pytest.approx(float(np.percentile(vals, q * 100)))


def test_latency_percentiles_excludes_missing_zeros() -> None:
    """★★ `total_latency_ms == 0` 是**未采集**，不是"很快"（实测占 80.1%）。

    不剔除 0 ⇒ `p50` 会变成 `0.0ms`（假象）—— 这是 T-09 在真实库上**实际踩到**的。
    """
    from lightcore.core.trace_metrics import latency_percentiles

    out = latency_percentiles([0, 0, 0, 100, 200, 300, 400])
    assert out["samples"] == 4, "只应统计非零样本"
    assert out["missing"] == 3, "缺口必须显式可见"
    assert out["p50"] == 250.0, "若把 0 计入，这里会是 0.0（假的'极快'）"
    assert out["max"] == 400
    assert latency_percentiles([])["p95"] == 0.0
    assert latency_percentiles([0, 0])["samples"] == 0


# ---- ③ ★★ 同源断言：两个端点必须用同一把尺子 --------------------------------------
def test_metrics_and_dev_eval_agree_on_classification() -> None:
    """★★ 同一批 events，`/metrics` 与 `dev_eval` 的分类计数**必须相同**。

    这条用例存在的唯一理由：T-09 开工前它们是**两份独立实现**（`routes_metrics` 自己数 done、
    `dev_eval` 只看工具成功率）。若日后有人再各写一份，本用例会红。
    """
    from lightcore.core.trace_metrics import summarize_traces
    from papermate.skills.dev_eval import _aggregate_agent

    batches = [
        [_ev_think(), _ev_done()],
        [_ev_think()],
        [_ev_err("llm_error")],
        [_ev_err(None)],
        [_ev_err("tool_error"), _ev_think()],
    ]
    direct = summarize_traces(batches)
    traces = [
        {"trace_id": f"t{i}", "events": b, "cost": 0.1, "steps": 2, "latency_ms": 100 * (i + 1)}
        for i, b in enumerate(batches)
    ]
    agg = _aggregate_agent(traces)
    assert agg["unfinished_breakdown"] == {
        "done": direct["done"],
        "interrupted": direct["interrupted"],
        "unknown": direct["unknown"],
        "failed": direct["failed"],
    }
    n = len(batches)
    assert agg["unfinished_rate"] == round((n - direct["done"]) / n, 4)
    # ★ 排除中断时**含 unknown**（缺口不能当成功）
    assert agg["unfinished_rate_excluding_interrupt"] == round(
        (direct["failed"] + direct["unknown"]) / n, 4
    )
    assert agg["failed_confirmed_rate"] == round(direct["failed"] / n, 4)


# ---- ② `/metrics` 端到端 --------------------------------------------------------
def _mk_trace(tid: str, events: list[dict], *, latency: int = 1000, cost: float = 0.01) -> Trace:
    return Trace(
        trace_id=tid, session_id=1,
        events_json=json.dumps(events, ensure_ascii=False),
        total_steps=2, total_cost=cost, total_latency_ms=latency,
    )


@pytest.fixture()
def client(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'm.db'}")
    init_db(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        s.add(ChatSession(id=1, title="t"))
        s.add(Message(id=1, session_id=1, role="assistant", content="a", cost=0.02))
        s.commit()

    class _S:
        _Session = Session

    app = FastAPI()
    app.state.chat_svc = _S()  # type: ignore[attr-defined]
    app.include_router(router)
    return TestClient(app), Session


def test_metrics_exposes_breakdown_and_percentiles(client) -> None:
    c, Session = client
    with Session() as s:
        s.add(_mk_trace("t1", [_ev_think(), _ev_done()], latency=100))
        s.add(_mk_trace("t2", [_ev_think()], latency=200))            # interrupted
        s.add(_mk_trace("t3", [_ev_err("llm_error")], latency=300))   # failed
        s.add(_mk_trace("t4", [_ev_think()], latency=400))            # interrupted
        s.add(_mk_trace("t5", [_ev_err(None)], latency=0))            # unknown + 未采集延迟
        s.commit()

    m = c.get("/metrics").json()
    assert m["num_traces"] == 5
    assert m["unfinished_breakdown"] == {
        "done": 1, "interrupted": 2, "unknown": 1, "failed": 1,
    }
    # 未收尾 4/5 = 0.8；★ 排除中断后 = (failed 1 + unknown 1)/5 = 0.4
    assert m["unfinished_rate"] == 0.8
    assert m["unfinished_rate_excluding_interrupt"] == 0.4
    assert m["failed_confirmed_rate"] == 0.2
    # 分位只在 [100,200,300,400] 上算；t5 的 0 记为缺口
    assert m["latency_samples"] == 4
    assert m["latency_missing"] == 1
    assert m["latency_p50_ms"] == 250.0
    assert m["latency_p99_ms"] > m["latency_p95_ms"] > m["latency_p50_ms"]


def test_metrics_empty_window_percentiles_are_zero(client) -> None:
    """空窗不能除零 —— 分位字段必须存在且为 0。"""
    c, _ = client
    m = c.get("/metrics").json()
    assert m["latency_p50_ms"] == 0.0
    assert m["latency_p95_ms"] == 0.0
    assert m["unfinished_breakdown"] == {
        "done": 0, "interrupted": 0, "unknown": 0, "failed": 0,
    }


def test_interrupted_is_not_counted_as_failure(client) -> None:
    """★ 口径核心：**用户中断不是失败** —— 不得进入"排除中断后的未收尾率"。"""
    c, Session = client
    with Session() as s:
        s.add(_mk_trace("t1", [_ev_think()]))   # 只有中断
        s.commit()
    m = c.get("/metrics").json()
    assert m["unfinished_rate"] == 1.0
    assert m["unfinished_rate_excluding_interrupt"] == 0.0
    assert m["failed_confirmed_rate"] == 0.0


def test_avg_latency_is_same_source_as_percentiles(client) -> None:
    """★ `avg_latency_ms` 必须与分位**同源**（都剔除未采集的 0）。

    实测差 5 倍（含 0 时 2967ms / 剔除后 14897ms）—— 若不同源，就会同时出现
    "avg 说 3 秒、p50 说 11 秒"的荒谬组合（本项目"两把尺子"的又一次现形）。
    """
    c, Session = client
    with Session() as s:
        s.add(_mk_trace("t1", [_ev_done()], latency=0))       # 未采集
        s.add(_mk_trace("t2", [_ev_done()], latency=1000))
        s.add(_mk_trace("t3", [_ev_done()], latency=3000))
        s.commit()
    m = c.get("/metrics").json()
    assert m["latency_samples"] == 2
    assert m["latency_missing"] == 1
    assert m["avg_latency_ms"] == 2000.0, "含 0 时会变成 1333.3（被未采集稀释）"
    assert m["latency_p50_ms"] == 2000.0
