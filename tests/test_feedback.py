"""E1 / E2（2026-09-21）：反馈入口 + 轨迹导出为评测样本。

守护的重点不只是"能用"，而是**语义正确**：
- 反馈可改分、取最后一次为准（否则按钮点了没反应 / 汇总数不对）
- 导出的 case **`gold` 必须是空**（把检索命中当标准答案 = 自证循环）
- `/traces/export` 不能被 `/traces/{trace_id}` 抢先匹配成 trace_id
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from papermate.api.routes_feedback import router as feedback_router
from papermate.api.routes_traces import router as traces_router
from papermate.models.db import Message, Trace, init_db, make_engine


@pytest.fixture()
def client(tmp_path):
    """独立临时库 + 一个最小 app（只挂被测试的路由）。"""
    engine = make_engine(f"sqlite:///{tmp_path/'t.db'}")
    init_db(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as s:
        from papermate.models.db import ChatSession

        sess_row = ChatSession(id=1, title="t")
        s.add(sess_row)
        s.add(Message(id=11, session_id=1, role="user", content="这篇论文用了什么数据集？"))
        s.add(Trace(
            id=1,
            trace_id="tr-1",
            session_id=1,
            events_json=json.dumps([
                {"idx": 0, "type": "tool_call", "data": {"name": "retrieve", "arguments": {"query": "数据集"}}},
                {"idx": 1, "type": "tool_result", "data": {"call_id": "c1", "ok": True, "output": {
                    "results": [{"id": "doc-19-c2"}, {"id": "doc-19-c5"}, {"id": "doc-19-c2"}]}}},
            ], ensure_ascii=False),
            total_steps=2,
        ))
        s.commit()

    class _Svc:  # 路由从 app.state.chat_svc._Session 取会话工厂
        _Session = Session

    app = FastAPI()
    app.state.chat_svc = _Svc()  # type: ignore[attr-defined]
    app.include_router(traces_router)
    app.include_router(feedback_router)
    return TestClient(app), Session


def test_submit_feedback_ok(client):
    c, _ = client
    r = c.post("/feedback", json={"session_id": 1, "message_id": 11, "trace_id": "tr-1", "rating": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rating"] == 1
    assert body["session_id"] == 1
    assert body["message_id"] == 11
    assert body["created_at"]


def test_rating_must_be_plus_or_minus_one(client):
    """rating 只接受 ±1：放宽会让好评率/均值的口径失真，且脏数据事后无法分辨。"""
    c, _ = client
    for bad in (0, 2, -2, 99):
        r = c.post("/feedback", json={"session_id": 1, "rating": bad})
        assert r.status_code == 422, (bad, r.status_code, r.text)


def test_unknown_session_returns_404(client):
    c, _ = client
    r = c.post("/feedback", json={"session_id": 9999, "rating": 1})
    assert r.status_code == 404


def test_change_rating_keeps_last(client):
    """同一消息可改分；汇总必须取**最后一次**（用户改主意是高频行为）。"""
    c, Session = client
    c.post("/feedback", json={"session_id": 1, "message_id": 11, "rating": 1})
    c.post("/feedback", json={"session_id": 1, "message_id": 11, "rating": -1})
    s = c.get("/feedback/summary").json()
    assert s["total"] == 2          # 两条记录都留（审计）
    assert s["positive"] == 1
    assert s["negative"] == 1
    assert s["rated_message_ids"]["11"] == -1   # 当前评分 = 最后一次


def test_summary_scoped_by_session(client):
    c, _ = client
    c.post("/feedback", json={"session_id": 1, "rating": 1})
    assert c.get("/feedback/summary", params={"session_id": 1}).json()["total"] == 1
    assert c.get("/feedback/summary", params={"session_id": 2}).json()["total"] == 0


def test_export_single_case_has_empty_gold(client):
    """★ 关键语义：gold 必须为空 —— 检索命中不是标准答案（自证循环）。"""
    c, _ = client
    r = c.get("/traces/tr-1/export")
    assert r.status_code == 200, r.text
    case = r.json()
    assert case["id"] == "tr-1"
    assert case["query"] == "这篇论文用了什么数据集？"   # 用户原话，不是检索改写后的 query
    assert case["kind"] == "candidate"
    assert case["gold"] == []                          # ← 不得预填
    assert set(case["candidates"]) == {"doc-19-c2", "doc-19-c5"}   # 去重
    assert case["docs"]["doc-19"] and len(case["docs"]["doc-19"]) == 2


def test_export_unknown_trace_404(client):
    c, _ = client
    assert c.get("/traces/nope/export").status_code == 404


def test_export_batch_jsonl_not_eaten_by_trace_id_route(client):
    """/traces/export 必须排在 /traces/{trace_id} 之前，否则被当成 trace_id 返回 404。"""
    c, _ = client
    r = c.get("/traces/export", params={"fmt": "jsonl"})
    assert r.status_code == 200, r.text
    lines = [ln for ln in r.text.strip().split("\n") if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "tr-1"

    rj = c.get("/traces/export", params={"fmt": "json"})
    assert isinstance(rj.json(), list)
