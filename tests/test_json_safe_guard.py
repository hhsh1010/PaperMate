"""JSON 解析容错**同源化**守护（O-04，2026-10-01）

### 背景

`meta_json` / `events_json` / `config_json` / `tool_calls_json` 都是**外部写入的 TEXT 列**，
损坏是迟早的事（项目已发生过 `meta_json` 损坏）。修复前：

| 位置 | 修复前 |
|---|---|
| `routes_sessions.py` | 同一函数里 `meta_json` **有**防护（注释明写"损坏不该让整页历史 500"），`tool_calls_json` **没有** |
| `routes_sessions._to_out` | `config_json` 裸 loads —— 它同时服务**会话列表** ⇒ 一条脏行让**整个列表** 500 |
| `routes_usage.py` | `events_json` / `config_json` 裸 loads —— 而 `routes_traces` 对**同一列**早有 `__safe_json` |

⇒ 收敛为唯一入口 `papermate/api/json_safe.py`。本文件守护两件事：
① 助手本身的边界行为；② **脏数据不再让接口 500**（端到端，走真实 HTTP）。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from papermate.api.json_safe import safe_json_dict, safe_json_list, safe_json_records
from papermate.models.db import ChatSession, Message, Trace


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_jsonsafe_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------- 单元层：助手的边界行为

def test_safe_json_list_edges() -> None:
    assert safe_json_list(None) == []
    assert safe_json_list("") == []
    assert safe_json_list("{not-json") == []
    assert safe_json_list('{"a": 1}') == []          # 非 list → 空（口径：保证 list）
    assert safe_json_list('[1, 2]') == [1, 2]


def test_safe_json_dict_edges() -> None:
    assert safe_json_dict(None) == {}
    assert safe_json_dict("[1,2]") == {}             # 非 dict → 空（口径：保证 dict）
    assert safe_json_dict('{"model": "x"}') == {"model": "x"}


def test_safe_json_records_filters_non_dict_elements() -> None:
    """逐条取字段的场景（`e.get(...)`）混入非 dict 元素会 AttributeError ⇒ 又是 500。"""
    out = safe_json_records('[{"type": "thinking"}, "oops", 42, {"type": "done"}]')
    assert out == [{"type": "thinking"}, {"type": "done"}]


# --------------------------------------------------------------- 端到端：脏数据 ⇒ 接口仍 200

def _corrupt(app, model, row_id: int, column: str, value: str) -> None:
    """直接把指定行的 JSON 列改成损坏内容（模拟落库时被截断/写坏）。"""
    SL = app.state.chat_svc._Session
    with SL() as db:
        row = db.get(model, row_id)
        assert row is not None
        setattr(row, column, value)
        db.commit()


def test_corrupt_session_config_json_does_not_break_list_or_detail(client) -> None:
    """★ 会话列表与详情都必须幸存（修复前列表接口会 500）。"""
    sid = client.post("/api/v1/sessions", json={"title": "脏配置"}).json()["id"]
    _corrupt(client.app, ChatSession, sid, "config_json", "{not-json")

    r_list = client.get("/api/v1/sessions")
    assert r_list.status_code == 200, r_list.text
    assert r_list.json(), "会话列表不应为空"

    r_detail = client.get(f"/api/v1/sessions/{sid}")
    assert r_detail.status_code == 200, r_detail.text
    body = r_detail.json()
    assert isinstance(body["config"], dict) and body["config"] == {}


def test_corrupt_message_tool_calls_json_does_not_break_detail(client) -> None:
    """★ 同一函数里 `meta_json` 有防护、`tool_calls_json` 没有 —— 统一后都该幸存。"""
    sid = client.post("/api/v1/sessions", json={"title": "脏 tool_calls"}).json()["id"]
    client.app.state.chat_svc.persist_turn(sid, "问题", [], [], "回答", "fake")

    SL = client.app.state.chat_svc._Session
    with SL() as db:
        msg = (
            db.query(Message)
            .filter(Message.session_id == sid, Message.role == "assistant")
            .order_by(Message.id.desc())
            .first()
        )
        assert msg is not None
        mid = msg.id
        msg.tool_calls_json = "{bad-json"
        db.commit()  # ★ 必须 commit：否则 `with` 退出即回滚，腐化从未落库 ⇒ 测试空转通过

    r = client.get(f"/api/v1/sessions/{sid}")
    assert r.status_code == 200, r.text
    target = next(m for m in r.json()["messages"] if m["id"] == mid)
    # 腐化 ⇒ 降级为空（`[]`，与"合法的空 tool_calls"同语义；前端两者都视为"无工具调用"）
    assert not target["tool_calls"], f"损坏的 tool_calls_json 应降级为空，实际 {target['tool_calls']!r}"


def test_corrupt_trace_events_json_does_not_break_usage(client) -> None:
    """★ `events_json` 在 `routes_traces` 早有 `__safe_json`，`routes_usage` 却裸 loads。"""
    sid = client.post("/api/v1/sessions", json={"title": "脏 events"}).json()["id"]
    client.app.state.chat_svc.persist_turn(sid, "问题", [], [], "回答", "fake")

    SL = client.app.state.chat_svc._Session
    with SL() as db:
        t = db.query(Trace).filter(Trace.session_id == sid).order_by(Trace.id.desc()).first()
        assert t is not None
        t.events_json = "[not-json"
        db.commit()  # ★ 同上：不 commit 会回滚 ⇒ 腐化不生效（已实测踩到）

    r = client.get(f"/api/v1/usage/sessions/{sid}")
    assert r.status_code == 200, r.text
    assert r.json()["messages"], "用量详情不应为空"
