"""O-09（2026-10-01）：N+1 批量查询消除的对拍测试。

三处改造，验收判据均为「改写前后同一输入结果完全一致」：
1. `ChatService.count_usage_all()`（GROUP BY 批量）↔ 逐会话 `count_usage()` 逐字段对拍；
2. `/traces/export` 批量路径（预取 user 消息 + 内存挑选）↔ `/traces/{id}/export`
   SQL 路径的 `query` 字段对拍（同一轨迹两种实现必须同值）；
3. `LongTermMemory` 的 `_norm` 缓存：文本变更（update / 合并刷新）后缓存必须同步，
   否则精确去重口径与文本脱节（这正是缓存类改动最危险的静默错误）。

数据层纯本地（SQLite / hash embedder），不触发大模型 / 网络。
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_o09_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------- 1) count_usage_all 对拍
def test_count_usage_all_matches_per_session(client):
    chat_svc = client.app.state.chat_svc
    SL = client.app.state.session_svc._Session
    from papermate.models.db import Message, ToolRun

    ids = []
    for i, title in enumerate(("有多条消息", "空会话", "只有工具无消息")):  # noqa: B007
        ids.append(client.post("/api/v1/sessions", json={"title": title}).json()["id"])

    now = datetime.now(UTC).replace(tzinfo=None)
    with SL() as db:
        for i in range(3):
            db.add(Message(session_id=ids[0], role="user" if i % 2 == 0 else "assistant",
                           content=f"m{i}", prompt_tokens=100 + i, completion_tokens=10 + i,
                           cost=0.001 * (i + 1), created_at=now - timedelta(minutes=i)))
        db.add(Message(session_id=ids[1], role="user", content="x"))  # 无 token 字段（None → 0）
        db.add(ToolRun(session_id=ids[0], tool_name="retrieve", args_json="{}", ok=True))
        db.add(ToolRun(session_id=ids[2], tool_name="retrieve", args_json="{}", ok=False))
        db.commit()

    batch = chat_svc.count_usage_all()
    for sid in ids:
        per = chat_svc.count_usage(sid)
        got = batch.get(sid)
        assert got is not None, f"会话 {sid} 在批量结果中缺失"
        for k in ("num_messages", "num_tool_runs", "prompt_tokens", "completion_tokens", "cost"):
            assert got[k] == per[k], f"session={sid} field={k}: batch={got[k]} vs per={per[k]}"


def test_usage_route_empty_session_zero_filled(client):
    """空会话（无任何消息/工具）必须回落零值兜底，不得 KeyError。"""
    s = client.post("/api/v1/sessions", json={"title": "空会话"}).json()
    rows = {r["session_id"]: r for r in client.get("/api/v1/usage").json()}
    row = rows[s["id"]]
    assert row["num_messages"] == 0 and row["num_tool_runs"] == 0
    assert row["prompt_tokens"] == 0 and row["completion_tokens"] == 0
    assert row["cost"] == 0.0 and row["last_at"] is None


# ---------------------------------------------------------- 2) traces/export 两路径对拍
def test_export_traces_batch_matches_single(client):
    from papermate.models.db import Message, Trace

    SL = client.app.state.session_svc._Session
    sid = client.post("/api/v1/sessions", json={"title": "导出对拍"}).json()["id"]
    now = datetime.now(UTC).replace(tzinfo=None)
    with SL() as db:
        db.add(Message(session_id=sid, role="user", content="第一条提问", created_at=now - timedelta(minutes=5)))
        db.add(Message(session_id=sid, role="user", content="第二条提问", created_at=now - timedelta(minutes=1)))
        db.add(Trace(trace_id="tr-a", session_id=sid, events_json="[]",
                     created_at=now - timedelta(minutes=4)))  # 之前只有第一条（-5min）
        db.add(Trace(trace_id="tr-b", session_id=sid, events_json="[]",
                     created_at=now - timedelta(minutes=30)))  # 早于全部 user 消息
        # 边界：trace.created_at 与某条 user 消息**恰好相等** ⇒ SQL `<=` 语义必须包含该消息
        # （反向自检踩过的坑：若测试数据全是严格小于，把 `<=` 改 `<` 也不会红）
        edge_t = now - timedelta(minutes=2)
        db.add(Message(session_id=sid, role="user", content="边界提问", created_at=edge_t))
        db.add(Trace(trace_id="tr-c", session_id=sid, events_json="[]", created_at=edge_t))
        db.commit()

    single_a = client.get("/api/v1/traces/tr-a/export").json()
    single_b = client.get("/api/v1/traces/tr-b/export").json()
    batch = {c["id"]: c for c in client.get("/api/v1/traces/export", params={"session_id": sid, "fmt": "json"}).json()}
    assert batch["tr-a"]["query"] == single_a["query"] == "第一条提问"
    assert batch["tr-b"]["query"] == single_b["query"] == ""  # 早于全部消息 → 空（SQL NULL 比较语义）
    single_c = client.get("/api/v1/traces/tr-c/export").json()
    assert batch["tr-c"]["query"] == single_c["query"] == "边界提问"  # 相等时刻被 `<=` 包含


# ---------------------------------------------------------- 3) _norm 缓存一致性
def test_norm_cache_follows_text_changes():
    from lightcore.memory.long_term import LongTermMemory
    from lightcore.rag.embedder import FakeHashEmbedder

    mem = LongTermMemory(embedder=FakeHashEmbedder(dim=32))
    it = mem.add("用户偏好用中文回答", importance=0.9)
    assert mem._norm_cached(it) == mem._norm("用户偏好用中文回答")

    # update 变更文本后：缓存必须同步，否则同新文本的精确去重会漏判
    mem.update(it.id, "用户偏好用英文回答", importance=0.9)
    assert mem._norm_cached(it) == mem._norm("用户偏好用英文回答")
    dup = mem._find_duplicate("用户偏好用英文回答", mem._embedder.encode(["用户偏好用英文回答"])[0])
    assert dup is not None and dup.id == it.id, "文本变更后缓存未刷新会导致精确匹配漏判"

    # 合并刷新路径（add 命中同义条目且新文本更长 → 文本被替换）：缓存同样必须同步
    it2 = mem.add("用户偏好用英文回答，包括代码注释", importance=0.95)
    assert it2.id == it.id
    assert mem._norm_cached(it) == mem._norm(it.text)

    # 删除后缓存不得残留（防 id 复用场景读到旧归一化值）
    assert mem.remove(it.id) is True
    assert it.id not in mem._norms
