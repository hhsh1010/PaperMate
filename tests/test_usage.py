"""F10 · 用量增强后端测试（P2 体验优化）

覆盖：
- 会话用量补齐「生效模型 / 最近活动」（模型取值与 build_agent 优先级一致：显式 > 全局默认）
- 按天聚合的用量趋势（同日合并、跨日分组）
- days 参数边界夹紧
- 趋势合计与会话用量合计同源可对账
- 会话明细返回 model / last_at

数据层纯本地（SQLite），不触发大模型 / 网络，可离线运行。
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
    tmp = tempfile.mkdtemp(prefix="papermate_usage_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _seed(client, sid: int, entries: list[tuple[int, int, int, float]]) -> None:
    """直接写入 message 模拟用量。entries = [(days_ago, prompt_tokens, completion_tokens, cost), ...]

    注意：SQLite 落库时间为 naive UTC（DateTime(timezone=True) 在 SQLite 不保留时区），
    故这里也按 naive UTC 写入，与生产行为一致。
    """
    from papermate.models.db import Message

    SL = client.app.state.session_svc._Session
    now = datetime.now(UTC).replace(tzinfo=None)
    with SL() as db:
        for i, (days_ago, p, c, cost) in enumerate(entries):
            db.add(
                Message(
                    session_id=sid,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"m{i}",
                    prompt_tokens=p,
                    completion_tokens=c,
                    cost=cost,
                    created_at=now - timedelta(days=days_ago),
                )
            )
        db.commit()


def _today_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def test_usage_row_includes_model_and_last_at(client):
    s1 = client.post("/api/v1/sessions", json={"title": "显式模型", "config": {"model": "qwen-plus"}}).json()
    _seed(client, s1["id"], [(0, 100, 50, 0.002)])
    s2 = client.post("/api/v1/sessions", json={"title": "回落默认"}).json()
    _seed(client, s2["id"], [(1, 10, 5, 0.0001)])

    rows = {r["session_id"]: r for r in client.get("/api/v1/usage").json()}
    # 显式指定优先
    assert rows[s1["id"]]["model"] == "qwen-plus"
    # 未指定时回落全局默认（不硬编码具体值，只要求非空）
    assert rows[s2["id"]]["model"]
    # last_at 存在且时间序正确（0 天前 晚于 1 天前）
    assert rows[s1["id"]]["last_at"] > rows[s2["id"]]["last_at"]


def test_usage_trend_groups_by_day(client):
    s = client.post("/api/v1/sessions", json={"title": "趋势"}).json()
    # 今日两笔（应合并） + 2 天前一笔（单独一天）
    _seed(client, s["id"], [(0, 100, 10, 0.001), (0, 200, 20, 0.002), (2, 50, 5, 0.0005)])

    t = client.get("/api/v1/usage/trend", params={"days": 7}).json()
    by_date = {p["date"]: p for p in t["points"]}
    assert len(by_date) == 2, by_date

    today = by_date[_today_local()]
    assert today["cost"] == pytest.approx(0.003)
    assert today["prompt_tokens"] == 300
    assert today["completion_tokens"] == 30


def test_usage_trend_days_clamped(client):
    assert client.get("/api/v1/usage/trend", params={"days": 0}).json()["days"] == 1
    assert client.get("/api/v1/usage/trend", params={"days": -5}).json()["days"] == 1
    assert client.get("/api/v1/usage/trend", params={"days": 99999}).json()["days"] == 365


def test_trend_total_matches_usage_total(client):
    """趋势与会话用量同源（同为 messages 表），合计必须一致。"""
    s = client.post("/api/v1/sessions", json={"title": "对账"}).json()
    _seed(client, s["id"], [(0, 10, 1, 0.001), (3, 20, 2, 0.002)])

    total_rows = sum(r["cost"] for r in client.get("/api/v1/usage").json())
    t = client.get("/api/v1/usage/trend", params={"days": 365}).json()
    assert t["total_cost"] == pytest.approx(total_rows, abs=1e-9)


def test_session_usage_detail_includes_model(client):
    s = client.post("/api/v1/sessions", json={"title": "明细", "config": {"model": "deepseek-chat"}}).json()
    _seed(client, s["id"], [(0, 5, 1, 0.0001)])

    d = client.get(f"/api/v1/usage/sessions/{s['id']}").json()
    assert d["usage"]["model"] == "deepseek-chat"
    assert d["usage"]["last_at"]
    # 明细三层结构仍在
    assert len(d["messages"]) == 1
    assert isinstance(d["tool_runs"], list)
    assert isinstance(d["traces"], list)


def test_session_usage_detail_404(client):
    assert client.get("/api/v1/usage/sessions/999999").status_code == 404
