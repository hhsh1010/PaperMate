"""用量接口 ↔ 数据库真值 的**独立口径**对拍测试（2026-10-05）。

## 为什么需要这个测试

`routes_usage.py` 的 docstring 承诺「趋势与会话用量**同源**（同为 messages 表），
因此趋势合计与会话用量合计天然可对账，不存在两套口径」——
但在这份测试之前，这句话只是**注释里的承诺**，没有任何测试守护它：

- `tests/test_usage.py::test_trend_total_matches_usage_total` 是 **API ↔ API** 对拍
  （趋势 vs 会话列表），两者**都**来自 `messages` 表 ⇒ 若聚合逻辑整体写错，两边一起错、测试照样绿。
- `tests/test_o09_batch_queries.py::test_count_usage_all_matches_per_session` 是
  **服务方法 ↔ 服务方法**对拍（批量 vs 逐会话），同样共享同一套字段口径。

⇒ 两者都属于 MEMORY 里记的「**两把尺子（量纲不同/同源）**」陷阱：数值同源只会带来**虚假安全感**。
本测试补上缺失的那把「**独立尺子**」：把原始行**逐条拉到 Python 内存里累加**
（不使用 SQL 的 COUNT/SUM，也不调用 `ChatService.count_usage*`），
再与 `/usage`、`/usage/trend`、`/usage/sessions/{id}` 的输出逐字段比对。

## 覆盖

1. `/usage` 每行 5 个字段 ↔ DB 真值（含空会话 / 仅工具无消息 / 字段为 NULL 三类边界）；
2. 用量页 4 张统计卡（`UsagePanel.tsx:170-176` 的会话数 / tokens / 工具数 / 成本）↔ DB 真值；
3. `/usage/trend` 的合计（成本 / prompt / completion）↔ DB 真值；
4. `/usage/sessions/{id}` 明细的 `usage` 块 ↔ 同一真值（与列表行同源）。

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
    tmp = tempfile.mkdtemp(prefix="papermate_usage_recon_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _truth_from_db(SL, session_id: int) -> dict:
    """**独立口径**：把该会话的原始行逐条累加（不复用被测的 SQL 聚合）。

    ★ 这是本测试的核心 —— 若改用 `db.query(func.sum(...))`，就与实现用了**同一把尺子**，
      实现写错时测试会跟着一起错（MEMORY 的「两把尺子」教训）。
    """
    from papermate.models.db import Message, ToolRun

    with SL() as db:
        msgs = db.query(Message).filter(Message.session_id == session_id).all()
        runs = db.query(ToolRun).filter(ToolRun.session_id == session_id).all()

    return {
        "num_messages": len(msgs),
        "num_tool_runs": len(runs),
        "prompt_tokens": sum(int(m.prompt_tokens or 0) for m in msgs),
        "completion_tokens": sum(int(m.completion_tokens or 0) for m in msgs),
        "cost": round(sum(float(m.cost or 0.0) for m in msgs), 6),
    }


def _truth_all(SL) -> dict:
    """全库真值（跨会话汇总），用于卡片合计与趋势对账。"""
    from papermate.models.db import Message, ToolRun

    with SL() as db:
        msgs = db.query(Message).all()
        runs = db.query(ToolRun).all()
    return {
        "num_sessions_with_messages": len({m.session_id for m in msgs}),
        "num_messages": len(msgs),
        "num_tool_runs": len(runs),
        "prompt_tokens": sum(int(m.prompt_tokens or 0) for m in msgs),
        "completion_tokens": sum(int(m.completion_tokens or 0) for m in msgs),
        "cost": round(sum(float(m.cost or 0.0) for m in msgs), 6),
    }


def _seed(client, session_id: int, entries: list[tuple]) -> None:
    """写入 message（entries = [(days_ago, prompt, completion, cost), ...]）。

    落库时间为 naive UTC（与生产一致：SQLite 的 DateTime(timezone=True) 不保留时区）。
    """
    from papermate.models.db import Message

    SL = client.app.state.session_svc._Session
    now = datetime.now(UTC).replace(tzinfo=None)
    with SL() as db:
        for i, (days_ago, p, c, cost) in enumerate(entries):
            db.add(
                Message(
                    session_id=session_id,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"m{i}",
                    prompt_tokens=p,
                    completion_tokens=c,
                    cost=cost,
                    created_at=now - timedelta(days=days_ago),
                )
            )
        db.commit()


def _seed_tool_runs(client, session_id: int, n: int) -> None:
    from papermate.models.db import ToolRun

    SL = client.app.state.session_svc._Session
    with SL() as db:
        for i in range(n):
            db.add(ToolRun(session_id=session_id, tool_name="retrieve", args_json="{}", ok=(i % 2 == 0)))
        db.commit()


# --------------------------------------------------------- 1) /usage 逐行对账
def test_usage_rows_match_db_truth(client):
    """`/usage` 每行 5 字段必须等于 DB 真值（含三类边界会话）。

    构造四种形态（缺一不可，否则这些分支不会被覆盖）：
    - A 正常多消息 + 工具；
    - B 空会话（无消息无工具）⇒ 必须零值兜底；
    - C 仅工具无消息 ⇒ num_tool_runs>0 但 tokens/cost 为 0
      （`count_usage_all` 只按 Message GROUP BY 会漏掉它，见 chat_service.py:989-990）；
    - D token 字段为 NULL ⇒ 必须按 0 计，不得让整行变 None。
    """
    from papermate.models.db import Message

    SL = client.app.state.session_svc._Session
    a = client.post("/api/v1/sessions", json={"title": "A 正常"}).json()["id"]
    b = client.post("/api/v1/sessions", json={"title": "B 空会话"}).json()["id"]
    c = client.post("/api/v1/sessions", json={"title": "C 仅工具"}).json()["id"]
    d = client.post("/api/v1/sessions", json={"title": "D NULL 字段"}).json()["id"]

    _seed(client, a, [(0, 100, 50, 0.002), (1, 200, 20, 0.003)])
    _seed_tool_runs(client, a, 2)
    _seed_tool_runs(client, c, 3)
    now = datetime.now(UTC).replace(tzinfo=None)
    with SL() as db:  # D：token/cost 全为 NULL
        db.add(Message(session_id=d, role="user", content="x", created_at=now))
        db.commit()

    rows = {r["session_id"]: r for r in client.get("/api/v1/usage").json()}
    assert set(rows) >= {a, b, c, d}, "每个会话都必须在 /usage 中出现（含空会话）"

    for sid in (a, b, c, d):
        truth = _truth_from_db(SL, sid)
        row = rows[sid]
        for k in ("num_messages", "num_tool_runs", "prompt_tokens", "completion_tokens"):
            assert row[k] == truth[k], f"session={sid} field={k}: api={row[k]} vs db={truth[k]}"
        assert row["cost"] == pytest.approx(truth["cost"], abs=1e-9), f"session={sid} cost 不一致"

    # 防"空测试假绿"：确认 A 确实非零，否则上面的循环可能全在比 0 == 0
    assert rows[a]["prompt_tokens"] == 300 and rows[a]["num_tool_runs"] == 2
    assert rows[a]["cost"] == pytest.approx(0.005)
    assert rows[c]["num_tool_runs"] == 3 and rows[c]["num_messages"] == 0
    # B 空会话：全部零值兜底，不得是 None（前端 reduce 会 NaN）
    assert rows[b]["num_messages"] == 0 and rows[b]["num_tool_runs"] == 0
    assert rows[b]["prompt_tokens"] == 0 and rows[b]["completion_tokens"] == 0
    assert rows[b]["cost"] == 0.0 and rows[b]["last_at"] is None


# --------------------------------------------------------- 2) 用量页卡片合计对账
def test_usage_card_totals_match_db_truth(client):
    """用量页 4 张卡（`UsagePanel.tsx:170-176`）的合计必须等于 DB 真值。

    卡片口径（前端原样复刻）：
      sessions = rows.length · tokens = Σ(prompt+completion)
      tools    = Σ num_tool_runs · cost = Σ cost
    """
    SL = client.app.state.session_svc._Session
    s1 = client.post("/api/v1/sessions", json={"title": "1"}).json()["id"]
    s2 = client.post("/api/v1/sessions", json={"title": "2"}).json()["id"]
    client.post("/api/v1/sessions", json={"title": "3 空"})  # 空会话也计入"会话数"

    _seed(client, s1, [(0, 100, 10, 0.001), (2, 250, 25, 0.0025)])
    _seed(client, s2, [(1, 40, 4, 0.0004)])
    _seed_tool_runs(client, s1, 1)
    _seed_tool_runs(client, s2, 4)

    rows = client.get("/api/v1/usage").json()
    truth = _truth_all(SL)

    # 前端卡片公式
    card_sessions = len(rows)
    card_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows)
    card_tools = sum(r["num_tool_runs"] for r in rows)
    card_cost = sum(r["cost"] for r in rows)

    assert card_sessions == 3, "会话数应含空会话"
    assert card_tokens == truth["prompt_tokens"] + truth["completion_tokens"], "tokens 卡对不上 DB"
    assert card_tools == truth["num_tool_runs"], "工具数卡对不上 DB"
    assert card_cost == pytest.approx(truth["cost"], abs=1e-9), "成本卡对不上 DB"
    # 非零护栏：确保上面的等式不是在比 0
    assert card_tokens > 0 and card_tools > 0 and card_cost > 0


# --------------------------------------------------------- 3) 趋势合计对账
def test_trend_total_matches_db_truth(client):
    """`/usage/trend` 的合计（成本 / prompt / completion）必须等于 DB 真值。

    ⚠️ 与 `/usage` 不同，趋势**有窗口**（`created_at >= cutoff`）—— 故全部样本都落在窗口内，
       否则"合计对不上"会变成窗口差异而非 bug。
    """
    SL = client.app.state.session_svc._Session
    s = client.post("/api/v1/sessions", json={"title": "趋势对账"}).json()["id"]
    _seed(client, s, [(0, 100, 10, 0.001), (1, 200, 20, 0.002), (4, 50, 5, 0.0005)])
    _seed_tool_runs(client, s, 2)

    t = client.get("/api/v1/usage/trend", params={"days": 365}).json()
    truth = _truth_all(SL)

    assert t["total_cost"] == pytest.approx(truth["cost"], abs=1e-9), "趋势总成本 ≠ DB"
    assert sum(p["prompt_tokens"] for p in t["points"]) == truth["prompt_tokens"], "趋势 prompt 合计 ≠ DB"
    assert sum(p["completion_tokens"] for p in t["points"]) == truth["completion_tokens"], "趋势 completion 合计 ≠ DB"
    assert truth["prompt_tokens"] > 0, "非零护栏"


# --------------------------------------------------------- 4) 会话明细 usage 块对账
def test_session_detail_usage_matches_db_truth(client):
    """`/usage/sessions/{id}` 的 `usage` 块必须与**同一真值**一致（明细与列表同源）。"""
    SL = client.app.state.session_svc._Session
    s = client.post("/api/v1/sessions", json={"title": "明细对账"}).json()["id"]
    _seed(client, s, [(0, 7, 3, 0.0007), (1, 11, 5, 0.0011)])
    _seed_tool_runs(client, s, 2)

    detail = client.get(f"/api/v1/usage/sessions/{s}").json()
    truth = _truth_from_db(SL, s)
    u = detail["usage"]

    for k in ("num_messages", "num_tool_runs", "prompt_tokens", "completion_tokens"):
        assert u[k] == truth[k], f"detail.usage.{k}: api={u[k]} vs db={truth[k]}"
    assert u["cost"] == pytest.approx(truth["cost"], abs=1e-9)

    # 明细里的逐条消息 + 工具行数也必须与真值自洽（防"汇总对、明细错"）
    assert len(detail["messages"]) == truth["num_messages"]
    assert len(detail["tool_runs"]) == truth["num_tool_runs"]


# --------------------------------------------------------- 5) 列表行 ↔ 明细块 交叉一致
def test_list_row_equals_detail_block(client):
    """同一会话：`/usage` 行 ↔ `/usage/sessions/{id}.usage` 必须逐字段一致。

    两者走**不同代码路径**（前者 `count_usage_all` 批量 GROUP BY，后者 `count_usage` 单会话），
    若哪天有人只改其中一条，这里会立刻变红。
    """
    s = client.post("/api/v1/sessions", json={"title": "两路径"}).json()["id"]
    _seed(client, s, [(0, 33, 3, 0.0033), (2, 66, 6, 0.0066)])
    _seed_tool_runs(client, s, 1)

    row = next(r for r in client.get("/api/v1/usage").json() if r["session_id"] == s)
    u = client.get(f"/api/v1/usage/sessions/{s}").json()["usage"]

    for k in ("num_messages", "num_tool_runs", "prompt_tokens", "completion_tokens", "cost"):
        assert row[k] == pytest.approx(u[k], abs=1e-9), f"field={k}: list={row[k]} vs detail={u[k]}"
