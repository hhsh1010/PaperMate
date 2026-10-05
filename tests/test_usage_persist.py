"""用量落库口径回归（2026-09-16，修复 `docs/用量与评测功能体检报告.md` 问题 2）

**修复前的两个语义错误**（`chat_service.persist_turn`）：

1. `cost = cost_override if cost_override is not None else estimate_cost(...)`
   —— 上游因"拿不到用量"回传的 `0.0` 也被当成真值 → **用量页显示 ¥0 而实际在花钱**；
2. `if total_tokens_override is not None: pt, ct = total_tokens_override, 0`
   —— `0` 同样命中 → **精确值 0 覆盖掉本地粗算**；
   顺带把输入/输出合并成一个总数（`ct` 恒 0），用量页失去分项。

实测症状（修复前）：正常走完 Agent 循环的轮次写 `0/0/0.0`，
反而只有没发 `done` 事件的轮次（技能/中断/失败兜底）才落到粗算兜底值 —— **语义反转**。

修复后规则：
- `cost_override` 为**假值（None / 0）**时一律回退本地估算（对未定价 / 假模型无副作用，两边都算 0）；
- 分项 override **只接受正数**（`0` = "上游没有这个数"，不是"真的是 0"）；
- 只有总数没有分项时，用总数补齐缺失的一侧（保证总量对得上）；
- 传了 0 成本但本地能算出值时落一条 WARNING，便于发现上游回归。

断言走**真实 HTTP 端点** `/usage/sessions/{id}`（与用量页同一数据源），
既验落库、又验端出口径。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod

PRICED_MODEL = "deepseek-v4-flash"  # _PRICING: in=0.001 / out=0.002（¥/1k tokens）


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_usage_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _turn(client, **kwargs) -> dict:
    """建会话 → 落一轮用量 → 返回 `/usage/sessions/{id}` 的响应体。"""
    sid = client.post("/api/v1/sessions", json={"title": "用量口径"}).json()["id"]
    client.app.state.chat_svc.persist_turn(sid, "问题", [], [], "回答", PRICED_MODEL, **kwargs)
    r = client.get(f"/api/v1/usage/sessions/{sid}")
    assert r.status_code == 200
    return r.json()


def _assistant(body: dict) -> dict:
    return next(m for m in body["messages"] if m["role"] == "assistant")


def test_exact_prompt_and_completion_tokens_are_persisted(client) -> None:
    """分项用量必须精确落库 —— 旧的 `total_tokens_override` 会把 ct 写成 0。"""
    body = _turn(client, cost_override=0.0005,
                 prompt_tokens_override=100, completion_tokens_override=200)
    msg = _assistant(body)

    assert (msg["prompt_tokens"], msg["completion_tokens"]) == (100, 200)
    assert msg["cost"] == pytest.approx(0.0005)
    assert body["usage"]["prompt_tokens"] == 100
    assert body["usage"]["completion_tokens"] == 200


def test_zero_override_falls_back_to_local_estimate(client) -> None:
    """上游回传 0（"没有用量"）时**不得**把 0 当精确值 —— 必须回退本地估算。"""
    body = _turn(client, cost_override=0.0,
                 prompt_tokens_override=0, completion_tokens_override=0)
    msg = _assistant(body)

    assert msg["prompt_tokens"] > 0, "精确值 0 覆盖了粗算兜底（修复前的症状）"
    assert msg["completion_tokens"] > 0
    assert msg["cost"] > 0, "0 成本被当真值 → 用量页会显示 ¥0 而实际在花钱"


def test_zero_cost_with_real_tokens_is_recomputed(client) -> None:
    """只给了 token 没给成本（或成本为 0）时，按定价表补算成本。"""
    body = _turn(client, cost_override=0.0,
                 prompt_tokens_override=1000, completion_tokens_override=1000)
    msg = _assistant(body)

    assert (msg["prompt_tokens"], msg["completion_tokens"]) == (1000, 1000)
    # 1000/1000*0.001 + 1000/1000*0.002 = 0.003
    assert msg["cost"] == pytest.approx(0.003)


def test_legacy_total_only_still_works(client) -> None:
    """向后兼容：只给总数（无分项）的遗留调用方仍能落库，且**总量不丢**。"""
    body = _turn(client, total_tokens_override=300)
    msg = _assistant(body)

    assert msg["prompt_tokens"] + msg["completion_tokens"] == 300


def test_total_tokens_fills_missing_side(client) -> None:
    """只给了 pt + 总数时，缺失的 ct 由总数补齐（避免总量对不上）。"""
    body = _turn(client, prompt_tokens_override=120, total_tokens_override=360)
    msg = _assistant(body)

    assert (msg["prompt_tokens"], msg["completion_tokens"]) == (120, 240)


def test_trace_mirrors_message_usage(client) -> None:
    """Trace 的 cost / latency 必须与消息一致（看板与用量页才能对账）。"""
    body = _turn(client, cost_override=0.0012,
                 prompt_tokens_override=10, completion_tokens_override=20, latency_ms=880)
    msg, trace = _assistant(body), body["traces"][-1]

    assert trace["total_cost"] == pytest.approx(msg["cost"]) == pytest.approx(0.0012)
    assert trace["total_latency_ms"] == 880, "Trace 耗时此前恒 0（流式路径没测延迟）"


def test_invalid_meta_unchanged_and_empty_answer_keeps_zero_ct(client) -> None:
    """回归保护：meta（invalid/notice）与用量互不影响；**真正的空回答** ct 保持 0。"""
    notice = {"kind": "fallback", "title": "⚠️ 未生成有效回答"}
    sid = client.post("/api/v1/sessions", json={"title": "空回答"}).json()["id"]
    # 空回答：final_text 为空 → 本地粗算 ct 也是 0，`completion_tokens_override=0` 与之自洽
    client.app.state.chat_svc.persist_turn(
        sid, "问题", [], [], "", PRICED_MODEL,
        cost_override=0.0, prompt_tokens_override=5, completion_tokens_override=0,
        meta={"invalid": True, "notice": notice},
    )
    body = client.get(f"/api/v1/usage/sessions/{sid}").json()
    msg = _assistant(body)

    assert msg["prompt_tokens"] == 5, "pt 有精确值应原样采用"
    assert msg["completion_tokens"] == 0, "空回答的 ct 确实是 0（不是缺失）"


# --------------------------------------------------------------- 问题 7：记录模型


def test_message_records_effective_model(client) -> None:
    """每轮必须记录**实际生效的模型** —— 成本异常时才能回答"这笔钱花在哪个模型上"。

    语义差别（正是要修的点）：
      - `usage.model` 来自会话 config → 回答"该用什么模型"；
      - `messages[i].model` 来自落库轮次 → 回答"**实际用了什么**"。
    回退 / 降级路径下两者会不一致，此前只有前者，于是"谁花的钱"无从查证。
    """
    body = _turn(client)
    msg = _assistant(body)

    assert msg["model"] == PRICED_MODEL, "assistant 消息应记录本轮实际模型"
    assert body["messages"][0]["model"] is None, "用户消息没有模型可言，如实留空"


def test_legacy_message_without_model_is_not_backfilled(client) -> None:
    """修复前落库的老消息 → model 为 None，**不得用会话当前模型倒填**（那是伪造历史）。"""
    from papermate.models.db import Message

    sid = client.post("/api/v1/sessions", json={"title": "老消息"}).json()["id"]
    SL = client.app.state.chat_svc._Session
    with SL() as db:
        db.add(Message(session_id=sid, role="assistant", content="历史回答",
                       prompt_tokens=11, completion_tokens=22, cost=0.0, meta_json="{}"))
        db.commit()

    body = client.get(f"/api/v1/usage/sessions/{sid}").json()

    assert _assistant(body)["model"] is None
    assert body["usage"]["model"] == PRICED_MODEL or isinstance(body["usage"]["model"], str)


def test_corrupt_meta_json_does_not_break_usage(client) -> None:
    """meta_json 损坏时用量接口必须仍可用（元数据不该让整页 500）。"""
    from papermate.models.db import Message

    sid = client.post("/api/v1/sessions", json={"title": "坏元数据"}).json()["id"]
    SL = client.app.state.chat_svc._Session
    with SL() as db:
        db.add(Message(session_id=sid, role="assistant", content="回答",
                       prompt_tokens=1, completion_tokens=2, cost=0.0, meta_json="{ not json"))
        db.commit()

    r = client.get(f"/api/v1/usage/sessions/{sid}")

    assert r.status_code == 200
    assert _assistant(r.json())["model"] is None
