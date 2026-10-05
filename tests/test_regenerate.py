"""重新生成（regenerate）测试：POST /sessions/{id}/messages/stream 带 regenerate=true

核验点：
1. 重新生成复用最后一条用户消息重跑，并裁掉旧 assistant 回复，**不重复落库**（user 消息数=1）。
2. 会话无任何消息时调用 → 400（无用户轮次可重跑）。
3. 重新生成后返回新的 message 事件（新回复内容到位）。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_regen_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    rs_mod._state = None
    cfg_mod._settings = None


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    cur_name = None
    cur_data = ""
    for line in text.splitlines():
        if line.startswith("event:"):
            cur_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            cur_data = line[len("data:"):].strip()
            if cur_name is not None:
                events.append((cur_name, __import__("json").loads(cur_data)))
                cur_name, cur_data = None, ""
    return events


def _stream(client, sid: int, content: str, regenerate: bool = False) -> tuple[str, list[str]]:
    r = client.post(
        f"/api/v1/sessions/{sid}/messages/stream",
        json={"content": content, "regenerate": regenerate},
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    names = [n for n, _ in events]
    message = next((d.get("text", "") for n, d in events if n == "message"), "")
    return message, names


def _role_counts(client, sid: int) -> dict[str, int]:
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    counts: dict[str, int] = {}
    for m in detail["messages"]:
        counts[m["role"]] = counts.get(m["role"], 0) + 1
    return counts


def test_regenerate_reruns_last_turn_without_duplicate_user(client):
    """重新生成只重跑最后一轮，旧 assistant 被裁掉，user 消息不重复。"""
    sid = client.post("/api/v1/sessions", json={"title": "重生成会话"}).json()["id"]
    msg1, _ = _stream(client, sid, "你好")

    msg2, names = _stream(client, sid, "", regenerate=True)
    assert "message" in names
    assert "done" in names
    assert msg2  # 重新生成产出非空回复

    counts = _role_counts(client, sid)
    assert counts.get("user", 0) == 1, f"user 消息不应重复：{counts}"
    assert counts.get("assistant", 0) == 1, f"assistant 应仅 1 条（旧的被裁）：{counts}"


def test_regenerate_trims_old_assistant_and_returns_new_reply(client):
    """多次重新生成后，始终只有 1 条 user + 1 条 assistant（旧的被持续裁掉）。"""
    sid = client.post("/api/v1/sessions", json={"title": "多次重生成"}).json()["id"]
    _stream(client, sid, "介绍一下量子计算")
    for _ in range(3):
        _stream(client, sid, "", regenerate=True)
    counts = _role_counts(client, sid)
    assert counts.get("user", 0) == 1
    assert counts.get("assistant", 0) == 1


def test_regenerate_without_prior_turn_returns_400(client):
    """会话无任何消息时调用重新生成 → 400（无用户轮次可重跑）。"""
    sid = client.post("/api/v1/sessions", json={"title": "空会话"}).json()["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "", "regenerate": True})
    assert r.status_code == 400


def test_trim_last_assistant_clears_only_current_turn_tool_runs_and_traces(client):
    """F1 回归：裁剪必须按「本轮时间归属」清理 ToolRun/Trace，而不是跨表比较 id。

    缺陷背景：Message / ToolRun / Trace 是三张**各自独立自增**的表（无 turn_id 外键）。
    旧实现 `ToolRun.id > last_user.id` 拿 Message 的 id 去比 ToolRun 的 id，
    常见情况下一条都不命中 → 被重新生成那轮的工具记录残留（用量 num_tool_runs 虚高、
    Trace 页出现孤儿记录）；工具密集时又可能误删更早轮次。

    本用例构造「上一轮 + 本轮」两组工具记录，断言：本轮的被清理、上一轮的保留。
    旧实现下两条 ToolRun 都会被留下（断言失败），可稳定复现该缺陷。
    """
    from datetime import UTC, datetime, timedelta

    from papermate.models.db import Message, ToolRun, Trace

    sid = client.post("/api/v1/sessions", json={"title": "裁剪工具记录"}).json()["id"]
    chat_svc = client.app.state.chat_svc
    now = datetime.now(UTC)

    with chat_svc._Session() as db:
        # ---- 上一轮（必须保留）----
        db.add(Message(session_id=sid, role="user", content="上一轮提问",
                       created_at=now - timedelta(minutes=10)))
        db.add(Message(session_id=sid, role="assistant", content="上一轮回答",
                       created_at=now - timedelta(minutes=9)))
        db.add(ToolRun(session_id=sid, tool_name="retrieve", args_json="{}",
                       created_at=now - timedelta(minutes=9)))
        db.add(Trace(session_id=sid, trace_id="old-turn", events_json="[]",
                     created_at=now - timedelta(minutes=9)))
        # ---- 本轮（必须清理）----
        db.add(Message(session_id=sid, role="user", content="本轮提问", created_at=now))
        db.add(Message(session_id=sid, role="assistant", content="本轮回答",
                       created_at=now + timedelta(minutes=1)))
        db.add(ToolRun(session_id=sid, tool_name="retrieve", args_json="{}",
                       created_at=now + timedelta(minutes=1)))
        db.add(Trace(session_id=sid, trace_id="cur-turn", events_json="[]",
                     created_at=now + timedelta(minutes=1)))
        db.commit()

    turn = chat_svc.trim_last_assistant(sid)
    assert turn is not None and turn[0] == "本轮提问"

    with chat_svc._Session() as db:
        left_runs = db.query(ToolRun).filter(ToolRun.session_id == sid).count()
        left_traces = db.query(Trace).filter(Trace.session_id == sid).count()
        assert left_runs == 1, f"本轮 ToolRun 应清理、上一轮保留（旧实现漏删 → 残留 2 条），实际 {left_runs}"
        assert left_traces == 1, f"本轮 Trace 应清理、上一轮保留，实际 {left_traces}"
        contents = [m.content for m in db.query(Message).filter(Message.session_id == sid).all()]
        assert "本轮回答" not in contents, "本轮 assistant 应被裁掉"
        assert "上一轮回答" in contents, "历史轮次不受影响"


def test_empty_reply_attaches_fallback_notice(client, monkeypatch):
    """F2 回归：回复为空走兜底文案时，done 事件必须携带 notice。

    缺陷背景：三处兜底路径原为「先赋非空 _FALLBACK_REPLY，再 `not final_text.strip()`」，
    条件恒 False → notice 恒 None → loop_risk 强打断（2026-09-03 死循环复盘产物）在
    兜底场景从未触发。修复后应先记录 _was_empty 再赋兜底文案。

    本用例用返回空内容的桩 LLM 强制走兜底路径，断言 notice 存在且 kind=fallback。
    """
    import papermate.services.chat_service as cs

    class _EmptyDelta:
        delta = ""
        finished = True
        prompt_tokens = 0
        completion_tokens = 0
        # 与 `LLMStreamChunk` 保持同形（2026-09-16 增补）：流式用量修复后新增了
        # latency_ms / cost 字段，替身跟上可避免"生产端要不要防御取字段"的误判。
        latency_ms = 0
        cost = 0.0

    class _EmptyLLM:
        """恒定产出空内容的桩：stream 只给一个 finished 事件，delta 为空。"""

        model_name = "stub-empty"
        thinking_mode = False

        async def stream(self, messages, temperature=0.7, web_search=False):
            yield _EmptyDelta()

        async def chat(self, messages, tools=None, temperature=0.3):
            class _Resp:
                class message:
                    content = ""

            return _Resp()

    # 隔离无关副作用（记忆反思 / 自动标题），专注断言兜底 notice
    async def _noop_reflect(self, session_id, llm, conversation):
        return None

    async def _noop_title(self, session_id, llm, text):
        return None

    monkeypatch.setattr(cs.ChatService, "_llm_from_raw", lambda self, raw: _EmptyLLM())
    monkeypatch.setattr(cs.ChatService, "reflect", _noop_reflect)
    monkeypatch.setattr(cs.ChatService, "auto_title", _noop_title)

    sid = client.post("/api/v1/sessions", json={"title": "空回复兜底"}).json()["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    assert r.status_code == 200, r.text

    events = _parse_sse(r.text)
    msg_text = next((d.get("text", "") for n, d in events if n == "message"), "")
    done = next((d for n, d in events if n == "done"), None)
    assert done is not None, f"缺少 done 事件：{[n for n, _ in events]}"
    assert msg_text, "空回复必须有兜底文案，绝不空白"
    assert done.get("notice") and done["notice"].get("kind") == "fallback", (
        f"兜底回复必须携带 fallback notice（旧实现 notice 恒 None，防死循环机制失效）：{done}"
    )
