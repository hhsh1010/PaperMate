"""溯源率口径回归（2026-09-16，修复 `docs/用量与评测功能体检报告.md` 问题 3）

**修复前的双重错配**（实测数据：最近 10 条 Trace 里 message 事件共 **11297** 个，
其中 `delta` 流式片段 **11279** 个、`content` 完整回答仅 **18** 个）：

1. **消费者读了一个生产者从不写入的字段**：`citation_rate` 读 `message.data.citations`，
   而全库没有任何代码把 citations 写进事件 —— 引用只走 SSE 给前端，**从未落库** → 指标恒 0；
2. **样本口径错**：即便把引用写进去，11279 个 `delta` 片段也会被当成"回答样本"灌进分母，
   占比被稀释到约 0。

修复分两半（缺一不可，与问题 1 的"两个缺陷叠加"同构）：
- **生产者**：`persist_turn(..., citations=[...])` 把本轮引用附加到**最终回答事件**的 data；
- **消费者**：样本只取"完整回答"（`content` 非空），排除 `delta` 片段。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.core.loop import LoopEvent, LoopState
from papermate.skills.dev_eval import collect_metrics

CITES = [{"id": "doc-1-c3", "page": 2, "doc_id": 1, "text": "证据段落"}]


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_citation_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _answer(content: str = "这是回答") -> LoopEvent:
    return LoopEvent(type="message", state=LoopState.FINISH, data={"content": content})


def _delta(n: int) -> list[LoopEvent]:
    return [LoopEvent(type="message", state=LoopState.THINK, data={"delta": f"片{i}"}) for i in range(n)]


def _persist(client, events, **kwargs) -> int:
    sid = client.post("/api/v1/sessions", json={"title": "溯源率"}).json()["id"]
    client.app.state.chat_svc.persist_turn(sid, "问题", [], events, "这是回答", "fake", **kwargs)
    return sid


def _trace_events(client) -> list[dict]:
    with client.app.state.chat_svc._Session() as db:
        from papermate.models.db import Trace

        t = db.query(Trace).order_by(Trace.id.desc()).first()
        return json.loads(t.events_json or "[]")


# ---------------- 生产者：引用必须真的落库 ----------------


def test_citations_are_written_into_final_answer_event(client) -> None:
    """引用要落到**最终回答事件**的 `data.citations` —— 这是溯源率唯一的数据来源。"""
    _persist(client, [*_delta(3), _answer()], citations=CITES)
    events = _trace_events(client)

    answers = [e for e in events if e["type"] == "message" and (e["data"] or {}).get("content")]
    assert len(answers) == 1
    assert answers[0]["data"]["citations"] == CITES

    deltas = [e for e in events if e["type"] == "message" and "delta" in (e["data"] or {})]
    assert all("citations" not in d["data"] for d in deltas), "引用只应挂在完整回答事件上"


def test_persist_does_not_mutate_caller_events(client) -> None:
    """落库时**新建 dict** 再附加引用，不得悄悄改动调用方手里的事件对象（隐藏副作用）。"""
    ev = _answer()
    _persist(client, [ev], citations=CITES)

    assert "citations" not in (ev.data or {}), "调用方的事件对象被就地修改了"


def test_empty_citations_leave_events_untouched(client) -> None:
    """直答路径 `citations=[]`（不检索）→ 不注入字段、不报错。"""
    _persist(client, [_answer()], citations=[])
    events = _trace_events(client)

    answers = [e for e in events if e["type"] == "message" and (e["data"] or {}).get("content")]
    assert "citations" not in answers[0]["data"]


# ---------------- 消费者：样本口径 ----------------


def test_delta_fragments_are_excluded_from_sample(client) -> None:
    """核心回归：大量 delta 片段不得稀释溯源率。

    修复前：11279 个 delta 进分母 → 1/11280 ≈ 0.0001（看起来就是"0%"）。
    修复后：分母只算完整回答 → 1/1 = 1.0。
    """
    _persist(client, [*_delta(200), _answer()], citations=CITES)
    m = collect_metrics(client.app.state.chat_svc._Session)

    assert m["agent"]["citation_rate"] == 1.0, f"delta 片段被当成回答样本：{m['agent']}"


def test_no_citation_answer_reports_zero_honestly(client) -> None:
    """没有引用的回答要如实记 0，不能因为"有检索证据"就虚报成 1。"""
    _persist(client, [_answer()], citations=[])
    m = collect_metrics(client.app.state.chat_svc._Session)

    assert m["agent"]["citation_rate"] == 0.0


def test_mixed_traces_give_expected_ratio(client) -> None:
    """一条带引用 + 一条不带引用 → 0.5（分子分母都只算完整回答）。"""
    _persist(client, [_answer("回答一")], citations=CITES)
    _persist(client, [*_delta(50), _answer("回答二")], citations=[])
    m = collect_metrics(client.app.state.chat_svc._Session)

    assert m["agent"]["citation_rate"] == pytest.approx(0.5), m["agent"]


def test_collect_metrics_reports_citations_for_dashboard(client) -> None:
    """看板取数：`collect_metrics` 是 `/eval/summary` 与 `/eval` 技能的唯一入口。"""
    _persist(client, [_answer()], citations=CITES)
    m = collect_metrics(client.app.state.chat_svc._Session)

    assert m["agent"]["available"] is True
    assert m["agent"]["count"] >= 1
    assert m["agent"]["citation_rate"] == 1.0
    assert m["recent_traces"], "看板仍需返回最近轨迹摘要"


# ---------------- B2-X1（2026-10-01）：历史 / 刷新后引用仍须可见 ----------------


def _history(client, sid: int) -> list[dict]:
    """取会话历史消息（前端"刷新 / 切换会话"走的就是这个接口）。"""
    r = client.get(f"/api/v1/sessions/{sid}")
    assert r.status_code == 200, r.text
    return r.json()["messages"]


def test_citations_are_returned_in_session_history(client) -> None:
    """★ B2-X1：引用必须随 `GET /sessions/{id}` 一起回传。

    修复前：引用只落进 trace 的事件，`messages` 侧没有任何痕迹 ⇒ 前端只能把历史消息的
    `citations` 写死成 `[]` ⇒ **当轮看得到、刷新即消失**（"指标有数、用户看不到"）。
    """
    sid = _persist(client, [_answer()], citations=CITES)

    assistant = [m for m in _history(client, sid) if m["role"] == "assistant"]
    assert assistant, "历史里没有 assistant 消息"
    assert assistant[-1]["citations"] == CITES


def test_citations_are_not_duplicated_into_meta(client) -> None:
    """引用提升为独立字段后**不得**在 meta 里再出现一次（否则响应载荷翻倍）。"""
    sid = _persist(client, [_answer()], citations=CITES)

    a = [m for m in _history(client, sid) if m["role"] == "assistant"][-1]
    assert "citations" not in a["meta"], "同一份引用被返回了两次（字段 + meta）"


def test_history_without_citations_returns_empty_list(client) -> None:
    """无引用的轮次：字段存在且为空数组（前端无需额外判空分支）。"""
    sid = _persist(client, [_answer()], citations=[])

    a = [m for m in _history(client, sid) if m["role"] == "assistant"][-1]
    assert a["citations"] == []
    assert "citations" not in a["meta"], "空引用不该往 meta 里塞无意义字段"


def test_corrupt_or_non_object_meta_does_not_break_history(client) -> None:
    """meta_json 损坏 / 不是对象时，历史接口仍须可用（与该不变量在用量接口上的既有测试同口径）。

    这条同时覆盖新增的"引用提升"逻辑：解析失败必须退化成空引用，而不是让整页历史 500。
    """
    from papermate.models.db import Message

    sid = client.post("/api/v1/sessions", json={"title": "坏元数据"}).json()["id"]
    SL = client.app.state.chat_svc._Session
    with SL() as db:
        db.add(Message(session_id=sid, role="assistant", content="回答1", meta_json="{ not json"))
        db.add(Message(session_id=sid, role="assistant", content="回答2", meta_json="[1, 2]"))
        db.add(Message(session_id=sid, role="assistant", content="回答3",
                       meta_json=json.dumps({"citations": "不是数组"})))
        db.commit()

    msgs = _history(client, sid)

    assert [m["content"] for m in msgs] == ["回答1", "回答2", "回答3"]
    assert all(m["citations"] == [] for m in msgs), "脏元数据必须退化为空引用"
