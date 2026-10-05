"""四类「未收尾 / 失败」的归因与处理（2026-09-30）。

**需求**：主动停止 / 对话崩溃 / 对话超时 / 工具调用失败 —— **都要有对应处理**。

"处理"= 三件事，缺一不可：
① **打点**（events 留下可识别标记）② **归因**（metrics 能分桶）③ **用户可见**。

本文件守护 ①②；③ 的 notice 是 `routes_search` 内的嵌套函数（不可从外部调用），
用**源码契约**断言它存在且是**中性提示**而非告警（项目无 testing-library 的既有做法）。

★ 一条关键口径：**打点缺口必须继续归 `unknown`** ——
新增了 `timeout` / `crash` 不代表可以"顺手"把老轨迹的缺口算成已知类别
（那正是 `unknown` 存在的意义：暴露缺口，而不是粉饰它）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from papermate.api.routes_metrics import router
from papermate.models.db import ChatSession, Message, Trace, init_db, make_engine

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _ev_done() -> dict:
    return {"type": "done", "data": {}}


def _ev_err(kind: str | None = "llm_error") -> dict:
    data = {} if kind is None else {"failure_kind": kind}
    return {"type": "error", "data": data}


def _ev_cancelled() -> dict:
    """「用户主动停止」的独立事件（2026-09-30 新增打点）。"""
    return {"type": "cancelled", "data": {"reason": "user_stop"}}


# ---- ① 常量 -------------------------------------------------------------------
def test_timeout_and_crash_are_real_failures() -> None:
    """超时 / 崩溃是**真故障** ⇒ 必须计入失败率；而 `cancelled` 必须不计。"""
    from lightcore.core import failure as mf

    for k in (mf.TIMEOUT, mf.CRASH):
        assert k in mf.ALL_KINDS
        assert k in mf.REAL_FAILURES, f"{k} 是真故障，必须计入失败率"
        assert k in mf.LABELS
    assert mf.CANCELLED not in mf.REAL_FAILURES, "用户取消不是故障（既有设计，勿改）"


# ---- ② 归因 -------------------------------------------------------------------
def test_cancelled_event_maps_to_interrupted() -> None:
    """★ 新增的 `cancelled` 事件 ⇒ `interrupted`（不是失败）。"""
    from lightcore.core.trace_metrics import classify_events

    assert classify_events([_ev_cancelled()]) == "interrupted"
    assert classify_events([{"type": "thinking"}, _ev_cancelled()]) == "interrupted"


def test_has_user_stop_mark_distinguishes_stop_from_crash() -> None:
    """★★ 这才是 `cancelled` 打点**真正的价值**：区分「确认用户停止」与「崩溃/未知中断」。

    ★ 教训（**反向自检发现的**）：`classify_events` 对这两者**都返回 `interrupted`**
      （因为崩溃来不及写事件，两者看上去一样）⇒ 若只断言分类，
      `cancelled` 打点**有没有都测不出差别**，等于白打。
      ⇒ 必须有一个"**有没有证据**"的独立判定，否则打点不可观测。
    """
    from lightcore.core.trace_metrics import classify_events, has_user_stop_mark

    stopped = [{"type": "thinking"}, _ev_cancelled()]
    crashed = [{"type": "thinking"}]
    assert has_user_stop_mark(stopped) is True
    assert has_user_stop_mark(crashed) is False
    # 两者分类**相同** ⇒ 分类不足以区分，必须靠 has_user_stop_mark
    assert classify_events(stopped) == classify_events(crashed) == "interrupted"


def test_crash_and_timeout_map_to_failed() -> None:
    from lightcore.core import failure as mf
    from lightcore.core.trace_metrics import classify_events

    assert classify_events([_ev_err(mf.CRASH)]) == "failed"
    assert classify_events([_ev_err(mf.TIMEOUT)]) == "failed"


def test_missing_kind_still_unknown_after_new_kinds() -> None:
    """★★ 打点缺口**仍然**归 `unknown` —— 不能被新类别"顺手吃掉"。"""
    from lightcore.core.trace_metrics import classify_events

    assert classify_events([_ev_err(None)]) == "unknown"
    assert classify_events([{"type": "error", "data": {}}]) == "unknown"


def test_error_beats_cancelled() -> None:
    """既有 cancelled 又有真 error ⇒ 以 error 为准（不能把故障藏进"用户停止"里）。"""
    from lightcore.core.trace_metrics import classify_events

    assert classify_events([_ev_cancelled(), _ev_err("llm_error")]) == "failed"


def test_tool_timeout_event_has_failure_kind() -> None:
    """★ 工具超时的 `tool_result` 事件必须带 `failure_kind=timeout`（源码契约）。

    `loop.py` 的 `_fail_kind` 是函数内的局部变量，无法从外部断言 ⇒ 用源码契约。
    """
    src = (_SRC_ROOT / "lightcore" / "core" / "loop.py").read_text(encoding="utf-8")
    assert "_fail_kind = failure.TIMEOUT" in src, "工具超时必须单独归因（与参数错区分）"
    assert '"failure_kind": _fail_kind' in src, "失败事件必须带归因分类"


# ---- ③ 用户可见 ----------------------------------------------------------------
def test_user_stop_notice_is_neutral_not_alarm() -> None:
    """★ 停止的回执必须是**中性提示**：标题不得含"错误/失败/异常"等告警措辞。

    理由：用户点「停止」是正常操作，染成告警会把正常行为渲染成"出问题了"
    （与 `unfinished_rate` 刻意不叫 error_rate 同源）。
    """
    src = (_SRC_ROOT / "papermate" / "api" / "routes_search.py").read_text(encoding="utf-8")
    assert "_user_stop_notice" in src
    assert "已停止本轮" in src
    assert '"interrupted": True, "notice": _user_stop_notice()' in src, (
        "停止时除 meta.interrupted 外，还必须落 notice（刷新后仍可见）"
    )


def test_sse_error_payload_carries_failure_kind() -> None:
    """★ 发给前端的 error 事件也要带 `failure_kind`（此前只有 reason、落库无痕）。"""
    src = (_SRC_ROOT / "papermate" / "api" / "routes_search.py").read_text(encoding="utf-8")
    assert '"failure_kind": m_failure.normalize(kind)' in src


# ---- ④ 端到端：停止与崩溃不再混为一谈 ------------------------------------------
def _mk_trace(tid: str, events: list[dict], *, latency: int = 1000) -> Trace:
    return Trace(
        trace_id=tid, session_id=1,
        events_json=json.dumps(events, ensure_ascii=False),
        total_steps=2, total_cost=0.01, total_latency_ms=latency,
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


def test_stopped_trace_is_interrupted_and_excluded_from_failure_rate(client) -> None:
    """端到端：带 `cancelled` 事件的轮次 ⇒ interrupted，**不计入**失败率。"""
    from lightcore.core import failure as mf

    c, Session = client
    with Session() as s:
        s.add(_mk_trace("t1", [{"type": "thinking"}, _ev_cancelled()]))     # 用户停止
        s.add(_mk_trace("t2", [_ev_done()]))                                # 正常
        s.add(_mk_trace("t3", [_ev_err(mf.CRASH)]))                         # 崩溃（真失败）
        s.add(_mk_trace("t4", [_ev_err(mf.TIMEOUT)]))                       # 超时（真失败）
        s.commit()
    m = c.get("/metrics").json()
    assert m["unfinished_breakdown"] == {
        "done": 1, "interrupted": 1, "unknown": 0, "failed": 2,
    }
    # ★ 用户停止被排除；崩溃与超时计入
    assert m["unfinished_rate_excluding_interrupt"] == 0.5
    assert m["failed_confirmed_rate"] == 0.5
    # ★★ cancelled 打点**必须可观测** —— 否则打了也看不出效果（反向自检的教训）
    assert m["confirmed_user_stop"] == 1
