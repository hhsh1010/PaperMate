"""请求级流取消令牌（F8）回归。

缺陷背景：取消事件按 session_id **单槽**存储（stream_cancels[session_id] = Event），
同会话并发流（多标签页 / 上一轮未结束就重新生成）会互相覆盖，
先前的流永远收不到停止信号；且条目从不清理。

现改为 (session_id, stream_id) 请求级令牌：
- /stop 带 stream_id → 只停止该条流，同会话其它流不受影响；
- /stop 不带 stream_id → 停止该会话全部进行中流（兼容旧客户端 / 兜底）；
- 流结束（正常 / 异常 / 客户端断开）时注销自己的令牌。
"""

from __future__ import annotations

import os
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_streamtok_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    rs_mod._state = None
    cfg_mod._settings = None


def test_stream_token_is_cleaned_up_after_stream_ends(client):
    """流结束后必须注销本请求的令牌（旧实现只写不删，条目残留）。"""
    sid = client.post("/api/v1/sessions", json={"title": "令牌清理"}).json()["id"]
    r = client.post(
        f"/api/v1/sessions/{sid}/messages/stream",
        json={"content": "你好", "stream_id": "tok-cleanup"},
    )
    assert r.status_code == 200, r.text
    cancels = client.app.state.stream_cancels
    assert (sid, "tok-cleanup") not in cancels, f"流结束后令牌应被注销，残留={cancels}"


def test_stop_with_stream_id_only_targets_that_stream(client):
    """带 stream_id：只停止该条流；不带：停止该会话全部流（兼容语义）。

    旧实现按 session_id 单槽存储，无法区分同一会话的不同流，
    「停止」会打到后启动的流上，先前的流永远停不掉。
    """
    cancels = client.app.state.stream_cancels
    ev_other = threading.Event()

    # ① 指定了一个不存在的 stream_id → 同会话其它流不应被误停
    cancels[(999001, "other-stream")] = ev_other
    try:
        r = client.post("/api/v1/sessions/999001/stop?stream_id=not-this-one")
        assert r.status_code == 200 and r.json() == {"ok": True}
        assert not ev_other.is_set(), "指定了 stream_id 时不应误停同会话的其它流"
    finally:
        cancels.pop((999001, "other-stream"), None)

    # ② 不带 stream_id → 会话级兜底：置位该会话全部流
    ev_other = threading.Event()
    cancels[(999001, "any-stream")] = ev_other
    try:
        r2 = client.post("/api/v1/sessions/999001/stop")
        assert r2.status_code == 200 and r2.json() == {"ok": True}
        assert ev_other.is_set(), "不带 stream_id 应停止该会话全部进行中流"
    finally:
        cancels.pop((999001, "any-stream"), None)
