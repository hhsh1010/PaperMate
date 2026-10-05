"""N6 回归（2026-09-11）：SSE 中断哨兵必须是**同一个对象**。

实测 bug（日志 2026-09-11 00:58:13）：`_watch_disconnect` 投递模块级 `_STREAM_STOP`，
而 agent 事件循环检查的是函数内新建的局部 `_STOP = object()`（另一个对象），`is` 永不成立 →
① 哨兵落进 `ev = item` → `ev.type` → `AttributeError: 'object' object has no attribute 'type'`（ASGI 500）；
② 中断语义失效——`loop_cancelled` 永不置位，点「停止」/切会话时后端仍跑到自然结束
   （已生成内容的落盘等中断处理从不执行）。
"""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod


@pytest.fixture
def client():
    import os
    import tempfile

    tmp = tempfile.mkdtemp(prefix="pm_n6_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_stop_sentinel_identity_is_consistent():
    """结构回归：agent 路径必须复用模块级 `_STREAM_STOP`，不得另建局部哨兵。

    用 AST 判定赋值语句（而非字符串包含——注释里也会出现 `_STOP = object()` 字样）。
    """
    import ast as _ast

    from papermate.api import routes_search as rs

    src = inspect.getsource(rs)
    tree = _ast.parse(src)
    assigned = {
        t.id
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Assign)
        for t in node.targets
        if isinstance(t, _ast.Name)
    }
    assert "_STOP" not in assigned, \
        "不得再新建局部中断哨兵——必须与 _watch_disconnect 投递的 _STREAM_STOP 同源"
    assert "_STREAM_STOP" in assigned, "模块级哨兵 _STREAM_STOP 必须存在"

    assert "if item is _STREAM_STOP:" in src, \
        "agent 消费循环必须用模块级 _STREAM_STOP 判等（否则 is 永不成立）"
    assert "q.put_nowait(_STREAM_STOP)" in src, \
        "watcher 投递的哨兵必须是同一个 _STREAM_STOP"


def test_interrupt_then_service_still_healthy(client):
    """行为回归：客户端中断流后，服务仍能正常处理后续请求（不因哨兵崩溃）。"""
    sid = client.post("/api/v1/sessions", json={"title": "interrupt"}).json()["id"]

    # 建立 SSE 后立刻断开（读到首个 chunk 就走）——模拟用户点停止 / 切会话
    try:
        with client.stream(
            "POST", f"/api/v1/sessions/{sid}/messages/stream",
            json={"content": "请介绍一下这篇论文"},
        ) as r:
            for _line in r.iter_lines():
                break  # 立即断开连接
    except Exception:  # noqa: BLE001  断开路径允许抛错，重点是服务不崩
        pass

    # 服务应仍然健康（旧实现在此路径会抛 AttributeError）
    assert client.get("/api/v1/healthz").status_code == 200
    assert client.get("/api/v1/sessions").status_code == 200
