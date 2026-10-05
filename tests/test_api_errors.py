"""统一错误出口测试（2026-09-15，P0-1 / P0-2 的回归守护）。

对应 `docs/后端工程体系评估.md` 中两项 P0：

- **P0-1**：422 校验错误的 `detail` 原是**对象数组** → 前端 `new Error(数组)` 得到
  `"[object Object]"`，用户看不到哪里错了。修复后 `detail` **恒为字符串**。
- **P0-2**：**无全局异常处理器** → 未捕获异常返回**非 JSON 的 500 纯文本**，
  且无错误码、无 trace_id。修复后为 JSON + `code` + `trace_id`，并可据 trace_id 关联日志。

本文件的断言同时锁住**向后兼容**：既有 `{"detail": <字符串>}` 契约与状态码不得变化
（多处既有测试依赖它们，见 `test_api.py` / `test_word_support.py` 等对 detail 文案的断言）。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod


@pytest.fixture
def app():
    tmp = tempfile.mkdtemp(prefix="papermate_err_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---------- P0-1：422 校验错误的 detail 必须是字符串 ----------


def test_validation_error_detail_is_readable_string(client):
    """路径参数类型错误 → 422，且 detail 是可读**字符串**（不是对象数组）。"""
    r = client.get("/api/v1/documents/abc/file")  # doc_id 应为 int
    assert r.status_code == 422
    body = r.json()

    detail = body["detail"]
    assert isinstance(detail, str), f"detail 必须是字符串，实际 {type(detail)}"
    assert detail.strip(), "detail 不能为空"
    # 关键回归：绝不能是 "[object Object]"（这正是修复前的用户可见表现）
    assert "[object" not in detail
    # 应当指出出错字段，便于用户定位
    assert "doc_id" in detail


def test_validation_error_has_code_and_structured_errors(client):
    """422 除可读 detail 外，仍保留机器可读 code 与结构化 errors（增强不破坏）。"""
    r = client.get("/api/v1/documents/abc/file")
    body = r.json()
    assert body["code"] == "validation_error"
    assert isinstance(body["trace_id"], str) and body["trace_id"]
    assert isinstance(body["errors"], list) and body["errors"], "结构化明细应保留"
    assert body["errors"][0]["loc"][-1] == "doc_id"


def test_js_coercion_of_old_behaviour_would_have_shown_object_object(client):
    """把「数组 detail」与「字符串 detail」的差别显式记录下来。

    修复前 detail 是 list → 前端 `new Error(list)` 的 message 会变成 `[object Object]`；
    修复后是 str，前端可直接展示。
    """
    detail = client.get("/api/v1/documents/abc/file").json()["detail"]
    assert not isinstance(detail, list)


# ---------- 向后兼容：HTTPException 的原语义不变 ----------


def test_http_exception_keeps_original_detail_and_status(client):
    """既有 `{"detail": <中文文案>}` 契约与状态码保持不变（126 处调用点语义不动）。"""
    r = client.get("/api/v1/sessions/999999")
    assert r.status_code == 404
    body = r.json()
    assert body["detail"] == "会话不存在"  # 原有中文文案逐字不变
    assert body["code"] == "not_found"
    assert body["trace_id"]


def test_not_found_route_also_json_with_code(client):
    r = client.get("/api/v1/__no_such_route__")
    assert r.status_code == 404
    body = r.json()
    assert isinstance(body["detail"], str)
    assert body["code"] == "not_found"


# ---------- P0-2：未捕获异常 → JSON 500 + trace_id ----------


def _boom_app(exc: Exception):
    """构造一个「只装统一错误出口」的最小 app，用于精确验证未捕获异常的表现。

    不用主 app：主 app 在 `with TestClient(...)` 启动时已构建 middleware stack，
    之后再 `@app.get(...)` 动态加路由不会生效（实测返回 404）。这里独立构造更纯粹。
    """
    from fastapi import FastAPI

    from papermate.api.errors import install_exception_handlers

    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    def _boom():  # pragma: no cover - 仅在测试中注册
        raise exc

    return app


def test_unhandled_exception_returns_json_500():
    """未捕获异常必须返回**可解析的 JSON**（此前是纯文本 `Internal Server Error`）。"""
    app = _boom_app(RuntimeError("boom-for-test"))
    # raise_server_exceptions=False：否则 TestClient 会把服务端异常重新抛出，拿不到响应
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")

    assert r.status_code == 500
    body = r.json()  # ← 修复前这一步会直接抛错（响应体是纯文本，无法解析）
    assert body["code"] == "internal_error"
    assert body["trace_id"]
    assert body["detail"] == "服务器内部错误，请稍后重试"


def test_unhandled_exception_does_not_leak_details():
    """异常文本不得回传客户端（可能含路径/凭据等敏感信息）。"""
    app = _boom_app(RuntimeError("SECRET_TOKEN=abc123"))
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")

    assert r.status_code == 500
    assert "SECRET_TOKEN" not in r.text
    assert "abc123" not in r.text


# ---------- trace_id 可跨端关联 ----------


def test_trace_id_reused_from_request_header(client):
    """请求头带 X-Request-Id 时复用（便于前端/代理与后端日志对齐）。"""
    r = client.get("/api/v1/sessions/999999", headers={"X-Request-Id": "trace-from-client-01"})
    assert r.status_code == 404
    assert r.json()["trace_id"] == "trace-from-client-01"


def test_trace_id_generated_when_absent(client):
    r = client.get("/api/v1/sessions/999999")
    trace = r.json()["trace_id"]
    assert isinstance(trace, str) and len(trace) == 12  # uuid4().hex[:12]
