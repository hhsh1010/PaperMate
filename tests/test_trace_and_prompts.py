"""C2 请求级 trace_id 中间件 + C7 提示词集中化（2026-09-21）。

两条都不是"功能"，是**防静默不一致**的基础设施，所以测试守护的是**不变量**：
- C2：每个响应都带 X-Trace-Id；上游传来的要**透传**（否则链路断成两截）
- C7：全库**不得**再有第二处 `SYSTEM_PROMPT` 定义 —— 这是防止"改一处漏一处"复发的闸门
"""

from __future__ import annotations

import pathlib
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from papermate.api.middleware import TraceIdMiddleware, current_trace_id
from papermate.prompts import PROMPT_VERSION, SYSTEM_PROMPT

SRC_ROOT = pathlib.Path(__file__).resolve().parent.parent / "src"


@pytest.fixture()
def client():
    app = FastAPI()
    app.add_middleware(TraceIdMiddleware)

    @app.get("/ping")
    def ping() -> dict:
        # 请求内应能取到同一个 trace_id（日志关联靠它）
        return {"ok": True, "trace_id": current_trace_id()}

    @app.get("/boom")
    def boom() -> dict:
        raise RuntimeError("x")

    return TestClient(app)


# ------------------------------------------------------------------ C2


def test_every_response_carries_trace_id(client):
    """成功的请求也必须带 trace_id —— 此前只有**出错时**才有。"""
    r = client.get("/ping")
    assert r.status_code == 200
    tid = r.headers.get("x-trace-id")
    assert tid, "响应缺少 X-Trace-Id"
    assert r.json()["trace_id"] == tid, "中间件与处理函数里取到的不是同一个 id"


def test_incoming_trace_id_is_propagated(client):
    """上游传来的 id 必须透传，否则链路被切成两截、无法跨服务对齐。"""
    r = client.get("/ping", headers={"X-Trace-Id": "upstream-123"})
    assert r.headers["x-trace-id"] == "upstream-123"
    assert r.json()["trace_id"] == "upstream-123"


def test_distinct_requests_get_distinct_ids(client):
    a = client.get("/ping").headers["x-trace-id"]
    b = client.get("/ping").headers["x-trace-id"]
    assert a != b


def test_outside_request_context_is_empty_string():
    """无请求上下文时返回空串而非抛错 —— 取用方不必到处 try/except。"""
    assert current_trace_id() == ""


def test_is_pure_asgi_not_base_http_middleware():
    """★ 必须是纯 ASGI：BaseHTTPMiddleware 会缓冲响应体，对 SSE 有已知风险。"""
    src = (SRC_ROOT / "papermate" / "api" / "middleware.py").read_text(encoding="utf-8")
    # 只禁「导入/继承」，不禁注释里出现该名字（文档里要解释为什么不用它）
    assert "import BaseHTTPMiddleware" not in src, "不得继承 BaseHTTPMiddleware（SSE 风险）"
    assert "BaseHTTPMiddleware)" not in src
    assert "async def __call__(self, scope, receive, send)" in src


def test_trace_survives_error_response():
    """出错时也要有 trace_id（否则错误响应与日志仍无法关联）。

    ⚠️ 必须装 `install_exception_handlers`：裸 FastAPI 的未捕获异常会一路抛到
    **最外层** `ServerErrorMiddleware` 生成 500 —— 那已经越过用户中间件，header 无法注入。
    真实 app 装了统一错误出口（异常在 ExceptionMiddleware 层被处理），响应会正常流经
    本中间件，这正是本用例要锁的行为。
    """
    from papermate.api.errors import install_exception_handlers

    app = FastAPI()
    app.add_middleware(TraceIdMiddleware)
    install_exception_handlers(app)

    @app.get("/boom")
    def boom() -> dict:
        raise RuntimeError("x")

    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/boom", headers={"X-Trace-Id": "up-777"})
    assert r.status_code >= 500
    # ★ 关键：错误体里的 trace_id 必须与请求级 id **同源**。
    #   此前 errors.py 自行生成，两者互不相干 → 用户报的错仍然无法关联日志（C2 的原意）。
    assert r.json().get("trace_id") == "up-777"


def test_bare_app_500_has_no_header_documented_gap():
    """记录已知边界：**无**统一错误出口时，未捕获异常的 500 不带 `X-Trace-Id` 响应头。

    原因不是我们漏了 —— Starlette 的 `ServerErrorMiddleware` **永远在最外层**，
    未捕获异常由它生成响应，用户中间件拿不到 `send`。真实 app 装了统一错误出口
    （`install_exception_handlers`），异常在 ExceptionMiddleware 层被处理，响应才流经本中间件。
    用测试固化这个事实，避免将来误把"没有 header"当成回归去乱改。
    """
    app = FastAPI()
    app.add_middleware(TraceIdMiddleware)

    @app.get("/boom")
    def boom() -> dict:
        raise RuntimeError("x")

    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/boom")
    assert r.status_code >= 500
    assert r.headers.get("x-trace-id") is None, "Starlette 中间件顺序若变了，需重新评估本结论"


# ------------------------------------------------------------------ C7


def test_single_source_of_system_prompt():
    """★ 防复发：全库只能有一处 `SYSTEM_PROMPT` 定义。"""
    defs = []
    for p in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
            if re.match(r"^SYSTEM_PROMPT\s*=", line):
                defs.append(f"{p.relative_to(SRC_ROOT)}:{i}")
    assert len(defs) == 1, f"SYSTEM_PROMPT 定义了 {len(defs)} 处（应只有 1 处）: {defs}"


def test_cli_and_chat_service_use_the_same_prompt():
    """★ CLI 与线上必须用同一份 —— 此前两份内容不同（简版缺全部硬性规则）。"""
    import papermate.cli as cli
    from papermate.services import chat_service

    assert cli.SYSTEM_PROMPT is SYSTEM_PROMPT
    assert chat_service.SYSTEM_PROMPT is SYSTEM_PROMPT


def test_merged_prompt_keeps_hard_rules():
    """合并基准是**线上长版**：防臆测、意图分类、回复格式、输出过滤都必须在。"""
    for marker in ("【硬性规则", "【意图分类", "【回复格式规范", "【输出过滤"):
        assert marker in SYSTEM_PROMPT, f"合并后丢失了 {marker}"


def test_prompt_version_exists():
    """版本号让"某次回答用的是哪版策略"可回溯。"""
    assert PROMPT_VERSION
    assert isinstance(PROMPT_VERSION, str)
