"""请求级 trace_id 中间件（C2，2026-09-21）。

**为什么必须是纯 ASGI 中间件**：
`BaseHTTPMiddleware` 会把响应体先缓冲再转发，对 **SSE 流式端点**有已知问题
（响应延迟/断流）。本项目 `/chat`、`/documents/*/progress` 都是流式接口，
所以这里**手写 ASGI 中间件**，直接透传 `send`，只在 `http.response.start` 阶段加响应头。

**做什么**：
1. 请求头带 `X-Trace-Id` 则**透传**（前端/上游已生成的链路 id），否则生成一个；
2. 存入 `ContextVar`，同一请求内的日志/代码可用 `current_trace_id()` 取到；
3. 响应头回写 `X-Trace-Id`，让"请求 ↔ 响应 ↔ 日志"三者可对齐。

⚠️ 这与 `api/errors.py` 里错误响应的 `trace_id` 是**两回事**：那个是**出错时才现造**的，
只在错误体里可见；这个是**每个请求都有**的，成功请求也能关联日志（清单 C2 要的正是后者）。
"""

from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

#: 当前请求的 trace_id（无请求上下文时为空串，取用方不必处理 None）
_trace_id_var: ContextVar[str] = ContextVar("papermate_trace_id", default="")

TRACE_HEADER = "x-trace-id"

#: 兼容旧名：错误出口早期只认 `X-Request-Id`（`errors.py` 的历史约定）。
#: ⚠️ 必须继续接受它 —— 否则外部已按旧名传链路 id 的调用方会**静默丢失关联**
#: （实测既有用例 `test_api_errors.py::test_trace_id_reused_from_request_header` 就是这种场景）。
TRACE_HEADERS_ACCEPTED = (TRACE_HEADER, "x-request-id")


def current_trace_id() -> str:
    """取当前请求的 trace_id；不在请求上下文内则返回空串。"""
    return _trace_id_var.get()


class TraceIdMiddleware:
    """纯 ASGI 中间件（不用 BaseHTTPMiddleware —— 对 SSE 有已知风险）。"""

    def __init__(self, app) -> None:  # noqa: ANN001  ASGI app 类型不统一
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            # websocket / lifespan 直接放行
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers") or []
        incoming = ""
        for k, v in headers:
            if k.decode("latin-1").lower() in TRACE_HEADERS_ACCEPTED:
                incoming = v.decode("latin-1").strip()
                break
        tid = incoming or uuid4().hex[:12]

        token = _trace_id_var.set(tid)

        async def send_with_trace(message) -> None:  # noqa: ANN001
            # 只在响应头阶段注入：SSE 的 body 分片不受影响（不缓冲、不改写）
            if message["type"] == "http.response.start":
                raw = message.setdefault("headers", [])
                raw.append((TRACE_HEADER.encode("latin-1"), tid.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        finally:
            _trace_id_var.reset(token)
