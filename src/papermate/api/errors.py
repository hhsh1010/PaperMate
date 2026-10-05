"""PaperMate · 统一错误出口（2026-09-15，P0-2）

背景（`docs/后端工程体系评估.md`）：
此前**没有全局异常处理器**（`exception_handler` 全库 0 处），错误处理分散在
12 个路由文件的 **126 处 `HTTPException`**；未捕获异常走 FastAPI/Starlette 默认路径 →
返回**非 JSON 的 500 纯文本**，既无错误码也无 trace_id → **用户报的错与后端日志无法关联**。

本模块统一三类出口，**并保持向后兼容**：

- 既有契约是 `{"detail": <字符串>}`（前端 `client.ts` 读它、11 个测试断言其文案）——
  因此 `detail` **恒为字符串**，新增字段只做增强，不改变既有语义；
- **状态码一律保持不变**（多处测试断言 400/404/409/422）。

三类出口：

| 异常 | 状态码 | 处理要点 |
|---|---|---|
| `RequestValidationError` | **422**（不变） | 默认 `detail` 是**对象数组** → 前端按字符串处理会显示 `[object Object]`；这里压平成可读字符串，结构化明细移到 `errors` |
| `StarletteHTTPException` | 原状态码（不变） | 保留原 `detail` 文案（126 处调用点的语义不动） |
| `Exception`（未捕获） | **500** | 转为 JSON + 全栈入日志 + `trace_id` 回传，避免"用户看到纯文本 500、日志里找不到对应请求" |

响应结构：

```json
{
  "detail": "人类可读消息（字符串，向后兼容）",
  "code": "validation_error",
  "trace_id": "9f2c1ab34de5",
  "errors": [ ... ]        // 仅 422 时出现：结构化校验明细
}
```
"""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .middleware import current_trace_id

logger = logging.getLogger(__name__)

#: HTTP 状态码 → 机器可读错误码（前端可据此分支，无需解析中文文案）
_CODE_BY_STATUS: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "too_many_requests",
    500: "internal_error",
    503: "service_unavailable",
}

#: 兼容旧名：早期只有错误出口认这个头（C2 统一后主名见 `middleware.TRACE_HEADER`）
_LEGACY_TRACE_HEADERS = ("X-Request-Id", "X-Trace-Id")


def _trace_id(request: Request) -> str:
    """取本次请求的链路标识。

    **C2（2026-09-21）修正**：此前这里自行生成/读取，与请求级中间件的 id **互不相干** ——
    结果是「错误体里的 trace_id」和「日志里的 trace_id」对不上，用户报错时仍然无法关联日志。
    这恰恰是 C2 要解决的问题，所以改为**优先复用中间件已建立的请求级 id**。
    """
    ctx = current_trace_id()
    if ctx:
        return ctx[:64]
    for name in _LEGACY_TRACE_HEADERS:
        incoming = request.headers.get(name)
        if incoming:
            return incoming[:64]
    return uuid4().hex[:12]


def _flatten_validation_error(exc: RequestValidationError) -> str:
    """把 FastAPI 的校验错误（对象数组）压平成**人类可读的单行字符串**。

    这是 P0-1 的根治点：原先前端拿到数组后 `new Error(数组)` → `message` 变成
    `"[object Object]"`，用户完全不知道哪里错了。
    """
    errors = exc.errors()
    if not errors:
        return "请求参数不合法"
    first = errors[0] or {}
    # loc 形如 ("body", "title") / ("path", "doc_id")；去掉 body 前缀更易读
    parts = [str(x) for x in (first.get("loc") or []) if x != "body"]
    where = ".".join(parts)
    msg = str(first.get("msg") or "参数不合法")
    # FastAPI 的 msg 是英文（如 "Input should be a valid integer"），保留原文更利于排查
    return f"请求参数不合法：{where} — {msg}" if where else f"请求参数不合法：{msg}"


def install_exception_handlers(app: FastAPI) -> None:
    """安装三类全局异常出口（在 `create_app` 中调用一次）。"""

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        trace = _trace_id(request)
        logger.info("参数校验失败 trace=%s path=%s errors=%s", trace, request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({
                "detail": _flatten_validation_error(exc),
                "code": "validation_error",
                "trace_id": trace,
                "errors": exc.errors(),  # 结构化明细保留，供调试与前端增强
            }),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        trace = _trace_id(request)
        # 兼容 Starlette 允许 detail 为非字符串的情形（如自定义 dict）
        detail = exc.detail if isinstance(exc.detail, str) else "请求失败"
        if exc.status_code >= 500:
            logger.warning("HTTP %s trace=%s path=%s detail=%s", exc.status_code, trace, request.url.path, detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": detail,
                "code": _CODE_BY_STATUS.get(exc.status_code, "http_error"),
                "trace_id": trace,
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        """未捕获异常：**JSON 化 + 全栈入日志 + trace_id 回传**。

        此前走 FastAPI 默认 → 纯文本 `Internal Server Error`，日志与用户所见无法对应。
        注意：不把异常文本回传给客户端（可能含路径/凭据等敏感信息），只回 trace_id。
        """
        trace = _trace_id(request)
        logger.exception("未处理异常 trace=%s method=%s path=%s", trace, request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务器内部错误，请稍后重试",
                "code": "internal_error",
                "trace_id": trace,
            },
        )
