"""日志降噪（2026-09-17）：摘掉「成功且幂等的高频请求」，把信噪比还给真正有用的行。

**实测依据**（`data/logs/app.log`，9695 行 / 969 KB）：

| 来源 | 行数 | 占比 |
|---|---|---|
| `uvicorn.access` | 7991 | 82.4% |
| ↳ 其中 **`GET /api/v1/healthz`** | **4560** | **47%**（前端每 5-6 秒轮询一次） |
| ↳ 其中文档预览静态 GET（`/file`、`/pages/*/image`、`/stream`） | 952 | 10% |
| `httpx` INFO（每个 HTTP 请求一行） | 479 | 4.9% |
| `faiss.loader` + `FlagEmbedding.*` INFO | 130 | 1.3% |

这些行**在成功时没有诊断价值**，却把 ERROR/WARNING 与业务 INFO 稀释在约 20 倍噪音里 ——
而日志是本地单用户场景下**唯一的排查入口**（`docs/` 里多处写明"排查第一步读 app.log"），
噪音直接抬高每次排查的成本。

**原则：只静音「成功且幂等」的请求**
- 非 2xx **一律保留**（失败必须有痕迹，否则等于把问题藏起来）；
- 写操作（POST/PATCH/DELETE）**一律保留**（有副作用，必须可追溯）；
- 只丢弃「GET + 2xx + 白名单路径」这一类。

> 🔑 与项目既有教义一致：**不能因为"吵"就把可能有用的信息一起删掉** ——
> 降噪的边界必须是"可证明无诊断价值"，而不是"看着像噪音"。
"""

from __future__ import annotations

import logging

#: 直接静音的路径（健康检查：前端定时轮询，成功时零信息量）
_QUIET_EXACT = frozenset({"/api/v1/healthz", "/healthz", "/api/v1/health"})

#: 文档预览类静态资源（GET + 2xx 时静音）：/documents/{id}/file、/documents/{id}/stream、
#: /documents/{id}/pages/{n}/image —— 一次翻页就是十几行，成功时无诊断价值。
_QUIET_DOC_SUFFIX = ("/file", "/stream")
_QUIET_DOC_CONTAINS = ("/pages/",)

#: 降级为 WARNING 的库 logger：它们的 INFO 是加载/连接级别的流水账，
#: 失败时通常自带 WARNING/ERROR（保留下层告警，只关掉流水账）。
_QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "faiss",
    "faiss.loader",
    "FlagEmbedding",
    "jieba",
)


def _is_noisy_access(args: object) -> bool:
    """判断一条 uvicorn.access 记录是否属于「成功且幂等的高频请求」。

    取 `record.args`（uvicorn 传的是 `(client, method, path, http_version, status)` 元组）
    而不是解析 message 文本 —— 前者是结构化数据，不受日志格式改动影响。
    """
    if not isinstance(args, tuple) or len(args) < 5:
        return False
    _client, method, path, _version, status = args[:5]
    try:
        code = int(status)
    except (TypeError, ValueError):
        return False
    if code >= 400:
        return False  # 失败一律保留
    if str(method).upper() != "GET":
        return False  # 写操作一律保留
    raw = str(path)
    if raw in _QUIET_EXACT:
        return True
    if raw.startswith("/api/v1/documents/"):
        return raw.endswith(_QUIET_DOC_SUFFIX) or any(k in raw for k in _QUIET_DOC_CONTAINS)
    return False


class AccessNoiseFilter(logging.Filter):
    """丢弃「成功且幂等的高频请求」的 access 记录（挂在 `uvicorn.access` 上）。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003  logging.Filter 规定的签名
        return not _is_noisy_access(record.args)


_applied = False


def apply_log_quieting() -> None:
    """挂载降噪（幂等，可重复调用）。

    调用时机：应用 **lifespan 启动时** —— 必须晚于 uvicorn 的 `dictConfig`，
    否则 access logger 可能在之后被重新配置（filter 挂 logger 对象上，dictConfig 不会移除，
    但仍以"最后挂载"为准更稳妥）。
    """
    global _applied
    if _applied:
        return
    _applied = True

    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, AccessNoiseFilter) for f in access.filters):
        access.addFilter(AccessNoiseFilter())

    for name in _QUIET_LOGGERS:
        lg = logging.getLogger(name)
        if lg.level == logging.NOTSET or lg.level < logging.WARNING:
            lg.setLevel(logging.WARNING)
