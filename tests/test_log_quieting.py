"""日志降噪 + 结构化上下文日志（2026-09-17）

**两部分，对应两个诉求**：

1. **降噪**（用户："日志越来越臃肿而且无效信息很多"）——
   实测 `data/logs/app.log` 9695 行里 `uvicorn.access` 占 82.4%，其中 **`/healthz` 一项 4560 行（47%）**
   （前端每 5-6 秒轮询）。本文件锁住"静音的边界"：
   **只丢「GET + 2xx + 白名单路径」** —— 非 2xx 保留、写操作保留、非白名单 GET 保留。
   这条边界很重要：降噪一旦越界，就会把"失败痕迹"一起删掉（那比噪音更危险）。

2. **结构化上下文日志**（用户："先做 1"）——
   实测单轮 prompt 达 24883 tokens，但能从 trace 反推的只有约 8600，缺口 1.6 万无法归因
   （`events_json` 里的 tool_result 是截断摘要）。新增按角色拆分的构成日志，把"prompt 为何这么大"
   变成可 grep 的事实。
"""

from __future__ import annotations

import asyncio

import pytest

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, Role
from lightcore.tools.spec import ToolExecutor, ToolRegistry
from papermate.api.log_quieting import AccessNoiseFilter, _is_noisy_access, apply_log_quieting


def _access_args(method: str, path: str, status: int):
    """uvicorn.access 的 record.args 形态：(client, method, path, http_version, status)。"""
    return ("127.0.0.1:1234", method, path, "HTTP/1.1", status)


# --------------------------------------------------------------- 降噪：静音边界


@pytest.mark.parametrize("path", [
    "/api/v1/healthz",
    "/healthz",
    "/api/v1/health",
])
def test_health_polling_is_silenced(path: str) -> None:
    """健康检查（前端定时轮询）成功时零信息量 → 静音。实测它一项占全日志 47%。"""
    assert _is_noisy_access(_access_args("GET", path, 200)) is True


@pytest.mark.parametrize("path", [
    "/api/v1/documents/14/file",
    "/api/v1/documents/14/stream",
    "/api/v1/documents/14/pages/3/image",
])
def test_document_preview_static_gets_are_silenced(path: str) -> None:
    """文档预览类静态 GET（一次翻页十几行）成功时无诊断价值 → 静音。"""
    assert _is_noisy_access(_access_args("GET", path, 200)) is True


@pytest.mark.parametrize("method,path,status,why", [
    ("GET", "/api/v1/healthz", 500, "健康检查失败必须留痕"),
    ("GET", "/api/v1/healthz", 404, "4xx 同样要留痕"),
    ("GET", "/api/v1/documents/14/file", 500, "静态资源失败要留痕（文件损坏/缺失）"),
    ("GET", "/api/v1/documents", 200, "文档列表是业务接口，不静音"),
    ("GET", "/api/v1/sessions/18", 200, "会话详情不静音"),
    ("POST", "/api/v1/documents/14/file", 200, "写操作一律保留"),
    ("PATCH", "/api/v1/sessions/18", 200, "写操作一律保留"),
    ("DELETE", "/api/v1/documents/14", 200, "写操作一律保留"),
])
def test_noise_filter_never_swallows_meaningful_records(method, path, status, why) -> None:
    """**核心边界**：降噪不能删掉任何"可能有诊断价值"的行。"""
    assert _is_noisy_access(_access_args(method, path, status)) is False, why


def test_malformed_args_are_kept() -> None:
    """形态不符合预期（第三方手写日志、args 非元组）→ 一律保留，不猜着丢。"""
    assert _is_noisy_access("不是元组") is False
    assert _is_noisy_access(("only", "two")) is False
    assert _is_noisy_access(("c", "GET", "/api/v1/healthz", "HTTP/1.1", "不是数字")) is False


def test_filter_is_idempotent_and_attaches_to_uvicorn_access() -> None:
    """`apply_log_quieting` 幂等，且 filter 真的挂到了 uvicorn.access 上。"""
    import logging

    apply_log_quieting()
    apply_log_quieting()  # 幂等：重复调用不应叠加多个 filter

    access = logging.getLogger("uvicorn.access")
    assert sum(1 for f in access.filters if isinstance(f, AccessNoiseFilter)) == 1
    # 库 logger 降级到 WARNING（流水账不再入日志）
    assert logging.getLogger("httpx").level >= logging.WARNING


def test_filter_drops_healthz_record_end_to_end() -> None:
    """端到端：真正的 LogRecord 经 filter 后被丢弃/保留（不只测纯函数）。"""
    import logging

    f = AccessNoiseFilter()
    quiet = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                              '%s - "%s %s HTTP/%s" %d', _access_args("GET", "/api/v1/healthz", 200), None)
    noisy = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                              '%s - "%s %s HTTP/%s" %d', _access_args("POST", "/api/v1/sessions", 200), None)

    assert f.filter(quiet) is False, "健康检查成功记录应被丢弃"
    assert f.filter(noisy) is True, "写操作必须保留"


# --------------------------------------------------- 结构化上下文：构成日志


class _BigCtxLLM(BaseLLM):
    """返回固定 prompt_tokens 的桩，用于触发/不触发上下文构成日志。"""

    def __init__(self, prompt_tokens: int) -> None:
        self._pt = prompt_tokens
        self.model_name = "stub"

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="答复"),
            prompt_tokens=self._pt, completion_tokens=10, model="stub", latency_ms=1, cost=0.0,
        )


def _messages_with_big_history(n: int) -> list[ChatMessage]:
    big = "这是一段用于撑大上下文的测试文本。" * 200  # 约 3000 字 → ≈2000 tokens
    msgs = [ChatMessage(role=Role.SYSTEM, content="系统提示")]
    msgs += [ChatMessage(role=Role.USER, content=big)] * n
    return msgs


def _loop(llm, **cfg) -> AgentLoop:
    reg = ToolRegistry()
    return AgentLoop(llm, ToolExecutor(reg), config=LoopConfig(**cfg))


def test_context_composition_logged_when_large(caplog) -> None:
    """上下文估算超过阈值 → 打出按**来源**拆分的构成。

    ⚠️ **2026-10-03 口径变更**（本条随之更新，属行为变更而**非**回归）：
    原先按**角色**拆（`system=` / `user=` / `assistant=` / `tool=`），那样有两个问题：
    ① `SYSTEM_PROMPT` 与「长期记忆注记」都是 `role=system` ⇒ **混算**，分不清提示词 vs 记忆；
    ② ★ **工具定义完全不在统计里**（它在 API 的 `tools` 参数里、不在 messages 里）
       ⇒ 用户会看到"工具占 0%"，而它是**每轮固定成本**。
    ⇒ 改为按**来源**拆（`system`/`notes`/`messages`/`tool_results`/`tool_defs`），
    与前端「上下文用量」面板一一对应（详见 `tests/test_ctx_breakdown.py`）。
    """
    loop = _loop(_BigCtxLLM(prompt_tokens=100))
    with caplog.at_level("INFO", logger="lightcore.core.loop"):
        loop._log_context_composition(_messages_with_big_history(6))

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "上下文构成" in text
    # ★ 新口径：对话消息合并为一项 `messages`（不再有 `user=` / `assistant=`）
    assert "messages=" in text, f"必须按来源拆分：{text}"
    # 本用例的 loop **没有** system_prompt ⇒ 那条 SYSTEM 消息归 `notes`（注记）而非 `system`
    assert "notes=" in text, f"未识别到注记分项：{text}"


def test_context_composition_silent_for_small_context(caplog) -> None:
    """正常小轮次**不打**日志 —— 否则每天几千条，等于又制造一批噪音。"""
    loop = _loop(_BigCtxLLM(prompt_tokens=100))
    with caplog.at_level("INFO", logger="lightcore.core.loop"):
        loop._log_context_composition([ChatMessage(role=Role.SYSTEM, content="短提示")])

    assert "上下文构成" not in "\n".join(r.getMessage() for r in caplog.records)


def test_estimate_vs_actual_gap_is_logged(caplog) -> None:
    """实测 prompt 远大于本地估算时补一条对照 —— 用来验证"估算口径是否系统性偏低"。

    这正是本轮诊断的核心怀疑：`_rough_tokens` 估算与 API 真实 prompt 差约 10 倍，
    而"历史压缩阈值"和"上下文熔断"都用的是这个估算/同源口径。
    """
    loop = _loop(_BigCtxLLM(prompt_tokens=99999))
    loop._last_ctx_estimate = 1000  # 伪造一个偏低的估算
    with caplog.at_level("INFO", logger="lightcore.core.loop"):
        asyncio.run(_drain(loop))

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "实测 prompt" in text and "倍" in text, f"偏差显著时必须留痕：{text}"


async def _drain(loop: AgentLoop) -> None:
    async for _ in loop.run_stream("问题"):
        pass


def test_context_logged_for_medium_context_that_old_threshold_missed(caplog) -> None:
    """估算落在 3000~8000 的轮次**必须**打出构成日志。

    这是 2026-09-17 把 `_CTX_LOG_THRESHOLD` 从 8000 降到 3000 的直接原因：
    `_rough_tokens` 实测偏低约 2 倍（1.88~1.99，见其 docstring），
    8000 的**估算**门槛等效于约 16000 真实 token —— 真实 1.5 万以下的轮次全都打不出来
    （首次上线时 4 步实测 9292~15015，却一条日志都没有）。本用例在旧阈值下会红。
    """
    loop = _loop(_BigCtxLLM(prompt_tokens=100))
    with caplog.at_level("INFO", logger="lightcore.core.loop"):
        loop._log_context_composition(_messages_with_big_history(3))

    text = "\n".join(r.getMessage() for r in caplog.records)
    est = loop._last_ctx_estimate
    assert 3000 <= est < 8000, f"用例构造的估算应落在 3000~8000 区间，实际 {est}"
    assert "上下文构成" in text, f"估算 {est} 对应真实规模上万，必须打出：{text}"


def test_ctx_log_threshold_stays_aligned_with_estimate_bias() -> None:
    """阈值不得被单独上调 —— 它与 `_rough_tokens` 的偏低倍数是一对。

    偏低约 2 倍 → 阈值必须相应取小约 2 倍，否则门控再次形同虚设。
    要改这里，必须先修 `_rough_tokens` 系数（那是行为变更，需单独决策）。
    """
    assert AgentLoop._CTX_LOG_THRESHOLD == 3000, (
        "阈值与 _rough_tokens 的偏差是一对：估算偏低 2 倍 → 阈值取小 2 倍。"
        "单独上调会让门控形同虚设（除非先修估算系数）"
    )
