"""LightCore · ModelRouter + OpenAI 兼容后端（docs/05 ModelRouter / docs/10 安全）

- 路由：按 name 选择后端。支持 OpenAI 官方、DeepSeek、Qwen/通义、以及任意 OpenAI 兼容端点。
- 安全：API key 一律从环境变量读取，**绝不入库**（NFR: 密钥不入库）。config 用 pydantic-settings 从 .env 加载。
- 成本：DeepSeek 为主用（docs/10 成本模型），本地嵌入/重排零成本。

注意：OAICompatibleLLM 在导入 httpx 调用真实端点，仅在有 key 且联网时可用。
FakeLLM 用于本地/Spike/测试，无需 key。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .base import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    Role,
    ToolCall,
    ToolSpecForLLM,
)
from .catalog import (
    BUILTIN_MODELS,
    LEGACY_MODEL_CONTEXT_WINDOW,
    LEGACY_MODEL_PRICING,
    LEGACY_MODEL_TO_PROVIDER,
    LEGACY_NATIVE_WEB_SEARCH,
    MAX_PRICE_IN,
    MAX_PRICE_OUT,
    enabled_models_by_provider,
)

logger = logging.getLogger("lightcore.llm")

# DeepSeek V4 系列模型（deepseek-v4-flash / deepseek-v4-pro / deepseek-v4-flash-vision-exp）：
# 官方文档（api-docs.deepseek.com/guides/thinking_mode + Oh My Pi 集成指南 + GitHub issue #1376）：
# - thinking 模式**默认开启**；
# - thinking 模式下**不接受 tool_choice**（"required" / 指定函数名 → HTTP 400 "Thinking mode does
#   not support this tool_choice"），且带工具调用的轮次要求历史回传 reasoning_content（否则 400）；
# - 显式传 {"thinking": {"type": "disabled"}} 可关闭思考，行为回到 deepseek-chat(V3.2) 一致。
# 本项目 Agent 循环依赖 tool_choice（绑文档会话强制 'required'）与标准 content 流式，
# 因此对 V4 系列一律显式禁用 thinking。
_V4_MODEL_PREFIX = "deepseek-v4"

# qwen3 系列思考模式：payload 用 enable_thinking（与 DeepSeek V4 的 thinking.type 格式不同，2026-08-25 接入）
# ★ 2026-10-05：改为**从 catalog 派生**（`thinking_style == "qwen3"`）。
#   旧实现是手写枚举 —— 新增 qwen3 系模型时极易漏加，后果是 thinking 参数不发送，
#   行为与预期不符**且不报错**（属"漏改一张表就静默降级"的典型）。
_QWEN3_THINKING_MODELS = {m.model_id for m in BUILTIN_MODELS if m.thinking_style == "qwen3"}

# doc-bound 会话「绑文档不可靠」模型名单（2026-09-03 实测确认，2026-09-16 起语义变更）：
# qwen3-vl-flash 在大 SYSTEM+历史上下文下，连 tool_choice='required'（流式+非流式补偿）都常不调用
# retrieve（最小上下文却会），属上下文依赖型工具服从缺口，无法纯代码根治。
#
# ⚠️ 2026-09-16 策略变更（用户决策）：**取消静默回落**。此前会把请求偷偷换成
# deepseek-v4-flash-vision-exp（DOC_BOUND_FALLBACK_MODEL，已删除），实测危害：
# 用户选了 Qwen、请求却发给 DeepSeek，报"DeepSeek 余额不足"——完全无法自救。
# 现改为：papermate 侧（chat_service.doc_bound_model_risk）给出**警示但放行**
# （2026-09-16 晚定：用户有权使用自费模型，决策权归用户，但风险必须可见）。
# 本名单语义从「回落触发条件」变为「风险提示名单」。
# ★ 2026-10-05：改为从 catalog 派生（`doc_bound_unreliable=True`），取值与改动前一致。
DOC_BOUND_UNRELIABLE_MODELS = {m.model_id for m in BUILTIN_MODELS if m.doc_bound_unreliable}

# 原生联网（DashScope enable_search=True）**实测支持**的模型清单（2026-09-17 定档，单一来源）。
#
# 判据（方法论，别用错尺子）：必须问「今天几号」这类**只有联网才知道**的硬事实 ——
# 不能拿模型本来就会的知识题当判据，否则分不清「联网查到的」与「训练时就会的」。
#
#   ✅ 实测支持：qwen3-max · qwen-max · qwen3.6-flash · qwen3.7-flash · qwen-turbo
#   ❌ 实测不支持：qwen-plus · qwen3-vl-flash · DeepSeek 全系列
#
# ⚠️ 2026-09-17 修复（用户报「联网提示闪烁且显示不正确」时揪出）：
# 此前 `supports_native_web_search()` **只看厂商**（`endpoint.name == "qwen"`），
# 于是同厂内不支持的 `qwen3-vl-flash` 也被判为「支持」→
# `GET /sessions/{id}/network-capability` 返回 `supports_native=true` →
# 前端「开关已开但能力不可用」的预警**永远不会触发**（用户以为联网生效，实际不搜）。
# **能力判定必须细到模型**：厂商级粒度会抹平同厂内的模型差异（vision 版尤其容易漏）。
# ★ 2026-10-05：改为从 catalog 派生（`native_web_search=True`）**并 union 退役模型的实测结论** ——
#   `qwen-max` / `qwen-turbo` 两个虽已退役，但 2026-09-17 实测支持联网；
#   若只取派生集合会**漏掉它们**（旧会话仍可能选中）⇒ 属行为回退，故显式并回。
NATIVE_WEB_SEARCH_MODELS: frozenset[str] = frozenset(
    {m.model_id for m in BUILTIN_MODELS if m.native_web_search}
) | LEGACY_NATIVE_WEB_SEARCH

# 常见可诊断 HTTP 错误 → 可操作的中文提示（用户侧可见，避免只透传英文 JSON）
_FRIENDLY_HTTP_ERRORS: dict[int, str] = {
    401: "API Key 无效或未授权：请检查该厂商的 API Key 是否正确（设置页「测试连接」可验证）。",
    402: "API 账户余额不足（Insufficient Balance）：请前往对应平台充值后重试"
         "（DeepSeek: platform.deepseek.com/top_up；Qwen: bailian.console.aliyun.com）。",
    403: "API 无权限访问该模型：请检查模型是否对当前 Key 开放。",
    404: "端点或模型不存在：请检查 base_url 与模型标识符是否正确。",
}


def _friendly_error(status: int, body_snippet: str) -> str:
    """把可诊断的 HTTP 状态码翻译为可操作提示；其余保留原始响应体。"""
    hint = _FRIENDLY_HTTP_ERRORS.get(status)
    if hint:
        return hint
    return f"模型服务返回错误（HTTP {status}）"

# 定价表：¥ / 1k tokens（近似值，随模型更新；仅用于本地成本估算 / 熔断，NFR-09/10）
# ★ 2026-10-05：改为从 catalog 派生（`price_in/price_out`）+ 保留退役模型的历史定价。
#   定价与窗口是 `ModelSpec` 的**必填参数** ⇒ 新增模型漏填会在 import 期报错，
#   而不是像旧实现那样"忘了填 ⇒ 成本恒 0 ⇒ ¥1 熔断静默失效"。
#   （取值与改动前**逐项一致**：6 条来自 catalog，3 条来自 LEGACY_MODEL_PRICING。）
_PRICING: dict[str, dict[str, float]] = {
    m.model_id: {"in": m.price_in, "out": m.price_out} for m in BUILTIN_MODELS
}
_PRICING.update(LEGACY_MODEL_PRICING)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """按模型定价估算成本（¥）。

    ★ 2026-10-05（随"模型目录化"一并修正）：未命中定价时**按目录内最贵档保守估算**，
    **不再 `return 0.0`**。

    为什么必须改：`0.0` 会让成本恒为 0 ⇒ **¥1/轮 熔断永不触发** ——
    一旦出现未登记定价的模型（新增模型漏改、或配置里模型名写错），
    护栏会在**不报任何错**的情况下失效。这是"静默降级"在成本侧的体现。

    现在的取舍：**宁可高估触发熔断**（用户看得见、可调 `cost_limit_per_turn`），
    也不要静默失去护栏；同时打 WARNING 指出该模型需登记进 `lightcore/llm/catalog.py`。
    """
    p = _PRICING.get(model)
    if not p:
        logger.warning(
            "模型 %r 不在定价表中 → 成本按**最贵档**保守估算（in=%.4f / out=%.4f 每 1k），"
            "以免熔断护栏静默失效；请把它登记进 lightcore/llm/catalog.py",
            model, MAX_PRICE_IN, MAX_PRICE_OUT,
        )
        return (
            prompt_tokens / 1000 * MAX_PRICE_IN + completion_tokens / 1000 * MAX_PRICE_OUT
        )
    return prompt_tokens / 1000 * p["in"] + completion_tokens / 1000 * p["out"]


# ---- 配置（pin 在环境变量 /.env，不入库） ----
@dataclass
class ModelEndpoint:
    name: str
    base_url: str
    api_key_env: str
    default_model: str


# 预置端点（key 名走环境变量）
PRESET_ENDPOINTS: dict[str, ModelEndpoint] = {
    # deepseek 默认模型 = DeepSeek-V4-Flash-Vision（官方 id：deepseek-v4-flash-vision-exp，实验性多模态）。
    # 仅该模型接受图片输入（图片仅限 user 消息，JPEG/PNG/GIF/WebP，≤384 tokens/张，计费与 V4-Flash 一致）。
    "deepseek": ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-flash-vision-exp"),
    # 2026-08-25：openai 厂商整体下线（gpt 系列移除）；qwen 默认模型切 qwen3.7-flash
    "qwen": ModelEndpoint("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen3.7-flash"),
}


# 各厂商可选模型（前端「设置 key 后选择模型」用；列表首位为该厂商默认模型）
# 2026-08-22 清理冗余：移除 openai 的具体日期版本号（gpt-4o-mini-2024-07-18）与
# 已被 qwen3-max 取代的旧款 qwen-max，避免下拉出现无用选项；兼容映射见 _HIDDEN_MODELS。
# ★ 2026-10-05：改为从 catalog 派生（仅 `enabled=True`；顺序 = 声明顺序）。
#   **新增模型只需在 `catalog.py` 追加一条，本表自动跟随** —— 这就是"预留模型接口"的落点，
#   不必再手改这里（旧实现下漏改本表 ⇒ 前端下拉看不到新模型，而后端其实已支持）。
PROVIDER_MODELS: dict[str, list[str]] = enabled_models_by_provider()

# 已从可选列表移除、但旧会话可能仍引用的模型 → 保留 provider 解析兼容（不展示在前端下拉）
# ★ 2026-10-05：数据迁至 `catalog.LEGACY_MODEL_TO_PROVIDER` 统一维护；
#   保留本变量名与语义（下游 `MODEL_TO_PROVIDER.update(_HIDDEN_MODELS)` 等引用不变）。
_HIDDEN_MODELS: dict[str, str] = dict(LEGACY_MODEL_TO_PROVIDER)

# 长上下文模型（≥128K），用于 S4 长文档总结路由（docs/统一方案 §4）
# 2026-08-25：gpt-4o / qwen-long 已下线，移出长上下文路由候选
# ★ 2026-10-05：改为从 catalog **派生**（`context_window >= LONG_CONTEXT_MIN`）。
#   旧实现要求手写维护本表，与窗口表**可能自相矛盾**（窗口写了 1M 却忘了进长文候选）——
#   派生后不可能矛盾：改窗口即自动进出长文候选。取值与改动前一致（同为那 6 个）。
LONG_CONTEXT_MODELS: set[str] = {m.model_id for m in BUILTIN_MODELS if m.long_context}

# 各模型上下文窗口（tokens，近似值，随模型更新），供总结等长上下文任务择优
# ★ 2026-10-05：改为从 catalog 派生 + 保留退役模型窗口（取值与改动前逐项一致）。
MODEL_CONTEXT_WINDOW: dict[str, int] = {
    m.model_id: m.context_window for m in BUILTIN_MODELS
}
MODEL_CONTEXT_WINDOW.update(LEGACY_MODEL_CONTEXT_WINDOW)

# 具体模型 id → 厂商（反向索引，供 chat_service 解析会话/全局所选项）
# ★ 2026-10-05：改为从 catalog 派生（**含 `enabled=False`**，保证退役 id 仍可解析）。
MODEL_TO_PROVIDER: dict[str, str] = {
    m.model_id: m.provider for m in BUILTIN_MODELS
}
# 兼容已移除的旧 id（旧会话 config 仍可解析，不展示在前端下拉）
MODEL_TO_PROVIDER.update(_HIDDEN_MODELS)

# 多模态（VLM）模型：可接收图片输入（figure_content._vlm / diagram_enrich 可选模型，2026-08-25）
# ★ 2026-10-05：改为从 catalog 派生（`multimodal=True`）。
MULTIMODAL_MODELS: set[str] = {m.model_id for m in BUILTIN_MODELS if m.multimodal}


class OAICompatibleLLM(BaseLLM):
    """OpenAI 兼容 Chat Completions 后端（支持 tools / function calling）。

    M1.1 验收点：chat/stream、重试+退避、延迟计时、成本估算。
    - 重试：网络错误(httpx.TransportError)与服务端/限流错误(429/5xx)自动指数退避重试；
            客户端错误(4xx 非 429)直接抛出不重试。每次重试记日志。
    - 延迟：真实测量请求耗时 latency_ms。
    - 成本：按 _PRICING 表估算 ¥，写入 LLMResponse.cost（NFR-09/10）。
    """

    #: **流式响应是否被整段缓冲** —— 单一来源，改 `stream()` 的传输方式时必须同步改这里，
    #: 否则耗时日志会撒谎（`_log_llm_timing` 读它决定要不要报 ttft）。
    #:
    #: **False = 真流式**（2026-09-27 改造后）：`_stream_impl` 用
    #: `async with client.stream("POST", ...)` ⇒ `aiter_lines()` **边收边解析** ⇒
    #: delta 逐块下发、`ttft_ms` 是真 TTFT。
    #:
    #: **历史**（为何曾有 True）：原实现是 `resp = await client.post(...)`，httpx 的 `post()`
    #: 走 `stream=False` ⇒ **整个 body 缓冲完才返回** ⇒ 所有 delta 在模型生成完后一次性下发。
    #: 实测 12/12 次调用 `ttft_ms ≈ latency_ms`（1742/1743、1014/1015、8919/8940…），
    #: "生成段"只有 1~21ms ⇒ 那是解析耗时，不是生成耗时。**用户体感"等满全程 → 整段突然出现"。**
    STREAM_BUFFERED: bool = False

    def __init__(
        self,
        endpoint: ModelEndpoint,
        model: str | None = None,
        *,
        api_key: str | None = None,
        client=None,
        max_retries: int = 3,
        timeout: float = 60.0,
        thinking_mode: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.model_name = model or endpoint.default_model
        self._api_key_override = api_key  # 运行时显式传入优先（避免依赖全局 os.environ 被并发请求污染）
        self._client = client  # 注入式，便于测试
        self._max_retries = max_retries
        self._timeout = timeout
        # 流式用量采集开关（2026-09-16）：默认请求 `stream_options.include_usage`；
        # 端点若明确拒绝该字段，会在首次 400 时置 False 并去掉字段重试（避免"为观测把对话打挂"）。
        self._stream_usage_enabled = True
        # 思考模式（DeepSeek V4，features.thinking 门控）：True → payload 显式启用 thinking，
        # 流式/非流式解析 reasoning_content；False → 显式禁用（默认，保证 tool_choice 兼容）。
        self.thinking_mode = thinking_mode

    def _api_key(self) -> str:
        # 显式传入的 key 优先；否则回退环境变量（.env）。两者皆空才报错。
        key = self._api_key_override or os.getenv(self.endpoint.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"缺少 {self.endpoint.name} 的 API Key（环境变量 {self.endpoint.api_key_env} 与运行时设置均未配置）。"
                f"密钥不入库（docs/10）。"
            )
        return key

    def api_key_available(self) -> bool:
        """密钥是否已配置（**不抛异常**）。

        供调用方在**长流程开始前**校验：评测判官要跑完几十秒检索 + GPU 才轮到 LLM 调用，
        若那时才发现缺密钥，白跑一轮（且容易误判为"判官没结论"）。
        """
        try:
            self._api_key()
        except RuntimeError:
            return False
        return True

    def _client_or_new(self) -> tuple[httpx.AsyncClient, bool]:
        """返回 (client, owned)：owned=True 表示本次调用新建、用完必须 aclose（2026-09-09 修复 K2）。

        原实现未注入 client 时每次调用各建一个 AsyncClient 且从不关闭——连接池与句柄
        随调用量无上限增长（本地高频调用放大）。现由 chat/stream 壳在 finally 中关闭
        owned client。**不**改为模块级共享单例的原因：httpx.AsyncClient 绑定创建它的
        事件循环，跨 loop 复用（测试 asyncio.run / 多 worker）会报
        "attached to a different loop"。
        """
        if self._client is not None:
            return self._client, False
        return httpx.AsyncClient(timeout=self._timeout), True

    @staticmethod
    def _backoff(attempt: int) -> float:
        # 指数退避：0.5s, 1s, 2s ... 上限 4s
        return min(0.5 * (2 ** (attempt - 1)), 4.0)

    async def _post_with_retry(self, client: httpx.AsyncClient, url: str, payload: dict, headers: dict) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            retryable = False
            resp: httpx.Response | None = None
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 429 or resp.status_code >= 500:
                    retryable = True
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                    logger.warning(
                        "[llm] %s 可重试 HTTP %d（第 %d/%d 次），%.1fs 后退避重试",
                        self.model_name, resp.status_code, attempt, self._max_retries,
                        self._backoff(attempt),
                    )
                elif resp.status_code >= 400:
                    # 客户端错误：把响应体也带上，便于排查（如 DashScope 400 schema）；
                    # 401/402/403/404 等常见问题翻译为可操作中文提示（余额不足/Key 无效等）
                    body_snippet = (resp.text or "")[:400]
                    raise httpx.HTTPStatusError(
                        f"{_friendly_error(resp.status_code, body_snippet)}"
                        f"（HTTP {resp.status_code}：{body_snippet}）",
                        request=resp.request, response=resp,
                    )
                else:
                    return resp
            except httpx.TransportError as e:
                retryable = True
                last_exc = e
                logger.warning(
                    "[llm] %s 网络错误（第 %d/%d 次），退避后重试：%s",
                    self.model_name, attempt, self._max_retries, type(e).__name__,
                )
            if retryable and attempt < self._max_retries:
                await asyncio.sleep(self._backoff(attempt))
                continue
            if last_exc is not None:
                raise last_exc
            if resp is not None:
                resp.raise_for_status()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _tools_payload(tools):
        """兼容 dict（来自 ToolSpec.to_llm）和 ToolSpecForLLM（来自单测）。"""
        out = []
        for t in tools:
            if isinstance(t, dict):
                out.append(t)
            else:
                out.append(t.to_openai())
        return out

    def _is_v4(self) -> bool:
        """DeepSeek V4 系列（含 vision-exp）：默认 thinking 模式与 tool_choice 不兼容，需显式控制。"""
        return self.model_name.startswith(_V4_MODEL_PREFIX)

    def _thinking_payload(self) -> dict | None:
        """thinking 显式控制，返回需注入 payload 的字段（不同厂商 key 不同）：
        - DeepSeek V4：{"thinking": {"type": "enabled"/"disabled"}}（开启=展示推理链；关闭=兼容 tool_choice）
        - qwen3 系（qwen3.7/3.6-flash、qwen3-vl-flash）：{"enable_thinking": True/False}
        - 其余模型：None（不传）
        默认一律 disabled，保证 tool_choice / Function Calling 兼容（AgentLoop 依赖）。
        """
        if self._is_v4():
            return {"thinking": {"type": "enabled" if self.thinking_mode else "disabled"}}
        if self.model_name in _QWEN3_THINKING_MODELS:
            return {"enable_thinking": self.thinking_mode}
        return None

    @staticmethod
    def _message_content(choice: dict) -> str:
        """提取 assistant 文本：优先 content；为空时回退 reasoning_content（防御性，
        防止某些厂商/配置下仅返回思考内容导致『回复空白』）。"""
        content = choice.get("content", "") or ""
        if not content:
            content = choice.get("reasoning_content", "") or ""
        return content

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpecForLLM] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,
        web_search: bool = False,
    ) -> LLMResponse:
        # 2026-09-09 修复 K2：owned client 在调用结束（含异常路径）时关闭，见 _client_or_new。
        # 原函数体平移至 _chat_impl，避免整段缩进 diff。
        client, owned = self._client_or_new()
        try:
            return await self._chat_impl(client, messages, tools=tools, temperature=temperature,
                                         max_tokens=max_tokens, tool_choice=tool_choice,
                                         web_search=web_search)
        finally:
            if owned:
                await client.aclose()

    async def _chat_impl(
        self,
        client: httpx.AsyncClient,
        messages: list[ChatMessage],
        tools: list[ToolSpecForLLM] | None,
        *,
        temperature: float,
        max_tokens: int | None,
        tool_choice: str | dict | None,
        web_search: bool,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        tp = self._thinking_payload()
        if tp is not None:
            # thinking 显式控制（DeepSeek V4: thinking.type / qwen3: enable_thinking）——
            # 开启=展示推理链（此时 tool_choice 由 AgentLoop 降级不强制）；关闭=兼容 tool_choice（默认）
            payload.update(tp)
        if tools:
            tool_payload = self._tools_payload(tools)
            if tool_payload:
                payload["tools"] = tool_payload
                # 透传 tool_choice：'auto' / 'required' / {"type":"function","function":{"name":"..."}}
                # 用 None 表示不强制。
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice
        # 原生联网搜索：当前仅 Qwen（DashScope）支持，通过顶层 enable_search=True 启用
        # （与 function calling 兼容，可同时检索文档）。DeepSeek / OpenAI 等在 chat/completions
        # 不接受 type:"web_search" 工具（实测 DeepSeek 报 400），且 DeepSeek 无可用 /responses 端点，
        # 故仅 Qwen 生效；其余厂商视为普通对话（不注入，避免 400 崩溃）。
        if web_search and self.supports_native_web_search():
            payload["enable_search"] = True
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        resp = await self._post_with_retry(client, f"{self.endpoint.base_url}/chat/completions", payload, headers)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            # 部分厂商在 200 响应体里直接返回 {"error": {...}}（如 key 无效），需提取可读原因
            err = data.get("error")
            detail = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(
                f"模型返回异常（缺少 choices 字段）：{detail or data}"
            )
        if "message" not in choices[0]:
            # 2026-09-09 修复 K6：部分兼容端点返回 delta 或纯 dict 而非 message，
            # 此前 `choices[0]["message"]` 直接 KeyError，绕过上方精心构造的可读错误提示。
            raise RuntimeError(f"模型返回异常（choices[0] 缺少 message 字段）：{choices[0]}")
        choice = choices[0]["message"]
        msg = ChatMessage(
            role=Role.ASSISTANT,
            content=self._message_content(choice),
            reasoning=choice.get("reasoning_content", "") or "",  # 思考模式：保留推理链
        )
        for tc in choice.get("tool_calls", []) or []:
            # 2026-09-09 修复 K1：与 stream() 侧（已 try/except 回退空参）口径一致——
            # 厂商返回的 tool_call 可能残缺（arguments 被截断 / 缺 name 字段），此前
            # KeyError 或 JSONDecodeError 会让**整轮对话**失败，而流式路径早已兜底。
            fn = tc.get("function") or {}
            try:
                arguments = json.loads(fn.get("arguments", "{}") or "{}")
            except (json.JSONDecodeError, TypeError, ValueError):
                arguments = {}
            msg.tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=arguments,
            ))
        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        return LLMResponse(
            message=msg,
            prompt_tokens=pt,
            completion_tokens=ct,
            model=self.model_name,
            latency_ms=latency_ms,
            cost=estimate_cost(self.model_name, pt, ct),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpecForLLM] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,
        web_search: bool = False,
    ) -> AsyncIterator[LLMStreamChunk]:
        """真实 SSE 增量流式：解析 OpenAI streaming 的 data: 行，逐 chunk 产出 delta。

        **2026-09-27 改造为真流式**：用 `client.stream()` 边收边解析（原实现用
        `client.post()` 会把整个 body 缓冲完再返回，导致 delta 一次性下发、`ttft ≈ latency`）。
        详见 `STREAM_BUFFERED` 的注释与 `_stream_impl` 里的完整说明。
        4xx/5xx 均抛带响应体的可读异常。
        """
        # 2026-09-09 修复 K2：owned client 在生成器耗尽/被关闭（finally）时 aclose。
        # 原函数体平移至 _stream_impl。
        client, owned = self._client_or_new()
        try:
            async for chunk in self._stream_impl(client, messages, tools=tools, temperature=temperature,
                                                 max_tokens=max_tokens, tool_choice=tool_choice,
                                                 web_search=web_search):
                yield chunk
        finally:
            if owned:
                await client.aclose()

    async def _stream_impl(
        self,
        client: httpx.AsyncClient,
        messages: list[ChatMessage],
        tools: list[ToolSpecForLLM] | None,
        *,
        temperature: float,
        max_tokens: int | None,
        tool_choice: str | dict | None,
        web_search: bool,
    ) -> AsyncIterator[LLMStreamChunk]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        tp = self._thinking_payload()
        if tp is not None:
            # 同 chat：thinking 显式控制（DeepSeek V4: thinking.type / qwen3: enable_thinking）
            payload.update(tp)
        if tools:
            tool_payload = self._tools_payload(tools)
            if tool_payload:
                payload["tools"] = tool_payload
                # 透传 tool_choice
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice
        # 原生联网搜索：仅 Qwen 生效（enable_search）；其余厂商忽略，避免 DeepSeek 对 type:"web_search" 报 400。
        if web_search and self.supports_native_web_search():
            payload["enable_search"] = True
        # 流式用量采集（2026-09-16 修复）：**OpenAI 兼容端点默认不在流式响应里带 usage**，
        # 必须显式请求 `stream_options.include_usage=true`，厂商才会在最后一个 chunk 顶层下发 usage。
        # 修复前这一项缺失 → pt/ct 恒 0 → 循环层 `_acc_tokens`/`_acc_cost` 恒 0
        # → **token 预算熔断与成本熔断形同虚设**（详见 docs/用量与评测功能体检报告.md）。
        if self._stream_usage_enabled:
            payload["stream_options"] = {"include_usage": True}
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        # ---- 真流式改造（2026-09-27）------------------------------------------------
        # 原实现 `resp = await client.post(...)`：httpx 的 `post()` 走 `stream=False`
        # ⇒ **把整个 body 缓冲完才返回** ⇒ 随后的 `aiter_lines()` 遍历的是**内存里的完整缓冲**
        # ⇒ 所有 delta 在模型**生成完之后一次性**下发。
        # 实测取证：12/12 次调用 `ttft_ms ≈ latency_ms`（1742/1743、1014/1015、8919/8940…），
        # 所谓"生成段"只有 1~21ms（那只是解析耗时）。**这就是"用户等满全程 → 整段突然出现"的根因**
        # —— 总时长不变，但**感知延迟 = 全程**。
        # 改用 `client.stream()` 后 `aiter_lines()` 边收边解析 ⇒ delta 真正逐块下发，
        # `ttft_ms` 才第一次成为**真 TTFT**。
        # （原 docstring 写"避免 httpx.stream 与重试纠缠不清"—— 那是当时的取舍；实测证明这个取舍
        #   让"流式"名存实亡，所以现在愿意付"重试与上下文管理器纠缠"的代价。）
        #
        # ⚠️ **重试在新实现下多了一条硬约束**：若**已经吐出过增量**再重试，用户会看到**重复内容**
        # （已下发的 delta 无法回滚）⇒ 只有 `emitted=False` 时才允许重试。
        t0 = time.perf_counter()  # 流式延迟起点（覆盖"建连 + 收流"全程）
        url = f"{self.endpoint.base_url}/chat/completions"
        emitted = False            # 是否已向调用方下发过增量（决定"能否重试"）
        last_exc: Exception | None = None
        sink: dict[str, Any] = {}  # 解析累计量（pt/ct/ttft/reasoning），由 `_parse_sse_stream` 回填
        for attempt in range(1, self._max_retries + 1):
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code == 429 or resp.status_code >= 500:
                        # 流式响应读完前 `.text` 会抛 ResponseNotRead ⇒ 必须先 aread()
                        await resp.aread()
                        last_exc = httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}：{resp.text[:300]}",
                            request=resp.request, response=resp,
                        )
                        logger.warning("[llm] %s 流式可重试 HTTP %d（第 %d/%d 次）", self.model_name, resp.status_code, attempt, self._max_retries)
                        if attempt < self._max_retries:
                            await asyncio.sleep(self._backoff(attempt))
                            continue
                        raise last_exc
                    if resp.status_code >= 400:
                        await resp.aread()
                        # 个别兼容实现不认 `stream_options` 会直接 400。**为了让用量可观测而把整条流式链路打挂
                        # 是不可接受的**：识别到该字段被拒 → 去掉它再试一次，并记住这次端点不支持。
                        if "stream_options" in payload and "stream_options" in resp.text[:600]:
                            logger.warning(
                                "[llm] %s 端点拒绝 stream_options，已关闭流式用量请求（成本/延迟回落为本地估算）",
                                self.model_name,
                            )
                            self._stream_usage_enabled = False
                            payload.pop("stream_options", None)
                            # 记下原因：若这已是最后一次尝试，应抛它而不是含糊的"连接未建立"
                            last_exc = httpx.HTTPStatusError(
                                f"端点拒绝 stream_options（HTTP {resp.status_code}）",
                                request=resp.request, response=resp,
                            )
                            continue
                        last_exc = httpx.HTTPStatusError(
                            f"{_friendly_error(resp.status_code, resp.text[:300])}"
                            f"（HTTP {resp.status_code}：{resp.text[:400]}）",
                            request=resp.request, response=resp,
                        )
                        raise last_exc
                    # ---- 200：边收边解析（**真流式**，此处才真正逐块下发）----
                    sink = {}   # 每次尝试都重置：失败的尝试不得留下半截累计量
                    async for chunk in self._parse_sse_stream(resp, sink, t0):
                        if chunk.delta or chunk.reasoning:
                            emitted = True  # 已有增量下发 ⇒ 后续出错不能再重试（会重复）
                        yield chunk
                    break  # 解析完成 → 跳出重试循环
            except httpx.TransportError as e:
                if emitted:
                    # 已产出增量：重放会造成重复内容，**只能如实报错**，不能静默重试
                    logger.warning(
                        "[llm] %s 流式中途传输中断，且已产出增量 ⇒ 不重试（重放会重复）：%s",
                        self.model_name, type(e).__name__,
                    )
                    raise
                last_exc = e
                logger.warning("[llm] %s 流式网络错误（第 %d/%d 次）：%s", self.model_name, attempt, self._max_retries, type(e).__name__)
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise
        else:
            # for 自然走完（未 break）⇒ 所有尝试都未成功 → 抛最后一次的原因
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("stream 连接未建立")
        pt = int(sink.get("pt", 0))
        ct = int(sink.get("ct", 0))
        if pt == 0 and ct == 0:
            # 厂商未下发 usage（端点不支持 / 未按 include_usage 返回）：
            # 显式告警而不是静默 0 —— 上游据此改走本地估算，熔断才不至于失效。
            logger.warning(
                "[llm] %s 流式响应未含 usage（pt/ct=0）：成本与 token 将回落为本地估算",
                self.model_name,
            )
        # ⚠️ `latency_ms` 在**收完流之后**测：它是"整段等待"；与 `ttft_ms`（首块到达）一起
        # 才把一次调用拆成「等服务端开口」与「逐字生成」两段。**真流式下这两个数才都可信。**
        yield LLMStreamChunk(
            finished=True, prompt_tokens=pt, completion_tokens=ct,
            reasoning=str(sink.get("reasoning", "")),
            latency_ms=int((time.perf_counter() - t0) * 1000),
            ttft_ms=int(sink.get("ttft_ms", 0)),
            cost=estimate_cost(self.model_name, pt, ct),
        )

    async def _parse_sse_stream(
        self, resp: httpx.Response, sink: dict[str, Any], t0: float
    ) -> AsyncIterator[LLMStreamChunk]:
        """解析 OpenAI 兼容 SSE 流，逐块产出增量；累计量写入 `sink`。

        T-05+（2026-09-27）：从 `_stream_impl` 抽出的。抽出的**唯一目的**是让这段解析体
        **不必再缩进一层** —— 改真流式后它必须位于 `async with client.stream(...)` 内部，
        直接内联会让 80 行代码整体右移 8 格，diff 巨大且易错。抽成方法后缩进层级不变。

        ⚠️ **`sink` 只在解析完整结束时写入**：中途异常（连接断）不会留下半截累计量，
        调用方据此判断"这次尝试是否成功"（配合 `emitted` 决定能否重试）。

        ⚠️ `t0` 由调用方传入，是**建连之前**的时刻 ⇒ `ttft_ms` 含建连 + 服务端排队 + prefill。
        在 `client.post()`（整段缓冲）时代它恒 ≈ `latency_ms`；改 `client.stream()` 后才成真 TTFT。
        """
        content_buf: list[str] = []
        reasoning_buf: list[str] = []  # 思考模式：reasoning_content 增量（DeepSeek V4）
        tool_acc: dict[str, dict[str, Any]] = {}
        pt = ct = 0
        ttft_ms = 0
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            choice = choices[0] if choices else {}
            delta = choice.get("delta", {})
            # 思考增量独立通道（前端 thinking 展示）；正文增量走 content
            r_delta = delta.get("reasoning_content") or ""
            if r_delta:
                reasoning_buf.append(r_delta)
                yield LLMStreamChunk(reasoning=r_delta)
            text_delta = delta.get("content") or ""
            # 首 token 判定：任一通道有增量即算「开口」（先判后 yield，避免多写三处）
            if not ttft_ms and (r_delta or text_delta or delta.get("tool_calls")):
                ttft_ms = max(1, int((time.perf_counter() - t0) * 1000))
            if text_delta:
                content_buf.append(text_delta)
                yield LLMStreamChunk(delta=text_delta)
            for tc in delta.get("tool_calls", []) or []:
                # 用 index 作为 key（同一调用的多块共享 index）；id 在 Qwen 流式后续块中
                # 常为 ""，不能直接当 key，否则同一调用被误拆成多个 ToolCall。
                idx_v = tc.get("index", 0)
                fid = f"idx{idx_v}"
                fn = tc.get("function", {})
                slot = tool_acc.setdefault(fid, {"name": "", "args": "", "id": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if fn.get("name"):
                    slot["name"] = fn["name"]
                slot["args"] += fn.get("arguments", "")
            # 流式用量：OpenAI 兼容规范把 usage 放在 **chunk 顶层**（仅当 include_usage 生效时下发），
            # 部分实现塞在 choices[0] 里 → 两处都认，顶层优先。
            # 修复前只读 `choice.get("usage")` → 即使厂商下发了 usage 也读不到（K7，2026-09-16）。
            usage = chunk.get("usage") or choice.get("usage")
            if usage:
                pt = usage.get("prompt_tokens", pt) or pt
                ct = usage.get("completion_tokens", ct) or ct
        # 解析完整结束 → **一次性**回填累计量（中途异常不写 ⇒ 调用方据此判"这次尝试是否成功"）
        sink.update(pt=pt, ct=ct, ttft_ms=ttft_ms, reasoning="".join(reasoning_buf))
        calls = []
        for fid, slot in tool_acc.items():
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=slot.get("id") or fid, name=slot["name"], arguments=args))
        if calls:
            yield LLMStreamChunk(tool_calls=calls)
        # ⚠️ **不在这里 yield `finished` 块**：它需要 `latency_ms`（收完流的墙钟），
        # 而 `t0` 在调用方手里。由 `_stream_impl` 在本生成器返回后统一发 ——
        # 这样解析层只回答"首块何时到"（ttft_ms），"整段等了多久"留给调用方。


    # ---------------- 原生联网搜索（2026-08-27 接入，2026-08-27 晚修复） ----------------
    # 让模型在生成时自带联网检索与引用，替代「工具检索 github/arxiv」模式。
    # 当前仅 Qwen（DashScope）支持：chat(web_search=True) → 顶层 enable_search=True。
    # DeepSeek / OpenAI 等在 chat/completions 不接受 type:"web_search" 工具（DeepSeek 实测 400），
    # 且其 /responses 端点不可用，故仅 Qwen 生效；其余厂商视为普通对话。
    def supports_native_web_search(self) -> bool:
        """本模型是否支持原生联网搜索（**厂商 + 模型**双重判定，见 NATIVE_WEB_SEARCH_MODELS）。

        ⚠️ 必须判到模型级：同厂内差异真实存在（qwen3-max 支持、qwen3-vl-flash 不支持）。
        白名单制 = 「已实测支持」，未列入的一律保守视为不支持 —— 宁可多提示一句
        （用户可关闭，且关掉开关即可消除），也不要漏报（用户以为联网生效、实际不搜）。
        """
        if self.endpoint.name != "qwen":
            return False
        return self.model_name in NATIVE_WEB_SEARCH_MODELS

    async def chat_web_search(
        self, messages: list[ChatMessage], *, temperature: float = 0.3
    ) -> LLMResponse:
        """原生联网搜索合成：模型自带检索+引用，返回含正文的 LLMResponse。

        当前仅 Qwen（enable_search=True）生效；其余厂商抛出 RuntimeError，由调用方优雅降级。
        异常（4xx/5xx/网络）向上抛，由调用方优雅降级。
        """
        if not self.supports_native_web_search():
            raise RuntimeError(
                f"模型 {self.model_name}（厂商 {self.endpoint.name}）不支持原生联网搜索"
                f"（已实测支持的模型见 NATIVE_WEB_SEARCH_MODELS）"
            )
        resp = await self.chat(messages, web_search=True, temperature=temperature)
        return resp


class ModelRouter:
    """按名路由 LLM 后端。"""

    def __init__(self) -> None:
        self._backends: dict[str, BaseLLM] = {}

    def register(self, name: str, llm: BaseLLM) -> None:
        self._backends[name] = llm

    def get(self, name: str) -> BaseLLM:
        if name not in self._backends:
            raise KeyError(f"未注册的模型后端: {name}，可选: {list(self._backends)}")
        return self._backends[name]

    @classmethod
    def from_presets(cls, prefer: str = "deepseek", fake: bool = True) -> ModelRouter:
        """默认注册 FakeLLM（无 key 可跑）+ 预置真实端点（按需 key 才可用）。"""
        router = cls()
        if fake:
            from .fake import FakeLLM

            router.register("fake", FakeLLM())
            router.register("default", FakeLLM())
        if prefer in PRESET_ENDPOINTS:
            router.register(prefer, OAICompatibleLLM(PRESET_ENDPOINTS[prefer]))
        return router
