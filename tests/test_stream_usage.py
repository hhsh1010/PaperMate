"""流式用量采集回归（2026-09-16，修复 `docs/用量与评测功能体检报告.md` 问题 1）

**修复前的三个缺陷**（任一都足以让用量/熔断失效）：
1. `stream()` 请求 payload **没带** `stream_options:{"include_usage":true}` → 厂商默认不下发 usage；
2. usage 解析读的是 `choices[0].usage`，而 OpenAI 兼容规范把它放在 **chunk 顶层**；
3. 流式路径构造 `LLMResponse` 时**不传** `cost` / `latency_ms`
   → 循环层 `_acc_cost` / `_acc_tokens` / `_acc_latency` 恒 0。

**后果**：`Trace.total_cost` / `total_latency_ms` 几乎全为 0；
**token 预算熔断与单轮成本熔断永不触发**（护栏变装饰）。

本文件把这三层逐层钉住，并额外锁住两条边界：
- 端点**拒绝** `stream_options` 时必须优雅降级（去掉字段重试），不能让"为了可观测"把流式对话打挂；
- 厂商**不下发** usage 时必须走本地估算兜底（否则熔断重新变装饰）。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from lightcore.core.loop import AgentLoop, LoopConfig, LoopState
from lightcore.llm.base import ChatMessage, LLMStreamChunk, Role
from lightcore.llm.fake import FakeLLM
from lightcore.llm.router import ModelEndpoint, OAICompatibleLLM
from lightcore.tools.spec import ToolExecutor, ToolRegistry

PRICED_MODEL = "deepseek-v4-flash"  # 在 _PRICING 表内（in=0.001 / out=0.002 ¥/1k）


def _endpoint(model: str = PRICED_MODEL) -> ModelEndpoint:
    return ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", model)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _sse(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n"


def _collect(llm: OAICompatibleLLM) -> list[LLMStreamChunk]:
    async def run():
        return [c async for c in llm.stream([ChatMessage(role=Role.USER, content="hi")])]

    return asyncio.run(run())


# ---------------- 层 1：请求形态 ----------------


def test_stream_payload_requests_include_usage():
    """必须显式请求 `stream_options.include_usage`，否则厂商不会下发 usage。"""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "好"}}]}),
                              headers={"Content-Type": "text/event-stream"})

    _collect(OAICompatibleLLM(_endpoint(), api_key="k", client=_client(handler)))

    assert captured["body"].get("stream_options") == {"include_usage": True}, \
        "流式请求必须带 stream_options.include_usage，否则 usage 永远不会下发"


def test_endpoint_rejecting_stream_options_degrades_gracefully():
    """端点明确拒绝 stream_options 时：去掉字段重试一次并成功，而不是让整条流式链路报错。"""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "stream_options" in body:
            return httpx.Response(400, text='{"error":{"message":"Unknown field: stream_options"}}')
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "ok"}}]}),
                              headers={"Content-Type": "text/event-stream"})

    llm = OAICompatibleLLM(_endpoint(), api_key="k", client=_client(handler))
    chunks = _collect(llm)

    assert any(c.finished for c in chunks), "降级后仍应正常拿到流式结果"
    assert len(seen) == 2 and "stream_options" in seen[0] and "stream_options" not in seen[1]
    assert llm._stream_usage_enabled is False, "记住该端点不支持，后续不再重复尝试"


# ---------------- 层 2：usage 解析 ----------------


def test_stream_reads_top_level_usage_and_sets_cost_and_latency():
    """usage 在 chunk **顶层**（OpenAI 兼容规范）时必须读到，并算出 cost / latency。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(
            {"choices": [{"delta": {"content": "你"}}]},
            {"choices": [{"delta": {"content": "好"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 200}},
        ), headers={"Content-Type": "text/event-stream"})

    chunks = _collect(OAICompatibleLLM(_endpoint(), api_key="k", client=_client(handler)))
    fin = next(c for c in chunks if c.finished)

    assert (fin.prompt_tokens, fin.completion_tokens) == (100, 200)
    # cost = 100/1000*0.001 + 200/1000*0.002 = 0.0005
    assert fin.cost == pytest.approx(0.0005), "流式 finished 块必须带成本（否则 _acc_cost 恒 0）"
    assert fin.latency_ms >= 0, "流式路径必须测延迟（修复前恒 0）"


def test_stream_reads_legacy_choice_level_usage():
    """兼容把 usage 塞在 choices[0] 里的实现（回退分支）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(
            {"choices": [{"delta": {"content": "x"}, "usage": {"prompt_tokens": 7, "completion_tokens": 9}}]},
        ), headers={"Content-Type": "text/event-stream"})

    fin = next(c for c in _collect(OAICompatibleLLM(_endpoint(), api_key="k", client=_client(handler)))
               if c.finished)
    assert (fin.prompt_tokens, fin.completion_tokens) == (7, 9)


def test_stream_matches_dashscope_documented_frame_sequence():
    """按阿里云百炼（DashScope）官方文档的真实帧序列钉住解析行为。

    官方样例（`help.aliyun.com/zh/model-studio/stream`）：
    - 每个内容帧形如 `{"choices":[{"delta":{...},"finish_reason":null}],...,"usage":null}`；
    - **最后一个帧不含 choices、只含 usage**（所以必须读 **chunk 顶层** usage，且不能因 choices
      为空就跳过该帧）。
    另据官方说明：OpenAI 兼容协议**默认不返回**用量，必须显式设置
    `stream_options={"include_usage": true}` —— 这正是本修复的一半。
    """

    def content_frame(text: str) -> dict:
        return {"choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}],
                "usage": None, "model": "qwen-plus"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(
            content_frame("你"),
            content_frame("好"),
            content_frame(""),
            {"choices": [], "usage": {"prompt_tokens": 26, "completion_tokens": 87, "total_tokens": 113}},
        ), headers={"Content-Type": "text/event-stream"})

    chunks = _collect(OAICompatibleLLM(_endpoint("qwen3.7-flash"), api_key="k", client=_client(handler)))
    fin = next(c for c in chunks if c.finished)

    assert (fin.prompt_tokens, fin.completion_tokens) == (26, 87), \
        "最后一帧没有 choices，只认『有 choices 的帧』会漏掉用量（官方帧序列如此）"
    assert "".join(c.delta for c in chunks) == "你好"


# ---------------- 层 3：循环层累加（决定熔断是否生效） ----------------


def _loop_with_stream(llm: FakeLLM, **cfg) -> AgentLoop:
    return AgentLoop(llm, ToolExecutor(ToolRegistry()), config=LoopConfig(**cfg))


class _StreamFake(FakeLLM):
    """可控流式替身：finished 块上的用量由测试给定；usage=None 模拟厂商不下发 usage。"""

    def __init__(self, usage: tuple[int, int, int, float] | None) -> None:
        super().__init__("stub-model")
        self._usage = usage

    async def stream(self, messages, tools=None, **kw):  # noqa: ARG002
        yield LLMStreamChunk(delta="这是一段足够长的回答，用于本地 token 估算兜底。")
        if self._usage is None:
            yield LLMStreamChunk(finished=True)  # 厂商未下发 usage
        else:
            pt, ct, lat, cost = self._usage
            yield LLMStreamChunk(finished=True, prompt_tokens=pt, completion_tokens=ct,
                                 latency_ms=lat, cost=cost)


def _done(events: list) -> dict:
    return next(e.data for e in events if e.type == "done")


def test_loop_accumulates_stream_usage_into_done_event():
    """核心回归：流式的 usage 必须进入 done 事件（→ 落库 → Trace → 用量页/看板）。"""
    async def run():
        loop = _loop_with_stream(_StreamFake((120, 340, 880, 0.0008)))
        return [e async for e in loop.run_stream("问题", stream_tokens=True)]

    data = _done(asyncio.run(run()))

    assert data["total_tokens"] == 460, f"token 未被累加：{data}"
    assert data["cost"] == pytest.approx(0.0008), f"成本未被累加：{data}"
    assert data["latency_ms"] == 880, f"延迟未被累加：{data}"


def test_loop_estimates_usage_when_provider_omits_it():
    """厂商不下发 usage 时必须本地估算 —— 否则 token/成本熔断重新变成装饰。"""
    async def run():
        loop = _loop_with_stream(_StreamFake(None))
        return [e async for e in loop.run_stream("问题", stream_tokens=True)]

    data = _done(asyncio.run(run()))

    assert data["total_tokens"] > 0, "缺少 usage 时必须本地估算，不能留 0（否则熔断失效）"
    assert data["cost"] >= 0.0


def test_context_fuse_fires_on_stream_with_large_prompt():
    """单次请求上下文超限必须在流式路径下真正触发（修复前 `_acc_tokens` 恒 0 → 永不触发）。

    2026-09-17 判据口径调整：本项从「整轮累计」改为「**单轮 prompt**」——
    累计口径另见 `test_token_budget_split.py::test_cumulative_tokens_over_turn_limit_aborts`。
    流式路径仍是同一处 `_think` 收尾逻辑，所以这条仍是"流式下熔断真的会触发"的守门测试。
    """
    async def run():
        # 单次调用的 prompt 就报 4000，上限 1000 → 应立即熔断
        loop = _loop_with_stream(_StreamFake((4000, 1000, 12, 0.0)), max_tokens_budget=1000)
        return [e async for e in loop.run_stream("问题", stream_tokens=True)]

    events = asyncio.run(run())
    aborts = [e for e in events if e.type == "error" and e.state == LoopState.ABORT]
    assert aborts, "单轮 prompt 超限必须中止本轮（修复前这条永远不会触发）"
    assert "上下文超限" in (aborts[0].data or {}).get("reason", "")


def test_cost_fuse_fires_on_stream_with_large_cost():
    """成本熔断同理：单轮成本超上限必须中止。"""
    async def run():
        loop = _loop_with_stream(_StreamFake((9000, 9000, 30, 5.0)), max_cost_per_turn=1.0)
        return [e async for e in loop.run_stream("问题", stream_tokens=True)]

    events = asyncio.run(run())
    aborts = [e for e in events if e.type == "error" and e.state == LoopState.ABORT]
    assert aborts, "单轮成本超上限必须中止（修复前这条永远不会触发）"
    assert "熔断" in (aborts[0].data or {}).get("reason", "")


def test_budget_aborts_carry_structured_kind():
    """④（2026-10-02）关键链路：预算熔断必须带**结构化** `failure_kind`。

    为什么这条是"用户可见文案正确"的前提：路由按 `failure_kind` 选文案
    （`services/failure_notice.fallback_text`）—— 若这里缺 kind，用户又会看到
    「可能是模型调用失败或返回内容为空」+「查余额/换模型」的**错误方向**。
    """
    async def run(spec, **cfg):
        loop = _loop_with_stream(_StreamFake(spec), **cfg)
        return [e async for e in loop.run_stream("问题", stream_tokens=True)]

    # ① 单次上下文超限
    ctx = asyncio.run(run((4000, 1000, 12, 0.0), max_tokens_budget=1000))
    ctx_aborts = [e for e in ctx if e.type == "error" and e.state == LoopState.ABORT]
    assert ctx_aborts, "前置条件：单次上下文超限应中止"
    assert (ctx_aborts[0].data or {}).get("failure_kind") == "budget_exceeded", (
        f"缺少结构化 kind ⇒ 路由无法给出正确归因：{ctx_aborts[0].data}"
    )

    # ② 单轮成本熔断
    cost = asyncio.run(run((9000, 9000, 30, 5.0), max_cost_per_turn=1.0))
    cost_aborts = [e for e in cost if e.type == "error" and e.state == LoopState.ABORT]
    assert cost_aborts, "前置条件：成本熔断应中止"
    assert (cost_aborts[0].data or {}).get("failure_kind") == "budget_exceeded", (
        f"缺少结构化 kind ⇒ 路由无法给出正确归因：{cost_aborts[0].data}"
    )
