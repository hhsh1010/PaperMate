"""四模型接入（2026-08-25）：qwen3.7-flash / qwen3.6-flash / qwen3-vl-flash / deepseek-v4-flash

断言：模型目录（PROVIDER_MODELS）/ 反向解析（MODEL_TO_PROVIDER）/ 上下文窗口（MODEL_CONTEXT_WINDOW）/
长上下文路由（LONG_CONTEXT_MODELS）/ 多模态集合（MULTIMODAL_MODELS）/ thinking payload（按厂商格式）。
纯逻辑，无网络调用。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from lightcore.llm.router import (
    LONG_CONTEXT_MODELS,
    MODEL_CONTEXT_WINDOW,
    MODEL_TO_PROVIDER,
    MULTIMODAL_MODELS,
    PRESET_ENDPOINTS,
    PROVIDER_MODELS,
    OAICompatibleLLM,
)
from lightcore.rag.figure_content import FigureEnricher

NEW_MODELS = {"qwen3.7-flash", "qwen3.6-flash", "qwen3-vl-flash", "deepseek-v4-flash"}


def test_new_models_in_provider_catalog():
    assert "qwen3.7-flash" in PROVIDER_MODELS["qwen"]
    assert "qwen3.6-flash" in PROVIDER_MODELS["qwen"]
    assert "qwen3-vl-flash" in PROVIDER_MODELS["qwen"]
    assert "deepseek-v4-flash" in PROVIDER_MODELS["deepseek"]


def test_retired_deepseek_models_removed_but_compat():
    # 官方 2026-07-24 退役 → 不再展示在前端下拉
    assert "deepseek-chat" not in PROVIDER_MODELS["deepseek"]
    assert "deepseek-reasoner" not in PROVIDER_MODELS["deepseek"]
    # 但旧会话 config 仍可解析（_HIDDEN_MODELS 兼容），不落 fake
    assert MODEL_TO_PROVIDER["deepseek-chat"] == "deepseek"
    assert MODEL_TO_PROVIDER["deepseek-reasoner"] == "deepseek"


def test_removed_qwen_and_gpt_models_not_in_catalog():
    # 2026-08-25：qwen-plus/turbo/long 与 gpt 系列已下线（前端不再展示）
    assert "qwen-plus" not in PROVIDER_MODELS["qwen"]
    assert "qwen-turbo" not in PROVIDER_MODELS["qwen"]
    assert "qwen-long" not in PROVIDER_MODELS["qwen"]
    assert "openai" not in PROVIDER_MODELS
    # 旧会话 config 仍可解析（_HIDDEN_MODELS 兼容）
    assert MODEL_TO_PROVIDER["qwen-plus"] == "qwen"
    assert MODEL_TO_PROVIDER["gpt-4o"] == "openai"


def test_model_to_provider_resolution():
    assert MODEL_TO_PROVIDER["qwen3-vl-flash"] == "qwen"
    assert MODEL_TO_PROVIDER["qwen3.7-flash"] == "qwen"
    assert MODEL_TO_PROVIDER["qwen3.6-flash"] == "qwen"
    assert MODEL_TO_PROVIDER["deepseek-v4-flash"] == "deepseek"


def test_context_window_and_long_context():
    assert MODEL_CONTEXT_WINDOW["qwen3.7-flash"] == 1_000_000
    assert MODEL_CONTEXT_WINDOW["qwen3.6-flash"] == 1_000_000
    assert MODEL_CONTEXT_WINDOW["qwen3-vl-flash"] == 262_144
    assert MODEL_CONTEXT_WINDOW["deepseek-v4-flash"] == 1_000_000
    assert NEW_MODELS <= LONG_CONTEXT_MODELS


def test_multimodal_models():
    assert MULTIMODAL_MODELS == {"deepseek-v4-flash-vision-exp", "qwen3-vl-flash"}


def test_thinking_payload_by_provider():
    # qwen3 系：enable_thinking（bool）
    llm_q = OAICompatibleLLM(PRESET_ENDPOINTS["qwen"], model="qwen3-vl-flash")
    assert llm_q._thinking_payload() == {"enable_thinking": False}
    llm_q_on = OAICompatibleLLM(PRESET_ENDPOINTS["qwen"], model="qwen3.7-flash", thinking_mode=True)
    assert llm_q_on._thinking_payload() == {"enable_thinking": True}
    # DeepSeek V4：thinking.type
    llm_d = OAICompatibleLLM(PRESET_ENDPOINTS["deepseek"], model="deepseek-v4-flash")
    assert llm_d._thinking_payload() == {"thinking": {"type": "disabled"}}
    # 其他模型（非 thinking 系列）：不传（2026-08-25：openai 已下线，用 qwen3-max 验证 None 分支）
    llm_o = OAICompatibleLLM(PRESET_ENDPOINTS["qwen"], model="qwen3-max")
    assert llm_o._thinking_payload() is None


def test_figure_enricher_vlm_model_configurable():
    # 显式配置 qwen3-vl-flash
    en = FigureEnricher(SimpleNamespace(figure_enrich={
        "enabled": True, "methods": ["vlm"], "vlm_model": "qwen3-vl-flash",
    }))
    assert en.vlm_model == "qwen3-vl-flash"
    # 未配置 → 默认 deepseek-v4-flash-vision-exp
    en2 = FigureEnricher(SimpleNamespace(figure_enrich={"enabled": True, "methods": ["vlm"]}))
    assert en2.vlm_model == "deepseek-v4-flash-vision-exp"


def test_stream_handles_empty_choices_chunk():
    """2026-08-25 回归：qwen3 流式会发 choices=[] 空 chunk（如 usage 包），
    解析必须跳过而非 IndexError（此前导致 qwen3.6/3.7-flash 对话无有效回复）。"""
    import httpx

    from lightcore.llm.base import ChatMessage, Role
    from lightcore.llm.router import ModelEndpoint, OAICompatibleLLM

    sse_body = (
        'data: {"choices": []}\n\n'
        'data: {"choices": [{"delta": {"content": "你"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "好"}}]}\n\n'
        'data: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse_body.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    ep = ModelEndpoint("mock", "https://example.com/v1", "MOCK_KEY", "mock-model")
    llm = OAICompatibleLLM(ep, client=client, api_key="sk-test")

    async def run() -> str:
        parts: list[str] = []
        async for c in llm.stream([ChatMessage(role=Role.USER, content="hi")]):
            if c.delta:
                parts.append(c.delta)
        await client.aclose()
        return "".join(parts)

    assert asyncio.run(run()) == "你好"
