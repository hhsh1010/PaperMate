"""OAICompatibleLLM 原生联网搜索测试（2026-08-27 接入，2026-08-27 晚收窄为仅 Qwen）。

覆盖：
- supports_native_web_search 厂商判定（仅 qwen=True；deepseek/openai 不支持）
- qwen：chat/completions 注入 enable_search=True，并正确解析 choices[0].message.content
- deepseek：chat() 不注入非法 web_search 工具（避免 400），chat_web_search 直接抛 RuntimeError
均用 httpx.MockTransport 离线验证请求形态与响应解析，不触网。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.router import ModelEndpoint, OAICompatibleLLM


def _qwen_endpoint(model: str = "qwen3.7-flash") -> ModelEndpoint:
    return ModelEndpoint("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "DASHSCOPE_API_KEY", model)


def _deepseek_endpoint(model: str = "deepseek-v4-flash") -> ModelEndpoint:
    return ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", model)


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------- 能力判定 ----------------
def test_supports_native_web_search_by_provider():
    assert OAICompatibleLLM(_qwen_endpoint()).supports_native_web_search() is True
    # DeepSeek chat/completions 不接受 type:"web_search" 工具（实测 400），故不支持
    assert OAICompatibleLLM(_deepseek_endpoint()).supports_native_web_search() is False
    # 非 qwen 厂商应返回 False
    assert OAICompatibleLLM(ModelEndpoint("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-x"),
                            ).supports_native_web_search() is False


# ---------------- qwen：chat/completions + enable_search ----------------
async def test_chat_web_search_qwen_injects_enable_search():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "答：根据最新资料，PyTorch 2.3 已发布（来源：xxx）。"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 20},
        })

    llm = OAICompatibleLLM(_qwen_endpoint(), api_key="sk-test", client=_client(handler))
    resp = await llm.chat_web_search(
        [ChatMessage(role=Role.USER, content="PyTorch 最新版本？")],
        temperature=0.3,
    )
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"].get("enable_search") is True, "qwen 应注入 enable_search=True"
    assert "PyTorch 2.3" in resp.message.content
    assert resp.prompt_tokens == 12 and resp.completion_tokens == 20


async def test_chat_web_search_qwen_stream_injects_enable_search():
    """流式路径同样注入 enable_search（对话链路走 stream）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        # OpenAI 流式：单条 data: 行结尾
        return httpx.Response(200, text="data: " + json.dumps({
            "choices": [{"delta": {"content": "联网答案片段"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }) + "\n\ndata: [DONE]\n", headers={"Content-Type": "text/event-stream"})

    llm = OAICompatibleLLM(_qwen_endpoint(), api_key="sk-test", client=_client(handler))
    chunks = [c async for c in llm.stream(
        [ChatMessage(role=Role.USER, content="最新 AI 新闻？")], web_search=True, temperature=0.3)]
    assert captured["body"].get("enable_search") is True
    assert "".join(c.delta or "" for c in chunks) == "联网答案片段"


# ---------------- deepseek：不注入 web_search（避免 400） ----------------
async def test_chat_deepseek_web_search_disabled_no_tool_injection():
    """DeepSeek 不支持原生联网：chat(web_search=True) 不得注入非法工具，避免 400。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "普通回答，无联网"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        })

    llm = OAICompatibleLLM(_deepseek_endpoint(), api_key="sk-test", client=_client(handler))
    resp = await llm.chat(
        [ChatMessage(role=Role.USER, content="Hi")], web_search=True, temperature=0.3)
    # 不应出现 type=web_search 的非法工具（否则 DeepSeek 会 400）
    assert not any(t.get("type") == "web_search" for t in captured["body"].get("tools", []))
    assert "普通回答" in resp.message.content


def test_chat_web_search_deepseek_raises():
    """deepseek 调用 chat_web_search 应直接抛 RuntimeError（由调用方降级）。"""
    llm = OAICompatibleLLM(_deepseek_endpoint(), api_key="sk-test")
    with pytest.raises(RuntimeError):
        asyncio.run(llm.chat_web_search([ChatMessage(role=Role.USER, content="x")], temperature=0.3))


def test_chat_web_search_unsupported_provider_raises():
    llm = OAICompatibleLLM(ModelEndpoint("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-x"),
                           api_key="sk-test")
    with pytest.raises(RuntimeError):
        asyncio.run(llm.chat_web_search([], temperature=0.3))
