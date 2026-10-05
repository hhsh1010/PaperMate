"""DeepSeek-V4-Flash-Vision 多模态模型更换测试（2026-08-22）。

覆盖：
- 模型目录：PRESET_ENDPOINTS 默认模型 / PROVIDER_MODELS 首位 / 定价 / 1M 上下文 / 长上下文集合；
- ChatMessage 多模态 content 块：to_openai 透传、text_only 提取、_rough_tokens 计费（文本 + 384/图）；
- OAICompatibleLLM：多模态消息序列化后 payload 含 text + image_url 块（MockTransport 断言）；
- routes_search：_build_user_content 组装、ChatRequest.images 校验（数量/格式 400）、带图 SSE 全链路不崩（fake）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.llm.base import ChatMessage, Role, _rough_tokens
from lightcore.llm.router import (
    LONG_CONTEXT_MODELS,
    MODEL_CONTEXT_WINDOW,
    MODEL_TO_PROVIDER,
    PRESET_ENDPOINTS,
    PROVIDER_MODELS,
    ModelEndpoint,
    OAICompatibleLLM,
    estimate_cost,
)
from papermate.api.routes_search import _build_user_content


# ---------------- 模型目录 ----------------
def test_deepseek_default_model_is_vision():
    ep = PRESET_ENDPOINTS["deepseek"]
    assert ep.default_model == "deepseek-v4-flash-vision-exp"
    # 官方 id 在可选模型目录首位（该厂商默认）
    assert PROVIDER_MODELS["deepseek"][0] == "deepseek-v4-flash-vision-exp"
    # 反向索引可解析
    assert MODEL_TO_PROVIDER["deepseek-v4-flash-vision-exp"] == "deepseek"


def test_vision_model_specs():
    # 定价存在（与 V4-Flash 一致，近似值）
    assert "deepseek-v4-flash-vision-exp" in estimate_cost.__globals__["_PRICING"]
    assert estimate_cost("deepseek-v4-flash-vision-exp", 1000, 1000) > 0
    # 1M 上下文窗口 + 长上下文集合（S4 长文档总结可用）
    assert MODEL_CONTEXT_WINDOW["deepseek-v4-flash-vision-exp"] == 1_000_000
    assert "deepseek-v4-flash-vision-exp" in LONG_CONTEXT_MODELS


# ---------------- ChatMessage 多模态 ----------------
def test_chat_message_multimodal_to_openai():
    blocks = [
        {"type": "text", "text": "这张图里有什么？"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    m = ChatMessage(role=Role.USER, content=blocks)
    out = m.to_openai()
    assert out["role"] == "user"
    assert out["content"] == blocks  # 块列表原样透传（vision 模型接口规范）
    # 纯文本消息保持字符串（回归）
    assert ChatMessage(role=Role.USER, content="hi").to_openai()["content"] == "hi"
    assert ChatMessage(role=Role.USER, content="").to_openai()["content"] == ""


def test_chat_message_text_only():
    m = ChatMessage(role=Role.USER, content=[
        {"type": "text", "text": "第一段"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "第二段"},
    ])
    assert m.text_only() == "第一段第二段"
    assert ChatMessage(role=Role.USER, content="纯文本").text_only() == "纯文本"


def test_rough_tokens_multimodal():
    # 文本按字数估；每张图按官方上限 384 tokens 计（本地成本估算/熔断用）
    blocks = [
        {"type": "text", "text": "十个字十个字十个字十个字十个字"},  # 30 字
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBBB"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,CCCC"}},
    ]
    assert _rough_tokens(blocks) == _rough_tokens("十个字十个字十个字十个字十个字") + 384 * 2
    assert _rough_tokens("abc") >= 1
    assert _rough_tokens("") == 0


# ---------------- OAICompatibleLLM 多模态序列化 ----------------
def test_oaicompatible_serializes_image_blocks():
    captured: dict = {}

    def fake_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    async def run():
        ep = ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-flash-vision-exp")
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake_handler))
        llm = OAICompatibleLLM(ep, client=client, api_key="sk-test")
        blocks = [
            {"type": "text", "text": "看图回答"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ]
        resp = await llm.chat([ChatMessage(role=Role.USER, content=blocks)])
        assert resp.message.content == "ok"
        assert captured["body"]["model"] == "deepseek-v4-flash-vision-exp"
        content = captured["body"]["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    asyncio.run(run())


# ---------------- routes_search：组装与校验 ----------------
def test_build_user_content():
    # 纯文本 → 原样字符串（历史消息/无图场景不变）
    assert _build_user_content("你好", None) == "你好"
    assert _build_user_content("你好", []) == "你好"
    # 带图 → OpenAI 兼容块列表（text + image_url）
    blocks = _build_user_content("看图", ["data:image/png;base64,AAAA"])
    assert blocks[0] == {"type": "text", "text": "看图"}
    assert blocks[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_mm_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _png_data_url() -> str:
    # 1x1 透明 PNG
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def test_chat_with_image_rejected_for_bad_mime(client):
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    sid = s["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "看图", "images": ["data:image/bmp;base64,AAAA"]})
    assert r.status_code == 400
    assert "JPEG/PNG/GIF/WebP" in r.json()["detail"]


def test_chat_with_image_rejected_too_many(client):
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    sid = s["id"]
    imgs = [_png_data_url() for _ in range(11)]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "看图", "images": imgs})
    assert r.status_code == 400
    assert "10" in r.json()["detail"]


def test_chat_with_image_streams_ok_on_fake(client):
    """带图请求在 fake 模型下全链路不崩：SSE 正常产出 done（多模态组装不阻断 Agent 循环）。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    sid = s["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "这张图里是什么？", "images": [_png_data_url()]})
    assert r.status_code == 200
    body = r.text
    assert "done" in body
