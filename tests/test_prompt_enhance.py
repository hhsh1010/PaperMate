"""提示词增强测试（routes_prompt / POST /prompt/enhance）

三硬约束核验：①后端 API 保留（路由恒挂载）；②开关 features.prompt_enhance 关→403；
③fake 模型优雅降级（applied=False 返回原文，不臆造）。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from papermate import config as cfg_mod
from papermate import runtime_settings
from papermate.api.routes_prompt import extract_enhanced
from papermate.services import chat_service as chat_svc_mod

TEXT = "帮我看看这篇文章讲得怎么样"


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_pe_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    # 进程级单例复位须在 mock with 块外（runtime_settings 污染隔离）
    runtime_settings._state = None
    cfg_mod._settings = None


class _Resp:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})()


class _StubLLM:
    """可返回预设文本的「真实模型」替身（非 FakeLLM → is_real_llm=True）。"""

    def __init__(self, content: str):
        self._content = content

    async def chat(self, messages, **kwargs):  # noqa: ANN001, ARG002
        return _Resp(self._content)


# ---------- 提取函数 ----------
class TestExtractEnhanced:
    def test_full_tag(self):
        assert extract_enhanced("开头\n<enhanced-prompt>\n增强结果A\n</enhanced-prompt>", "原") == "增强结果A"

    def test_no_tag_falls_back_to_original(self):
        # 无标记不信任模型裸输出：回退原文，避免把模型废话灌进输入框
        assert extract_enhanced("模型直接输出的一段废话", "原始输入") == "原始输入"

    def test_empty_tag_falls_back(self):
        assert extract_enhanced("<enhanced-prompt>   </enhanced-prompt>", "原") == "原"

    def test_empty_model_output_falls_back(self):
        assert extract_enhanced("", "原") == "原"


# ---------- API ----------
class TestPromptEnhanceAPI:
    def test_feature_off_returns_403(self, client):
        runtime_settings.update({"features": {"prompt_enhance": False}})
        resp = client.post("/api/v1/prompt/enhance", json={"text": TEXT})
        assert resp.status_code == 403

    def test_empty_text_returns_400(self, client):
        resp = client.post("/api/v1/prompt/enhance", json={"text": "   "})
        assert resp.status_code == 400

    def test_fake_model_degrades_gracefully(self, client):
        """默认 fake：不调模型，返回原文 + applied=False + reason（绝不臆造）。"""
        resp = client.post("/api/v1/prompt/enhance", json={"text": TEXT})
        assert resp.status_code == 200
        data = resp.json()
        assert data["enhanced"] == TEXT and data["applied"] is False
        assert data["reason"]  # 有降级原因说明

    def test_stub_real_model_extracts_tag(self, client, monkeypatch):
        out = f"<enhanced-prompt>{TEXT}，请逐条给出优点与不足，并标注页码依据</enhanced-prompt>"
        monkeypatch.setattr(chat_svc_mod.ChatService, "resolve_llm",
                            lambda self, session_id=None: _StubLLM(out))
        resp = client.post("/api/v1/prompt/enhance", json={"text": TEXT})
        assert resp.status_code == 200
        data = resp.json()
        assert data["applied"] is True
        assert "逐条给出优点与不足" in data["enhanced"]

    def test_unconformant_output_not_trusted(self, client, monkeypatch):
        """模型没按标记输出：不信任裸文本，回退原文（宁缺毋滥）。"""
        monkeypatch.setattr(chat_svc_mod.ChatService, "resolve_llm",
                            lambda self, session_id=None: _StubLLM("直接输出没有标记的增强"))
        resp = client.post("/api/v1/prompt/enhance", json={"text": TEXT})
        data = resp.json()
        assert data["applied"] is False
        assert data["enhanced"] == TEXT

    def test_llm_exception_degrades(self, client, monkeypatch):
        class _Boom:
            async def chat(self, messages, **kwargs):  # noqa: ANN001, ARG002
                raise RuntimeError("boom")

        monkeypatch.setattr(chat_svc_mod.ChatService, "resolve_llm",
                            lambda self, session_id=None: _Boom())
        resp = client.post("/api/v1/prompt/enhance", json={"text": TEXT})
        assert resp.status_code == 200
        data = resp.json()
        assert data["applied"] is False and data["enhanced"] == TEXT
