"""段落 AI 改写测试（routes_transform / POST /workbench/para-transform）

三硬约束核验：①后端 API 保留（路由恒挂载）；②开关 features.para_transform 关→403；
③fake 模型优雅降级（applied=False 返回原文，不臆造）。
另覆盖：非法 action 400 / 空文本 400 / 标记提取 / 真实路径（StubLLM）。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from papermate import config as cfg_mod
from papermate import runtime_settings
from papermate.api.routes_transform import extract_transformed

TEXT = "本文提出一种轻量化检测方法，在保持精度的同时降低了计算量。"


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_pt_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
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


def _patch_llm(monkeypatch, content: str):

    # create_app 后 app.state.chat_svc.resolve_llm 指向服务实例方法——monkeypatch 类方法
    from papermate.services.chat_service import ChatService

    monkeypatch.setattr(ChatService, "resolve_llm", lambda self, session_id=None: _StubLLM(content))


# ---------- 提取函数 ----------
def test_extract_transformed_full_tag():
    assert extract_transformed("前言\n<transformed>\n改写结果\n</transformed>", "原") == "改写结果"


def test_extract_transformed_no_tag_falls_back():
    assert extract_transformed("模型裸输出", "原始段落") == "原始段落"


# ---------- 端点行为 ----------
def test_para_transform_ok(client, monkeypatch):
    _patch_llm(monkeypatch, "<transformed>\n本文提出一种轻量化目标检测方法（改写版）。\n</transformed>")
    r = client.post(
        "/api/v1/workbench/para-transform",
        json={"text": TEXT, "action": "polish", "context_before": "上一段", "context_after": "下一段"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["applied"] is True
    assert "轻量化目标检测方法" in out["transformed"]
    assert out["action_label"] == "润色"


def test_para_transform_fake_llm_degrades(client):
    """默认 fake 模型：applied=False 返回原文（不臆造）。"""
    r = client.post("/api/v1/workbench/para-transform", json={"text": TEXT, "action": "expand"})
    assert r.status_code == 200
    out = r.json()
    assert out["applied"] is False
    assert out["transformed"] == TEXT
    assert out["reason"]


def test_para_transform_disabled_403(client, monkeypatch):
    monkeypatch.setattr(
        runtime_settings, "get_state", lambda: {"features": {"para_transform": False}}
    )
    r = client.post("/api/v1/workbench/para-transform", json={"text": TEXT, "action": "polish"})
    assert r.status_code == 403


def test_para_transform_invalid_action_and_empty(client):
    assert client.post(
        "/api/v1/workbench/para-transform", json={"text": TEXT, "action": "magic"}
    ).status_code == 400
    assert client.post(
        "/api/v1/workbench/para-transform", json={"text": "", "action": "polish"}
    ).status_code == 400
