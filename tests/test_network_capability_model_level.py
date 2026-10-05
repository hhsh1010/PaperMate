"""联网能力判定：**模型级**粒度 + 端点字段语义（2026-09-17）。

背景（用户报「联网提示还是会闪烁出现，无法正常显示」时揪出的两个缺陷）：

1. **能力判定粒度错**：`supports_native_web_search()` 原先只看厂商
   （`endpoint.name == "qwen"`），于是同厂内**实测不支持**的 `qwen3-vl-flash` 也被判为支持 →
   `GET /sessions/{id}/network-capability` 返回 `supports_native=true` →
   前端「开关已开但能力不可用」的预警**永不触发**（用户以为联网生效，实际不搜）。
   **能力判定必须细到模型**：同厂内差异真实存在（vision 版尤其容易漏）。

2. **`available` 语义容易被误用**：`available = enabled and supports` 表达的是
   「**当前是否真的可用**」，**不是**「能力是否具备」—— 开关关着时它同样是 `false`。
   前端若用 `!available` 判断能力，就会把"开关关着"误读成"模型不支持"，
   进而显示一句**错误归因**的提示（2026-09-17 实测闪现约 50ms 的正是这句）。
   故这里把语义钉死：`supports_native` 才是纯能力，与 `enabled` 相互独立。

判据方法论（别用错尺子）：联网能力必须用「今天几号」这类**只有联网才知道**的硬事实来测，
不能拿模型本来就会的知识题当判据 —— 否则分不清「联网查到的」与「训练时就会的」。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.llm.router import NATIVE_WEB_SEARCH_MODELS, ModelEndpoint, OAICompatibleLLM


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_netcap_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _qwen(model: str) -> OAICompatibleLLM:
    return OAICompatibleLLM(
        ModelEndpoint(
            "qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", model
        )
    )


def _set_cfg(client, session_id: int, **config) -> None:
    r = client.patch(f"/api/v1/sessions/{session_id}", json={"config": config})
    assert r.status_code == 200, r.text


def _capability(client, session_id: int) -> dict:
    r = client.get(f"/api/v1/sessions/{session_id}/network-capability")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------- ① 模型级判定 ----------------


def test_supports_native_web_search_is_model_level_not_provider_level():
    """同厂内必须区分模型：qwen3-vl-flash 实测**不支持**联网。

    旧实现是 `endpoint.name == "qwen"` → 本用例会红。这正是"预警永不触发"的根因。
    """
    assert _qwen("qwen3-vl-flash").supports_native_web_search() is False, (
        "qwen3-vl-flash 实测不支持原生联网；接受厂商级判定会让前端预警永不触发"
    )


def test_measured_supported_models_pass():
    """实测支持清单内的模型（判据 = 能答出「今天几号」这类只有联网才知道的硬事实）。"""
    for m in ("qwen3-max", "qwen3.6-flash", "qwen3.7-flash", "qwen-turbo", "qwen-max"):
        assert m in NATIVE_WEB_SEARCH_MODELS, f"{m} 应在实测支持清单内"
        assert _qwen(m).supports_native_web_search() is True, m


def test_non_qwen_provider_always_unsupported():
    """DeepSeek / OpenAI 的 chat/completions 不接受 enable_search（实测 DeepSeek 报 400）。"""
    for name, model in (("deepseek", "deepseek-v4-flash-vision-exp"), ("openai", "gpt-x")):
        ep = ModelEndpoint(name, "https://example.invalid/v1", "K", model)
        assert OAICompatibleLLM(ep).supports_native_web_search() is False, name


def test_unknown_qwen_model_defaults_to_unsupported():
    """白名单制：未实测过的模型保守视为不支持。

    **方向是刻意的**：多提示一句（用户可关闭、关掉开关即消除）好过漏报
    （用户以为联网生效、实际不搜 —— 那才是端点存在要防的事）。
    """
    assert _qwen("qwen3-future-unknown").supports_native_web_search() is False


# ---------------- ② 端点字段语义（前端判据依赖它） ----------------


def test_capability_reports_unsupported_model_when_switch_on(client):
    """开关已开 + 模型不支持 → `supports_native=False`、`available=False`、reason 指明模型名。"""
    s = client.post("/api/v1/sessions", json={"title": "cap"}).json()
    _set_cfg(client, s["id"], model="qwen3-vl-flash", enable_network=True)

    cap = _capability(client, s["id"])

    assert cap["supports_native"] is False
    assert cap["available"] is False
    assert cap["effective_model"] == "qwen3-vl-flash", "不再静默换模型"
    assert "qwen3-vl-flash" in cap["reason"], f"reason 会直接展示给用户，必须指明模型：{cap['reason']}"


def test_available_false_does_not_mean_incapable(client):
    """**关键语义**：开关关着时 `available` 也是 False，但 `supports_native` 仍为 True。

    前端因此**不得**用 `!available` 判断"能力不具备" —— 那会把"开关关着"误读成
    "模型不支持"，显示一句错误归因的提示（2026-09-17 实际闪过的就是这句）。
    """
    s = client.post("/api/v1/sessions", json={"title": "cap2"}).json()
    _set_cfg(client, s["id"], model="qwen3-max", enable_network=False)

    cap = _capability(client, s["id"])

    assert cap["enabled"] is False
    assert cap["available"] is False  # 因为开关关着
    assert cap["supports_native"] is True  # ← 但能力是具备的（两者独立）
    assert cap["reason"] == "", "开关关着不构成「不可用」，不该给 reason（否则前端会错误归因）"


def test_enabled_and_supported_model_is_available(client):
    s = client.post("/api/v1/sessions", json={"title": "cap3"}).json()
    _set_cfg(client, s["id"], model="qwen3-max", enable_network=True)

    cap = _capability(client, s["id"])

    assert cap["available"] is True
    assert cap["supports_native"] is True
    assert cap["reason"] == ""


def test_capability_404_for_missing_session(client):
    assert client.get("/api/v1/sessions/999999/network-capability").status_code == 404
