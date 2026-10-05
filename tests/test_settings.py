"""运行时设置端点：GET/PUT /api/v1/settings（前端填 key 的落盘与脱敏，docs/04 §4.6）

覆盖：
- GET 默认只返回脱敏视图，绝不泄露真实 key；
- PUT 保存后 GET 仍脱敏，且真实 key 已落盘本地文件；
- PUT 只更新出现的字段，不传的 provider/字段保持不变（不清空已保存的 key）。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_settings_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None  # 重置运行时缓存，确保用临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_get_defaults_no_secrets(client):
    r = client.get("/api/v1/settings")
    assert r.status_code == 200
    data = r.json()
    assert data["default_model"] == "fake"
    for name in ("deepseek", "qwen"):  # 2026-08-25：openai 已下线
        assert name in data["providers"]
        assert data["providers"][name]["configured"] is False
        assert data["providers"][name]["key_masked"] == ""
        assert "api_key" not in data["providers"][name]  # 不泄露真实 key


def test_put_and_get_roundtrip_masks_key(client):
    r = client.put(
        "/api/v1/settings",
        json={
            "default_model": "deepseek",
            "providers": {"deepseek": {"api_key": "sk-abcdefghijklmnop", "base_url": ""}},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["default_model"] == "deepseek"
    assert body["providers"]["deepseek"]["configured"] is True
    assert body["providers"]["deepseek"]["key_masked"].startswith("sk-a")
    assert body["providers"]["deepseek"]["key_masked"].endswith("mnop")
    assert "sk-abcdefghijklmnop" not in str(body)  # 真实 key 不返回

    g = client.get("/api/v1/settings").json()
    assert g["providers"]["deepseek"]["configured"] is True

    # 真实 key 已落盘（本地文件，等价 .env）
    f = os.path.join(os.environ["PAPERMATE_DATA_DIR"], "provider_settings.json")
    saved = json.loads(open(f, encoding="utf-8").read())
    assert saved["providers"]["deepseek"]["api_key"] == "sk-abcdefghijklmnop"


def test_put_only_updates_provided_fields(client):
    client.put("/api/v1/settings", json={"providers": {"deepseek": {"api_key": "sk-dddddddddddd"}}})
    # 只更新 qwen，不传 deepseek → deepseek 应保持已配置（2026-08-25：openai 已下线）
    client.put("/api/v1/settings", json={"providers": {"qwen": {"api_key": "sk-qqqqqqqqqqqq"}}})
    g = client.get("/api/v1/settings").json()
    assert g["providers"]["deepseek"]["configured"] is True
    assert g["providers"]["qwen"]["configured"] is True


def test_put_model_persisted_and_available_models_exposed(client):
    r = client.put(
        "/api/v1/settings",
        json={"providers": {"deepseek": {"api_key": "sk-abcdefghijklmnop", "model": "deepseek-reasoner"}}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["providers"]["deepseek"]["model"] == "deepseek-reasoner"
    # GET 暴露可选模型目录，供前端「设 key 后选模型」（2026-08-25：deepseek-chat/reasoner 已随
    # 官方 V4 发布退役，改为 v4-flash + vision-exp）
    assert "deepseek-v4-flash" in body["providers"]["deepseek"]["available_models"]
    assert "deepseek-v4-flash-vision-exp" in body["providers"]["deepseek"]["available_models"]
    assert "deepseek-chat" not in body["providers"]["deepseek"]["available_models"]


def test_test_endpoint_no_key_reports_error(client):
    # qwen 未配置 key → 测试连接应返回 ok=False 且给出明确原因（不抛 500）
    r = client.post("/api/v1/settings/test", json={"provider": "qwen"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "error" in body


def test_test_endpoint_invalid_key_reports_error(client):
    # 配置一个无效 key，测试连接应返回 ok=False（401 或网络错误均算明确失败，绝不静默）
    client.put("/api/v1/settings", json={"providers": {"deepseek": {"api_key": "sk-invalidjunkkey"}}})
    r = client.post("/api/v1/settings/test", json={"provider": "deepseek"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "error" in body
