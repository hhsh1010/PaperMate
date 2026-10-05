"""doc-bound 会话模型策略：**不回落，直接阻断**（2026-09-16 用户决策）

历史：2026-09-03 为根治 qwen3-vl-flash 绑文档不检索（上下文依赖型工具服从缺口），
引入了**静默回落**——doc-bound 会话自动把请求换成 deepseek-v4-flash-vision-exp。

实测危害（2026-09-16）：用户选了 Qwen（qwen3-vl-flash），请求却被发给 DeepSeek，
于是报"DeepSeek 余额不足（402）"—— 用户完全无法自救（他以为在用 Qwen）。

本文件验证**新策略**（离线，不触发真实网络）：
- test 1：doc-bound + 不可靠模型 → effective_model **原样返回**（绝不换模型）+ risk 给出可操作提示
- test 2：doc-bound + 可靠模型 → 无冲突
- test 3：非 doc-bound + qwen3-vl-flash → 无冲突（多模态自由使用，不阻断）
- test 4：对话**放行**（真的走检索循环）+ 风险落在回答上（meta.model_risk），且不污染 invalid 口径
- test 5：network-capability 暴露 model_risk，供前端常驻提示
- test 6：doc-bound + fake 模型 → 正常走 agent（风险提示不误伤可靠模型）
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.llm.fake import FakeLLM


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_dbfb_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


DOC = "量子纠缠是量子力学中一种非局域关联现象。对其中一个粒子测量会瞬间决定另一个的状态。"


def _upload(client, text: str, session_id: int) -> dict:
    files = {"file": ("doc.md", text.encode("utf-8"), "text/markdown")}
    r = client.post(f"/api/v1/documents?session_id={session_id}", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def _set_model(client, session_id: int, model: str) -> None:
    r = client.patch(f"/api/v1/sessions/{session_id}", json={"config": {"model": model}})
    assert r.status_code == 200, r.text


def _svc(client):
    return client.app.state.chat_svc


def _sse_events(text: str) -> list[dict]:
    out = []
    for ln in text.splitlines():
        if ln.startswith("data:"):
            try:
                out.append(json.loads(ln[5:].strip()))
            except json.JSONDecodeError:
                pass
    return out


def test_doc_bound_unreliable_model_never_swapped_but_blocked(client):
    """核心：**绝不静默换模型**；不可用必须给出含推荐模型的、可操作的原因。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    _upload(client, DOC, s["id"])
    _set_model(client, s["id"], "qwen3-vl-flash")
    svc = _svc(client)

    # ① effective_model 恒等于用户所选（旧行为：返回 deepseek-v4-flash-vision-exp）
    assert svc.effective_model(s["id"]) == "qwen3-vl-flash"
    # ② 冲突原因可操作：说明"不可用" + 给出推荐模型
    risk = svc.doc_bound_model_risk(s["id"])
    assert risk, "doc-bound + 不可靠模型必须给出风险提示"
    assert "qwen3-vl-flash" in risk
    assert "qwen3-max" in risk, "提示必须包含推荐模型，让用户一步自救"
    # ③ build_agent 也用原模型（旧断言 agent.llm.model_name == deepseek-… 已作废）
    agent = asyncio.run(svc.build_agent(s["id"]))
    assert agent.llm.model_name == "qwen3-vl-flash"


def test_reliable_model_no_conflict_in_doc_bound(client):
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    _upload(client, DOC, s["id"])
    _set_model(client, s["id"], "deepseek-v4-flash-vision-exp")
    svc = _svc(client)

    assert svc.effective_model(s["id"]) == "deepseek-v4-flash-vision-exp"
    assert svc.doc_bound_model_risk(s["id"]) == ""


def test_no_conflict_when_no_doc(client):
    """非 doc-bound + qwen3-vl-flash：多模态自由使用，不阻断。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    _set_model(client, s["id"], "qwen3-vl-flash")
    svc = _svc(client)

    assert svc.effective_model(s["id"]) == "qwen3-vl-flash"
    assert svc.doc_bound_model_risk(s["id"]) == ""


def test_stream_warns_but_does_not_block(client, monkeypatch):
    """策略 B（2026-09-16 晚定）：**警示但放行** —— 请求照常发出，风险落在该条回答上。

    与上一版（硬阻断）的关键差别：用户有权使用自己付费的模型，
    我们只负责让风险可见，不替他决定是否调用。
    """
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    _upload(client, DOC, s["id"])
    _set_model(client, s["id"], "qwen3-vl-flash")
    svc = _svc(client)

    # 离线替身：不真调网络，但**必须真的走 agent 循环**，否则测不到"放行"
    monkeypatch.setattr(svc, "_llm_from_raw", lambda raw: FakeLLM())

    r = client.post(
        f"/api/v1/sessions/{s['id']}/messages/stream",
        json={"content": "文档里量子纠缠是什么？"},
    )
    assert r.status_code == 200, r.text
    evs = _sse_events(r.text)

    # ① 不阻断：既没有 model_unavailable 错误，又真的走了检索循环
    assert [e for e in evs if e.get("code") == "model_unavailable"] == []
    names = [ln.split(":", 1)[1].strip() for ln in r.text.splitlines() if ln.startswith("event:")]
    assert "tool_call" in names, "必须真的调用检索（证明放行，而不是被拦在入口）"

    # ② 风险可见：落库的该条回答带 `model_risk`（刷新后仍看得到"要核对引用"）
    detail = client.get(f"/api/v1/sessions/{s['id']}").json()
    assistants = [m for m in detail["messages"] if m["role"] == "assistant"]
    assert assistants, "放行后应正常落库"
    assert (assistants[-1].get("meta") or {}).get("model_risk") is True
    # ③ 风险标记**不得**污染「无效回复」口径（否则会被升级为 loop_risk 强打断）
    assert (assistants[-1].get("meta") or {}).get("invalid") is not True
    assert not (assistants[-1].get("meta") or {}).get("notice")


def test_capability_exposes_model_risk(client):
    """network-capability 暴露 model_risk（前端据此常驻提示，发送不受阻）。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    _upload(client, DOC, s["id"])
    _set_model(client, s["id"], "qwen3-vl-flash")

    cap = client.get(f"/api/v1/sessions/{s['id']}/network-capability").json()

    assert cap["model_risk"] is True
    assert "qwen3-vl-flash" in cap["model_risk_reason"]
    assert "qwen3-max" in cap["model_risk_reason"]
    assert cap["effective_model"] == "qwen3-vl-flash", "不再静默换模型"


def test_doc_bound_fake_model_still_works(client):
    """阻断不误伤：可靠模型（fake）在 doc-bound 下照常走 agent。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    _upload(client, DOC, s["id"])
    _set_model(client, s["id"], "fake")
    svc = _svc(client)

    assert svc.doc_bound_model_risk(s["id"]) == ""
    r = client.post(
        f"/api/v1/sessions/{s['id']}/messages/stream",
        json={"content": "文档里量子纠缠是什么？"},
    )
    assert r.status_code == 200
    names = [ln.split(":", 1)[1].strip() for ln in r.text.splitlines() if ln.startswith("event:")]
    assert "tool_call" in names and "tool_result" in names
    assert "error" not in names
