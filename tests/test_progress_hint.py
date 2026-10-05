"""O-12 档 1（2026-10-02）：首字之前的进度提示 `progress` 事件。

**为什么需要**：doc-bound 首轮的首字延迟实测中位 **≈4.07s**
（LLM1 决策 1748ms + 检索 1488ms + LLM2 首 token 832ms，见
`docs/O-12-TTFT探针-首字构成实测.md`）；这段窗口内**零字节正文**，用户只看到一个空气泡。

**三条硬约束（缺一即"用错通道换观感"）**：
1. ★ **必须走独立事件**，不能发 `message_delta` —— 那会进 `partial_text` → 落库 → 污染回答正文与引用；
2. ★ **不能借用 `thinking`** —— 那会虚增前端「推理过程（N 步）」的计数（`steps` 也按 thinking 事件统计）；
3. ★ **只在"必然发生"时发** —— 即 `force_choice` 为真（会话已关联文档 ⇒ 协议级要求必须调工具）；
   无文档的直答路径**不得**发（否则是对用户撒谎："正在检索"而根本不检索）。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod

DOC = "量子纠缠是量子力学中一种非局域关联现象。\n\n两个粒子发生纠缠后，对其中一个测量会瞬间决定另一个的状态。"


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_progress_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    cur_name, cur_data = None, ""
    for line in text.splitlines():
        if line.startswith("event:"):
            cur_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            cur_data = line[len("data:"):].strip()
            if cur_name is not None:
                events.append((cur_name, json.loads(cur_data)))
                cur_name, cur_data = None, ""
    return events


def _stream(client, session_id: int, content: str) -> list[tuple[str, dict]]:
    r = client.post(f"/api/v1/sessions/{session_id}/messages/stream", json={"content": content})
    assert r.status_code == 200, r.text
    return _parse_sse(r.text)


def _upload(client, text: str, session_id: int) -> dict:
    files = {"file": ("doc.md", text.encode("utf-8"), "text/markdown")}
    r = client.post(f"/api/v1/documents?session_id={session_id}", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def test_progress_emitted_on_doc_bound_before_first_delta(client):
    """doc-bound：必须有 progress，且**排在首个 message_delta 之前**（否则毫无意义）。"""
    s = client.post("/api/v1/sessions", json={"title": "progress-doc"}).json()
    assert _upload(client, DOC, s["id"])["status"] == "ready"

    events = _stream(client, s["id"], "文档里提到的量子纠缠是什么？")
    names = [n for n, _ in events]

    assert "progress" in names, f"doc-bound 首轮必须发进度提示：{names}"
    first_progress = names.index("progress")
    first_delta = names.index("message_delta")
    assert first_progress < first_delta, f"进度提示必须在首个正文增量之前：{names}"

    # 载荷：phase 供程序判定、text 供展示
    payload = next(d for n, d in events if n == "progress")
    assert payload.get("phase") == "retrieving"
    assert payload.get("text"), "必须带可展示文案"
    # ★ 不得污染正文：progress 的文案不能出现在最终回答里
    final_text = next((d["text"] for n, d in events if n == "message"), "")
    assert "正在检索" not in final_text, "进度文案串进回答正文了（通道用错）"


def test_progress_sets_no_thinking_step(client):
    """★ 硬约束②：不得借用 `thinking` 通道 —— 否则虚增「推理过程（N 步）」。"""
    s = client.post("/api/v1/sessions", json={"title": "progress-nostep"}).json()
    assert _upload(client, DOC, s["id"])["status"] == "ready"

    events = _stream(client, s["id"], "文档里提到的量子纠缠是什么？")
    progress_payload = next(d for n, d in events if n == "progress")
    # progress 必须是**独立事件**（有自己的 type），而不是被塞进 thinking
    assert "step" not in progress_payload, "进度提示不得伪装成推理步骤"
    assert "reasoning" not in progress_payload


def test_no_progress_without_docs(client):
    """★ 硬约束③：无关联文档**不得**发 —— 否则是撒谎说"正在检索"。

    ⚠️ **本用例两次修正，记录在此以免后人再踩**：
    初版用"什么是注意力机制？"—— 那是 `general_qa` + 无文档 ⇒ 走**直答**路径，
    根本不经过发事件那段代码 ⇒ 反向自检**不红**（弱测试：绿得没道理）。
    二版改用"文档里提到的…"—— 命中 `has_document_reference` ⇒ 被**缺文档协议级拦截**
    （只回 reminder、不检索）⇒ 仍然到不了 Agent 路径。
    正解：用只命中 `_DOC_QA_MARKERS`（如"摘要"）而**不含任何强指代词**（"这篇/文中/文档里…"，
    它们都在 `_DOC_OWNED_MARKERS` 里、会被协议级拦截）的问法 ⇒
    `doc_qa` 意图 + 绕过拦截 + 无文档 ⇒ `force_choice=None` ⇒ **真·Agent 路径**。
    实测反向自检（去掉 `force_choice` 门控）**变红** ✓
    """
    s = client.post("/api/v1/sessions", json={"title": "no-doc"}).json()
    events = _stream(client, s["id"], "摘要部分通常包含哪些要素？")
    names = [n for n, _ in events]
    assert "progress" not in names, f"无文档却发了'正在检索'进度提示（在撒谎）：{names}"


def test_progress_emission_is_guarded_by_force_choice():
    """源码契约：progress 的发出必须以 `force_choice` 为条件（防有人改成无条件发）。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py").read_text(
        encoding="utf-8")
    assert 'if force_choice:\n            yield sse_event("progress"' in src, (
        "进度提示必须由 force_choice 门控（仅在'必然要检索'时才发）"
    )
