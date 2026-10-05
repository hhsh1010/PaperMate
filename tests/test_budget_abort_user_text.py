"""④ 路由级 e2e（2026-10-02）：预算熔断时，**用户看到的那条回答**必须归因正确。

## 为什么必须做路由级 e2e（而不是只测单元 + 源码契约）

本项守护的用户可见行为是"**熔断时气泡里写的是'预算熔断'而不是'可能是模型调用失败'**"。
这条链有三段，任何一段断了用户都会重新被误导：
`loop 发 error 事件（带 failure_kind）` → `路由捕获 kind` → `落库文案按 kind 分流`。
反向自检实测：只把路由的 kind 分流去掉时，**只有源码契约用例会红** ——
说明单靠单元测试仍有"链断了但没红"的缝隙，故补这条端到端。

## 做法

把运行时设置里的 `max_tokens_budget` 调到允许的最小值（1000），
上传一份文档（检索结果会撑大 prompt），再问一句 —— 于是**单次上下文超限**必然触发。
断言两条落库产物：
① 回答正文（气泡）含「预算熔断」且**不含**误导性的「账户余额 / 换个模型」；
② `meta.notice.reason_kind == "budget_exceeded"`（结构化归因落库，刷新后仍可见）。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from papermate import runtime_settings

DOC = "量子纠缠是量子力学中一种非局域关联现象。\n\n" + "两个粒子发生纠缠后测量其中一个会瞬间决定另一个的状态。" * 40


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_budget_notice_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    runtime_settings._STATE = None  # type: ignore[attr-defined]  清缓存，确保读临时目录
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


def test_budget_abort_text_is_actionable_not_misleading(client, monkeypatch):
    # ① 预算调到最小值（设置校验下限 1000）
    runtime_settings.update({"max_tokens_budget": 1000})

    s = client.post("/api/v1/sessions", json={"title": "budget-abort"}).json()
    files = {"file": ("doc.md", DOC.encode("utf-8"), "text/markdown")}
    up = client.post(f"/api/v1/documents?session_id={s['id']}", files=files)
    assert up.status_code == 200, up.text
    assert up.json()["status"] == "ready"

    r = client.post(f"/api/v1/sessions/{s['id']}/messages/stream",
                    json={"content": "文档里提到的量子纠缠是什么？"})
    assert r.status_code == 200, r.text
    sse = _parse_sse(r.text)

    # ② 落库的最后一条 assistant = 用户看到的气泡
    hist = client.get(f"/api/v1/sessions/{s['id']}").json()
    replies = [m for m in hist["messages"] if m["role"] == "assistant"]
    assert replies, f"失败轮次也必须落一条回答（绝不空白）：{hist}"
    text = replies[-1]["content"]

    # ★ 核心断言：归因正确、不误导
    assert "预算熔断" in text, f"熔断文案未按 kind 分流：{text[:200]}"
    for wrong in ("账户余额", "换个模型重试"):
        assert wrong not in text, f"仍在把用户引向「{wrong}」：{text[:200]}"

    # ③ 结构化归因落库（刷新后警示卡仍能给出正确方向）
    meta = replies[-1].get("meta") or {}
    notice = meta.get("notice") or {}
    assert notice.get("kind") == "stream_failed", "前端依赖 stream_failed 渲染警示卡"
    assert notice.get("reason_kind") == "budget_exceeded", f"结构化归因缺失：{notice}"
    assert any(n == "error" for n, _ in sse), "失败轮次必须发 error 事件"
