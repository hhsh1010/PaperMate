"""B8（2026-09-29）：**SSE 保活帧** —— 首个字节前的静默窗口不得静默。

## 被修的缺口

三个 SSE 生成器都包了 `_with_stream_cleanup`，但**全库无 keepalive**。
而**首个 SSE 字节之前**真实可能发生：**管线加载**（历史实测单篇可达 **33s**）+
**历史摘要**（≤20s）⇒ 该窗口内**零字节下行** ⇒ 代理/网关易判超时并断开。
本地单用户风险低，但属健壮性缺口。

## 为什么用 SSE **注释帧**（`: keepalive`）而不是一个自定义事件

1. **前端已原生忽略**：`web/src/lib/sse.ts:55` 有 `if (line.startsWith(":")) continue;`，
   且整帧只有注释时 `parseFrame` 得空对象 ⇒ 返回 `null` ⇒ **完全丢弃、不产生 UI 事件**
   （前端侧已由 `web/src/lib/sse.test.ts` 钉住）。
2. **不污染任何口径**：它不是 `LoopEvent` ⇒ **不进 `events`** ⇒ trace / 用量 / 引用都不受影响。
3. 若改发自定义事件（如 `event: ping`），前端 reducer 必须新增分支去忽略它 —— 那是**改协议 + 改前端**，
   成本与风险都远高于一个注释帧。

## 本文件在守什么

1. **首帧就是保活**（第一条字节即为 `: keepalive`）——"静默窗口被覆盖"的唯一可观测证据。
2. **不打扰下游**：保活帧之后 `done` 仍正常出现，帧解析不受影响。
3. **不进 trace**：`events_json` 里**不得**出现保活帧。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.rag.pipeline import RetrievedChunk
from papermate.models.db import Trace

DOC = "量子纠缠是量子力学中一种非局域关联现象。对其中一个粒子测量会瞬间决定另一个的状态。"

KEEPALIVE = ": keepalive\n\n"


class _HitPipe:
    """检索必然命中的管线。"""

    def get_chunks(self):  # noqa: ANN201  pdf_extract 需要
        return [RetrievedChunk(id="doc-9-c0", text="正文", page=1, score=0.5)]

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        return [RetrievedChunk(id="doc-9-c1", text="命中段落的正文内容足够长可以保留",
                               page=1, score=0.9)][:top_k]


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_b8_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _prepare(client, monkeypatch, pipe) -> int:
    doc_svc = client.app.state.doc_svc
    sid = client.post("/api/v1/sessions", json={"title": "b8"}).json()["id"]
    files = {"file": ("doc.md", DOC.encode(), "text/markdown")}
    assert client.post(f"/api/v1/documents?session_id={sid}", files=files).status_code == 200
    client.patch(f"/api/v1/sessions/{sid}", json={"config": {"model": "fake"}})

    async def _fake_get_pipeline(_doc_id, **_kw):
        return pipe

    monkeypatch.setattr(doc_svc, "get_pipeline", _fake_get_pipeline)
    return sid


def _ask(client, sid: int) -> str:
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "文档里的量子纠缠是什么？"})
    assert r.status_code == 200, r.text
    return r.text


def _frame_names(raw: str) -> list[str]:
    """按空行切块，取每块的 `event:` 行（无 event 的注释块返回空串）。"""
    out: list[str] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        line = next((ln for ln in block.splitlines() if ln.startswith("event:")), "")
        out.append(line[len("event:"):].strip() if line else "")
    return out


def test_keepalive_is_the_very_first_bytes(client, monkeypatch):
    """★ 主用例：响应的**第一段**必须是保活注释帧 —— 否则"静默窗口"根本没被覆盖。

    这是"首个 SSE 字节之前会静默"这一缺口的**唯一可观测证据**：只要首字节是保活帧，
    客户端/代理在管线加载（可达 33s）期间就**始终有字节下行**。
    """
    sid = _prepare(client, monkeypatch, _HitPipe())
    body = _ask(client, sid)
    assert body.startswith(KEEPALIVE), (
        f"首个字节不是保活帧（B8 缺陷本体）：{body[:40]!r}")


def test_keepalive_does_not_disturb_downstream_frames(client, monkeypatch):
    """保活帧**不得**干扰后续帧解析：`done` 仍正常出现，且保活块不产生事件名。"""
    sid = _prepare(client, monkeypatch, _HitPipe())
    body = _ask(client, sid)

    names = _frame_names(body)
    assert names and names[0] == "", f"首块应是注释帧（无 event 名）：{names[:3]}"
    assert "done" in names, f"保活帧之后 done 丢失：{names[-5:]}"


def test_keepalive_is_not_persisted_into_trace(client, monkeypatch):
    """★ 保活帧**不是** LoopEvent ⇒ 不得进 `events_json`（否则污染轨迹与用量口径）。"""
    sid = _prepare(client, monkeypatch, _HitPipe())
    _ask(client, sid)

    with client.app.state.chat_svc._Session() as s:  # noqa: SLF001
        t = s.query(Trace).filter(Trace.session_id == sid).order_by(Trace.id.desc()).first()
        assert t is not None, "本轮没有落 trace"
        evs = json.loads(t.events_json or "[]")
    assert evs, "trace 里没有事件"
    # 任何一条事件都不该是保活帧/空类型
    assert all(isinstance(e, dict) and e.get("type") for e in evs), \
        f"trace 里混入了非事件条目：{evs[:3]}"
    assert not any("keepalive" in json.dumps(e, ensure_ascii=False) for e in evs), \
        "保活帧被写进了 trace"
