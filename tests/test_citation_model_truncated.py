"""B2（2026-09-29）：引用卡片如实标注「模型所见已截断」。

## 被修的缺口

- **引用文本**取自工具的**完整**输出（`routes_search` 的 `tool_result` 处理）。
- **注入模型**的是 `retrieve` 的 **200 字/段**压缩稿（`loop._prepare_tool_output`
  → `_compact_retrieve_output`）；`pdf_extract` 才是全文。
⇒ 两者**不同源**：模型可能"引用"它**并未看全**的段落，而用户看到引用卡片
会以为模型是基于全文作答的。这是**判据与事实不同源**（与 B2 审查结论一致）。

## 处置（方案 A：如实标注，不改变任何注入内容）

给引用打一个 `model_truncated` 标记 → 经 SSE `message` 投影到前端 → 卡片显示 muted 的
「节选」标记 + 脚注说明。**只加标记字段，不改提示词、不改检索、不改回答**。

## 为什么判据要复用 `model_injection_truncates`（而不是在路由里写 `len(text) > 200`）

若路由另写一份长度判断（或按 `output["source"] == "rag"` 近似），就与
`_prepare_tool_output` 的判据（按 `name == "retrieve"` + `_RETRIEVE_MAX_TEXT`）
**不同源** ⇒ 又是一把尺子。本文件同时钉住"两者判据一致"。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.core.loop import _RETRIEVE_MAX_TEXT, model_injection_truncates
from lightcore.rag.pipeline import RetrievedChunk

DOC = "量子纠缠是量子力学中一种非局域关联现象。对其中一个粒子测量会瞬间决定另一个的状态。"


# ------------------------------------------------ ① 判据（与注入逻辑同源）


@pytest.mark.parametrize("tool,text,expected", [
    # retrieve 走 200 字/段压缩 ⇒ 超过即"模型所见已截断"
    ("retrieve", "x" * (_RETRIEVE_MAX_TEXT + 1), True),
    ("retrieve", "x" * _RETRIEVE_MAX_TEXT, False),          # 恰好等于上限 ⇒ 未截断
    ("retrieve", "x" * 10, False),
    ("retrieve", None, False),
    # pdf_extract 是**全文**注入 ⇒ 多长都不算截断
    ("pdf_extract", "x" * 5000, False),
    ("", "x" * 5000, False),
])
def test_model_injection_truncates_contract(tool, text, expected):
    """★ 判据必须与 `_prepare_tool_output` 一致：只有 `retrieve` 会截断。"""
    assert model_injection_truncates(tool, text) is expected


def test_criterion_matches_compaction_result():
    """★ **同源断言**：判据说"会截断"时，`_compact_retrieve_output` 必须真的变短了。

    这条把"标记"与"实际注入内容"绑在一起 —— 若哪天改了压缩阈值却忘了改判据，它会红。
    """
    from lightcore.core.loop import _compact_retrieve_output

    long_text = "长" * (_RETRIEVE_MAX_TEXT + 50)
    out = _compact_retrieve_output({"results": [{"id": "c1", "text": long_text}]})
    injected = out["results"][0]["text"]
    assert model_injection_truncates("retrieve", long_text) is True
    assert len(injected) < len(long_text), "判据说会截断，实际却没截断（判据与实现脱节）"


# ------------------------------------------------ ② 端到端：标记出现在 SSE


class _LongSegmentPipe:
    """>200 字命中 ⇒ 注入模型时会被截断。"""

    def get_chunks(self):  # noqa: ANN201
        return [RetrievedChunk(id="doc-9-c0", text="正文", page=1, score=0.5)]

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        return [RetrievedChunk(id="doc-9-c1", text="这是一段足够长的命中正文内容。" * 40,
                               page=1, score=0.9)][:top_k]


class _ShortSegmentPipe:
    """≤200 字命中 ⇒ 模型看到的就是全文，不应标注。"""

    def get_chunks(self):  # noqa: ANN201
        return [RetrievedChunk(id="doc-9-c0", text="正文", page=1, score=0.5)]

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        # ⚠️ 必须 ≥16 字（`clean_citations` 的下限）且 ≤ `_RETRIEVE_MAX_TEXT`（不被截断）
        return [RetrievedChunk(id="doc-9-c2", text="这是一句较短但足够长的命中正文内容。",
                               page=1, score=0.9)][:top_k]


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_b2_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _prepare(client, monkeypatch, pipe) -> int:
    doc_svc = client.app.state.doc_svc
    sid = client.post("/api/v1/sessions", json={"title": "b2"}).json()["id"]
    files = {"file": ("doc.md", DOC.encode(), "text/markdown")}
    assert client.post(f"/api/v1/documents?session_id={sid}", files=files).status_code == 200
    client.patch(f"/api/v1/sessions/{sid}", json={"config": {"model": "fake"}})

    async def _fake_get_pipeline(_doc_id, **_kw):
        return pipe

    monkeypatch.setattr(doc_svc, "get_pipeline", _fake_get_pipeline)
    return sid


def _message_citations(client, sid: int) -> list[dict]:
    """取 SSE 里最终 `message` 帧的 citations（带 text 的那个，非 delta）。"""
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "文档里的量子纠缠是什么？"})
    assert r.status_code == 200, r.text
    for block in r.text.split("\n\n"):
        if not block.startswith("event: message"):
            continue
        data_line = next((ln for ln in block.splitlines() if ln.startswith("data: ")), None)
        if not data_line:
            continue
        payload = json.loads(data_line[len("data: "):])
        if payload.get("citations"):
            return payload["citations"]
    return []


def test_long_segment_citation_is_marked_model_truncated(client, monkeypatch):
    """★ 主用例：>200 字的检索片段 ⇒ 卡片侧必须能知道"模型只看了开头"。"""
    sid = _prepare(client, monkeypatch, _LongSegmentPipe())
    cits = _message_citations(client, sid)
    assert cits, "没有拿到引用"
    assert cits[0].get("model_truncated") is True, (
        f"长片段未被标注（模型所见已截断）：{cits[0].get('model_truncated')}")


def test_short_segment_citation_is_not_marked(client, monkeypatch):
    """反向：≤200 字的片段，模型看到的就是全文 ⇒ **不得**误标（否则标记会失去意义）。"""
    sid = _prepare(client, monkeypatch, _ShortSegmentPipe())
    cits = _message_citations(client, sid)
    assert cits, "没有拿到引用"
    assert cits[0].get("model_truncated") is False, (
        f"短片段被误标为已截断：{cits[0].get('model_truncated')}")


def test_tool_result_event_carries_tool_name(client, monkeypatch):
    """★ 判据依赖的工具名必须真的在 `tool_result` 事件里（否则消费方只能改用近似判据）。"""
    sid = _prepare(client, monkeypatch, _ShortSegmentPipe())
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "文档里的量子纠缠是什么？"})
    assert r.status_code == 200, r.text
    names = []
    for block in r.text.split("\n\n"):
        if block.startswith("event: tool_result"):
            data_line = next((ln for ln in block.splitlines() if ln.startswith("data: ")), None)
            if data_line:
                names.append(json.loads(data_line[len("data: "):]).get("name"))
    assert "retrieve" in names, f"tool_result 事件没带工具名：{names}"
