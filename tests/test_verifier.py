"""P1-2.1 回复逐条核验测试：
- verify_l1：引用文本 ↔ 页码确定性校验（命中 / 错页 / 编造文本 / 无 id / 无页文本）
- verify_reply 开关：默认关闭 + settings 往返持久化
- SSE `verify` 事件：开启后 stream 响应含 verify（fake 模型下 L2 降级 skipped），
  且不改变既有 message 文本与事件流（关闭时行为与现状一致）
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from papermate.services.verifier import only_suspects, verify_l1, verify_reply


def _make_grid_pdf(path: str) -> None:
    """含 3×3 矢量边框表格的 PDF（页码确定 = 1，供 L1 校验）。"""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    x0, y0, x1, y1 = 50, 80, 250, 160
    page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=1)
    for dy in (25, 50):
        page.draw_line(pymupdf.Point(x0, y0 + dy), pymupdf.Point(x1, y0 + dy), color=(0, 0, 0), width=1)
    for dx in (66, 133):
        page.draw_line(pymupdf.Point(x0 + dx, y0), pymupdf.Point(x0 + dx, y1), color=(0, 0, 0), width=1)
    cells = [("Model", "Acc", "Params"), ("BERT", "92.1", "110M"), ("GPT", "88.4", "175B")]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            page.insert_text(pymupdf.Point(x0 + 5 + c * 66, y0 + 18 + r * 25), val, fontsize=10)
    doc.save(path)
    doc.close()


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_verify_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None  # 重置运行时缓存，确保用临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _upload_pdf(client) -> int:
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_grid_pdf(fpath)
        content = open(fpath, "rb").read()
    finally:
        os.unlink(fpath)
    r = client.post("/api/v1/documents", files={"file": ("grid.pdf", content, "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _chunk_page1(client, doc_id: int) -> str:
    """取文档第 1 页任一 chunk 文本（L1 校验的基准文本）。"""
    from papermate.api.app import create_app  # noqa: F401  确保 app 模块已加载

    app = client.app
    with app.state.chat_svc._Session() as db:
        from papermate.models.db import Chunk

        row = db.query(Chunk).filter(Chunk.document_id == doc_id, Chunk.page == 1).first()
        assert row is not None
        return row.content


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    cur_name: str | None = None
    cur_data = ""
    for line in text.splitlines():
        if line.startswith("event:"):
            cur_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            cur_data = line[len("data:"):].strip()
            if cur_name is not None:
                events.append((cur_name, json.loads(cur_data)))
                cur_name, cur_data = None, ""
    return events


# ---------------- verify_l1 确定性校验 ----------------
def test_l1_hit_citation_is_ok(client):
    doc_id = _upload_pdf(client)
    page_text = _chunk_page1(client, doc_id)
    with client.app.state.chat_svc._Session() as db:
        out = verify_l1(
            [{"id": f"doc-{doc_id}-c0", "page": 1, "text": page_text[:60]}],
            db,
        )
    assert out["status"] == "ok"
    assert out["issues"] == []


def test_l1_wrong_page_is_suspect(client):
    doc_id = _upload_pdf(client)
    page_text = _chunk_page1(client, doc_id)
    with client.app.state.chat_svc._Session() as db:
        out = verify_l1(
            [{"id": f"doc-{doc_id}-c0", "page": 99, "text": page_text[:60]}],
            db,
        )
    assert out["status"] == "issues"
    assert any("第 99 页" in i["reason"] for i in out["issues"])


def test_l1_fabricated_text_is_suspect(client):
    doc_id = _upload_pdf(client)
    with client.app.state.chat_svc._Session() as db:
        out = verify_l1(
            [{"id": f"doc-{doc_id}-c0", "page": 1, "text": "这段内容是模型编造的，绝不在原文中"}],
            db,
        )
    assert out["status"] == "issues"
    assert any("未能在第 1 页找到" in i["reason"] for i in out["issues"])


def test_l1_missing_id_is_suspect(client):
    doc_id = _upload_pdf(client)
    page_text = _chunk_page1(client, doc_id)
    with client.app.state.chat_svc._Session() as db:
        out = verify_l1(
            [{"id": None, "page": 1, "text": page_text[:60]}],
            db,
        )
    assert out["status"] == "issues"
    assert any("无法定位" in i["reason"] for i in out["issues"])


def test_l1_empty_citations_ok(client):
    with client.app.state.chat_svc._Session() as db:
        out = verify_l1([], db)
    assert out["status"] == "ok"


def _upload_text_doc(client, text: str) -> int:
    """上传文本文档（parse_text → 块 page=None，模拟 docx 类无物理页码文档）。"""
    r = client.post("/api/v1/documents", files={"file": ("a.md", text.encode("utf-8"), "text/markdown")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _first_chunk(client, doc_id: int) -> tuple[str, int]:
    from papermate.models.db import Chunk

    app = client.app
    with app.state.chat_svc._Session() as db:
        row = db.query(Chunk).filter(Chunk.document_id == doc_id).first()
        assert row is not None
        return row.content, row.page


def test_l1_no_page_doc_with_zero_page_citation_warns(client):
    """无物理页码文档（docx/md/txt）引用 → warn「无页码可核验」（防御性：即便传入遗留 page=0 也视作无页）。"""
    doc_id = _upload_text_doc(client, "量子纠缠是量子力学中一种非局域关联现象。\n\n两个粒子纠缠后，测量一个会瞬间决定另一个。")
    text, page = _first_chunk(client, doc_id)
    assert page is None  # 确认文档块无物理页码（page=None，而非 0）
    with client.app.state.chat_svc._Session() as db:
        out = verify_l1([{"id": f"doc-{doc_id}-c0", "page": 0, "text": text[:40]}], db)
    assert out["status"] == "issues"
    assert any(i["severity"] == "warn" and "页码" in i["reason"] for i in out["issues"])


def test_l1_no_page_doc_with_null_page_citation_warns(client):
    doc_id = _upload_text_doc(client, "Transformer 通过自注意力实现并行化序列建模。")
    text, _ = _first_chunk(client, doc_id)
    with client.app.state.chat_svc._Session() as db:
        out = verify_l1([{"id": f"doc-{doc_id}-c0", "page": None, "text": text[:40]}], db)
    assert out["status"] == "issues"
    assert any(i["severity"] == "warn" and "页码" in i["reason"] for i in out["issues"])


def test_l1_pdf_citation_without_page_is_suspect(client):
    """有物理页码的 PDF 文档，引用却没标页码 → suspect（页码缺失/标注错误）。"""
    doc_id = _upload_pdf(client)
    page_text = _chunk_page1(client, doc_id)
    with client.app.state.chat_svc._Session() as db:
        out = verify_l1([{"id": f"doc-{doc_id}-c0", "page": None, "text": page_text[:60]}], db)
    assert out["status"] == "issues"
    assert any(i["severity"] == "suspect" and "未标注页码" in i["reason"] for i in out["issues"])


# ---------------- verify_reply 开关 ----------------
def test_verify_reply_default_off(client):
    data = client.get("/api/v1/settings").json()
    assert data["features"]["verify_reply"] is False


def test_verify_reply_roundtrip(client):
    r = client.put("/api/v1/settings", json={"features": {"verify_reply": True}})
    assert r.status_code == 200
    assert r.json()["features"]["verify_reply"] is True
    assert client.get("/api/v1/settings").json()["features"]["verify_reply"] is True
    # 关回默认
    client.put("/api/v1/settings", json={"features": {"verify_reply": False}})
    assert client.get("/api/v1/settings").json()["features"]["verify_reply"] is False


# ---------------- SSE verify 事件 ----------------
def test_stream_emits_verify_when_enabled(client):
    client.put("/api/v1/settings", json={"features": {"verify_reply": True}})
    sid = client.post("/api/v1/sessions", json={"title": "核验会话"}).json()["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    names = [n for n, _ in events]
    assert "message" in names
    assert "verify" in names  # verify 事件已发出
    v = next(d for n, d in events if n == "verify")
    assert v["level"] >= 1
    # fake 模型下 L2 解析失败 → 降级 skipped，但事件仍携带（L1 结果）
    assert v.get("l2") == "skipped" or v["status"] in ("ok", "issues")
    # verify 在 done 之前（保序），且 message 文本未被改动
    assert names.index("verify") < names.index("done")
    msg = next(d for n, d in events if n == "message")
    assert msg["text"]  # 非空


def test_stream_no_verify_when_disabled(client):
    # 默认关闭（fixture 重置）→ 事件流与现状一致，无 verify
    sid = client.post("/api/v1/sessions", json={"title": "无核验会话"}).json()["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    assert r.status_code == 200, r.text
    names = [n for n, _ in _parse_sse(r.text)]
    assert "verify" not in names
    assert "message" in names and "done" in names


# ---------------- verify_reply 顶层（含 L2 降级路径） ----------------
def test_verify_reply_fake_llm_l2_skipped(client):
    doc_id = _upload_pdf(client)
    page_text = _chunk_page1(client, doc_id)
    from lightcore.llm.fake import FakeLLM

    with client.app.state.chat_svc._Session() as db:
        out = asyncio_run(
            verify_reply(
                user_query="第 1 页的表格说了什么",
                final_text="第 1 页的表格对比了 BERT 与 GPT 的准确率。",
                citations=[{"id": f"doc-{doc_id}-c0", "page": 1, "text": page_text[:60]}],
                db=db,
                llm=FakeLLM(),
            )
        )
    assert out["level"] == 2
    assert out["l2"] == "skipped"  # fake 输出非 JSON → 降级
    assert out["status"] == "ok"  # L1 通过


# ---------------- P0（2026-09-16）：L1 常开 + 低噪下发 ----------------
#
# 背景：L1（确定性、零 LLM、零延迟，引用↔页码比对）此前被 `features.verify_reply`
# ——一个**为省 L2 的 LLM 调用而设的开关**——**连坐关闭**：最便宜的一道防线，
# 跟着最贵的那道一起被关掉了。现改为 L1 常开，但必须同时解决噪音问题。


def test_only_suspects_filters_warn_and_keeps_suspect():
    """策略函数：只留 `suspect`（确凿问题），`warn` 静音；无 suspect → None（不必下发）。"""
    warn_only = {"level": 1, "status": "issues",
                 "issues": [{"severity": "warn", "reason": "无页码可核验"}]}
    assert only_suspects(warn_only) is None

    mixed = {"level": 1, "status": "issues", "issues": [
        {"severity": "warn", "reason": "无页码可核验"},
        {"severity": "suspect", "reason": "引用文本未能在第 2 页找到"},
    ]}
    out = only_suspects(mixed)
    assert out is not None
    assert out["level"] == 1 and out["status"] == "issues"
    assert [i["severity"] for i in out["issues"]] == ["suspect"]

    # 边界：ok / 缺 issues 键 / issues 为 None —— 不得抛异常
    assert only_suspects({"level": 1, "status": "ok", "issues": []}) is None
    assert only_suspects({"level": 1, "status": "ok"}) is None
    assert only_suspects({"status": "issues", "issues": None}) is None


def test_suspect_finding_is_delivered_with_switch_off(client):
    """**常开的核心价值**：开关关闭（默认）时，L1 抓到确凿问题**仍要下发**。"""
    doc_id = _upload_pdf(client)
    with client.app.state.chat_svc._Session() as db:
        out = asyncio_run(verify_reply(
            user_query="第 1 页的表格说了什么？",
            final_text="第 1 页给出了模型准确率对比。",
            # 引用文本是编造的（该 PDF 第 1 页没有这段）→ L1 应判 suspect
            citations=[{"id": f"doc-{doc_id}-c0", "page": 1,
                        "text": "此处为伪造的引用文本，第 1 页并不存在这一段"}],
            db=db,
            llm=None,  # ← 开关关闭时的传参：L1 照跑、L2 跳过
        ))
    assert out["level"] == 1 and "l2" not in out, "llm=None 时不得执行 L2"
    report = only_suspects(out)
    assert report is not None and report["issues"], "确凿问题必须下发，不能被开关一起关掉"
    assert all(i["severity"] == "suspect" for i in report["issues"])


def test_warn_only_finding_is_silenced_with_switch_off(client):
    """噪音控制：无物理页码文档（docx/md）的「无法核验页码」属**正常情况** → 关闭时静音。"""
    doc_id = _upload_text_doc(client, "稀疏向量检索按词权重排序，与稠密向量互补。")
    text, page = _first_chunk(client, doc_id)
    assert page is None  # 确认该文档块无物理页码
    with client.app.state.chat_svc._Session() as db:
        out = asyncio_run(verify_reply(
            user_query="稀疏向量有什么用？",
            final_text="用于按词权重匹配。",
            citations=[{"id": f"doc-{doc_id}-c0", "page": None, "text": text[:40]}],
            db=db,
            llm=None,
        ))
    assert out["status"] == "issues"  # L1 确实报了（warn 级）
    assert only_suspects(out) is None, (
        "warn 级必须静音 —— 否则每个带引用的 docx 回复都会弹核验卡，用户很快学会忽略它"
    )


def test_run_verify_keeps_l1_on_when_switch_off():
    """结构回归：`_run_verify` 不得再因开关关闭而**直接返回**（那会把 L1 一起关掉）。

    该函数是 `stream_message` 内的闭包、无法直接调用，故用源码断言：
    必须看到「开关只决定是否传 `llm`」与「关闭时走 `only_suspects` 过滤」两处。
    """
    import inspect

    from papermate.api import routes_search as rs

    src = inspect.getsource(rs)
    start = src.index("async def _run_verify(")
    block = src[start:src.index("\n    # 组装本轮 user 消息", start)]

    assert 'l2_on = bool(_feat.get("verify_reply", False))' in block
    assert "llm=(llm if l2_on else None)" in block, "L2 必须只由开关控制（传 None 即跳过）"
    assert "return result if l2_on else only_suspects(result)" in block
    # 2026-09-23 更新：无引用分支的写法变了（先做「doc-bound 零引用」检查，再按开关决定返回），
    # 但**本条测试的核心意图丝毫未变**：开关关闭**不得**让 L1 提前返回。
    # 所以只更新断言形式，并**新增**对新行为的守护 —— 而不是删掉这条。
    assert "if not cites:" in block, "无引用分支必须存在（它是唯一的合法提前返回点）"
    assert "zero_citation_notice(" in block, "doc-bound 零引用提示必须接在这个分支里"
    # 反例守卫（保留原文）：原先「开关关闭 → 立刻 return None」的写法必须消失
    assert 'if not _feat.get("verify_reply", False):' not in block


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
