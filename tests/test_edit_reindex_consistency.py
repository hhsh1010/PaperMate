"""改稿后检索一致性回归（2026-09-30，T-06 P0 修复的守卫）。

探针 `scripts/probe_edit_consistency.py` 证实（真实 BGE-M3）：apply-edits 只改 **docx 源文件**，
而 `_trigger_reindex` 原先调用的 `reindex()` 从 **DB chunks** 重建管线 —— chunks 未变 ⇒
signature 原样命中磁盘 state ⇒ **0.01s 秒回旧向量**，检索到的仍是**改稿前**文本（假生效）。

修复：`_trigger_reindex` 改调 `reprocess()`（完整重处理：重新解析 → 分块 → 嵌入）。
本文件把「**改稿后检索必须反映新内容**」钉死为回归契约：
- 主断言走 **chunks 表精确文本**（fake embedder 的语义相似度不可靠，不作断言依据）；
- 反向自检已做：把 `_job` 改回 `svc.reindex(doc_id)` ⇒ 本文件用例变红。
"""
from __future__ import annotations

import io
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
from docx import Document

import papermate.config as cfg_mod

SENT_A = "The quick brown fox jumps over the lazy dog near the river bank."
SENT_B = "Deep reinforcement learning controls autonomous drone swarms in crowded cities."


def make_docx_bytes(texts: list[str]) -> bytes:
    buf = io.BytesIO()
    d = Document()
    for t in texts:
        d.add_paragraph(t)
    d.save(buf)
    return buf.getvalue()


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_edit_consistency_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # noqa: SLF001  隔离环境需重置单例
    from papermate.api.app import create_app

    app = create_app()
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def _upload_docx(client, texts: list[str]) -> int:
    r = client.post(
        "/api/v1/documents",
        files={"file": ("paper.docx", make_docx_bytes(texts),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert r.status_code == 200, (r.status_code, r.text)
    return r.json()["id"]


def _chunks_texts() -> list[str]:
    db = Path(os.environ["PAPERMATE_DATA_DIR"]) / "papermate.db"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return [r[0] or "" for r in con.execute("select content from chunks")]
    finally:
        con.close()


def test_edit_then_background_reprocess_reflects_new_content(client) -> None:
    """★★ 改稿 → 后台重处理 → chunks/检索必须反映**新**内容（修复前是假生效）。"""
    doc_id = _upload_docx(client, ["引言。", SENT_A])

    # 改稿：P1 整段替换为句 B。TestClient 会在响应后排空 background tasks
    # ⇒ 返回时 `_job`（reprocess）已执行完，可直接断言终态。
    r = client.post(
        f"/api/v1/documents/{doc_id}/apply-edits",
        json={"edits": [{"pid": "P1", "op": "replace", "text": SENT_B}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"], f"编辑未生效：{body}"
    assert body["reindex_triggered"] is True

    # ① chunks 表（**精确文本**，不依赖 embedder 语义）
    doc_now = client.get(f"/api/v1/documents/{doc_id}").json()
    texts = _chunks_texts()
    assert texts, "chunks 为空（重处理未落库）"
    assert any(SENT_B[:40] in t for t in texts), (
        f"改稿后 chunks 不含新文本 ⇒ reprocess 假生效（回归）；doc status={doc_now.get('status')} "
        f"chunks 内容预览={[t[:150] for t in texts]}"
    )
    assert not any("quick brown fox" in t for t in texts), (
        "改稿后 chunks 仍含旧文本 ⇒ 检索会返回改稿前内容（回归）"
    )

    # ② 检索接口可用且能取到该文档的块（fake embedder 下只断言结构，不断言语义排序）
    q = client.post(
        "/api/v1/search",
        json={"query": "drone swarms", "top_k": 3, "doc_ids": [doc_id]},
    )
    assert q.status_code == 200, q.text
    assert isinstance(q.json(), list)


def test_reprocess_is_exposed_on_service(client) -> None:
    """守卫：`reprocess` 公共方法必须存在（`_trigger_reindex` 依赖它，勿删）。"""
    from papermate.services.document_service import DocumentService

    assert hasattr(DocumentService, "reprocess")
