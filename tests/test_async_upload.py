"""异步上传（2026-09-02）：POST /documents/async 立即返回，后台处理；
轮询 GET /documents/{id}/progress 看进度，POST /documents/{id}/cancel 可取消。

保留同步 POST /documents（block=True）的旧语义由 test_word_support 等既有用例覆盖，
本文件只验证新的异步路径与取消契约。
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_async_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _upload_async(client, text: bytes, name: str = "doc.md"):
    return client.post(
        "/api/v1/documents/async",
        files={"file": (name, text, "text/markdown")},
    )


def test_async_upload_returns_immediately_then_reaches_ready(client):
    r = _upload_async(client, b"# Title\n\nSome body text for chunking.")
    assert r.status_code == 200
    doc = r.json()
    assert doc["status"] == "uploaded", "异步上传应立即可见 status=uploaded"

    did = doc["id"]
    # 轮询进度直到 ready（后台任务在 fake embedder 下很快完成）
    ready = False
    for _ in range(60):
        p = client.get(f"/api/v1/documents/{did}/progress")
        if p.status_code == 200 and p.json().get("status") == "ready":
            ready = True
            break
        time.sleep(0.1)
    assert ready, "异步上传应在合理时间内达到 ready"

    final = client.get(f"/api/v1/documents/{did}")
    assert final.status_code == 200
    assert final.json()["status"] == "ready"
    assert final.json()["num_chunks"] >= 1


def test_cancel_on_ready_is_idempotent_noop(client):
    r = _upload_async(client, b"# Title\n\nBody text.")
    did = r.json()["id"]
    # 等它处理完
    for _ in range(60):
        p = client.get(f"/api/v1/documents/{did}/progress")
        if p.status_code == 200 and p.json().get("status") == "ready":
            break
        time.sleep(0.1)
    cr = client.post(f"/api/v1/documents/{did}/cancel")
    assert cr.status_code == 200
    body = cr.json()
    assert body["ok"] is True
    # 已就绪 → cancelled=False（幂等无副作用）
    assert body.get("cancelled") is False
    # 文档仍在
    assert client.get(f"/api/v1/documents/{did}").status_code == 200


def test_cancel_midflight_cleans_up(client, monkeypatch):
    """注入 2s 延迟到 get_pipeline，制造可取消窗口；取消后应清理半成品（404 / 终态）。"""
    from papermate.services.document_service import DocumentService

    orig = DocumentService.get_pipeline

    async def slow_get_pipeline(self, doc_id, **kw):
        await asyncio.sleep(2)  # 卡在 embedding 阶段，给取消留出时间窗
        return await orig(self, doc_id, **kw)

    monkeypatch.setattr(DocumentService, "get_pipeline", slow_get_pipeline)

    r = _upload_async(client, b"# Title\n\nBody text for a cancellable upload.")
    did = r.json()["id"]
    assert r.json()["status"] == "uploaded"

    # 等到进入处理中阶段再取消
    triggered = False
    for _ in range(60):
        p = client.get(f"/api/v1/documents/{did}/progress")
        st = p.json().get("status") if p.status_code == 200 else None
        if st in ("parsing", "chunking", "embedding"):
            cr = client.post(f"/api/v1/documents/{did}/cancel")
            assert cr.status_code == 200
            assert cr.json().get("cancelled") in (True, False)
            triggered = True
            break
        time.sleep(0.1)

    # 最终应被清理（404）或进入 cancelled/failed 终态
    gone = False
    for _ in range(60):
        pr = client.get(f"/api/v1/documents/{did}/progress")
        if pr.status_code == 404:
            gone = True
            break
        st = pr.json().get("status")
        if st in ("cancelled", "failed"):
            gone = True
            break
        time.sleep(0.1)
    assert gone, f"取消后文档应被清理或置为终态；triggered={triggered}"
