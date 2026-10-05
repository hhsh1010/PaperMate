"""删除文档回归测试：uploads/<id>/ 含 images/ 子目录时不再假失败。

根因（2026-09-03）：上传管线会在 uploads/<id>/images/ 下落页面图，旧 delete()
用 iterdir+unlink 清理，遇子目录在 Windows 上抛 PermissionError —— 此时 DB 行
已删（级联 chunks），路由 500 → 前端提示「删除失败」但刷新后文档消失，且留孤儿。
修复：shutil.rmtree 递归删 + 清理失败容错（warning 不冒泡）+ 删除前取消进行中任务。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from papermate import config as cfg_mod


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_del_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    cfg_mod._settings = None


def _upload_txt(client: TestClient, name: str, content: str) -> dict:
    resp = client.post(
        "/api/v1/documents",
        files={"file": (name, content.encode("utf-8"), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestDeleteWithImagesSubdir:
    def test_delete_doc_with_images_subdir_ok(self, client):
        """uploads/<id>/images/ 子目录存在时删除必须 200 且目录整体移除（回归：PermissionError 假失败）。"""
        doc = _upload_txt(client, "demo.txt", "# 标题\n\n正文内容用于切片。")
        doc_id = doc["id"]
        settings = cfg_mod.get_settings()
        upload_dir = settings.upload_dir / str(doc_id)
        assert upload_dir.exists()
        # 模拟上传管线落下的页面图子目录
        images = upload_dir / "images"
        images.mkdir()
        (images / "p1-d1.png").write_bytes(b"\x89PNG fake")
        (upload_dir / "p2-x9.png").write_bytes(b"\x89PNG fake")

        resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"ok": True}
        # 目录（含子目录）与解析文本均已移除，无孤儿
        assert not upload_dir.exists()
        assert not (settings.parsed_dir / f"{doc_id}.txt").exists()
        assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404

    def test_delete_locked_file_still_ok_no_false_failure(self, client, monkeypatch):
        """清理失败（文件被占用）不应 500：DB 已删成功，路由仍返 ok（孤儿记 warning）。"""
        import shutil as _shutil

        doc = _upload_txt(client, "demo2.txt", "# 标题\n\n正文内容用于切片。")
        doc_id = doc["id"]
        settings = cfg_mod.get_settings()
        upload_dir = settings.upload_dir / str(doc_id)
        assert upload_dir.exists()

        def _boom(path, *a, **k):  # 模拟 Windows 文件占用
            raise OSError(32, "文件被占用（模拟）")

        monkeypatch.setattr(_shutil, "rmtree", _boom)
        # document_service 内部 `import shutil` 取的是同一模块对象，monkeypatch 生效
        resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200, resp.text
        assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404

    def test_delete_missing_doc_is_ok_idempotent(self, client):
        """删除不存在的文档：当前路由语义为幂等 ok（不 404）。"""
        assert client.delete("/api/v1/documents/999999").status_code == 200
