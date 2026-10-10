"""「另存为」落盘的路径校验 + 写盘端点（P-04/P-05，2026-10-09）。

安全边界是本组最重要的部分：`dest` 必须来自用户自己的原生「另存为」对话框，
后端**只写不读**、拒绝相对路径与超限内容。这些必须逐条钉死。
"""

from __future__ import annotations

import base64

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from papermate.api.dest_path import MAX_EXPORT_BYTES, require_abs_dest
from papermate.api.routes_files import router


class TestRequireAbsDest:
    def test_rejects_empty(self):
        with pytest.raises(HTTPException) as ei:
            require_abs_dest("   ")
        assert ei.value.status_code == 400

    def test_rejects_relative_path(self):
        with pytest.raises(HTTPException) as ei:
            require_abs_dest("sub/out.docx")
        assert ei.value.status_code == 400

    def test_rejects_missing_parent(self, tmp_path):
        with pytest.raises(HTTPException) as ei:
            require_abs_dest(str(tmp_path / "nope" / "a.csv"))
        assert ei.value.status_code == 400

    def test_accepts_absolute_existing_parent(self, tmp_path):
        p = require_abs_dest(str(tmp_path / "a.csv"))
        assert p == tmp_path / "a.csv"


class TestWriteLocalFile:
    def _app(self):
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return app

    def test_writes_base64_to_dest(self, tmp_path):
        dest = tmp_path / "out.jsonl"
        body = base64.b64encode(b'{"a":1}\n').decode()
        with TestClient(self._app()) as client:
            r = client.post("/files/write", json={"dest": str(dest), "content_b64": body})
        assert r.status_code == 200, r.text
        assert dest.read_text(encoding="utf-8") == '{"a":1}\n'
        assert r.json()["bytes"] == len('{"a":1}\n')

    def test_rejects_relative_dest(self, tmp_path):
        with TestClient(self._app()) as client:
            r = client.post("/files/write", json={"dest": "relative.csv", "content_b64": "eA=="})
        assert r.status_code == 400

    def test_rejects_invalid_base64(self, tmp_path):
        with TestClient(self._app()) as client:
            r = client.post(
                "/files/write", json={"dest": str(tmp_path / "x.bin"), "content_b64": "%%%not-base64%%%"}
            )
        assert r.status_code == 400

    def test_rejects_oversized(self, tmp_path):
        too_big = base64.b64encode(b"0" * (MAX_EXPORT_BYTES + 1)).decode()
        with TestClient(self._app()) as client:
            r = client.post(
                "/files/write", json={"dest": str(tmp_path / "big.bin"), "content_b64": too_big}
            )
        assert r.status_code == 400
