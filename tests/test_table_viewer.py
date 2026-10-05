"""表格查看器后端 + 图片增强(OCR)运行时开关测试：
- GET /documents/{id}/tables 返回表格块（markdown/rows/cols/header/page）
- GET/PUT /settings 持久化 figure_enrich（默认关闭，可开启）
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.rag.table_extractor import TableExtractor


def _make_grid_pdf(path: str) -> None:
    """含 3×3 矢量边框表格的 PDF（TableExtractor 默认策略可检出）。"""
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
    tmp = tempfile.mkdtemp(prefix="pm_tableview_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None  # 重置运行时缓存，确保用临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------- figure_enrich 运行时开关 ----------------
def test_figure_enrich_default_off(client):
    data = client.get("/api/v1/settings").json()
    fe = data["figure_enrich"]
    assert fe["enabled"] is False
    assert "ocr" in fe["methods"]  # 预置方法，仅 enabled 决定是否执行
    assert "figure_enrich_deps" in data  # 依赖探测随设置返回（前端提示用）


def test_put_figure_enrich_roundtrip(client):
    r = client.put("/api/v1/settings", json={"figure_enrich": {"enabled": True, "methods": ["ocr", "table"]}})
    assert r.status_code == 200
    assert r.json()["figure_enrich"] == {"enabled": True, "methods": ["ocr", "table"]}
    # 再读 GET 持久化生效
    assert client.get("/api/v1/settings").json()["figure_enrich"]["enabled"] is True


def test_put_figure_enrich_rejects_unknown_methods(client):
    r = client.put("/api/v1/settings", json={"figure_enrich": {"enabled": True, "methods": ["ocr", "evil"]}})
    assert r.status_code == 200
    assert r.json()["figure_enrich"]["methods"] == ["ocr"]  # 非法方法被过滤


# ---------------- /documents/{id}/tables 表格查看 ----------------
def _upload_grid(client):
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


def test_list_document_tables_returns_markdown(client):
    doc_id = _upload_grid(client)
    lst = client.get(f"/api/v1/documents/{doc_id}/tables")
    assert lst.status_code == 200
    items = lst.json()
    assert len(items) >= 1
    t = items[0]
    assert {"page", "markdown", "rows", "cols", "bbox", "header"} <= set(t.keys())
    assert t["rows"] == 3 and t["cols"] == 3
    assert t["header"] == ["Model", "Acc", "Params"]
    assert "| BERT | 92.1 | 110M |" in t["markdown"]


def test_list_document_tables_404_for_missing_doc(client):
    assert client.get("/api/v1/documents/999999/tables").status_code == 404


def test_list_document_tables_empty_for_text_doc(client):
    r = client.post("/api/v1/documents", files={"file": ("a.md", b"# T\n\nbody text", "text/markdown")})
    assert r.status_code == 200
    lst = client.get(f"/api/v1/documents/{r.json()['id']}/tables")
    assert lst.status_code == 200
    assert lst.json() == []


def test_table_extractor_still_detects_grid(tmp_path):
    pdf = tmp_path / "grid.pdf"
    _make_grid_pdf(str(pdf))
    assert len(TableExtractor().extract(str(pdf))) >= 1


# ---------------- P0-2.3：页面截图端点 /documents/{id}/pages/{n}/image ----------------
def test_page_image_renders_png(client):
    doc_id = _upload_grid(client)
    r = client.get(f"/api/v1/documents/{doc_id}/pages/1/image")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG 魔数


def test_page_image_cached_second_hit(client):
    doc_id = _upload_grid(client)
    url = f"/api/v1/documents/{doc_id}/pages/1/image?dpi=150"
    first = client.get(url)
    second = client.get(url)
    assert first.status_code == 200 and second.status_code == 200
    assert first.content == second.content  # 命中缓存，字节一致


def test_page_image_bbox_clip(client):
    doc_id = _upload_grid(client)
    r = client.get(f"/api/v1/documents/{doc_id}/pages/1/image", params={"bbox": "40,70,260,170"})
    assert r.status_code == 200, r.text
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_page_image_invalid_bbox(client):
    doc_id = _upload_grid(client)
    assert client.get(f"/api/v1/documents/{doc_id}/pages/1/image", params={"bbox": "10,20,5,30"}).status_code == 400
    assert client.get(f"/api/v1/documents/{doc_id}/pages/1/image", params={"bbox": "1,2,3"}).status_code == 400
    assert client.get(f"/api/v1/documents/{doc_id}/pages/1/image", params={"bbox": "a,b,c,d"}).status_code == 400


def test_page_image_out_of_range_page(client):
    doc_id = _upload_grid(client)
    assert client.get(f"/api/v1/documents/{doc_id}/pages/999/image").status_code == 404
    assert client.get(f"/api/v1/documents/{doc_id}/pages/0/image").status_code == 400


def test_page_image_dpi_bounds(client):
    doc_id = _upload_grid(client)
    assert client.get(f"/api/v1/documents/{doc_id}/pages/1/image", params={"dpi": 999}).status_code == 422
    assert client.get(f"/api/v1/documents/{doc_id}/pages/1/image", params={"dpi": 5}).status_code == 422


def test_page_image_requires_pdf(client):
    r = client.post("/api/v1/documents", files={"file": ("a.md", b"# T\n\nbody", "text/markdown")})
    assert r.status_code == 200
    doc_id = r.json()["id"]
    assert client.get(f"/api/v1/documents/{doc_id}/pages/1/image").status_code == 400


def test_page_image_missing_doc(client):
    assert client.get("/api/v1/documents/999999/pages/1/image").status_code == 404
