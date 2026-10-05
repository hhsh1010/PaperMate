"""图片/图表提取（ER-03 图表部分）：
- FigureExtractor：get_image_info 提取嵌入图片 + caption 启发式 + figure/image_table 分类
- 集成：混合 PDF（文本表格 + 截图表格 + 纯图片）→ 图片落盘、figure chunk 入库、
  meta 诊断（num_images / extraction_warnings）、图片端点可访问
- 覆盖根因修复：截图型表格/纯图片此前 find_tables 完全失效且静默吞错
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.rag.figure_extractor import FigureExtractor


def _make_mixed_pdf(path: str) -> None:
    """构造混合 PDF：页1 文本表格；页2 截图表格（表格渲染成图片嵌入）；页3 纯图片。"""
    import pymupdf

    doc = pymupdf.open()
    p1 = doc.new_page(width=300, height=400)
    p1.insert_text((30, 30), "Table 1: Model comparison", fontsize=12)
    x0, y0, x1, y1 = 30, 60, 270, 150
    p1.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=1)
    for dy in (25, 50):
        p1.draw_line(pymupdf.Point(x0, y0 + dy), pymupdf.Point(x1, y0 + dy), color=(0, 0, 0), width=1)
    for dx in (80, 160):
        p1.draw_line(pymupdf.Point(x0 + dx, y0), pymupdf.Point(x0 + dx, y1), color=(0, 0, 0), width=1)
    cells = [("Model", "Acc", "Size"), ("BERT", "92.1", "110M"), ("GPT", "88.4", "175B")]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            p1.insert_text((x0 + 6 + c * 80, y0 + 18 + r * 25), val, fontsize=10)
    # 截图表格（图片型表格）
    pix = p1.get_pixmap(clip=pymupdf.Rect(20, 50, 280, 160), dpi=120)
    png = pix.tobytes("png")
    p2 = doc.new_page(width=300, height=400)
    p2.insert_text((30, 30), "Table 2: image-based table", fontsize=11)
    p2.insert_image(pymupdf.Rect(40, 50, 270, 170), stream=png)
    # 纯图片（图 1）
    p3 = doc.new_page(width=300, height=400)
    p3.insert_text((30, 30), "Figure 1: architecture overview", fontsize=11)
    p3.insert_image(pymupdf.Rect(40, 60, 270, 340), stream=png)
    doc.save(path)
    doc.close()


def _make_scanned_pdf(path: str) -> None:
    """构造扫描件 PDF：整页嵌入一张大图（含表格），无文本层。"""
    import pymupdf

    tmp = pymupdf.open()
    p = tmp.new_page(width=600, height=800)
    x0, y0, x1, y1 = 50, 80, 550, 400
    p.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=2)
    for dy in (80, 160, 240, 320):
        p.draw_line(pymupdf.Point(x0, y0 + dy), pymupdf.Point(x1, y0 + dy), color=(0, 0, 0), width=1)
    for dx in (250, 400):
        p.draw_line(pymupdf.Point(x0 + dx, y0), pymupdf.Point(x0 + dx, y1), color=(0, 0, 0), width=1)
    cells = [("Model", "Acc", "Params", "Year"), ("BERT", "92.1", "110M", "2018"),
             ("GPT", "88.4", "175B", "2019"), ("T5", "89.0", "60M", "2020")]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            p.insert_text(pymupdf.Point(x0 + 5 + c * 150, y0 + 30 + r * 80), val, fontsize=12)
    pix = p.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    img = pix.tobytes("png")
    tmp.close()
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_image(pymupdf.Rect(0, 0, 600, 800), stream=img)  # 整页图，无文本层
    doc.save(path)
    doc.close()


class TestFigureExtractor:
    def test_extracts_embedded_images(self, tmp_path):
        pdf = tmp_path / "mixed.pdf"
        _make_mixed_pdf(str(pdf))
        figures, warnings = FigureExtractor().extract(str(pdf))
        # 页2 截图表格 + 页3 纯图片 → 至少 2 张
        assert len(figures) >= 2
        # 图注启发式命中（Figure 1）
        caps = [f.caption for f in figures]
        assert any("Figure 1" in c for c in caps)
        # 分类：截图表格（宽幅）→ image_table
        kinds = [f.kind for f in figures]
        assert "image_table" in kinds
        assert any("第" in w and "图片" in w for w in warnings)

    def test_export_images_to_dir(self, tmp_path):
        pdf = tmp_path / "mixed.pdf"
        _make_mixed_pdf(str(pdf))
        img_dir = tmp_path / "images"
        figures, _ = FigureExtractor().extract(str(pdf), image_dir=img_dir)
        assert len(figures) >= 2
        files = list(img_dir.iterdir())
        assert len(files) == len(figures)  # 每张图都落盘
        assert all(f.stat().st_size > 0 for f in files)

    def test_corrupt_pdf_returns_empty_with_warning(self, tmp_path):
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"%PDF-1.4 broken")
        figures, warnings = FigureExtractor().extract(str(pdf))
        assert figures == []
        assert warnings  # 有诊断提示而非静默

    def test_full_page_image_classified_as_image_table(self, tmp_path):
        """扫描件（整页图 ≥60% 页面面积）→ 标记 image_table + 提示需 OCR，而非静默丢弃。"""
        pdf = tmp_path / "scanned.pdf"
        _make_scanned_pdf(str(pdf))
        figures, warnings = FigureExtractor().extract(str(pdf))
        assert len(figures) == 1
        assert figures[0].kind == "image_table"
        assert any("整页图片" in w and "OCR" in w for w in warnings)


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_fig_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


class TestFigureIntegration:
    def test_upload_mixed_pdf_figures_land(self, client):
        import tempfile as _tf

        with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            fpath = f.name
        try:
            _make_mixed_pdf(fpath)
            content = open(fpath, "rb").read()
        finally:
            os.unlink(fpath)
        r = client.post("/api/v1/documents", files={"file": ("mixed.pdf", content, "application/pdf")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ready"
        meta = body["meta"]
        # 诊断信息可见（不再静默）
        assert meta["num_tables"] >= 1
        assert meta["num_images"] >= 2
        assert meta["extraction_warnings"]  # 图片型表格提示非空
        # figure chunk 入库
        from papermate.models.db import Chunk

        doc_id = body["id"]
        with client.app.state.doc_svc._Session() as db:
            fig_chunks = [
                c for c in db.query(Chunk).filter(Chunk.document_id == doc_id).all()
                if json.loads(c.meta_json or "{}").get("chunk_type") in ("figure", "image_table")
            ]
        assert len(fig_chunks) >= 2
        assert all(c.page >= 1 for c in fig_chunks)
        # 图片文件落盘 + 端点可访问
        fig = fig_chunks[0]
        fname = json.loads(fig.meta_json)["file"]
        img_resp = client.get(f"/api/v1/documents/{doc_id}/images/{fname}")
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"].startswith("image/")
        # parsed 文本包含图片标记（检索可命中图注）
        parsed = client.app.state.doc_svc.get_parsed_text(doc_id)
        assert "【图片" in parsed

    def test_image_endpoint_blocks_path_traversal(self, client):
        assert client.get("/api/v1/documents/1/images/..%2F..%2Fsecret").status_code in (400, 404)

    def test_text_doc_no_images(self, client):
        r = client.post("/api/v1/documents", files={"file": ("a.md", b"# T\n\nbody text", "text/markdown")})
        assert r.status_code == 200
        assert r.json()["meta"].get("num_images", 0) == 0

    def test_list_document_images_returns_figures(self, client):
        import tempfile as _tf

        with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            fpath = f.name
        try:
            _make_mixed_pdf(fpath)
            content = open(fpath, "rb").read()
        finally:
            os.unlink(fpath)
        r = client.post("/api/v1/documents", files={"file": ("mixed.pdf", content, "application/pdf")})
        assert r.status_code == 200
        doc_id = r.json()["id"]
        # 列表端点返回图片清单（含 filename/page/kind/caption）
        lst = client.get(f"/api/v1/documents/{doc_id}/images")
        assert lst.status_code == 200
        items = lst.json()
        assert isinstance(items, list) and len(items) >= 2
        keys = {*items[0].keys()}
        assert {"filename", "page", "kind", "caption"} <= keys
        assert items[0]["kind"] in ("figure", "image_table")
        # 清单里的 filename 可被单图端点直接服务
        one = next(i for i in items if i["kind"] == "image_table")
        img = client.get(f"/api/v1/documents/{doc_id}/images/{one['filename']}")
        assert img.status_code == 200
        assert img.headers["content-type"].startswith("image/")

    def test_list_document_images_404_for_missing_doc(self, client):
        assert client.get("/api/v1/documents/999999/images").status_code == 404

    def test_list_document_images_empty_for_text_doc(self, client):
        r = client.post("/api/v1/documents", files={"file": ("a.md", b"# T\n\nbody text", "text/markdown")})
        doc_id = r.json()["id"]
        lst = client.get(f"/api/v1/documents/{doc_id}/images")
        assert lst.status_code == 200
        assert lst.json() == []
