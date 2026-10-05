"""Word(.docx) 内嵌图片提取（ER-03 图表部分，docx 版）：
- DocxImageExtractor：inline_shapes 提取内嵌图片 + 宽幅分类 image_table + blob 去重命名
- 集成：含图 docx 上传 → 图片落盘、figure/image_table chunk 入库、meta 诊断、
  num_images、图片列表/单图端点可访问；无图 docx → num_images == 0
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import zipfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.rag.docx_image_extractor import DocxImageExtractor

#: 1x1 红色 PNG（docx 内图片尺寸以 XML 的 cx/cy 为准，像素内容仅占位）
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>"""


def _make_docx(path: str, *, wide: bool = False, with_image: bool = True) -> None:
    cx = 2000000 if wide else 1000000
    cy = 900000 if wide else 800000
    drawing = ""
    if with_image:
        drawing = f"""<w:p><w:r><w:drawing>
          <wp:inline distT="0" distB="0" distL="0" distR="0">
            <wp:extent cx="{cx}" cy="{cy}"/>
            <wp:effectExtent l="0" t="0" r="0" b="0"/>
            <wp:docPr id="1" name="Picture 1"/>
            <wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>
            <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                  <pic:nvPicPr><pic:cNvPr id="0" name="image1.png"/><pic:cNvPicPr/></pic:nvPicPr>
                  <pic:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
                  <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
                </pic:pic>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing></w:r></w:p>"""
    doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    <w:p><w:r><w:t xml:space="preserve">Figure 1: architecture overview</w:t></w:r></w:p>
    {drawing}
  </w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/document.xml", doc_xml)
        if with_image:
            z.writestr("word/_rels/document.xml.rels", _DOC_RELS)
            z.writestr("word/media/image1.png", _PNG)


class TestDocxImageExtractor:
    def test_extracts_wide_image_as_image_table(self, tmp_path):
        docx = tmp_path / "fig.docx"
        _make_docx(str(docx), wide=True)
        img_dir = tmp_path / "images"
        figures, warnings = DocxImageExtractor().extract(str(docx), image_dir=img_dir)
        assert len(figures) == 1
        assert figures[0].kind == "image_table"
        assert any("宽幅" in w for w in warnings)
        files = list(img_dir.iterdir())
        assert len(files) == 1
        assert files[0].stat().st_size > 0
        assert figures[0].filename.startswith("docx-")

    def test_extracts_normal_image_as_figure(self, tmp_path):
        docx = tmp_path / "fig.docx"
        _make_docx(str(docx), wide=False)
        figures, _ = DocxImageExtractor().extract(str(docx))
        assert len(figures) == 1
        assert figures[0].kind == "figure"

    def test_no_image_docx_returns_empty(self, tmp_path):
        docx = tmp_path / "text.docx"
        _make_docx(str(docx), with_image=False)
        figures, warnings = DocxImageExtractor().extract(str(docx))
        assert figures == []
        assert warnings == []  # 无图不是错误，不报警

    def test_corrupt_docx_returns_empty_with_warning(self, tmp_path):
        docx = tmp_path / "bad.docx"
        docx.write_bytes(b"PK\x03\x04 not a real docx")
        figures, warnings = DocxImageExtractor().extract(str(docx))
        assert figures == []
        assert warnings  # 有诊断提示而非静默


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_docx_img_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


class TestDocxImageIntegration:
    def test_upload_docx_with_image_figures_land(self, client):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            fpath = f.name
        try:
            _make_docx(fpath, wide=True)
            content = open(fpath, "rb").read()
        finally:
            os.unlink(fpath)
        r = client.post(
            "/api/v1/documents",
            files={"file": ("fig.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ready"
        meta = body["meta"]
        assert meta["num_images"] >= 1
        assert meta["extraction_warnings"]  # 宽幅图片提示非空
        from papermate.models.db import Chunk

        doc_id = body["id"]
        with client.app.state.doc_svc._Session() as db:
            fig_chunks = [
                c for c in db.query(Chunk).filter(Chunk.document_id == doc_id).all()
                if json.loads(c.meta_json or "{}").get("chunk_type") in ("figure", "image_table")
            ]
        assert len(fig_chunks) >= 1
        assert all(c.page is None for c in fig_chunks)  # docx 无页码概念 → None
        fname = json.loads(fig_chunks[0].meta_json)["file"]
        img_resp = client.get(f"/api/v1/documents/{doc_id}/images/{fname}")
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"].startswith("image/")
        parsed = client.app.state.doc_svc.get_parsed_text(doc_id)
        assert "【图片" in parsed

    def test_upload_docx_without_image_no_figures(self, client):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            fpath = f.name
        try:
            _make_docx(fpath, with_image=False)
            content = open(fpath, "rb").read()
        finally:
            os.unlink(fpath)
        r = client.post(
            "/api/v1/documents",
            files={"file": ("text.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert r.status_code == 200
        assert r.json()["meta"].get("num_images", 0) == 0

    def test_list_document_images_for_docx(self, client):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            fpath = f.name
        try:
            _make_docx(fpath, wide=True)
            content = open(fpath, "rb").read()
        finally:
            os.unlink(fpath)
        r = client.post(
            "/api/v1/documents",
            files={"file": ("fig.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert r.status_code == 200
        doc_id = r.json()["id"]
        lst = client.get(f"/api/v1/documents/{doc_id}/images")
        assert lst.status_code == 200
        items = lst.json()
        assert isinstance(items, list) and len(items) >= 1
        assert {"filename", "page", "kind", "caption"} <= set(items[0].keys())
        assert items[0]["kind"] in ("figure", "image_table")
        one = next(i for i in items if i["kind"] == "image_table")
        img = client.get(f"/api/v1/documents/{doc_id}/images/{one['filename']}")
        assert img.status_code == 200
        assert img.headers["content-type"].startswith("image/")
