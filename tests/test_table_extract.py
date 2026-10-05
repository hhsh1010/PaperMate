"""P1-3 表格结构化提取（ER-03 表格优先）：
- TableExtractor 纯函数（markdown/平铺文本/bbox）
- 集成：生成含表格的 PDF 上传 → 表格块以 chunk_type=table 入库、parsed 文本可检索
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.rag.table_extractor import TableExtractor, _table_to_markdown, md_table_lint


def _make_grid_pdf(path: str) -> None:
    """生成含 3×3 网格表格的测试 PDF（fitz 绘制）。"""
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


def _make_borderless_pdf(path: str) -> None:
    """无矢量边框、仅靠文本列对齐的表格（find_tables 默认 lines 策略会漏检）。"""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    rows = [("Model", "Accuracy", "Parameters", "Year"),
            ("BERT", "92.1", "110M", "2018"),
            ("GPT-3", "88.4", "175B", "2020"),
            ("T5", "89.0", "60M", "2020")]
    for r, row in enumerate(rows):
        y = 120 + r * 40
        for c, val in enumerate(row):
            page.insert_text(pymupdf.Point(60 + c * 140, y), val, fontsize=12)
    doc.save(path)
    doc.close()


def _make_paragraph_pdf(path: str) -> None:
    """纯段落页：text 策略易误检为大表，验证负向用例不误检。"""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for i in range(20):
        page.insert_text(
            pymupdf.Point(60, 80 + i * 30),
            f"This is a normal paragraph line number {i} with some running text.",
            fontsize=11,
        )
    doc.save(path)
    doc.close()


class TestTableExtractor:
    def test_extracts_grid_table(self, tmp_path):
        pdf = str(tmp_path / "grid.pdf")
        _make_grid_pdf(pdf)
        tables = TableExtractor().extract(pdf)
        assert len(tables) >= 1
        t = tables[0]
        assert t.page >= 1
        assert len(t.bbox) == 4 and all(isinstance(v, float) for v in t.bbox)
        assert t.rows == 3 and t.cols == 3
        assert t.header == ["Model", "Acc", "Params"]
        assert "BERT" in t.text and "175B" in t.text

    def test_corrupt_pdf_returns_empty(self, tmp_path):
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"%PDF-1.4 not a real pdf")
        assert TableExtractor().extract(str(pdf)) == []

    def test_extracts_borderless_text_table(self, tmp_path):
        """无边框（纯文本列对齐）表格：lines 策略漏检，text 策略兜底命中（ER-03 修复）。"""
        pdf = tmp_path / "borderless.pdf"
        _make_borderless_pdf(pdf)
        tables = TableExtractor().extract(str(pdf))
        assert len(tables) >= 1
        t = tables[0]
        assert t.rows == 4 and t.cols == 4
        assert t.header == ["Model", "Accuracy", "Parameters", "Year"]
        assert "BERT" in t.text and "175B" in t.text

    def test_paragraph_page_not_detected_as_table(self, tmp_path):
        """纯段落页不得被 text 策略误检为「大表」（防检索污染）。"""
        pdf = tmp_path / "para.pdf"
        _make_paragraph_pdf(str(pdf))
        tables = TableExtractor().extract(str(pdf))
        assert tables == []


class TestTableMarkdown:
    def test_markdown_with_header_separator(self):
        data = [["a", "b"], ["1", "2"]]
        md = _table_to_markdown(data)
        assert md.splitlines()[0].startswith("| a | b |")
        assert "---" in md.splitlines()[1]
        assert "| 1 | 2 |" in md

    def test_ragged_rows_padded(self):
        data = [["a", "b"], ["1"]]
        md = _table_to_markdown(data)
        assert "| 1 |  |" in md

    def test_empty_data(self):
        assert _table_to_markdown([]) == ""


class TestMdTableLint:
    """P0-2.2 表格结构校验（md_table_lint）。"""

    def test_well_formed_passes(self):
        md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        assert md_table_lint(md) == []

    def test_missing_separator(self):
        md = "| a | b |\n| 1 | 2 |"
        issues = md_table_lint(md)
        assert any("分隔行" in i for i in issues)

    def test_empty_header(self):
        md = "|  |  |\n| --- | --- |\n| 1 | 2 |"
        issues = md_table_lint(md)
        assert any("表头为空" in i for i in issues)

    def test_ragged_columns(self):
        md = "| a | b | c |\n| --- | --- | --- |\n| 1 | 2 |\n| 1 | 2 | 3 | 4 |"
        issues = md_table_lint(md)
        assert any("列数不一致" in i for i in issues)

    def test_too_few_lines(self):
        assert md_table_lint("| a |") != []

    def test_extractor_output_passes_lint(self, tmp_path):
        """真实提取产物应通过结构校验（表头/分隔行/列数一致由 _table_to_markdown 保证）。"""
        pdf = tmp_path / "grid.pdf"
        _make_grid_pdf(str(pdf))
        tables = TableExtractor().extract(str(pdf))
        assert tables
        for t in tables:
            assert md_table_lint(t.markdown) == []


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_table_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


class TestTableIntegration:
    def test_upload_pdf_creates_table_chunk(self, client):
        import tempfile as _tf

        with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            fpath = f.name
        try:
            _make_grid_pdf(fpath)
            with open(fpath, "rb") as f:
                content = f.read()
        finally:
            os.unlink(fpath)
        r = client.post("/api/v1/documents", files={"file": ("grid.pdf", content, "application/pdf")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ready"
        assert body["meta"].get("num_tables", 0) >= 1
        # 表格块以 chunk_type=table 入库
        from papermate.models.db import Chunk

        with client.app.state.doc_svc._Session() as db:
            table_chunks = [
                c for c in db.query(Chunk).filter(Chunk.document_id == body["id"]).all()
                if json.loads(c.meta_json or "{}").get("chunk_type") == "table"
            ]
        assert len(table_chunks) >= 1
        tc = table_chunks[0]
        meta = json.loads(tc.meta_json)
        assert meta["rows"] == 3
        assert "BERT" in tc.content
        # parsed 文本包含表格内容（向量检索可命中）
        parsed = client.app.state.doc_svc.get_parsed_text(body["id"])
        assert parsed and "【表格" in parsed and "BERT" in parsed

    def test_text_doc_has_no_tables(self, client):
        r = client.post("/api/v1/documents", files={"file": ("a.md", b"# Title\n\nSome text here.", "text/markdown")})
        assert r.status_code == 200
        assert r.json()["meta"].get("num_tables", 0) == 0
