"""页码解析与 1 索引回归测试（2026-08-24 根因修复）

覆盖：
- parse_pdf 对多页 PDF 返回 1 起算页码（1,2,3…），绝不出现第 0 页；
- 非分页文档（txt/docx）块 page=None（不再用 0 占位）；
- RAG 管线从数据库 Chunk 表构建后，检索/ pdf_extract 返回的页码与落库一致（修复「检索 page=0」）；
- normalize_page / validate_requested_page 边界：0/负数/None/浮点/bool → None，越界页码给出提示；
- /search 与 pdf_extract 不再输出无效页码。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.rag.page_utils import normalize_page, validate_requested_page
from lightcore.rag.parser import DocumentParser
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry


def _make_multi_page_pdf(path: str, n_pages: int = 3) -> None:
    import pymupdf

    doc = pymupdf.open()
    for i in range(n_pages):
        pg = doc.new_page()
        # 每页一句唯一 ASCII 关键字（CJK 默认字体提取不稳，用 ASCII 保证可检索），便于按页验证
        pg.insert_text((72, 72), f"Unique keyword beta{i + 1} located on page {i + 1} sample body.")
    doc.save(path)
    doc.close()


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_page_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _upload_multi_page_pdf(client, n_pages: int = 3) -> int:
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_multi_page_pdf(fpath, n_pages)
        content = open(fpath, "rb").read()
    finally:
        os.unlink(fpath)
    r = client.post("/api/v1/documents", files={"file": ("multi.pdf", content, "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ---------- 1. 解析层：1 索引、无 0 页 ----------

def test_parse_pdf_uses_1_indexed_pages():
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_multi_page_pdf(fpath, n_pages=3)
        blocks = DocumentParser().parse_pdf(fpath)
    finally:
        os.unlink(fpath)
    pages = sorted({b.page for b in blocks})
    assert pages == [1, 2, 3], f"多页 PDF 应返回 1 起算页码，实际 {pages}"
    assert all(b.page is not None and b.page >= 1 for b in blocks)
    assert 0 not in pages


def test_parse_text_has_no_physical_page():
    blocks = DocumentParser().parse_text("第一段。\n\n第二段。")
    assert all(b.page is None for b in blocks)


# ---------- 2. 归一化与边界 ----------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, None), (-1, None), (None, None), (1.5, None), (True, None),
        (1, 1), (3, 3), (100, 100),
    ],
)
def test_normalize_page(raw, expected):
    assert normalize_page(raw) == expected


def test_validate_requested_page_out_of_range():
    norm, note = validate_requested_page(99, {1, 2, 3})
    assert norm == 99 and note is not None and "无第 99 页" in note


def test_validate_requested_page_invalid():
    norm, note = validate_requested_page(0, {1, 2, 3})
    assert norm is None and "从 1 开始" in note


def test_validate_requested_page_valid():
    norm, note = validate_requested_page(2, {1, 2, 3})
    assert norm == 2 and note is None


# ---------- 3. 管线从 DB 块构建后保留页码（核心回归） ----------

def test_pipeline_from_db_chunks_preserves_pages(client):
    """上传多页 PDF 后，管线检索/ pdf_extract 返回正确的 1 索引页码，而非 0。"""
    import asyncio

    doc_id = _upload_multi_page_pdf(client, n_pages=3)
    svc = client.app.state.doc_svc
    pipe = asyncio.run(svc.get_pipeline(doc_id))

    reg = ToolRegistry()
    register_builtin_tools(reg, rag_pipeline=pipe)
    exe = ToolExecutor(reg)

    # pdf_extract 按页码精确取页：第 2 页应返回 page=2 且内容非空
    res = asyncio.run(exe.execute("pdf_extract", {"page": 2}))
    assert res.ok, res.error if not res.ok else ""
    assert res.output["page"] == 2, f"应按第 2 页返回，实际 {res.output.get('page')}"
    assert res.output["num_chunks"] > 0, "第 2 页应有内容"
    assert "beta2" in res.output["text"], "应命中第 2 页唯一关键字"

    # 所有页都应能取到（1,2,3）
    for p in (1, 2, 3):
        r2 = asyncio.run(exe.execute("pdf_extract", {"page": p}))
        assert r2.ok and r2.output["page"] == p and r2.output["num_chunks"] > 0


def test_retrieve_returns_valid_pages_not_zero(client):
    """/search 检索 PDF 返回的引用页码均为有效 ≥1，绝不出现 0/None。"""
    doc_id = _upload_multi_page_pdf(client, n_pages=3)
    r = client.post("/api/v1/search", json={"query": "beta", "doc_ids": [doc_id], "top_k": 5})
    assert r.status_code == 200, r.text
    results = r.json()
    assert results, "应返回检索结果"
    for c in results:
        assert c["page"] is not None and c["page"] >= 1, f"页码应为有效值，实际 {c['page']}"


def test_pdf_extract_boundary_invalid_and_out_of_range(client):
    import asyncio

    doc_id = _upload_multi_page_pdf(client, n_pages=3)
    pipe = asyncio.run(client.app.state.doc_svc.get_pipeline(doc_id))
    reg = ToolRegistry()
    register_builtin_tools(reg, rag_pipeline=pipe)
    exe = ToolExecutor(reg)

    # 页码 0：无效，给出提示且不返回内容
    res0 = asyncio.run(exe.execute("pdf_extract", {"page": 0}))
    assert res0.ok and res0.output["num_chunks"] == 0 and "从 1 开始" in res0.output["note"]

    # 越界页码 99：给出提示
    res99 = asyncio.run(exe.execute("pdf_extract", {"page": 99}))
    assert res99.ok and res99.output["num_chunks"] == 0 and "无第 99 页" in res99.output["note"]
