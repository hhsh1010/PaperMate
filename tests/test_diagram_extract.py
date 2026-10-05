"""矢量结构图提取（P0）测试：
- DiagramExtractor：矢量路径聚类 → bbox/标签/页码（1 索引）；装饰/过小过滤
- 与位图 figure 的 bbox 去重（交叠 > 50% 只保留位图）
- 上传链路：meta.num_diagrams + GET /documents/{id}/diagrams + 图片可访问
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.rag.diagram_extractor import DiagramExtractor


def _make_diagram_pdf(path: str, n_pages: int = 2, diagram_page: int = 2) -> None:
    """含矢量结构图（两节点矩形 + 连线 + 文字标签）的 PDF。"""
    import pymupdf

    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 60), f"Page {i + 1} body text.")
        if i + 1 == diagram_page:
            page.draw_rect(pymupdf.Rect(100, 100, 220, 150), color=(0, 0, 0), width=1)
            page.draw_rect(pymupdf.Rect(300, 100, 420, 150), color=(0, 0, 0), width=1)
            page.draw_line(pymupdf.Point(220, 125), pymupdf.Point(300, 125), color=(0, 0, 0), width=1)
            page.insert_text((112, 123), "Node A")
            page.insert_text((312, 123), "Node B")
    doc.save(path)
    doc.close()


def _make_decoration_pdf(path: str) -> None:
    """只有细横线/小方块（装饰）的 PDF → 不应提取出结构图。"""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_line(pymupdf.Point(50, 100), pymupdf.Point(550, 100), color=(0, 0, 0), width=1)  # 细横线
    page.draw_rect(pymupdf.Rect(200, 200, 240, 240), color=(0, 0, 0), width=1)  # 40×40 小方块
    page.insert_text((72, 60), "Body only, no diagram.")
    doc.save(path)
    doc.close()


def _make_mixed_pdf(path: str) -> None:
    """位图 + 其上矢量标注（同一区域）→ 去重后只保留位图 figure。"""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50))
    pix.clear_with(200)  # 纯色小位图
    png = pix.tobytes("png")
    page.insert_image(pymupdf.Rect(100, 100, 420, 220), stream=png)
    page.draw_rect(pymupdf.Rect(140, 120, 260, 170), color=(0, 0, 0), width=1)
    page.draw_rect(pymupdf.Rect(280, 120, 400, 170), color=(0, 0, 0), width=1)
    page.insert_text((150, 145), "Label over bitmap")
    doc.save(path)
    doc.close()


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_diagram_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _upload(client, fname: str, maker, **kw) -> int:
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        maker(fpath, **kw)
        content = open(fpath, "rb").read()
    finally:
        os.unlink(fpath)
    r = client.post("/api/v1/documents", files={"file": (fname, content, "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()


# ---------- 1. 提取器单元测试 ----------

def test_extract_vector_diagram_with_labels_and_page():
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_diagram_pdf(fpath)
        diagrams = DiagramExtractor().extract(fpath)
    finally:
        os.unlink(fpath)
    assert len(diagrams) == 1, f"应提取到 1 个结构图，实际 {len(diagrams)}"
    d = diagrams[0]
    assert d.page == 2, f"结构图应在第 2 页，实际 page={d.page}"
    assert "Node A" in d.labels and "Node B" in d.labels
    assert "Node A" in d.text and "Node B" in d.text
    x0, y0, x1, y1 = d.bbox
    assert x1 - x0 >= 300 and y1 - y0 >= 40  # 覆盖两个节点矩形
    assert 0 not in [d.page]


def test_decoration_and_small_shapes_ignored():
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_decoration_pdf(fpath)
        diagrams = DiagramExtractor().extract(fpath)
    finally:
        os.unlink(fpath)
    assert diagrams == [], f"装饰线/小方块不应被当作结构图，实际 {diagrams}"


def test_multi_page_only_diagram_page_extracted():
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_diagram_pdf(fpath, n_pages=3, diagram_page=3)
        diagrams = DiagramExtractor().extract(fpath)
    finally:
        os.unlink(fpath)
    assert len(diagrams) == 1 and diagrams[0].page == 3


def test_extract_saves_png_when_image_dir_given(tmp_path):
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_diagram_pdf(fpath)
        img_dir = tmp_path / "imgs"
        diagrams = DiagramExtractor().extract(fpath, image_dir=img_dir)
    finally:
        os.unlink(fpath)
    assert diagrams and (img_dir / diagrams[0].filename).exists()
    assert (img_dir / diagrams[0].filename).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------- 2. 上传链路集成 ----------

def test_upload_diagram_meta_and_endpoint(client):
    body = _upload(client, "diagram.pdf", _make_diagram_pdf)
    assert body["status"] == "ready"
    assert body["meta"]["num_diagrams"] >= 1
    doc_id = body["id"]
    lst = client.get(f"/api/v1/documents/{doc_id}/diagrams")
    assert lst.status_code == 200
    items = lst.json()
    assert isinstance(items, list) and len(items) >= 1
    one = items[0]
    assert {"page", "bbox", "labels", "file", "text"} <= set(one.keys())
    assert one["page"] == 2
    assert "Node A" in one["labels"]
    img = client.get(f"/api/v1/documents/{doc_id}/images/{one['file']}")
    assert img.status_code == 200
    assert img.headers["content-type"].startswith("image/png")


def test_upload_diagram_dedup_with_bitmap_figure(client):
    body = _upload(client, "mixed.pdf", _make_mixed_pdf)
    assert body["status"] == "ready"
    doc_id = body["id"]
    # 位图覆盖同一区域 → 去重后仅保留 figure，diagram 列表为空
    diagrams = client.get(f"/api/v1/documents/{doc_id}/diagrams").json()
    assert diagrams == [], f"位图区域不应重复生成 diagram，实际 {diagrams}"
    imgs = client.get(f"/api/v1/documents/{doc_id}/images").json()
    assert len(imgs) >= 1, "位图应保留为 figure"
