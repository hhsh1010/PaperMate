"""结构图多模态识别（P1）测试：
- DiagramEnricher 单元：VLM 输出 → Mermaid + 描述拆分；失败/无代码块降级
- _split_mermaid 纯函数边界
- 集成：features.diagram_enrich 开关控制上传期识别（开 → /diagrams 带 mermaid；关 → 空）
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.rag.figure_content import DiagramEnricher, _split_mermaid


def _make_diagram_pdf(path: str) -> None:
    """含矢量结构图（两节点矩形 + 连线 + 文字标签）的 PDF，图在第 2 页。"""
    import pymupdf

    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 60), "Page 1 body text.")
    page2 = doc.new_page()
    page2.insert_text((72, 60), "Page 2 body text.")
    page2.draw_rect(pymupdf.Rect(100, 100, 220, 150), color=(0, 0, 0), width=1)
    page2.draw_rect(pymupdf.Rect(300, 100, 420, 150), color=(0, 0, 0), width=1)
    page2.draw_line(pymupdf.Point(220, 125), pymupdf.Point(300, 125), color=(0, 0, 0), width=1)
    page2.insert_text((112, 123), "Node A")
    page2.insert_text((312, 123), "Node B")
    doc.save(path)
    doc.close()


class _FakeVLM:
    """注入 DiagramEnricher 的假多模态 LLM：chat 返回预设 content。"""

    def __init__(self, content: str) -> None:
        self._content = content

    async def chat(self, messages, **kwargs):  # noqa: ANN001
        return SimpleNamespace(message=SimpleNamespace(content=self._content))


def test_enrich_splits_mermaid_and_description(tmp_path):
    import pymupdf

    pdf = tmp_path / "d.pdf"
    _make_diagram_pdf(str(pdf))
    with pymupdf.open(str(pdf)) as doc:
        pix = doc[1].get_pixmap()
        png = tmp_path / "d.png"
        png.write_bytes(pix.tobytes("png"))
    vlm = _FakeVLM(
        "```mermaid\ngraph TD\nA[Node A] --> B[Node B]\n```\n【描述】这是一个示例结构图。"
    )
    out = asyncio.run(DiagramEnricher(vlm_llm=vlm).enrich(png))
    assert out["mermaid"] == "graph TD\nA[Node A] --> B[Node B]"
    assert "示例结构图" in out["description"]


def test_enrich_degrades_when_no_mermaid_block(tmp_path):
    png = tmp_path / "x.png"
    png.write_bytes(b"not-really-png")
    out = asyncio.run(DiagramEnricher(vlm_llm=_FakeVLM("无法识别")).enrich(png))
    assert out["mermaid"] == ""
    assert out["description"] == "无法识别"


def test_enrich_missing_file(tmp_path):
    out = asyncio.run(DiagramEnricher(vlm_llm=_FakeVLM("x")).enrich(tmp_path / "nope.png"))
    assert out == {"mermaid": "", "description": ""}


def test_split_mermaid_boundaries():
    assert _split_mermaid("```mermaid\ngraph TD\nA-->B\n```\n描述文字") == {
        "mermaid": "graph TD\nA-->B",
        "description": "描述文字",
    }
    assert _split_mermaid("无代码块") == {"mermaid": "", "description": "无代码块"}
    assert _split_mermaid("```mermaid\nX\n```\n【描述】标题")["description"] == "标题"


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_diagram_enrich_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _upload_diagram_pdf(client) -> int:
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_diagram_pdf(fpath)
        content = open(fpath, "rb").read()
    finally:
        os.unlink(fpath)
    r = client.post("/api/v1/documents", files={"file": ("diagram.pdf", content, "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_diagram_enrich_off_by_default(client, monkeypatch):
    """默认关闭：/diagrams 的 mermaid/description 为空（不产生多模态调用）。"""
    from lightcore.rag.figure_content import DiagramEnricher

    called = {"n": 0}

    async def _fake_enrich(self, image_path):  # noqa: ANN001
        called["n"] += 1
        return {"mermaid": "x", "description": "y"}

    monkeypatch.setattr(DiagramEnricher, "enrich", _fake_enrich)
    doc_id = _upload_diagram_pdf(client)
    items = client.get(f"/api/v1/documents/{doc_id}/diagrams").json()
    assert items, "应提取到结构图"
    assert called["n"] == 0, "开关关闭时不应触发 VLM 识别"
    assert items[0]["mermaid"] == "" and items[0]["description"] == ""


def test_diagram_enrich_on_saves_mermaid(client, monkeypatch):
    """开启 features.diagram_enrich：上传后 /diagrams 返回 Mermaid 与描述。"""
    from lightcore.rag.figure_content import DiagramEnricher

    rs_mod.update({"features": {"diagram_enrich": True}})

    async def _fake_enrich(self, image_path):  # noqa: ANN001
        return {"mermaid": "graph TD\nA-->B", "description": "识别出的结构图"}

    monkeypatch.setattr(DiagramEnricher, "enrich", _fake_enrich)
    doc_id = _upload_diagram_pdf(client)
    items = client.get(f"/api/v1/documents/{doc_id}/diagrams").json()
    assert items
    assert items[0]["mermaid"] == "graph TD\nA-->B"
    assert "识别出的结构图" in items[0]["description"]
