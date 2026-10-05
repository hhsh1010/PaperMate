"""`pdf_extract` 多文档隔离（2026-10-03）。

## 修的缺陷
`page` 模式原先是 `sel = [c for c in chunks if c.page == norm]` ⇒ **不区分文档**。
实测：在绑定 **8 篇**文档的会话里，`page=9` 会把 **6 篇**文档的第 9 页拼在一起
（22~38 chunks），再撞 `_PDF_EXTRACT_MAX_TOTAL = 4000` 被截断
⇒ 模型拿到「**多篇混排 + 被截断**」的内容 ⇒ **静默降准**。

## 修法
1. 新增 `doc_id` 参数（**与 schema 同源** —— `ToolExecutor` 是 `handler(**arguments)`，
   不会套用 schema 的 default，声明与实现不一致会直接 `TypeError`）；
2. 未指定 `doc_id` 但**实际命中多篇**时，返回值带 `note` **如实告知是混排**。
   ⚠️ 判据用**实际命中的文档数**而非"会话绑了几篇" —— 只有真拼上了才提示，
   否则单文档会话会**误报**。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lightcore.tools.builtin.basic import _doc_id_of, _make_pdf_extract


class _Pipe:
    """最小 pipeline 替身（`_all_chunks` 只要求 `get_all_chunks`）。"""

    def __init__(self, chunks):
        self._chunks = chunks

    def get_all_chunks(self):
        return self._chunks


def _c(doc: int, seq: int, page: int, text: str):
    return SimpleNamespace(id=f"doc-{doc}-c{seq}", page=page, text=text, doc_id=doc)


#: 两篇文档都有第 1 页与第 9 页（复现真实的跨文档命中）
CHUNKS = [
    _c(10, 0, 1, "A文档-第1页-片段1"),
    _c(10, 1, 1, "A文档-第1页-片段2"),
    _c(10, 2, 9, "A文档-第9页-片段"),
    _c(13, 0, 1, "B文档-第1页-片段1"),
    _c(13, 1, 9, "B文档-第9页-片段"),
    _c(13, 2, 5, "B文档-第5页-片段"),
]


class TestDocIdOf:
    def test_parses_doc_number(self):
        assert _doc_id_of(SimpleNamespace(id="doc-16-c7")) == 16

    @pytest.mark.parametrize("bad", [None, "", "weird", "doc-x-c1", "doc--c1"])
    def test_never_raises(self, bad):
        """★ 容错优先：解析不出返回 `None`，**绝不抛**（观测/分流辅助不该弄崩调用）。"""
        assert _doc_id_of(SimpleNamespace(id=bad)) is None


class TestMultiDocIsolation:
    def _call(self, **kw):
        return asyncio.run(_make_pdf_extract(_Pipe(CHUNKS))(page=kw.pop("page", 1), **kw))

    def test_doc_id_filters_to_single_document(self):
        """★ 核心：指定 doc_id ⇒ 只返回该文档的片段（不再混排）。"""
        out = self._call(page=1, doc_id=13)
        assert out.output["num_chunks"] == 1
        assert "B文档" in out.output["text"]
        assert "A文档" not in out.output["text"]

    def test_no_note_when_single_document_requested(self):
        """指定了 doc_id ⇒ 已单篇，**不该**再报"混排"（否则是误报噪音）。"""
        out = self._call(page=1, doc_id=13)
        assert "混排" not in (out.output.get("note") or "")

    def test_multi_doc_without_doc_id_is_flagged(self):
        """★ 未指定且**实际命中多篇** ⇒ 必须如实告知是混排（不许静默给拼接）。"""
        out = self._call(page=1)
        assert out.output["num_chunks"] == 3  # 两篇的第 1 页合起来
        note = out.output.get("note") or ""
        assert "混排" in note
        assert "10" in note and "13" in note

    def test_single_hit_page_is_not_flagged(self):
        """★ 第 5 页只有一篇有 ⇒ **不该**报混排（判据是实际命中数，不是会话文档数）。"""
        out = self._call(page=5)
        assert out.output["num_chunks"] == 1
        assert not (out.output.get("note") or "")

    def test_unknown_doc_id_is_explicit(self):
        """doc_id 不在会话里 ⇒ 明确说明（而不是静默空结果）。"""
        out = self._call(page=1, doc_id=999)
        assert out.output["num_chunks"] == 0
        assert "999" in (out.output.get("note") or "")

    def test_page_without_doc_id_keeps_original_behavior(self):
        """未指定 doc_id 时**行为不变**（仍是取该页全部片段）—— 这是向后兼容的底线。"""
        out = self._call(page=9)
        assert out.output["num_chunks"] == 2
        assert "A文档-第9页" in out.output["text"]
        assert "B文档-第9页" in out.output["text"]


class TestSchemaSync:
    def test_schema_declares_doc_id(self):
        """★ schema 与实现**必须同源**：`ToolExecutor` 直接 `handler(**arguments)`，
        不套用 schema 的 default；实现加了参数而 schema 没声明 ⇒ 模型根本传不进来。"""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / "src" / "lightcore" / "tools" / "builtin" / "basic.py"
        ).read_text(encoding="utf-8")
        assert '"doc_id": {"type": "integer"' in src
        assert "doc_id: int | None = None," in src

    def test_schema_doc_id_description_mentions_multi_doc(self):
        """★ 描述里要说明"多文档必填"—— 否则模型不知道该传什么。"""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / "src" / "lightcore" / "tools" / "builtin" / "basic.py"
        ).read_text(encoding="utf-8")
        i = src.index('"doc_id": {"type": "integer"')
        assert "多文档" in src[i:i + 400]
