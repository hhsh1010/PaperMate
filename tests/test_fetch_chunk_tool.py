"""`fetch_chunk` 工具（retrieve 被压缩后按 id 取全文）。

★ 背景（2026-10-23）：`retrieve` 每段 text 截断到 ≤200 字，但**保留 id**。
此前模型只能换关键词重检，拿不回原段 ⇒ 容易"以为文档就这么多"（静默降准）。
本文件钉住：id 解析规则与 retrieve 一致、能取到全文、不存在时明确报错。
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from lightcore.tools.spec import ToolRegistry
from papermate.models.db import Base, Chunk, Document
from papermate.services.doc_tools import parse_chunk_id, register_doc_tools


@pytest.fixture()
def sf(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'c.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        db.add(Document(id=3, filename="a.pdf", storage_path="x", status="parsed", content_hash=""))
        db.add(Chunk(document_id=3, seq=12, content="完整原文" * 400, page=7, heading="方法"))
        db.commit()
    return factory


def test_parse_matches_retrieve_id_format() -> None:
    assert parse_chunk_id("doc-3-c12") == (3, 12)
    assert parse_chunk_id("doc-12-c34") == (12, 34)
    assert parse_chunk_id(" doc-1-c8 ") == (1, 8)
    assert parse_chunk_id("bad") is None
    assert parse_chunk_id("") is None
    assert parse_chunk_id("doc-x-c1") is None


def test_fetch_returns_full_text(sf) -> None:
    reg = ToolRegistry()
    register_doc_tools(reg, sf)
    res = asyncio.run(reg.get("fetch_chunk").handler(chunk_id="doc-3-c12"))
    assert res.ok is True
    out = res.output
    assert out["page"] == 7 and out["heading"] == "方法"
    assert len(out["text"]) == len("完整原文" * 400)   # ★ 全文，未截断


def test_missing_chunk_gives_clear_error(sf) -> None:
    reg = ToolRegistry()
    register_doc_tools(reg, sf)
    res = asyncio.run(reg.get("fetch_chunk").handler(chunk_id="doc-3-c999"))
    assert res.ok is False
    assert "找不到" in (res.error or "")


def test_bad_id_rejected(sf) -> None:
    reg = ToolRegistry()
    register_doc_tools(reg, sf)
    res = asyncio.run(reg.get("fetch_chunk").handler(chunk_id="oops"))
    assert res.ok is False
    assert "格式" in (res.error or "")
