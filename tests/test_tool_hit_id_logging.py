"""命中 id 日志必须记录**完整**列表 —— 2026-09-28 补的观测契约。

### 为什么补这条日志
「同一批 chunk 被反复取回多少次」是**去重换策略**的唯一量化依据。
而改造前：

- `retrieve` 只记 `（第一次命中 doc-13-c16）` —— **只记第一条**；
- `pdf_extract` **完全没有命中日志**（而它是不压缩注入的最大来源）。

⇒ 只看第一条既算不出重合度、也分不清「不同 query 命中同一批块」这种情况。
实测（09-28 评估）单请求原始命中 **74,531 字 / 语料 54,884 字 = 136%**，
重复下限 36% —— 但**精确值算不出来**，就是因为缺这份 id 列表。

### 本测试守什么
三个取数入口的日志都必须包含**全部**命中 id：
`retrieve` · `pdf_extract(page=)` · `pdf_extract(query=)`。

⚠️ 这些日志**只用于观测**，不得参与任何业务分支（本测试也仅断言日志内容，
不触碰返回值——返回值由既有测试覆盖）。
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from lightcore.tools.builtin.basic import _make_pdf_extract, _make_retrieve

LOGGER_NAME = "lightcore.tools.builtin.basic"
_IDS = ("doc-1-c0", "doc-1-c7", "doc-1-c23")


class _Chunk:
    """最小 chunk 替身（`_all_chunks` / page 模式按属性取用）。"""

    def __init__(self, cid: str, page: int = 1) -> None:
        self.id = cid
        self.page = page
        self.text = f"正文 {cid}"
        self.score = 0.5


class _Pipe:
    """最小管线替身：同步 `query` + `get_all_chunks`。

    刻意返回**多于一条**结果 —— 只记第一条的旧实现会在这里被抓到。
    """

    def __init__(self, n: int = 3, page: int = 1) -> None:
        self._chunks = [_Chunk(cid, page) for cid in _IDS[:n]]

    def query(self, q, top_k: int = 6):  # noqa: ANN001, ARG002
        return list(self._chunks)

    def get_all_chunks(self):
        return list(self._chunks)


def _logs(caplog) -> str:  # noqa: ANN001
    return "\n".join(r.getMessage() for r in caplog.records if r.name == LOGGER_NAME)


def test_retrieve_logs_all_hit_ids(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    handler = _make_retrieve(_Pipe())
    asyncio.run(handler("术语", top_k=3))

    text = _logs(caplog)
    assert "命中 id=" in text, text
    for cid in _IDS:
        assert cid in text, f"命中 {cid} 未出现在日志里（是否只记了第一条？）：{text}"


def test_pdf_extract_page_logs_all_hit_ids(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    handler = _make_pdf_extract(_Pipe(page=8))
    asyncio.run(handler(page=8))

    text = _logs(caplog)
    assert "pdf_extract(page)" in text, text
    for cid in _IDS:
        assert cid in text, f"page 模式漏记 {cid}：{text}"


def test_pdf_extract_query_logs_all_hit_ids(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    handler = _make_pdf_extract(_Pipe())
    asyncio.run(handler(query="abstract", top_k=3))

    text = _logs(caplog)
    assert "pdf_extract(query)" in text, text
    for cid in _IDS:
        assert cid in text, f"query 模式漏记 {cid}：{text}"


def test_empty_hits_still_log_a_placeholder(caplog):
    """零命中也要留痕（`无`）—— 否则"这次取了但没命中"与"根本没调用"在日志里无法区分。"""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    class _Empty(_Pipe):
        def query(self, q, top_k: int = 6):  # noqa: ANN001, ARG002
            return []

    asyncio.run(_make_retrieve(_Empty())("无此内容", top_k=3))
    assert "命中 id=无" in _logs(caplog)


@pytest.mark.parametrize("name", ["retrieve", "pdf_extract"])
def test_logging_does_not_change_tool_result(name):
    """观测代码**不得改变被观测对象** —— 日志只上报，返回值必须原样。"""
    if name == "retrieve":
        res = asyncio.run(_make_retrieve(_Pipe())("q", top_k=3))
    else:
        res = asyncio.run(_make_pdf_extract(_Pipe(page=8))(page=8))
    assert res.ok is True
    assert res.output
