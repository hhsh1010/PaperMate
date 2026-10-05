"""跨文档合并检索：行为刻画 + 「部分失败可见」（2026-09-23，建议④）。

## 这份测试要守两件事

### 一、合并行为此前**零测试覆盖**
`CompositeRAGPipeline.query` 的合并语义（跨篇合并 → 按 score 降序 → 取 top_k）
在本次之前没有任何测试 —— 只有 `test_phase_a.py` 测过 `get_all_chunks` 的聚合。
行为没人守，就随时可能在重构中被改掉而无人察觉。这里把它**如实记录**下来。

其中「单篇可占满全部 top_k 名额」是**有意行为，不是缺陷**：
用户问某篇的内容时，把名额让给不相关篇目会**降低**回答质量。
（是否要按篇配额属"相关性 vs 覆盖度"的取舍，需要产品决策 —— 故此处只如实刻画，
不擅自改默认行为。）

### 二、部分失败此前**无声**
全失败会抛 `RetrievalUnavailable`（F7，有完整信号：`ok=False` + `retrieval_unavailable`
+ 面向模型的 note）。但**部分失败只 `logger.warning` 就继续** ——
调用方拿到「较少的结果」，与「文档里确实没有相关内容」**在数据上无法区分**。
用户于是以为检索了全部文档，而回答实际只基于其中一部分，模型也不会声明这一点。
这与 F7 要修的错**同一类**，只是从"全部"缩小到"部分"。
"""

from __future__ import annotations

import asyncio

import pytest

from lightcore.rag.pipeline import RetrievedChunk
from lightcore.tools.builtin.basic import _make_retrieve, _partial_failure_of
from papermate.services.chat_service import CompositeRAGPipeline


def _chunk(cid: str, score: float, text: str = "片段文本") -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text, page=1, score=score)


class _DocPipeline:
    """替身：按给定分数返回若干命中（模拟某一篇文档的检索管线）。"""

    def __init__(self, hits: list[RetrievedChunk] | None = None, boom: bool = False) -> None:
        self._hits = hits or []
        self._boom = boom
        self.calls = 0

    def query(self, query, top_k=3):
        self.calls += 1
        if self._boom:
            raise RuntimeError("索引损坏 / 嵌入服务异常")
        return list(self._hits)[:top_k]

    def get_chunks(self):
        return list(self._hits)


# --------------------------- 一、合并行为刻画 ---------------------------


def test_merges_across_docs_and_sorts_by_score_desc():
    """★ 跨篇合并 + 按 score **降序** + 只取 top_k。"""
    p1 = _DocPipeline([_chunk("a1", 0.9), _chunk("a2", 0.2)])
    p2 = _DocPipeline([_chunk("b1", 0.5)])
    got = CompositeRAGPipeline([p1, p2]).query("问题", top_k=2)
    assert [c.id for c in got] == ["a1", "b1"], "应按 score 降序取前 2"
    assert len(got) == 2


def test_one_doc_may_occupy_all_slots():
    """★ 记录**有意行为**：单篇可以占满全部 top_k 名额。

    这不是缺陷 —— 用户问某篇的内容时，把名额让给不相关篇目反而**降低**回答质量。
    强制按篇配额属"相关性 vs 覆盖度"的取舍，需产品决策；故此处只**钉住现状**，
    以免将来有人"顺手"改成配额而无人知情。
    """
    p1 = _DocPipeline([_chunk("a1", 0.9), _chunk("a2", 0.8), _chunk("a3", 0.7)])
    p2 = _DocPipeline([_chunk("b1", 0.1)])
    got = CompositeRAGPipeline([p1, p2]).query("问题", top_k=3)
    assert [c.id for c in got] == ["a1", "a2", "a3"], "单篇占满即为当前（有意）行为"


def test_no_pipelines_returns_empty():
    """无关联文档 → 空结果（且**不**算失败）。"""
    assert CompositeRAGPipeline([]).query("问题") == []


# --------------------------- 二、部分失败可见 ---------------------------


def test_partial_failure_is_recorded_with_doc_ids():
    """★★ 核心：部分失败必须**被记录**，且能报出**具体是哪几篇**。"""
    good = _DocPipeline([_chunk("a1", 0.9)])
    bad = _DocPipeline(boom=True)
    pipe = CompositeRAGPipeline([good, bad], doc_ids=[10, 15])
    got = pipe.query("问题")
    assert [c.id for c in got] == ["a1"], "可用结果照常返回（F7 语义不变）"
    info = pipe.last_partial_failure
    assert info is not None, "部分失败不得静默"
    assert info["failed_doc_ids"] == [15], "要能指出具体是哪一篇"
    assert info["failed_count"] == 1 and info["total"] == 2


def test_no_failure_clears_the_record():
    """★ 上一轮失败不得残留到本轮无失败的查询（否则会误报「检索不完整」）。"""
    good = _DocPipeline([_chunk("a1", 0.9)])
    bad = _DocPipeline(boom=True)
    pipe = CompositeRAGPipeline([good, bad], doc_ids=[10, 15])
    pipe.query("问题")
    assert pipe.last_partial_failure is not None
    bad._boom = False  # 故障恢复
    pipe.query("问题")
    assert pipe.last_partial_failure is None, "恢复后必须清空，否则会持续误报"


def test_without_doc_ids_reports_count_only():
    """不传 doc_ids 时仍报失败**数量**（ids 为空）——不能因为拿不到 id 就退回静默。"""
    pipe = CompositeRAGPipeline([_DocPipeline([_chunk("a1", 0.9)]), _DocPipeline(boom=True)])
    pipe.query("问题")
    info = pipe.last_partial_failure
    assert info is not None and info["failed_count"] == 1 and info["failed_doc_ids"] == []


def test_all_failure_raises_and_does_not_record_partial():
    """全失败走 F7 的异常路径（那条路径已有专门信号），不得再记成「部分失败」。"""
    from lightcore.rag.pipeline import RetrievalUnavailable

    pipe = CompositeRAGPipeline([_DocPipeline(boom=True), _DocPipeline(boom=True)], doc_ids=[1, 2])
    with pytest.raises(RetrievalUnavailable):
        pipe.query("问题")
    assert pipe.last_partial_failure is None


# --------------------------- 三、工具层如实告知 ---------------------------


def _retrieve(pipeline, query="问题"):
    handler = _make_retrieve(pipeline)
    return asyncio.run(handler(query))


def test_retrieve_marks_partial_failure_and_keeps_ok_true():
    """★★ 工具输出必须带 `partial_failure` + 面向模型的 note；且 **`ok` 仍为 True**。

    为何不改 `ok`：部分结果**仍然可用**。改 `ok=False` 会把它计入工具失败预算，
    可能触发 `max_tool_failures` 提前中止 —— 把「覆盖不全」升级成「整轮失败」。
    """
    pipe = CompositeRAGPipeline(
        [_DocPipeline([_chunk("a1", 0.9)]), _DocPipeline(boom=True)], doc_ids=[10, 15]
    )
    res = _retrieve(pipe)
    assert res.ok is True, "部分失败 ≠ 工具失败（数据仍可用）"
    assert res.output["partial_failure"]["failed_doc_ids"] == [15]
    assert res.output["partial_failure"]["searched_count"] == 1
    assert "本次检索不完整" in res.output["note"]
    assert "不要谎称已检索全部文档" in res.output["note"]


def test_retrieve_has_no_note_when_all_docs_succeed():
    """全成功 → **不得**出现任何"不完整"提示（避免噪声与警示疲劳）。"""
    pipe = CompositeRAGPipeline(
        [_DocPipeline([_chunk("a1", 0.9)]), _DocPipeline([_chunk("b1", 0.5)])], doc_ids=[10, 15]
    )
    res = _retrieve(pipe)
    assert "partial_failure" not in res.output
    assert "本次检索不完整" not in (res.output.get("note") or "")


def test_partial_failure_survives_multi_query_wrapper():
    """★★ `features.multi_query` 开启时管线被包一层，失败信息挂在内层 composite 上。

    若不穿透包装，这条最常用的路径上修复会**静默失效** —— 那比不修更糟
    （看起来已修，实际没有）。
    """
    from lightcore.rag.multi_query import MultiQueryRAGPipeline

    class _Expander:
        async def expand(self, query):
            return [query]

    inner = CompositeRAGPipeline(
        [_DocPipeline([_chunk("a1", 0.9)]), _DocPipeline(boom=True)], doc_ids=[10, 15]
    )
    wrapped = MultiQueryRAGPipeline(inner, _Expander())
    res = _retrieve(wrapped)
    assert res.output["partial_failure"]["failed_doc_ids"] == [15], "必须穿透包装读到内层信息"
    assert "本次检索不完整" in res.output["note"]


def test_helper_returns_none_for_unknown_pipeline():
    """未知结构（无该属性）→ `None`，且不得因结构异常而抛/死循环。"""

    class _Weird:
        def query(self, q, top_k=3):
            return []

    assert _partial_failure_of(_Weird()) is None
    assert _partial_failure_of(None) is None
