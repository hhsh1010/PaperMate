"""检索故障（F7）回归：全部检索管线失败 ≠ 文档无相关内容。

缺陷背景：CompositeRAGPipeline.query 对每个子管线 `except Exception: continue`，
全部失败时静默返回 []，与「检索到但无命中」在数据结构上无法区分。
上层据此按系统提示词回答「知识库中没有相关内容」，把**系统故障**误报成
"文档里没有"，诱导用户重复上传而不是排查索引/嵌入服务。
"""

from __future__ import annotations

import pytest

from papermate.services.chat_service import CompositeRAGPipeline


class _BoomPipeline:
    """恒定失败的管线替身（模拟索引损坏 / 嵌入或重排服务异常）。"""

    def query(self, query, top_k=3):
        raise RuntimeError("索引损坏 / 嵌入服务异常")

    def get_chunks(self):
        return []


class _OkPipeline:
    """正常但无命中的管线替身（检索成功、结果为空）。"""

    def query(self, query, top_k=3):
        return []

    def get_chunks(self):
        return []


def test_all_pipelines_failed_raises_retrieval_unavailable():
    """全部管线失败 → 抛 RetrievalUnavailable，不得静默返回空列表。"""
    from lightcore.rag.pipeline import RetrievalUnavailable

    pipe = CompositeRAGPipeline([_BoomPipeline(), _BoomPipeline()])
    with pytest.raises(RetrievalUnavailable):
        pipe.query("任意问题")


def test_partial_failure_does_not_fail_whole_retrieval():
    """部分管线失败不拖垮整体：仍返回可用结果，且不抛异常。"""
    pipe = CompositeRAGPipeline([_BoomPipeline(), _OkPipeline()])
    assert pipe.query("任意问题") == [], "单篇故障应被跳过，整体仍可用（不抛异常）"


def test_no_pipeline_is_not_a_failure():
    """无关联文档（无管线）→ 正常空结果，不属于检索故障。"""
    pipe = CompositeRAGPipeline([])
    assert pipe.query("任意问题") == []
