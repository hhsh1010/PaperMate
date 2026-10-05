"""LightCore · RAG 子系统单元测试（docs/08 单元层 / M3）"""

from __future__ import annotations

import asyncio

from lightcore.rag.embedder import FakeHashEmbedder
from lightcore.rag.pipeline import RAGPipeline
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry

SAMPLE = """
LightCore 是一种从零自研的轻量级 Agent 内核。它不依赖 LangChain 或 AutoGen 等高层框架。
Agent 循环采用 ReAct 与 Function Calling 双协议。状态机包含思考、行动、观察、错误处理、中止与完成。
记忆系统分为短期与长期。短期是滚动窗口并做摘要压缩；长期是向量记忆，按相关、重要与时效打分召回。
生产级 RAG 使用 BGE-M3 与 BM25 混合检索，再经 Cross-Encoder 重排，最后给出带引用的回答。
"""


def test_pipeline_add_and_query():
    pipe = RAGPipeline(embedder=FakeHashEmbedder(dim=32))
    n = pipe.add_document(SAMPLE, doc_id="paper")
    assert n > 0, "应至少切出 1 个块"
    results = pipe.query("LightCore 的核心贡献是什么", top_k=2)
    assert results, "应返回检索结果"
    # 相关块应排前：应包含 'Agent 内核' 或 '从零自研'
    assert any("内核" in r.text or "自研" in r.text for r in results)


def test_pipeline_citation_has_id_and_page():
    pipe = RAGPipeline(embedder=FakeHashEmbedder(dim=32))
    pipe.add_document(SAMPLE, doc_id="paper")
    r = pipe.query("RAG 怎么重排", top_k=1)[0]
    # 纯文本无物理页码 → page 为 None；PDF 场景应为 ≥1（见 normalize_page）
    assert r.id and (r.page is None or r.page >= 1) and r.score is not None


def test_retrieve_tool_with_pipeline():
    pipe = RAGPipeline(embedder=FakeHashEmbedder(dim=32))
    pipe.add_document(SAMPLE, doc_id="paper")
    reg = ToolRegistry()
    register_builtin_tools(reg, rag_pipeline=pipe)
    exe = ToolExecutor(reg)
    res = asyncio.run(exe.execute("retrieve", {"query": "记忆系统", "top_k": 2}))
    assert res.ok
    assert res.output["source"] == "rag"
    assert res.output["results"]


def test_retrieve_tool_page_normalized_to_none_for_text_doc():
    """docx/md/txt 等无物理页码文档 → 工具结果 page=None（不再把 0 当页码喂给 LLM）。"""
    pipe = RAGPipeline(embedder=FakeHashEmbedder(dim=32))
    pipe.add_document(SAMPLE, doc_id="paper")  # SAMPLE 是纯文本 → 块 page=None
    reg = ToolRegistry()
    register_builtin_tools(reg, rag_pipeline=pipe)
    exe = ToolExecutor(reg)
    res = asyncio.run(exe.execute("retrieve", {"query": "记忆系统", "top_k": 2}))
    assert res.ok and res.output["results"]
    for r in res.output["results"]:
        assert r["page"] is None  # 无物理页码，绝不输出 0


def test_retrieve_tool_no_pipeline_refuses():
    reg = ToolRegistry()
    register_builtin_tools(reg)  # 无 rag_pipeline（会话未关联文档）
    exe = ToolExecutor(reg)
    res = asyncio.run(exe.execute("retrieve", {"query": "LightCore", "top_k": 1}))
    # 不再回退假数据：明确 source=none 且 results 为空，避免给出与文档无关的答案
    assert res.ok
    assert res.output["source"] == "none"
    assert res.output["results"] == []
    assert res.output.get("note")
