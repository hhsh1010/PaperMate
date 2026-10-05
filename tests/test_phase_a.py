"""Phase A 实现验证（S1/S2/S7 核心 RAG）

覆盖：
- P0-2 检索注入截断（_compact_retrieve_output / _format_tool_result）
- P0-3 上下文摘要压缩（AgentLoop._compact_history：Fake 滚动截断 + 真实 LLM 摘要）
- 新增工具 pdf_extract / citation_lookup（含无文档降级）
- RAGPipeline.get_chunks / CompositeRAGPipeline.get_all_chunks 聚合
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from lightcore.core.loop import (
    AgentLoop,
    LoopConfig,
    _compact_retrieve_output,
    _format_tool_result,
)
from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, Role, _rough_tokens
from lightcore.llm.fake import FakeLLM
from lightcore.rag.embedder import FakeHashEmbedder
from lightcore.rag.pipeline import RAGPipeline
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry


# ---------------- P0-2 检索注入截断 ----------------
def test_compact_retrieve_output_truncates_and_keeps_meta():
    out = {
        "query": "x",
        "source": "rag",
        "results": [
            {"id": "doc-1-c0", "page": 2, "text": "A" * 500, "score": 0.9},
            {"id": "doc-1-c1", "page": 3, "text": "short", "score": 0.5},
        ],
    }
    c = _compact_retrieve_output(out)
    assert c["results"][0]["text"] == "A" * 200 + "…"
    assert c["results"][0]["id"] == "doc-1-c0"
    assert c["results"][0]["page"] == 2
    assert c["results"][0]["score"] == 0.9
    assert c["results"][1]["text"] == "short"  # 未超长不截断

    # 序列化格式保持可被 ast.literal_eval 解析（与 chat_service._parse_result 对齐）
    import ast

    s = _format_tool_result("retrieve", out)
    assert s.startswith("RESULT: ")
    parsed = ast.literal_eval(s[len("RESULT: "):])
    assert parsed["results"][0]["text"].endswith("…")
    # 非 retrieve 工具原样包裹
    assert _format_tool_result("noop", "abc") == "RESULT: abc"


# ---------------- P0-3 上下文摘要压缩 ----------------
def test_compact_history_rolling_for_fake():
    async def run():
        big = "内容" * 2000  # 单条约 1600 tokens
        history = [ChatMessage(role=Role.USER, content=big) for _ in range(5)]
        loop = AgentLoop(FakeLLM(), ToolExecutor(ToolRegistry()), config=LoopConfig(max_tokens_budget=8000))
        compacted = await loop._compact_history(history)
        est = sum(_rough_tokens(m.content) for m in compacted)
        assert est <= 4000  # cap = max_tokens_budget // 2
        assert len(compacted) < len(history)

    asyncio.run(run())


def test_compact_history_keeps_pinned_and_summarizes_for_real_llm():
    class StubLLM(BaseLLM):
        model_name = "stub"

        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
            return LLMResponse(
                message=ChatMessage(role=Role.ASSISTANT, content="摘要：用户问了注意力机制。"),
                prompt_tokens=1, completion_tokens=1, model="stub",
            )

    async def run():
        big = "内容" * 2000
        history = [
            ChatMessage(role=Role.SYSTEM, content="【长期记忆】用户偏好中文"),
            ChatMessage(role=Role.USER, content=big),
            ChatMessage(role=Role.USER, content=big),
            ChatMessage(role=Role.USER, content=big),
            ChatMessage(role=Role.USER, content=big),
        ]
        loop = AgentLoop(StubLLM(), ToolExecutor(ToolRegistry()), config=LoopConfig(max_tokens_budget=8000))
        compacted = await loop._compact_history(history)
        # 置顶 SYSTEM 注记（长期记忆）应保留
        assert any("长期记忆" in (m.content or "") for m in compacted)
        # 真实 LLM 应生成【历史摘要】注记（Fake 则跳过）
        assert any("历史摘要" in (m.content or "") for m in compacted)

    asyncio.run(run())


# ---------------- 新增工具 pdf_extract / citation_lookup ----------------
def _stub_pipeline(chunks):
    return SimpleNamespace(get_all_chunks=lambda: chunks)


def test_pdf_extract_and_citation_lookup():
    chunks = [
        SimpleNamespace(id="doc-3-c0", page=1, text="注意力机制改进了 Transformer。"),
        SimpleNamespace(id="doc-3-c1", page=2, text="消融实验显示模块 A 贡献最大。"),
        SimpleNamespace(id="doc-7-c0", page=1, text="另一篇论文的内容。"),
    ]
    reg = ToolRegistry()
    register_builtin_tools(reg, rag_pipeline=_stub_pipeline(chunks))
    exe = ToolExecutor(reg)
    assert "pdf_extract" in reg.names() and "citation_lookup" in reg.names()

    # pdf_extract 按页提取
    r = asyncio.run(exe.execute("pdf_extract", {"page": 1}))
    assert r.ok and r.output["source"] == "pdf"
    assert "注意力" in r.output["text"]
    assert r.output["num_chunks"] == 2  # doc-3-c0 与 doc-7-c0 同为第 1 页

    # citation_lookup 核实引用元数据（防幻觉）
    c = asyncio.run(exe.execute("citation_lookup", {}))
    assert c.ok and c.output["source"] == "pdf"
    docs = {d["doc_id"]: d for d in c.output["docs"]}
    assert "doc-3" in docs and "doc-7" in docs
    assert docs["doc-3"]["num_chunks"] == 2
    assert docs["doc-3"]["page_min"] == 1 and docs["doc-3"]["page_max"] == 2


def test_pdf_extract_citation_lookup_no_doc_degrade():
    reg = ToolRegistry()
    register_builtin_tools(reg)  # 未关联文档（rag_pipeline=None）
    exe = ToolExecutor(reg)
    r = asyncio.run(exe.execute("pdf_extract", {"query": "x"}))
    assert r.ok and r.output["source"] == "none" and r.output.get("note")
    c = asyncio.run(exe.execute("citation_lookup", {}))
    assert c.ok and c.output["source"] == "none" and c.output["docs"] == []


# ---------------- RAG 聚合方法 ----------------
def test_rag_pipeline_get_chunks():
    p = RAGPipeline(embedder=FakeHashEmbedder())
    n = p.add_document("第一段。\n\n第二段。\n\n第三段。", doc_id="doc-1")
    assert n > 0
    chunks = p.get_chunks()
    assert len(chunks) == n
    assert all(c.id.startswith("doc-1-c") for c in chunks)


def test_composite_get_all_chunks():
    from papermate.services.chat_service import CompositeRAGPipeline

    p1 = RAGPipeline(embedder=FakeHashEmbedder())
    p1.add_document("A 段。\n\nB 段。", doc_id="doc-1")
    p2 = RAGPipeline(embedder=FakeHashEmbedder())
    p2.add_document("C 段。\n\nD 段。", doc_id="doc-2")
    comp = CompositeRAGPipeline([p1, p2])
    assert len(comp.get_all_chunks()) == len(p1.get_chunks()) + len(p2.get_chunks())
