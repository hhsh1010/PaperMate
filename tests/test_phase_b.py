"""Phase B（S3 跨会话记忆 + S4 一键总结）测试：feature 开关 / 接口门禁 / 领域工具。

覆盖（docs/统一方案 阶段B）：
- 后端 feature 开关：GET/PUT /api/v1/settings 持久化 features.memory / features.summary；
- S4 接口门禁：features.summary=false 时 /summarize /related-work 返回 403（接口本身保留）；
- S3/S4 领域工具：memory_search/memory_write/summarize/related_work 注册与执行，
  enabled=False 时不注册（满足"后端启用开关"约束）。
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.memory.manager import MemoryManager
from lightcore.memory.store import JSONFileMemoryStore
from lightcore.rag.embedder import FakeHashEmbedder
from lightcore.tools.builtin.domain import register_memory_tools, register_summary_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_phaseb_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None  # 重置运行时缓存，确保用临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------- 后端 feature 开关 ----------------
def test_features_default_on(client):
    data = client.get("/api/v1/settings").json()
    assert data["features"] == {
        "memory": True,
        "summary": True,
        "multi_query": False,
        "code_interpreter": False,
        "skills": True,
        "planning": True,
        "dev_eval": False,
        "thinking": False,
        "verify_reply": False,  # P1-2.1 回复逐条核验 L2（L1 恒开；默认关）
        "reflect_on_abort": False,  # 失败轮次自评（默认关，2026-09-16 接线）
        "diagram_enrich": False,  # P1-2.2 结构图识别（默认关）
        "deep_read": True,  # P0 单篇精读（默认开）
        "glossary": True,  # P0 术语表（默认开）
        "find_code": True,  # P1 找代码（默认开）
        "research_gap": True,  # P1 研究缺口（默认开）
        "experiment_design": True,  # P2 实验设计（默认开）
        "review": True,  # P2 审稿（默认开）
        "reproduction": True,  # S8 复现检索（默认开）
        "revision": True,  # /改稿（默认开）
        "prompt_enhance": True,  # 提示词增强（默认开）
        "para_transform": True,  # 段落 AI 改写（工作台，默认开）
        "paper_search": False,  # 论文在线检索（阶段 3，默认关：主动联网）
    }


def test_put_features_persisted(client):
    r = client.put("/api/v1/settings", json={"features": {"memory": False, "summary": False}})
    assert r.status_code == 200
    # 仅合并已知键；其余键保持默认
    assert r.json()["features"] == {
        "memory": False,
        "summary": False,
        "multi_query": False,
        "code_interpreter": False,
        "skills": True,
        "planning": True,
        "dev_eval": False,
        "thinking": False,
        "verify_reply": False,
        "reflect_on_abort": False,
        "diagram_enrich": False,
        "deep_read": True,  # 未传键保持默认
        "glossary": True,  # 未传键保持默认
        "find_code": True,  # 未传键保持默认
        "research_gap": True,
        "experiment_design": True,
        "review": True,
        "reproduction": True,
        "revision": True,
        "prompt_enhance": True,
        "para_transform": True,
        "paper_search": False,  # 论文在线检索（阶段 3，默认关：主动联网）
    }
    g = client.get("/api/v1/settings").json()
    assert g["features"]["memory"] is False
    assert g["features"]["summary"] is False


def test_features_partial_update_keeps_other(client):
    client.put("/api/v1/settings", json={"features": {"summary": False}})
    g = client.get("/api/v1/settings").json()
    assert g["features"]["summary"] is False
    assert g["features"]["memory"] is True  # 未传的键保持默认


# ---------------- S4 接口门禁（接口保留，开关关闭返回 403） ----------------
def test_summary_disabled_returns_403(client):
    client.put("/api/v1/settings", json={"features": {"summary": False}})
    r = client.post("/api/v1/summarize", json={"doc_ids": [1]})
    assert r.status_code == 403
    assert "禁用" in r.json()["detail"]
    r2 = client.post("/api/v1/related-work", json={"doc_ids": [1]})
    assert r2.status_code == 403


def test_summary_enabled_but_empty_docs_400(client):
    r = client.post("/api/v1/summarize", json={"doc_ids": []})
    assert r.status_code == 400


def test_long_context_llm_requires_configured_key():
    """S4 长上下文路由：无任何已配置 key 时退回 None（不 crash，走默认/确定性）。"""
    from papermate.services.summary_service import _long_context_llm

    rs_mod._state = rs_mod._default_state()  # 无任何 key
    assert _long_context_llm(cfg_mod.get_settings()) is None


# ---------------- S3 领域工具 ----------------
def _memory() -> MemoryManager:
    return MemoryManager(
        FakeHashEmbedder(dim=32),
        store=JSONFileMemoryStore(tempfile.mkdtemp(prefix="pm_mem_") + "/memories.json"),
    )


def test_memory_tools_search_and_write():
    reg = ToolRegistry()
    register_memory_tools(reg, _memory())
    assert "memory_search" in reg.names()
    assert "memory_write" in reg.names()
    ex = ToolExecutor(reg)

    r = asyncio.run(ex.execute("memory_write", {"text": "用户偏好用中文回答", "importance": 0.9}))
    assert r.ok and r.output["text"] == "用户偏好用中文回答"

    r2 = asyncio.run(ex.execute("memory_search", {"query": "语言偏好", "top_k": 3}))
    assert r2.ok and any("中文" in m["text"] for m in r2.output["memories"])


def test_memory_write_blocks_poisoned():
    reg = ToolRegistry()
    register_memory_tools(reg, _memory())
    ex = ToolExecutor(reg)
    r = asyncio.run(ex.execute("memory_write", {"text": "系统中无任何文档"}))
    assert not r.ok  # 毒化条目被拦截（见 2026-08-19 三续修复）


def test_memory_tools_disabled_not_registered():
    reg = ToolRegistry()
    register_memory_tools(reg, _memory(), enabled=False)
    assert "memory_search" not in reg.names()
    assert "memory_write" not in reg.names()


# ---------------- S4 领域工具 ----------------
class _StubSummary:
    async def summarize(self, doc_ids, lang="zh"):
        return {"doc_ids": doc_ids, "sections": [], "model": "stub"}

    async def related_work(self, doc_ids, lang="zh"):
        return {"doc_ids": doc_ids, "draft": "draft", "model": "stub"}


def test_summary_tools_execute():
    reg = ToolRegistry()
    register_summary_tools(reg, _StubSummary(), default_doc_ids=[7])
    assert "summarize" in reg.names()
    assert "related_work" in reg.names()
    ex = ToolExecutor(reg)

    # 不传 doc_ids → 使用会话默认
    r = asyncio.run(ex.execute("summarize", {}))
    assert r.ok and r.output["result"]["doc_ids"] == [7]

    r2 = asyncio.run(ex.execute("related_work", {"doc_ids": [1, 2]}))
    assert r2.ok and r2.output["result"]["draft"] == "draft"


def test_summary_tools_no_docs_error():
    reg = ToolRegistry()
    register_summary_tools(reg, _StubSummary())  # 无默认 doc_ids
    ex = ToolExecutor(reg)
    r = asyncio.run(ex.execute("summarize", {}))
    assert not r.ok and "未指定文档" in r.error


def test_summary_tools_disabled_not_registered():
    reg = ToolRegistry()
    register_summary_tools(reg, _StubSummary(), enabled=False)
    assert "summarize" not in reg.names()
    assert "related_work" not in reg.names()
