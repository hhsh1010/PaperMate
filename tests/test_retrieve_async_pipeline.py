"""N4 回归（2026-09-10）：retrieve / pdf_extract 必须支持 **async** 的 pipeline.query。

真实根因（e2e 探针复现）：`MultiQueryRAGPipeline.query` 是协程（features.multi_query 开启时使用），
而工具 handler 是同步函数、直接 `for r in pipeline.query(...)` → 拿到协程对象 → TypeError →
工具连续失败 3 次 → agent 提前中止 → 用户只看到「抱歉，本次没有生成有效回答」。

修复：handler 改 async，并对 `pipeline.query(...)` 结果做 **awaitable 兼容**
（同步 RAGPipeline.query 与异步 MultiQueryRAGPipeline.query 都要能跑）。
"""

from __future__ import annotations

import asyncio

from lightcore.rag.pipeline import RetrievedChunk
from lightcore.tools.builtin.basic import _make_pdf_extract, _make_retrieve  # noqa: PLC2701
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolSpec


class _AsyncPipeline:
    """模拟 MultiQueryRAGPipeline：query 是协程。"""

    def __init__(self) -> None:
        self.called = 0

    def get_chunks(self):  # noqa: ANN201  pdf_extract 需先取全量 chunk
        return [RetrievedChunk(id="doc-9-c75", text="2. 5对比试验", page=5, score=0.93)]

    async def query(self, _q: str, top_k: int = 6):  # noqa: ANN201
        self.called += 1
        return [RetrievedChunk(id="doc-9-c75", text="2. 5对比试验", page=5, score=0.93)][:top_k]


class _SyncPipeline:
    """模拟普通 RAGPipeline：query 是同步函数。"""

    def __init__(self) -> None:
        self.called = 0

    def query(self, _q: str, top_k: int = 6):  # noqa: ANN201
        self.called += 1
        return [RetrievedChunk(id="doc-9-c83", text="对比以上两种模型检测结果", page=6, score=0.88)][:top_k]


def test_retrieve_awaits_async_pipeline_query():
    """N4 回归：异步管线必须被 await——原先同步取值 → 协程不可迭代 → 工具必然失败。"""
    pipe = _AsyncPipeline()
    handler = _make_retrieve(pipe)

    res = asyncio.run(handler("文档中提到的对比模型都有哪些"))
    assert res.ok, res.error
    assert pipe.called == 1
    assert res.output["results"][0]["id"] == "doc-9-c75"
    assert res.output["results"][0]["page"] == 5


def test_retrieve_still_supports_sync_pipeline_query():
    """兼容性：普通同步管线不受影响（awaitable 兼容不得只服务异步路径）。"""
    pipe = _SyncPipeline()
    handler = _make_retrieve(pipe)

    res = asyncio.run(handler("对比实验"))
    assert res.ok, res.error
    assert pipe.called == 1
    assert res.output["results"][0]["id"] == "doc-9-c83"


def test_pdf_extract_awaits_async_pipeline_query():
    """N4 回归：pdf_extract 的 query 分支同样是 async 兼容（同一根因的第二处）。"""
    pipe = _AsyncPipeline()
    handler = _make_pdf_extract(pipe)

    res = asyncio.run(handler(page=None, query="对比试验"))
    assert res.ok, res.error
    assert res.output["results"][0]["id"] == "doc-9-c75"


def test_executor_runs_async_handler_end_to_end():
    """端到端：注册到 ToolRegistry 后经 ToolExecutor 执行，异步 handler 被正确 await。

    （ToolExecutor 早已支持 awaitable handler，本用例守住这条约定不被回退。）
    """
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="retrieve",
        description="rag",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission=Permission.SAFE,
        handler=_make_retrieve(_AsyncPipeline()),
    ))
    ex = ToolExecutor(reg)

    res = asyncio.run(ex.execute("retrieve", {"query": "对比模型"}))
    assert res.ok, res.error
    assert res.output["results"][0]["page"] == 5
