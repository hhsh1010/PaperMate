"""检索调用口径的跨层契约（O-01 / O-05，2026-10-01）

### 背景

本项目的检索管线有**两种形态**，调用点上长得一模一样：

- **同步**：`RAGPipeline` / `CompositeRAGPipeline`（BGE-M3 编码 + 混合检索 + CrossEncoder 重排，**GPU 密集**）
- **异步**：`MultiQueryRAGPipeline`（`features.multi_query` 开启时外层包一层）

历史上为此踩过两次坑：
- **N4**：漏判异步 ⇒ 同步取值拿到**协程对象** ⇒ `TypeError: 'coroutine' object is not iterable`；
- **N9**：忘了下放线程池 ⇒ 同步管线阻塞事件循环（实测 262.82s，「停止」无效）。

⇒ 已收敛为唯一入口 `lightcore.rag.invoke.call_query`。本文件守护它，并**钉死漏网调用点**
（防止"改了三处、漏了第四处"再次发生）。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from lightcore.llm.fake import FakeLLM
from lightcore.rag.invoke import call_query
from lightcore.rag.pipeline import RetrievedChunk
from lightcore.tools.builtin.domain import run_planning


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(id=f"c{i}", text=f"证据段落 {i}", page=1, score=0.9)


class _AsyncPipe:
    """模拟 MultiQueryRAGPipeline：`query` 是 **async**。"""

    async def query(self, q: str, top_k: int = 3) -> list[RetrievedChunk]:
        return [_chunk(1)]


class _SyncPipe:
    """模拟 RAGPipeline / CompositeRAGPipeline：`query` 是**同步**且（现实中）GPU 密集。"""

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None

    def query(self, q: str, top_k: int = 3) -> list[RetrievedChunk]:
        self.thread = threading.current_thread()
        return [_chunk(1)]


def test_call_query_awaits_async_pipeline() -> None:
    """异步管线：直接 await，不经过线程池。"""
    out = asyncio.run(call_query(_AsyncPipe(), "x", top_k=3))
    assert [c.id for c in out] == ["c1"]


def test_call_query_runs_sync_pipeline_off_the_event_loop() -> None:
    """★ N9 回归：同步 GPU 管线**必须**下放线程池，否则阻塞整个事件循环。

    断言的是"实际在哪个线程执行"——若有人把 `to_thread` 去掉改回直接调用，
    这里必然变红（而不是悄悄退化成阻塞）。
    """
    pipe = _SyncPipe()
    out = asyncio.run(call_query(pipe, "x", top_k=3))
    assert len(out) == 1
    assert pipe.thread is not None
    assert pipe.thread is not threading.main_thread(), "同步管线未下放线程池（N9 回归）"


def test_call_query_awaits_awaitable_return() -> None:
    """兜底分支：同步实现**返回 awaitable**（自定义管线）时也要 await 一次（N4 兜底）。"""

    class _WeirdPipe:
        def query(self, q: str, top_k: int = 3):  # noqa: ANN201  故意返回 awaitable
            async def _inner() -> list[RetrievedChunk]:
                return [_chunk(2)]

            return _inner()

    out = asyncio.run(call_query(_WeirdPipe(), "x", top_k=3))
    assert [c.id for c in out] == ["c2"]


def test_call_query_raises_without_query() -> None:
    """管线没有 `query()` 时抛 TypeError（由调用方决定是否容错）。"""
    with pytest.raises(TypeError):
        asyncio.run(call_query(object(), "x", top_k=3))


def test_run_planning_collects_evidence_from_async_pipeline() -> None:
    """★★ O-01 核心回归：`features.multi_query` 开启时 rag 是**异步**管线。

    修复前：`rag.query(...)` 同步取值拿到**协程对象** ⇒ `for c in chunks` 抛 TypeError ⇒
    被"单步取证容错"的 `except Exception` 吞掉 ⇒ **每一步 evidence 静默为空**；
    而 K4 的"检索服务故障"标记**不会触发**（那是 `RetrievalUnavailable`，不是 TypeError）
    ⇒ 把**系统故障伪装成"文档里没有"**——模型据此回答"文献未提及"。

    修复后：每步都应有真实证据。本用例在修复前**必然红**（见报告的反向自检）。
    """
    out = asyncio.run(
        run_planning("先解析论文，然后检索段落，最后生成摘要", FakeLLM(), _AsyncPipe(), max_steps=3)
    )
    assert out["ok"] is True, out
    assert out["execution"], "规划未产出执行步骤"
    empty_steps = [ex["step"] for ex in out["execution"] if not ex["evidence"]]
    assert not empty_steps, f"步骤 {empty_steps} 的证据为空 ⇒ 异步管线未被 await（O-01 回归）"


def test_run_planning_also_works_with_sync_pipeline() -> None:
    """反向约束：同步管线（duck-type，如 `planning_skill._query`）同样必须正常取证。"""
    out = asyncio.run(
        run_planning("先解析论文，然后检索段落，最后生成摘要", FakeLLM(), _SyncPipe(), max_steps=3)
    )
    assert out["ok"] is True, out
    assert out["execution"]
    assert all(ex["evidence"] for ex in out["execution"]), "同步管线取证失败"


def test_no_call_site_invokes_pipeline_query_directly() -> None:
    """源码契约：检索调用点必须走 `call_query`（否则又会从漏网处长出同一类缺陷）。

    ⚠️ 断言的是**调用形态**（`xxx.query(`），不是"标识符出现过"——
    项目已两次栽在"源码里有关键词但不在被执行的那段"（注释/死分支）上。
    """
    root = Path(__file__).resolve().parents[1] / "src"
    forbidden = {
        "lightcore/tools/builtin/domain.py": "rag.query(",
        "lightcore/tools/builtin/basic.py": "pipeline.query(",
        "papermate/skills/deep_read.py": "pipe.query(",
        "papermate/api/routes_search.py": "CompositeRAGPipeline(pipelines).query(",
    }
    for rel, bad in forbidden.items():
        src = (root / rel).read_text(encoding="utf-8")
        assert bad not in src, f"{rel} 仍存在裸调用 `{bad}`（应统一走 call_query）"
        assert "call_query" in src, f"{rel} 未使用 call_query"
