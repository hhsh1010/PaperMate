"""get_pipeline 事件循环保护回归（2026-09-07）。

现象：往工作间加入一篇文档后，首次构建该文档的 RAG 管线（BGE-M3 编码 + jieba 分词
+ FAISS 建索引，CPU 下分钟级）直接跑在事件循环上，整个后端冻结——期间所有 HTTP
请求（含预览的 /documents/{id}/file）排队超时，前端预览"全部加载失败"。
修复：get_pipeline 的无进度路径（cancel_event/on_progress 均为 None）与上传路径
同样走 asyncio.to_thread。本测试用桩管线断言：编码在非主线程执行，且编码期间
事件循环仍在持续调度（ping 计数不冻结）。
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import papermate.config as cfg_mod
from papermate.services import document_service as ds
from papermate.services.document_service import DocumentService


class _Query:
    def filter(self, *a):  # noqa: ANN002, ANN003
        return self

    def order_by(self, *a):  # noqa: ANN002, ANN003
        return self

    def all(self):
        return []  # 无 Chunk 行 → 走扁平文本回退分支


class _Session:
    def __enter__(self):
        return self

    def __exit__(self, *a):  # noqa: ANN002, ANN003
        return False

    def query(self, *a):  # noqa: ANN002, ANN003
        return _Query()


class _SessionMaker:
    def __call__(self):
        return _Session()


class _StubPipeline:
    """记录执行线程 + 阻塞 0.3s（模拟 CPU 编码），供事件循环响应性断言。"""

    executed_on_main_thread: bool | None = None

    def __init__(self, **kw) -> None:  # noqa: ANN003
        pass

    def add_document(self, text, doc_id=None, **kw) -> None:  # noqa: ANN002, ANN003
        _StubPipeline.executed_on_main_thread = threading.current_thread() is threading.main_thread()
        time.sleep(0.3)

    def add_chunks(self, *a, **kw) -> None:  # noqa: ANN002, ANN003
        _StubPipeline.executed_on_main_thread = threading.current_thread() is threading.main_thread()
        time.sleep(0.3)


def test_get_pipeline_encoding_off_event_loop(monkeypatch, tmp_path):
    os.environ["PAPERMATE_DATA_DIR"] = str(tmp_path)
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    import papermate.runtime_settings as rs

    rs._state = None  # 隔离单例（读取 tmp 下不存在的 provider_settings.json → 默认值）

    from papermate.config import get_settings

    settings = get_settings()
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    (settings.parsed_dir / "42.txt").write_text("演示文本，用于扁平回退分支。 " * 50, encoding="utf-8")

    monkeypatch.setattr(ds, "RAGPipeline", _StubPipeline)

    svc = DocumentService(_SessionMaker(), settings)

    async def _fake_component():
        return object()

    svc._get_embedder = _fake_component
    svc._get_reranker = _fake_component
    svc._get_tokenizer = _fake_component

    async def run():
        ticks = {"n": 0}
        stop = asyncio.Event()

        async def ping():
            """每 10ms 自增一次：若事件循环被编码阻塞，计数将几乎不增长。"""
            while not stop.is_set():
                ticks["n"] += 1
                await asyncio.sleep(0.01)

        pinger = asyncio.create_task(ping())
        await asyncio.sleep(0.05)  # 让 ping 先跑起来
        baseline = ticks["n"]

        await svc.get_pipeline(42)

        during = ticks["n"] - baseline  # 编码期间 ping 的推进次数
        stop.set()
        await asyncio.gather(pinger, return_exceptions=True)
        return during

    during = asyncio.run(run())

    # 1) 编码不在主线程（= 不在事件循环线程）执行
    assert _StubPipeline.executed_on_main_thread is False, (
        "get_pipeline 无进度路径仍在事件循环上同步编码（会冻结整个后端）"
    )
    # 2) 0.3s 编码期间 ping 仍在被调度（0.01s 间隔理论 ~30 次，容忍调度抖动取 ≥10）
    assert during >= 10, f"编码期间事件循环被冻结（ping 仅推进 {during} 次）"
