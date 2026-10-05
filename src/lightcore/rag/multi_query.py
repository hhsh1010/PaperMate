"""LightCore · 多查询融合（P1-6，docs/统一方案 阶段C / 设计文档 §3.2）

把用户 query 改写为 2–3 个子查询，分别检索后 RRF 合并去重，提升召回（尤其长尾/多义查询）。

- MultiQueryExpander：用真实 LLM 生成子查询；无 LLM（None）或 FakeLLM 时回退单查询（不阻断、零成本）。
- MultiQueryRAGPipeline：包裹任意含 `query`/`get_all_chunks` 的管线（RAGPipeline 或 CompositeRAGPipeline），
  透明增强 retrieve 工具；fake 模式下与单查询行为完全一致。

模块零副作用、可纯单测（用 StubLLM 注入即可验证改写与合并逻辑）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..llm.base import BaseLLM, ChatMessage, Role
from ..llm.fake import FakeLLM
from .pipeline import RetrievedChunk

logger = logging.getLogger(__name__)


class MultiQueryExpander:
    """把 query 改写为多个子查询（LLM 增强）；无 key / fake 时回退单查询。"""

    def __init__(self, llm: BaseLLM | None = None, n: int = 3) -> None:
        self._llm = llm
        self._n = max(1, min(n, 5))

    async def expand(self, query: str, n: int | None = None) -> list[str]:
        q = (query or "").strip()
        if not q:
            return [q]
        llm = self._llm
        if llm is None or isinstance(llm, FakeLLM):
            return [q]  # 离线/占位：单查询，行为不变
        want = max(1, min(n or self._n, 5))
        prompt = (
            "你是一个检索查询改写器。请把下面的用户问题改写成最多 "
            f"{want} 个不同角度的检索子查询，以便更全面地召回相关段落。\n"
            "要求：每行一个子查询，不要编号、不要解释、不要使用 Markdown；\n"
            "子查询应保留关键实体与术语，可换种说法或拆解子问题。\n\n"
            f"用户问题：{q}"
        )
        try:
            resp = await llm.chat(
                [ChatMessage(role=Role.USER, content=prompt)],
                temperature=0.2,
            )
            text = (resp.message.content or "").strip()
        except Exception:  # noqa: BLE001  LLM 不可用 → 回退单查询，绝不阻断
            return [q]
        subs = [ln.strip("0123456789.、-) \t") for ln in text.splitlines() if ln.strip()]
        subs = [s for s in subs if s and s != q][: want - 1]
        # 子查询 + 原查询去重合并（原查询必含，保证基线召回）
        out: list[str] = [q]
        seen: set[str] = {q}
        for s in subs:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out


class MultiQueryRAGPipeline:
    """多查询融合检索管线：改写 → 各子查询检索 → RRF 合并去重 → top_k。"""

    def __init__(self, base: Any, expander: MultiQueryExpander | None = None) -> None:
        self._base = base
        self._expander = expander or MultiQueryExpander(None)

    def get_all_chunks(self) -> list:
        if hasattr(self._base, "get_all_chunks"):
            return self._base.get_all_chunks()
        if hasattr(self._base, "get_chunks"):
            return self._base.get_chunks()
        return []

    async def query(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        subqueries = await self._expander.expand(query)
        # N9 修复（2026-09-11）：**base.query 是同步且 GPU 密集的**（BGE-M3 编码 + 混合检索
        # + CrossEncoder 重排）。原先在 async 函数里直接调用会**阻塞整个事件循环** ——
        # 用户实测一次工具调用耗时 262.82s，期间 `_watch_disconnect`（每 0.3s 轮询断连的协程）
        # 完全没机会运行 → 「停止」无效、其余请求无响应、GPU 持续满载。
        # N10 优化（同日）：**子查询并发**。原先串行（3 个 × 每次 2~4s = 8~14s，实测复现）。
        # ⚠️ **实测结论（务必保留）**：并发**无速度收益**（4 子查询：串行 6.02s vs 并发 6.01s）——
        # 各子查询共享同一块 GPU，CUDA 操作在同一 stream 上**天然串行**，下放线程池只是改变
        # 排队位置，总计算量不变。保留此写法是因为它**无害且更简洁**（gather 替代 for 循环，
        # 未来多卡/异构硬件下会受益）；**不要**把它当作性能优化依据。
        # 检索提速的真正杠杆在「减少重排候选数」与「减少检索轮次」，见 docs/待修清单 第十八节。
        _t0 = time.perf_counter()
        results_list = await asyncio.gather(
            *(asyncio.to_thread(self._base.query, subq, top_k=top_k) for subq in subqueries),
            return_exceptions=True,
        )
        merged: dict[str, dict[str, Any]] = {}
        for res in results_list:
            if isinstance(res, BaseException):  # 单条子查询失败不拖垮整体
                continue
            for rank, c in enumerate(res):
                entry = merged.setdefault(c.id, {"chunk": c, "rrf": 0.0, "best": c.score})
                entry["rrf"] += 1.0 / (60 + rank)  # RRF k=60（与 hybrid 一致）
                entry["best"] = max(entry["best"], c.score)
        logger.info(
            "多查询检索：%d 个子查询并发，合并后 %d 个候选，耗时 %.2fs",
            len(subqueries), len(merged), time.perf_counter() - _t0,
        )
        out: list[RetrievedChunk] = []
        for _cid, e in merged.items():
            c = e["chunk"]
            out.append(RetrievedChunk(id=c.id, text=c.text, page=c.page, score=round(e["rrf"], 6)))
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]
