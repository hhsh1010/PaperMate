"""文档块的**按 id 重取**工具（`fetch_chunk`）—— 2026-10-23。

## 为什么需要
`retrieve` 的结果会被压缩（每段 text 截断到 ≤200 字，见 `loop._compact_retrieve_output`），
但它**保留了 `id`**（形如 `doc-12-c34`）✓。此前模型被截断后只能**换关键词重新检索** ✗
—— 拿不回原来那一段，容易"以为文档里就这么多"（静默降准）。

`pdf_extract` 那边已经踩过并解决了同类问题（注释里写着"必须告诉模型可以分页取全文"），
这里把同一思路推广到 `retrieve`：**给一个按 id 取全文的工具**。

## 接口
`fetch_chunk(chunk_id) -> {chunk_id, page, heading, text}`
id 解析规则与 `retrieve` 返回的 `id` 完全一致（`doc-{document_id}-c{seq}`）。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from lightcore.tools.spec import Permission, ToolResult, ToolSpec

from ..models.db import Chunk

logger = logging.getLogger(__name__)

#: 与 `loop._compact_retrieve_output` 生成 id 的规则保持一致
_CHUNK_ID_RE = re.compile(r"^doc-(\d+)-c(\d+)$")


def parse_chunk_id(chunk_id: str) -> tuple[int, int] | None:
    """`doc-12-c34` → (12, 34)；格式不对返回 None（纯函数，便于写判据）。"""
    m = _CHUNK_ID_RE.match((chunk_id or "").strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def register_doc_tools(registry: Any, session_factory: Any) -> None:
    """注册 `fetch_chunk`（需 DB 访问 ⇒ 放在 papermate 层，不放进 lightcore）。"""

    async def _fn(chunk_id: str) -> ToolResult:
        parsed = parse_chunk_id(chunk_id)
        if parsed is None:
            return ToolResult(
                ok=False, output=None,
                error=f"chunk_id 格式不对：{chunk_id!r}。应为 `doc-<文档id>-c<序号>`（如 doc-3-c12）。",
            )
        doc_id, seq = parsed
        try:
            with session_factory() as db:
                c = (
                    db.query(Chunk)
                    .filter(Chunk.document_id == doc_id, Chunk.seq == seq)
                    .first()
                )
                if c is None:
                    return ToolResult(ok=False, output=None,
                                      error=f"找不到片段 {chunk_id}（可能文档已删除或重建过索引）。")
                out = {
                    "chunk_id": chunk_id,
                    "document_id": c.document_id,
                    "seq": c.seq,
                    "page": c.page,
                    "heading": c.heading or "",
                    "text": c.content,
                }
        except Exception as e:  # noqa: BLE001  工具失败交给 Agent 循环处理
            logger.warning("fetch_chunk 失败 chunk_id=%s", chunk_id, exc_info=True)
            return ToolResult(ok=False, output=None, error=f"{type(e).__name__}: {e}")
        return ToolResult(ok=True, output=out)

    registry.register(ToolSpec(
        name="fetch_chunk",
        description="按 id 取回**某个文档片段的完整原文**（`retrieve` 返回的是压缩版，每段 ≤200 字）。"
                    "当你需要完整的一句话/公式/数值，而 retrieve 的结果被截断（末尾有「…」）时用它。"
                    "id 直接复制 retrieve 结果里的 `id` 字段（形如 doc-12-c34）。",
        parameters={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string",
                             "description": "片段 id，形如 `doc-12-c34`（取自 retrieve 结果的 id）"},
            },
            "required": ["chunk_id"],
        },
        permission=Permission.SAFE,
        handler=_fn,
    ))
