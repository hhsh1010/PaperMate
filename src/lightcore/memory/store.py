"""LightCore · 记忆持久化存储（docs/05 §记忆系统）

把 LongTermMemory 的 MemoryItem 列表落盘，使其跨会话（跨进程）存活。
默认提供 JSONFileMemoryStore（零依赖，嵌入向量以 list 一并保存）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from .long_term import SCOPE_GLOBAL, SCOPE_SESSION, MemoryItem


@runtime_checkable
class MemoryStore(Protocol):
    def load(self) -> list[MemoryItem]: ...
    def save(self, items: list[MemoryItem]) -> None: ...


def _item_to_dict(it: MemoryItem) -> dict:
    return {
        "id": it.id,
        "text": it.text,
        "embedding": it.embedding,
        "importance": it.importance,
        "created_at": it.created_at,
        "last_accessed": it.last_accessed,
        "source_session_id": it.source_session_id,
        "scope": it.scope,
    }


def _dict_to_item(d: dict) -> MemoryItem:
    scope = d.get("scope")
    if scope not in (SCOPE_GLOBAL, SCOPE_SESSION):
        scope = SCOPE_GLOBAL  # 旧数据/非法值回落 global，行为向后兼容
    return MemoryItem(
        id=d["id"],
        text=d["text"],
        embedding=list(d.get("embedding", [])),
        importance=float(d.get("importance", 0.5)),
        created_at=float(d.get("created_at", 0.0)),
        last_accessed=float(d.get("last_accessed", 0.0)),
        # 来源会话（2026-08-29 C 档）：旧数据缺失时为 None
        source_session_id=(int(d["source_session_id"]) if d.get("source_session_id") is not None else None),
        scope=scope,
    )


class JSONFileMemoryStore:
    """把记忆存为 JSON 文件（嵌入向量一并保存）；文件缺失时返回空列表。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[MemoryItem]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        return [_dict_to_item(d) for d in data]

    def save(self, items: list[MemoryItem]) -> None:
        self._path.write_text(
            json.dumps([_item_to_dict(it) for it in items], ensure_ascii=False),
            encoding="utf-8",
        )
