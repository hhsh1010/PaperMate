"""PaperMate · 记忆管理路由（ER-16：长期记忆管理面板，docs/功能增强规格说明 ER-16）

长期记忆以 JSON 文件存储（data/memories.json，见 lightcore.memory.store）。本路由提供
查看 / 编辑 / 删除 / 清空 接口，供设置页「记忆管理」面板使用。

CRITICAL：操作的是应用自身数据（data/ 目录下的 memories.json），不涉及个人文件，可安全操作；
但清空为破坏性动作，仍要求二次确认（由前端发起，这里只负责执行 Clear）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..memory_hub import get_memory_manager
from ..schemas import (
    MemoryBatchDeleteOut,
    MemoryListOut,
    MemoryUpdateOut,
    OkRemovedOut,
)

router = APIRouter(tags=["memory"])


def _manager(request: Request):
    """共享的长期记忆管理器（与对话服务同一实例，见 papermate.memory_hub）。

    2026-08-29 A 档修复：此前本模块另建 store 直写 JSON 文件，与对话侧进程内存
    不同步，面板编辑/删除需重启后端才生效。
    """
    return get_memory_manager(request.app.state.settings)


@router.get("/memories", response_model=MemoryListOut)
def list_memories(request: Request) -> dict:
    items = _manager(request).long.all_items()
    return {
        "total": len(items),
        "items": [
            {
                "id": it.id,
                "text": it.text,
                "importance": it.importance,
                "created_at": it.created_at,
                "last_accessed": it.last_accessed,
                "source_session_id": it.source_session_id,  # 来源会话（C 档：面板可溯源）
                "scope": it.scope,  # 作用域（分层 scope：global 跨会话 / session 仅来源会话）
            }
            for it in items
        ],
    }


@router.post("/memories/batch-delete", response_model=MemoryBatchDeleteOut)
def batch_delete_memories(body: dict, request: Request) -> dict:
    """批量删除记忆（2026-08-29 C 档：面板支持勾选后一次清理多条）。

    body: {"ids": ["id1", "id2", ...]}；返回实际删除条数与未命中的 id。
    """
    ids = body.get("ids") or []
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="ids 必须是字符串数组")
    ltm = _manager(request).long
    removed = 0
    missing: list[str] = []
    for mid in ids:
        mid_str = str(mid)
        if ltm.remove(mid_str):
            removed += 1
        else:
            missing.append(mid_str)
    return {"ok": True, "removed": removed, "missing": missing}


@router.put("/memories/{mem_id}", response_model=MemoryUpdateOut)
def update_memory(mem_id: str, body: dict, request: Request) -> dict:
    """编辑记忆文本 / 重要性 / 作用域。

    2026-08-29 A 档修复：改用共享单例（改动即时对对话生效）+ **重算 embedding**
    （否则召回仍按旧文本的向量匹配，语义永久错位）；importance 缺省时保留原值，
    仅在请求体显式传入时更新（原实现用关键词启发式覆盖 LLM 评估值，会丢信息）。
    2026-08-29 分层 scope：scope 可单独修改（改归属不需重传文本）。
    text / importance / scope 三者任一缺省时保留原值，至少应提供其一。
    """
    ltm = _manager(request).long
    target_item = ltm.get(mem_id)
    if target_item is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    # M10 修复（2026-09-10）：text 非字符串（如 {"text": 123}）时 `.strip()` 会
    # AttributeError → 500；显式判型返回 400，让调用方知道是入参问题。
    raw_text = body.get("text")
    if raw_text is not None and not isinstance(raw_text, str):
        raise HTTPException(status_code=400, detail="text 必须是字符串")
    text = (raw_text or "").strip()
    if not text:
        text = target_item.text  # 未传文本则保留原文（便于仅改作用域/重要性）
    imp_raw = body.get("importance")
    imp = None
    if imp_raw is not None:
        try:
            imp = max(0.0, min(1.0, float(imp_raw)))
        except (TypeError, ValueError):
            imp = None
    # scope（2026-08-29 分层 scope）：显式传入才更新，支持面板「设为全局 / 限本会话」
    scope = body.get("scope")
    if scope is not None and scope not in ("global", "session"):
        raise HTTPException(status_code=400, detail="scope 仅接受 global / session")
    target = ltm.update(mem_id, text, importance=imp, scope=scope)
    if target is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"ok": True, "id": target.id, "text": target.text,
            "importance": target.importance, "scope": target.scope}


@router.delete("/memories/{mem_id}", response_model=OkRemovedOut)
def delete_memory(mem_id: str, request: Request) -> dict:
    ltm = _manager(request).long
    if not ltm.remove(mem_id):
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"ok": True, "removed": 1}


@router.post("/memories/clear", response_model=OkRemovedOut)
def clear_memories(request: Request) -> dict:
    removed = _manager(request).long.clear()
    return {"ok": True, "removed": removed}
