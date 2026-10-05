"""PaperMate · 工作间路由（A1：工作间后端基座）

- POST /workspaces：原子创建工作间（文献集 + 工作间会话）。
- GET  /workspaces/{session_id}：从会话 config_json 解析工作间配置（只读，零额外表）。

三硬约束：本路由不涉及大模型 API / 开关，仅本地数据组织，属安全默认（不开独立开关）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.exc import IntegrityError

from ..schemas import WorkspaceAddDocs, WorkspaceCreate, WorkspaceOut, WorkspaceUpdate
from ..services.workspace_service import WorkspaceError, WorkspaceNotFound

router = APIRouter(tags=["workspaces"])


@router.post("/workspaces", response_model=WorkspaceOut)
def create_workspace(body: WorkspaceCreate, request: Request) -> WorkspaceOut:
    svc = request.app.state.workspace_svc
    try:
        out = svc.create(
            name=body.name,
            description=body.description,
            doc_ids=body.doc_ids,
            collection_id=body.collection_id,
            draft_doc_id=body.draft_doc_id,
            mode=body.mode,
        )
    except WorkspaceNotFound as e:
        raise HTTPException(status_code=404, detail=f"文献集不存在：{e}") from e
    except (ValueError, WorkspaceError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except IntegrityError:
        raise HTTPException(status_code=400, detail="文档不存在或已删除，请刷新后重试") from None
    return WorkspaceOut(**out)


@router.post("/workspaces/{session_id}/documents", response_model=WorkspaceOut)
def add_workspace_documents(
    session_id: int, body: WorkspaceAddDocs, request: Request
) -> WorkspaceOut:
    """B1：往工作间文献集批量加入文档（幂等去重）。非工作间会话 → 404。"""
    svc = request.app.state.workspace_svc
    try:
        out = svc.add_documents(session_id, body.doc_ids)
    except WorkspaceNotFound as e:
        raise HTTPException(status_code=404, detail="工作间不存在或该会话不是工作间") from e
    except (ValueError, WorkspaceError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except IntegrityError:
        raise HTTPException(status_code=400, detail="文档不存在或已删除，请刷新后重试") from None
    return WorkspaceOut(**out)


@router.delete("/workspaces/{session_id}/documents/{doc_id}", response_model=WorkspaceOut)
def remove_workspace_document(
    session_id: int, doc_id: int, request: Request
) -> WorkspaceOut:
    """B1 反向：从工作间文献集移除单一文档关联（不删除源 Document 本身）。

    非工作间会话 / 文档不在该文献集内 → 404。返回移除后的工作间投影。
    """
    svc = request.app.state.workspace_svc
    try:
        out = svc.remove_document(session_id, doc_id)
    except WorkspaceNotFound as e:
        raise HTTPException(
            status_code=404,
            detail="工作间不存在、该会话不是工作间，或文档不在工作间文献集内",
        ) from e
    return WorkspaceOut(**out)


@router.patch("/workspaces/{session_id}", response_model=WorkspaceOut)
def update_workspace(session_id: int, body: WorkspaceUpdate, request: Request) -> WorkspaceOut:
    """B2：PATCH 语义更新工作间配置（主文献/模式）—— 只更新显式传入的字段。

    `{"draft_doc_id": 5}` 设置主文献；`{"draft_doc_id": null}` 清除；`{}` 不动任何字段。
    以 `model_fields_set` 区分「未传」与「显式 null」。
    """
    svc = request.app.state.workspace_svc
    fs = body.model_fields_set
    try:
        out = svc.update(
            session_id,
            set_mode="mode" in fs,
            mode=body.mode,
            set_draft="draft_doc_id" in fs,
            draft_doc_id=body.draft_doc_id,
        )
    except WorkspaceNotFound as e:
        raise HTTPException(status_code=404, detail="工作间不存在或该会话不是工作间") from e
    except (ValueError, WorkspaceError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return WorkspaceOut(**out)


@router.get("/workspaces/{session_id}", response_model=WorkspaceOut)
def get_workspace(session_id: int, request: Request) -> WorkspaceOut:
    """从会话 config_json 解析工作间配置；非工作间会话 / 不存在 → 404。"""
    svc = request.app.state.workspace_svc
    out = svc.resolve(session_id)
    if out is None:
        raise HTTPException(status_code=404, detail="工作间不存在或该会话不是工作间")
    return WorkspaceOut(**out)
