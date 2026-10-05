"""PaperMate · 文献集路由（F5：文献集 / 项目分组 Collections）

- CRUD：POST /collections、GET /collections、GET /collections/{id}、PATCH /collections/{id}、
  DELETE /collections/{id}
- 文档增删：POST /collections/{id}/documents、DELETE /collections/{id}/documents/{doc_id}
- 会话通过 config.collection_id 关联文献集，由 chat_service / routes_search 在查询时解析为有效文档。

三硬约束：本路由不涉及大模型 API / 开关，仅本地数据组织，属安全默认（不开独立开关）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from ..schemas import CollectionListOut, CollectionOut, OkOut
from ..services.collection_service import CollectionService

router = APIRouter(tags=["collections"])


class CollectionCreate(BaseModel):
    name: str
    description: str = ""
    doc_ids: list[int] = []


class CollectionPatch(BaseModel):
    name: str | None = None
    description: str | None = None


class CollectionDocAdd(BaseModel):
    doc_id: int


def _svc(request: Request) -> CollectionService:
    return request.app.state.collection_svc


@router.post("/collections", response_model=CollectionOut)
def create_collection(body: CollectionCreate, request: Request) -> dict:
    svc = _svc(request)
    try:
        cid = svc.create(body.name, body.description, body.doc_ids)
    except ValueError as e:
        # 写入前校验发现无效 doc_id → 明确 400，避免 IntegrityError 冒泡成 500
        raise HTTPException(status_code=400, detail=str(e)) from e
    except IntegrityError as e:
        # 竞态兜底：校验通过后、提交瞬间文档被删除
        raise HTTPException(status_code=400, detail="文档不存在或已删除，请刷新后重试") from e
    return svc.get(cid)


@router.get("/collections", response_model=CollectionListOut)
def list_collections(request: Request) -> dict:
    return {"items": _svc(request).list()}


@router.get("/collections/{cid}", response_model=CollectionOut)
def get_collection(cid: int, request: Request) -> dict:
    out = _svc(request).get(cid)
    if out is None:
        raise HTTPException(status_code=404, detail="文献集不存在")
    return out


@router.patch("/collections/{cid}", response_model=CollectionOut)
def patch_collection(cid: int, body: CollectionPatch, request: Request) -> dict:
    out = _svc(request).rename(cid, body.name, body.description)
    if out is None:
        raise HTTPException(status_code=404, detail="文献集不存在")
    return out


@router.delete("/collections/{cid}", response_model=OkOut)
def delete_collection(cid: int, request: Request) -> dict:
    if not _svc(request).delete(cid):
        raise HTTPException(status_code=404, detail="文献集不存在")
    return {"ok": True}


@router.post("/collections/{cid}/documents", response_model=OkOut)
def add_document(cid: int, body: CollectionDocAdd, request: Request) -> dict:
    svc = _svc(request)
    try:
        ok = svc.add_document(cid, body.doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except IntegrityError as e:
        raise HTTPException(status_code=400, detail="文档不存在或已删除，请刷新后重试") from e
    if not ok:
        raise HTTPException(status_code=404, detail="文献集不存在")
    return {"ok": True}


@router.delete("/collections/{cid}/documents/{doc_id}", response_model=OkOut)
def remove_document(cid: int, doc_id: int, request: Request) -> dict:
    if not _svc(request).remove_document(cid, doc_id):
        raise HTTPException(status_code=404, detail="文献集或文档关联不存在")
    return {"ok": True}
