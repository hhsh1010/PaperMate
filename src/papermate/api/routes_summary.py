"""PaperMate · 应用服务路由（A-02：结构化总结 / 相关工作 / 多篇对比，docs/功能增强规格说明 A-02）

- POST /summarize      ：对指定文档集合产出「背景/方法/实验/结论/局限」五段式 + 引用。
- POST /related-work   ：产出「相关工作」综述草稿 + 引文来源。
- POST /compare        ：多篇文档对比（逐维度并列 + 共性/差异叙述 + 共引分析）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .. import runtime_settings
from ..schemas import CompareOut, RelatedWorkOut, SummaryOut, SummaryRequest
from ..services.summary_service import SummaryService

router = APIRouter(tags=["app-services"])


def _get_svc(request: Request) -> SummaryService:
    svc: SummaryService = request.app.state.summary_svc
    return svc


def _summary_enabled() -> bool:
    """Phase B：一键总结功能开关（features.summary）。关闭时接口返回 403。

    接口本身始终保留（满足"后端保留对应 API 接口"约束），仅按开关启用/停用。
    """
    return runtime_settings.get_state().get("features", {"summary": True}).get("summary", True)


@router.post("/summarize", response_model=SummaryOut)
async def summarize(body: SummaryRequest, request: Request) -> dict:
    if not _summary_enabled():
        raise HTTPException(status_code=403, detail="一键总结功能当前已禁用（请在设置页开启『一键总结』）")
    if not body.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids 不能为空")
    svc = _get_svc(request)
    # ★ 2026-10-22：带 session_id ⇒ 模型跟随会话所选（与对话/提示词增强同源）
    model = None
    if body.session_id is not None:
        model = request.app.state.chat_svc._resolve_raw_model(body.session_id)
    try:
        return await svc.summarize(body.doc_ids, lang=body.lang, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/related-work", response_model=RelatedWorkOut)
async def related_work(body: SummaryRequest, request: Request) -> dict:
    if not _summary_enabled():
        raise HTTPException(status_code=403, detail="一键总结功能当前已禁用（请在设置页开启『一键总结』）")
    if not body.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids 不能为空")
    svc = _get_svc(request)
    try:
        return await svc.related_work(body.doc_ids, lang=body.lang)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/compare", response_model=CompareOut)
async def compare(body: SummaryRequest, request: Request) -> dict:
    """多篇文档对比（S5）：逐维度并列各篇五段式 + 共性/差异叙述 + 共引分析。"""
    if not _summary_enabled():
        raise HTTPException(status_code=403, detail="一键总结功能当前已禁用（请在设置页开启『一键总结』）")
    if len(body.doc_ids or []) < 2:
        raise HTTPException(status_code=400, detail="对比至少需要 2 篇文档")
    svc = _get_svc(request)
    try:
        return await svc.compare(body.doc_ids, lang=body.lang)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

