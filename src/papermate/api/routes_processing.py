"""PaperMate · 文档处理路由（功能增强规格说明 §2 文档处理模块 + 批量/版本/统计增强）

端点一览：
- POST   /documents/process            单文档分析（解析/提取/转换），可选 persist=true 落库
- POST   /documents/process/batch      批量异步提交（任务队列，含容错重试）
- GET    /documents/process/jobs       任务列表（status/limit）
- GET    /documents/process/jobs/{id}  任务详情（含处理结果）
- GET    /documents/process/stats      队列统计摘要
- POST   /documents/{doc_id}/process   对已上传文档执行处理并持久化（结构化存储 + 注册版本）
- GET    /documents/{doc_id}/structured 已持久化的结构化视图
- GET    /documents/{doc_id}/export?format=md|json|txt  格式转换导出
- GET    /documents/{doc_id}/versions  版本历史
- POST   /documents/{doc_id}/versions  注册当前文件为新版本
- GET    /documents/compare            两文档差异对比（left/right doc_id）

注意：compare 路由必须在 /documents/{doc_id} 类路由之前注册，避免路径参数吞掉 "compare"。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse

from ..processing.schemas import (
    BatchSubmitOut,
    DiffOut,
    JobOut,
    ProcessedDocumentOut,
    ProcessingResultOut,
    StatsOut,
    VersionOut,
)

router = APIRouter(tags=["documents-processing"])


def _svc(request: Request):
    return request.app.state.processing_svc


def _doc_svc(request: Request):
    return request.app.state.doc_svc


# ---------- 单文档处理 ----------

@router.post("/documents/process", response_model=ProcessingResultOut)
async def process_document(
    file: UploadFile,
    request: Request,
    persist: bool = Query(False, description="处理结果写入 documents.meta_json['processing']（需要先上传）"),
    doc_id: int | None = Query(None, description="persist=true 时指定目标文档 id"),
) -> ProcessingResultOut:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="空文件")
    if persist and doc_id is None:
        raise HTTPException(status_code=400, detail="persist=true 时必须提供 doc_id")
    name = (file.filename or "document.txt").strip()
    result = _svc(request).process_bytes(name, content, doc_id=doc_id if persist else None)
    if persist:
        _svc(request).persist_processing(doc_id, result)
    return _to_result_out(result)


# ---------- 批量异步队列 ----------

@router.post("/documents/process/batch", response_model=BatchSubmitOut)
async def process_batch(files: list[UploadFile], request: Request) -> BatchSubmitOut:
    items: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        if not content:
            continue  # 空文件跳过（容错）
        items.append(((f.filename or "document.txt").strip(), content))
    if not items:
        raise HTTPException(status_code=400, detail="没有可处理的文件")
    jobs = await _svc(request).submit_batch(items)
    return BatchSubmitOut(
        batch_id="batch",
        total=len(jobs),
        jobs=[_to_job_out(j) for j in jobs],
    )


@router.get("/documents/process/jobs", response_model=list[JobOut])
async def list_process_jobs(
    request: Request,
    status: str | None = Query(None, pattern="^(pending|running|retrying|done|failed)$"),
    limit: int = Query(50, ge=1, le=500),
) -> list[JobOut]:
    return [_to_job_out(j) for j in _svc(request).list_jobs(status=status, limit=limit)]


@router.get("/documents/process/jobs/{job_id}", response_model=JobOut)
async def get_process_job(job_id: str, request: Request) -> JobOut:
    job = _svc(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _to_job_out(job)


@router.get("/documents/process/stats", response_model=StatsOut)
async def process_stats(request: Request) -> StatsOut:
    return StatsOut(**_svc(request).batch_summary())


# ---------- 差异对比（字面路径，须在 {doc_id} 路由之前注册；app.py 中本路由也先于 documents_router 挂载） ----------

@router.get("/documents/compare", response_model=DiffOut)
async def compare_documents(
    request: Request,
    left: int = Query(..., description="左文档 id"),
    right: int = Query(..., description="右文档 id"),
) -> DiffOut:
    svc = _svc(request)
    if _doc_svc(request).get_document(left) is None or _doc_svc(request).get_document(right) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    diff = svc.compare_documents(left, right)
    return DiffOut(
        left_title=diff.left_title,
        right_title=diff.right_title,
        similarity=diff.similarity,
        added_lines=diff.added_lines,
        removed_lines=diff.removed_lines,
        changed_ratio=diff.changed_ratio,
        unified=diff.unified,
        summary=diff.summary,
    )


# ---------- 对已上传文档的处理与持久化 ----------

@router.post("/documents/{doc_id}/process", response_model=ProcessedDocumentOut)
async def process_existing_document(doc_id: int, request: Request) -> ProcessedDocumentOut:
    svc = _svc(request)
    doc_svc = _doc_svc(request)
    doc = doc_svc.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    raw = doc_svc.read_original(doc_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="原始文件不可用")
    # 2026-09-08 修复 H4：必须在 persist_processing **之前**取旧解析文本作为 diff 基线——
    # 原实现在处理之后才读 get_parsed_text，此时文档已被新内容覆盖，prev_text == text，
    # 版本摘要恒为「内容基本相同（仅格式差异）」，真实的内容变更被完全掩盖。
    prev_text = doc_svc.get_parsed_text(doc_id) if svc.list_versions(doc_id) else None
    result = svc.process_bytes(doc.filename, raw, doc_id=doc_id)
    svc.persist_processing(doc_id, result)
    version = None
    if result.status == "ok":
        # 版本对比用规范解析文本（与 POST /versions 同源，保证指纹一致可去重）
        text = doc_svc.get_parsed_text(doc_id) or _plain_of(result)
        version = svc.register_version(doc_id, text, raw, prev_text=prev_text)
    return ProcessedDocumentOut(
        doc_id=doc_id,
        filename=doc.filename,
        status=doc.status,
        processing=_to_result_out(result),
        versions=[_to_version_out(v) for v in svc.list_versions(doc_id)] if version else [],
    )


@router.get("/documents/{doc_id}/structured")
async def get_structured(doc_id: int, request: Request) -> dict:
    proc = _svc(request).get_processing(doc_id)
    if proc is None:
        raise HTTPException(status_code=404, detail="该文档尚未处理（先 POST /documents/{id}/process）")
    return proc.get("structured", {})


@router.get("/documents/{doc_id}/export")
async def export_document(
    doc_id: int,
    request: Request,
    format: str = Query("md", pattern="^(md|json|txt)$"),
):
    result = _svc(request).export(doc_id, format)
    if result is None:
        # 2026-09-06：export 已支持三级回退（产物 → meta_json["processing"] → 解析文本兜底），
        # 返回 None 只可能是文档尚无解析文本。旧文案「先 POST /documents/{id}/process」
        # 会误导用户去调一个前端并不存在的接口。
        raise HTTPException(status_code=404, detail="导出内容不可用（文档尚未解析完成，请等待处理结束后重试）")
    content_type, body = result
    return PlainTextResponse(body, media_type=content_type, headers={"X-Export-Format": format})


# ---------- 版本管理 ----------

@router.get("/documents/{doc_id}/versions", response_model=list[VersionOut])
async def get_versions(doc_id: int, request: Request) -> list[VersionOut]:
    svc = _svc(request)
    if _doc_svc(request).get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return [_to_version_out(v) for v in svc.list_versions(doc_id)]


@router.post("/documents/{doc_id}/versions", response_model=VersionOut)
async def register_document_version(doc_id: int, request: Request) -> VersionOut:
    svc = _svc(request)
    doc_svc = _doc_svc(request)
    doc = doc_svc.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    raw = doc_svc.read_original(doc_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="原始文件不可用")
    text = doc_svc.get_parsed_text(doc_id)
    if not text:
        # 文档尚未被解析过：现场用处理管线解析一次，保证版本对比有文本基础
        res = svc.process_bytes(doc.filename, raw)
        text = _plain_of(res) if res.status == "ok" else ""
    info = svc.register_version(doc_id, text, raw)
    return _to_version_out(info)


# ---------- 转换辅助 ----------

def _plain_of(result) -> str:
    """ProcessingResult → 纯文本（版本对比用正文）。"""
    return "\n".join(f"{s.heading}\n{s.text}" if s.heading else s.text for s in result.sections)


def _to_result_out(result) -> ProcessingResultOut:
    from ..processing.schemas import MetadataOut, SectionOut

    meta = result.metadata
    return ProcessingResultOut(
        filename=result.filename,
        source_format=result.source_format,
        status=result.status,
        title=result.title,
        num_chunks=result.num_chunks,
        duration_ms=result.duration_ms,
        metadata=(
            MetadataOut(
                title=meta.title,
                language=meta.language,
                chars=meta.chars,
                words=meta.words,
                paragraphs=meta.paragraphs,
                sentences=meta.sentences,
                headings=meta.headings,
                max_heading_level=meta.max_heading_level,
                has_code_block=meta.has_code_block,
                keywords=meta.keywords,
                issues=[{"severity": i.severity, "code": i.code, "message": i.message} for i in meta.issues],
            )
            if meta
            else None
        ),
        sections=[
            SectionOut(order=s.order, level=s.level, heading=s.heading, page=s.page, text=s.text)
            for s in result.sections
        ],
        markdown=result.markdown,
        structured=result.structured,
        warnings=result.warnings,
        error=result.error,
        error_code=result.error_code,
        artifacts=result.artifacts,
    )


def _to_job_out(job) -> JobOut:
    return JobOut(
        id=job.id,
        filename=job.filename,
        status=job.status,
        attempts=job.attempts,
        max_retries=job.max_retries,
        error=job.error,
        error_code=job.error_code,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=_to_result_out(job.result) if job.result is not None else None,
    )


def _to_version_out(v) -> VersionOut:
    return VersionOut(
        version=v.version,
        content_hash=v.content_hash,
        fingerprint=v.fingerprint,
        summary=v.summary,
        created_at=v.created_at,
    )
