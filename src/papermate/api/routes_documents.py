"""PaperMate · 文档路由（docs/04 §4.6.1：上传/列表/详情/删除/重建索引）

2026-08-23：支持 Word（.docx/.doc）上传——扩展名白名单 + 大小上限 + 魔数内容校验
（防改后缀伪装）；新增 GET /documents/{id}/text 解析文本预览（docx/txt/md 统一文本预览，
PDF 仍走 pdf.js 渲染）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from ..schemas import (
    ApplyEdits,
    CancelUploadOut,
    DocDiagramOut,
    DocImageOut,
    DocParagraphOut,
    DocTableOut,
    DocTextOut,
    DocumentOut,
    EditApplyOut,
    OkOut,
    PrecheckOut,
    ReindexOut,
    RestoreBackupOut,
    UploadProgressOut,
)
from ..services import docx_edit
from .pagination import DEFAULT_LIMIT, LimitParam, OffsetParam

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

#: 支持的上传扩展名白名单（PDF / Markdown / 纯文本 / Word）
_ALLOWED_EXT = {".pdf", ".md", ".markdown", ".mdown", ".mkd", ".txt", ".doc", ".docx"}
#: 单文件大小上限 50MB（NFR：防超大文件拖垮解析）
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
#: 魔数：docx = zip 头；doc = OLE 复合文档头；pdf = %PDF
_DOCX_MAGIC = b"PK\x03\x04"
_DOC_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
_PDF_MAGIC = b"%PDF"


def _prune_page_shots(cache_dir: Path, max_bytes: int = 200 * 1024 * 1024) -> None:
    """page_shots 渲染缓存按体积淘汰（2026-09-06）：超 200MB 时按 mtime 删最旧 20%。

    此前缓存只增不减——PDF 阅读器每页/dpi 组合都会落盘 PNG（MB 级），长使用会累积到 GB 级。"""
    try:
        shots = [f for f in cache_dir.iterdir() if f.is_file()]
        total = sum(f.stat().st_size for f in shots)
        if total <= max_bytes:
            return
        excess = int(total * 0.2)
        for f in sorted(shots, key=lambda x: x.stat().st_mtime):
            size = f.stat().st_size
            f.unlink(missing_ok=True)
            excess -= size
            if excess <= 0:
                break
    except OSError:
        pass  # 清理失败不影响正常响应


async def _read_upload_content(file: UploadFile, request: Request) -> bytes:
    """流式读取上传内容并强制大小上限（2026-09-06）。

    此前 `await file.read()` 先全量入内存、之后才判 50MB——单请求即可打出等额内存尖峰，
    并发时被放大为 DoS 面。现改为：
    1. Content-Length 预检（零成本；multipart 封装开销留 1MB 余量）；
    2. 分块流式读取，累计超限立即 413，不再继续读入。
    """
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > _MAX_UPLOAD_BYTES + 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB 上限")
    total, parts = 0, []
    while True:
        chunk = await file.read(256 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"文件超过 {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB 上限")
        parts.append(chunk)
    return b"".join(parts)


def _validate_upload(filename: str, content: bytes) -> str:
    """上传前校验（扩展名 / 大小 / 内容魔数），返回规范化文件名或抛 400。

    内容安全检测说明：魔数校验保证「扩展名与真实格式一致」（防伪装文件）；
    病毒/恶意内容扫描需外部引擎（如 ClamAV），本项目不内置——见 docs/实施记录 2026-08-23。
    """
    name = (filename or "").strip()
    if not name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="文件名不合法")
    ext = Path(name).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型「{ext or '（无扩展名）'}」；支持：PDF / Markdown / TXT / Word（.docx/.doc）",
        )
    if not content:
        raise HTTPException(status_code=400, detail="空文件")
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"文件超过 {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB 上限")
    # 魔数内容校验（防改后缀伪装）
    if ext == ".pdf" and not content.startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail="文件内容与扩展名不符（不是有效的 PDF）")
    if ext == ".docx" and not content.startswith(_DOCX_MAGIC):
        raise HTTPException(status_code=400, detail="文件内容与扩展名不符（不是有效的 Word .docx）")
    if ext == ".doc" and content[:8] != _DOC_MAGIC:
        raise HTTPException(status_code=400, detail="文件内容与扩展名不符（不是有效的 Word .doc）")
    return name


def _reject_duplicate_if_any(doc_svc, name: str, content: bytes, allow_duplicate: bool):
    """服务端内容去重（2026-09-06）：content_hash 命中**已就绪**文档 → 409 + 已有文档信息。

    此前去重只有 precheck 一个调用方（且前端 precheck 属死代码组件），异步上传链路
    从未做服务端去重 → 同一文档可无限重复入库（浪费解析/向量化与存储）。
    - 只拦截 ``status == "ready"`` 的重复：旧文档 failed/cancelled 时允许重传（旧的是垃圾）；
    - ``allow_duplicate=true`` 为显式逃生门（前端确认框/CLI 强制重传用）。
    命中时返回 JSONResponse(409)，未命中返回 None。"""
    if allow_duplicate:
        return None
    dup = doc_svc.check_duplicate(name, content)
    existing = dup.get("existing")
    # 只放行 failed/cancelled 的旧文档（重传合理）；uploaded/解析中/ready 的重复一律拦截
    # ——uploaded 意味着后台正在处理，放行会产生并发重复入库。
    if (
        dup.get("duplicate")
        and existing is not None
        and existing.status not in ("failed", "cancelled")
    ):
        return JSONResponse(
            status_code=409,
            content={
                "detail": f"内容与已有文档重复：「{existing.filename}」（文档 #{existing.id}），已跳过上传。"
                          "如确需重复上传，请在请求中加 allow_duplicate=true。",
                "existing_id": existing.id,
                "existing_filename": existing.filename,
            },
        )
    return None


@router.post("/documents", response_model=DocumentOut)
async def upload_document(
    file: UploadFile,
    request: Request,
    session_id: int | None = Query(None, description="可选：上传后自动把文档关联到该会话"),
    allow_duplicate: bool = Query(False, description="显式允许上传与已有文档内容完全相同的文件"),
) -> DocumentOut:
    content = await _read_upload_content(file, request)
    name = _validate_upload(file.filename or "", content)
    doc_svc = request.app.state.doc_svc
    blocked = _reject_duplicate_if_any(doc_svc, name, content, allow_duplicate)
    if blocked is not None:
        return blocked
    try:
        out = await doc_svc.upload_and_process(name, content, session_id=session_id)
    except ValueError as e:
        # 解析失败（如 .doc 老格式）：返回 400 明示，不静默
        raise HTTPException(status_code=400, detail=str(e)) from e
    return out


@router.post("/documents/async", response_model=DocumentOut)
async def upload_document_async(
    file: UploadFile,
    request: Request,
    session_id: int | None = Query(None, description="可选：上传后自动把文档关联到该会话"),
    allow_duplicate: bool = Query(False, description="显式允许上传与已有文档内容完全相同的文件"),
) -> DocumentOut:
    """异步上传：保存文件后立即返回 ``status=uploaded`` 的文档，重活交给后台任务处理。

    前端随后轮询 ``GET /documents/{id}/progress`` 看进度、可 ``POST /documents/{id}/cancel``
    取消（协作式，见 2026-09-02）。保留旧 ``POST /documents``（同步 block=True）供 CLI / 测试使用。
    2026-09-06：服务端内容去重——content_hash 命中已就绪文档时返回 409（可 allow_duplicate 覆盖）。
    """
    content = await _read_upload_content(file, request)
    name = _validate_upload(file.filename or "", content)
    doc_svc = request.app.state.doc_svc
    blocked = _reject_duplicate_if_any(doc_svc, name, content, allow_duplicate)
    if blocked is not None:
        return blocked
    try:
        out = await doc_svc.upload_and_process(name, content, session_id=session_id, block=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return out


@router.post("/documents/precheck", response_model=PrecheckOut)
async def precheck_document(file: UploadFile, request: Request) -> dict:
    """上传前重复检测：按内容 sha256 + 文件名返回是否重复。

    返回 {"duplicate", "same_name", "existing"(DocumentOut|null), "filename", "sha256"}
    前端据此弹「查看已有 / 继续上传 / 取消」确认框（见 DocumentPanel，2026-08-20）。
    """
    content = await _read_upload_content(file, request)
    name = _validate_upload(file.filename or "", content)
    doc_svc = request.app.state.doc_svc
    result = doc_svc.check_duplicate(name, content)
    result["filename"] = name
    result["sha256"] = hashlib.sha256(content).hexdigest()
    return result


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(
    request: Request,
    response: Response,
    limit: LimitParam = DEFAULT_LIMIT,
    offset: OffsetParam = 0,
) -> list[DocumentOut]:
    """文档列表。

    C4（2026-09-21）：支持 `limit`/`offset`，总数走 `X-Total-Count` 响应头
    （**保持数组返回** → 前端零改动，决策理由见 `api/pagination.py`）。
    """
    svc = request.app.state.doc_svc
    response.headers["X-Total-Count"] = str(svc.count_documents())
    return svc.list_documents(limit=limit, offset=offset)


@router.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, request: Request) -> DocumentOut:
    out = request.app.state.doc_svc.get_document(doc_id)
    if out is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return out


@router.delete("/documents/{doc_id}", response_model=OkOut)
def delete_document(doc_id: int, request: Request) -> dict:
    request.app.state.doc_svc.delete(doc_id)
    return {"ok": True}


@router.get("/documents/{doc_id}/progress", response_model=UploadProgressOut)
def upload_progress(doc_id: int, request: Request):
    """异步上传进度快照（前端轮询可视化用）。

    返回 {"doc_id","status","stage","done","total","pct","error","updated_at"}；
    文档已被取消清理（记录不存在）→ 404，前端据此停止轮询并刷新列表。
    """
    prog = request.app.state.doc_svc.get_upload_progress(doc_id)
    if prog is None:
        raise HTTPException(status_code=404, detail="文档不存在或没有上传进度")
    return prog


@router.post("/documents/reindex/cancel", response_model=OkOut)
async def cancel_reindex(request: Request) -> dict:
    """取消正在进行中的全量重建索引。

    **协作式**：取消后正在处理的那篇会跑完（单篇是同步嵌入推理，无法中途打断），
    随后循环停止并向流中产出一条 ``type=="cancelled"`` 事件。前端应同时
    ``abort`` 掉 SSE 连接——否则即使后端停了，前端仍会停在「进行中」。

    当前没有进行中的重建时调用无副作用（幂等），返回 ``{"ok": true}``。
    """
    request.app.state.doc_svc.cancel_reindex()
    return {"ok": True}


@router.post("/documents/{doc_id}/cancel", response_model=CancelUploadOut)
def cancel_upload(doc_id: int, request: Request) -> dict:
    """取消正在进行的异步上传（协作式：在阶段边界 / 下一批次生效，与重建索引同范式）。

    幂等——文档已处于终态（ready/failed/cancelled）时调用无副作用，返回 ``{"ok":true,"cancelled":false}``。
    """
    return request.app.state.doc_svc.cancel_upload(doc_id)


# ---------- C2：改稿直写 docx（段落视图 + track-changes 应用） ----------

def _trigger_reindex(background_tasks: BackgroundTasks, request: Request, doc_id: int) -> bool:
    """改稿内容已变化 → 后台**完整重处理**（fire-and-forget，不阻塞响应；失败仅记日志）。

    C3：apply-edits 与 restore-backup 内容变更后调用，使 RAG 检索与新文本一致。

    ★ 2026-09-30 修复（探针 `scripts/probe_edit_consistency.py` 证实）：改稿只改 docx
    源文件，原先调用的 `reindex()` 从 **DB chunks** 重建管线 —— chunks 未变 ⇒
    signature 原样命中旧 state ⇒ **0.01s 秒回旧向量**，检索到的仍是改稿前内容（假生效）。
    ⇒ 改为 `reprocess()`：重新解析 → 分块 → 嵌入的**完整流程**（后台执行，进度见
    `GET /documents/{id}/progress`；纯文本 ~20s，图密集无 VLM ~187s）。

    用 FastAPI ``BackgroundTasks`` 调度：后台任务在事件循环中被 ``await``，
    规避此前同步 endpoint（`apply_doc_edits`/`restore_doc_backup` 走线程池、无 running
    loop）内 ``asyncio.create_task`` 抛 ``RuntimeError`` 被吞、导致重索引静默失效的缺陷。
    """
    svc = request.app.state.doc_svc

    async def _job() -> None:
        try:
            await svc.reprocess(doc_id)
        except Exception:  # noqa: BLE001
            logger.warning("改稿后重处理失败 doc_id=%s", doc_id, exc_info=True)

    background_tasks.add_task(_job)
    return True


@router.get("/documents/{doc_id}/paragraphs", response_model=list[DocParagraphOut])
def list_doc_paragraphs(doc_id: int, request: Request) -> list[dict]:
    """段落视图（``P{i}`` 正文 + ``T..R..C..P{pi}`` 表格）：改稿工作台 / LLM 大纲的数据源。

    仅 .docx；每项含 pid / style / text / editable（空段、域代码段、媒体段不可编辑）。
    """
    path = request.app.state.doc_svc.storage_path(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="文档不存在或原文件已丢失")
    try:
        return docx_edit.list_paragraphs(path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"段落解析失败：{e}") from e


@router.get("/documents/{doc_id}/stream", response_model=list[dict])
def list_doc_stream(doc_id: int, request: Request) -> list[dict]:
    """文档流视图（C4·P1）：按 body 实际顺序输出段落/表格，供高保真渲染层按序对齐。

    段落 ``B{i}``；表格 ``B{i}``（cells 内为 ``B{i}.R{ri}.C{ci}.P{pi}``）。
    ``sectPr`` 不编址（渲染层不输出该元素，保证编址与渲染顺序一致）。
    """
    path = request.app.state.doc_svc.storage_path(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="文档不存在或原文件已丢失")
    try:
        return docx_edit.list_stream(path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"文档流解析失败：{e}") from e


@router.post("/documents/{doc_id}/apply-edits", response_model=EditApplyOut)
def apply_doc_edits(doc_id: int, body: ApplyEdits, request: Request, background_tasks: BackgroundTasks) -> dict:
    """改稿直写：按 edit ops 应用修改（track-changes 写入，quote 校验阀宁拒不错）。

    落盘前自动创建 `.bak-{ts}` 备份（C3）；成功且有实际写入时后台触发重嵌入。
    返回 {"applied", "rejected", "total", "paragraph_count_after", "backup", "reindex_triggered"}。
    """
    path = request.app.state.doc_svc.storage_path(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="文档不存在或原文件已丢失")
    try:
        out = docx_edit.apply_edits(path, [e.model_dump() for e in body.edits])
    except docx_edit.DocxNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (docx_edit.DocxNotEditable, docx_edit.DocxEditError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    out["reindex_triggered"] = bool(out["applied"]) and _trigger_reindex(background_tasks, request, doc_id)
    return out


@router.post("/documents/{doc_id}/restore-backup", response_model=RestoreBackupOut)
def restore_doc_backup(doc_id: int, request: Request, background_tasks: BackgroundTasks) -> dict:
    """一键撤销（C3，2026-09-06 多级撤销）：回退到上一个用户操作前的快照，并后台触发重嵌入。

    撤销走显式 undo/redo 双栈（.baks.json）：撤销把当前状态转入 redo 栈（不进 undo 栈，
    不再出现"撤销又被撤销"的横跳）；新操作清空 redo。undo 栈空 → 400（没有可撤销的操作）。
    """
    path = request.app.state.doc_svc.storage_path(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="文档不存在或原文件已丢失")
    try:
        out = docx_edit.restore_from_backup(path)
    except docx_edit.DocxNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except docx_edit.DocxEditError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    out["reindex_triggered"] = _trigger_reindex(background_tasks, request, doc_id)
    return out


@router.get("/documents/{doc_id}/download")
def export_document(doc_id: int, request: Request, mode: str = Query("tracked")) -> Response:
    """修改后保存导出（C4 增补）。注意路径为 /download —— /export 已被格式转换导出占用。

    - ``mode=tracked``（默认带痕版）：下载当前文件 —— 修订痕迹完整保留，Word 中可继续接受/拒绝；
    - ``mode=clean``（干净版）：接受全部修订后的 docx（ins 解包、del 移除），仅 .docx 支持。

    以附件（attachment）形式触发浏览器下载；文件名沿用文档名（干净版加后缀）。
    """
    svc = request.app.state.doc_svc
    doc = svc.get_document(doc_id)
    path = svc.storage_path(doc_id)
    if doc is None or path is None:
        raise HTTPException(status_code=404, detail="文档不存在或原文件已丢失")

    from urllib.parse import quote

    if mode == "tracked":
        return FileResponse(path, filename=doc.filename, content_disposition_type="attachment")
    if mode != "clean":
        raise HTTPException(status_code=400, detail=f"mode 仅支持 tracked / clean，收到：{mode!r}")

    try:
        data = docx_edit.accept_all_revisions(path)
    except docx_edit.DocxNotEditable as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"导出失败：{e}") from e

    stem = Path(doc.filename).stem or "document"
    out_name = f"{stem}_干净版.docx"
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(out_name)}"},
    )


@router.post("/documents/{doc_id}/reindex", response_model=DocumentOut)
async def reindex_document(doc_id: int, request: Request) -> DocumentOut:
    try:
        out = await request.app.state.doc_svc.reindex(doc_id)
    except Exception as e:  # noqa: BLE001  暴露真实原因（如 real 模式权重未就绪），而非泛化 500
        raise HTTPException(status_code=500, detail=f"重建索引失败：{e}") from e
    if out is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return out


@router.post("/documents/reindex", response_model=ReindexOut)
async def reindex_all_documents(request: Request) -> dict:
    """全量重建索引：切换 embedding 模式 / BGE 稀疏 / 中文分词后调用，使新配置对所有文档生效。

    返回 {"reindexed": int, "failed": [{id, error}], "total": int}。单篇失败计入 failed 不影响其余文档，
    与前端的「重建索引」按钮对应。
    """
    try:
        return await request.app.state.doc_svc.reindex_all()
    except Exception as e:  # noqa: BLE001  暴露真实原因（如配置读取失败），而非泛化 500
        raise HTTPException(status_code=500, detail=f"重建索引失败：{e}") from e


@router.post("/documents/reindex/stream")
async def reindex_all_stream(request: Request):
    """全量重建索引（SSE 流式进度）：前端用于实时展示「已处理 X / 总数 Y」与逐篇状态。

    事件格式（每行 ``data: <json>``）：
      {"type":"progress","doc_id":int,"status":"ok|skip|error","error":str|null,"done":int,"total":int}
      {"type":"done","reindexed":int,"failed":[{"id":int,"error":str}],"total":int}

    异常路径也会产出一条 ``type=="done"`` 且带 ``error`` 的事件——生成器若直接抛错，
    SSE 连接会静默中断，前端会一直停在「进行中」永远等不到结束。
    """
    svc = request.app.state.doc_svc

    async def event_gen():
        try:
            async for ev in svc.reindex_all_stream():
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            logger.error("reindex stream 异常：%s", e, exc_info=True)
            err_ev = {
                "type": "done",
                "reindexed": 0,
                "failed": [],
                "total": 0,
                "error": f"重建索引中断：{e}",
            }
            yield f"data: {json.dumps(err_ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用反代缓冲，保证进度逐条下发
        },
    )


@router.get("/documents/{doc_id}/file")
def get_document_file(doc_id: int, request: Request):
    """返回文档原始文件（PDF/文本），供 ER-22 阅读器引用跳转使用。

    属于应用自身数据（data/uploads/），可按需预览；非个人目录，安全。
    content_disposition_type="inline"：允许前端 iframe 内联渲染（文档预览面板 /
    PdfReader 引用跳转），否则浏览器会把响应当附件弹出下载。
    显式下载由前端「导出」按钮带 download 属性 / blob 实现。
    """
    out = request.app.state.doc_svc.get_document(doc_id)
    if out is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    doc = None
    with request.app.state.chat_svc._Session() as db:
        from ..models.db import Document

        doc = db.get(Document, doc_id)
    if doc is None or not doc.storage_path:
        raise HTTPException(status_code=404, detail="原始文件不可用")
    path = Path(doc.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="原始文件已缺失")
    # 改稿工作台会反复就地修改此文件（apply-edits），再重新渲染图文视图。
    # 1) 必须禁用缓存：否则浏览器可能复用旧字节，导致「已改写但视图仍是原文」。
    # 2) 改为一次性读入内存返回（而非 FileResponse 流式）：FileResponse 在传输期间
    #    一直持有源文件句柄，紧随其后的 apply-edits 做 os.replace 会撞 WinError 32
    #    共享冲突。单文件上限 50MB，本地单用户场景内存直读无压力。
    import mimetypes
    from urllib.parse import quote

    media_type = mimetypes.guess_type(doc.filename)[0] or "application/octet-stream"
    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(doc.filename)}",
            "Cache-Control": "no-store, must-revalidate",
        },
    )


@router.get("/documents/{doc_id}/text", response_model=DocTextOut)
def get_document_text(doc_id: int, request: Request) -> dict:
    """返回文档解析后的纯文本（parsed_dir 落盘 → chunk 表回退）。

    用途：Word（.docx）、Markdown、TXT 等非 PDF 格式的在线文本预览——直接 fetch 原始
    二进制会乱码，统一走解析文本；PDF 仍由前端 pdf.js 渲染。与全文检索共用同一解析产物，
    保证「预览内容 == 检索内容」一致。
    """
    doc_svc = request.app.state.doc_svc
    if doc_svc.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    text = doc_svc.get_parsed_text(doc_id) or ""
    return {"doc_id": doc_id, "text": text}


@router.get("/documents/{doc_id}/images/{filename}")
def get_document_image(doc_id: int, filename: str, request: Request):
    """返回 PDF 解析时提取的图片资源（upload_dir/{doc_id}/images/）。

    ER-03 图表提取：引用卡/阅读器可用该端点渲染图片缩略图。
    文件名做 basename 校验，防路径穿越。
    """
    safe = Path(filename or "").name
    if not safe or safe != filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    img_dir = request.app.state.settings.upload_dir / str(doc_id) / "images"
    path = img_dir / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(path)


@router.get("/documents/{doc_id}/images", response_model=list[DocImageOut])
def list_document_images(doc_id: int, request: Request) -> list[dict]:
    """列出文档提取的图片资源清单（供前端缩略图渲染）。

    从 chunks 表中 chunk_type 为 figure / image_table 的记录回取 file/caption/page/kind。
    图片字节仍由 GET /documents/{doc_id}/images/{filename} 提供。
    """
    if request.app.state.doc_svc.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    out: list[dict] = []
    with request.app.state.chat_svc._Session() as db:
        from ..models.db import Chunk

        rows = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
    for c in rows:
        try:
            m = json.loads(c.meta_json)
        except Exception:  # noqa: BLE001  损坏 meta 跳过
            continue
        ct = m.get("chunk_type")
        if ct in ("figure", "image_table") and m.get("file"):
            out.append(
                {
                    "filename": m["file"],
                    "page": c.page,
                    "kind": ct,
                    "caption": m.get("caption") or "",
                }
            )
    return out


@router.get("/documents/{doc_id}/tables", response_model=list[DocTableOut])
def list_document_tables(doc_id: int, request: Request) -> list[dict]:
    """列出文档提取的表格清单（供前端表格查看器渲染）。

    从 chunks 表中 chunk_type 为 table 的记录回取 markdown/rows/cols/header/page。
    图片型表格（扫描件/截图）不在其中——它们以 image_table 图块形式由 /images 列表提供。
    """
    if request.app.state.doc_svc.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    out: list[dict] = []
    with request.app.state.doc_svc._Session() as db:
        from ..models.db import Chunk

        rows = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
    for c in rows:
        try:
            m = json.loads(c.meta_json)
        except Exception:  # noqa: BLE001  损坏 meta 跳过
            continue
        if m.get("chunk_type") != "table":
            continue
        out.append(
            {
                "page": c.page,
                "markdown": c.content,
                "rows": m.get("rows", 0),
                "cols": m.get("cols", 0),
                "bbox": m.get("bbox"),
                "header": m.get("header", []),
            }
        )
    return out


@router.get("/documents/{doc_id}/diagrams", response_model=list[DocDiagramOut])
def list_document_diagrams(doc_id: int, request: Request) -> list[dict]:
    """列出文档提取的矢量结构图清单（供前端结构图查看器渲染，P0 后端先行）。

    从 chunks 表中 chunk_type 为 diagram 的记录回取 page/bbox/labels/file。
    结构图栅格化 PNG 由 GET /documents/{doc_id}/images/{filename} 提供（同一 images 目录）。
    """
    if request.app.state.doc_svc.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    out: list[dict] = []
    with request.app.state.doc_svc._Session() as db:
        from ..models.db import Chunk

        rows = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
    for c in rows:
        try:
            m = json.loads(c.meta_json)
        except Exception:  # noqa: BLE001  损坏 meta 跳过
            continue
        if m.get("chunk_type") != "diagram":
            continue
        out.append(
            {
                "page": c.page,
                "bbox": m.get("bbox"),
                "labels": m.get("labels", []),
                "file": m.get("file"),
                "mermaid": m.get("mermaid", ""),
                "description": m.get("description", ""),
                "text": c.content,
            }
        )
    return out


@router.get("/documents/{doc_id}/pages/{page}/image")
def get_document_page_image(
    doc_id: int,
    page: int,
    request: Request,
    dpi: int = Query(120, ge=20, le=300, description="渲染分辨率（72 基准，上限 300 防超大图）"),
    bbox: str | None = Query(None, description="页面坐标 x0,y0,x1,y1（PDF 点坐标，可省略=整页）"),
) -> Response:
    """PDF 页面栅格化截图（P0-2.3）：原图缺失/扫描件整页预览/整页存图兜底。

    零新依赖：pymupdf get_pixmap 渲染整页或按 bbox 区域；产物落盘缓存
    （upload_dir/{doc_id}/page_shots/，按 page+dpi+bbox 哈希命名），重复请求直接返回。
    bbox 与页面求交集后渲染，避免越界。仅支持 PDF（docx 无页面概念，其图片已单独提取）。
    """
    doc_svc = request.app.state.doc_svc
    if doc_svc.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    with request.app.state.chat_svc._Session() as db:
        from ..models.db import Document

        doc = db.get(Document, doc_id)
    if doc is None or not doc.storage_path:
        raise HTTPException(status_code=404, detail="原始文件不可用")
    path = Path(doc.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="原始文件已缺失")
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="仅支持 PDF 页面截图")
    if page < 1:
        raise HTTPException(status_code=400, detail="页码从 1 开始")

    clip = None
    if bbox:
        try:
            nums = [float(x) for x in bbox.split(",")]
        except ValueError as e:
            raise HTTPException(status_code=400, detail="bbox 格式应为 x0,y0,x1,y1（逗号分隔 4 个数字）") from e
        if len(nums) != 4 or nums[0] >= nums[2] or nums[1] >= nums[3]:
            raise HTTPException(status_code=400, detail="bbox 应为 x0<x1 且 y0<y1")

        import pymupdf  # 惰性：仅截图场景

        clip = pymupdf.Rect(nums[0], nums[1], nums[2], nums[3])

    # 缓存：page+dpi+bbox 哈希命名，重复请求零渲染
    cache_dir = request.app.state.settings.upload_dir / str(doc_id) / "page_shots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{page}|{dpi}|{bbox or ''}".encode()).hexdigest()[:16]
    cache_path = cache_dir / f"p{page}-{key}.png"
    if not cache_path.exists():
        _prune_page_shots(cache_dir)  # 2026-09-06：渲染前先清理（此前缓存无淘汰，磁盘只增不减）
        try:
            import pymupdf  # 惰性：仅截图场景

            with pymupdf.open(str(path)) as pdf:
                if page > pdf.page_count:
                    raise HTTPException(status_code=404, detail=f"页码超出范围（共 {pdf.page_count} 页）")
                pg = pdf[page - 1]
                rect = pg.rect
                if clip is not None:
                    clip &= rect  # 与页面求交集，防越界
                    if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                        raise HTTPException(status_code=400, detail="bbox 超出页面范围")
                    target = clip
                else:
                    target = rect
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), clip=target)
                cache_path.write_bytes(pix.tobytes("png"))
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001  渲染失败给通用错误
            raise HTTPException(status_code=500, detail="页面渲染失败") from None
    return Response(content=cache_path.read_bytes(), media_type="image/png")
