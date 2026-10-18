"""PaperMate · 论文检索与导入路由（阶段 2）

- `GET  /papers/search`            检索 arXiv（**只返回元数据，绝不下载**）
- `POST /papers/import`            提交导入任务（下载 → 去重 → 交文档管线）
- `GET  /papers/import/{task_id}`  任务状态（`downloading` 时带字节进度）
- `POST /papers/import/{task_id}/cancel` 取消

## 三条硬约束（网络功能）
1. **独立开关** `PAPERMATE_ENABLE_PAPER_SEARCH`，**默认关**；关时一律 **403**（不是静默失败）；
2. **只允许 arXiv 域**（防 SSRF）：校验在 `paper_import.validate_pdf_url`，前端回传的 url 视为不可信；
3. **下载有上限**（默认 60MB），且 `Content-Type` 与文件头都必须是 PDF。

## 与"会话级联网"的区别
会话的 `enable_network` 管的是**对话里**的联网；论文检索是**全局功能**（不属于任何会话），
所以用独立开关。设置页开关与前端一起在阶段 3 做。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.paper_import import (
    PaperImportError,
    PaperImportService,
    paper_import_enabled,
)
from ..services.paper_search import (
    SCOPES,
    ArxivClient,
    PaperSearchError,
    has_han,
    normalize_categories,
    translate_zh_query,
)

router = APIRouter(tags=["papers"])

#: 单次导入的条目上限（防止一次点"全选"把 50 篇一起排队）
MAX_IMPORT_ITEMS = 20


# ------------------------------------------------------------------ 出参模型

class PaperHitOut(BaseModel):
    source: str
    source_id: str
    version: str = ""
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    updated: str = ""
    pdf_url: str = ""
    abs_url: str = ""
    categories: list[str] = Field(default_factory=list)
    primary_category: str = ""
    comment: str = ""
    doi: str = ""


class TranslationOut(BaseModel):
    zh: str
    en: str


class PaperSearchOut(BaseModel):
    #: 用户原样输入的查询
    query: str
    #: 实际发给 arXiv 的查询（中文术语被替换后会与 query 不同）
    query_used: str = ""
    #: 做了哪些替换 —— 界面**必须**明示，不能悄悄换词
    translations: list[TranslationOut] = Field(default_factory=list)
    #: 替换后是否仍含汉字（true 表示"没翻干净"，前端应提示改英文）
    remaining_chinese: bool = False
    items: list[PaperHitOut]


class ImportItemIn(BaseModel):
    pdf_url: str
    source: str = "arxiv"
    source_id: str = ""
    version: str = ""
    title: str = ""


class ImportRequestIn(BaseModel):
    items: list[ImportItemIn]


class ImportTaskOut(BaseModel):
    id: str
    status: str
    pdf_url: str = ""
    source: str
    source_id: str
    version: str
    title: str
    filename: str
    downloaded: int
    total: int
    doc_id: int | None = None
    duplicate: bool = False
    error: str = ""


class ImportCreatedOut(BaseModel):
    tasks: list[ImportTaskOut]


class CancelOut(BaseModel):
    ok: bool
    cancelled: bool


# ------------------------------------------------------------------ 工具

def _require_enabled() -> None:
    """全局开关：默认关；关时**明确 403**并给出开启方式（不静默失败）。"""
    if not paper_import_enabled():
        raise HTTPException(
            status_code=403,
            detail="论文检索与导入默认关闭（网络功能需显式开启）："
                   "请设置环境变量 PAPERMATE_ENABLE_PAPER_SEARCH=1 后重启后端。",
        )


def _search_svc(request: Request) -> ArxivClient:
    return request.app.state.paper_search


def _import_svc(request: Request) -> PaperImportService:
    return request.app.state.paper_import


# ------------------------------------------------------------------ 接口

@router.get("/papers/search", response_model=PaperSearchOut)
async def search_papers(request: Request, q: str = "", limit: int = 10,
                        start: int = 0, sort: str = "relevance",
                        scope: str = "all", categories: str = "") -> PaperSearchOut:
    """检索 arXiv。**只拿元数据**：搜索页不会触发任何 PDF 下载。

    - `scope`：`all` | `ti` | `au` | `abs`
    - `categories`：逗号分隔，如 `cs.CL,cs.IR`
    ⚠️ 「参数不合法」是**客户端错误（400）**，只有上游失败才是 502 —— 两者不能混。
    """
    _require_enabled()
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="检索关键词不能为空")
    if (scope or "all").strip().lower() not in SCOPES:
        raise HTTPException(status_code=400,
                            detail=f"不支持的检索范围：{scope}（可选：{'/'.join(SCOPES)}）")
    try:
        cats = normalize_categories(categories)
    except PaperSearchError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # ★ 术语映射**只在本接口内**发生（唯一调用点）：客户端 ArxivClient 与 build_query 都不翻译
    #   ⇒ 其它调用方（脚本、未来的 Agent 工具）拿到的仍是原样关键词，是否翻译由调用方自己决定。
    used, applied = translate_zh_query(query)
    try:
        hits = await _search_svc(request).search(used, limit=limit, start=start, sort=sort,
                                                 scope=scope, categories=cats)
    except PaperSearchError as e:
        # 上游不可用（超时/限流/解析失败）⇒ 502，前端可提示"稍后重试"
        raise HTTPException(status_code=502, detail=str(e)) from e
    return PaperSearchOut(
        query=query,
        query_used=used,
        translations=[TranslationOut(zh=a, en=b) for a, b in applied],
        remaining_chinese=has_han(used),
        items=[PaperHitOut(**h.as_dict()) for h in hits],
    )


@router.post("/papers/import", response_model=ImportCreatedOut)
async def import_papers(payload: ImportRequestIn, request: Request) -> ImportCreatedOut:
    """提交导入任务（**立即返回**，下载在后台跑）。"""
    _require_enabled()
    items = payload.items or []
    if not items:
        raise HTTPException(status_code=400, detail="没有要导入的论文")
    if len(items) > MAX_IMPORT_ITEMS:
        raise HTTPException(status_code=400,
                            detail=f"单次最多导入 {MAX_IMPORT_ITEMS} 篇（收到 {len(items)} 篇）")
    svc = _import_svc(request)
    tasks = []
    for i, it in enumerate(items):
        try:
            tasks.append(svc.submit({
                "pdf_url": it.pdf_url,
                "source": it.source,
                "source_id": it.source_id,
                "version": it.version,
                "title": it.title,
            }))
        except PaperImportError as e:
            # 一条不合法 ⇒ 整批拒绝（避免"部分成功"让前端状态难以对账）
            raise HTTPException(status_code=400, detail=f"第 {i + 1} 条不合法：{e}") from e
    return ImportCreatedOut(tasks=[ImportTaskOut(**t.as_dict()) for t in tasks])


@router.get("/papers/import", response_model=ImportCreatedOut)
def list_import_tasks(request: Request, limit: int = 50) -> ImportCreatedOut:
    """列出最近的导入任务。

    ★ 前端切页/刷新后据此**恢复进度**（任务仍在后端跑，进度不该因为换了页面就消失）。
    注意：本路由必须声明在 `/papers/import/{task_id}` **之前**（字面路径优先）。
    """
    _require_enabled()
    svc = _import_svc(request)
    return ImportCreatedOut(tasks=[ImportTaskOut(**t.as_dict()) for t in svc.list_tasks(limit=limit)])


@router.get("/papers/import/{task_id}", response_model=ImportTaskOut)
def get_import_task(task_id: str, request: Request) -> ImportTaskOut:
    """查任务状态。`ingesting` 阶段请同时轮询 `/documents/{doc_id}/progress` 看细粒度进度。"""
    task = _import_svc(request).get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在（服务重启后未完成的导入需重新发起）")
    return ImportTaskOut(**task.as_dict())


@router.post("/papers/import/{task_id}/cancel", response_model=CancelOut)
def cancel_import_task(task_id: str, request: Request) -> CancelOut:
    """取消导入；已结束的返回 `cancelled=false`（幂等）。"""
    svc = _import_svc(request)
    task = svc.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return CancelOut(ok=True, cancelled=svc.cancel(task_id))
