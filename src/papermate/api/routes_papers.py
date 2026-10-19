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

import asyncio
import logging

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
from ..services.paper_translate import (
    MAX_OUTPUT_TOKENS,
    MAX_TITLES,
    build_messages,
    describe_llm_error,
    is_fake_model,
    parse_titles_json,
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
    #: ★ 总命中条数（arXiv 的 totalResults）——不让用户误以为"这 20 条就是全部"
    total: int = 0
    #: 是否还有更多（total > start + len(items)）
    has_more: bool = False
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
    created_at_iso: str = ""


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
    # ★ 防越界：arXiv 的深层分页很慢、位置太深还会直接报错 ⇒ 在入口就挡掉
    #   （前端上限是 100 条，这里留足余量 500，不做成"正常使用会撞到"的限制）
    if max(0, int(start)) + max(1, int(limit)) > 500:
        raise HTTPException(
            status_code=400,
            detail="翻得太深了（最多取前 500 条）；建议收窄关键词或用分类筛选",
        )
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
    svc = _search_svc(request)
    try:
        reader = getattr(svc, "search_page", None)
        if reader is not None:
            hits, total = await reader(used, limit=limit, start=start, sort=sort,
                                       scope=scope, categories=cats)
        else:
            # 兼容只实现 search() 的替身（测试桩/第三方实现）：没有总数就按 0 上报，不瞎猜
            hits, total = await svc.search(used, limit=limit, start=start, sort=sort,
                                           scope=scope, categories=cats), 0
    except PaperSearchError as e:
        # 上游不可用（超时/限流/解析失败）⇒ 502，前端可提示"稍后重试"
        raise HTTPException(status_code=502, detail=str(e)) from e
    return PaperSearchOut(
        query=query,
        query_used=used,
        translations=[TranslationOut(zh=a, en=b) for a, b in applied],
        remaining_chinese=has_han(used),
        total=total,
        has_more=total > (max(0, int(start)) + len(hits)),
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


# ------------------------------------------------------------------ 标题中译

class TranslateItemIn(BaseModel):
    source_id: str
    title: str


class TranslateIn(BaseModel):
    items: list[TranslateItemIn]


class TranslatedItemOut(BaseModel):
    source_id: str
    #: 中文标题；**空串表示这条没译出来**（前端保留英文原文，不要显示空）
    zh: str = ""


class TranslateOut(BaseModel):
    items: list[TranslatedItemOut]


@router.post("/papers/translate-titles", response_model=TranslateOut)
async def translate_paper_titles(payload: TranslateIn, request: Request) -> TranslateOut:
    """把检索结果的**英文标题批量译成中文**（一次 LLM 调用，最多 20 条）。

    ★ 为什么必须拦住 fake：`_llm_from_raw` 对未知/缺失模型会**静默退化成 ApiFakeLLM**，
      那种情况下会返回占位文本 ⇒ 用户看到一堆假标题，比直接报错更糟。
    ★ 成本：一次调用 ≈ 20 条短标题，输出上限 2000 tokens，**不点按钮不会产生任何调用**。
    """
    _require_enabled()
    items = (payload.items or [])[:MAX_TITLES]
    if not items:
        raise HTTPException(status_code=400, detail="没有要翻译的标题")

    from .. import runtime_settings  # 延迟导入：避免模块级循环依赖

    model = str(runtime_settings.get_state().get("default_model") or "")
    if is_fake_model(model):
        raise HTTPException(
            status_code=400,
            detail="当前没有配置可用的模型（或仍是 fake），无法翻译标题；请先在设置页配置模型与密钥。",
        )

    llm = request.app.state.chat_svc._llm_from_raw(model)
    # ★ 必须构造 ChatMessage（不能传普通 dict）：router 里 to_openai() 要取 role.value，
    #   传 dict 会 `AttributeError: 'dict' object has no attribute 'to_openai'`（线上实际踩过）。
    from lightcore.llm.base import ChatMessage, Role  # 局部导入，避免 API 层顶部多一条跨层依赖

    raw_msgs = build_messages([it.title for it in items])
    msgs = [
        ChatMessage(
            role=Role.SYSTEM if m["role"] == "system" else Role.USER,
            content=m["content"],
        )
        for m in raw_msgs
    ]
    try:
        resp = await asyncio.wait_for(
            llm.chat(msgs, temperature=0.2, max_tokens=MAX_OUTPUT_TOKENS),
            timeout=45.0,
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail="翻译超时（45s），请重试或减少条数") from e
    except Exception as e:  # noqa: BLE001
        # ★ 模型调用失败 ⇒ **把上游状态码翻成人话**（402 余额 / 429 限流 / 403 免费额度 / 401 密钥…），
        #   只报 "HTTPStatusError" 用户无法判断下一步做什么；同时**不要**漏成 500。
        code, msg = describe_llm_error(e)
        # ★ 必须把"给用户的那句话"落到日志：4xx 不会被全局错误中间件记 detail，
        #   结果排查时只剩"HTTP 400"，完全无法判断是密钥/余额/额度/模型名（实际发生过）。
        logging.getLogger(__name__).warning("[paper_translate] 模型调用失败 → %s %s", code, msg)
        raise HTTPException(status_code=code, detail=msg) from e

    raw = getattr(resp, "content", "") or ""
    parsed = parse_titles_json(raw, len(items))
    if all(x is None for x in parsed):
        # ★ 必须把原始输出落到日志（截 500 字）：否则"无法解析"时**完全看不到模型说了什么**，
        #   就没法判断该放宽解析器还是该改 prompt（2026-10-19 实际发生过）。
        logging.getLogger(__name__).warning(
            "[paper_translate] 输出无法解析，raw(前 500 字)=%r", (raw or "")[:500]
        )
        raise HTTPException(status_code=502, detail="翻译服务返回的内容无法解析（不是预期的 JSON），请重试")
    return TranslateOut(items=[
        TranslatedItemOut(source_id=it.source_id, zh=(zh or ""))
        for it, zh in zip(items, parsed, strict=True)
    ])
