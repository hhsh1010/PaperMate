"""PaperMate · FastAPI 应用工厂（docs/04 架构 + docs/07 M3.1）

组装：数据层引擎 → SessionLocal → service 单例 → 挂载 REST 路由 + healthz。
service 单例挂在 app.state，路由通过 request.app.state 取用（依赖方向 api → services → lightcore）。

注意：processing_router 必须先于 documents_router 挂载——其 `/documents/compare` 为字面路径，
须抢在 `/documents/{doc_id}` 参数路由之前注册，否则会被 {doc_id} 吞掉。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from ..config import get_settings
from ..memory_hub import reset_memory_manager
from ..models.db import init_db, make_engine
from ..services.chat_service import ChatService
from ..services.collection_service import CollectionService
from ..services.document_service import DocumentService
from ..services.processing_service import ProcessingService
from ..services.session_service import SessionService
from ..services.summary_service import SummaryService
from ..services.workspace_service import WorkspaceService
from .errors import install_exception_handlers
from .log_quieting import apply_log_quieting
from .middleware import TraceIdMiddleware
from .routes_collections import router as collections_router
from .routes_documents import router as documents_router
from .routes_eval import router as eval_router
from .routes_feedback import router as feedback_router
from .routes_memory import router as memory_router
from .routes_metrics import router as metrics_router
from .routes_processing import router as processing_router
from .routes_prompt import router as prompt_router
from .routes_search import router as search_router
from .routes_sessions import router as sessions_router
from .routes_settings import router as settings_router
from .routes_summary import router as summary_router
from .routes_traces import router as traces_router
from .routes_transform import router as transform_router
from .routes_usage import router as usage_router
from .routes_workspaces import router as workspaces_router

logger = logging.getLogger(__name__)

#: 是否在服务启动后**后台预热** RAG 模型（T-05）。默认开，受 `PAPERMATE_PREWARM` 控制。
#:
#: ⚠️ **为什么必须有这个开关**（实测踩到）：预热会真的加载 BGE-M3（**22s**）。
#: 而测试环境 `rag_embedder` 也是 `real` → 每个 `create_app()` 都会拉起一个加载线程；
#: 多线程争 GPU/磁盘会让测试又慢又不稳。
#: 因此 `tests/conftest.py` 里设 `PAPERMATE_PREWARM=0` 关闭它。
#: 生产（`python -m papermate.api`）不设该变量 → 保持预热。
_PREWARM_ENV_VAR = "PAPERMATE_PREWARM"
_PREWARM_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def _prewarm_enabled() -> bool:
    """读环境变量判断是否预热（抽成函数是为了**可测** —— 模块级常量测不到）。"""
    return os.environ.get(_PREWARM_ENV_VAR, "1").strip().lower() not in _PREWARM_OFF_VALUES


async def _prewarm_rag(doc_svc: DocumentService) -> None:
    """后台预热重型 RAG 模型（T-05，2026-09-26）。**永不把异常抛给调用方**。

    背景见 `_lifespan` 里的注释；核心数字：冷构造 embedder + reranker 实测 17~23s，
    而单篇索引冷加载只要 ≈100ms —— 预热就是把这 17~23s 从用户请求挪到启动期。

    失败语义：只记 warning。首个真实检索请求会重试，并由 `_get_embedder` 既有的
    熔断逻辑（同签名快速失败）避免逐篇重试。
    """
    t0 = time.perf_counter()
    try:
        timings = await doc_svc.warm_up()
    except asyncio.CancelledError:
        logger.info("RAG 模型预热被取消（服务关闭中）")
        raise
    except Exception:  # noqa: BLE001  预热失败绝不影响服务可用性
        logger.warning("RAG 模型预热失败（不影响服务；首个检索请求会重试）", exc_info=True)
        return
    if timings.get("skipped"):
        logger.debug("RAG 模型预热跳过：%s", timings.get("reason", ""))
    else:
        logger.info(
            "RAG 模型预热完成：embedder %.2fs · reranker %.2fs · 合计 %.2fs"
            "（预热后用户首问只需付索引冷加载 ≈100ms/篇，无需等模型）",
            timings.get("embedder_s", 0.0), timings.get("reranker_s", 0.0),
            time.perf_counter() - t0,
        )


def create_app() -> FastAPI:
    settings = get_settings()
    # 2026-08-29 A 档：长期记忆改为进程级共享单例（papermate.memory_hub），这里先重置，
    # 保证新建 app 时按当前 settings.data_dir 重新加载（测试隔离 / 单进程多 app 场景）。
    reset_memory_manager()
    engine = make_engine(settings.db_url)
    init_db(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    session_svc = SessionService(SessionLocal)
    doc_svc = DocumentService(SessionLocal, settings)
    summary_svc = SummaryService(doc_svc, settings, SessionLocal)
    chat_svc = ChatService(session_svc, doc_svc, settings, SessionLocal, summary_svc=summary_svc, collection_svc=CollectionService(SessionLocal))
    processing_svc = ProcessingService(doc_svc, settings, SessionLocal)

    # 启动时回填旧文档 content_hash（重复上传检测用；缺失/失败自动跳过）
    try:
        doc_svc.backfill_hashes()
    except Exception:  # noqa: BLE001  回填失败不影响服务启动
        pass

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # 2026-09-17 日志降噪：静音"成功且幂等的高频请求"（实测 healthz 轮询一项就占日志 47%）。
        # 挂在 lifespan 启动时机，确保晚于 uvicorn 的 dictConfig。见 api/log_quieting.py。
        apply_log_quieting()
        # C5（2026-09-21）：中断任务恢复 —— 把上次进程遗留的中间态文档标记为 failed。
        # 为什么放在启动路径：这些状态**只由进程内存驱动**，重启后不可能自行恢复；
        # 不处理的话它们会永久停在"处理中"（用户既看不到原因，也没有重试入口）。
        # 放在 to_thread：DB 写入是同步阻塞调用，不能直接跑在 async 函数里（既有纪律）。
        await asyncio.to_thread(processing_svc.recover_interrupted)
        # T-05（2026-09-26）：**后台预热重型模型**。
        #
        # ★ 为什么不 await：冷构造 BGE-M3 + reranker 实测 **17~23s**
        #   （`scripts/probe_cold_start.py`）。await 会把"服务就绪"推迟同样久 ——
        #   健康检查、文档列表、预览全都要等模型读完盘，比现状更糟
        #   （现状至少非检索接口可用）。用 `create_task`：服务**立即就绪**，模型后台加载。
        #
        # ★ 为什么预热值得做：单篇索引冷加载其实只要 **≈100ms**
        #   （CPU 部分 1.51s ÷ 15 篇 = faiss 26ms + pickle 1ms + BM25/jieba 73ms），
        #   但用户首个检索请求此前要独自承担 17~23s 的模型加载。
        #   预热把这段挪到启动后，**并填充单例缓存** → 用户请求只需付那 100ms。
        #
        # ★ 为什么不阻塞事件循环：`warm_up` 内部走 `_get_embedder` / `_get_reranker`，
        #   二者已改为 `asyncio.to_thread`（同轮修复）→ 加载期间其它请求照常响应。
        #
        # ⚠️ 受 `PAPERMATE_PREWARM` 控制（默认开）；测试环境关闭 —— 每个 app 测试都拉起
        #   一个 22s 的加载线程会让测试又慢又不稳（见 `_PREWARM_ENABLED` 注释）。
        prewarm_task = (
            asyncio.create_task(_prewarm_rag(doc_svc), name="t05-prewarm-rag")
            if _prewarm_enabled() else None
        )
        yield
        # 关闭时：先取消未完成的预热（避免后台线程在 shutdown 后被 join 卡住），再停 worker。
        if prewarm_task is not None and not prewarm_task.done():
            prewarm_task.cancel()
            # 预热任务自身已吞掉异常；这里只等它收尾，不让关闭路径被它中断。
            await asyncio.gather(prewarm_task, return_exceptions=True)
        # 关闭时停止批量处理 worker（幂等；未启动则 no-op）
        await processing_svc.shutdown()

    app = FastAPI(title="PaperMate API", version="v1", lifespan=_lifespan)
    # 统一错误出口（2026-09-15，P0-2）：422 校验错误压平为可读字符串（原为对象数组，
    # 前端会显示 "[object Object]"）、未捕获异常转 JSON 500、所有错误附 code + trace_id。
    # 见 docs/后端工程体系评估.md 与 api/errors.py。
    install_exception_handlers(app)
    # C2（2026-09-21）：请求级 trace_id —— **纯 ASGI** 中间件（BaseHTTPMiddleware 对 SSE 有风险）。
    # 让每个请求（含成功的）都能与日志关联；此前只有出错时才有 trace_id。
    # ⚠️ add_middleware 必须在**路由注册之前**、且在 lifespan 之外；Starlette 会把中间件
    # 逆序包裹，这里注册即最外层，先于 CORS/错误处理拿到请求。
    app.add_middleware(TraceIdMiddleware)
    app.state.settings = settings
    app.state.session_svc = session_svc
    app.state.doc_svc = doc_svc
    app.state.chat_svc = chat_svc
    app.state.summary_svc = summary_svc
    app.state.processing_svc = processing_svc
    app.state.collection_svc = chat_svc._collections
    app.state.workspace_svc = WorkspaceService(SessionLocal)
    # 会话级流取消事件（2026-09-07）：/sessions/{id}/stop 置位，流式端点轮询消费
    app.state.stream_cancels = {}

    app.include_router(collections_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(processing_router, prefix="/api/v1")  # 先于 documents_router（compare 字面路径）
    app.include_router(prompt_router, prefix="/api/v1")
    app.include_router(transform_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(eval_router, prefix="/api/v1")
    app.include_router(search_router, prefix="/api/v1")
    app.include_router(usage_router, prefix="/api/v1")
    app.include_router(settings_router, prefix="/api/v1")
    app.include_router(summary_router, prefix="/api/v1")
    app.include_router(traces_router, prefix="/api/v1")
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(memory_router, prefix="/api/v1")
    app.include_router(metrics_router, prefix="/api/v1")
    app.include_router(workspaces_router, prefix="/api/v1")

    @app.get("/api/v1/healthz")
    def healthz():
        from ..runtime_settings import get_state

        rt = get_state()
        # 2026-09-06：移除 "db": str(engine.url)——healthz 此前泄露本地绝对路径；
        # 并增加真实探活（SELECT 1），DB 锁死/损坏时状态灯不再误报"在线"。
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception:  # noqa: BLE001  探活失败本身要报告而非中断
            db_ok = False
        return {
            "status": "ok" if db_ok else "degraded",
            "db_ok": db_ok,
            # 2026-09-15：是否监听非回环地址。本项目**无鉴权**，暴露状态应当可见
            # （前端可据此提示；启动时另有醒目告警）。判定口径见 config.Settings.network_exposed。
            "network_exposed": settings.network_exposed,
            # 口径修正（2026-09-06）：此前 default_model 读配置文件默认值（恒 "fake"），
            # 与用户实际使用的运行时模型不符——统一走 runtime_settings 运行时口径。
            "default_model": rt.get("default_model", settings.default_model),
            "default_embedder": rt.get("default_embedder", settings.default_embedder),
            "rag_embedder": rt.get("rag_embedder", settings.default_embedder),
            "use_bge_sparse": rt.get("use_bge_sparse", settings.use_bge_sparse),
            "use_jieba": rt.get("use_jieba", settings.use_jieba),
        }

    # ---- 生产部署：托管前端构建产物（2026-09-06，此前 deploy/ 为空、web/dist 无服务方）----
    # 必须挂载在全部 API router 之后，否则 "/" 会吞掉 /api/v1/* 路由。
    # html=True：未知路径回退 index.html，支持前端 hash/hash 路由刷新。
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    # parents: [0]=api [1]=papermate [2]=src [3]=项目根
    dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
    else:
        logging.getLogger(__name__).warning(
            "未找到前端构建产物 %s（生产部署请先在 web/ 下执行 npm run build）；"
            "当前仅提供 API。",
            dist,
        )

    return app
