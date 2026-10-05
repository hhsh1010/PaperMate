"""PaperMate · 文档服务（docs/06 + docs/07 M3.1）

上传 → 解析 → 分块 → 嵌入 → 索引 的状态机（uploaded→parsing→chunking→embedding→ready/failed），
并把每个文档构建好的 RAGPipeline 缓存在内存（按 doc_id），供会话级 CompositeRAGPipeline 组合检索。
删除文档级联删块 + 清理落盘文件。

嵌入器：默认 FakeHashEmbedder（秒级、无需权重）；配置 default_embedder=real 时走已下载的 BGE-M3。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from lightcore.rag.chunker import Chunker
from lightcore.rag.parser import DocumentParser
from lightcore.rag.pipeline import IndexingCancelled, RAGPipeline

from .. import runtime_settings
from ..config import Settings
from ..models.db import Chunk, Document, FigureEnrichCache, iso_utc
from ..schemas import DocumentOut
from .db_retry import write_with_retry

logger = logging.getLogger(__name__)

#: BGE-M3 权重目录的**显式覆盖**环境变量（与 `lightcore.rag.embedder` 消费同一变量）。
#: ★ 换权重目录 ⇒ **向量空间完全改变** ⇒ 旧索引不可复用（否则检索静默出错）。
_BGE_M3_PATH_ENV = "PAPERMATE_BGE_M3_PATH"


def _embedder_identity(rt: dict) -> str:
    """**嵌入器身份**（2026-10-04）：判定「换没换嵌入权重」的**唯一来源**。

    ## 为什么必须有它
    此前两条缓存链都只看 `rag_embedder` 这个**模型名字符串**：
    - `_get_embedder` 的单例 key（`DocumentService` 内）
    - `_pipeline_signature` 的磁盘索引签名

    而 BGE-M3 的权重目录可由 `PAPERMATE_BGE_M3_PATH` **单独覆盖**（`lightcore.rag.embedder`
    会读它并优先于 HF 缓存）⇒ **模型名不变、权重已换** ⇒ 两条链都不失效
    ⇒ 换权重后**继续用旧向量**，检索结果错得没有任何报错。

    ## 为什么用 **env 原始值**而不是解析后的快照路径
    `resolve_local_hf_snapshot` 要 `glob` + 多次 `isfile`，而本函数在 `_get_embedder()`
    的**缓存 key** 里每次调用都会算 ⇒ 用解析路径会给每次调用加文件系统开销。
    env 读取几乎零成本，且覆盖了换权重的**主场景**（显式指定权重目录）。

    ## ★ 诚实边界（不覆盖的场景）
    「直接在 HF 缓存目录里替换权重文件、且**不改** `PAPERMATE_BGE_M3_PATH`」不会被检测到。
    属罕见场景；要强制失效可 `reindex` 或改 env 后重启。
    """
    return f'{rt.get("rag_embedder", "")}|{os.environ.get(_BGE_M3_PATH_ENV, "").strip()}'

#: 哨兵：重排器「不可用 / 已降级」状态（区别于尚未初始化的 None），
#: 用于缓存降级结果，避免 real 模式下逐篇重试加载 CrossEncoder 权重。
_RERANK_UNAVAILABLE = object()

#: T-05：`_pipelines` 的默认容量上限（LRU 淘汰）。
#:
#: 取值依据：单篇管线常驻估算 1~5 MB（faiss 索引 + 块文本 + BM25 倒排），
#: T-05 前置分析给的内存预算是 **+300MB** → 32 篇 ≈ 96~160MB，留有余量。
#: 而真实业务文档规模为 15 篇（`data/papermate.db` 实测）→ 默认值不会触发淘汰，
#: 只在文档量增长时起保护作用。
_DEFAULT_MAX_PIPELINES = 32


def _resolve_max_pipelines() -> int:
    """读 `PAPERMATE_MAX_PIPELINES`（<1 或非法值 → 回退默认，**不允许 0/负数**）。

    `0` 曾被考虑表示"不限制"，但那与"设上限防 OOM"的初衷相反
    —— 一个手滑的 0 会让保护静默失效。所以非法值一律回退到默认值。
    """
    raw = os.environ.get("PAPERMATE_MAX_PIPELINES", "")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_PIPELINES
    return val if val >= 1 else _DEFAULT_MAX_PIPELINES


#: T-05b（2026-09-27）：预热时顺便构建 pipeline + 跑一次空检索的**文档篇数**。
#: 默认 1 —— 冷开销（CUDA kernel / faiss 首次搜索）是**进程级**的，
#: 跑通一篇就足以触发；多篇只增加常驻内存，收益递减。
_DEFAULT_PREWARM_DOCS = 1


def _resolve_prewarm_docs() -> int:
    """读 `PAPERMATE_PREWARM_DOCS`（`0` = 只预热模型，不碰 pipeline）。

    与 `_resolve_max_pipelines` 不同：**这里 0 是合法值**（表示"关闭 pipeline 预热"），
    因为它是优化项而非保护项 —— 关掉不会造成 OOM，只是首个请求多付约 2s。
    """
    raw = os.environ.get("PAPERMATE_PREWARM_DOCS", "")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_PREWARM_DOCS
    return val if val >= 0 else _DEFAULT_PREWARM_DOCS


def _safe_filename(name: str | None) -> str:
    """清洗上传文件名，防路径穿越（`../../x` 逃逸 upload_dir）。取 basename 且非空。"""
    base = Path(name or "").name
    return base if base and base not in (".", "..") else "document.txt"


def _bbox_overlap_ratio(a, b) -> float:
    """两 bbox 交集面积占 a 面积的比例（0~1），用于位图 figure 与矢量 diagram 去重。"""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area_a


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class _FigureEnrichCache:
    """`FigureEnrichCache` 表的轻量读写适配器（T-06 P1，2026-09-30）。

    注入 `FigureEnricher(cache=...)`：lightcore 层不依赖 ORM，只见
    `get(key)` / `put(key, value)` 两个动作；键语义（内容哈希+配置）由 enricher 生成，
    本类不理解也不需要理解。读写失败由 enricher 侧兜底（退回真实增强，不阻断）。
    """

    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    def get(self, key: str) -> dict | None:
        with self._sf() as db:
            row = db.get(FigureEnrichCache, key)
            if row is None:
                return None
            return {
                "ocr_text": row.ocr_text or "",
                "table_md": row.table_md or "",
                "vlm_caption": row.vlm_caption or "",
            }

    def put(self, key: str, value: dict) -> None:
        with self._sf() as db:
            db.merge(
                FigureEnrichCache(
                    cache_key=key,
                    ocr_text=str(value.get("ocr_text") or ""),
                    table_md=str(value.get("table_md") or ""),
                    vlm_caption=str(value.get("vlm_caption") or ""),
                )
            )
            db.commit()


def _figure_search_text(f, extra: dict) -> str:
    """把图片块组装为可检索文本：基础标记 + OCR 文字 + 表格结构 + VLM 描述。"""
    # docx 等无物理页码 → 页标记省略（page=None）；PDF 才显示「第 N 页」
    page_tag = f" · 第 {f.page} 页" if f.page else ""
    parts = [f"【图片{page_tag}】{f.caption or '（未识别图注）'}（图片文件：{f.filename}）"]
    ocr = (extra or {}).get("ocr_text", "")
    tbl = (extra or {}).get("table_md", "")
    vlm = (extra or {}).get("vlm_caption", "")
    if ocr:
        parts.append("图片文字(OCR)：" + ocr)
    if tbl:
        parts.append("表格结构：" + tbl)
    if vlm and vlm != (f.caption or ""):
        parts.append("内容描述：" + vlm)
    return "\n".join(parts)


def _diagram_search_text(d, extra: dict) -> str:
    """把结构图块组装为可检索文本：标签 + Mermaid 结构 + 一句话描述。"""
    parts = [f"【结构图 · 第 {d.page} 页】\n{d.text}"]
    mermaid = (extra or {}).get("mermaid", "")
    desc = (extra or {}).get("description", "")
    if mermaid:
        parts.append("结构(Mermaid)：\n" + mermaid)
    if desc:
        parts.append("结构描述：" + desc)
    return "\n".join(parts)


class DocumentService:
    def __init__(self, SessionLocal: sessionmaker, settings: Settings) -> None:
        self._Session = SessionLocal
        self._settings = settings
        self._parser = DocumentParser()
        self._chunker = Chunker()
        # T-05（2026-09-26）：`_pipelines` 改为**有序字典**并设容量上限（LRU 淘汰）。
        # 用 OrderedDict 是因为淘汰需要"最近使用"顺序：
        # `move_to_end(doc_id)` 标记最近使用、`popitem(last=False)` 取最久未用。
        # 上限见 `_max_pipelines`（env `PAPERMATE_MAX_PIPELINES`，默认 32）。
        self._pipelines: OrderedDict[int, RAGPipeline] = OrderedDict()
        self._max_pipelines = _resolve_max_pipelines()
        self._locks: dict[int, asyncio.Lock] = {}
        # G1（2026-09-10）：文档删除代际——delete 时递增，用于识别「构建期间被删除」
        # 的过期构建，避免写入孤儿管线缓存（代际守卫模式，与 J2/L1/N2 同款）。
        self._pipeline_gen: dict[int, int] = {}
        # 嵌入器 / 分词器单例缓存：按 RAG 配置签名复用，避免 real 模式 BGE-M3(~2.2GB) 在
        # 全量重建索引时被逐篇重复加载；同一时刻只保留当前配置的一个实例（切换回 fake 即释放显存/内存）。
        self._embedder = None
        self._embedder_sig: tuple | None = None
        self._embedder_lock = asyncio.Lock()
        # 加载失败熔断：BGE-M3 权重缺失/下载失败时，若不记住失败，全量重建会对每篇文档
        # 重试一次完整下载（N 篇 = N 次 ~2.2GB 拉取 + N 条同样错误）。记住后同签名直接快速失败。
        self._embedder_err: str | None = None
        self._embedder_err_sig: tuple | None = None
        self._tokenizer = None
        self._tokenizer_sig: tuple | None = None
        self._tokenizer_lock = asyncio.Lock()
        # 重排器单例缓存：real 模式下 CrossEncoderReranker 会加载 bge-reranker-v2-m3（~2.2GB），
        # 原先逐篇重建 = 重复加载 N 次；缓存后只加载一次。加载失败则缓存「不可用」哨兵，
        # 后续文档直接走 ScoreReranker，不再每篇重试。
        self._reranker = None
        self._reranker_sig: tuple | None = None
        self._reranker_lock = asyncio.Lock()
        # 上传异步化：每篇上传的取消事件 + 进度追踪 + 后台任务句柄
        # （前端轮询进度可视化、随时取消；与重建索引的 _reindex_cancel 同级范式）
        self._upload_cancel: dict[int, threading.Event] = {}
        self._upload_progress: dict[int, dict] = {}
        self._upload_tasks: dict[int, asyncio.Task] = {}
        # 全量重建索引的取消标志（协作式：在**文档边界**生效）。
        # 单篇 get_pipeline 是同步重型调用（DB 查询 + 嵌入推理），一旦开始无法中途打断，
        # 因此取消只能保证「当前这篇跑完后不再处理下一篇」，而非立即中断。
        self._reindex_cancel = asyncio.Event()

    # ---------- 嵌入器 ----------
    def _rag_rt(self) -> dict:
        """读取 RAG 底座运行期配置：优先 runtime_settings（前端可切换、持久化），
        回退 config.py 的 env 值。避免在模块加载期引入 runtime_settings，惰性导入防环。"""
        try:
            from ..runtime_settings import get_state

            st = get_state()
        except Exception:  # noqa: BLE001  runtime 不可用时退回 env 配置
            st = {}
        return {
            "rag_embedder": st.get("rag_embedder", self._settings.default_embedder),
            "use_bge_sparse": st.get("use_bge_sparse", self._settings.use_bge_sparse),
            "use_jieba": st.get("use_jieba", self._settings.use_jieba),
            # T-03 重排加速档位：runtime 不可用时回落 "关/不限长" = 改动前行为（fail-safe 方向）
            "rag_reranker_fp16": st.get("rag_reranker_fp16", False),
            "rag_reranker_max_length": st.get("rag_reranker_max_length", 0),
        }

    def _make_embedder(self):
        if self._rag_rt()["rag_embedder"] == "real":
            from lightcore.rag.embedder import BgeM3Embedder

            return BgeM3Embedder()
        from lightcore.rag.embedder import FakeHashEmbedder

        return FakeHashEmbedder()

    def _make_tokenizer(self):
        """BM25 分词器：开启 jieba 且已安装时按词切分，否则字符级兜底（零依赖）。"""
        from lightcore.rag.bm25 import _char_tokenize, jieba_tokenize

        return jieba_tokenize if self._rag_rt()["use_jieba"] else _char_tokenize

    async def _get_embedder(self):
        """缓存 embedder 单例（按 RAG 配置签名）。

        real 模式下 BGE-M3 权重 ~2.2GB，全量重建索引若逐篇新建会重复加载 N 次 →
        缓存后只在配置切换时重建一次，且只保留当前配置的一个实例（切换回 fake 即释放显存/内存）。

        失败熔断：加载抛错时记住 (签名, 错误)，同签名后续调用**直接快速失败**，
        避免 N 篇文档各重试一次完整下载。`invalidate_rag_components()` 或新一轮
        reindex 会清除该标记，给予重试机会（例如用户补齐权重后）。
        """
        key = (_embedder_identity(self._rag_rt()),)
        if self._embedder is not None and self._embedder_sig == key:
            return self._embedder
        if self._embedder_err is not None and self._embedder_err_sig == key:
            raise RuntimeError(f"嵌入器加载失败（已熔断，避免逐篇重试）：{self._embedder_err}")
        async with self._embedder_lock:
            if self._embedder is not None and self._embedder_sig == key:
                return self._embedder
            if self._embedder_err is not None and self._embedder_err_sig == key:
                raise RuntimeError(f"嵌入器加载失败（已熔断，避免逐篇重试）：{self._embedder_err}")
            try:
                # T-05（2026-09-26）：**必须走 to_thread**。实测冷构造 BGE-M3（GPU）需
                # **12.7s**（首次）/ 1.6s（二次），而它原先直接跑在 async 函数里 →
                # 加载期间**整个事件循环冻结**，预览/列表/健康检查全部排队。
                # 探针：`scripts/probe_cold_start.py`（六阶段分解）。
                emb = await asyncio.to_thread(self._make_embedder)
            except Exception as e:  # noqa: BLE001  记住失败原因，后续文档快速失败
                self._embedder_err = str(e)
                self._embedder_err_sig = key
                # 必须带 exc_info：熔断后同签名请求只会拿到这一句字符串，
                # 不在此处留全栈，真因（如 __cause__ 里的底层 ImportError）就永久丢失
                # —— 2026-09-18 的线上事故正是这样丢掉根因的（日志只剩一句误导性文案）。
                logger.error("嵌入器加载失败，已熔断后续同配置请求：%s", e, exc_info=True)
                raise
            self._embedder = emb
            self._embedder_sig = key
            self._embedder_err = None
            self._embedder_err_sig = None
            return emb

    async def _get_tokenizer(self):
        """缓存分词器单例（按 jieba 开关签名），与 embedder 同生命周期策略。"""
        key = (self._rag_rt()["use_jieba"],)
        if self._tokenizer is not None and self._tokenizer_sig == key:
            return self._tokenizer
        async with self._tokenizer_lock:
            if self._tokenizer is not None and self._tokenizer_sig == key:
                return self._tokenizer
            tok = self._make_tokenizer()
            self._tokenizer = tok
            self._tokenizer_sig = key
            return tok

    async def warm_up(self) -> dict:
        """预热重型 RAG 组件（T-05，2026-09-26）。返回各阶段耗时（秒），供日志/探针使用。

        ## 为什么需要它（实测，`scripts/probe_cold_start.py`）
        进程内冷启动总计 **25.4s**，其中：
        - **`BgeM3Embedder()` 12.7~18.3s**（71.8%）+ **`CrossEncoderReranker()` 4.6~5.1s**（19.9%）
        - CPU 部分（读 faiss + pickle + BM25 重建）仅 **1.51s** ← 前置分析曾把它当成主因（**归因错误**）

        预热把这 17~23s 从「**用户第一个检索请求**」挪到「**服务启动后**」。
        次要构造实测仍有 3.4s（权重载入本身），因此预热**必须配合单例缓存**
        （`_get_embedder` / `_get_reranker` 已有），否则用户请求还是要付这 3.4s。

        ## 边界
        - **幂等**：命中单例即立即返回（重复调用零成本）；
        - **fake 模式直接返回**：`FakeHashEmbedder` 构造是微秒级，无需预热
          —— 这也让绝大多数测试（用 fake）不受影响；
        - **失败不抛出**：预热失败只记日志，首个真实请求会重试（与既有熔断逻辑一致）。
        """
        if self._rag_rt()["rag_embedder"] != "real":
            return {"skipped": True, "reason": "fake embedder 无需预热"}

        ## T-05b（2026-09-27 复查修订）：上面两步**只加载了权重**，不够。
        ## 实测首个真实请求仍多付 ≈2.1s，来源两部分：
        ##   ① jieba 词典首次加载 ≈0.6s（在首个请求里才发生，日志可查）
        ##   ② 首次检索 ≈1.5s：embedder/reranker 的 CUDA kernel 初始化 + faiss 首次搜索
        ## 实测取证（2026-09-27 00:24 重启后）：
        ##   第一次 retrieve 2148ms（top_k=6，无并发）vs 后续 641/670ms（top_k=10，无并发）
        ##   ⇒ top_k 更小反而更慢，多出的部分与数据量无关，是"首次使用"开销。
        ## 为什么 T-05 当时判"预热 pipeline 收益有限"：
        ##   依据是探针 `probe_cold_start.py`，而**探针只测了加载、从未跑过一次真实检索**
        ##   ⇒ 探针缺口导致的误判。本轮补上（见 `_warm_first_use`）。
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        await self._get_embedder()
        timings["embedder_s"] = round(time.perf_counter() - t0, 3)
        t1 = time.perf_counter()
        await self._get_reranker()
        timings["reranker_s"] = round(time.perf_counter() - t1, 3)
        # 首次使用开销（jieba + pipeline + 一次空检索）
        t2 = time.perf_counter()
        await self._warm_first_use(timings)
        timings["first_use_s"] = round(time.perf_counter() - t2, 3)
        timings["total_s"] = round(time.perf_counter() - t0, 3)
        return timings

    async def _warm_first_use(self, timings: dict[str, float]) -> None:
        """预热「首次使用」的冷开销：jieba 词典 + 最近 N 篇 pipeline + 一次空检索。

        **幂等**：命中单例/pipeline 缓存即几乎零成本，重复调用安全。
        **失败不抛**：预热失败只影响首个请求快慢，不影响可用性（首个请求会自行加载）。
        """
        # ① jieba 词典（≈0.6s）—— 只有 `use_jieba` 打开时才有这一步。
        # ⚠️ 用 `.get()`：预热不该因配置字典缺字段而抛（替身/老配置都可能没有这一项）。
        if self._rag_rt().get("use_jieba"):
            try:
                tok = await self._get_tokenizer()
                t = time.perf_counter()
                await asyncio.to_thread(tok, "预热分词")
                timings["tokenizer_s"] = round(time.perf_counter() - t, 3)
            except Exception:  # noqa: BLE001
                timings["tokenizer_s"] = -1.0
                logger.warning("预热分词器失败（首个请求会自行加载）", exc_info=True)

        n_docs = _resolve_prewarm_docs()
        if n_docs <= 0:
            timings["prewarmed_docs"] = 0
            return
        try:
            doc_ids = await asyncio.to_thread(self._recent_ready_doc_ids, n_docs)
        except Exception:  # noqa: BLE001
            timings["prewarmed_docs"] = 0
            logger.warning("预热：读取文档列表失败，跳过 pipeline 预热", exc_info=True)
            return
        timings["prewarmed_docs"] = len(doc_ids)
        for doc_id in doc_ids:
            try:
                t = time.perf_counter()
                pipe = await self.get_pipeline(doc_id)
                if pipe is None:
                    timings[f"doc_{doc_id}_s"] = 0.0
                    continue
                # ② 一次真实检索：触发 embedder 前向 + faiss 首次搜索 + reranker 前向
                await asyncio.to_thread(pipe.query, "预热", 1)
                timings[f"doc_{doc_id}_s"] = round(time.perf_counter() - t, 3)
            except Exception:  # noqa: BLE001
                timings[f"doc_{doc_id}_s"] = -1.0
                logger.warning("预热文档 %s 失败（首个请求会自行加载）", doc_id, exc_info=True)

    def _recent_ready_doc_ids(self, limit: int) -> list[int]:
        """最近 `limit` 篇 `ready` 文档的 id（同步，调用方负责 `to_thread`）。"""
        with self._Session() as db:
            rows = (
                db.query(Document.id)
                .filter(Document.status == "ready")
                .order_by(Document.id.desc())
                .limit(limit)
                .all()
            )
            return [r[0] for r in rows]

    async def _get_reranker(self):
        """缓存重排器单例（按 real/fake 签名），real 模式返回 CrossEncoderReranker。

        原先 real 模式下每个 RAGPipeline 都会 new 一个 CrossEncoderReranker（加载
        bge-reranker-v2-m3 ~2.2GB），全量重建 N 篇 = 加载 N 次。缓存后只加载一次。

        返回 ``None`` 表示「用默认 ScoreReranker」——包括 fake 模式，以及 real 模式下
        CrossEncoder 初始化失败（权重缺失等）的降级；降级结果同样缓存，避免逐篇重试。

        ⚠️ T-03（2026-09-24）：缓存签名**必须包含加速档位**。此前 key 只有 rag_embedder，
        若沿用旧 key，用户在设置页改 fp16 / max_length 后**单例不会重建 → 设置静默不生效**
        （"改了没反应"这类最难排查的缺陷）。签名带档位后，改设置即触发重建。
        """
        rt = self._rag_rt()
        key = (rt["rag_embedder"], rt["rag_reranker_fp16"], rt["rag_reranker_max_length"])
        if self._reranker is not None and self._reranker_sig == key:
            return None if self._reranker is _RERANK_UNAVAILABLE else self._reranker
        async with self._reranker_lock:
            if self._reranker is not None and self._reranker_sig == key:
                return None if self._reranker is _RERANK_UNAVAILABLE else self._reranker
            if rt["rag_embedder"] != "real":
                self._reranker = _RERANK_UNAVAILABLE  # fake：零开销 ScoreReranker
                self._reranker_sig = key
                return None
            try:
                from lightcore.rag.rerank import CrossEncoderReranker

                # T-05（2026-09-26）：与 `_get_embedder` 同理 —— 冷构造实测 **4.6s**，
                # 直接跑在 async 函数里会阻塞事件循环。走 to_thread。
                self._reranker = await asyncio.to_thread(
                    CrossEncoderReranker,
                    max_length=rt["rag_reranker_max_length"],
                    fp16=rt["rag_reranker_fp16"],
                )
                logger.info(
                    "CrossEncoderReranker 已加载（bge-reranker-v2-m3）· fp16=%s · max_length=%s"
                    "（T-03 加速档位）",
                    rt["rag_reranker_fp16"], rt["rag_reranker_max_length"] or "不限",
                )
            except Exception as e:  # noqa: BLE001  降级：重排退化为融合分数，不阻断索引
                logger.warning("CrossEncoderReranker 初始化失败，回退 ScoreReranker：%s", e)
                self._reranker = _RERANK_UNAVAILABLE
            self._reranker_sig = key
            return None if self._reranker is _RERANK_UNAVAILABLE else self._reranker

    def invalidate_rag_components(self) -> None:
        """切换 RAG 配置后主动释放 embedder / tokenizer 单例（避免旧 BGE-M3 常驻显存/内存）。

        注：reindex 时若配置已变，``_get_embedder`` 会因签名不匹配自动重建，无需手动调用；
        此方法供「仅改配置、暂不重建」的场景显式回收资源，同时清除加载失败熔断标记
        （使补齐权重后可再次尝试）。
        """
        self._embedder = None
        self._embedder_sig = None
        self._embedder_err = None
        self._embedder_err_sig = None
        self._tokenizer = None
        self._tokenizer_sig = None
        self._reranker = None
        self._reranker_sig = None

    def reset_reranker_pipelines(self) -> None:
        """T-03：重排加速档位变更后，让新档位对**已缓存的管线**立即生效。

        **为什么必须清 `_pipelines`**：``get_pipeline`` 命中缓存时**不校验签名**
        （重排器只在"首次构建管线"时才取用），所以只重置 `_reranker` 单例是不够的 ——
        已建好的管线仍持有旧档位的重排器实例 → 表现为"设置页改了、实际一直没变"，
        属于最难排查的**静默失效**。改设置即清缓存，才能兑现"改完即时生效、无需重启"。

        安全性：只清本字典的引用，在途请求仍持有自己的 pipeline 对象；
        下次查询按需重建（索引缓存命中时无需重新编码）。
        """
        self._pipelines.clear()
        self._reranker = None
        self._reranker_sig = None

    # ---------- 上传 + 处理（状态机）----------
    async def upload_and_process(
        self,
        filename: str,
        content: bytes,
        session_id: int | None = None,
        *,
        block: bool = True,
    ) -> DocumentOut:
        """上传并（同步或异步）处理一篇文档。

        - ``block=True``（默认）：保存文件后**同步**在当前请求内跑完解析→分块→向量化，
          返回终态 DocumentOut（status=ready/failed），解析失败等异常原样抛出（路由转 400）。
          保留旧调用方（CLI / 测试 ``POST /documents``）语义，避免破坏既有的「上传后立即可检索」断言。
        - ``block=False``：保存文件后立即返回 ``status=uploaded`` 的 DocumentOut，真正的重活交给
          后台任务 ``_process_phase2``；前端轮询 ``GET /documents/{id}/progress`` 看进度、
          可 ``POST /documents/{id}/cancel`` 取消（异步上传，见 2026-09-02）。
        """
        filename = _safe_filename(filename)  # 防路径穿越（见 _safe_filename）
        digest = _content_hash(content)
        # 1) 落库拿 id
        def _insert_doc() -> int:
            with self._Session() as db:
                doc = Document(filename=filename, storage_path="", status="uploaded", content_hash=digest)
                db.add(doc)
                db.commit()
                db.refresh(doc)
                return doc.id

        doc_id = write_with_retry(_insert_doc)

        try:
            # 2) 保存原始文件
            folder = self._settings.upload_dir / str(doc_id)
            folder.mkdir(parents=True, exist_ok=True)
            fpath = folder / filename
            fpath.write_bytes(content)

            def _mark_saved() -> None:
                with self._Session() as db:
                    d = db.get(Document, doc_id)
                    assert d is not None
                    d.storage_path = str(fpath)
                    d.status = "uploaded"
                    db.commit()

            write_with_retry(_mark_saved)
        except Exception:
            # 落库成功但文件保存失败 → 回滚空记录，避免留下无文件的僵尸文档
            self._cleanup_upload_record(doc_id)
            raise

        if block:
            # 同步模式：直接在当前请求内跑完处理（与旧行为一致），异常原样抛出 → 路由转 400。
            # 不派发后台任务、不初始化进度条目，保持调用方零改动。
            out = await self._process_phase2(doc_id, filename, session_id, reraise=True)
            return out if out is not None else DocumentOut(
                id=doc_id, filename=filename, status="failed", meta={}, num_chunks=0
            )

        # 异步模式：初始化上传状态（取消事件 + 进度追踪），派发后台任务后立即返回。
        self._upload_cancel[doc_id] = threading.Event()
        self._upload_progress[doc_id] = {
            "status": "uploaded",
            "stage": "queued",
            "done": 0,
            "total": 0,
            "pct": 0,
            "error": None,
            "updated_at": time.time(),
        }
        task = asyncio.create_task(self._process_phase2(doc_id, filename, session_id, reraise=False))
        self._upload_tasks[doc_id] = task
        task.add_done_callback(lambda _t: self._upload_tasks.pop(doc_id, None))
        # 异步模式：立即返回，前端负责轮询进度 / 取消
        return DocumentOut(id=doc_id, filename=filename, status="uploaded", meta={}, num_chunks=0)

    async def _process_phase2(
        self,
        doc_id: int,
        filename: str,
        session_id: int | None = None,
        *,
        reraise: bool = False,
    ) -> DocumentOut | None:
        """后台异步处理一篇已落库（status=uploaded）的文档。

        流程与原 ``upload_and_process`` 同步主流程一致（解析 → 分块入库 → 向量化索引，
        含 VLM 图片/结构图增强，保持 async 不变），增加三处协作式增强：

        * 各阶段切换写 ``self._upload_progress[doc_id]``（前端轮询可视化）；
        * 解析/分块后检查 ``self._upload_cancel[doc_id]``；向量化阶段经 ``get_pipeline`` 的
          ``cancel_event``/``on_progress`` 钩子实现「分批可取消 + 进度回调」；
        * 失败标记 ``failed``；被取消则 ``_cleanup_upload_record`` 清理半成品（删记录 + 落盘）。

        返回终态 ``DocumentOut``；被取消或异常时返回 ``None``（清理已自行完成）。
        """
        cancel = self._upload_cancel.get(doc_id)

        try:
            # 1) 解析
            self._set_upload_progress(doc_id, status="parsing", stage="parsing")

            def _mark_parsing() -> None:
                with self._Session() as db:
                    d = db.get(Document, doc_id)
                    assert d is not None
                    d.status = "parsing"
                    db.commit()

            write_with_retry(_mark_parsing)
            if cancel is not None and cancel.is_set():
                raise IndexingCancelled("解析前已取消")

            # 2) 解析（与原流程一致；文本类从落盘文件读取，避免额外持有大字节）
            tables: list = []
            figures: list = []
            diagrams: list = []
            extract_warnings: list[str] = []
            figure_extras: list[dict] = []
            diagram_extras: list[dict] = []
            img_dir = self._settings.upload_dir / str(doc_id) / "images"
            fpath = self._settings.upload_dir / str(doc_id) / filename
            if filename.lower().endswith(".pdf"):
                blocks = self._parser.parse_pdf(fpath)
                # P1-3（ER-03）：表格结构化提取（文本型表格，零新依赖 find_tables）
                try:
                    from lightcore.rag.table_extractor import TableExtractor, md_table_lint

                    tables = TableExtractor().extract(fpath)
                    for _t in tables:
                        for _issue in md_table_lint(_t.markdown):
                            extract_warnings.append(f"表格校验（第 {_t.page} 页）：{_issue}")
                except Exception:  # noqa: BLE001  表格提取失败不阻断文档处理
                    tables = []
                # 图片/图表提取（ER-03 图表部分）
                try:
                    from lightcore.rag.figure_extractor import FigureExtractor

                    figures, img_warnings = FigureExtractor().extract(fpath, image_dir=img_dir)
                    extract_warnings.extend(img_warnings)
                except Exception:  # noqa: BLE001  图片提取失败不阻断文档处理
                    figures = []
                # 矢量结构图提取（ER-03 扩展）
                try:
                    from lightcore.rag.diagram_extractor import DiagramExtractor

                    diagrams = DiagramExtractor().extract(fpath, image_dir=img_dir)
                    diagrams = [
                        d for d in diagrams
                        if not any(_bbox_overlap_ratio(d.bbox, f.bbox) > 0.5 for f in figures)
                    ]
                except Exception:  # noqa: BLE001  结构图提取失败不阻断文档处理
                    diagrams = []
            elif filename.lower().endswith((".docx", ".doc")):
                blocks = self._parser.parse_docx(fpath)
                try:
                    from lightcore.rag.docx_image_extractor import DocxImageExtractor

                    figures, docx_img_warnings = DocxImageExtractor().extract(fpath, image_dir=img_dir)
                    extract_warnings.extend(docx_img_warnings)
                except Exception:  # noqa: BLE001  图片提取失败不阻断文档处理
                    figures = []
            else:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                blocks = self._parser.parse_text(text)
            # ER-03 Level 2：图片内容增强（OCR / 表格结构还原 / VLM 图注）
            if figures:
                try:
                    from types import SimpleNamespace

                    from lightcore.rag.figure_content import FigureEnricher

                    enrich_cfg = (
                        runtime_settings.get_state().get("figure_enrich")
                        or getattr(self._settings, "figure_enrich", None)
                        or {}
                    )
                    # T-06 P1：注入内容哈希缓存 ⇒ 改稿重建时未变图 0 次 OCR/VLM 调用
                    enricher = FigureEnricher(
                        SimpleNamespace(figure_enrich=enrich_cfg),
                        cache=_FigureEnrichCache(self._Session),
                    )
                    for f in figures:
                        extra = await enricher.enrich(img_dir / f.filename, kind=f.kind, page=f.page)
                        figure_extras.append(extra)
                        if extra.get("vlm_caption"):
                            f.caption = extra["vlm_caption"]
                except Exception:  # noqa: BLE001
                    figure_extras = [{} for _ in figures]
            # P1：结构图多模态识别（Mermaid + 一句话描述）
            if diagrams:
                try:
                    from lightcore.rag.figure_content import DiagramEnricher

                    _feat = runtime_settings.get_state().get("features", {})
                    if _feat.get("diagram_enrich", False):
                        enricher = DiagramEnricher()
                        for dg in diagrams:
                            diagram_extras.append(await enricher.enrich(img_dir / dg.filename))
                    else:
                        diagram_extras = [{} for _ in diagrams]
                except Exception:  # noqa: BLE001  结构识别失败不阻断文档处理
                    diagram_extras = [{} for _ in diagrams]
            parsed_text = "\n\n".join(b.text for b in blocks)
            if tables:
                parsed_text += "\n\n" + "\n\n".join(
                    f"【表格 · 第 {t.page} 页】\n{t.markdown}" for t in tables
                )
            if figures:
                parsed_text += "\n\n" + "\n\n".join(
                    _figure_search_text(f, figure_extras[i] if i < len(figure_extras) else {})
                    for i, f in enumerate(figures)
                )
            if diagrams:
                parsed_text += "\n\n" + "\n\n".join(
                    _diagram_search_text(d, diagram_extras[i] if i < len(diagram_extras) else {})
                    for i, d in enumerate(diagrams)
                )

            if cancel is not None and cancel.is_set():
                raise IndexingCancelled("解析后已取消")
            # 3) 分块入库
            self._set_upload_progress(doc_id, status="chunking", stage="chunking")

            def _mark_chunking() -> None:
                with self._Session() as db:
                    d = db.get(Document, doc_id)
                    assert d is not None
                    d.status = "chunking"
                    db.commit()

            write_with_retry(_mark_chunking)

            chunks = self._chunker.chunk_blocks(blocks, doc_id=f"doc-{doc_id}")

            def _store_chunks() -> None:
                with self._Session() as db:
                    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
                    for seq, c in enumerate(chunks):
                        db.add(Chunk(
                            document_id=doc_id, seq=seq, content=c.text,
                            token_count=c.token_count, page=c.page,
                        ))
                    for ti, t in enumerate(tables):
                        db.add(Chunk(
                            document_id=doc_id, seq=len(chunks) + ti, content=t.markdown,
                            token_count=max(1, len(t.text) // 4), page=t.page,
                            meta_json=json.dumps({
                                "chunk_type": "table", "bbox": list(t.bbox),
                                "rows": t.rows, "cols": t.cols, "header": t.header,
                            }, ensure_ascii=False),
                        ))
                    for fi, f in enumerate(figures):
                        extra = figure_extras[fi] if fi < len(figure_extras) else {}
                        fig_text = _figure_search_text(f, extra)
                        db.add(Chunk(
                            document_id=doc_id, seq=len(chunks) + len(tables) + fi,
                            content=fig_text, token_count=max(1, len(fig_text) // 4), page=f.page,
                            meta_json=json.dumps({
                                "chunk_type": f.kind, "bbox": list(f.bbox), "caption": f.caption,
                                "file": f.filename, "xref": f.xref,
                                "ocr_text": extra.get("ocr_text", ""), "table_md": extra.get("table_md", ""),
                            }, ensure_ascii=False),
                        ))
                    for di, d in enumerate(diagrams):
                        dextra = diagram_extras[di] if di < len(diagram_extras) else {}
                        d_text = _diagram_search_text(d, dextra)
                        db.add(Chunk(
                            document_id=doc_id, seq=len(chunks) + len(tables) + len(figures) + di,
                            content=d_text, token_count=max(1, len(d_text) // 4), page=d.page,
                            meta_json=json.dumps({
                                "chunk_type": "diagram", "bbox": list(d.bbox), "labels": d.labels,
                                "file": d.filename, "mermaid": dextra.get("mermaid", ""),
                                "description": dextra.get("description", ""),
                            }, ensure_ascii=False),
                        ))
                    d = db.get(Document, doc_id)
                    assert d is not None
                    d.status = "embedding"
                    db.commit()

            write_with_retry(_store_chunks)

            (self._settings.parsed_dir / f"{doc_id}.txt").write_text(parsed_text, encoding="utf-8")

            # 4) 向量化 + 索引（构建并缓存管线）；分批取消 / 进度回调
            self._set_upload_progress(doc_id, status="embedding", stage="embedding", total=len(chunks))

            def _on_progress(done: int, total: int) -> None:
                self._set_upload_progress(doc_id, status="embedding", stage="embedding", done=done, total=total)

            await self.get_pipeline(doc_id, cancel_event=cancel, on_progress=_on_progress)

            # 5) 收尾：标记 ready + 关联会话
            def _mark_ready() -> DocumentOut:
                with self._Session() as db:
                    d = db.get(Document, doc_id)
                    assert d is not None
                    d.status = "ready"
                    d.meta_json = json.dumps(
                        {
                            "num_blocks": len(blocks),
                            "num_chunks": len(chunks),
                            "num_tables": len(tables),
                            "num_images": len(figures),
                            "num_diagrams": len(diagrams),
                            "extraction_warnings": extract_warnings,
                        },
                        ensure_ascii=False,
                    )
                    db.commit()
                    return DocumentOut(
                        id=d.id, filename=d.filename, status=d.status,
                        meta=json.loads(d.meta_json), num_chunks=len(chunks),
                    )

            out = write_with_retry(_mark_ready)
            self._set_upload_progress(
                doc_id, status="ready", stage="ready", done=len(chunks), total=len(chunks)
            )
            # 2026-09-06 资源回收：任务终态后释放取消 Event 与进度条目——此前仅在
            # 失败/取消/删除路径清理，成功路径的条目永久滞留（长生命周期进程内存缓慢增长）。
            # 进度快照留最后一帧（status=ready），供前端轮询在收到 200 响应前仍可读到终态。
            self._upload_cancel.pop(doc_id, None)
            if session_id is not None:
                self._attach_to_session(session_id, doc_id)
            return out
        except IndexingCancelled:
            # 协作式取消：清理半成品（删记录 + 落盘 + 状态条目），前端轮询将收到 404 / cancelled
            logger.info("上传 %s 被取消，清理半成品", doc_id)
            self._cleanup_upload_record(doc_id)
            self._set_upload_progress(doc_id, status="cancelled", stage="cancelled")
            return None
        except Exception as e:  # noqa: BLE001
            err_msg = str(e)
            logger.error("上传 %s 处理失败：%s", doc_id, e, exc_info=True)
            try:
                def _mark_failed() -> None:
                    with self._Session() as db:
                        d = db.get(Document, doc_id)
                        if d is not None:
                            d.status = "failed"
                            d.meta_json = json.dumps({"error": err_msg}, ensure_ascii=False)
                            db.commit()

                write_with_retry(_mark_failed)
            except Exception:  # noqa: BLE001  失败标记也写不进时，保留原异常
                pass
            self._pipelines.pop(doc_id, None)
            self._set_upload_progress(doc_id, status="failed", stage="failed", error=err_msg)
            # 同步模式（block=True）：把解析失败等异常原样抛出，路由据此转 400（保持旧语义）；
            # 异步模式：吞掉并返回 None，由进度轮询把状态置为 failed，避免「未捕获任务异常」告警。
            if reraise:
                raise
            return None

    # ---------- 上传进度 / 取消（异步模式）----------
    @staticmethod
    def _upload_pct(done: int, total: int) -> int:
        if not total:
            return 0
        return max(0, min(100, int(round(100 * done / total))))

    def _set_upload_progress(
        self,
        doc_id: int,
        *,
        status: str | None = None,
        stage: str | None = None,
        done: int | None = None,
        total: int | None = None,
        error: str | None = None,
    ) -> None:
        """更新（或初始化）一篇上传的进度快照；每次整体替换保证轮询端读到一致视图。"""
        cur = self._upload_progress.get(doc_id, {})
        nxt = {
            "status": status if status is not None else cur.get("status", "uploaded"),
            "stage": stage if stage is not None else cur.get("stage", "queued"),
            "done": done if done is not None else cur.get("done", 0),
            "total": total if total is not None else cur.get("total", 0),
            "error": error if error is not None else cur.get("error"),
            "updated_at": time.time(),
        }
        nxt["pct"] = self._upload_pct(nxt["done"], nxt["total"])
        self._upload_progress[doc_id] = nxt

    def get_upload_progress(self, doc_id: int) -> dict | None:
        """返回上传进度快照；无条目时回退到 DB 当前状态（已完成很久的文档仍可被轮询到）。

        文档已被取消清理（记录不存在）→ 返回 ``None``（路由转 404，前端据此停止轮询并刷新列表）。
        """
        prog = self._upload_progress.get(doc_id)
        if prog is not None:
            return {"doc_id": doc_id, **prog}
        d = self.get_document(doc_id)
        if d is None:
            return None
        pct = 100 if d.status == "ready" else 0
        return {
            "doc_id": doc_id,
            "status": d.status,
            "stage": d.status,
            "done": d.num_chunks,
            "total": d.num_chunks,
            "pct": pct,
            "error": None,
            "updated_at": time.time(),
        }

    def cancel_upload(self, doc_id: int) -> dict:
        """请求取消正在进行的异步上传（协作式，按阶段生效，与重建索引同范式）。

        返回 ``{"ok", "cancelled", "status"}``：``cancelled=True`` 表示本次调用触发了取消
        （置位取消事件，后台任务在阶段边界 / 下一批次检测到后清理半成品）；
        已处于终态（ready/failed/cancelled）时 ``cancelled=False``，幂等无副作用。
        """
        prog = self._upload_progress.get(doc_id)
        status = prog.get("status") if prog else None
        if status in ("ready", "failed", "cancelled"):
            return {"ok": True, "cancelled": False, "status": status}
        ev = self._upload_cancel.get(doc_id)
        if ev is not None:
            ev.set()
        return {"ok": True, "cancelled": True, "status": status or "uploaded"}

    def _cleanup_upload_record(self, doc_id: int) -> None:
        """取消 / 失败时清理半成品：删 DB 记录（级联块）+ 落盘文件 + 上传状态条目。

        G2 修复（2026-09-10）：**不**经 `delete()`——delete 会 pop 当前上传任务并
        `cancel()`，而该任务往往正是调用本方法的协程自身（自取消反模式）：任务以
        cancelled 语义结束，外部 `await task` 会收到 CancelledError 并掩盖真实失败原因。
        这里只清理数据与上传状态条目，**任务生命周期交给调用方**（谁启动谁收尾）。
        """
        try:
            self._drop_document_data(doc_id)
        except Exception:  # noqa: BLE001  清理尽力而为，不掩盖原始失败/取消
            logger.warning("清理文档 %s 半成品失败（留待手动清理）", doc_id, exc_info=True)
        self._upload_cancel.pop(doc_id, None)
        self._upload_progress.pop(doc_id, None)
        # 注意：**不** pop/cancel _upload_tasks —— 当前执行者可能就是该任务自身（G2）

    def _attach_to_session(self, session_id: int, doc_id: int) -> None:
        from ..models.db import ChatSession

        def _attach() -> None:
            with self._Session() as db:
                s = db.get(ChatSession, session_id)
                if s is None:
                    return
                cfg = json.loads(s.config_json or "{}")
                doc_ids = list(cfg.get("doc_ids", []))
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                cfg["doc_ids"] = doc_ids
                s.config_json = json.dumps(cfg, ensure_ascii=False)
                db.commit()

        write_with_retry(_attach)

    # ---------- 按文档的 RAGPipeline（缓存）----------
    def _cache_pipeline(self, doc_id: int, pipeline: RAGPipeline, gen: int) -> None:
        """写管线缓存前比对删除代际（G1 修复 2026-09-10，代际守卫模式第五用）。

        `delete()` 与管线构建可能并发：若构建完成后文档已被删除，直接把 pipeline
        写进 `_pipelines` 会留下**孤儿缓存**（常驻内存、无对应文档、永不回收）。
        `_drop_document_data` 递增 `_pipeline_gen[doc_id]`，此处比对构建开始时的代际，
        不一致即放弃写入（构建成果随对象释放回收）。

        注意：**不用** DB 查询校验文档存在——那会让本方法依赖完整 session API、
        并对「无 DB 行的内存管线」（测试/特殊路径）产生误伤；代际方案零额外依赖。
        """
        if self._pipeline_gen.get(doc_id, 0) != gen:
            logger.info("文档 %s 在构建期间被删除，放弃写入管线缓存（避免孤儿）", doc_id)
            return
        self._pipelines[doc_id] = pipeline
        self._pipelines.move_to_end(doc_id)   # T-05：记为最近使用
        self._evict_pipelines_if_needed()

    def _evict_pipelines_if_needed(self) -> None:
        """T-05（2026-09-26）：`_pipelines` 超出上限时淘汰**最久未用**的管线。

        ## 为什么需要
        原先 `_pipelines` 是**无上限**的：文档只有 `delete()` / `invalidate_rag_components()`
        才会被移出。长会话累积几十篇后，每篇的 faiss + 文本 + BM25 索引都会常驻内存，
        "常驻缓存"会变成事实上只增不减的内存泄漏。

        ## 为什么淘汰是安全的
        淘汰只**丢弃内存对象**，`data/faiss/doc-{id}.{faiss,state}` 仍在磁盘 →
        下次 `get_pipeline` 走 `load_state` 重建，实测 **≈100ms/篇**
        （faiss 26ms + pickle 1ms + BM25/jieba 73ms，见 `scripts/probe_cold_start.py`）。
        即"淘汰 = 用 100ms 换回内存"，代价可接受。

        ## 与既有机制的交互（改动此方法前必读）
        - **G1 删除代际**：淘汰与删除是两件事 —— 删除会 `pop` + 递增代际；
          淘汰只是让对象失去强引用，`_pipeline_gen[doc_id]` **保持不变**
          （文档仍在，只是缓存不在了）→ 不会误伤"构建期间被删除"的判定。
        - **正在被并发检索使用的管线**：调用方持有对象引用 → 淘汰只影响字典，
          对象本身在本次查询结束前不会被 GC，**不会出现"用到一半索引消失"**。
        - **`invalidate_rag_components()`**：清空整个字典（配置变更），与本淘汰互不干扰。
        """
        limit = self._max_pipelines
        while len(self._pipelines) > limit:
            doc_id, _ = self._pipelines.popitem(last=False)   # 最久未用
            logger.info("管线缓存超出上限 %d，淘汰最久未用的文档 %s（下次访问约 100ms 重建）",
                        limit, doc_id)

    async def get_pipeline(
        self,
        doc_id: int,
        *,
        cancel_event: threading.Event | None = None,
        on_progress=None,
    ) -> RAGPipeline | None:
        if doc_id in self._pipelines:
            self._pipelines.move_to_end(doc_id)   # T-05：命中即标记为最近使用（LRU）
            return self._pipelines[doc_id]
        lock = self._locks.setdefault(doc_id, asyncio.Lock())
        # 2026-09-11：**等锁耗时日志**——定位「新文档首次对话非常慢」。
        # 上传流水线正在编码同一 doc 时，对话会在此等待这把锁（此前无任何日志线索，
        # 用户实测出现过 52s 空档）。打点后可一眼区分「等编码」与「检索本身慢」。
        _waited = lock.locked()
        _t_wait = time.perf_counter()
        if _waited:
            logger.info("文档 %s 管线正在构建/加载中，本请求等待锁…", doc_id)
        async with lock:
            if _waited:
                logger.info("文档 %s 等待锁结束，耗时 %.2fs", doc_id, time.perf_counter() - _t_wait)
            if doc_id in self._pipelines:  # 双重检查
                self._pipelines.move_to_end(doc_id)   # T-05：同上（LRU）
                return self._pipelines[doc_id]
            gen = self._pipeline_gen.get(doc_id, 0)  # G1：记录构建开始时的删除代际
            # 优先从数据库 Chunk 表构建管线：块已正确记录 1 起算页码（PDF），
            # 这样 retrieve / pdf_extract 返回的页码与落库页码一致，修复「检索 page=0」。
            with self._Session() as db:
                from ..models.db import Chunk

                rows = (
                    db.query(Chunk)
                    .filter(Chunk.document_id == doc_id)
                    .order_by(Chunk.seq)
                    .all()
                )
            if rows:
                items = [(f"doc-{doc_id}-c{r.seq}", r.content, r.page) for r in rows]
                rt = self._rag_rt()
                # reranker 显式传入（单例缓存）；use_cross_encoder 必须为 False，
                # 否则 RAGPipeline 在 reranker=None 时会自行 new 一个（逐篇加载 2.2GB）。
                embedder = await self._get_embedder()
                reranker = await self._get_reranker()
                tokenize = await self._get_tokenizer()
                # 2026-09-07：索引状态落盘（faiss_dir 复活）——嵌入向量只存内存的话，
                # 每次重启后首次访问都会全量重编码（分钟级）。签名绑定分块内容+嵌入器配置，
                # 内容或配置变化自动失效；命中则免重编码直接恢复管线。
                faiss_path = self._settings.faiss_dir / f"doc-{doc_id}.faiss"
                state_path = self._settings.faiss_dir / f"doc-{doc_id}.state"
                sig = self._pipeline_signature(items, rt)
                pipeline = await asyncio.to_thread(
                    RAGPipeline.load_state, faiss_path=faiss_path, state_path=state_path,
                    signature=sig, embedder=embedder, reranker=reranker,
                    use_bge_sparse=rt["use_bge_sparse"], tokenize=tokenize,
                )
                if pipeline is not None:
                    logger.info("文档 %s 索引缓存命中，跳过重编码", doc_id)
                    self._cache_pipeline(doc_id, pipeline, gen)
                    return pipeline
                pipeline = RAGPipeline(
                    embedder=embedder,
                    reranker=reranker,
                    use_cross_encoder=False,
                    use_bge_sparse=rt["use_bge_sparse"],
                    tokenize=tokenize,
                )
                if cancel_event is not None or on_progress is not None:
                    # 异步上传路径：分批写入并支持中途取消 + 进度回调；
                    # 在 worker 线程跑嵌入编码，释放事件循环让进度轮询/取消端点可响应。
                    await asyncio.to_thread(
                        pipeline.add_chunks, items, f"doc-{doc_id}",
                        cancel_event=cancel_event, on_progress=on_progress,
                    )
                else:
                    # 2026-09-07 修复：无进度路径（工作间加文档/问答/技能首次建管线）同样必须
                    # 在 worker 线程跑——add_chunks 含 BGE-M3 编码（CPU 下分钟级）+ jieba 分词
                    # + FAISS 建索引，同步跑在事件循环上会冻结整个后端：期间所有 HTTP 请求
                    # （含预览的 /documents/{id}/file）排队超时，前端预览"全部加载失败"。
                    await asyncio.to_thread(pipeline.add_chunks, items, f"doc-{doc_id}")
                try:
                    await asyncio.to_thread(pipeline.save_state, faiss_path, state_path, sig)
                except Exception:  # noqa: BLE001  落盘失败仅损失下次重启的缓存
                    logger.warning("文档 %s 索引状态落盘失败", doc_id, exc_info=True)
                self._cache_pipeline(doc_id, pipeline, gen)
                return pipeline
            # 回退：无块（极旧文档）→ 用扁平文本构建（此时 page=None，无页码溯源）
            text = self._read_parsed_text(doc_id)
            if text is None:
                return None
            rt = self._rag_rt()
            # 与上面分支保持一致：走 embedder/tokenizer/reranker 单例缓存，
            # 否则 real 模式下无块的文档会各加载一次 ~2.2GB BGE-M3。
            pipeline = RAGPipeline(
                embedder=await self._get_embedder(),
                reranker=await self._get_reranker(),
                use_cross_encoder=False,
                use_bge_sparse=rt["use_bge_sparse"],
                tokenize=await self._get_tokenizer(),
            )
            if cancel_event is not None or on_progress is not None:
                await asyncio.to_thread(
                    pipeline.add_document, text, f"doc-{doc_id}",
                    cancel_event=cancel_event, on_progress=on_progress,
                )
            else:
                # 同上（2026-09-07）：回退分支同样移入 worker 线程
                await asyncio.to_thread(pipeline.add_document, text, doc_id=f"doc-{doc_id}")
            self._cache_pipeline(doc_id, pipeline, gen)
            return pipeline

    def _pipeline_signature(self, items: list[tuple[str, str, int | None]], rt: dict) -> str:
        """管线缓存签名（2026-09-07）：分块内容 + 嵌入器配置共同决定向量是否仍有效。

        任一分块的 id/text/page 变化，或 rag_embedder/sparse/jieba 配置切换 → 签名变 → 缓存失效。

        ★ 2026-10-04：模型名部分改用 `_embedder_identity(rt)`，把
        **`PAPERMATE_BGE_M3_PATH`（权重目录覆盖）**也纳入签名。
        此前只签 `rag_embedder` 名字 ⇒ 换权重后签名不变 ⇒ `load_state` 命中
        ⇒ **继续用旧向量、检索结果错且无任何报错**。
        """
        h = hashlib.sha256()
        h.update(f'{_embedder_identity(rt)}|{rt["use_bge_sparse"]}|{rt["use_jieba"]}'.encode())
        for i, t, p in items:
            h.update(f"{i}\x1f{t}\x1f{p}\x1e".encode())
        return h.hexdigest()[:32]

    def _read_parsed_text(self, doc_id: int) -> str | None:
        p = self._settings.parsed_dir / f"{doc_id}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
        # 回退：用 chunk 表重建文本
        with self._Session() as db:
            rows = db.query(Chunk).filter(Chunk.document_id == doc_id).order_by(Chunk.seq).all()
            if not rows:
                return None
            return "\n\n".join(r.content for r in rows)

    def get_parsed_text(self, doc_id: int) -> str | None:
        """公开：读取该文档的解析文本（parsed_dir 落盘 → chunk 表回退）。供文档处理/版本模块复用。"""
        return self._read_parsed_text(doc_id)

    def read_original(self, doc_id: int) -> bytes | None:
        """公开：读取该文档的原始文件字节（storage_path 缺失/文件丢失返回 None）。"""
        with self._Session() as db:
            d = db.get(Document, doc_id)
            if d is None or not d.storage_path:
                return None
            path = Path(d.storage_path)
        if not path.exists():
            return None
        try:
            return path.read_bytes()
        except OSError:  # noqa: BLE001  文件被占用/损坏时按缺失处理
            return None

    def storage_path(self, doc_id: int) -> Path | None:
        """公开：返回原始文件磁盘路径（C2 改稿直写用）；文档不存在/无路径/文件丢失 → None。"""
        with self._Session() as db:
            d = db.get(Document, doc_id)
            if d is None or not d.storage_path:
                return None
            path = Path(d.storage_path)
        return path if path.exists() else None

    # ---------- 重复上传检测（2026-08-20）----------
    def check_duplicate(self, filename: str, content: bytes) -> dict:
        """按内容 sha256 检测是否已有相同文档；另报告同名但内容不同的情况。

        返回 {"duplicate": bool, "same_name": bool, "existing": DocumentOut | None}
        """
        digest = _content_hash(content)
        with self._Session() as db:
            # 1) 内容完全相同（主判据）
            by_hash = (
                db.query(Document)
                .filter(Document.content_hash == digest)
                .order_by(Document.id.desc())
                .first()
            )
            if by_hash is not None:
                meta = json.loads(by_hash.meta_json or "{}")
                n = db.query(Chunk).filter(Chunk.document_id == by_hash.id).count()
                return {
                    "duplicate": True,
                    "same_name": by_hash.filename == filename,
                    "existing": DocumentOut(
                        id=by_hash.id, filename=by_hash.filename, status=by_hash.status,
                        meta=meta, num_chunks=n,
                    ),
                }
            # 2) 同名但内容不同（辅助信号）
            by_name = (
                db.query(Document)
                .filter(Document.filename == filename)
                .order_by(Document.id.desc())
                .first()
            )
            if by_name is not None:
                meta = json.loads(by_name.meta_json or "{}")
                n = db.query(Chunk).filter(Chunk.document_id == by_name.id).count()
                return {
                    "duplicate": False,
                    "same_name": True,
                    "existing": DocumentOut(
                        id=by_name.id, filename=by_name.filename, status=by_name.status,
                        meta=meta, num_chunks=n,
                    ),
                }
            return {"duplicate": False, "same_name": False, "existing": None}

    def backfill_hashes(self) -> int:
        """启动时回填旧文档的 content_hash（读落盘文件算 sha256），返回回填条数。"""
        filled = 0
        with self._Session() as db:
            rows = db.query(Document).filter(Document.content_hash == "").all()
            for d in rows:
                if not d.storage_path:
                    continue
                p = Path(d.storage_path)
                if not p.exists():
                    continue
                try:
                    d.content_hash = _content_hash(p.read_bytes())
                    filled += 1
                except Exception:  # noqa: BLE001  单条失败跳过
                    continue
            if filled:
                db.commit()
        return filled

    # ---------- 列表 / 详情 / 删除 / 重建 ----------
    def count_documents(self) -> int:
        """文档总数（C4 分页用）。与 `list_documents` 同源，保证 `X-Total-Count` 与实际列表一致。"""
        with self._Session() as db:
            return db.query(Document).count()

    def list_documents(self, limit: int | None = None, offset: int = 0) -> list[DocumentOut]:
        """文档列表。分块计数用一次 GROUP BY 聚合（消除 N+1 查询）。

        C4（2026-09-21）：新增 `limit`/`offset`。**默认 `limit=None` 表示全量**，
        保持既有调用方行为不变（分页是新增能力，不该悄悄改变老调用的语义）。
        """
        with self._Session() as db:
            q = db.query(Document).order_by(Document.id.desc())
            if limit is not None:
                q = q.offset(offset).limit(limit)
            rows = q.all()
            if not rows:
                return []
            counts: dict[int, int] = {}
            for chunk_doc_id, n in (
                db.query(Chunk.document_id, func.count(Chunk.id))
                .filter(Chunk.document_id.in_([d.id for d in rows]))
                .group_by(Chunk.document_id)
                .all()
            ):
                counts[chunk_doc_id] = n
            out = []
            for d in rows:
                meta = json.loads(d.meta_json or "{}")
                out.append(DocumentOut(
                    id=d.id, filename=d.filename, status=d.status, meta=meta,
                    num_chunks=counts.get(d.id, 0),
                    created_at=iso_utc(d.created_at),
                ))
            return out

    def get_document(self, doc_id: int) -> DocumentOut | None:
        with self._Session() as db:
            d = db.get(Document, doc_id)
            if d is None:
                return None
            meta = json.loads(d.meta_json or "{}")
            n = db.query(Chunk).filter(Chunk.document_id == d.id).count()
            return DocumentOut(
                id=d.id, filename=d.filename, status=d.status, meta=meta, num_chunks=n,
                created_at=iso_utc(d.created_at),
            )

    def delete(self, doc_id: int) -> None:
        """删除文档：清数据/缓存/落盘 + 取消在飞上传任务。

        G2（2026-09-10）：数据清理已抽到 `_drop_document_data`，与 `_cleanup_upload_record`
        共用——后者**不能**走本方法（会取消调用者自身）。任务取消只应发生在
        「外部主动删除」这一侧。
        """
        # 若有进行中的上传后台任务，先取消，避免其继续持有/写入即将被删的文件（Windows 文件锁）
        task = self._upload_tasks.pop(doc_id, None)
        if task is not None and not task.done():
            task.cancel()
        self._upload_cancel.pop(doc_id, None)
        self._upload_progress.pop(doc_id, None)
        self._drop_document_data(doc_id)

    def _drop_document_data(self, doc_id: int) -> None:
        """清理文档的数据/缓存/落盘产物（**不含**任务与上传状态条目）。

        由 `delete()`（外部删除）与 `_cleanup_upload_record()`（取消/失败回收）共用。
        G1 修复（2026-09-10）：**不** pop `_locks` ——
        原先 delete 把锁从 `_locks` 移除后，新协程的 `setdefault` 会**重建一把新锁**，
        与仍在飞行的旧构建协程持有的锁不是同一把 → 两个协程并发构建同一文档
        （重复 BGE-M3 编码，昂贵 + 孤儿管线）。锁对象极轻，且 doc_id 复用时复用
        同一把锁更安全，故保留。
        """
        self._pipelines.pop(doc_id, None)
        # G1：递增删除代际——使「构建期间被删除」的在飞构建放弃写入缓存（见 _cache_pipeline）
        self._pipeline_gen[doc_id] = self._pipeline_gen.get(doc_id, 0) + 1
        # 2026-09-07：清理落盘索引缓存（否则残留的 faiss/state 文件成为孤儿垃圾）
        for suffix in (".faiss", ".state"):
            try:
                (self._settings.faiss_dir / f"doc-{doc_id}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass

        def _delete_row() -> None:
            with self._Session() as db:
                d = db.get(Document, doc_id)
                if d is None:
                    return
                db.delete(d)  # 级联删 chunks
                db.commit()

        write_with_retry(_delete_row)
        # 清理落盘。此时 DB 行已删，清理失败若抛错会让路由 500 → 前端误报「删除失败」
        # （文档实际已删）。故容错处理：失败记 warning 留待人工/孤儿清理，不冒泡。
        # 用 shutil.rmtree 递归删除：uploads/<id>/ 下现含 images/ 子目录（页面图），
        # 旧实现的 iterdir+unlink 遇子目录在 Windows 上抛 PermissionError（删除假失败根因）。
        folder = self._settings.upload_dir / str(doc_id)
        if folder.exists():
            try:
                shutil.rmtree(folder)
            except OSError as e:  # noqa: BLE001  文件被占用等：留孤儿但不影响删除结果
                logger.warning("删除文档 %s 的上传目录失败（留待清理）：%s", doc_id, e)
        parsed = self._settings.parsed_dir / f"{doc_id}.txt"
        if parsed.exists():
            try:
                parsed.unlink()
            except OSError as e:  # noqa: BLE001
                logger.warning("删除文档 %s 的解析文本失败（留待清理）：%s", doc_id, e)

    async def reindex(self, doc_id: int) -> DocumentOut | None:
        """单篇重建：弹缓存后用当前配置重建管线。

        返回语义：文档不存在 → ``None``（路由转 404）；文档存在但无内容可索引
        （无块且无解析文本）→ 仍返回 DocumentOut。原实现用
        ``await get_pipeline(...) and get_document(...)`` 短路，会把「无内容」误报成
        「文档不存在」404，掩盖真实原因。
        """
        self._pipelines.pop(doc_id, None)
        out = self.get_document(doc_id)
        if out is None:
            return None
        await self.get_pipeline(doc_id)
        return out

    async def reprocess(self, doc_id: int) -> DocumentOut | None:
        """改稿后**完整重处理**：重新解析 → 分块入库 → 向量化索引。

        ★ 为什么 `reindex` 不够（2026-09-30 探针证实，`scripts/probe_edit_consistency.py`）：
        改稿（apply-edits）只改 **docx 源文件**；`reindex` 从 **DB chunks** 重建管线 ——
        chunks 未变 ⇒ signature 原样命中磁盘 state ⇒ **0.01s 秒回旧向量**，
        检索到的仍是**改稿前**内容且无任何报错（假生效）。

        与 `reindex` 的区别：本方法走 `_process_phase2` 全流程（会重写 chunks 与
        faiss/state），耗时 = 一次完整首建（纯文本 ~20s；图密集无 VLM ~187s；
        开 figure_enrich 时更久）⇒ **只适合后台 fire-and-forget**，进度可经
        `GET /documents/{id}/progress` 轮询。
        """
        out = self.get_document(doc_id)
        if out is None:
            return None
        self._pipelines.pop(doc_id, None)
        return await self._process_phase2(doc_id, out.filename, reraise=False)

    async def _iter_reindex_all(self):
        """生成器：逐篇重建并产出进度事件。供 `reindex_all`（汇总）与 `reindex_all_stream`（SSE）共用。

        单篇失败（如 real 模式下 BGE-M3 权重未就绪、缺块且无解析文本）**不影响其余文档**，
        失败项记入 ``failed`` 列表并保留真实错误信息。

        支持**协作式取消**：每篇开始前检查 ``_reindex_cancel``，已取消则停止循环并产出
        ``type=="cancelled"`` 汇总事件。注意单篇 ``get_pipeline`` 是同步重型调用
        （DB 查询 + 嵌入推理），已开始的那篇会跑完——取消的粒度是「文档边界」而非即时中断。
        """
        # 新一轮重建：清除嵌入器加载失败熔断，给予重试机会（如用户已补齐权重/修好镜像）
        self._embedder_err = None
        self._embedder_err_sig = None
        # 新一轮重建：重置取消标志（上一轮若被取消，不应影响这一轮）
        self._reindex_cancel.clear()
        try:
            docs = self.list_documents()
        except Exception as e:  # noqa: BLE001  列表失败 → 产出错误汇总，而非抛栈打断 SSE
            logger.error("reindex_all: 获取文档列表失败：%s", e)
            yield {"type": "done", "reindexed": 0, "failed": [], "total": 0, "error": str(e)}
            return
        total = len(docs)
        reindexed = 0
        processed = 0
        failed: list[dict] = []
        for d in docs:
            if self._reindex_cancel.is_set():
                logger.info("reindex_all: 收到取消请求，在文档 %s 前停止（已处理 %s/%s）", d.id, processed, total)
                yield {
                    "type": "cancelled",
                    "reindexed": reindexed,
                    "failed": failed,
                    "total": total,
                    "done": processed,
                }
                return
            self._pipelines.pop(d.id, None)  # 弹缓存，强制下次 get_pipeline 用新配置重建
            status = "ok"
            error: str | None = None
            try:
                if not await self.get_pipeline(d.id):
                    status = "skip"
                    error = "无块且无解析文本，跳过"
            except Exception as e:  # noqa: BLE001  单篇失败隔离，不让整请求 500
                err = str(e)
                logger.warning("reindex_all: doc %s 重建失败：%s", d.id, err)
                status = "error"
                error = err
                failed.append({"id": d.id, "error": err})
            if status == "ok":
                reindexed += 1
            # ok / skip / error 三者均计入进度：若只用 reindexed+len(failed)，
            # 被跳过的文档不计入 → 前端进度条永远到不了 total，看起来像卡死。
            processed += 1
            yield {
                "type": "progress",
                "doc_id": d.id,
                "status": status,  # ok | skip | error
                "error": error,
                "done": processed,
                "total": total,
            }
        yield {"type": "done", "reindexed": reindexed, "failed": failed, "total": total}

    async def reindex_all(self) -> dict:
        """全量重建：按当前 RAG 配置（embedding 模式 / 稀疏 / 分词）重建所有已索引文档的管线缓存。

        用途：切换 rag_embedder / use_bge_sparse / use_jieba 后必须调用，否则 get_pipeline
        返回的是旧配置缓存（缓存仅按 doc_id 维度，不感知配置变化）。
        单篇失败**不影响其余文档**，失败项记入 returned ``failed`` 列表并保留真实错误信息。
        返回 {"reindexed": int, "failed": [{id, error}], "total": int}（CLI / scripts 同步调用）。
        """
        final: dict = {"reindexed": 0, "failed": [], "total": 0}
        async for ev in self._iter_reindex_all():
            final = ev  # 末次事件为 type=="done" 的汇总
        return final

    def cancel_reindex(self) -> None:
        """请求取消正在进行的全量重建索引（协作式，在文档边界生效）。

        已在处理的那篇会跑完（同步嵌入推理无法中途打断），随后循环停止并产出
        ``type=="cancelled"`` 事件。若当前没有正在进行的重建，调用无副作用。
        """
        self._reindex_cancel.set()

    async def reindex_all_stream(self):
        """逐篇重建并产出 SSE 进度事件（前端实时可视化用）。详见 routes_documents.reindex_all_stream。

        每产出一条事件后 ``await asyncio.sleep(0)`` 主动让出事件循环：`get_pipeline` 内部是
        同步重型调用（DB 查询 + 嵌入推理，real 模式单篇可达数秒~数十秒），会阻塞事件循环；
        不让出的话 SSE 事件无法逐条 flush，前端进度条表现为「长时间不动 → 最后瞬间跳 100%」。
        """
        async for ev in self._iter_reindex_all():
            yield ev
            await asyncio.sleep(0)
