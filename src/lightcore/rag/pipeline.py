"""LightCore · RAG 流水线（docs/05 §RAG / 组装 + 引用溯源）

串联：解析 → 分块 → 嵌入 → (FAISS 向量 + BM25) → 混合检索(RRF k=60) → 重排 → 引用溯源。

★ 2026-10-04（O9）修正过时描述：此处原写「重排(**Top50**→5)」，而
`__init__` 的 `top_n_hybrid` 默认已是 **20**（N16，2026-09-12 由 50 降到 20，
见 `:100` 与 `:112-115` 的实测依据：Top5 与 top_n=50 完全重叠、重排耗时 -54%）。
⇒ 融合池 = 3 路 × 20 = 60，重排后取 `top_k`（默认 5）。改文档时以**代码为准**。
对外暴露 add_document / query，返回带 id、页码、分数的片段，供 retrieve 工具与 SSE 引用卡片使用。

默认 Embedder=FakeHashEmbedder（可插 BgeM3Embedder）；默认 Reranker=ScoreReranker（可插 CrossEncoderReranker）。
- use_cross_encoder=True 时，在真实 BGE-M3 部署下自动装配 CrossEncoderReranker（精度更高），
  rag extra 缺失时自动回退 ScoreReranker，fake 部署零开销。
- use_bge_sparse=True 时，启用 BGE-M3 稀疏向量作为与 BM25 并行的词法信号（默认关闭，需 eval 量化后启用）。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25Retriever, _char_tokenize
from .chunker import Chunker
from .embedder import Embedder, FakeHashEmbedder
from .hybrid import HybridRetriever
from .index import VectorIndex
from .parser import DocumentParser
from .rerank import Reranker, ScoreReranker
from .sparse_index import SparseIndex

logger = logging.getLogger("lightcore.rag.pipeline")

#: 检索阶段计时开关（默认**关**）。未开启时所有打点都在 `if not _TRACE` 短路之后 → 正常使用零开销。
#:
#: **为什么用环境变量而不是 `logger.debug`**：本项目已有日志降噪机制（2026-09-17 静音高频请求），
#: 而 `query()` 是全项目最高频路径之一。若挂在 DEBUG 级，"为排查别的问题而开 DEBUG"会突然
#: 获得 3~6 行/次的噪声；显式环境变量则让"要测量的人明确打开"，也避免"以为在测、其实没开"。
#:
#: **用法**：需在**导入本模块之前**设置（模块级常量，import 时求值）：
#: `PAPERMATE_RAG_TRACE=1`。脚本里写成 `os.environ["PAPERMATE_RAG_TRACE"] = "1"` 再 import 即可。
_TRACE = os.environ.get("PAPERMATE_RAG_TRACE", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class RetrievedChunk:
    """检索结果片段（**不可变对象**）。

    2026-09-15 加固：`RAGPipeline.query` 命中查询缓存时返回 `list(cached)` —— 这是**浅拷贝**，
    只复制列表容器，元素仍指向同一批 RetrievedChunk 对象。因此浅拷贝防得住
    「调用方 append/pop/sort 改坏缓存」，**防不住 `chunk.score = x` 这类字段级原地修改**：
    那会静默污染缓存，使其后所有命中都返回被篡改的结果（且极难排查）。

    改为 `frozen=True`，把这条原本只存在于注释里的隐式约定升级为**类型层面的硬保证** ——
    字段赋值直接抛 FrozenInstanceError，而不是悄悄改坏缓存。
    当前全库无字段赋值（调用方均为「读取字段 → 构造新的输出结构」），故零破坏性。
    """

    id: str
    text: str
    page: int | None  # None=无物理页码；PDF 为 1 起算
    score: float


@dataclass
class _IndexChunk:
    """管线内部回查块（供 get_chunks / pdf_extract / citation_lookup 复用）。"""

    id: str
    text: str
    page: int | None


class IndexingCancelled(Exception):
    """索引过程被取消（上传 / 重建中途 abort）。

    由调用方（上传后台任务 / 重建生成器）捕获并清理半成品——管线本身不负责回滚。
    """


class RetrievalUnavailable(RuntimeError):
    """检索服务不可用：**全部**检索管线均失败（2026-09-07 F7）。

    与「检索成功但没有命中任何段落」有本质区别：后者是正常的空结果，
    前者是系统故障。若把二者混为一谈，上层会把故障误报成
    「文档中没有相关内容」，诱导用户重复上传而非排查系统。

    由检索工具（retrieve）捕获，转成带 error 标记的 ToolResult，
    供路由层生成独立的「检索故障」提示（notice kind=retrieval_error）。
    """

    def __init__(self, message: str = "检索服务暂时不可用", *, failed: int = 0, total: int = 0) -> None:
        super().__init__(message)
        self.failed = failed
        self.total = total


class RAGPipeline:
    def __init__(self, embedder: Embedder | None = None, reranker: Reranker | None = None,
                 *, use_cross_encoder: bool = False, use_bge_sparse: bool = False,
                 tokenize: Callable[[str], list[str]] = _char_tokenize,
                 top_n_hybrid: int = 20, top_k: int = 5) -> None:
        self.embedder = embedder or FakeHashEmbedder()
        # 重排器优先级：显式传入 > 自动 CrossEncoder（仅 real embedder + 配置开启）> 零依赖 ScoreReranker
        if reranker is not None:
            self.reranker = reranker
        elif use_cross_encoder:
            self.reranker = self._build_cross_encoder_reranker()
        else:
            self.reranker = ScoreReranker()
        # BGE-M3 稀疏向量（lexical）增强：仅当 embedder 支持 encode_sparse 且开启时装配。
        # 与 BM25 同源词法信号，但同处 BGE-M3 向量空间，融合更一致（默认关闭，待 eval 量化后启用）。
        self.use_bge_sparse = bool(use_bge_sparse) and hasattr(self.embedder, "encode_sparse")
        # N16（2026-09-12）：top_n_hybrid 默认 50 → 20。
        # 实测（doc 9，146 chunks，CrossEncoder）：重排占单次检索耗时 **97%**（1444/1491ms），
        # 候选数与耗时近似线性；top_n=20 时 Top5 与 top_n=50 **完全重叠（5/5）**、重排 670ms（-54%）。
        # top_n=10 重叠也是 5/5（338ms），保守取 20（融合池 3×20=60 仍有充足冗余）。
        self.top_n = top_n_hybrid
        self.top_k = top_k
        # N16（2026-09-12）：query 结果缓存（FIFO，容量 32）。
        # Agent 多轮对话常重复同一查询（如改稿场景反复检索同一段落相关内容）；
        # 命中时跳过 编码+三路检索+重排（约 1.5s/次）。索引变更（add_chunks）时整体失效，
        # 管线重建（load_state/重编码）产生新实例，缓存自然隔离。
        self._query_cache: dict[tuple[str, int], list[RetrievedChunk]] = {}
        self._query_cache_order: list[tuple[str, int]] = []
        self._parser = DocumentParser()
        self._chunker = Chunker()
        self._chunks: list = []  # list[Chunk]
        self._vidx = VectorIndex(self.embedder.dim)
        self._bm25 = BM25Retriever(tokenize=tokenize)
        self._sidx = SparseIndex() if self.use_bge_sparse else None
        self._hybrid = HybridRetriever(
            vector_search=self._dense_search,
            bm25_search=self._bm25_search,
            sparse_search=self._sparse_search if self._sidx is not None else None,
        )
        #: 最近一次 `query()` 的**阶段耗时**（毫秒），仅 `PAPERMATE_RAG_TRACE=1` 时填充。
        #:
        #: 为什么要有它：项目内检索延迟存在 **100× 跨度**（评测报告中位 10541ms vs 本机补测 79ms），
        #: 而既有观测只有一个"总数"—— 无法判断时间花在查询编码、三路召回还是重排上。
        #: T-00「查询级分阶段剖析」的数据来源就是它。
        #:
        #: ⚠️ 它**不参与** `save_state()` 的 bundle（那里是**显式白名单**：signature/texts/metas/
        #: chunks/sparse），故新增此属性**不影响索引缓存兼容性**；也**不要**把它写进 bundle ——
        #: 那是「索引状态」，这是「运行期观测」，混进去会让缓存内容随一次测量而变。
        self.last_stage_ms: dict[str, float] = {}

    # ------------------------------------------------------------ 三路召回（含阶段计时）

    def _acc(self, key: str, dt_s: float) -> None:
        """累加一个阶段的耗时（毫秒）。同一次 query 内同名阶段可能被调用多次（如编码）。"""
        self.last_stage_ms[key] = self.last_stage_ms.get(key, 0.0) + dt_s * 1000.0

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """稠密召回 = **查询编码** + ANN 搜索。两者分开计时 ——
        它们是可独立优化的（前者可缓存/换 ONNX，后者取决于索引实现），合并成一个数就无法判断该优化谁。
        """
        if not _TRACE:
            return self._vidx.search(self.embedder.encode([query])[0], k)
        t0 = time.perf_counter()
        vec = self.embedder.encode([query])[0]
        t1 = time.perf_counter()
        out = self._vidx.search(vec, k)
        t2 = time.perf_counter()
        self._acc("encode_dense", t1 - t0)
        self._acc("ann_search", t2 - t1)
        return out

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """BGE-M3 稀疏（lexical）召回；与稠密路径同理，编码与检索分开计时。

        注：仅在 `use_bge_sparse` 生效时才会被装配（见 `__init__` 的 `self._sidx is not None`）。
        """
        if not _TRACE:
            return self._sidx.search(self.embedder.encode_sparse([query])[0], k)
        t0 = time.perf_counter()
        vec = self.embedder.encode_sparse([query])[0]
        t1 = time.perf_counter()
        out = self._sidx.search(vec, k)
        t2 = time.perf_counter()
        self._acc("encode_sparse", t1 - t0)
        self._acc("sparse_search", t2 - t1)
        return out

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """BM25 召回（含 jieba 分词开销）—— 语料越大越慢，故单列一项以观察其随规模的增长。"""
        if not _TRACE:
            return self._bm25.search(query, k)
        t0 = time.perf_counter()
        out = self._bm25.search(query, k)
        self._acc("bm25_search", time.perf_counter() - t0)
        return out

    @staticmethod
    def _build_cross_encoder_reranker() -> Reranker:
        try:
            from .rerank import CrossEncoderReranker

            return CrossEncoderReranker()
        except Exception as e:  # noqa: BLE001
            logger.warning("[pipeline] CrossEncoderReranker 初始化失败，回退 ScoreReranker：%s", e)
            return ScoreReranker()

    def add_document(
        self,
        source: str | Path | list[str],
        doc_id: str = "doc",
        *,
        cancel_event=None,
        on_progress=None,
        batch_size: int = 64,
    ) -> int:
        blocks = self._parser.parse(source)
        chunks = self._chunker.chunk_blocks(blocks, doc_id=doc_id)
        if not chunks:
            return 0
        items = [(c.id, c.text, c.page) for c in chunks]
        return self.add_chunks(
            items, doc_id=doc_id, cancel_event=cancel_event, on_progress=on_progress, batch_size=batch_size
        )

    def add_chunks(
        self,
        items: list[tuple[str, str, int | None]],
        doc_id: str = "doc",
        *,
        cancel_event=None,
        on_progress=None,
        batch_size: int = 64,
    ) -> int:
        """直接索引已分块、带页码的 (id, text, page) 三元组（不再经「去页码文本」重建）。

        用于从数据库 Chunk 表（已正确记录 1 起算页码）构建管线，确保 retrieve /
        pdf_extract / citation_lookup 返回的页码与落库页码一致——这是修复「检索返回
        page=0」的根因（此前管线由 page-less 的扁平文本重建，所有块 page=0）。

        分批写入以支持「中途取消」与「进度回调」：每批之间检查 ``cancel_event``（置位则
        抛 ``IndexingCancelled``），并通过 ``on_progress(done, total)`` 上报进度。默认路径
        （cancel_event/on_progress 为 None）行为与旧版一致——一次性编码全部块。
        """
        if not items:
            return 0
        # N16：索引内容变化 → 查询缓存整体失效（命中结果与索引强相关）
        self._query_cache.clear()
        self._query_cache_order.clear()
        total = len(items)
        done = 0
        for start in range(0, total, batch_size):
            if cancel_event is not None and cancel_event.is_set():
                raise IndexingCancelled(f"索引在 {done}/{total} 处被取消")
            batch = items[start : start + batch_size]
            ids = [i for i, _, _ in batch]
            texts = [t for _, t, _ in batch]
            pages = [p for _, _, p in batch]
            vectors = self.embedder.encode(texts)
            self._vidx.add(vectors, texts, meta=[{"id": i, "page": p} for i, p in zip(ids, pages, strict=False)])
            self._bm25.add(texts)
            if self._sidx is not None:
                self._sidx.add(self.embedder.encode_sparse(texts))
            self._chunks.extend(
                _IndexChunk(id=i, text=t, page=p) for i, t, p in zip(ids, texts, pages, strict=False)
            )
            done += len(batch)
            if on_progress is not None:
                on_progress(done, total)
        return done

    # ---------- 索引状态持久化（2026-09-07：重启后免重编码） ----------

    def save_state(self, faiss_path: Path, state_path: Path, signature: str) -> None:
        """持久化索引状态：dense 向量（faiss 原生格式）+ 稀疏 postings/分块（pickle）。

        供上层 service 在重启后直接恢复管线，免去 BGE-M3 全量重编码（分钟级）。
        ``signature`` 绑定分块内容与嵌入器配置：不一致时 load_state 返回 None → 重编码。
        BM25 不落盘——由 texts 在 load 时廉价重建（纯分词统计，无模型）。
        """
        import pickle

        import faiss

        faiss.write_index(self._vidx._index, str(faiss_path))
        bundle = {
            "signature": signature,
            "texts": [c.text for c in self._chunks],
            "metas": [{"id": c.id, "page": c.page} for c in self._chunks],
            "chunks": [{"id": c.id, "text": c.text, "page": c.page} for c in self._chunks],
            "sparse": None if self._sidx is None else self._sidx.export_vecs(),
        }
        state_path.write_bytes(pickle.dumps(bundle))

    @classmethod
    def load_state(
        cls,
        *,
        faiss_path: Path,
        state_path: Path,
        signature: str,
        embedder: Embedder,
        reranker: Reranker | None,
        use_bge_sparse: bool,
        tokenize: Callable[[str], list[str]] = _char_tokenize,
    ) -> RAGPipeline | None:
        """从落盘状态恢复管线；签名不符/文件损坏/结构校验失败 → None（调用方走重编码）。"""
        try:
            import pickle

            import faiss

            # 2026-09-07 日志分级：缓存文件不存在 = 首次访问/旧版文档迁移（预期路径），
            # 打 INFO 且不带堆栈；文件存在但损坏/签名不符才是异常，保留 WARNING。
            if not Path(state_path).exists():
                logger.info("无索引缓存（首次访问或功能上线前的旧文档），将现场编码并落盘")
                return None
            bundle = pickle.loads(Path(state_path).read_bytes())
            if not isinstance(bundle, dict) or bundle.get("signature") != signature:
                return None
            inst = cls(embedder=embedder, reranker=reranker, use_cross_encoder=False,
                       use_bge_sparse=use_bge_sparse, tokenize=tokenize)
            texts: list[str] = bundle["texts"]
            metas: list[dict] = bundle["metas"]
            vi = inst._vidx
            vi._index = faiss.read_index(str(faiss_path))
            vi._texts = list(texts)
            vi._meta = list(metas)
            inst._bm25.add(texts)
            if inst._sidx is not None and bundle.get("sparse"):
                inst._sidx.add(bundle["sparse"])
            inst._chunks.extend(_IndexChunk(**c) for c in bundle["chunks"])
            # 结构一致性校验：三路状态必须对齐，否则视为缓存损坏
            if not (len(vi) == len(inst._bm25._docs) == len(inst._chunks) == len(texts)):
                logger.warning("索引缓存结构不一致（faiss=%s bm25=%s chunks=%s），放弃缓存",
                               len(vi), len(inst._bm25), len(inst._chunks))
                return None
            return inst
        except Exception:  # noqa: BLE001  缓存损坏不应阻断建管线，退回重编码
            logger.warning("索引缓存加载失败，将重新编码", exc_info=True)
            return None

    def query(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        # N16：查询缓存命中 → 跳过 编码+三路检索+重排（实测约 1.5s/次）
        key = (query, top_k)
        cached = self._query_cache.get(key)
        if cached is not None:
            if _TRACE:
                # ★ 命中必须**显式标注**：否则测出来是 0ms，会被读成"检索飞快"。
                #   2026-09-23 实测踩过这个坑 —— 预热调用把结果写进缓存，随后同 query 全得 0ms，
                #   差点据此得出"检索只要 0ms"的荒谬结论。
                self.last_stage_ms = {"cache_hit": 1.0}
                logger.info(
                    "[rag-trace] 查询缓存命中 → 本次**未做真实检索**（此 0ms 不可用作性能依据）"
                )
            return list(cached)
        t_all = time.perf_counter() if _TRACE else 0.0
        if _TRACE:
            self.last_stage_ms = {}
        candidates = self._hybrid.retrieve(query, top_n=self.top_n)
        # CrossEncoderReranker 需要候选 chunk 文本才能计算 (query, chunk) 相关性；
        # ScoreReranker 忽略 texts，向后兼容。
        # 契约（2026-09-03）：texts 与 candidates 按位置一一对应（不是按下标索引的全量文本）。
        texts = [self._chunks[idx].text for idx, _ in candidates]
        t_r = time.perf_counter() if _TRACE else 0.0
        try:
            ranked = self.reranker.rerank(query, candidates, top_k=top_k, texts=texts)
        except Exception:  # noqa: BLE001  重排失败不得摧毁整条检索：退化为融合分数排序
            logger.warning("重排失败，退化为融合分数排序", exc_info=True)
            ranked = sorted(candidates, key=lambda kv: kv[1], reverse=True)[:top_k]
        finally:
            # 异常路径**同样计时** —— 降级排序也在花时间；只在成功路径打点会漏掉
            # 最需要诊断的那种场景（"这次为什么慢"往往伴随降级）。
            if _TRACE:
                self._acc("rerank", time.perf_counter() - t_r)
        out: list[RetrievedChunk] = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            out.append(RetrievedChunk(id=chunk.id, text=chunk.text, page=chunk.page, score=score))
        # 写缓存（FIFO，容量 32）：dict 操作在 GIL 下原子，竞态至多多算一次检索，无害
        self._query_cache[key] = out
        self._query_cache_order.append(key)
        if len(self._query_cache_order) > 32:
            old_key = self._query_cache_order.pop(0)
            self._query_cache.pop(old_key, None)
        # 2026-09-15 加固：**所有返回路径一律返回副本**（与命中路径的 `list(cached)` 对齐）。
        # 此前非命中路径直接 `return out` —— 交出去的就是缓存本体，调用方一次
        # `clear()` / `append()` 就把缓存改坏了，命中路径的副本保护因此形同虚设
        # （由 tests/test_retrieved_chunk_immutable.py 用例 2 复现）。
        if _TRACE:
            self.last_stage_ms["total"] = (time.perf_counter() - t_all) * 1000.0
            self._log_stage_trace(query, len(candidates))
        return list(out)

    def _log_stage_trace(self, query: str, n_candidates: int) -> None:
        """输出一行阶段耗时（仅 `PAPERMATE_RAG_TRACE=1` 时调用）。

        各阶段**互斥**（`total` 除外），故可相加对照 `total`；差值即 RRF 融合等未单独打点的部分。
        若差值出现**负数**，说明有阶段被重复计入（重叠），也是有用信号 —— 故不夹断为 0。
        """
        ms = self.last_stage_ms
        total = ms.get("total", 0.0)
        known = sum(v for k, v in ms.items() if k != "total")
        detail = " ".join(f"{k}={v:.1f}" for k, v in sorted(ms.items()) if k != "total")
        logger.info(
            "[rag-trace] chunks=%d cand=%d top_n=%d sparse=%s total=%.1fms | %s | "
            "未单独打点(含RRF融合)=%.1fms | q=%r",
            len(self._chunks), n_candidates, self.top_n, self.use_bge_sparse,
            total, detail, total - known, query[:40],
        )

    def get_chunks(self) -> list:
        """返回已索引的全部原文 chunk（含 id/page/text），供 pdf_extract / citation_lookup 等工具复用。"""
        return list(self._chunks)
