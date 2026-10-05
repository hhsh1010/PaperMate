"""LightCore · 重排（docs/05 §RAG / Cross-Encoder 重排）

★ 2026-10-04（O9）修正过时描述：标题原写「重排 **Top50**→5」，而
`RAGPipeline.__init__` 的 `top_n_hybrid` 默认已是 **20**（N16，2026-09-12 由 50 → 20）。
⇒ 本模块**不决定**候选池大小，只负责把传入的候选（默认 20 条）重排后取 `top_k`。
改文档时以 `pipeline.py` 的 `top_n` 为准。

- ScoreReranker   ：用混合检索的融合分数直接重排（默认，零依赖），产出 Top-K。
- CrossEncoderReranker：真实 BGE-reranker 适配器（sentence-transformers），需 rag extra + 权重；
                        对 (query, chunk) 对打分重排，精度更高。

两者接口一致：rerank(query, candidates, top_k) -> list[(idx, score)]
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """重排器统一接口：输入候选 (idx, score) 与可选文本，输出重排后的 (idx, score)。"""

    def rerank(self, query: str, candidates: Sequence[tuple[int, float]], top_k: int = 5,
               texts: Sequence[str] | None = None) -> list[tuple[int, float]]:
        ...


class ScoreReranker:
    """直接用已有融合分数重排（HybridRetriever 输出）。"""

    def rerank(self, query: str, candidates: Sequence[tuple[int, float]], top_k: int = 5,
               texts: Sequence[str] | None = None):
        ordered = sorted(candidates, key=lambda kv: kv[1], reverse=True)
        return list(ordered[:top_k])


class CrossEncoderReranker:
    """BGE-reranker 适配器（懒加载，需 sentence-transformers + 模型权重）。

    **本地优先加载**：与 :mod:`lightcore.rag.embedder` 的 BGE-M3 同一套策略
    （见 :mod:`lightcore.rag.hf_local`）——先看 hub 缓存里有没有完整快照，
    有就把快照目录当本地模型目录传入并置 ``local_files_only=True``，
    transformers 对本地目录不会发起任何网络请求。

    原因：国内直连 ``huggingface.co`` 会超时（10s × 5 次重试 ≈ 50s 卡顿），
    而 ST 即使模型已缓存也会为校验 revision 发 HEAD 请求；命中本地快照后
    这一步被完全跳过，重建索引不再被联网拖住。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None,
                 max_length: int | None = None, fp16: bool = False,
                 batch_size: int | None = None) -> None:
        """T-03（2026-09-24）：新增三个**推理加速**参数，默认值与 2026-09-24 之前**完全一致**。

        三个参数都是**显式注入**（本层不读任何配置）——线上由 papermate 层从 runtime_settings
        取值后传入，与 `RAGPipeline(use_bge_sparse=...)` 的注入风格保持一致（lightcore 不反向依赖 papermate）。

        实测依据（4 候选 / RTX 4070 Laptop / 与线上同源口径；详见
        `.workbuddy/artifacts/2026-09-24-T03重排四臂对照.md`）：

        ===========================  =========  =============  =============
        配置                          p50        vs baseline    打分 max-abs-diff
        ===========================  =========  =============  =============
        默认(8192/fp32 = 今天的行为)   208.7 ms   ×1.00          —
        ``max_length=512``            188.3 ms   ×1.09          2.53e-03
        ``max_length=256``            91.2 ms    ×2.24          3.62e-02
        ``max_length=512`` + fp16     62.6 ms    ×3.27          2.66e-03
        ``max_length=384`` + fp16     48.1 ms    ×4.34          1.14e-02
        ``max_length=256`` + fp16     33.3 ms    ×6.26          3.65e-02
        ===========================  =========  =============  =============

        **五档的 Top-5 顺序全部一致（0/5 换位）**。默认档位 2026-10-05 起由 384 上调为 **512**
        （质量优先实测：512 相对 384 的 hit@5 +3.27pp / nDCG@5 +5.15pp / MRR@10 +8.36pp，
        代价 +51ms；384 反而截断约 35% 候选 ⇒ 属欠拟合档）。
        ⚠️ 上表"顺序全一致"是**同池内**排序稳定性，**不能**推论"截断不伤召回" ——
        221 case 实测 256 档 hit@5 **−4.92pp**（顺序稳定 ≠ 质量不变）。
        依据：docs/rerank参数下调-可行性评估-2026-10-05.md

        参数
        ----
        max_length
            送入 Cross-Encoder 的最大 token 数。``None``/``0`` = **不限长**（严格等于改动前行为：
            sentence-transformers 会回落到 tokenizer 的 ``model_max_length`` = 8192）。
            截断的收益不在"平均缩短"，而在**砍掉长尾** —— 一个 batch 的成本由 batch 内最长序列
            决定（padding 到最大值），所以砍掉少数超长候选收益远大于按平均值估算。
        fp16
            是否以半精度推理。⚠️ **只在 CUDA 上生效**：非 CUDA 设备上 fp16 通常**更慢**
            （无半精度加速单元），故强制回落 fp32 **并显式告警**（不静默）。
        batch_size
            交给 sentence-transformers 的 batch。``None`` = 用其默认（32）。
            实测 batch 8/16/32 的 p50 为 63.5/62.8/63.5 ms —— **几乎完全平坦**，
            即该项在本负载下**没有调优价值**，保留只为将来换模型时留出口。
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 sentence-transformers，请先 `uv sync --extra rag`") from e
        # 设备自适应：显式传入 > PAPERMATE_DEVICE env > torch.cuda 探测 > cpu（与 BgeM3Embedder 一致）
        if device is None:
            from .hf_local import auto_device

            device = auto_device()
        self.device = device
        # 归一化：0/负数一律视为"不限长"，避免把候选截成空串
        self.max_length = max_length if (isinstance(max_length, int) and max_length > 0) else None
        self.batch_size = batch_size if (isinstance(batch_size, int) and batch_size > 0) else None
        want_fp16 = bool(fp16) and str(device).startswith("cuda")
        if fp16 and not want_fp16:
            logger.warning("reranker fp16 只在 CUDA 上启用（当前 device=%s）→ 回落 fp32", device)
        self.fp16 = want_fp16
        local = self._resolve_local(model_name)
        if local:
            self._model = CrossEncoder(local, device=device, local_files_only=True,
                                       max_length=self.max_length)
        else:
            # 无本地快照：兜底指向国内镜像后再走常规下载（避免直连被墙域名干等）
            try:
                from .hf_local import ensure_hf_mirror

                ensure_hf_mirror()
            except Exception:  # noqa: BLE001  镜像兜底失败不影响主流程
                pass
            self._model = CrossEncoder(model_name, device=device, max_length=self.max_length)
        if self.fp16:
            self._model.model.half()

    @staticmethod
    def _resolve_local(model_name: str) -> str | None:
        """解析本机缓存快照目录；解析失败（含未装 hf_local）一律返回 None 走常规路径。"""
        try:
            from .hf_local import resolve_local_hf_snapshot

            return resolve_local_hf_snapshot(model_name, env_var="PAPERMATE_RERANKER_PATH")
        except Exception:  # noqa: BLE001  本地探测属加速手段，不应阻断加载
            return None

    def rerank(self, query: str, candidates: Sequence[tuple[int, float]], top_k: int = 5,
               texts: Sequence[str] | None = None) -> list[tuple[int, float]]:
        """重排并按 (chunk_idx, score) 返回 Top-K。

        契约（2026-09-03 修正）：``texts`` 与 ``candidates`` **按位置一一对应**
        （texts[i] 是 candidates[i] 对应 chunk 的文本），不是「按 chunk 下标索引的全量文本」。
        此前实现写成 ``texts[idx]``（idx=candidates[0]，全局 chunk 下标），
        与 RAGPipeline.query 的实际传参不符：chunk 下标远大于候选数 →
        IndexError → 整条检索抛错 → 上层退化成「未关联任何文档」。
        """
        if texts is None:
            raise ValueError("CrossEncoderReranker.rerank 需要 texts 以计算相关性")
        if len(texts) != len(candidates):
            raise ValueError(
                f"CrossEncoderReranker.rerank：texts 长度({len(texts)}) 必须等于候选数({len(candidates)})"
            )
        if not candidates:
            # M9 修复（2026-09-10）：空输入直接返回 []——`predict([])` 对空 batch 通常
            # 抛错（tokenizer 无法确定 padding 长度），与 ScoreReranker 的空输入行为
            # （返回空列表）保持一致。当前唯一调用方有兜底，但直连调用处会炸。
            return []
        pairs = [(query, t) for (_, _), t in zip(candidates, texts, strict=False)]
        # 2026-09-11：**显式关闭 tqdm 进度条**——`predict` 默认 `show_progress_bar=None`，
        # 会继承实例属性；实测每次提问都在后端日志刷 2~6 行 `Batches: 2/2`，
        # 被误读为「又在重建索引」。重排是每次检索的常规环节（候选 ~50-100 条 → 2 个 batch），
        # 进度条无信息量；真正的索引重建走 `add_chunks` 的 on_progress 回调，与此无关。
        # T-03：`batch_size` 只在显式设过时才传 —— `predict(batch_size=None)` 会在
        # 「按 batch 切片」处炸（None 不可用于 range/DataLoader），不能图省事直接传 None。
        kw: dict = {"show_progress_bar": False}
        if self.batch_size:
            kw["batch_size"] = self.batch_size
        scores = self._model.predict(pairs, **kw)
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]
