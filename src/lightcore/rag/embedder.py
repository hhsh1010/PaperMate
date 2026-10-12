"""LightCore · 嵌入抽象（docs/05 RAG / docs/13 Spike S2）

Embedder 是记忆系统（long_term）与 RAG 子系统共用的依赖注入点：
- FakeHashEmbedder：确定性哈希向量，零依赖，用于本地/Spike/测试（docs/08 FakeLLM 同思想）。
- BgeM3Embedder    ：真实 BGE-M3 适配器（GPU/CPU 自适应，GPU 优先 + fp16），依赖官方 FlagEmbedding 库
                      （`uv sync --extra rag` 安装），首次需联网下载约 2GB 权重（BAAI/bge-m3，走 HF 镜像缓存）。

二者接口一致（encode(texts)->list[vec]），切换不影响上层。
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Sequence

import numpy as np

from ..runtime.resource_lock import hold_model_load

logger = logging.getLogger(__name__)

#: 首次导入失败后的重试退避（秒）。长度 = 额外尝试次数；总尝试次数 = 1 + len(该元组)。
#: 取值依据：`transformers` 首次导入本身就要 1~3 秒，故给足时间窗而不是"立刻重试"。
_IMPORT_BACKOFF_S: tuple[float, ...] = (0.3, 0.8)


def _detect_local_bge_m3() -> str | None:
    """探测 HF hub 缓存里已下载的 BAAI/bge-m3 快照目录（保留旧名以兼容既有调用/文档）。

    BGE-M3 原版仓库含一个 ``imgs/.DS_Store``，hf-mirror 镜像对其返回 403，
    导致 ``BGEM3FlagModel`` 走 Hub 自动下载时整批失败。但推理实际只需要
    config.json / 权重 / tokenizer*，这些往往已落在 hub 缓存快照里。
    直接把该快照目录当本地路径传给 FlagEmbedding 即可绕过下载、也不碰 .DS_Store。

    实现下沉到 ``hf_local.resolve_local_hf_snapshot``（重排器复用同一套逻辑）。
    """
    from .hf_local import resolve_local_hf_snapshot

    return resolve_local_hf_snapshot("BAAI/bge-m3", env_var="PAPERMATE_BGE_M3_PATH")


def _ensure_transformers_npu_symbol(module: object | None = None) -> bool:
    """兜底补齐 FlagEmbedding 对 ``transformers.is_torch_npu_available`` 的硬依赖。

    **背景（2026-09-18 线上实测，会话 s18）**：FlagEmbedding 的 ``AbsEmbedder`` 第 14 行写死
    ``from transformers import is_torch_npu_available``。该符号一旦在顶层 ``transformers``
    取不到，导入即抛 ``ImportError: cannot import name 'is_torch_npu_available' from 'transformers'``
    —— 而调用方用 ``except ImportError`` 统一包装成「未安装 FlagEmbedding」，
    **真因被掩盖**（排查因此被带偏两轮：先怀疑解释器、再怀疑环境），
    同时整条 RAG 链路熔断（文档检索全不可用）。

    由于该符号**只用于 NPU 探测**，本机（CUDA / CPU）恒为 False，与推理路径无关，
    补齐它对结果没有任何影响 —— 所以这里做兜底而不是直接失败。

    实现顺序：优先取上游权威实现；上游也取不到时才注入恒 False 的等价实现。
    返回 True 表示**确实做了注入**（即上游缺失），调用方可据此决定是否重试导入。
    """
    if module is None:
        try:
            import transformers

            module = transformers
        except Exception as exc:  # noqa: BLE001 连 transformers 都取不到 → 无可兜底
            # 分支①：这个分支本身也是重要信息（说明 transformers 处于不可导入状态）
            logger.error("兜底检查：`import transformers` 失败，无法兜底：%s", exc, exc_info=True)
            return False

    try:
        if hasattr(module, "is_torch_npu_available"):
            # 分支②：符号其实在 —— 说明失败是"时序"而非"缺失"，重试就有机会
            logger.info("兜底检查：transformers 顶层此刻已有 is_torch_npu_available（无需注入）")
            return False
    except Exception:  # noqa: BLE001
        logger.error("兜底检查：hasattr(transformers, 'is_torch_npu_available') 抛错", exc_info=True)
        return False

    try:
        from transformers.utils.import_utils import is_torch_npu_available as upstream

        impl, source = upstream, "上游 transformers.utils.import_utils"
    except Exception:  # noqa: BLE001 上游也取不到 → 用语义等价的本地实现

        def impl(check_device: bool = False) -> bool:  # type: ignore[misc]
            return False

        source = "本地实现（恒 False，等同 NPU 缺失时的上游语义）"

    try:
        module.is_torch_npu_available = impl  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.warning("注入 is_torch_npu_available 失败，继续尝试导入 FlagEmbedding", exc_info=True)
        return False

    logger.warning(
        "transformers 顶层缺少 is_torch_npu_available（transformers=%s）→ 已注入兜底实现（来源：%s）。"
        "该符号仅用于 NPU 探测，CUDA/CPU 路径不受影响；此告警出现即说明环境有异常，请留档。",
        getattr(module, "__version__", "?"),
        source,
    )
    return True


def _snapshot_transformers_state(exc: BaseException) -> dict:
    """首次导入失败时的**可诊断快照**。

    这类失败只在特定的"进程内导入时序"下出现（同一份代码换个进程就跑得过），
    若不把当时状态落下来，下次仍然只能靠猜 —— 2026-09-18 为此空转了三轮。
    """
    tf = sys.modules.get("transformers")
    utils = sys.modules.get("transformers.utils")
    struct = getattr(tf, "_import_structure", None) if tf is not None else None

    snap: dict[str, object] = {
        "异常": f"{type(exc).__name__}: {exc}",
        "sys.executable": sys.executable,
        "线程": threading.current_thread().name,
        "sys.modules 有 transformers": tf is not None,
        "sys.modules 有 transformers.utils": utils is not None,
        "sys.modules 有 torch": "torch" in sys.modules,
        "sys.modules 有 sentence_transformers": "sentence_transformers" in sys.modules,
    }
    if tf is not None:
        snap["transformers 类型"] = type(tf).__name__
        snap["transformers 文件"] = getattr(tf, "__file__", None)
        snap["顶层此刻可解析该符号"] = hasattr(tf, "is_torch_npu_available")
        if isinstance(struct, dict):
            snap["_import_structure['utils'] 含该符号"] = "is_torch_npu_available" in struct.get("utils", ())
        else:
            snap["_import_structure"] = "缺失（可能未替换为 LazyModule）"
    if utils is not None:
        snap["utils 模块有该符号"] = hasattr(utils, "is_torch_npu_available")
    return snap


def _import_bgem3_flag_model():
    """导入 ``FlagEmbedding.BGEM3FlagModel``：失败时**暴露真因 + 落快照 + 总是重试一次**。

    **为什么不能只写一句"未安装 FlagEmbedding"**：这条导入至少有三种失败原因 ——
    ① 真的没装；② 装好了但内部依赖（`transformers` 的某个符号）取不到；③ 包目录损坏。
    旧实现把三者合成一句话，导致线上排查连续两次走错方向（先怀疑解释器、再怀疑环境）。

    **为什么必须"总是重试"**：2026-09-18 线上实测（会话 s18）该导入失败属**进程内导入时序问题** ——
    失败瞬间 `from transformers import is_torch_npu_available` 取不到符号，
    但**片刻之后**在同一进程里 `hasattr(transformers, ...)` 已为 True（状态已自愈）。
    旧实现只在"兜底注入成功"时才重试，于是这种"已自愈"的情形被直接抛错 → 整条 RAG 熔断。
    现改为**有界重试**（`_IMPORT_BACKOFF_S`），每次失败都落证据。
    """
    first_exc: BaseException | None = None
    last_exc: BaseException | None = None
    injected: bool | None = None
    total = 1 + len(_IMPORT_BACKOFF_S)

    for attempt in range(total):
        if attempt:
            time.sleep(_IMPORT_BACKOFF_S[attempt - 1])
        try:
            model = importlib.import_module("FlagEmbedding").BGEM3FlagModel
        except (ImportError, AttributeError) as exc:
            last_exc = exc
            if first_exc is None:
                first_exc = exc
                logger.error(
                    "FlagEmbedding 首次导入失败，环境快照：%s",
                    _snapshot_transformers_state(exc),
                    exc_info=True,
                )
                injected = _ensure_transformers_npu_symbol()
            else:
                logger.warning("FlagEmbedding 第 %d 次尝试仍失败：%s: %s", attempt + 1, type(exc).__name__, exc)
            continue
        if first_exc is not None:
            logger.warning(
                "FlagEmbedding 首次导入失败但**第 %d 次尝试成功**（兜底注入=%s）→ 属进程内导入时序问题，"
                "已留环境快照；首次真因：%s: %s",
                attempt + 1,
                injected,
                type(first_exc).__name__,
                first_exc,
            )
        return model

    raise RuntimeError(
        f"FlagEmbedding 导入失败（共尝试 {total} 次；兜底注入={injected}）。"
        f"真因：{type(last_exc).__name__}: {last_exc} —— 详见 app.log 里的「环境快照」；"
        "若已执行过 `uv sync --extra rag`，请按快照字段排查（见 pyproject[rag]）"
    ) from last_exc


class Embedder:
    dim: int = 0

    def encode(self, texts: Sequence[str]) -> list[list[float]]:  # noqa: D401
        raise NotImplementedError


class FakeHashEmbedder(Embedder):
    """确定性哈希嵌入：把文本哈希成定长向量，零依赖。仅用于流程验证，不可用于真实语义检索。"""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for i, ch in enumerate(t):
                h = int(hashlib.md5(f"{i}:{ch}".encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0 if (h & 1) else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class BgeM3Embedder(Embedder):
    """真实 BGE-M3 适配器（GPU/CPU 自适应）。依赖官方 ``FlagEmbedding`` 库（rag extra 下安装）。

    BGE-M3 的稀疏（lexical）输出依赖原版 BAAI/bge-m3 的 ``FlagModel`` 建模代码
    （``BGEM3FlagModel``），它**不在** sentence-transformers 导出的稠密快照里——
    那只有 XLM-RoBERTa 底座，``encode(return_sparse=...)`` 会被 ST 包装器直接拒绝
    （"this model does not accept any additional keyword arguments: ['return_sparse']"）。
    因此改用官方 ``FlagEmbedding`` 直接加载，原生支持 dense/sparse/colbert，
    彻底绕开 ST 的 kwarg 校验坑。
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        # 导入经 _import_bgem3_flag_model：失败时带真因，并在"符号缺失"时先兜底重试
        # （2026-09-18 线上事故：旧文案把"符号缺失"误报成"未安装包"，把排查带偏）。
        BGEM3FlagModel = _import_bgem3_flag_model()
        # 模型路径解析优先级：
        #   1) 显式传入 model_name
        #   2) 环境变量 PAPERMATE_BGE_M3_PATH（指向含 config.json / pytorch_model.bin / tokenizer* 的本地目录）
        #   3) 自动探测 HF hub 缓存里已下载的 BAAI/bge-m3 快照（绕过 hf-mirror 对 imgs/.DS_Store 的 403）
        #   4) 回退 "BAAI/bge-m3" 走 Hub 自动下载（国内镜像可能 403 imgs/.DS_Store，建议用上面 2/3）
        if model_name is None:
            model_name = os.environ.get("PAPERMATE_BGE_M3_PATH")
        if model_name is None:
            model_name = _detect_local_bge_m3()
        if model_name is None:
            model_name = "BAAI/bge-m3"
        # 设备自适应：显式传入 > PAPERMATE_DEVICE env > torch.cuda 探测 > cpu。
        # use_fp16 仅在 GPU 有意义（显存减半、推理 ~2x 提速）；CPU 关闭以保证数值稳定。
        # 注意 FlagEmbedding 的 kwarg 是 devices（复数，1.3+ 新 API），不是 device。
        if device is None:
            from .hf_local import auto_device

            device = auto_device()
        self.device = device
        # G1：重模型构造走**跨进程互斥**。实测本步 12.7~18.3s（占冷启动 71.8%），
        # 两个进程同时构造会互抢 8GB 显存 ⇒ 双方死锁（CPU 0%、GPU 占满、零产出）。
        # ⚠️ 本函数由调用方经 `asyncio.to_thread` 执行（见 `document_service`：'必须走 to_thread'），
        #    故这里的**阻塞等待是安全的**；若将来有调用方在事件循环里直接调用它，会冻结后端。
        with hold_model_load("嵌入模型 bge-m3"):
            self._model = BGEM3FlagModel(
                model_name, use_fp16=(device == "cuda"), devices=device
            )
        logger.info("BgeM3Embedder 已加载（model=%s, device=%s, fp16=%s）",
                    model_name, device, device == "cuda")
        self.dim = getattr(self._model, "dim", 1024)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = self._model.encode(
            list(texts),
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # FlagEmbedding 返回键为 "dense_vecs"（np.ndarray, shape=(N,dim)）
        arr = np.asarray(out["dense_vecs"], dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()

    def encode_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """BGE-M3 稀疏（lexical）表示：token_id -> weight。供 SparseIndex 做词法检索。

        走 FlagEmbedding 原生 ``return_sparse=True``（与 ST 包装器无关，无 kwarg 兼容坑）。
        FlagEmbedding 返回键为 ``lexical_weights``，类型为 ``List[Dict[str, float]]``
        （每句一个 dict，键是 token_id 字符串、值是权重）——``_sparse_to_dict`` 负责归一为
        ``dict[int, float]``。兼容性兜底：若稀疏抽取出错，降级为空稀疏向量并告警，
        避免单篇/整批 reindex 因稀疏失败而 500；稠密向量与 BM25 不受影响。
        """
        try:
            out = self._model.encode(
                list(texts),
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            lexical = out["lexical_weights"]  # List[Dict[str, float]], 每句一个
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "BgeM3Embedder.encode_sparse 降级为空稀疏（稠密+BM25 仍可用）：%s", e
            )
            return [{} for _ in texts]
        return [self._sparse_to_dict(v) for v in lexical]

    @staticmethod
    def _sparse_to_dict(v) -> dict[int, float]:
        # FlagEmbedding 的 lexical_weights 是 dict（键为 token_id 字符串，值为权重）
        if isinstance(v, dict):
            return {int(k): float(w) for k, w in v.items()}
        try:
            import scipy.sparse as sp

            if sp.issparse(v):
                v = v.tocsr()
                return {int(v.indices[i]): float(v.data[i]) for i in range(v.nnz)}
        except Exception:  # noqa: BLE001
            pass
        return {}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
