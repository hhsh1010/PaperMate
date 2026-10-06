"""PaperMate · 真实语义检索模型的**显式安装器**（2026-10-05 桌面客户端需求）

背景（为什么必须有这一层）
──────────────────────────
真实语义检索 = BGE-M3 嵌入 + bge-reranker-v2-m3 重排，合计约 **4.5GB** 权重。
桌面 App **装完必须能用、且绝不静默下载 4GB** ⇒ 首启默认 `fake`（见 `config.default_embedder`），
真实权重由用户**安装后在设置页显式点一次下载**，下载完成后**也不自动启用**
（切成 real 会使既有索引失效，必须"手动启用 + 重建索引"，这个代价要由用户知情确认）。

设计要点（每条都对应一个真实踩过的坑）
────────────────────────────────────
1. **逐个文件下载**（`hf_hub_download`）而非 `snapshot_download`：
   - `hf-mirror` 对某些仓库的无关文件（实测 `BAAI/bge-m3` 的 `imgs/.DS_Store`）返回 **403**，
     `snapshot_download` 会在**最后一个文件**上整批 abort（详见 `lightcore.rag.hf_local` 文档串）；
   - 逐文件可拿到**真实进度**（第 i/N 个文件）、可**取消**、天然**断点续传**（已缓存的直接跳过）；
   - 且能对"无关文件下载失败"**单独容错**：失败后统一用 `_has_weight()/_missing_required()` 复核，
     **只要推理必需的文件齐了就算成功** —— 与线上 `hf_local` 的"快照解析"口径同源。
2. **下载前做磁盘余量检查**：先 fail fast（错误明确），而不是下到一半 `No space left`。
3. **后台线程 + 可轮询状态**：不在请求线程里跑分钟级任务（会阻塞事件循环 / 触发网关超时）。
4. **失败 alike**：错误统一 `ModelInstallError`，由路由层转 400/409，**不裸抛堆栈给前端**。

★ LLM 模块硬约束：本模块**不调用任何 LLM**（纯权重下载），三项硬约束 N/A（保持记录习惯）。
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass

from lightcore.rag.hf_local import (
    _WEIGHT_FILES as HF_WEIGHT_FILES,  # 与线上推理**同源**的权重文件名表（不另造一份）
)
from lightcore.rag.hf_local import (
    default_hub_cache_dir,
    ensure_hf_mirror,
    resolve_local_hf_snapshot,
)

logger = logging.getLogger(__name__)

GB = 1024 ** 3


# --------------------------------------------------------------------------- #
# 模型清单（单一来源：设置页、UI 文案、下载编排共用）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RagModelSpec:
    """一个可按需安装的本地模型。"""

    key: str          # 稳定标识（API 里用它）
    repo_id: str      # HuggingFace 仓库
    label: str        # 设置页展示名
    approx_gb: float  # 预计下载体积（仅用于磁盘余量提示，非精确值）
    allow_patterns: tuple[str, ...]   # 只下载这些（fnmatch），其余一律跳过
    required_files: tuple[str, ...]   # 推理必需（缺一即判安装失败）


#: 推理必需的**非权重**文件
_TEXT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

#: 权重文件（含分片）。缺任一**形式**都算没权重，但与 `_TEXT_FILES` 分开判定
#: ——因为不同仓库可能是 `model.safetensors` 或 `pytorch_model.bin`，二者取其一即可。
_WEIGHTS = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
_SHARD_WEIGHTS = ("model-*.safetensors",)  # 分片权重（如 model-00001-of-00002.safetensors）

EMBEDDER_SPEC = RagModelSpec(
    key="embedder",
    repo_id="BAAI/bge-m3",
    label="BGE-M3 嵌入模型（真实语义召回）",
    approx_gb=2.3,
    allow_patterns=(
        *_TEXT_FILES,
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "vocab.txt",
        "1_Pooling/config.json",   # sentence-transformers 池化层配置，缺失会加载失败
        *_WEIGHTS,
        *_SHARD_WEIGHTS,
    ),
    required_files=_TEXT_FILES,
)

RERANKER_SPEC = RagModelSpec(
    key="reranker",
    repo_id="BAAI/bge-reranker-v2-m3",
    label="bge-reranker-v2-m3 重排模型（提升 Top-K 精度）",
    approx_gb=2.2,
    allow_patterns=(
        *_TEXT_FILES,
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "vocab.txt",
        *_WEIGHTS,
        *_SHARD_WEIGHTS,
    ),
    required_files=_TEXT_FILES,
)

MODEL_SPECS: dict[str, RagModelSpec] = {s.key: s for s in (EMBEDDER_SPEC, RERANKER_SPEC)}

#: 磁盘余量安全系数：HF 缓存可能同时存在旧分片 / 临时 *.incomplete，实测需留余量
_DISK_SAFETY = 1.15


class ModelInstallError(RuntimeError):
    """安装失败（磁盘不足 / 仓库不可达 / 必需文件缺失）。消息可直接展示给用户。"""


class InstallCancelled(RuntimeError):
    """用户取消安装。"""


class AlreadyRunning(RuntimeError):
    """已有安装任务在跑。"""


# --------------------------------------------------------------------------- #
# 纯函数层（可单测、无副作用、无网络）
# --------------------------------------------------------------------------- #
def cache_dir() -> str:
    """HF hub 缓存根目录。

    ★ 复用 `lightcore.rag.hf_local.default_hub_cache_dir`（`HF_HUB_CACHE` → `HF_HOME/hub`
      → `~/.cache/huggingface/hub`），与**实际下载落盘位置**、与**推理侧探测**三者同源；
      自己再实现一份必然漂移（2026-10-06 首装仿真实测：`~/.cache` 硬编码会让设置页
      在仿真环境里误报"权重已就绪"）。
    """
    return default_hub_cache_dir()


def plan_files(files: list[str], spec: RagModelSpec) -> list[str]:
    """从仓库文件清单里筛出"本次要下载"的文件。

    顺序：按**仓库清单顺序**过滤（更贴近 HF 自己的下载行为），命中任一 `allow_patterns` 即保留；
    同一文件被多个模式命中时只保留一次（`dict.fromkeys` 去重且保序）。
    """
    wanted: list[str] = []
    for f in files:
        if any(fnmatch.fnmatch(f, pat) for pat in spec.allow_patterns):
            wanted.append(f)
    return list(dict.fromkeys(wanted))


def _has_weight(keys: list[str]) -> bool:
    """判定是否已拿到任一种形式的权重文件。"""
    for k in keys:
        if any(fnmatch.fnmatch(k, p) for p in (*_WEIGHTS, *_SHARD_WEIGHTS)):
            return True
        # 兜底：与 hf_local 同源的权重名（将来表里新增也能识别）
        if k in HF_WEIGHT_FILES:
            return True
    return False


def missing_required(spec: RagModelSpec, got: list[str]) -> list[str]:
    """返回缺失的必需项（权重整体算一项：`权重文件`）。"""
    missing = [n for n in spec.required_files if n not in got]
    if not _has_weight(got):
        missing.append("权重文件(model.safetensors / pytorch_model.bin)")
    return missing


def _existing_ancestor(path: str) -> str:
    """返回 ``path`` 最近的**已存在**祖先目录（自身不存在则逐级上跳）。

    ★ 为什么需要：新用户装完 App 时 HF 缓存目录**还不存在**。若直接 `disk_usage(不存在目录)`
      会抛异常 → 首屏显示"磁盘余量未知"，而这恰恰是首装最需要的信息（够不够 4.5GB）。
    """
    p = os.path.abspath(path)
    while p and not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            return os.curdir
        p = parent
    return p or os.curdir


def free_space_gb() -> float | None:
    """HF 缓存所在磁盘的可用 GB；无法探测时返回 ``None``（不抛错）。

    与 `check_free_space` 共用 `_existing_ancestor` ⇒ 展示口径与拦截口径**同源**，
    不会出现"界面说够、点下载说不够"（用户最不信任的一种不一致）。
    """
    try:
        return round(shutil.disk_usage(_existing_ancestor(cache_dir())).free / GB, 2)
    except Exception as e:  # noqa: BLE001  展示属增强信息，探测失败不应让整个卡片挂掉
        logger.info("磁盘余量探测失败：%s", e)
        return None


def check_free_space(needed_gb: float, *, dirpath: str | None = None, usage=None) -> float:
    """磁盘余量检查：不足抛 `ModelInstallError`；返回可用 GB（便于提示）。

    `usage` 可注入（`shutil.disk_usage` 同签名），便于单元测试构造"磁盘不足"。
    """
    target = dirpath or cache_dir()
    probe = target if os.path.isdir(target) else (os.path.dirname(target) or ".")
    try:
        du = (usage or shutil.disk_usage)(probe)
    except Exception as e:  # noqa: BLE001  路径不存在/无权限 → 显式失败而非继续下载
        raise ModelInstallError(f"无法探测磁盘余量（{probe}）：{type(e).__name__}") from e
    free_gb = du.free / GB
    need = needed_gb * _DISK_SAFETY
    if free_gb < need:
        raise ModelInstallError(
            f"磁盘余量不足：可用 {free_gb:.1f}GB，本次安装预计需要 {needed_gb:.1f}GB"
            f"（含 {int((_DISK_SAFETY - 1) * 100)}% 安全余量，需 ≥{need:.1f}GB）。"
            "请清理空间，或把 HF_HOME 指向空间更大的磁盘后重试。"
        )
    return round(free_gb, 2)


def probe_installed(spec: RagModelSpec, *, resolver=resolve_local_hf_snapshot) -> str | None:
    """解析本机是否已有该模型的可用快照；返回本地目录，没有则 `None`。

    ★ 复用 `lightcore.rag.hf_local.resolve_local_hf_snapshot` —— 与**线上推理的探测口径完全一致**，
      避免"设置页说已安装、推理侧却找不到"（这正是本仓库反复强调的『评测/线上同源』纪律）。
    """
    return resolver(spec.repo_id)


# --------------------------------------------------------------------------- #
# 安装编排（后台线程 + 可轮询状态）
# --------------------------------------------------------------------------- #
def _idle_state() -> dict:
    return {
        "status": "idle",        # idle | running | success | cancelled | error
        "models": [],            # 本轮目标 model keys
        "current": None,         # 当前正在下的 key
        "current_file": None,    # 当前文件名（真实进度的最小单位）
        "file_index": 0,
        "file_total": 0,
        "percent": 0.0,
        "message": "",
        "error": None,
        "started_at": None,
        "finished_at": None,
        "elapsed_s": 0.0,
    }


class ModelInstaller:
    """真实语义权重的安装器（进程内单例，`get_model_installer()` 获取）。

    线程模型：**一个**后台工作线程 + 一把状态锁；状态 dict 的读写都在锁内，
    请求线程只读（不阻塞下载）。取消用 `threading.Event`，工作线程在**每个文件间隙**检查
    ——4GB 下载不可能秒级中断，但能保证"下一个文件不会再开始"。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _idle_state()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    # ---------------- 只读查询 ----------------
    def task(self) -> dict:
        """安装任务的当前状态快照（含实时 elapsed_s）。"""
        with self._lock:
            snap = dict(self._state)
        if snap["started_at"]:
            end = snap["finished_at"] or time.time()
            snap["elapsed_s"] = round(end - snap["started_at"], 1)
        return snap

    def models(self) -> dict:
        """模型清单 + 本机安装状态 + 当前启用情况 + 磁盘余量（设置页首屏用）。"""
        from ..runtime_settings import get_state  # 惰性导入：避免与 settings 模块构成导入环

        rt = get_state()
        enabled = rt.get("rag_embedder") == "real"
        items = []
        for spec in MODEL_SPECS.values():
            path = probe_installed(spec)
            items.append({
                "key": spec.key,
                "repo_id": spec.repo_id,
                "label": spec.label,
                "approx_gb": spec.approx_gb,
                "installed": bool(path),
                "local_path": path,
            })
        all_installed = all(it["installed"] for it in items)
        if enabled:
            hint = "已启用真实语义检索。切换（尤其是从 real 切回）或新增文档后，**重建索引**才会生效。"
        elif all_installed:
            hint = ("权重已就绪，但当前仍是占位嵌入。请在上方把『嵌入模型』切到 real 并保存，"
                    "然后到文档列表执行**重建索引** —— 不重建则检索结果不会变。")
        else:
            hint = ("当前使用占位嵌入（零依赖、秒级，但**没有真实语义**）。"
                    "点『下载模型』获取真实语义能力（约 4.5GB，取决于网络可能需 10~60 分钟）。")
        return {
            "models": items,
            "all_installed": all_installed,
            "enabled": enabled,
            "embedder": rt.get("rag_embedder", "fake"),
            # ★ 目录不存在（新用户首装）时回退到最近的存在祖先，而不是给 None ——
            #   "磁盘够不够 4.5GB" 是首屏决策信息，给"未知"等于让用户盲点下载。
            "disk_free_gb": free_space_gb(),
            "hint": hint,
        }

    # ---------------- 动作 ----------------
    def install(self, keys: list[str] | None = None) -> dict:
        """启动后台安装。返回**立即快照**（不等下载）。

        - `keys` 为空 / None ⇒ 装全部；
        - 有校验：未知 key 直接 `ModelInstallError`（路由层转 400）；
        - 已有任务在跑 ⇒ `AlreadyRunning`（路由层转 409）。
        """
        wanted = list(MODEL_SPECS) if not keys else list(keys)
        unknown = [k for k in wanted if k not in MODEL_SPECS]
        if unknown:
            raise ModelInstallError(f"未知模型：{'、'.join(unknown)}（可选：{'、'.join(MODEL_SPECS)}）")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise AlreadyRunning("已有下载任务在进行中，请等待完成或先点『取消』")
            self._cancel = threading.Event()
            self._state = _idle_state()
            self._state.update(status="running", models=wanted, started_at=time.time(),
                               message="正在准备下载…")
            self._thread = threading.Thread(
                target=self._worker, args=(wanted, self._cancel),
                name="rag-model-installer", daemon=True,
            )
            self._thread.start()
        return self.task()

    def cancel(self) -> dict:
        """请求取消当前下载（下一个文件开始前生效）。"""
        with self._lock:
            if self._state["status"] != "running":
                raise ModelInstallError("当前没有正在进行的下载任务")
        self._cancel.set()
        return self.task()

    # ---------------- 内部 ----------------
    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    def _worker(self, keys: list[str], cancel: threading.Event) -> None:
        n = len(keys)
        try:
            for mi, key in enumerate(keys):
                spec = MODEL_SPECS[key]
                base = mi / n * 100.0
                self._set(current=key, current_file=None, file_index=0, file_total=0,
                          percent=round(base, 1), message=f"准备下载 {spec.label}…")
                self._download(spec, cancel, base=base, span=100.0 / n)
                self._set(percent=round((mi + 1) / n * 100.0, 1), current_file=None)
            self._set(status="success", percent=100.0, finished_at=time.time(),
                      current=None, message="下载完成。请在设置页启用真实语义检索并重建索引。")
        except InstallCancelled:
            self._set(status="cancelled", finished_at=time.time(), current=None,
                      error="已取消下载", message="已取消。已下载的部分会被缓存，下次继续。")
        except ModelInstallError as e:
            logger.warning("语义检索模型安装失败：%s", e)
            self._set(status="error", finished_at=time.time(), current=None,
                      error=str(e), message="下载失败")
        except Exception as e:  # noqa: BLE001  兜底：任何异常都不许让后台线程静默消失
            logger.exception("语义检索模型安装出现未预期异常")
            self._set(status="error", finished_at=time.time(), current=None,
                      error=f"{type(e).__name__}：{e}", message="下载失败")

    def _download(self, spec: RagModelSpec, cancel: threading.Event, *, base: float, span: float) -> None:
        """下载单个模型的全部目标文件（同步，跑在工作线程里）。"""
        ensure_hf_mirror()   # 国内镜像兜底（仅当 HF_ENDPOINT 未显式设置）
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as e:  # rag extra 未安装
            raise ModelInstallError(
                f"缺少 huggingface_hub 依赖（{e}）。服务端请执行 `uv sync --extra rag` 后重启。"
            ) from e

        # ① 磁盘余量：fail fast，别下到一半才炸
        check_free_space(spec.approx_gb)

        # ② 列文件：网络不可达时这里就失败，给出可读错误
        try:
            files = HfApi().list_repo_files(spec.repo_id, repo_type="model")
        except Exception as e:  # noqa: BLE001
            raise ModelInstallError(
                f"无法访问模型仓库 {spec.repo_id}（{type(e).__name__}）。"
                "请检查网络；国内建议设置 HF_ENDPOINT=https://hf-mirror.com 后重试。"
            ) from e
        wanted = plan_files(list(files), spec)
        if not wanted:
            raise ModelInstallError(f"仓库 {spec.repo_id} 中没有匹配到需要的文件（仓库结构可能已变）")

        # ③ 逐文件下载：可中断 / 可容错 / 断点续传（已缓存的直接返回本地路径）
        got: list[str] = []
        failures: list[str] = []
        for i, name in enumerate(wanted, 1):
            if cancel.is_set():
                raise InstallCancelled()
            self._set(current_file=name, file_index=i, file_total=len(wanted),
                      percent=round(base + span * (i - 1) / len(wanted), 1),
                      message=f"下载 {spec.label}（{i}/{len(wanted)}）")
            try:
                hf_hub_download(spec.repo_id, name)
                got.append(name)
            except Exception as e:  # noqa: BLE001  单个文件失败不整体 abort
                failures.append(f"{name}（{type(e).__name__}）")
                logger.warning("模型文件下载失败 %s/%s：%s", spec.repo_id, name, e)
        if cancel.is_set():
            raise InstallCancelled()

        # ④ 以"推理必需是否齐备"判定成败 —— 与 hf_local 的快照口径同源，
        #    这样 imgs/.DS_Store 之类的 403 不会让整次安装失败。
        missing = missing_required(spec, got)
        if missing:
            detail = f"；另有 {len(failures)} 个文件下载失败：{'、'.join(failures[:5])}" if failures else ""
            raise ModelInstallError(
                f"{spec.label} 安装不完整，缺少：{'、'.join(missing)}{detail}。"
                "请检查网络后重试（已下载的文件会被复用）。"
            )
        if failures:
            logger.info("模型 %s 安装成功，但有 %d 个非必需文件下载失败（已忽略）", spec.key, len(failures))


# --------------------------------------------------------------------------- #
# 进程内单例
# --------------------------------------------------------------------------- #
_INSTALLER: ModelInstaller | None = None
_INST_LOCK = threading.Lock()


def get_model_installer() -> ModelInstaller:
    global _INSTALLER
    with _INST_LOCK:
        if _INSTALLER is None:
            _INSTALLER = ModelInstaller()
        return _INSTALLER


def reset_model_installer() -> None:
    """丢弃单例（仅供测试：避免用例之间的任务状态串味）。"""
    global _INSTALLER
    with _INST_LOCK:
        _INSTALLER = None
