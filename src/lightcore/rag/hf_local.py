"""LightCore · HF 本地缓存解析（离线 / 镜像受限环境）

国内网络下 ``huggingface.co`` 直连会超时（默认 10s × 5 次重试 ≈ 50s 卡顿），
而 ``hf-mirror`` 镜像对部分仓库文件（如 ``BAAI/bge-m3`` 的 ``imgs/.DS_Store``）
返回 403，会让 ``snapshot_download`` 在最后一个文件上整批 abort。

但推理真正需要的文件（``config.json`` + 权重 + ``tokenizer*``）**往往已经**落在
hub 缓存快照里。本模块把该快照目录解析成本地路径——transformers / FlagEmbedding
对本地目录直接 ``from_pretrained``，**不会发起任何网络请求**。

⚠️ 缓存根目录的解析**必须**走 ``default_hub_cache_dir()``（读 ``HF_HUB_CACHE`` / ``HF_HOME``），
不可硬编码 ``~``：桌面端把 ``HF_HOME`` 指向 ``%LOCALAPPDATA%``，硬编码会导致
「下载在一处、探测在另一处」的静默失效（2026-10-06 首装仿真实测发现）。

用途：让 BGE-M3 嵌入器与 BGE-reranker 重排器都走「本地优先」，彻底摆脱下载/超时。
"""

from __future__ import annotations

import glob
import logging
import os

logger = logging.getLogger(__name__)

#: 权重文件候选名（命中其一即认为权重就绪）。分片权重用 index.json 表示。
_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")

#: 国内可用的 HF 镜像；仅在 HF_ENDPOINT 未设置时作为兜底，避免直连被墙域名干等超时。
_DEFAULT_HF_MIRROR = "https://hf-mirror.com"


def ensure_hf_mirror() -> None:
    """``HF_ENDPOINT`` 未设置时兜底指向国内镜像，避免直连 huggingface.co 超时。

    已显式设置（含设为空以外的值）时**不做任何改动**，尊重用户/运维配置。
    """
    if os.environ.get("HF_ENDPOINT", "").strip():
        return
    os.environ["HF_ENDPOINT"] = _DEFAULT_HF_MIRROR
    logger.info("HF_ENDPOINT 未设置，已兜底为国内镜像 %s", _DEFAULT_HF_MIRROR)


def auto_device() -> str:
    """探测推理设备：env ``PAPERMATE_DEVICE`` 显式指定 > torch.cuda 可用则 ``cuda`` > ``cpu``。

    torch 未安装 / 探测失败一律安全回退 ``cpu``（嵌入/重排在无 torch 环境本就不可用，
    此处不抛错以保证调用方可统一走探测-降级路径）。
    """
    env = os.environ.get("PAPERMATE_DEVICE", "").strip().lower()
    if env:
        return env
    try:
        import torch  # 惰性导入：非 rag 环境不引入重依赖

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001  探测属加速手段，失败不影响主流程
        return "cpu"


def default_hub_cache_dir() -> str:
    """HF **hub 缓存**目录（与 ``huggingface_hub`` 自身的解析口径一致）。

    优先级：
      1. ``HF_HUB_CACHE``（最具体，huggingface_hub 的第一顺位）；
      2. ``HF_HOME`` + ``hub``（huggingface_hub 的默认布局）；
      3. ``~/.cache/huggingface/hub``（两者都未设时的默认）。

    ★★ 2026-10-06：这三个 env **必须**被尊重，不能硬编码 ``~``。
       桌面客户端把 ``HF_HOME`` 指向 ``%LOCALAPPDATA%/PaperMate/huggingface``（见
       ``desktop/src-tauri/src/lib.rs``），而实际下载也落在那里。若此处仍去 ``~`` 下找，
       就会发生**「下载在一处、探测在另一处」** ⇒ 设置页误报"权重已就绪"，用户启用 real
       后却加载失败 —— 属于典型的**静默失效**，且在开发机（有 ``~`` 缓存）上完全测不出来。
    """
    hub = os.environ.get("HF_HUB_CACHE", "").strip()
    if hub:
        return os.path.abspath(os.path.expanduser(hub))
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return os.path.join(os.path.abspath(os.path.expanduser(home)), "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def resolve_local_hf_snapshot(repo_id: str, env_var: str | None = None) -> str | None:
    """解析 ``repo_id`` 在本机 HF hub 缓存中的完整快照目录，找不到返回 ``None``。

    优先级：
      1. 环境变量 ``env_var`` 显式指定的目录（需含 ``config.json``）；
      2. ``default_hub_cache_dir()`` 下 ``models--<org>--<name>/snapshots/*`` 的
         **最新**且「config.json + 权重」齐备的快照。

    ``env_var`` 指向的目录不合法时只告警并继续探测缓存，不抛异常（不阻断加载）。
    """
    if env_var:
        explicit = os.environ.get(env_var, "").strip()
        if explicit:
            if os.path.isfile(os.path.join(explicit, "config.json")):
                logger.info("HF 模型 %s：使用环境变量 %s 指定的本地目录 %s", repo_id, env_var, explicit)
                return explicit
            logger.warning("环境变量 %s=%s 不是含 config.json 的模型目录，忽略并回退缓存探测", env_var, explicit)

    safe = repo_id.replace("/", "--")
    root = os.path.join(default_hub_cache_dir(), f"models--{safe}", "snapshots")
    if not os.path.isdir(root):
        logger.info("HF 模型 %s：无 hub 缓存目录 %s（需联网下载）", repo_id, root)
        return None
    for d in sorted(glob.glob(os.path.join(root, "*")), reverse=True):
        if not os.path.isfile(os.path.join(d, "config.json")):
            continue
        if any(os.path.isfile(os.path.join(d, w)) for w in _WEIGHT_FILES):
            logger.info("HF 模型 %s：命中本地缓存快照 %s", repo_id, d)
            return d
    logger.warning("HF 模型 %s：%s 下无完整快照（缺 config.json 或权重），将尝试联网", repo_id, root)
    return None
