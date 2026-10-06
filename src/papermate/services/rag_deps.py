"""PaperMate · 真实语义检索的**重依赖**探测与安装（2026-10-06）

背景：为什么需要这个模块
----------------------
桌面客户端首启原本执行 ``uv sync --extra rag --frozen``，而 ``rag`` extra 里含
``torch==2.13.0+cu126``。**实测**（本机 venv 逐包统计，非估算）：

===========================  ==========
包                体积
===========================  ==========
torch 2.13.0+cu126      3997 MB   ← 其中 CUDA 库约 3.5GB（torch/lib 内 49 个文件）
pymupdf / jieba / faiss   107 MB   ← **核心路径必需**（PDF 解析 / 分词 / 向量索引）
基础依赖（FastAPI 等）    656 MB
===========================  ==========
首启合计                  4765 MB
===========================  ==========

即：**用户装完 App，为了用"假检索"，先要下 4.8GB** —— 与产品"装完即用、绝不静默下载
大体积"的原则冲突（该原则已在模型层贯彻：默认 fake + 显式下载）。

本模块的解法（**不改 pyproject.toml、不改 uv.lock**）
--------------------------------------------------
``uv sync`` 支持 ``--no-install-package``，因此把首启拆成两段：

1. **首启（轻）**：``uv sync --extra rag --frozen --no-install-package torch
   --no-install-package FlagEmbedding --no-install-package sentence-transformers``
   ⇒ 约 0.76GB，**核心功能（上传/解析/检索流程/假嵌入）完全可用**。
2. **用户点「准备运行环境」**：``uv sync --extra rag --frozen`` ⇒ 补装 4.0GB。

★ 为什么不用"新增 extra"拆分：那需要重新生成 ``uv.lock``（本机 PyPI 可达性未验证），
  而 ``--no-install-package`` **零改动**即可达成同一效果 —— 失败方向更安全。

★ 为什么**不需要动 Rust**：``uv.exe`` 与后端源码是**同级目录**
（``resources/{python,backend}``），后端能自己定位并调用，无需 Tauri IPC。

本模块只负责三件事（**纯函数优先，便于单测**）：
1. **探测**：重依赖是否可导入（可缓存、可失效）；
2. **定位**：捆绑的 ``uv.exe`` 与 venv 路径；
3. **执行**：阻塞式跑 ``uv sync``（由上层编排进后台线程）。
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: 真实语义检索才需要的重依赖。
#:
#: ★★ **必须区分「发行包名」与「导入名」** —— 这是两个不同命名空间，混用必然出错：
#:   - 发行包名（给 ``uv sync --no-install-package`` / ``pyproject.toml``）：
#:     ``sentence-transformers``（**连字符**）
#:   - 导入名（给 ``__import__`` / ``from ... import``）：
#:     ``sentence_transformers``（**下划线**）
#:   实测踩到：曾把两者混成一个列表，导致开发机上明明装好了却被判为"缺失"，
#:   界面会骗用户去重下 4GB 已有的东西。
HEAVY_DISTRIBUTIONS: tuple[str, ...] = ("torch", "FlagEmbedding", "sentence-transformers")
HEAVY_IMPORTS: tuple[str, ...] = ("torch", "FlagEmbedding", "sentence_transformers")

#: 探测结果缓存（``import torch`` 首次要 1~2s，不能每次请求都付）
_CACHE: dict | None = None


@dataclass
class DepsStatus:
    """重依赖就绪状态。"""

    ready: bool
    missing: list[str] = field(default_factory=list)
    #: 无法探测时为 True（例如 import 抛了非 ImportError 的异常）
    unknown: bool = False
    torch_cuda: bool | None = None
    torch_version: str | None = None

    def as_dict(self) -> dict:
        return {
            "ready": self.ready,
            "missing": list(self.missing),
            "unknown": self.unknown,
            "torch_cuda": self.torch_cuda,
            "torch_version": self.torch_version,
        }


def _import_ok(name: str) -> tuple[bool, str | None]:
    """尝试 import，返回 (是否成功, 异常摘要)。"""
    try:
        __import__(name)
        return True, None
    except ImportError as e:  # 未安装是最常见情况
        return False, f"未安装（{e}）"
    except Exception as e:  # noqa: BLE001  装了但坏了（版本冲突/二进制不兼容）也算缺失
        return False, f"导入失败（{type(e).__name__}: {e}）"


def probe(*, use_cache: bool = True) -> DepsStatus:
    """探测重依赖是否就绪。

    ★ 必须区分"没装"与"装了但导入失败"：后者（如 torch 与 CUDA DLL 不匹配）是**真故障**，
    不能被当成"未安装 ⇒ 去下载吧"从而反复重试 4GB。两种都归入 ``missing``，
    但 ``missing`` 里的描述不同，便于日志与提示区分。
    """
    global _CACHE
    if use_cache and _CACHE is not None:
        return _CACHE

    missing: list[str] = []
    for name in HEAVY_IMPORTS:
        ok, detail = _import_ok(name)
        if not ok:
            missing.append(f"{name}：{detail}" if detail else name)

    cuda = None
    ver = None
    if not any(m.startswith("torch：") for m in missing):
        try:
            import torch  # 惰性：只有 torch 真的在时才 import

            cuda = bool(torch.cuda.is_available())
            ver = getattr(torch, "__version__", None)
        except Exception as e:  # noqa: BLE001
            logger.info("torch 已安装但探测 CUDA 失败：%s", e)

    st = DepsStatus(ready=not missing, missing=missing, torch_cuda=cuda, torch_version=ver)
    _CACHE = st
    return st


def deps_ready(*, use_cache: bool = True) -> bool:
    """重依赖是否齐备（快捷方式）。"""
    return probe(use_cache=use_cache).ready


def invalidate_cache() -> None:
    """装完依赖后必须调用，否则进程内会一直用旧的"缺失"结论。"""
    global _CACHE
    _CACHE = None


# --------------------------------------------------------------------------- #
# 定位：uv.exe 与 venv
# --------------------------------------------------------------------------- #
def backend_root() -> Path:
    """捆绑后端副本的根目录（``resources/backend``）。

    ``parents`` 依赖：``[0]``=services `[1]`=papermate `[2]`=src `[3]`=backend。
    ★ `app.py` 托管 SPA 时也用 ``parents[3] / "web" / "dist"``，两处口径必须一致。
    """
    return Path(__file__).resolve().parents[3]


def find_uv() -> Path | None:
    """定位捆绑的 ``uv.exe``。

    优先级：
      1. 环境变量 ``PAPERMATE_UV_EXE``（运维/测试可显式指定）；
      2. **捆绑位置** ``<backend>/../python/uv.exe``（桌面客户端布局）；
      3. PATH 上的 ``uv``（开发机：``uv sync`` 流程本身就用它）。
    """
    env = os.environ.get("PAPERMATE_UV_EXE", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    bundled = backend_root().parent / "python" / "uv.exe"
    if bundled.is_file():
        return bundled
    from shutil import which

    found = which("uv")
    return Path(found) if found else None


def venv_dir() -> Path | None:
    """本应用应当安装依赖的 venv 目录。

    优先级：
      1. ``UV_PROJECT_ENVIRONMENT``（``lib.rs`` 引导 uv 时用的同一个变量）；
      2. ``<data_dir 的上级>/venv`` —— 对应 ``lib.rs`` 的
         ``local_data/venv`` 与 ``local_data/data`` 布局；
      3. ``None`` —— 解析不出时**如实降级**（见 :func:`describe_target`）。
    """
    env = os.environ.get("UV_PROJECT_ENVIRONMENT", "").strip()
    if env:
        return Path(env)
    try:
        from ..config import get_settings

        data_dir = Path(get_settings().resolved_data_dir)
    except Exception:  # noqa: BLE001  配置不可用时视为解析不出
        return None
    cand = data_dir.parent / "venv"
    return cand if cand.exists() else None


def python_exe_for_venv(venv: Path) -> Path:
    """venv 内的解释器路径（Windows / POSIX 都覆盖）。"""
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def describe_target() -> dict:
    """给前端/日志用：当前能否在本机安装重依赖，以及原因。

    ★ 「解析不出 venv」必须**明确告诉用户**，不能静默失败后让人以为点了没用。
    """
    uv = find_uv()
    venv = venv_dir()
    return {
        "uv_found": uv is not None,
        "uv_path": str(uv) if uv else None,
        "venv_found": venv is not None,
        "venv_path": str(venv) if venv else None,
        "in_bundled_layout": (backend_root().parent / "python").is_dir(),
        "reason": (
            None if (uv and venv)
            else ("未找到捆绑的 uv.exe（请确认安装包完整，或设置 PAPERMATE_UV_EXE）" if not uv
                  else "未找到应用私有 venv（不在桌面端安装环境中，无法自动安装依赖）")
        ),
    }


# --------------------------------------------------------------------------- #
# 执行：阻塞式 uv sync（由上层编排进后台线程）
# --------------------------------------------------------------------------- #
class DepsInstallError(RuntimeError):
    """依赖安装失败（消息可直接展示给用户）。"""


def sync_heavy_deps(
    *,
    log: str | None = None,
    cancel=None,
    timeout: int = 3600,
) -> str:
    """执行 ``uv sync --extra rag --frozen`` 补装重依赖（**阻塞**，约 4GB / 数分钟）。

    与首启的"轻量 sync"是**同一份 lockfile 的两种选择**，因此结果一致、可重复执行；
    已安装的包会被 uv 跳过，所以重复调用是廉价且幂等的。

    返回 uv 的输出尾部（便于前端展示）；失败抛 :class:`DepsInstallError`。
    """
    if deps_ready():
        return "重依赖已就绪，跳过安装"

    target = describe_target()
    if not target["uv_found"]:
        raise DepsInstallError(target["reason"] or "未找到 uv.exe，无法安装依赖")
    if not target["venv_found"]:
        raise DepsInstallError(target["reason"] or "未找到应用私有 venv，无法安装依赖")

    uv, venv = Path(target["uv_path"]), Path(target["venv_path"])
    proj = backend_root()
    py = python_exe_for_venv(venv)
    if not py.is_file():
        raise DepsInstallError(f"venv 解释器不存在：{py}（venv 可能未初始化完成）")

    args = [
        str(uv), "sync",
        "--extra", "rag",
        "--frozen",                      # 严格按 uv.lock，不重新解析版本
        "--python", str(py),
    ]
    env = {
        "UV_PROJECT_ENVIRONMENT": str(venv),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        "UV_HTTP_TIMEOUT": "600",         # 单个 wheel 较大，默认 30s 会超时
        "UV_HTTP_LOW_SPEED_LIMIT": "10",
    }
    logger.info("开始补装重依赖：%s", " ".join(args))
    try:
        proc = subprocess.run(  # noqa: S603  参数全为本地绝对路径，无 shell
            args,
            cwd=str(proj),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise DepsInstallError(f"依赖安装超时（>{timeout}s）") from e
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[-600:]
        raise DepsInstallError(f"依赖安装失败（uv 退出码 {proc.returncode}）：{tail}")
    if cancel is not None and cancel.is_set():
        raise DepsInstallError("已取消")

    invalidate_cache()
    st = probe(use_cache=False)
    if not st.ready:
        raise DepsInstallError(f"安装命令已结束，但依赖仍不可导入：{'、'.join(st.missing)}")
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return out[-800:] if out else (log or "依赖安装完成")
