"""图片增强（Level 2）依赖的**按需安装**（设置页可选下载）。

## 为什么需要一条独立通道
`pyproject.toml` 把 OCR / 表格结构所需的重依赖放在**独立的 `[figure]` extra**
（`paddleocr` + `paddlepaddle` + `transformers` + `torch` + `pillow`），注释写明
「与 rag 分离，避免拖重核心 RAG 安装；启用 figure_enrich 前才需安装」。
而此前**应用内没有任何安装入口**（`rag_deps` 只管 rag）⇒ 用户把「图片增强」打开后
OCR 永远不生效（2026-10-09 实测反馈：设置页显示"OCR 未装"且无从安装）。

## 与 `rag_deps` 的关系（同源，不重造）
- **复用**其 uv / venv / 缓存解析（`find_uv` / `venv_dir` / `python_exe_for_venv` /
  `uv_cache_dir` / `backend_root` / `describe_target`）—— 两条安装通道必须**同一口径**。
- **区别**：只有"装包"一段（没有模型下载段）；调用的 extra 不同。

## ★★ `--inexact` 不是可选项（血泪，两次踩过）
`uv sync` 默认是**精确模式**：会删除"不在目标期望集里的包"。本命令的期望集只含
`base + figure` ⇒ **rag 的重依赖（faiss / FlagEmbedding / sentence-transformers）会成为
"多余包"被静默剪掉** ⇒ 用户的真实语义检索直接失效（2026-10-07 与 10-08 两度踩坑的同一机制）。
`--inexact`（官方帮助原文：*Do not remove extraneous packages present in the environment*）
是唯一安全阀；`tests/test_figure_deps.py` 用静态断言把它**钉死**，防止后人"顺手删掉看起来多余的参数"。

## 探测口径（三能力）
`probe()` 是**唯一实现**（`runtime_settings._enrich_deps` 委托到它，消除重复）：
- `ocr`   ← `import paddleocr`
- `table` ← `import torch` + `PIL.Image` + `transformers.AutoImageProcessor`
- `vlm`   ← 恒 `True`（走已配置的 LLM 通道；是否真能用由「模型连接」的密钥决定）
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from . import rag_deps

logger = logging.getLogger(__name__)


class FigureDepsError(RuntimeError):
    """图片增强依赖安装失败（消息可直接展示给用户）。"""


class AlreadyRunning(RuntimeError):
    """已有安装任务在跑。"""


def probe() -> dict:
    """探测图片增强三能力是否可用（**每次现场 import**，装完即变 True，无需重启）。

    ⚠️ 不要缓存结果：import 失败**不会**被 Python 记住（`sys.modules` 只存成功的），
    所以装完再调即可拿到新值。缓存反而会让"装完仍显示未装"（那正是 2026-10-09 实测到的
    另一处缺陷——前端把探测结果随设置缓存了，见 `docs/_internal/待批量改动-队列.md` P-06）。
    """
    deps = {"ocr": False, "table": False, "vlm": True}
    try:
        from paddleocr import PaddleOCR  # noqa: F401

        deps["ocr"] = True
    except Exception:  # noqa: BLE001  缺依赖/装坏了都算"不可用"
        pass
    try:
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
        from transformers import AutoImageProcessor  # noqa: F401

        deps["table"] = True
    except Exception:  # noqa: BLE001
        pass
    return deps


def needed() -> bool:
    """是否还需要安装（OCR 与表格结构缺任一即需要）。"""
    d = probe()
    return not (d["ocr"] and d["table"])


def _extra_args() -> list[str]:
    """uv 参数（★ 单独成函数以便静态契约测试逐项断言）。"""
    return [
        "sync",
        "--extra", "figure",
        "--frozen",  # 严格按 uv.lock（图 3.7.0/paddlepaddle 等均已锁定）
        # ★★ 见模块 docstring：少了它就会把 rag 重依赖静默剪掉
        "--inexact",
    ]


def sync_figure_deps(*, cancel=None, timeout: int = 3600) -> str:
    """执行 ``uv sync --extra figure --frozen --inexact``（**阻塞**）。

    ★ 体积/耗时（2026-10-09 开发版实测，非估算）：
      - **下载 ≈ 150~250MB**（wheel 压缩包；paddlepaddle 单个 99.9MB）
      - **落盘 ≈ 559MB**（venv 增量；`paddle` 368MB + `cv2` 122MB 占大头；uv-cache 同步 559MB）
      - **耗时 73.6s**（25 个包，含 1 次 editable 重装 papermate）
    ⚠️ 若尚未「准备运行环境（rag）」，同一次命令还会补 torch 等（+约 4GB）——
       figure 与 rag 共享 torch，故 UI 文案必须提示这一点。

    已装则直接返回（幂等、廉价）；失败抛 :class:`FigureDepsError`（消息含 uv 输出尾部）。
    """
    if not needed():
        return "图片增强依赖已就绪，跳过安装"

    target = rag_deps.describe_target()
    if not target["uv_found"]:
        raise FigureDepsError(target["reason"] or "未找到 uv.exe，无法安装依赖")
    if not target["venv_found"]:
        raise FigureDepsError(target["reason"] or "未找到应用私有 venv，无法安装依赖")

    uv, venv = Path(target["uv_path"]), Path(target["venv_path"])
    py = rag_deps.python_exe_for_venv(venv)
    if not py.is_file():
        raise FigureDepsError(f"venv 解释器不存在：{py}（venv 可能未初始化完成）")

    args = [str(uv), *_extra_args(), "--python", str(py)]
    env = {
        "UV_PROJECT_ENVIRONMENT": str(venv),
        "UV_HTTP_TIMEOUT": "600",  # 单个 wheel 较大，uv 默认 30s 会超时
        "UV_HTTP_LOW_SPEED_LIMIT": "10",
    }
    cache = rag_deps.uv_cache_dir()
    if cache is not None:
        env["UV_CACHE_DIR"] = str(cache)

    logger.info("开始安装图片增强依赖：%s", " ".join(args))
    try:
        proc = subprocess.run(  # noqa: S603  参数全为本地绝对路径，无 shell
            args,
            cwd=str(rag_deps.backend_root()),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise FigureDepsError(f"依赖安装超时（>{timeout}s）") from e
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[-600:]
        raise FigureDepsError(f"依赖安装失败（uv 退出码 {proc.returncode}）：{tail}")
    if cancel is not None and cancel.is_set():
        raise FigureDepsError("已取消")

    # ★ 能力探测为判据（不认退出码）：uv 退出 0 ≠ 真的 import 得到
    #   （wheel 与本机不兼容、下载被截断都会"装成功"却不可用）
    after = probe()
    if not after["ocr"]:
        raise FigureDepsError(
            "安装命令已结束，但 `import paddleocr` 仍失败 ⇒ 请把 uv 输出尾部反馈（见 app.log）"
        )
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    logger.info("图片增强依赖安装完成：ocr=%s table=%s", after["ocr"], after["table"])
    return out[-800:] if out else "图片增强依赖安装完成"


# --------------------------------------------------------------------------- #
# 安装编排（后台线程 + 可轮询状态）——骨架照 `model_installer.ModelInstaller`
# --------------------------------------------------------------------------- #
def _idle_state() -> dict:
    return {
        "status": "idle",  # idle | running | success | cancelled | error
        "message": "",
        "error": None,
        "started_at": None,
        "finished_at": None,
        "elapsed_s": 0.0,
        #: 安装结束后的三能力快照（前端据此刷新"已装/未装"提示）
        "deps": None,
    }


class FigureInstaller:
    """图片增强依赖安装器（进程内单例）。

    线程模型：**一个**后台工作线程 + 一把状态锁；请求线程只读不阻塞。
    取消用 `threading.Event`：`subprocess.run` 是阻塞调用 ⇒ 取消只在**命令返回后**生效
    （与 `rag_deps.sync_heavy_deps` 一致，属既有取舍）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _idle_state()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    def task(self) -> dict:
        with self._lock:
            snap = dict(self._state)
        if snap["started_at"]:
            end = snap["finished_at"] or time.time()
            snap["elapsed_s"] = round(end - snap["started_at"], 1)
        return snap

    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    def install(self) -> dict:
        """启动后台安装（已在跑 ⇒ 抛 :class:`AlreadyRunning`）。"""
        with self._lock:
            if self._state["status"] == "running":
                raise AlreadyRunning("图片增强依赖正在安装中（请等待完成或先取消）")
            self._cancel.clear()
            self._state = _idle_state()
            self._state.update(status="running", started_at=time.time(),
                               message="正在安装图片增强依赖（paddleocr / 表格结构）…")
        self._thread = threading.Thread(target=self._worker, name="figure-deps-install", daemon=True)
        self._thread.start()
        return self.task()

    def cancel(self) -> dict:
        with self._lock:
            if self._state["status"] != "running":
                raise AlreadyRunning("当前没有正在进行的安装任务")
        self._cancel.set()
        self._set(message="正在取消…（当前命令返回后生效）")
        return self.task()

    def _worker(self) -> None:
        try:
            sync_figure_deps(cancel=self._cancel)
        except FigureDepsError as e:
            logger.warning("图片增强依赖安装失败：%s", e)
            self._set(status="error", error=str(e), finished_at=time.time(),
                      message="图片增强依赖安装失败", deps=probe())
            return
        except Exception as e:  # noqa: BLE001  兜底：后台线程不许静默消失
            logger.exception("图片增强依赖安装出现未预期异常")
            self._set(status="error", error=f"{type(e).__name__}：{e}", finished_at=time.time(),
                      message="图片增强依赖安装失败", deps=probe())
            return
        if self._cancel.is_set():
            self._set(status="cancelled", finished_at=time.time(),
                      message="已取消。已下载的部分会被缓存，下次继续。", deps=probe())
            return
        self._set(status="success", finished_at=time.time(),
                  message="图片增强依赖安装完成。请重新上传文档以对图片生效。", deps=probe())


_INSTALLER: FigureInstaller | None = None
_INSTALLER_LOCK = threading.Lock()


def get_figure_installer() -> FigureInstaller:
    global _INSTALLER
    with _INSTALLER_LOCK:
        if _INSTALLER is None:
            _INSTALLER = FigureInstaller()
        return _INSTALLER
