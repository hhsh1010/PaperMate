"""重模型加载的**跨进程互斥**（G1）。

## 为什么需要
`bge-m3` + `reranker` 各约 **2.3GB**；两个进程同时构造会互抢同一张 8GB 显存 ⇒ **双方死锁**
（实测：CPU 0%、GPU 占满、10 分钟零产出、进程 pid 不断变化 —— 是**死锁不是慢**）。
而现有的管线缓存 `document_service._pipelines` 只在**进程内**有效 ⇒ 拦不住跨进程。

实测这类构造占冷启动的 **92%**（`BgeM3Embedder()` 12.7~18.3s + `CrossEncoderReranker()` 4.6~5.1s）
⇒ 临界区约 **17~23 秒**，值得为它加互斥。

## 设计取舍（每条都有理由）
1. **用 OS 级文件锁**（`msvcrt.locking` / `fcntl.flock`）：**持有者崩溃时锁自动释放**
   ⇒ 不需要"僵尸锁"探测这一整类新问题（自研 pid 文件方案会引入它）。
2. **只守"加载窗口"，不守每次推理** ⇒ 不会把产品退化成串行。
3. **等待有上限**（默认 120s，同量级于工具超时），**超时显式报错**而不是无限等
   —— 否则用户看到"卡住"却没有任何解释。
4. ⚠️ **调用方必须在工作线程里等**（`asyncio.to_thread`）：在事件循环里同步等会冻结整个后端
   （本项目 1 号事故：同步重计算跑在 async 里导致后端冻结）。
   现有的模型构造点已经满足这一点（`document_service.py` 注明"必须走 to_thread"）。
5. 锁文件放**数据目录**（`PAPERMATE_DATA_DIR`，默认 `data/`），**不放仓库根**
   ⇒ 多实例/多用户互不干扰。

## 边界（明确不做）
- 不做**请求级限流**（那是服务的职责，本地单用户非刚需）；
- 不做**跨机器的分布式锁**（本项目的定位是本地优先单用户）。
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

#: 锁文件名（放数据目录；以 `.` 开头避免出现在用户可见的文件列表里）
_LOCK_NAME = ".model_load.lock"

#: 轮询间隔（秒）：临界区是分钟级，0.2s 足够，也不至于让等待方抢得太凶
_POLL_S = 0.2

#: 默认等待上限（秒）——可用 `PAPERMATE_MODEL_LOCK_WAIT_S` 覆盖；<=0 表示"不等待，抢不到就报错"
_DEFAULT_WAIT_S = 120.0


class ModelBusyError(RuntimeError):
    """另一个进程正在加载重模型，且本进程等待超时。

    刻意**不吞掉**：调用方应把它转成用户可读的错误（"另一个实例正在加载模型，请稍后重试"），
    而不是静默重试或无限等待 —— 静默会退化成"看起来在跑、实际卡住"。
    """


def lock_path() -> Path:
    """锁文件路径：**数据目录**下（不依赖 papermate 层，故直接读环境变量）。"""
    data_dir = Path(os.environ.get("PAPERMATE_DATA_DIR") or "data")
    return data_dir / _LOCK_NAME


def wait_seconds() -> float:
    """等待上限（秒）。非法/空值回落到默认值（**绝不因为配置写错就变成无限等待**）。"""
    raw = os.environ.get("PAPERMATE_MODEL_LOCK_WAIT_S", "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_WAIT_S


def _try_once(fd: int) -> bool:
    """尝试一次非阻塞抢锁；成功 True，被别人占用 False。"""
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def acquire(path: Path | None = None, timeout_s: float | None = None) -> int | None:
    """阻塞抢锁至拿到或超时。返回 fd（**须持有到不再需要**）；超时返回 None。

    ★ 只能在**工作线程**里调用（见模块 docstring 第 4 条）。
    """
    p = path or lock_path()
    budget = wait_seconds() if timeout_s is None else timeout_s
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:  # 数据目录不可写等 → 明确失败，别静默放行
        raise ModelBusyError(f"无法创建模型加载锁目录 {p.parent}：{e}") from e
    fd = _open_with_retry(p)

    deadline = time.monotonic() + max(budget, 0.0)
    waited = False
    while True:
        if _try_once(fd):
            _write_owner(fd)
            if waited:
                logger.info("[模型锁] 已获得加载权（曾等待 %.1fs）", time.monotonic() - (deadline - budget))
            return fd
        if not waited:
            waited = True
            logger.warning(
                "[模型锁] 另一个进程正在加载重模型，本进程最多等待 %.0fs（锁=%s）", budget, p
            )
        if time.monotonic() >= deadline:
            os.close(fd)
            return None
        time.sleep(_POLL_S)


#: 打开锁文件的重试预算（秒）
_OPEN_RETRY_S = 3.0


def _open_with_retry(p: Path) -> int:
    """打开锁文件；对**瞬时**占用做有限重试。

    ★ 为什么需要（真机验收抓出来的）：Windows 上杀软/搜索索引器会扫刚创建的文件，
    此时 `os.open(O_RDWR)` 会瞬时返回 `EACCES`。原先"一次失败就抛错"会让**正常启动被误杀**；
    而无限重试又会退化成"卡住"。故：**有限重试 + 明确失败**。
    """
    deadline = time.monotonic() + _OPEN_RETRY_S
    while True:
        try:
            return os.open(str(p), os.O_CREAT | os.O_RDWR)
        except OSError as e:
            if time.monotonic() >= deadline:
                raise ModelBusyError(
                    f"无法打开模型加载锁文件 {p}（重试 {_OPEN_RETRY_S:.0f}s 仍失败）：{e}"
                ) from e
            time.sleep(_POLL_S)


def _write_owner(fd: int) -> None:
    """把持有者 pid 写进锁文件（**尽力而为**）：只为让"是谁占着"可排查。

    只在偏移 1 之后写 —— 第 0 字节是**被锁定的区间**，不能动。
    任何失败都忽略：诊断信息不值得让加载失败。
    """
    try:
        os.lseek(fd, 1, os.SEEK_SET)
        os.write(fd, f"pid={os.getpid()}\n".encode())
    except OSError:
        pass


def read_owner(path: Path | None = None) -> str:
    """读持有者信息（供错误文案/排查用）；读不到返回空串。

    ⚠️ 必须**从偏移 1 开始读**：第 0 字节是被锁定的区间，Windows 下**他人连读都会被拒**
    （实测：用 `read_bytes()` 从 0 读 ⇒ `PermissionError` ⇒ 永远拿不到持有者信息）。
    """
    try:
        with open(path or lock_path(), "rb") as f:
            f.seek(1)  # ★ 跳过锁定字节
            return f.read().decode("utf-8", "ignore").strip()
    except OSError:
        return ""


def release(fd: int) -> None:
    """释放锁：**关闭 fd 即释放**（OS 语义，无需显式解锁）。"""
    with contextlib.suppress(OSError):
        os.close(fd)


@contextlib.contextmanager
def hold_model_load(label: str = "model", *, timeout_s: float | None = None,
                    path: Path | None = None) -> Iterator[None]:
    """`with` 形式：进入临界区时抢锁，退出时释放；超时抛 `ModelBusyError`。

    ★ 只能在**工作线程**里用（`asyncio.to_thread`）。
    """
    fd = acquire(path, timeout_s)
    if fd is None:
        owner = read_owner(path)
        raise ModelBusyError(
            f"另一个进程正在加载{label}（等待已超 {wait_seconds():.0f}s"
            + (f"，持有者 {owner}" if owner else "")
            + "）。请等待其完成，或关闭其它 PaperMate 实例后重试。"
        )
    try:
        yield
    finally:
        release(fd)
