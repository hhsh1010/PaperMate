"""带**单实例锁**的 pytest 启动器（T-10，2026-09-30）。

## 为什么需要它
本开发环境的**沙箱会重放 Bash 命令**（同一条命令执行两遍）。而 pytest-xdist
每个 worker 实测占 **≈960MB** ⇒ 两套 worker 同时跑约 15GB ⇒ 内存耗尽：
- 后起的 worker 直接起不来（`EOFError: expected 1 bytes, got 0`）
- 已起的 worker 崩在 `ACCESS_DENIED`（`0x80070005`）/ `stack overflow`
- ★ 且**遗留孤儿进程**（实测遗留 8 个 960MB 进程，必须手工清理）

⇒ 用一把**跨进程文件锁**兜住：第二个实例发现锁被占用就**立刻退出**，
不启动任何 worker、不占任何内存。

## 用法（等价于 `python -m pytest`，参数原样透传）
    python scripts/run_tests.py                     # 全量（含 slow）
    python scripts/run_tests.py -m "not slow"       # 日常门禁
    python scripts/run_tests.py tests/test_api.py -q
    python scripts/run_tests.py -n 0                # 退回串行（排障 flaky）
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: ★ 锁路径用**仓库内固定路径**（2026-09-30）。
#:
#: 原实现放在 `tempfile.gettempdir()`（系统临时目录）—— 同一 Bash 调用里并发两个实例时实测有效
#: （第二个被拒 `PermissionError`）。但**「turn 重放」产生的另一个实例锁不住**（怀疑其 TEMP
#: 解析到别处 ⇒ 各锁各的文件）。两个实例改的是**同一个仓库**（已实测：另一个实例创建的文件
#: 会出现在本实例的仓库视图里），所以把锁放进仓库根目录能**确定**撞上同一把锁。
#: （锁文件是 0~1 字节的常驻小文件；字节锁在进程退出时自动释放，**残留不会阻塞**后续运行。）
#:
#: ⚠️ **不要**用 `bringing up nodes...` 的出现次数判断"有几个实例"：实测**单次运行就会打 2 次**
#: （2026-09-30 自证：一次 `-n 3` 小规模运行 = 2 次该行 + **仅 1 条汇总行**）。
#: 真判据：同一日志出现**两条汇总行**（`N passed`）／提示 `检测到已有 pytest 实例在跑`／
#: 出现**自己没创建过的文件**。
_LOCK_PATH = str(Path(__file__).resolve().parents[1] / ".pytest_instance.lock")


def _try_lock(path: str) -> int | None:
    """非阻塞地抢锁；成功返回 fd（需持有到 pytest 结束），失败返回 None。"""
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _other_pytest_running() -> bool:
    """★ **跨上下文**实例检测（2026-09-30 新增，文件锁的兜底）。

    为什么光靠文件锁不够：本环境的「turn 重放」实例可能拥有**独立的文件系统视图**
    （同一路径、不同物理文件）⇒ 两把"同路径"的锁互不可见 ⇒ 都放行。
    实测证据：并发期间进程表里出现 **2 个 `run_tests.py` + 12 个 xdist worker**，
    且 CPU 仅 **~30%**（在等 GPU/IO）⇒ 全量从 `-n 3` 的 ~115s 退化到 **412s**。

    **进程表是全局的**，不受文件视图影响 ⇒ 用它兜底。判据是 cmdline 里的
    `run_tests.py`（另一个启动器）或 `-m pytest`（直跑的 pytest）。

    ### 2026-10-01 修正（实测踩到）：**必须限定 python 进程 + 排除祖先链**

    曾出现"环境明明静默却 `RESULT=SKIPPED`"——根因：**启动本脚本的 bash 包装进程**
    的 cmdline 里同样含有 `run_tests.py`（`... bash -c "... python scripts/run_tests.py ..."`），
    旧实现只排除了自身 pid、没排除**祖先**，也没限定进程名 ⇒ **把自己拦下了**。
    ⇒ ① 只统计**名字含 python** 的进程（真正的 pytest / 启动器都是 python，
      bash / pwsh 包装进程不再误伤）；② 排除自身与**全部祖先**。
    """
    try:
        import psutil
    except ImportError:  # psutil 是开发依赖；缺了也不该让启动器挂掉
        return False
    me = os.getpid()

    # 排除自身 + 全部祖先（父链上的 bash/pwsh 的 cmdline 含 "run_tests.py" 字样）
    excluded: set[int] = {me}
    try:
        cur = psutil.Process(me)
        while True:
            cur = cur.parent()  # type: ignore[assignment]
            if cur is None:
                break
            excluded.add(cur.pid)
    except psutil.Error:  # 链条断裂不影响主流程（还有一个 python 名字过滤兜着）
        pass

    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in excluded:
                continue
            name = (p.info["name"] or "").lower()
            if "python" not in name:  # bash/pwsh/cmd 包装进程不是 pytest，跳过
                continue
            c = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "run_tests.py" in c or "-m pytest" in c:
            return True
    return False


def main() -> int:
    fd = _try_lock(_LOCK_PATH)
    # 文件锁（进程内/同视图互斥） + 进程表（跨视图兜底）：任一判定"已在跑"就退出
    if fd is None or _other_pytest_running():
        if fd is not None:
            os.close(fd)
        print(
            "[run_tests] ⛔ RESULT=SKIPPED —— 本次**没有跑测试**（不是通过！）\n"
            "            检测到已有 pytest 实例在跑（或本命令被沙箱重放）⇒ 跳过。\n"
            "            原因：xdist 每 worker ≈960MB，两套并行会耗尽内存（实测会遗留孤儿进程），\n"
            "            且两套 worker 争抢 GPU/IO 会让全量从 ~115s 退化到 ~412s。",
            file=sys.stderr,
        )
        return 0
    try:
        return subprocess.call([sys.executable, "-m", "pytest", *sys.argv[1:]])
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
