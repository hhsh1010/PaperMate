"""G1：重模型加载的跨进程互斥（`lightcore/runtime/resource_lock.py`）的判据。

## 为什么必须有这些测试
`bge-m3` + `reranker` 各约 2.3GB，两个进程同时构造会互抢 8GB 显存 ⇒ **双方死锁**
（实测：CPU 0%、GPU 占满、10 分钟零产出）。这类问题**在单进程测试里永远复现不出来**，
所以判据必须**真的起第二个进程**（而不是 mock 掉"跨进程"这件事）。

## 五条判据
| 编号 | 判据 | 反例（如果没守住会怎样）|
|---|---|---|
| V1 | 被占用时：**等待到超时后明确报错**（可读文案，含持有者） | 无限等待 ⇒ 用户看到"卡住"却毫无解释 |
| V2 | 空闲时：正常拿到锁 | 误判占用 ⇒ 正常启动被拒 |
| V3 | **持有者被强杀 ⇒ 锁自动释放**（OS 级文件锁的意义） | 自研 pid 文件方案会留下"僵尸锁"，永久锁死 |
| V4 | ★ 等待发生在**工作线程**里 ⇒ **事件循环不被冻结** | 在 async 里同步等 ⇒ 重演本项目 1 号事故（后端整体冻结）|
| V5 | 真的加载点（embedder/rerank）**走了守卫** | 守卫写了但没接上 ⇒ 等于没做 |
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lightcore.runtime import resource_lock as rl

REPO = Path(__file__).resolve().parents[1]

#: 子进程脚本：抢锁并报告结果（用 `argv` 传参，避开 Windows 引号/反斜杠的坑）
_CHILD = """
import sys, time
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from lightcore.runtime import resource_lock as rl
lock, wait = Path(sys.argv[2]), float(sys.argv[3])
fd = rl.acquire(lock, timeout_s=wait)
if fd is None:
    print("BUSY", flush=True)
else:
    print("GOT", flush=True)
    time.sleep(float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
    rl.release(fd)
    print("DONE", flush=True)
"""


def _start_child(lock: Path, wait: float, hold_s: float = 0.0) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(REPO), str(lock), str(wait), str(hold_s)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


# ------------------------------------------------------------------ V1 / V2


def test_v1_busy_times_out_with_readable_error(tmp_path: Path) -> None:
    """V1：被占用 ⇒ 等待到超时后**明确报错**（而不是无限等或静默放行）。"""
    lock = tmp_path / "l.lock"
    fd = rl.acquire(lock, timeout_s=5)
    assert fd is not None
    try:
        child = _start_child(lock, wait=0.6)
        out, _ = child.communicate(timeout=30)
        assert "BUSY" in out, f"子进程应报占用超时，实际输出：{out!r}"
        # 持有者信息必须**能被别人读到**（写入点刻意放在偏移 1，避开被锁的第 0 字节）
        assert str(os.getpid()) in rl.read_owner(lock), rl.read_owner(lock)
    finally:
        rl.release(fd)


def test_v2_free_lock_is_acquired(tmp_path: Path) -> None:
    """V2：空闲 ⇒ 正常拿到（避免"误判占用导致正常启动被拒"）。"""
    lock = tmp_path / "l.lock"
    child = _start_child(lock, wait=5)
    out, _ = child.communicate(timeout=30)
    assert "GOT" in out and "DONE" in out, out


# ------------------------------------------------------------------ V3


def test_v3_killed_holder_releases_lock(tmp_path: Path) -> None:
    """★ V3：持有者被**强杀** ⇒ 锁必须自动释放（这是选 OS 级文件锁的核心理由）。"""
    lock = tmp_path / "l.lock"
    child = _start_child(lock, wait=5, hold_s=30)
    # 等它确实拿到锁
    assert child.stdout is not None
    deadline = time.monotonic() + 30
    got = ""
    while time.monotonic() < deadline and "GOT" not in got:
        got += child.stdout.readline()
    assert "GOT" in got, f"子进程未拿到锁：{got!r}"

    child.kill()          # 模拟进程崩溃（不是优雅退出）
    child.wait(timeout=30)

    fd = rl.acquire(lock, timeout_s=3)   # 崩溃后应当**立刻**可拿
    assert fd is not None, "持有者被强杀后仍拿不到锁 ⇒ 出现了僵尸锁"
    rl.release(fd)


# ------------------------------------------------------------------ V4


def test_v4_waiting_does_not_freeze_event_loop(tmp_path: Path) -> None:
    """★ V4：等待发生在 `to_thread` 里 ⇒ 事件循环**必须继续推进**。

    反例：若在事件循环里同步等锁，本测试的 ticker 会停止推进 —— 那正是本项目 1 号事故
    （同步重计算跑在 async 里 ⇒ 后端整体冻结）。
    """
    lock = tmp_path / "l.lock"
    fd = rl.acquire(lock, timeout_s=5)   # 先占住
    assert fd is not None
    ticks = 0

    async def main() -> None:
        nonlocal ticks

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(20):
                await asyncio.sleep(0.05)
                ticks += 1

        t = asyncio.create_task(ticker())
        # 正确用法：在工作线程里等锁（会超时，因为上面占着）
        got = await asyncio.to_thread(rl.acquire, lock, 1.2)
        assert got is None
        await t

    try:
        asyncio.run(main())
    finally:
        rl.release(fd)
    assert ticks >= 15, f"等待期间事件循环被冻结（只推进了 {ticks} 次）⇒ 必须走 to_thread"


# ------------------------------------------------------------------ V5


def test_v5_embedder_and_reranker_use_the_guard(monkeypatch, tmp_path: Path) -> None:
    """★ V5：真正的加载点**确实走了守卫**（否则守卫只是"写了没用上"）。

    ⚠️ 打桩打在**使用方模块**（`lightcore.rag.embedder` / `lightcore.rag.rerank`）：
    它们用的是 `from ... import hold_model_load`，打源模块对它们无效（本项目踩过的坑）。
    """
    import lightcore.rag.embedder as emb_mod
    import lightcore.rag.rerank as rrk_mod

    calls: list[str] = []

    class _FakeGuard:
        def __init__(self, label: str):
            self.label = label

        def __enter__(self) -> None:
            calls.append(self.label)

        def __exit__(self, *exc: object) -> bool:
            return False

    monkeypatch.setattr(emb_mod, "hold_model_load", lambda label=None: _FakeGuard(label or ""))
    monkeypatch.setattr(rrk_mod, "hold_model_load", lambda label=None: _FakeGuard(label or ""))

    # embedder：把重量级导入换成假模型，只验证"构造被守卫包住"
    class _FakeFlag:
        dim = 1024

        def __init__(self, *a, **kw) -> None:
            pass

        def encode(self, *a, **kw):
            return {"dense_vecs": []}

    monkeypatch.setattr(emb_mod, "_import_bgem3_flag_model", lambda: _FakeFlag)
    emb_mod.BgeM3Embedder(model_name="fake", device="cpu")
    assert any("bge-m3" in c for c in calls), f"embedder 构造未走守卫：{calls}"

    # reranker：本地快照路径（避免真的联网下载）
    monkeypatch.setattr(rrk_mod.CrossEncoderReranker, "_resolve_local", lambda self, m: "/tmp/fake")
    monkeypatch.setitem(sys.modules, "sentence_transformers", types_stub())
    rrk_mod.CrossEncoderReranker(device="cpu")
    assert any("reranker" in c for c in calls), f"reranker 构造未走守卫：{calls}"


def types_stub():  # noqa: ANN201
    """最小 `sentence_transformers` 替身：只提供 `CrossEncoder`。"""
    import types

    mod = types.ModuleType("sentence_transformers")

    class _FakeCE:
        def __init__(self, *a, **kw) -> None:
            pass

    mod.CrossEncoder = _FakeCE  # type: ignore[attr-defined]
    return mod


# ------------------------------------------------------------------ 配置边界


def test_wait_seconds_falls_back_on_garbage(monkeypatch) -> None:
    """配置写错 ⇒ 回落默认值，**绝不变成无限等待**。"""
    monkeypatch.setenv("PAPERMATE_MODEL_LOCK_WAIT_S", "abc")
    assert rl.wait_seconds() == 120.0
    monkeypatch.setenv("PAPERMATE_MODEL_LOCK_WAIT_S", "5")
    assert rl.wait_seconds() == 5.0


def test_lock_path_follows_data_dir(monkeypatch, tmp_path: Path) -> None:
    """锁文件必须落在**数据目录**（而不是仓库根）⇒ 多实例/多用户互不干扰。"""
    monkeypatch.setenv("PAPERMATE_DATA_DIR", str(tmp_path))
    assert rl.lock_path().parent == tmp_path


# ------------------------------------------------------- 打开锁文件的健壮性
# 背景（真机验收抓出）：Windows 上杀软/索引器会扫刚创建的文件，此时 `os.open(O_RDWR)`
# 会瞬时返回 EACCES。原先"一次失败就抛错"会让**正常启动被误杀** ⇒ 加了有限重试。


def test_transient_open_denial_is_retried(tmp_path: Path, monkeypatch) -> None:
    """瞬时被拒（模拟杀软扫文件）⇒ **有限重试后仍应成功**。"""
    real_open = os.open
    calls = {"n": 0}

    def flaky_open(path, flags, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "Permission denied")
        return real_open(path, flags, *a, **kw)

    monkeypatch.setattr(rl.os, "open", flaky_open)
    fd = rl.acquire(tmp_path / "l.lock", timeout_s=5)
    assert fd is not None, "瞬时被拒后应当重试成功"
    assert calls["n"] == 4
    rl.release(fd)


def test_persistent_open_denial_fails_clearly(tmp_path: Path, monkeypatch) -> None:
    """持续被拒 ⇒ **明确报错**（既不无限重试，也不静默放行）。"""
    monkeypatch.setattr(
        rl.os, "open",
        lambda *a, **kw: (_ for _ in ()).throw(PermissionError(13, "Permission denied")),
    )
    t0 = time.monotonic()
    with pytest.raises(rl.ModelBusyError) as err:
        rl.acquire(tmp_path / "l.lock", timeout_s=5)
    spent = time.monotonic() - t0
    assert spent < 6, f"应在 3s 重试预算内失败，实际 {spent:.1f}s（不能无限等）"
    assert "无法打开" in str(err.value)


@pytest.mark.parametrize("hold", [0.0, 0.1])
def test_release_is_idempotent_safe(tmp_path: Path, hold: float) -> None:
    """释放语义：关闭 fd 即释放；重复释放不得抛错。"""
    fd = rl.acquire(tmp_path / "l.lock", timeout_s=5)
    assert fd is not None
    rl.release(fd)
    rl.release(fd)  # 二次释放：静默容忍
    fd2 = rl.acquire(tmp_path / "l.lock", timeout_s=1)
    assert fd2 is not None, "释放后应可再次获得"
    rl.release(fd2)
