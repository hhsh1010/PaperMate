"""G1 真机验收：两个进程**同时加载重模型**，验证跨进程守卫能否防住显存死锁。

## 为什么要真机（不能只靠单测）
单测证明的是"锁的语义正确"；这里量的是**真机行为**：两个进程同时加载真实模型时的
**完成情况与显存峰值**。

## ★ 实测结论（2026-10-12，真机 · RTX 4070 Laptop 8GB）
| 组 | 两进程是否完成 | 显存峰值 |
|---|---|---|
| **正例（走守卫）** | ✅ 2/2（47.4s） | **5046 MB** |
| **反例（绕过守卫）** | ✅ **2/2（40.5s）** ✗ | 6304 MB |

**⚠️ 反例并没有复现死锁** —— 也就是说：**"两个进程同时构造模型"本身**在当前机器/模型规模下
并不会必然死锁（8GB 卡放得下两份的加载窗口峰值）。历史那次真死锁（CPU 0%、10 分钟零产出）
发生在**批跑/评测进程**上（且当时命令被环境重放成两份），负载比"纯构造"更重 ⇒
**根因未能归一到"仅构造重叠"** ✗。

## 那守卫还有什么用（如实说）
- **可测到的收益**：加载窗口显存峰值 **5046 vs 6304 MB**（串行化确实降低了并发持有）；
- **代价**：两进程同时启动时各多花约 3~8s（36.4s/43.7s vs 35.6s/36.0s）；
- **定位**：它是一条**便宜、可解释的资源防线**（卡更小 / 模型更大 / 进程更多时收益放大），
  但**不能宣称"它修好了那次死锁"** —— 那次的根因未证实。

## 用法
    python scripts/probe_model_load_guard.py --orchestrate            # 正例（走守卫）
    python scripts/probe_model_load_guard.py --orchestrate --no-guard # 反例（绕过守卫）
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: 硬超时：反例组预期会卡死，必须由外部强制终止（绝不"无限等"）
HARD_TIMEOUT_S = 75.0


def _log(tag: str, msg: str) -> None:
    print(f"[{tag} t={time.time() % 1000:08.3f}] {msg}", flush=True)


def run_role(role: str, no_guard: bool) -> int:
    """子进程角色：加载两个重模型并报告时间点。"""
    if no_guard:
        # ★ 反例：把守卫换成"空上下文" —— 注意必须打在**使用方模块**
        #   （`embedder.py`/`rerank.py` 用的是 `from ... import hold_model_load`，
        #    打在源模块 `resource_lock` 上对它们无效 —— 本项目踩过的坑）
        import lightcore.rag.embedder as emb
        import lightcore.rag.rerank as rrk

        emb.hold_model_load = lambda *a, **kw: contextlib.nullcontext()  # type: ignore[assignment]
        rrk.hold_model_load = lambda *a, **kw: contextlib.nullcontext()  # type: ignore[assignment]
        _log(role, "已绕过守卫（反例组）")

    _log(role, "开始加载")
    from lightcore.rag.embedder import BgeM3Embedder
    from lightcore.rag.rerank import CrossEncoderReranker

    t0 = time.time()
    emb_obj = BgeM3Embedder()
    _log(role, f"嵌入模型就绪（{time.time() - t0:.1f}s）")
    rrk_obj = CrossEncoderReranker(device="cuda", fp16=True, max_length=512)
    _log(role, f"重排模型就绪，全部完成（累计 {time.time() - t0:.1f}s）")
    del emb_obj, rrk_obj
    return 0


def _sample_vram(stop: threading.Event, out: list[int]) -> None:
    """每秒采一次显存占用，记录峰值（MB）。"""
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            val = int(r.stdout.strip().splitlines()[0])
            out.append(val)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)


def orchestrate(no_guard: bool) -> int:
    # 隔离数据目录：**每轮全新**（避免残留锁文件/残留状态影响判据 —— 上一版用固定目录 + 清空文件，
    # 在 Windows 上被瞬时占用撞过 PermissionError，是探针自身的毛病，不是守卫的）
    data_dir = REPO / ".l0_tmp" / f"g1-acceptance-{int(time.time())}"
    data_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PAPERMATE_DATA_DIR": str(data_dir)}
    label = "反例（绕过守卫）" if no_guard else "正例（走守卫）"
    print(f"\n===== G1 真机验收 · {label} =====")
    print(f"  锁文件：{data_dir / '.model_load.lock'}\n")

    vram: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, vram), daemon=True)
    sampler.start()

    base = [sys.executable, str(Path(__file__).resolve())]
    extra = ["--no-guard"] if no_guard else []
    procs = [
        subprocess.Popen(
            [*base, "--role", role, *extra],
            cwd=str(REPO), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for role in ("A", "B")
    ]

    t0 = time.time()
    outcomes = []
    for role, p in zip(("A", "B"), procs, strict=True):
        try:
            out, _ = p.communicate(timeout=HARD_TIMEOUT_S)
            outcomes.append((role, p.returncode, out or ""))
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            outcomes.append((role, None, out or ""))

    elapsed = time.time() - t0
    stop.set()
    sampler.join(timeout=3)

    for role, rc, out in outcomes:
        print(f"--- 进程 {role}（退出码 {'超时被杀' if rc is None else rc}）---")
        print("\n".join("    " + ln for ln in out.strip().splitlines()) or "    （无输出）")

    peak = max(vram) if vram else 0
    # ★ 判据 A 必须看**退出码为 0** —— 上一版把"非超时退出"当完成，结果**崩溃也被算成成功** ✗
    ok_rc = [r for r, rc, _ in outcomes if rc == 0]
    crashed = [r for r, rc, _ in outcomes if rc not in (0, None)]
    print("\n===== 判据 =====")
    print(
        f"  A. 两个进程都**正常完成**（退出码 0）：{'✅' if len(ok_rc) == 2 else '❌'}"
        f"（成功 {len(ok_rc)}/2"
        + (f"，崩溃 {crashed}" if crashed else "")
        + (f"，超时被杀 {[r for r, rc, _ in outcomes if rc is None]}" if any(rc is None for _, rc, _ in outcomes) else "")
        + f"，总耗时 {elapsed:.1f}s）"
    )
    print(f"  C. 显存峰值：{peak} MB")
    # ★ 判据 B（2026-10-12 真机实测后修正）：守卫是**按"每个模型构造"串行**，不是"整个进程串行"
    #   —— 日志显示两者交错推进（A 嵌→B 嵌→A 重排→B 重排），这正是期望行为（双方都能推进）。
    #   因此判据不能用"B 开始 ≥ A 结束"（那是错的），而要用**物理量**：显存峰值。
    #   并发持有两份模型 ⇒ 峰值会到 9GB+（超卡）；串行持有 ⇒ 峰值约 5GB。
    single_set_max = 6500   # MB：单份模型 + 推理余量；超过即说明"同时持有两份"
    print(
        f"  B. 加载窗口**未并发持有两份模型**（显存峰值 ≤ {single_set_max}MB）："
        f"{'✅' if 0 < peak <= single_set_max else '❌'}"
    )
    if no_guard:
        print("     ⚠️ 反例组：若进程被超时强杀，即为**死锁**的表现（历史实测：双方 CPU 0%、10 分钟零产出）")
    return 0 if (len(ok_rc) == 2 and (no_guard or 0 < peak <= single_set_max)) else 1


def _ts(text: str, marker: str) -> float | None:
    for ln in text.splitlines():
        if marker in ln and "t=" in ln:
            with contextlib.suppress(ValueError, IndexError):
                return float(ln.split("t=")[1].split("]")[0])
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orchestrate", action="store_true")
    ap.add_argument("--role", default=None)
    ap.add_argument("--no-guard", action="store_true")
    args = ap.parse_args(argv)
    if args.orchestrate:
        return orchestrate(args.no_guard)
    return run_role(args.role or "?", args.no_guard)


if __name__ == "__main__":
    raise SystemExit(main())
