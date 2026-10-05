"""T-04+ ④ 三臂对照驱动：`legacy` / `sections` / `v2` 各跑一遍全量 221 例。

为什么用脚本而不是 shell 循环：
  1. 沙箱会**重放 bash 命令两遍** → 循环会跑两轮（实测踩过）；
  2. 需要**单次守护**（锁文件）避免并发/重复执行；
  3. 三臂必须**串行**（GPU 单卡，并发会争用且互相污染延迟数字）。

用法（需 GPU；约 7 分钟）：
    .venv/Scripts/python.exe scripts/run_rewrite_arms.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "eval" / "_t04plus_rewrite_arms.lock"
ARMS = ["legacy", "sections", "v2"]


def main() -> None:
    # ── 单次守护：锁文件已存在 → 直接退出（防沙箱重放跑两轮）──
    if LOCK.exists():
        print(f"[SKIP] 锁文件已存在（{LOCK.name}），本次不重复执行。")
        print("       如需重跑，先删除该锁文件。")
        return
    LOCK.write_text(f"pid={Path(sys.executable).name}\n", encoding="utf-8")

    results: list[tuple[str, int, float]] = []
    for arm in ARMS:
        out_json = ROOT / "eval" / f"report_t04plus_rewrite_{arm}_k4.json"
        log = ROOT / "eval" / f"_t04plus_rw_{arm}.log"
        cmd = [
            sys.executable, str(ROOT / "eval" / "run_eval.py"),
            "--cases", "cases/golden_real.jsonl",
            "--embedder", "real", "--profile", "prod",
            "--merge", "composite", "--retrieve-top-k", "4", "--k", "4",
            "--query-rewrite", arm,
            "--out", str(out_json),
        ]
        print(f"\n{'=' * 70}\n▶ 臂 {arm} 开始（{time.strftime('%H:%M:%S')}）\n{'=' * 70}")
        t0 = time.perf_counter()
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        results.append((arm, proc.returncode, dt))
        print(f"  臂 {arm} 结束：exit={proc.returncode} · {dt / 60:.1f}min · 报告 → {out_json.name}")

    print(f"\n{'=' * 70}\n汇总\n{'=' * 70}")
    for arm, rc, dt in results:
        flag = "✅" if rc == 0 else "❌"
        print(f"  {flag} {arm:<10} exit={rc}  {dt / 60:.1f}min")


if __name__ == "__main__":
    main()