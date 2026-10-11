"""消融实验批量运行器（**单实例锁 + 幂等 + 结果自校验**）。

## 为什么需要它（三个真实踩坑，都记在案）
1. **本环境的沙箱会重放 Bash 命令**（同一条命令执行两遍）⇒ 两份批跑同时抢同一张 8GB 显卡
   （bge-m3 + reranker 各约 2.3GB）⇒ **双方都卡死**（实测 CPU 0%、pid 不断变化）。
   ⇒ 用**仓库内单实例文件锁**兜住，重放副本立即退出（照搬 `scripts/run_tests.py` 的做法）。
2. **命令行写了参数 ≠ 参数生效 / 输入正确**：曾漏写 `--cases`，评测**静默回退到 20 条冒烟集**，
   指标全退化成 0.25 的底值 —— 报告看起来正常，数字全是废的。
   ⇒ 每个配置跑完**必须回报告核对** `cases_file` / `num_cases` / 旋钮值，不符即判失败。
3. 重排档位（384/256）**不是 CLI 开关**，走 `runtime_settings` ⇒ 只能写
   `<数据目录>/provider_settings.json`；**必须用隔离的数据目录**，不能碰真实 `data/`。

## 用法
    python scripts/run_ablation.py                 # 跑全部配置（已完成的自动跳过）
    python scripts/run_ablation.py --only sparse_off rerank384
    python scripts/run_ablation.py --skip rerank_off   # 跳过最慢的那个（约 70 分钟）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval"
LOCK = REPO / ".ablation.lock"
DATA_DIR = Path(r"D:\papermate-eval-tmp")  # ★ 隔离目录：绝不写真实 data/
CASES = "cases/golden_real.jsonl"
N_CASES = 221

#: (名字, 隔离设置里的档位覆盖, 额外 CLI 参数)
CONFIGS: list[tuple[str, dict, list[str]]] = [
    ("sparse_off", {"rag_reranker_max_length": 512, "rag_reranker_fp16": True}, ["--sparse", "off"]),
    ("rerank384", {"rag_reranker_max_length": 384, "rag_reranker_fp16": True}, []),
    ("rerank256", {"rag_reranker_max_length": 256, "rag_reranker_fp16": True}, []),
    ("qr_legacy", {"rag_reranker_max_length": 512, "rag_reranker_fp16": True}, ["--query-rewrite", "legacy"]),
    # ⚠️ 最后跑：`--rerank off` = fp16 关 + 不限长 = **重排优化前的档位**（不是"关闭重排"），实测约 19s/例
    ("rerank_off", {"rag_reranker_max_length": 0, "rag_reranker_fp16": False}, ["--rerank", "off"]),
]

#: 报告里用于自校验的字段与期望值
EXPECT = {
    "cases_file": "golden_real.jsonl",
    "num_cases": N_CASES,
    "merge_strategy": "composite",
}


def _try_lock(path: Path) -> int | None:
    """非阻塞抢锁；成功返回 fd（须持有到结束），失败返回 None。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
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


def _report_ok(name: str) -> tuple[bool, str]:
    """读报告做**自校验**：用例集/条数/合并口径/旋钮必须与预期一致。"""
    p = EVAL / f"report_ablation_{name}.json"
    if not p.exists():
        return False, "报告不存在"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, f"报告解析失败：{e}"
    bad = [f"{k}={d.get(k)!r}（期望 {v!r}）" for k, v in EXPECT.items() if d.get(k) != v]
    if bad:
        return False, "；".join(bad)
    return True, f"cases={d['num_cases']} recall@5={d.get('avg_recall@5')} mrr@10={d.get('avg_mrr@10')}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args(argv)

    fd = _try_lock(LOCK)
    if fd is None:
        print(
            "[ablation] RESULT=SKIPPED —— 本次没有跑实验（不是通过！）\n"
            "           检测到已有实例在跑（本环境沙箱会重放命令）⇒ 直接退出，避免抢 GPU 死锁。",
            file=sys.stderr,
        )
        return 0

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        todo = [c for c in CONFIGS if (args.only is None or c[0] in args.only) and c[0] not in args.skip]
        print(f"[ablation] 待跑 {len(todo)} 个配置：{[c[0] for c in todo]}\n")
        results: list[tuple[str, bool, str, float]] = []
        for name, settings, extra in todo:
            ok, why = _report_ok(name)
            if ok:
                print(f"[skip] {name}：已存在且校验通过（{why}）")
                results.append((name, True, why, 0.0))
                continue
            (DATA_DIR / "provider_settings.json").write_text(json.dumps(settings), encoding="utf-8")
            print(f"[run ] {name}：settings={settings} extra={extra}", flush=True)
            t0 = time.time()
            r = subprocess.run(
                [
                    sys.executable, "run_eval.py", "--embedder", "real", "--merge", "composite",
                    "--profile", "prod", "--k", "5", "--cases", CASES, *extra,
                    "--out", f"report_ablation_{name}.json",
                ],
                cwd=str(EVAL),
                env={**os.environ, "PAPERMATE_DATA_DIR": str(DATA_DIR)},
                capture_output=True,
                text=True,
            )
            dt = time.time() - t0
            ok, why = _report_ok(name)
            flag = "✓" if (r.returncode == 0 and ok) else "✗"
            print(f"[{flag} ] {name}：exit={r.returncode} 用时 {dt/60:.1f} 分钟 | {why}", flush=True)
            if not ok:
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
                print(f"      末尾输出：{tail}", flush=True)
            results.append((name, ok, why, dt))
        (DATA_DIR / "provider_settings.json").unlink(missing_ok=True)

        print("\n===== 汇总 =====")
        for name, ok, why, dt in results:
            print(f"  {'✓' if ok else '✗'} {name:<12} {dt:5.1f} 分钟  {why}")
        return 0 if all(ok for _, ok, _, _ in results) else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
