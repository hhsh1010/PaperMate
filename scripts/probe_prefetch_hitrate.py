"""O-12 档 2 · P0 命中率统计（零成本，只读日志）。

用法：开启探针跑若干次真实 doc-bound 提问后：
    PAPERMATE_PREFETCH_PROBE=1   # 启动后端时设置
    .venv/Scripts/python.exe scripts/probe_prefetch_hitrate.py

判据（见 `docs/O-12-档2-并行预取-影响面与去重口径设计.md` §6）：
- **命中率 ≥50%** ⇒ 值得进入 P1（真做预取复用）；
- **t_first_retrieve 中位 > 检索耗时中位** ⇒ 预取大概率能在模型提问前完成
  （检索耗时取自同一日志里的 `[工具] retrieve … 耗时=…ms`）。
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "data" / "logs" / "app.log"

RE_PROBE = re.compile(r"\[预取探针\] 命中=(?P<hit>[是否]) t_first_retrieve=(?P<t>[\d.]+|nan)ms")
RE_TOOL = re.compile(r"\[工具\] retrieve ok 耗时=(?P<ms>\d+)ms")


def main() -> int:
    if not LOG.exists():
        print(f"未找到日志：{LOG}")
        return 1
    text = LOG.read_text(encoding="utf-8", errors="ignore")
    probes = RE_PROBE.findall(text)
    tools = [int(m) for m in RE_TOOL.findall(text)]

    if not probes:
        print("[无数据] 日志里没有 `[预取探针]` 行 —— 确认启动后端时设置了 "
              "PAPERMATE_PREFETCH_PROBE=1，并跑过 doc-bound 提问（会话需关联文档）。")
        return 0

    hits = [h for h, _ in probes]
    n = len(probes)
    n_hit = sum(1 for h in hits if h == "是")
    ts = [float(t) for _, t in probes if t != "nan"]

    print(f"样本（doc-bound 轮次）= {n} · 命中 = {n_hit} ⇒ 命中率 = {n_hit / n:.1%}")
    print(f"判据：≥50% 才值得做 P1 —— 当前{'达标' if n_hit / n >= 0.5 else '**未达标**'}")
    if ts:
        print(f"t_first_retrieve：中位 {statistics.median(ts):.0f}ms（n={len(ts)}）"
              f" · p90 {sorted(ts)[min(len(ts) - 1, int(len(ts) * 0.9))]:.0f}ms")
    if tools:
        med_tool = statistics.median(tools)
        print(f"检索耗时（同一日志的 [工具] retrieve）：中位 {med_tool:.0f}ms（n={len(tools)}）")
        if ts:
            faster = sum(1 for t in ts if t > med_tool)
            print(f"⇒ 预取**来得及**的轮次（模型提问晚于一次检索耗时）：{faster}/{len(ts)}")
    print("\n注：命中率是上限估计 —— P1 启用后若模型 query 与用户 query 相同，即可省下整段检索。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
