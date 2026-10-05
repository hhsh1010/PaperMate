"""O-12 探针（零成本版）：从 `data/logs/app.log` 解析 doc-bound 首字延迟的三段构成。

数据源（均已存在，不需要新打点、不发任何 API 请求）：
- `[LLM] mode=… step=N 墙钟=…ms HTTP实测=…ms 外层开销=…ms | ttft=…ms（生成段 …ms）| … 工具调用=K | model=…`
- `[本轮] steps=N 墙钟=…ms = LLM …ms + 工具 …ms + 其它 …ms | …`

判定 doc-bound 首轮：该轮 step=0 的 LLM 调用带 `工具调用>=1`（= 协议级强制工具调用，
见 `chat_service.should_force_tool_call`）⇒ 首字必然经历 `LLM1(决策) → 工具(检索) → LLM2首token`。
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "data" / "logs" / "app.log"

RE_CALL = re.compile(
    r"\[LLM\] mode=(?P<mode>\S+) step=(?P<step>\d+) 墙钟=(?P<wall>\d+)ms "
    r"HTTP实测=(?P<http>\d+)ms 外层开销=(?P<outer>\d+)ms \| ttft=(?P<ttft>\d+)ms"
    r"(?:（生成段 (?P<gen>\d+)ms）)? \| prompt=(?P<pt>\d+) completion=(?P<ct>\d+) "
    r"\| 工具调用=(?P<tc>\d+) \| model=(?P<model>\S+)"
)
RE_TURN = re.compile(
    r"\[本轮\] steps=(?P<steps>\d+) 墙钟=(?P<wall>\d+)ms = LLM (?P<llm>\d+)ms \+ "
    r"工具 (?P<tool>\d+)ms \+ 其它 (?P<other>\d+)ms \| LLM HTTP实测合计=(?P<http>\d+)ms"
)


def parse(path: Path) -> tuple[list[dict], list[dict]]:
    """按**行序**返回 (LLM 调用行, 轮次汇总行)。行序即时间序，用于相邻配对。"""
    calls: list[dict] = []
    turns: list[dict] = []
    if not path.exists():
        return calls, turns
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_CALL.search(line)
        if m:
            calls.append({k: (int(v) if k not in ("mode", "model") else v) for k, v in m.groupdict().items()})
            continue
        t = RE_TURN.search(line)
        if t:
            turns.append({k: int(v) for k, v in t.groupdict().items()})
    return calls, turns


def _turn_tool_ms(turns: list[dict], *, steps: int) -> float | None:
    """`steps` 恰好等于给定值的轮次里，工具段耗时的中位数（= 该轮仅有的那几次工具调用）。"""
    vals = [t["tool"] for t in turns if t["steps"] == steps]
    return statistics.median(vals) if vals else None


def _retrieval_latency_ms() -> float | None:
    """检索耗时的兜底来源（见下）。

    ★ 实测事实：`tool_runs.latency_ms` **全库为 0**（226 条 retrieve 无一采集，2026-10-02 实测）
    ⇒ 检索耗时**不在任何持久化数据里**。唯一可用的书面数据是 `[本轮]` 汇总的工具段合计：
    当该轮 `steps==2`（= 一次决策 + 一次生成，中间恰好一次工具调用）时，工具段**就等于那一次检索**。
    """
    return None


def main() -> None:
    calls, turns = parse(LOG)
    print(f"[data] {LOG.name}: 单次调用 {len(calls)} 条 · 轮次汇总 {len(turns)} 条")
    if not calls:
        print("（日志里没有计时行 —— 检查是否启用了 python -m papermate.api 启动，日志只在它下面配置）")
        return

    by_model: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        by_model[c["model"]].append(c)
    print("\n=== 按 model 分组（★ 跨模型混合会污染结论，必须先分组）===")
    for model, lst in sorted(by_model.items()):
        ttfts = [c["ttft"] for c in lst if c.get("ttft")]
        if not ttfts:
            continue
        print(f"  {model:28s} n={len(lst):3d} ttft 中位={statistics.median(ttfts):7.0f}ms "
              f"p90={sorted(ttfts)[int(len(ttfts) * 0.9)]:7.0f}ms")

    # step=0 且带工具调用 ⇒ doc-bound 首轮的第一段（决策）
    first_calls = [c for c in calls if c["step"] == 0]
    doc_bound = [c for c in first_calls if c["tc"] >= 1]
    print("\n=== doc-bound 首轮识别（step=0 且 工具调用>=1）===")
    print(f"  step=0 调用 {len(first_calls)} 条，其中带工具调用 {len(doc_bound)} 条")
    if doc_bound:
        walls = [c["wall"] for c in doc_bound]
        print(f"  第 1 段 LLM1（决策）墙钟：中位 {statistics.median(walls):.0f}ms（n={len(walls)}）")

    if turns:
        print("\n=== 轮次汇总（工具段占比）===")
        ratios = []
        for t in turns:
            tot = t["wall"] or 1
            ratios.append(t["tool"] / tot)
        print(f"  工具段占整轮比：中位 {statistics.median(ratios):.1%}（n={len(turns)}）")
        for t in turns[-3:]:
            print(f"    例：墙钟 {t['wall']}ms = LLM {t['llm']}ms + 工具 {t['tool']}ms + 其它 {t['other']}ms")

    # ★ 用户首字（TTFT）的三段构成。**口径必须写清**（否则会得出错误结论）：
    #   ① LLM1 决策墙钟   —— 日志可精确配对（step=0 且带工具调用）
    #   ③ LLM2 首 token   —— 日志可精确配对（相邻 step=1 的 ttft）
    #   ② 检索（工具段）  —— **日志里没有逐轮的检索耗时**，用另一来源近似：
    #        `tool_runs.latency_ms`（tool_name='retrieve'）的中位数。
    #   ⚠️ 曾经的错误口径：把 `LLM2.墙钟 - LLM2.ttft` 当成"检索耗时" —— 那其实是
    #     **首 token 之后的生成时间**（不属于 TTFT）。测量口径写错比不测量更危险。
    comp: list[tuple[int, int, int]] = []
    for i, c in enumerate(calls[:-1]):
        nxt = calls[i + 1]
        if c["step"] == 0 and c["tc"] >= 1 and nxt["step"] == 1:
            comp.append((c["wall"], nxt["ttft"], nxt["wall"] - nxt["ttft"]))
    print("\n=== ★ doc-bound 首字构成（口径见脚本注释）===")
    if comp:
        l1 = [x[0] for x in comp]
        t2 = [x[1] for x in comp]
        gen = [x[2] for x in comp]
        retr = _turn_tool_ms(turns, steps=2)  # steps=2 ⇒ 该轮恰好一次工具调用（= 检索）
        print(f"  配对样本 n={len(comp)}")
        print(f"  ① LLM1 决策墙钟    中位 {statistics.median(l1):7.0f}ms  （日志实测）")
        if retr:
            print(f"  ② 检索（工具段）   中位 {retr:7.0f}ms  （来自 [本轮] steps=2 的工具段，"
                  "= 该轮仅有的那一次工具调用）")
            print(f"  ⇒ 用户首字估算     中位 {statistics.median(l1) + retr + statistics.median(t2):7.0f}ms"
                  f"  = ①{statistics.median(l1):.0f} + ②{retr:.0f} + ③{statistics.median(t2):.0f}")
        else:
            print("  ② 检索（工具段）   无 steps=2 样本可配对（工具段无法从日志逐轮取出）")
        print(f"  ③ LLM2 首 token    中位 {statistics.median(t2):7.0f}ms  （日志实测）")
        print(f"  [参考] LLM2 首字之后的生成段 中位 {statistics.median(gen):.0f}ms（**不计入首字**）")
    else:
        print("  （未配对到 step=0→step=1 相邻序列；日志样本不足）")

    print("\n[注] 日志样本有限（app.log 会轮转）；统计级结论需提高日志级别后多跑若干次")


if __name__ == "__main__":
    sys.exit(main())
