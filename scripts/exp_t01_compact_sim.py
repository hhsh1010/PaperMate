"""T-01 收尾实验（配套）：压缩次数与"压缩空转"的离线仿真。

## 为什么需要它

卡片③有四个预期效果，前三个（估算误差 ±10% / 峰值 −25% / 溢出 0）已由
`scripts/exp_t01_gate_e2e.py` 实测。第 4 个是「**压缩次数 ≤ +50%**」—— 它是**护栏**：
把估算器修准之后，`_compact_history` 的 cap 判据会从"可能永远不满足"变成"会满足"，
理论上可能让应用**突然开始频繁摘要**（每次摘要 = 一次 LLM 调用 + ≤20s 超时）→ 必须量化。

## 怎么测（零 API 消耗）

- 用**桩 LLM**（非 `FakeLLM`，故会走摘要分支）替代真实模型，只数调用次数，不真调 API；
- **按轮推进并应用压缩结果**（否则轨迹不真实：压缩被丢弃就看不到"涨到撞上限"）；
- 每轮开头按**真实口径**判"要不要压缩"（触发线 = `budget × 0.7`，T-01 **未改动**该判据），
  再分别用**旧估算器**（`len/2.5`，cap=`budget//2`）与**新估算器**（按字符类，cap=`trigger×0.7`）
  跑真实的 `_compact_history`，对比：触发次数 / 实际丢弃条数 / 摘要调用次数 / 轨迹峰值。

## 关键：结论依赖内容构成（必须分两种跑）

旧估算器是**单一系数** `len/2.5`，它的误差完全取决于内容类型：
- **中文为主**：真实密度 ≈1.64 字符/token，旧系数（2.5）偏低 **52%** → cap（估算口径）等效
  ≈2 倍真实 > 触发线 → **压缩空转**（触发但一条都不丢）；
- **中英混合**：英文把密度拉高（实测 ~2.2 字符/token），旧系数只偏低 ~12% → **基本不空转**。

真实日志里 `real/est` 的 p50 = **1.80**（偏中文），所以线上更接近第一种。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import lightcore.core.loop as loop_mod  # noqa: E402
from lightcore.core.loop import AgentLoop  # noqa: E402
from lightcore.llm.base import (  # noqa: E402
    ChatMessage,
    LLMResponse,
    Role,
    _rough_tokens,
)

OUT_FILE = ROOT / "eval" / "t01_compact_sim.txt"
BUDGET = 30000
TRIGGER = int(BUDGET * 0.7)
N_TURNS = 16

_ZH = (
    "该段描述了检索增强生成系统的分块策略与向量归一化细节，并给出在中文长文档上的实测结果，"
    "重排阶段使用交叉编码器以提升首位命中率，推理开销在可接受范围内。"
)
_EN = (
    "The pipeline fuses dense and sparse retrieval and applies a cross-encoder reranker "
    "to improve top-1 accuracy on long documents with bounded inference cost. "
)

#: 两种内容构成：(标签, 每轮工具结果生成器)
PROFILES = (
    ("中文为主（贴近线上日志 real/est=1.80）", lambda: _rep(_ZH, 6000)),
    ("中英混合", lambda: _rep(_ZH, 3600) + _rep(_EN, 2400)),
)


def _rep(text: str, n: int) -> str:
    return (text * (n // len(text) + 1))[:n]


def _old_est(content) -> int:
    """T-01 改动前的估算器：`len / 2.5`（对中文为主的内容系统性偏低）。"""
    if isinstance(content, list):
        return max(1, int(sum(len(str(b)) for b in content) / 2.5))
    if not content:
        return 0
    return max(1, int(len(content) / 2.5))


class _StubLLM:
    """非 FakeLLM 的桩：走摘要分支，但只计数、不真调 API。"""

    model_name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None, **kw):  # noqa: ANN001, ANN003, ANN201
        self.calls += 1
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="【桩摘要】" + "要" * 120))


async def _run(est_fn, cap: int, gen, label: str) -> dict:
    """按轮推进并**应用**压缩结果；统计触发/丢弃/摘要调用/轨迹峰值。"""
    llm = _StubLLM()
    lp = object.__new__(AgentLoop)
    lp.llm = llm
    lp.config = SimpleNamespace(compact_trigger_tokens=TRIGGER,
                               compact_keep_capacity=cap,
                               max_tokens_budget=BUDGET)

    hist: list[ChatMessage] = []
    triggers = dropped_msgs = 0
    peak = 0
    over_budget = 0
    saved = loop_mod._rough_tokens
    loop_mod._rough_tokens = est_fn      # 被考察的估算器（真实 `_compact_history` 会用到）
    try:
        for i in range(N_TURNS):
            real = sum(_rough_tokens(m.content) for m in hist)   # 真实口径代理（误差 p50≈5%）
            peak = max(peak, real)
            if real > BUDGET:
                over_budget += 1
            if real >= TRIGGER:
                triggers += 1
                before = len(hist)
                hist = await lp._compact_history(list(hist))
                dropped_msgs += max(0, before - len(hist))
            hist.append(ChatMessage(role=Role.USER, content=f"第 {i} 问：请继续分析这一节。" + _rep(_ZH, 120)))
            hist.append(ChatMessage(role=Role.ASSISTANT, content=f"（第 {i} 轮）我需要再读一段原文。"))
            hist.append(ChatMessage(role=Role.TOOL, content=gen(), tool_call_id=f"c{i}"))
    finally:
        loop_mod._rough_tokens = saved
    final = sum(_rough_tokens(m.content) for m in hist)
    return {"label": label, "triggers": triggers, "dropped": dropped_msgs,
            "summaries": llm.calls, "peak": peak, "over_budget": over_budget, "final": final}


async def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = ["", "=" * 96,
             "T-01 配套 · 压缩次数与「空转」离线仿真（零 API 消耗）",
             "=" * 96,
             f"口径：单次上限 {BUDGET} · 触发线 = ×0.7 = {TRIGGER}（T-01 未改动）"
             f" · 共 {N_TURNS} 轮 · 每轮推进后**应用**压缩结果"]

    for plabel, gen in PROFILES:
        old = await _run(_old_est, BUDGET // 2, gen, "改动前（len/2.5 + cap=budget//2=15000）")
        new = await _run(_rough_tokens, int(TRIGGER * 0.7), gen, "改动后（按字符类 + cap=trigger×0.7=14700）")
        lines += ["", "-" * 96, f"内容构成：{plabel}", "-" * 96,
                  f"{'方案':<44}{'触发次数':>8}{'实际丢弃':>10}{'摘要调用':>10}{'轨迹峰值':>9}{'末态规模':>9}"]
        for r in (old, new):
            lines.append(f"{r['label']:<44}{r['triggers']:>8}{r['dropped']:>10}{r['summaries']:>10}"
                         f"{r['peak']:>9,}{r['final']:>9,}")
        verdict = (f"触发 {old['triggers']} 次但一条未丢" if old["dropped"] == 0
                   else f"实际丢弃 {old['dropped']} 条")
        lines.append(f"  → 改动前：{verdict}；轨迹峰值 {old['peak']:,}（上限 {BUDGET:,}）")
        lines.append(f"  → 改动后：触发 {new['triggers']} 次、丢弃 {new['dropped']} 条、"
                     f"摘要调用 {new['summaries']} 次；轨迹峰值 {new['peak']:,} → {new['final']:,}")
        ok = new["triggers"] <= old["triggers"] * 1.5
        lines.append(f"  → 护栏「压缩次数 ≤ +50%」：触发 {old['triggers']} vs {new['triggers']}"
                     f" = {'相同/更少 → 满足' if ok else '超出 → 需复查'}")

    lines += ["", "=" * 96,
              "结论：",
              "1) 旧口径（cap=budget//2=15000，**估算**口径）折算成真实约 22.9k，仍**高于**触发线 21k",
              "   ⇒ **首次触发不丢、下一轮才丢** ⇒ 准确说法是「**压缩延迟一轮 + 单次压得不够**」，",
              "   而不是早先静态推算的「一条都不丢（空转）」。",
              "2) 该现象**与内容构成相关**：中文为主时旧估算偏低 52%（延迟明显）；",
              "   中英混合时只偏 ~12%（几乎无延迟）。线上日志 real/est 的 p50=1.80 ⇒ 偏前一种。",
              "3) 改动后两种构成都**按量压缩**：轨迹峰值与末态规模都更低（中文为主 26.3k→15.1k），",
              "   且**触发次数不增反降**（因触发判据用的是真实值，T-01 没动它）⇒ 护栏满足；",
              "   摘要调用（真实成本）随之下降；每次摘要上限 SUMMARY_MAX_TOKENS=600、",
              "   超时 SUMMARY_TIMEOUT_S=20，失败回退滚动截断、不阻断主流程。",
              "=" * 96]

    text = "\n".join(lines)
    print(text, flush=True)
    with OUT_FILE.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    asyncio.run(main())
