"""A0（2026-10-02，② 的前置）：汇总「实测 prompt / 本地估算」的比值分布。

## 为什么需要这个脚本（对应诊断文档 §9.2）

闸门投影用的是**本地估算**，熔断判定用的是**厂商实测**。二者之比若长期偏大 ⇒ 预警迟到
（表现为"突然熔断"）。要判断"该不该上校正系数 k"，需要的是**分布**而不是单点：
- p50 ≤ 1.15 ⇒ 漂移不成立 ⇒ **② 直接取消**；
- p50 > 1.3 且离散度小 ⇒ 可以考虑常数 k；
- 离散度大（p90/p10 > 2）⇒ **不该用全局 k**（会误伤某些轮次）。

**必须按 model 分组**：T-01 的密度标定是在特定模型上做的，跨模型平均毫无意义。

## 用法

    .venv/Scripts/python.exe scripts/probe_estimate_ratio.py [日志路径]

零成本：只读 `data/logs/app.log`，不发任何请求、不加载模型。
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "data" / "logs" / "app.log"

RE_RATIO = re.compile(
    r"\[估算比值\] model=(?P<model>\S+) step=(?P<step>\d+) "
    r"prompt=(?P<prompt>\d+) est=(?P<est>\d+) ratio=(?P<ratio>[\d.]+)"
)

# 判据阈值（与报备书 §9.2 一致；**不要**随手改，改了要同步改报备）
P50_OK = 1.15        # ≤ 此值：漂移不成立，② 取消
P50_HIGH = 1.30       # > 此值：才考虑常数 k
DISPERSION_MAX = 2.0  # p90/p10 > 此值：不该用全局 k


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


def parse(path: Path) -> dict[str, list[float]]:
    """按 model 分组返回 ratio 列表（保持出现顺序，便于看时间趋势）。"""
    out: dict[str, list[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_RATIO.search(line)
        if m:
            out[m.group("model")].append(float(m.group("ratio")))
    return out


def verdict(p50: float, dispersion: float) -> str:
    if p50 <= P50_OK:
        return f"漂移不成立（p50={p50:.2f} ≤ {P50_OK}）⇒ **② 直接取消**，不上校正系数"
    if p50 > P50_HIGH and dispersion <= DISPERSION_MAX:
        return f"比值大但稳定（p50={p50:.2f}，p90/p10={dispersion:.2f}）⇒ 可考虑**常数 k**（取值上偏安全 + 可一键关）"
    return (
        f"比值大且**离散**（p50={p50:.2f}，p90/p10={dispersion:.2f}）⇒ **不要全局 k**；"
        "考虑分桶修正或只加告警"
    )


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LOG
    if not path.exists():
        print(f"未找到日志：{path}")
        return 1
    data = parse(path)
    if not data:
        print("[无数据] 日志里没有 `[估算比值]` 行。\n"
              "  可能原因：① 后端未重启（本次打点随 2026-10-02 的代码生效）；\n"
              "            ② 日志已轮转（app.log 只留最近一段）。\n"
              "  下一步：重启后端 → 正常提问几轮（doc-bound 长会话更有代表性）→ 再跑本脚本。")
        return 0

    print(f"样本来源：{path}\n")
    print(f"{'model':34s} {'n':>4s} {'p10':>6s} {'p50':>6s} {'p90':>6s} {'p90/p10':>7s}")
    for model, ratios in sorted(data.items(), key=lambda kv: -len(kv[1])):
        p10, p50, p90 = _pct(ratios, 0.1), _pct(ratios, 0.5), _pct(ratios, 0.9)
        disp = (p90 / p10) if p10 > 0 else float("inf")
        print(f"{model:34s} {len(ratios):4d} {p10:6.2f} {p50:6.2f} {p90:6.2f} {disp:7.2f}")
        print(f"  ⇒ {verdict(p50, disp)}")

    print("\n口径说明：")
    print("  · est = 该次调用**发出前**对 messages 的估算（content + tool_calls.arguments + reasoning，")
    print("         即 `message_tokens` 唯一入口；不含系统提示与工具 schema —— 那是**所有轮次共有**的固定项，")
    print("         对『比值是否稳定』这个判断不敏感，但会让绝对比值略微偏高，读数时请知悉）。")
    print("  · prompt = 厂商返回的**真实** prompt_tokens。")
    print("  · ⚠️ 样本量 < 10 的结论**只能当趋势**，不能当分布。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))