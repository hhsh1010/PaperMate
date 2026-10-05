"""失败归因分类（A4，2026-09-21）。

**要解决的问题**（复检 §3 A4）：
轨迹只记了事件流，没有"这次失败属于检索错 / 规划错 / 工具错 / 格式错"的分类，
于是**同一种失败反复出现却无法定位到责任模块**，调优只能靠人肉归因。

**关键设计决策：在"产生失败的那一行"标注分类，而不是事后从 reason 文本去猜。**

理由：`reason` 是给人看的中文文案（"连续无进展步数过多，疑似死循环，已中止"），
用它反推分类 = **把展示文案当协议**。改一个字的文案就会静默失配，
而且失配后**不会报错**，只会让统计悄悄变空 —— 这正是本项目反复吃过的亏
（参见 `_rough_tokens` 与"同名两份提示词"的教训）。

所以：分类由**产生方**显式给出（`kind=` 参数），文案随便改，归因始终成立。

**覆盖边界（诚实说明）**：本模块只归因**确定性的中止/错误**（可机械判定）。
"回答了但答得不好"（检索召回差、引用缺失、格式不合规）属于**质量类**问题，
需要判官（A3 / A2）才能判定，不在本模块范围内。
"""

from __future__ import annotations

#: 失败分类 —— 取值即"责任模块"，便于直接回流到对应部分。
#:
#: - `llm_error`       LLM 调用本身失败（网络/鉴权/超时）→ 责任在 LLM 层
#: - `tool_error`      工具连续失败超限 → 责任在工具实现或调用参数
#: - `no_progress`     连续多步无新信息（疑似死循环）→ 责任在规划/提示词
#: - `max_iterations`  达到最大迭代仍未收敛 → 责任在规划（预算或任务分解）
#: - `budget_exceeded` token/成本熔断 → 责任在上下文管理或任务规模
#: - `retrieval_empty` 检索返回空（有文档但没命中）→ 责任在检索链路
#: - `cancelled`       用户主动取消 —— ⚠️ **不是失败**，单列以便统计时排除
#: - `unknown`         未标注（老轨迹 / 新增路径漏标）—— 出现即说明有打点缺口
#: - `timeout`         工具 / LLM 执行**超时**（2026-09-30 新增）→ 责任在单次执行耗时。
#:                     ★ 为什么从 `tool_error` 里拆出来：二者的**自救方向完全不同** ——
#:                       超时要"缩小问题规模 / 减少关联文档"，参数错要"修调用参数"。
#:                       混在一个桶里就只能给"请重试"这种无信息量的建议。
#: - `crash`           服务崩溃 / SSE 流意外中断（2026-09-30 新增）→ 责任在运行环境。
#:                     ★ 为什么从 `interrupted` 里拆出来：`interrupted` 长期把
#:                       "用户主动停止"与"进程崩了"混在一起 ⇒ 前者正常、后者要修。
LLM_ERROR = "llm_error"
TOOL_ERROR = "tool_error"
NO_PROGRESS = "no_progress"
MAX_ITERATIONS = "max_iterations"
BUDGET_EXCEEDED = "budget_exceeded"
RETRIEVAL_EMPTY = "retrieval_empty"
CANCELLED = "cancelled"
TIMEOUT = "timeout"
CRASH = "crash"
UNKNOWN = "unknown"

ALL_KINDS = (
    LLM_ERROR,
    TOOL_ERROR,
    NO_PROGRESS,
    MAX_ITERATIONS,
    BUDGET_EXCEEDED,
    RETRIEVAL_EMPTY,
    CANCELLED,
    TIMEOUT,
    CRASH,
    UNKNOWN,
)

#: 真正算"系统出问题"的分类（用于计算失败率）。
#: ⚠️ `cancelled` 必须排除：用户主动停止不是故障 ——
#: 把它算进去会把正常行为报成缺陷（08-19 的 `unfinished_rate` 命名教训同源）。
#: ⚠️ `timeout` / `crash` **算**故障（二者都需要人去修，与"取消"性质不同）。
REAL_FAILURES = (
    LLM_ERROR,
    TOOL_ERROR,
    NO_PROGRESS,
    MAX_ITERATIONS,
    BUDGET_EXCEEDED,
    RETRIEVAL_EMPTY,
    TIMEOUT,
    CRASH,
)

#: 中文标签（仅用于展示；**归因不依赖它**，见模块 docstring）
LABELS = {
    LLM_ERROR: "LLM 调用失败",
    TOOL_ERROR: "工具连续失败",
    NO_PROGRESS: "连续无进展",
    MAX_ITERATIONS: "达到最大迭代",
    BUDGET_EXCEEDED: "预算熔断",
    RETRIEVAL_EMPTY: "检索无结果",
    CANCELLED: "用户取消",
    TIMEOUT: "执行超时",
    CRASH: "服务中断",
    UNKNOWN: "未标注",
}


def normalize(kind: str | None) -> str:
    """把任意输入收敛为已知分类；未知/为空一律 `unknown`（绝不抛错、绝不静默丢弃）。"""
    if kind in ALL_KINDS:
        return kind  # type: ignore[return-value]
    return UNKNOWN


def label(kind: str) -> str:
    return LABELS.get(normalize(kind), kind)
