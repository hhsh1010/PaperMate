"""轨迹级指标口径的**单一来源**（T-09，2026-09-30）。

## 为什么需要这个模块
同一批 trace，`/metrics`（全量窗口）与 `/eval/summary`（最近 10 条）**各写了一份聚合**
⇒ 同名字段（如 `avg_latency_ms`）在两个端点上**口径不同**。这正是本项目反复吃过的
「两把尺子」（评测同源化已踩四次）。本模块把口径收敛到一处，两边都调它。

## 未收尾的**四分类**（本模块核心）
此前 `/metrics` 只有 `unfinished_rate = 1 - done率` **一个数** ⇒ **无法归因**：
实测 246 条轨迹里 135 条未收尾（54.9%），既有"用户主动停止"也有"真失败"，
一个数看不出**该不该修**（用户主动停止不是缺陷）。

| 分类 | 判据 | 语义 |
|---|---|---|
| `done` | 有 `done` 事件 | 正常收尾 |
| `interrupted` | 无 `done`、**无 `error`**（或 error 的 kind = `cancelled`） | 中断：用户停止 / 断连 / 进程退出 |
| `unknown` | 无 `done`、有 error 但 **`failure_kind` 缺失** | **打点缺口** —— 不知道发生了什么 |
| `failed` | 无 `done`、有 error 且 kind 明确 | 真失败（细分见 `failure_kinds`） |

★ **为什么必须把 `unknown` 单列**：它是"**不知道**"，**不能算成功也不能算失败**。
  实测 51 条 failed 轨迹里 **45 条的 kind 缺失**（多为 A4 打点之前的老轨迹）
  ⇒ 若把它并进 failed，"排除中断后的未收尾率"会被**高估 7.5 倍**（20.7% vs 真失败 2.4%）。

★ **`cancelled` 归 interrupted 而不是 failed**：`lightcore/core/failure.py` 早就定义了
  `CANCELLED` 且**刻意不放进 `REAL_FAILURES`** —— 设计意图就是"它是一种 failure_kind，
  但不是故障"。本模块按该意图归类 ⇒ **将来若在取消路径补上打点，无需改这里**。
  ⚠️ 现状：取消路径**没有任何 events 打点**（只写 `Message.meta.interrupted`），
  所以今天拿到的 `interrupted` **无法再区分**"用户主动停止"与"进程崩溃"。

## 延迟分位：必须先剔掉"未采集"的 0
★ **实测 197/246（80.1%）条轨迹的 `total_latency_ms == 0`**（且**包含 63 条已收尾的轮次**）
⇒ 直接对全样本算分位会得到 **p50 = 0**，看起来"极快"，实际是**采集缺口**
（平均回推不出分位，而"带缺口的样本"连分位都会骗人）。
⇒ 本模块**只对非零样本算分位**，并**显式返回 `samples` / `missing`** 让缺口可见。
"""
from __future__ import annotations

from . import failure as m_failure

_EV_DONE = "done"
_EV_ERROR = "error"
#: 「用户主动停止」的**独立事件类型**（2026-09-30 新增打点）。
#: ★ 为什么不是 `error`：前端 `reducer.ts` 的 `case "error"` 会把消息染成**错误态**，
#:   而"用户点了停止"是正常行为 ⇒ 用独立类型，前端 `default` 分支安全忽略它。
_EV_CANCELLED = "cancelled"

#: 分类结果的全部取值（稳定顺序，便于前端/文档对齐）。
CATEGORIES = ("done", "interrupted", "unknown", "failed")


def classify_events(events: object) -> str:
    """把**一条**轨迹的 events 归为 `done` / `interrupted` / `unknown` / `failed`。

    - 解析失败 / 非 list ⇒ `interrupted`（**不**判 failed：数据坏了 ≠ 系统失败）。
    - 有 `done` ⇒ `done`（即使中途出过错又恢复，也算收尾成功 —— 否则会高估失败）。
    - 无 done、无 error ⇒ `interrupted`。
    - 无 done、有 error：kind 缺失 ⇒ `unknown`；kind == `cancelled` ⇒ `interrupted`；其余 ⇒ `failed`。
    """
    if not isinstance(events, list):
        return "interrupted"
    has_done = False
    has_cancelled = False
    error_kind: str | None = None
    has_error = False
    for e in events:
        if not isinstance(e, dict):
            continue
        et = e.get("type")
        if et == _EV_DONE:
            has_done = True
        elif et == _EV_CANCELLED:
            has_cancelled = True
        elif et == _EV_ERROR:
            has_error = True
            data = e.get("data")
            k = (data or {}).get("failure_kind") if isinstance(data, dict) else None
            error_kind = k if isinstance(k, str) and k else None
    if has_done:
        return "done"
    # 明确的"用户主动停止"标记 ⇒ interrupted（且与"进程崩溃"区分开）
    if has_cancelled and not has_error:
        return "interrupted"
    if not has_error:
        return "interrupted"
    if error_kind is None:
        return "unknown"
    return "interrupted" if error_kind == m_failure.CANCELLED else "failed"


def has_user_stop_mark(events: object) -> bool:
    """是否有**明确的**「用户主动停止」标记。

    ⚠️ 为什么需要它：`interrupted` 是"没跑完、也没报错"—— 其中**既可能是用户点的停止，
    也可能是进程崩了**（崩溃来不及写任何事件）。只看 `interrupted` **区分不了**。
    本函数判"有没有**证据**表明是用户主动停止"：`cancelled` 事件，或 error 的 kind=`cancelled`。

    ⇒ 因此 `interrupted` 的语义应读作「**中断（原因未确认）**」，
       而 `confirmed_user_stop` 才是「**确认是用户停的**」。
       老轨迹没有该打点 ⇒ 计数为 0，这本身就是"打点缺口"的度量。
    """
    if not isinstance(events, list):
        return False
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("type") == _EV_CANCELLED:
            return True
        if e.get("type") == _EV_ERROR:
            data = e.get("data")
            k = (data or {}).get("failure_kind") if isinstance(data, dict) else None
            if k == m_failure.CANCELLED:
                return True
    return False


def summarize_traces(all_events: list[object]) -> dict[str, int]:
    """对**多条**轨迹的 events 做四分类计数（输入 = 每条轨迹的 events 列表）。

    返回 `{"done", "interrupted", "unknown", "failed", "total"}` + **`stopped`**
    （有**明确**用户停止标记的条数，见 `has_user_stop_mark`）。
    ★ 分类逻辑与 `classify_events` **同一处实现** ⇒ 两个端点不会分叉。
    """
    out: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    stopped = 0
    for ev in all_events:
        out[classify_events(ev)] += 1
        if has_user_stop_mark(ev):
            stopped += 1
    out["stopped"] = stopped
    out["total"] = len(all_events)
    return out


def percentile(values: list[float], q: float) -> float:
    """**线性插值**分位（与 `numpy.percentile` 的默认 `linear` 一致，但零依赖）。

    ★ 为什么不建直方图：`/metrics` / `/eval/summary` **每次查询都把样本全量取进内存**
    （`total_latency_ms` 列）⇒ 原始样本本就在手，直接排序即可。
    （若将来改为增量预聚合，才需要可合并直方图。）

    `q` 取 0~1；空样本返回 0.0。
    """
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo]) * (1.0 - frac) + float(s[hi]) * frac


def latency_percentiles(latencies: list[float]) -> dict[str, float | int]:
    """延迟分布：`avg` / `max` / `p50` / `p95` / `p99`（ms）+ **`samples` / `missing`**。

    ⚠️⚠️ **只统计非零样本**：实测 80.1% 的轨迹 `total_latency_ms == 0`（**未采集**，
    且含已收尾的轮次）⇒ 若把 0 当真实耗时，`p50` 会变成 **0ms**（假的"极快"）。
    调用方**必须**看 `samples`/`missing` 判断这条分位可不可信：
    样本太少（如 `/eval/summary` 只看 10 条）时，分位**没有统计意义**，只能看趋势。
    """
    vals = [float(x) for x in latencies if x]  # 剔除 0 = 未采集
    miss = len(latencies) - len(vals)
    if not vals:
        return {"avg": 0.0, "max": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "samples": 0, "missing": miss}
    return {
        "avg": round(sum(vals) / len(vals), 1),
        "max": int(max(vals)),
        "p50": round(percentile(vals, 0.50), 1),
        "p95": round(percentile(vals, 0.95), 1),
        "p99": round(percentile(vals, 0.99), 1),
        "samples": len(vals),
        "missing": miss,
    }


#: done 事件里承载**服务端首字延迟**的键（O-12，2026-10-02）。
#: 埋在 `routes_search._with_stream_cleanup`，随 done 落 trace；此处是读取口径的**唯一来源**。
TTFT_KEY = "ttft_ms"


def extract_ttft_ms(events: object) -> int | None:
    """从一条轨迹的 `events` 里取出服务端首字延迟（ms）；没有则 `None`。

    口径声明（写进 docstring 以免又被读错）：
    - 这是**服务端**首字延迟 = 请求进入路由处理器 → 首个正文增量（`message_delta`），
      **不含网络往返**；与单次 LLM 调用的 `ttft`（建连+排队+prefill）**不是一回事**；
    - **老轨迹没有这个键**（埋点 2026-10-02 才加）⇒ 一律 `None`，**绝不用 0 冒充**
      （与 `total_latency_ms` 的 80.1% 空洞同一处理原则：缺口必须可辨识）。
    """
    if not isinstance(events, list):
        return None
    for e in events:
        if not isinstance(e, dict) or e.get("type") != "done":
            continue
        data = e.get("data")
        if not isinstance(data, dict):
            continue
        v = data.get(TTFT_KEY)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return int(v)
    return None


def ttft_percentiles(ttfts: list[int | None]) -> dict[str, float | int]:
    """首字延迟分布（`avg`/`p50`/`p95`/`p99`）+ `samples`/`missing`。

    `missing` = 未采集（老轨迹 / 未产出即结束）⇒ 与 `latency_percentiles` 同口径：
    **只对已采集样本算分位**，缺口单独计数，不做任何填补。
    """
    vals = [float(v) for v in ttfts if v is not None and v > 0]
    miss = len(ttfts) - len(vals)
    if not vals:
        return {"avg": 0.0, "max": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "samples": 0, "missing": miss}
    return {
        "avg": round(sum(vals) / len(vals), 1),
        "max": int(max(vals)),
        "p50": round(percentile(vals, 0.50), 1),
        "p95": round(percentile(vals, 0.95), 1),
        "p99": round(percentile(vals, 0.99), 1),
        "samples": len(vals),
        "missing": miss,
    }


# ============================================================
# 上下文水位（2026-10-02）：让"额度够不够"在事发前可见，而不是事后翻日志
# ============================================================
#: `done.data` 里的**本轮最大单次** prompt（与 `prompt_tokens` 的"整轮累计"严格区分）
PEAK_PROMPT_KEY = "peak_prompt_tokens"
#: `done.data` 里的**本轮首次** prompt = 下一轮的跨轮基线（**不**用峰值，原因见 loop.py 注释）
FIRST_PROMPT_KEY = "first_prompt_tokens"
#: 闸门裁剪条数 / 历史压缩丢弃条数（"证据被动过刀"的可核对事实）
GATE_HITS_KEY = "_gate_hits"
COMPACTED_DROPPED_KEY = "_compacted_dropped"


def _done_data(events: object) -> dict | None:
    """取出该轨迹 `done` 事件的 `data`；没有则 `None`（未收尾 / 结构损坏）。"""
    if not isinstance(events, list):
        return None
    for e in events:
        if isinstance(e, dict) and e.get("type") == "done":
            data = e.get("data")
            if isinstance(data, dict):
                return data
    return None


def _int_or_none(data: dict | None, key: str) -> int | None:
    """取**正整数**；缺键 / 0 / bool / 非数值一律 `None`（**绝不用 0 冒充"没量到"**）。"""
    if not isinstance(data, dict):
        return None
    v = data.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        return None
    return int(v)


def extract_context_waterline(events: object) -> dict[str, int | None]:
    """取出一条轨迹的上下文水位（tokens）。

    ★ **三个量各自的含义，混用会得出完全错误的结论**（已在 2026-10-02 实测踩过）：

    | 字段 | 口径 | 实测可得性 |
    |---|---|---|
    | `peak` | 本轮**最大单次** prompt（≈ 每轮最大的那一次调用） | 库内 35/115 条 |
    | `first` | 本轮**首次** prompt = **下一轮的跨轮基线** | 库内 5/115 条（很新） |
    | `total` | 本轮**整轮累计** prompt（**不是**单次上下文！） | 普遍可得 |

    ⚠️ `total` 常被误读成"上下文用了多少"（库内某会话单条高达 82,057）
    ⇒ 实际读数请用 `peak`；`first` 才是"进入该轮时的基线"。

    缺失一律 `None`（老轨迹字段尚未埋），**绝不用 0 冒充**。
    """
    d = _done_data(events)
    return {
        "peak": _int_or_none(d, PEAK_PROMPT_KEY),
        "first": _int_or_none(d, FIRST_PROMPT_KEY),
        "total": _int_or_none(d, "prompt_tokens"),
        "gate_hits": _int_or_none(d, GATE_HITS_KEY),
        "compacted_dropped": _int_or_none(d, COMPACTED_DROPPED_KEY),
    }


def context_waterline(all_events: list[object], budget: int) -> dict[str, object]:
    """聚合上下文水位 + **相对额度的余量**（判断"还能不能塞进一轮"）。

    `budget` 传 `LoopConfig.max_tokens_budget`（**单次**熔断线），本函数据此算：
    - `peak_p50` / `peak_max`：单次 prompt 的中位 / 峰值；
    - `headroom_median` = `budget − peak_p50` ⇒ **中位情况下还剩多少余量**；
    - `over_trigger` / `over_budget`：越过 70% 触发线 / 直接越熔断线的**轨迹数**；
    - `samples` / `missing`：**缺口必须可辨识**（老轨迹无 `peak` 字段）。
    """
    marks = [extract_context_waterline(ev) for ev in all_events]
    peaks = [m["peak"] for m in marks]
    firsts = [m["first"] for m in marks]
    got_peak = [p for p in peaks if p is not None]
    got_first = [f for f in firsts if f is not None]
    trigger = int(budget * 0.7)

    # 「被动过刀」的轨迹数：闸门裁剪或历史压缩任一发生即为 1（**按轨迹计一次**，
    # 与 `failure_kinds` 同口径 —— 否则同一轮多次裁剪会让分母与其他分类不同）。
    touched = sum(
        1 for m in marks
        if (m["gate_hits"] or 0) > 0 or (m["compacted_dropped"] or 0) > 0
    )
    res: dict[str, object] = {
        "budget": int(budget),
        "trigger": trigger,
        "peak_samples": len(got_peak),
        "peak_missing": len(peaks) - len(got_peak),
        "first_samples": len(got_first),
        "over_trigger": sum(1 for p in got_peak if p > trigger),
        "over_budget": sum(1 for p in got_peak if p > budget),
        "elided_traces": touched,
    }
    if got_peak:
        p50 = percentile([float(x) for x in got_peak], 0.50)
        res["peak_p50"] = int(p50)
        res["peak_max"] = int(max(got_peak))
        res["headroom_median"] = int(budget - p50)
    else:
        # ⚠️ **没有样本时不得报 0 当"峰值很小"** —— 报 None = "量不到"，前端须显式区分
        res["peak_p50"] = None
        res["peak_max"] = None
        res["headroom_median"] = None
    return res

