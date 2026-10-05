"""PaperMate · 运行指标路由（D1 + A6，2026-09-21）

- GET /metrics?days=7&session_id=   ：成功率 / 时延 / 成本 / 好评率 + 阈值告警

**为什么是 JSON 而不是 Prometheus**：
本机是**单用户本地应用**，没有 Prometheus 抓取端；引入 `prometheus_client` 只为得到一个
文本格式，收益与新增依赖不成比例。这里用**零依赖 JSON**，字段自解释，前端直接消费。
将来真要接 Grafana，加一个 `/metrics/prometheus` 适配层即可，不必改聚合逻辑。

**A6 的"告警"是怎么定的**：
阈值是**常量**（见 `_THRESHOLDS`），并在响应里**回显**，避免"告警响了却不知道阈值多少"。
⚠️ 关键：**小样本不告警** —— 只有 1 条反馈就按"差评率 100%"报警，是典型的
**度量尺与阈值不匹配**（本项目在 `_rough_tokens` 上吃过同样的亏）。
所以每条告警都带 `sample` 且要求**样本量达标**才判定。
"""

from __future__ import annotations

import json
from datetime import timedelta

from fastapi import APIRouter, Query, Request
from sqlalchemy import func as sqlfunc

from lightcore.core import failure as m_failure
from lightcore.core import trace_metrics as tm

from ..models.db import Feedback, Message, Trace, _now, iso_utc
from ..runtime_settings import resolve_budget_for
from ..schemas import MetricAlert, MetricsOut

router = APIRouter(tags=["metrics"])

#: 阈值（与响应回显同源 —— 单一来源，改这里即可）。
#: 单位必须与聚合口径一致：latency 用 **毫秒**、cost 用 **元/轮**、rate 用 **0~1**。
#: ★ T-09（2026-09-30）告警口径切换：「未收尾率」含用户主动停止（实测 54.9% 里 34.1%
#:   是正常停止），拿它告警会把正常行为报成故障 ⇒ 改为监控**排除中断后**的保守率，
#:   并新增「确认失败率」双层监控（前者红看 breakdown 定位，后者红直接修）。
_THRESHOLDS = {
    "avg_latency_ms": 60_000,
    "avg_cost": 0.05,
    "unfinished_rate_excluding_interrupt": 0.20,
    "failed_confirmed_rate": 0.05,
    "tool_error_rate": 0.20,
    "bad_rate": 0.30,
}

#: 判定告警所需的最小样本量：样本太小，比率类指标会剧烈抖动（1 条差评 = 100%）。
_MIN_SAMPLES = {
    "avg_latency_ms": 3,
    "avg_cost": 3,
    "unfinished_rate_excluding_interrupt": 5,
    "failed_confirmed_rate": 5,
    "tool_error_rate": 5,
    "bad_rate": 5,
}

#: 单次最多解析多少条轨迹的事件流（事件流可达上千条/轨迹，全量解析会拖慢端点）。
_MAX_TRACES = 100


def _classify(alerts: list[dict], key: str, value: float, sample: int) -> None:
    """按阈值判定并追加告警。**样本量不足则直接跳过**（不产生噪音告警）。"""
    thr = _THRESHOLDS[key]
    if sample < _MIN_SAMPLES[key] or value <= thr:
        return
    alerts.append(
        {
            "code": key,
            "level": "warn",
            "value": round(value, 6),
            "threshold": thr,
            "sample": sample,
            "message": f"{key} = {value:.4g} 超过阈值 {thr:.4g}（样本 {sample} 条）",
        }
    )


@router.get("/metrics", response_model=MetricsOut)
def get_metrics(
    request: Request,
    days: int = Query(7, ge=1, le=365),
    session_id: int | None = Query(None),
) -> dict:
    """运行指标 + 告警。

    聚合源：`traces`（时延/成本/步数/是否收尾）、`messages`（token 与成本）、
    `feedback`（E1 的真实评价）。三者同源可交叉验证，不另起统计表。
    """
    db = request.app.state.chat_svc._Session
    since = _now() - timedelta(days=days)

    with db() as sess:
        q = sess.query(Trace).filter(Trace.created_at >= since)
        if session_id is not None:
            q = q.filter(Trace.session_id == session_id)
        # 窗口内总数（不受 _MAX_TRACES 限制）—— 与"本次实际解析的 n"区分开，
        # 否则窗口内 300 条时前端会以为只有 100 条，比率的分母对不上。
        total_traces = q.count()
        traces = q.order_by(Trace.id.desc()).limit(_MAX_TRACES).all()

        fb_q = sess.query(Feedback).filter(Feedback.created_at >= since)
        if session_id is not None:
            fb_q = fb_q.filter(Feedback.session_id == session_id)
        # 同一消息改分只算最后一次：与 /feedback/summary 口径一致（升序遍历覆盖）
        rated: dict[int | None, int] = {}
        for f in fb_q.order_by(Feedback.id.asc()).all():
            rated[f.message_id] = f.rating

        msg_q = sess.query(
            sqlfunc.count(Message.id),
            sqlfunc.coalesce(sqlfunc.sum(Message.cost), 0.0),
        ).filter(Message.created_at >= since, Message.role == "assistant")
        if session_id is not None:
            msg_q = msg_q.filter(Message.session_id == session_id)
        num_answers, msg_cost = msg_q.one()

    # ---- 轨迹聚合：时延 / 成本 / 步数 / 是否收尾 / 工具失败
    n = len(traces)
    latencies = [t.total_latency_ms or 0 for t in traces]
    costs = [t.total_cost or 0.0 for t in traces]
    steps = [t.total_steps or 0 for t in traces]

    tool_total = tool_fail = 0
    failure_kinds: dict[str, int] = {}
    # T-09：把每条轨迹的 events 收起来，交给**单一来源**做收尾分类 ——
    # 此前这里自己数 `done`（与 `dev_eval` 各写一份 ⇒ 同名字段两个口径，即"两把尺子"）。
    trace_events: list = []
    for t in traces:
        try:
            events = json.loads(t.events_json or "[]")
        except (json.JSONDecodeError, TypeError):
            events = []
        if not isinstance(events, list):
            events = []
        trace_events.append(events)
        # C3（2026-09-28）：检索空命中**按轨迹**计数（见下方 `retrieval_empty_in_trace`）
        retrieval_empty_in_trace = False
        for e in events:
            if not isinstance(e, dict):
                continue
            if e.get("type") == "tool_result":
                tool_total += 1
                tdata = e.get("data") or {}
                if tdata.get("ok") is not True:
                    tool_fail += 1
                # C3：标记由**路由层**打在 tool_result 事件上（`routes_search.is_empty_retrieval`）。
                # 为什么不是 `error` 事件：空命中不是错误 —— 本轮照常继续作答。
                if tdata.get("retrieval_empty") is True:
                    retrieval_empty_in_trace = True
            elif e.get("type") == "error":
                # A4（2026-09-21）：失败归因。分类由**产生方显式标注**（core/failure.py），
                # 这里只计数；老轨迹没有该字段 → 归入 unknown（暴露打点缺口，而不是丢弃）。
                kind = m_failure.normalize((e.get("data") or {}).get("failure_kind"))
                failure_kinds[kind] = failure_kinds.get(kind, 0) + 1
        # ★ 口径统一（易错点）：`failure_kinds` 里每一格的含义都是「**有多少条轨迹**出现了该情况」——
        # `error` 事件在循环里必然终止本轮 ⇒ 每条轨迹至多 1 个；而空命中可以在同一轮里发生多次
        # （多次 query 都没命中）。若按**次数**累加，`retrieval_empty` 会与其他分类**分母不同**，
        # 同一份直方图里混两种单位正是本项目「两把尺子」的老账。故这里**按轨迹计一次**。
        if retrieval_empty_in_trace:
            failure_kinds[m_failure.RETRIEVAL_EMPTY] = (
                failure_kinds.get(m_failure.RETRIEVAL_EMPTY, 0) + 1
            )

    # T-09：收尾分类（**单一来源** `lightcore.core.trace_metrics`）。
    # `interrupted` = 用户停止 / 断连 —— **不是缺陷**；`failed` = 真失败（原因见 failure_kinds）。
    outcome = tm.summarize_traces(trace_events)
    done_cnt = outcome["done"]
    lat = tm.latency_percentiles(latencies)
    # O-12：首字延迟同样走**单一来源**（`trace_metrics.extract_ttft_ms` / `ttft_percentiles`）
    ttft = tm.ttft_percentiles([tm.extract_ttft_ms(ev) for ev in trace_events])
    # ★ 上下文水位（2026-10-02）：`budget` 必须取**当前生效的**运行时配置，
    # 否则面板会显示"水位 84% / 额度 30000"而实际线上跑的是别的数 ⇒ 又一把尺子。
    # ★ 读取方式与 `chat_service.build_agent`（:484）**完全一致**（同一入口、同一默认值）。
    # ★ 读取方式与 `chat_service.build_agent`（:484）**完全一致**（同一入口、同一默认值）：
    # `runtime_settings.get_state()["max_tokens_budget"]` —— 它已合并 `.env` 与
    # `data/provider_settings.json`，是**当前生效**的额度。
    # ★ 2026-10-03：预算改为**按模型自动推断** ⇒ 这里也必须同源解析，否则
    # 「水位 x% / 额度 30000」会和线上实际生效的额度对不上（又一把尺子）。
    # ⚠️ **已知限制（如实记录）**：不指定 `session_id` 时统计的是**跨会话**数据，
    # 可能混了不同窗口的模型 ⇒ 没有唯一的"正确额度"。此时按显式/默认值算，
    # 让水位只作**粗略参考**（要看精确值请指定 session_id）。
    _model = (request.app.state.chat_svc.effective_model(session_id)
              if session_id is not None else None)
    _budget = resolve_budget_for(_model)
    ctx = tm.context_waterline(trace_events, _budget)

    # ★ T-09：延迟均值**必须与分位同源** —— 也剔除"未采集的 0"。
    # 实测差 5 倍：含 0 时 avg=2967ms、剔除后 avg=14897ms（80% 轨迹的 0 把均值稀释了）。
    # 若这里仍用 `sum(latencies)/n`，就会出现"avg 说 3 秒、p50 说 11 秒"的荒谬组合。
    avg_latency = float(lat["avg"])
    avg_cost = sum(costs) / n if n else 0.0
    avg_steps = sum(steps) / n if n else 0.0
    # 「未收尾」= 没写 done 事件的轨迹。⚠️ **包含用户主动点「停止」** ——
    # 所以叫 unfinished 不叫 error：把它叫错误率会把"用户不想等了"误判成系统故障
    # （实测 43% 未收尾，其中绝大多数是主动取消，正是这个命名的由来）。
    # ★ T-09：现在**可归因**了 —— 判断"该不该修"请看
    # `unfinished_rate_excluding_interrupt`（= failed/n），而不是这个含中断的总数。
    unfinished_rate = (n - done_cnt) / n if n else 0.0
    # ★ 排除中断时**必须连 `unknown` 一起算未收尾** —— 它是"不知道发生了什么"，
    # 既不能当成功（那是假通过），也不能当确认失败（那是栽赃）。
    unfinished_rate_excluding_interrupt = (
        (outcome["failed"] + outcome["unknown"]) / n if n else 0.0
    )
    tool_error_rate = tool_fail / tool_total if tool_total else 0.0

    # ---- 反馈（E1 真实评价）
    fb_vals = list(rated.values())
    fb_n = len(fb_vals)
    pos = sum(1 for v in fb_vals if v > 0)
    neg = sum(1 for v in fb_vals if v < 0)
    bad_rate = neg / fb_n if fb_n else 0.0

    alerts: list[dict] = []
    _classify(alerts, "avg_latency_ms", avg_latency, n)
    _classify(alerts, "avg_cost", avg_cost, n)
    # ★ T-09：告警挂**排除中断**的保守率（含 unknown 打点缺口，缺口不当成功），
    #   外加「确认失败」的严格率 —— 双层：前者红看 breakdown 定位缺口，后者红直接修。
    #   含中断的 `unfinished_rate` 只展示、不告警（54.9% 里 34.1% 是用户正常停止）。
    _classify(
        alerts, "unfinished_rate_excluding_interrupt",
        unfinished_rate_excluding_interrupt, n,
    )
    _classify(alerts, "failed_confirmed_rate", outcome["failed"] / n if n else 0.0, n)
    _classify(alerts, "tool_error_rate", tool_error_rate, tool_total)
    _classify(alerts, "bad_rate", bad_rate, fb_n)

    return {
        "window_days": days,
        "since": iso_utc(since),
        "session_id": session_id,
        "num_traces": n,
        "num_traces_total": total_traces,
        "num_answers": int(num_answers or 0),
        "num_sessions": len({t.session_id for t in traces}),
        "avg_steps": round(avg_steps, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "max_latency_ms": max(latencies) if latencies else 0,
        # T-09：分位（此前只有 avg/max —— 平均回推不出分位，也看不出长尾）。
        # ⚠️⚠️ **必须同时看 `latency_samples` / `latency_missing`**：实测 80.1% 的轨迹
        # `total_latency_ms == 0`（未采集，且含已收尾轮次）⇒ 分位只对**非零样本**计算，
        # 若缺口很大，这几个数字就**不可信**（不是"很快"）。
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "latency_p99_ms": lat["p99"],
        "latency_samples": lat["samples"],
        "latency_missing": lat["missing"],
        # ★ O-12（2026-10-02）：**服务端首字延迟**（TTFT）分布。
        # 口径：请求进入路由 → 首个正文增量（不含网络往返）；与单次 LLM 调用的 `ttft` 不同。
        # ⚠️ 老轨迹（埋点 2026-10-02 之前）一律计入 `ttft_missing`，**不用 0 冒充**；
        # 与 `latency_*` 同规则：只看 `ttft_samples` 足够大时的分位。
        "ttft_avg_ms": ttft["avg"],
        "ttft_p50_ms": ttft["p50"],
        "ttft_p95_ms": ttft["p95"],
        "ttft_samples": ttft["samples"],
        "ttft_missing": ttft["missing"],
        # ★★ 上下文水位（2026-10-02）：让"额度够不够"**事发前可见**，不必事后翻日志。
        # 数据源是 `done` 事件里**本来就有的** `peak_prompt_tokens`/`first_prompt_tokens`
        # （loop 侧零改动、零 DDL）⇒ 下列数字与 `[本轮]` 日志、`done` 事件**同源**。
        # ⚠️ 口径**不可混用**：`peak_*` = 本轮**最大单次** prompt（这才是"单次上下文"）；
        # `messages.prompt_tokens` 是**整轮累计**（库里某条高达 82,057，绝不是上下文用量）。
        # ⚠️ 老轨迹（`peak` 字段未埋）一律计入 `ctx_peak_missing`、分位报`null`（= 量不到），
        # **绝不用 0 冒充"水位很低"**。
        "ctx_budget": ctx["budget"],
        "ctx_trigger": ctx["trigger"],
        "ctx_peak_p50": ctx["peak_p50"],
        "ctx_peak_max": ctx["peak_max"],
        # ★ 余量 = budget − peak_p50：**中位情况下还剩多少空间可塞**（负数=常在越线）。
        "ctx_headroom_median": ctx["headroom_median"],
        "ctx_peak_samples": ctx["peak_samples"],
        "ctx_peak_missing": ctx["peak_missing"],
        "ctx_over_trigger": ctx["over_trigger"],
        "ctx_over_budget": ctx["over_budget"],
        # 「被动过刀」的轨迹数（闸门裁剪 / 历史压缩）—— 证据被裁过，**judge 测不出**那部分损失。
        "ctx_elided_traces": ctx["elided_traces"],
        "avg_cost": round(avg_cost, 6),
        "total_cost": round(sum(costs) + float(msg_cost or 0.0), 6),
        "unfinished_rate": round(unfinished_rate, 4),
        # ★ T-09：判断"该不该修"要看这个（已排除用户主动停止 / 断连）。
        # `unfinished_rate` 含中断 ⇒ 单看它会高估问题（实测 54.9% vs 20.7%）。
        "unfinished_rate_excluding_interrupt": round(unfinished_rate_excluding_interrupt, 4),
        # ★ 上面那个数里，**确认**有失败原因的部分（`unknown` 是打点缺口，不算确认失败）。
        # 实测：20.7% 里只有 2.4% 是确认失败，其余 18.3% 是 `failure_kind` 缺失的历史轨迹。
        "failed_confirmed_rate": round(outcome["failed"] / n, 4) if n else 0.0,
        "unfinished_breakdown": {
            "done": outcome["done"],
            "interrupted": outcome["interrupted"],
            "unknown": outcome["unknown"],
            "failed": outcome["failed"],
        },
        # ★ 有**明确**「用户主动停止」标记的条数（见 trace_metrics.has_user_stop_mark）。
        #   `interrupted` 里混了"用户停止"与"进程崩溃"，只有这个数是**确认**的停止。
        #   老轨迹无此打点 ⇒ 为 0 —— 这本身就是"打点缺口"的度量。
        "confirmed_user_stop": outcome["stopped"],
        "tool_error_rate": round(tool_error_rate, 4),
        # A4：失败归因分布（按分类计数）。`unknown` 非零 = 有打点缺口，需补。
        "failure_kinds": failure_kinds,
        "failure_kinds_excluding_cancel": {
            k: v for k, v in failure_kinds.items() if k in m_failure.REAL_FAILURES
        },
        "tool_calls": tool_total,
        "feedback": {
            "total": fb_n,
            "positive": pos,
            "negative": neg,
            "good_rate": round(pos / fb_n, 4) if fb_n else 0.0,
        },
        "thresholds": dict(_THRESHOLDS),
        "min_samples": dict(_MIN_SAMPLES),
        "alerts": [MetricAlert(**a) for a in alerts],
        "healthy": len(alerts) == 0,
    }
