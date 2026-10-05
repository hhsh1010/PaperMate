"""PaperMate · dev_eval 技能（S10 开发评测，docs/统一方案 阶段D）

命令：/eval /评测
产出「指标卡 + 门禁判定」：
- 检索基线：按**可信度优先**读 eval/ 下的评测报告（`report_real.json` > `eval_report.json`），
  并校验口径（真实嵌入 + 线上检索 profile）—— 口径不可信时门禁**不出结论**。
- 近期 Agent 指标：从 Trace 表聚合 工具成功率 / 溯源率 / 平均成本 / 平均延迟 / 平均步数。
- 门禁：检索 avg_recall@5 低于阈值（默认 0.8）则标「未达基线」，否则「达标」。
受 features.dev_eval 门控（SkillRouter 校验）。
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from lightcore.skills.spec import SkillContext, SkillResult, SkillSpec

#: 评测目录（测试通过 monkeypatch 本模块变量注入临时目录，见 tests/test_eval_baseline.py）
_EVAL_DIR = Path(__file__).resolve().parents[3] / "eval"

#: 报告候选（按可信度降序）：真实文档集报告优先于早期合成集报告。
#: 2026-09-16 修复（体检问题 5）：此前**硬编码**读 `eval_report.json` ——
#: 那是「20 条合成集 + fake 哈希嵌入 + 未开重排」的报告，门禁据此显示"✅ 达标"；
#: 而它不会被任何脚本自动重跑（数字可无限期陈旧），真正实跑的真实口径报告
#: `report_real.json`（25 条真实 PDF / real / prod）反而**零消费者**。
_REPORT_CANDIDATES = ("report_real.json", "eval_report.json")

#: 「可信基线」判据 —— 必须**同时**满足，缺一不可：
#:   - `embedder == "real"`：真实 BGE-M3 语义嵌入（fake 哈希嵌入与语义检索质量无关）；
#:   - `profile  == "prod"`：与线上同口径（CrossEncoder 重排 + jieba + 稀疏）。
#: 判据来自 l0 实测：同一语料 fast=0.80 vs prod=0.975 —— 口径不同，结论就不同。
_TRUSTED_EMBEDDER = "real"
_TRUSTED_PROFILE = "prod"

_RECALL_GATE = 0.8  # 检索 avg_recall@5 门禁阈值（接入真实嵌入后语义检索才有参考性）

#: 采样窗口时效护栏（2026-10-05）。看板取「最近 N 条非空 Trace」，
#: 但这份样本可能**又旧又少**，而它会被当成"现状"来读 —— 实测踩到：
#: 窗口 = 10-03 14:28 ~ 10-04 13:13（最新一条已 **27 小时**前），且 **8/10 早于 10-03 的一次修复**
#: ⇒ 看板在展示"修复前的世界"。⇒ 必须把"这份样本有多可信"显式标出来，而不是静默展示。
_SAMPLE_MAX_AGE_HOURS = 24.0


def _window_age_hours(newest_at: str | None, *, now: datetime | None = None) -> float | None:
    """窗口内**最新一条**距现在的小时数（`None` = 无样本 / 时间无法解析）。

    ⚠️ 落库时间是 **naive UTC**（SQLite 的 `DateTime` 不保留时区），必须**显式补 UTC** 再比较；
    否则会被按本地时区解释 —— 中国区整体偏差 **8 小时**（本项目「时区口径」的老账）。
    `now` 可注入，便于单测而不依赖真实时钟。
    """
    if not newest_at:
        return None
    try:
        dt = datetime.fromisoformat(str(newest_at))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return round(((now or datetime.now(UTC)) - dt).total_seconds() / 3600.0, 1)


def _sample_quality(window: dict, *, now: datetime | None = None) -> dict:
    """把「这份样本有多可信」算成结构化字段（喂给看板做护栏）。

    两个**独立**的判据，都只描述事实、不替代指标：
      - `stale`：最新一条距今 > `_SAMPLE_MAX_AGE_HOURS` ⇒ 指标可能已不代表现状；
      - `insufficient`：`sampled < limit` ⇒ 表里**连窗口都没填满**（比"n 小"更强的信号）。

    `now` 可注入：判据必须**与当前时钟无关地可复现**（否则测试只能"相对"断言，钉不住边界）。
    """
    age = _window_age_hours(window.get("to"), now=now)
    return {
        "age_hours": age,
        "max_age_hours": _SAMPLE_MAX_AGE_HOURS,
        "stale": (age is not None and age > _SAMPLE_MAX_AGE_HOURS),
        "insufficient": window.get("sampled", 0) < window.get("limit", 0),
    }


def _load_retrieval_report() -> dict:
    """挑选检索评测报告，并把**取数口径**与**可信性**一并返回。

    返回值恒为 dict（不再返回 None），便于调用方区分三种状态：
      ① `available=False` —— 没有任何报告；
      ② `available=True, trusted=False` —— 有报告但口径不可信（fake 嵌入 / 未开重排 /
         报告与用例集不一致），**门禁不得据此通过**；
      ③ `available=True, trusted=True` —— 可用于门禁判定。

    设计取向：**宁可不给结论，也不给假通过**。"达标"若建立在合成集或近似口径上，
    比"没有基线"更危险 —— 它让人以为评测在守着质量。
    """
    for name in _REPORT_CANDIDATES:
        path = _EVAL_DIR / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 报告损坏 → 尝试下一个候选
            continue

        embedder = data.get("embedder")
        profile = data.get("profile")
        reasons: list[str] = []
        if embedder != _TRUSTED_EMBEDDER:
            reasons.append(f"嵌入={embedder or '未知'}（需 real）")
        if profile != _TRUSTED_PROFILE:
            reasons.append(f"检索口径={profile or '未知'}（需 prod）")
        # 陈旧检测：报告记录的用例集指纹与当前用例集不一致 → 评测已过时
        stale = _stale_reason(data)
        if stale:
            reasons.append(stale)

        by_kind = data.get("metrics_by_kind") or {}
        return {
            "available": True,
            "file": name,
            "trusted": not reasons,
            "reason": "、".join(reasons),
            "embedder": embedder,
            "profile": profile,
            "cases_file": data.get("cases_file"),
            "num_cases": data.get("num_cases"),
            "num_answerable": data.get("num_answerable"),
            "avg_recall@5": (data.get("metrics") or {}).get("recall@5", data.get("avg_recall@5")),
            "avg_mrr@10": (data.get("metrics") or {}).get("mrr@10", data.get("avg_mrr@10")),
            "generated_at": data.get("generated_at"),
            # 分组指标：单 gold 下 recall 结构性饱和，分辨力在 multi_gold / nDCG —— 透传给看板
            "by_kind": {
                k: {
                    "n": (v or {}).get("n"),
                    "recall@5": ((v or {}).get("metrics") or {}).get("recall@5"),
                    "ndcg@5": ((v or {}).get("metrics") or {}).get("ndcg@5"),
                }
                for k, v in by_kind.items()
            },
        }
    return {
        "available": False,
        "file": None,
        "trusted": False,
        "reason": "未找到评测报告",
        # 键必须**恒存在**（值为 None）：前端直接取 `retrieval["avg_recall@5"]`，
        # 缺键会变成 undefined（"契约不存在"）而不是"无数据"，两种语义不应混同。
        "embedder": None,
        "profile": None,
        "cases_file": None,
        "num_cases": None,
        "num_answerable": None,
        "avg_recall@5": None,
        "avg_mrr@10": None,
        "generated_at": None,
        "by_kind": {},
    }


def _stale_reason(report: dict) -> str:
    """报告是否与**当前用例集 + 当前检索实现**一致。无法判定时返回空串（不阻断）。

    两类陈旧都要抓：
      - 用例集改了没重跑（`cases_sha256`）；
      - **用例集没变但检索实现/参数改了**（`pipeline_digest`）—— 后者原先漏掉，
        结果是"改了分块或重排参数，报告仍被判新鲜"，门禁守的是过去。
    指纹算法统一来自 `lightcore.rag.provenance`（单一来源）。
    """
    problems: list[str] = []

    recorded = report.get("cases_sha256")
    cases_file = report.get("cases_file")
    if recorded and cases_file:
        path = _EVAL_DIR / "cases" / cases_file
        try:
            import hashlib

            current = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        except Exception:  # noqa: BLE001 用例集缺失 → 无法比对，交由其它判据
            current = ""
        if current and current != recorded:
            problems.append(f"用例集已变（{recorded} → {current}）")

    recorded_pipe = report.get("pipeline_digest")
    if recorded_pipe:
        try:
            from lightcore.rag import provenance

            current_pipe = provenance.digest()
        except Exception:  # noqa: BLE001 内核不可导入时不做判定（不阻断）
            current_pipe = ""
        if current_pipe and current_pipe != recorded_pipe:
            problems.append(f"检索实现已变（{recorded_pipe} → {current_pipe}）")

    if not problems:
        return ""
    return "报告已过时：" + "、".join(problems) + "，请重跑评测"


#: 空轨迹判据：`total_steps=0` 且无任何事件 —— 这类行是"没跑起来的轮次"
#: （被中断/技能直答/失败），把它算进分母会**系统性拉低**平均步数与平均成本。
_EMPTY_TRACE_WHERE = (
    "COALESCE(total_steps, 0) = 0 AND (events_json IS NULL OR events_json IN ('[]', ''))"
)


class _TraceSamples(list):
    """list 子类：既是样本列表，又携带采样窗口元信息（避免为一个元信息改动所有调用签名）。"""

    def __init__(self, items: list[dict], meta: dict) -> None:
        super().__init__(items)
        self.meta = meta


def _load_recent_traces(db, limit: int = 10) -> list[dict] | None:
    """读取最近 limit 条**非空** Trace 的原始样本（单一取样点）。

    工具成功率、溯源率、成本、延迟共用同一批样本，避免拆成多次查询造成口径漂移。
    DB 不可用返回 None；样本为空返回 []。

    2026-09-16 修复（体检问题 6）：排除空轨迹 —— 实测真实库 179 条 trace 里
    **42 条（23%）为空**（steps=0、events=[]），且集中在最近的记录里（因此必然进采样窗口）。
    它们不代表"Agent 跑得很差"，而是"这一轮没走 Agent 循环"，混入分母只会制造假象。
    注意：**只修正分母，不虚构数字** —— 排除后若无样本，指标仍返回 None。
    """
    if db is None:
        return None
    try:
        with db() as session:
            rows = session.execute(
                text(
                    "SELECT trace_id, events_json, total_steps, total_cost, total_latency_ms, created_at "
                    "FROM traces "
                    f"WHERE NOT ({_EMPTY_TRACE_WHERE}) "
                    "ORDER BY id DESC LIMIT :lim"
                ).bindparams(lim=limit)
            ).fetchall()
            excluded = int(
                session.execute(
                    text(f"SELECT COUNT(*) FROM traces WHERE {_EMPTY_TRACE_WHERE}")
                ).scalar()
                or 0
            )
            total = int(session.execute(text("SELECT COUNT(*) FROM traces")).scalar() or 0)
    except Exception:  # noqa: BLE001 DB 不可用 → 跳过
        return None
    out: list[dict] = []
    for r in rows:
        tid, ev_json, tsteps, tcost, tlat, created = r
        try:
            evs = json.loads(ev_json or "[]")
        except Exception:  # noqa: BLE001
            evs = []
        out.append({
            "trace_id": tid,
            "events": evs,
            "steps": int(tsteps or 0),
            "cost": float(tcost or 0.0),
            "latency_ms": int(tlat or 0),
            "created_at": str(created) if created is not None else None,
        })
    # 把采样窗口信息挂在 list 上（保持返回类型不变，调用方无需改签名）
    out_window = {
        "limit": limit,
        "sampled": len(out),
        "table_total": total,
        "excluded_empty": excluded,
        "from": out[-1]["created_at"] if out else None,
        "to": out[0]["created_at"] if out else None,
    }
    # 2026-10-05：附上「样本时效 / 是否填满窗口」——看板据此出护栏，避免把过旧/过小的样本当现状读。
    out_window.update(_sample_quality(out_window))
    out_meta = {"window": out_window}
    return _TraceSamples(out, out_meta)


def _aggregate_agent(traces: list[dict] | None) -> dict:
    """从样本计算 Agent 指标：工具成功率 / 溯源率 / 平均成本 / 平均延迟 / 平均步数。

    工具成功率取所有事件中的 tool_call/tool_result 配对；溯源率取**完整回答**事件中带引用的占比；
    二者同源，口径一致（审查 P2-3）。

    2026-09-16（体检问题 4/6）修正三点，让"0"与"健康"都可被解释而不是误导：
      ① **分母排除空轨迹**（见 `_load_recent_traces`）—— 此前 42/179 条空 trace 混入，
         把平均步数从 3.9 拉低、平均成本进一步压向 0；
      ② 新增 `costed_count` —— 成本为 0 的样本**可能是真的**（未定价模型 / fake 模型），
         所以不能简单剔除；但必须让人看到"这 10 条里有几条真的计过价"，
         否则"平均成本 ¥0"既可能是"没采集"也可能是"都是免费模型"，无法区分；
      ③ 新增 `avg_latency_ms` —— `traces.total_latency_ms` 修好前一直是 0，
         看板索性没用它；现在两条路径都实测延迟，指标从无到有。
    """
    if traces is None:
        return {"available": False}
    window = getattr(traces, "meta", {}).get("window")
    if not traces:
        return {
            "available": True, "count": 0,
            "tool_success_rate": None, "citation_rate": None,
            "avg_cost": 0.0, "avg_steps": 0.0, "avg_latency_ms": 0.0,
            "costed_count": 0, "window": window, "traces": [],
        }
    from lightcore.rag.metrics import citation_rate, tool_success_rate  # 单一来源，延迟导入

    all_events: list[dict] = []
    costs = []
    steps = []
    latencies = []
    trace_summaries = []
    answers = []
    # T-09：每条轨迹的 events 单独留一份，交给**单一来源**做"有没有收尾"的分类
    # （此前本看板只看工具成功率/溯源率，缺"这轮究竟跑完了没"这一层）。
    trace_events: list = []
    for t in traces:
        all_events.extend(t["events"])
        trace_events.append(t["events"])
        costs.append(t["cost"])
        steps.append(t["steps"])
        latencies.append(t.get("latency_ms") or 0)
        trace_summaries.append({
            "trace_id": t["trace_id"], "steps": t["steps"], "cost": round(t["cost"], 4),
        })
        for e in t["events"]:
            # 「溯源率」的样本只取**完整回答**事件（`content` 非空），排除流式增量片段。
            # 2026-09-16 修复：此前把**所有** message 事件都当答案样本 —— 实测最近 10 条 trace 里
            # message 事件共 11297 个，其中 11279 个是 `delta` 片段、只有 18 个是 `content`，
            # 即便引用已落库也会被这些片段把占比稀释到约 0（与"读不到字段"构成双重错配）。
            if isinstance(e, dict) and e.get("type") == "message" and (e.get("data") or {}).get("content"):
                answers.append(e.get("data") or {})
    from lightcore.core import trace_metrics as tm  # T-09：口径的单一来源

    tsr = tool_success_rate(all_events) if all_events else None
    cite = citation_rate(answers) if answers else None
    outcome = tm.summarize_traces(trace_events)
    lat = tm.latency_percentiles(latencies)
    n_tr = outcome["total"]
    return {
        "available": True,
        "count": len(traces),
        "tool_success_rate": round(tsr, 4) if tsr is not None else None,
        "citation_rate": round(cite, 4) if cite is not None else None,
        "avg_cost": round(sum(costs) / len(costs), 4) if costs else 0.0,
        "costed_count": sum(1 for c in costs if c > 0),
        "avg_steps": round(sum(steps) / len(steps), 2) if steps else 0.0,
        # ★ 与分位**同源**（同样剔除未采集的 0）：否则会出现"avg 说 3 秒、p50 说 11 秒"
        "avg_latency_ms": round(lat["avg"]) if lat["samples"] else 0,
        # T-09：以下与 `/metrics` **同一口径**（同调 `lightcore.core.trace_metrics`），
        # 避免同一批 trace 在两个端点上算出两个数（本项目「两把尺子」的老账）。
        # ⚠️ 本看板只看**最近 N 条**（`_load_recent_traces` 的 window）⇒ 这里的 p95/p99
        #    是**小样本分位**，只用于看趋势，**不要当 SLA 数字读**（样本数见 `count`）。
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "latency_p99_ms": lat["p99"],
        # ⚠️ 分位只在**非零样本**上计算（0 = 未采集）；样本极小时看这两个数判断可信度
        "latency_samples": lat["samples"],
        "latency_missing": lat["missing"],
        "unfinished_breakdown": {
            "done": outcome["done"],
            "interrupted": outcome["interrupted"],
            "unknown": outcome["unknown"],
            "failed": outcome["failed"],
        },
        "unfinished_rate": round((n_tr - outcome["done"]) / n_tr, 4) if n_tr else 0.0,
        # ★ 判断"该不该修"用这个（已排除中断，但**含 unknown** —— 打点缺口不能当成功）
        "unfinished_rate_excluding_interrupt": round(
            (outcome["failed"] + outcome["unknown"]) / n_tr, 4
        ) if n_tr else 0.0,
        "failed_confirmed_rate": round(outcome["failed"] / n_tr, 4) if n_tr else 0.0,
        "window": window,
        "traces": trace_summaries,
    }


def _gate(ret: dict) -> dict:
    """检索基线门禁判定。**口径不可信时不出结论**（passed=None），绝不假通过。"""
    if not ret.get("available"):
        return {
            "threshold": _RECALL_GATE,
            "passed": None,
            "label": "无检索评测报告（先跑 python eval/run_eval.py --embedder real --profile prod）",
        }
    if not ret.get("trusted"):
        return {
            "threshold": _RECALL_GATE,
            "passed": None,
            "label": f"基线口径不可信，门禁不出结论：{ret.get('file')} —— {ret.get('reason') or '口径不符'}",
        }
    recall = ret.get("avg_recall@5")
    if recall is None:
        return {"threshold": _RECALL_GATE, "passed": None, "label": "评测报告缺少 recall@5 指标"}
    passed = recall >= _RECALL_GATE
    return {
        "threshold": _RECALL_GATE,
        "passed": passed,
        "label": f"达标（{ret.get('file')}）" if passed else f"未达基线（{ret.get('file')}）",
    }


def collect_metrics(db) -> dict:
    """结构化评测指标（单一数据口径来源）。

    /eval 技能渲染 Markdown 与 GET /api/v1/eval/summary 看板共用此函数，
    避免两处各算一遍导致指标口径漂移。
    """
    ret = _load_retrieval_report()
    gate = _gate(ret)

    agent = _aggregate_agent(_load_recent_traces(db))
    # 审查 P2-5：移除 agent.traces 冗余（与 recent_traces 同源），前端只消费 recent_traces
    recent_traces = (agent.pop("traces", []) or [])[:5]
    return {
        "retrieval": ret,
        "gate": gate,
        "agent": agent,
        "recent_traces": recent_traces,
    }


async def run(ctx: SkillContext) -> SkillResult:
    # 单一口径：与 GET /api/v1/eval/summary 看板共用 collect_metrics
    m = await asyncio.to_thread(collect_metrics, ctx.db)
    ret = m["retrieval"]
    gate = m["gate"]
    agent = m["agent"]

    lines = ["## 开发评测指标卡（dev_eval）", ""]
    lines.append(f"### 检索基线（{ret.get('file') or '无报告'}）")
    if ret["available"]:
        lines.append(f"- 用例集：{ret.get('cases_file') or '（老报告未记录）'}　样例数：{ret.get('num_cases')}")
        lines.append(f"- 取数口径：嵌入={ret.get('embedder')} / 检索 profile={ret.get('profile')}")
        lines.append(f"- avg Recall@5：{ret.get('avg_recall@5')}")
        lines.append(f"- avg MRR@10：{ret.get('avg_mrr@10')}")
        by_kind = ret.get("by_kind") or {}
        if by_kind:
            parts = [
                f"{k} (n={v.get('n')}, recall@5={v.get('recall@5')}, ndcg@5={v.get('ndcg@5')})"
                for k, v in sorted(by_kind.items())
            ]
            lines.append(f"- 分组：{'；'.join(parts)}")
    else:
        lines.append("- （缺失）")

    if gate["passed"] is None:
        gate_line = f"⏭️ {gate['label']}"
    elif gate["passed"]:
        gate_line = f"✅ 达标（avg_recall@5={ret['avg_recall@5']} ≥ 阈值 {_RECALL_GATE}）"
    else:
        gate_line = f"⚠️ 未达基线（avg_recall@5={ret['avg_recall@5']} < 阈值 {_RECALL_GATE}，建议重跑检索管线）"
    lines.append(f"\n### 门禁判定\n{gate_line}")
    w = agent.get("window") or {}
    lines.append("\n### 近期 Agent 指标（Trace 表）")
    if agent.get("available") and agent.get("count"):
        lines.append(f"- 采样轨迹数：{agent['count']}"
                     + (f"（表内共 {w.get('table_total')} 条，已排除 {w.get('excluded_empty')} 条空轨迹）"
                        if w else ""))
        if w.get("from") and w.get("to"):
            lines.append(f"- 采样窗口：{str(w['from'])[:19]} → {str(w['to'])[:19]}")
        # 2026-10-05：样本护栏随文本一并给出（与看板同口径），否则过旧/过小的样本会被当现状读。
        _notes = []
        if w.get("stale"):
            _notes.append(f"最新一条距今 {w.get('age_hours')}h（> {w.get('max_age_hours')}h，指标可能已过时）")
        if w.get("insufficient"):
            _notes.append(f"窗口未填满（{w.get('sampled')}/{w.get('limit')} 条，表内非空 {w.get('table_total')} 条）")
        if _notes:
            lines.append(f"- ⚠️ 样本可信度：{'；'.join(_notes)}")
        lines.append(f"- 工具成功率：{agent.get('tool_success_rate')}")
        lines.append(f"- 溯源率：{agent.get('citation_rate')}")
        lines.append(f"- 平均成本（¥）：{agent.get('avg_cost')}"
                     f"（其中 {agent.get('costed_count')}/{agent['count']} 条有计价）")
        lines.append(f"- 平均延迟：{agent.get('avg_latency_ms')} ms")
        lines.append(f"- 平均步数：{agent.get('avg_steps')}")
        lines.append("\n**调试轨迹（最近）**：")
        for t in m["recent_traces"]:
            lines.append(f"- `{t['trace_id']}`：steps={t['steps']}，cost={t['cost']}")
    else:
        lines.append("- （无 Trace 数据或 DB 不可用）")
    lines.append(
        "\n> 说明：忠实度（faithfulness）与跨厂商成本需真实 LLM + temperature=0 评估，"
        "见 judge_prompts/；本技能给出检索基线与运行期指标卡，作为阶段门禁参考。"
    )
    return SkillResult(ok=True, content="\n".join(lines), citations=[])


SPEC = SkillSpec(
    name="dev_eval",
    commands=["/eval", "/评测"],
    description="运行开发评测指标卡（检索基线 + Agent 指标 + 门禁判定）",
    handler=run,
    feature="dev_eval",
    icon="📊",
    category="评测",
)
