"""PaperMate · 用量与成本统计路由（docs/07 M3.1：用量与成本统计接口）

F10（2026-08-30 体验优化 P2）：
- 会话用量补齐「生效模型 / 最近活动」两个维度，模型取值与 ChatService.build_agent 的
  优先级一致（会话显式指定 > 全局默认），保证用量页与对话实际使用的模型口径统一；
- 新增按天聚合的用量趋势（成本趋势图数据源）。趋势与 /usage **同源**（同为 messages 表），
  因此趋势合计与会话用量合计天然可对账，不存在两套口径。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request

from .. import runtime_settings
from ..models.db import Message, ToolRun, Trace, iso_utc
from ..schemas import SessionUsageDetailOut, UsageRow, UsageTrendOut, UsageTrendPoint
from .json_safe import safe_json_dict, safe_json_records

router = APIRouter(tags=["usage"])

_MAX_TREND_DAYS = 365

#: 无消息会话的用量零值兜底（与 ChatService.count_usage 的空结果逐字段一致）
_EMPTY_USAGE: dict = {
    "num_messages": 0,
    "num_tool_runs": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost": 0.0,
    "last_at": None,
}


def _local_date(dt: datetime) -> str:
    """把消息时间归到「本地日历日」。

    SQLite 无原生时区类型，DateTime(timezone=True) 落库后读回多为 naive（语义为 UTC）；
    这里兼容 naive / aware 两种，否则趋势图会整体偏移一个时区。
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone().strftime("%Y-%m-%d")


def _message_model(meta_json: str | None) -> str | None:
    """从 `meta_json` 取本轮实际生效的模型名（坏 JSON / 老消息 → None，绝不伪造）。"""
    try:
        model = (safe_json_dict(meta_json) or {}).get("model")
    except Exception:  # noqa: BLE001 元数据损坏不应让用量接口整体 500
        return None
    return model if isinstance(model, str) and model else None


@router.get("/usage", response_model=list[UsageRow])
def usage(request: Request) -> list[UsageRow]:
    session_svc = request.app.state.session_svc
    chat_svc = request.app.state.chat_svc
    default_model = runtime_settings.get_state().get("default_model", "fake")
    # O-09（2026-10-01）：批量聚合。此前每会话各调一次 count_usage ⇒ 每会话 2 条聚合 SQL
    # （N+1）；现改为两条 GROUP BY 一次取回，结果与逐会话调用逐字段一致（有对拍测试）。
    usage_all = chat_svc.count_usage_all()
    rows = []
    for s in session_svc.list():
        u = usage_all.get(s.id) or _EMPTY_USAGE
        cfg = safe_json_dict(s.config_json)
        rows.append(UsageRow(
            session_id=s.id,
            title=s.title,
            num_messages=u["num_messages"],
            num_tool_runs=u["num_tool_runs"],
            prompt_tokens=u["prompt_tokens"],
            completion_tokens=u["completion_tokens"],
            cost=u["cost"],
            model=cfg.get("model") or default_model,
            last_at=iso_utc(u.get("last_at")),
        ))

    # ★★ 2026-10-23：把「服务类调用」也汇总进来。
    #   翻译/总结/提示词增强**没有会话**，此前完全不记账 ⇒ `/usage` 的总额是**黑账** ✗
    #   现按总量汇总成一行（session_id 用 -1 作哨兵，title 明示来源）。
    #   ⚠️ 用 -1 而不是 0/NULL：前端按 session_id 匹配会话，负数不会撞上任何真实会话。
    try:
        from sqlalchemy import func

        from ..models.db import ServiceUsage

        with chat_svc._Session() as db:  # noqa: SLF001  与 chat_svc 共用同一 sessionmaker
            agg = db.query(
                func.coalesce(func.sum(ServiceUsage.prompt_tokens), 0),
                func.coalesce(func.sum(ServiceUsage.completion_tokens), 0),
                func.coalesce(func.sum(ServiceUsage.cost), 0.0),
                func.count(ServiceUsage.id),
                func.max(ServiceUsage.created_at),
            ).one()
        if int(agg[3] or 0) > 0:  # 没有服务调用就不出这行，避免界面上多一条 0
            rows.append(UsageRow(
                session_id=-1,
                title="服务调用（翻译 / 总结 / 提示词增强）",
                num_messages=int(agg[3]),
                num_tool_runs=0,
                prompt_tokens=int(agg[0]),
                completion_tokens=int(agg[1]),
                cost=float(agg[2]),
                model="(多个)",
                last_at=iso_utc(agg[4]),
            ))
    except Exception:  # noqa: BLE001  聚合失败不能拖垮整页
        import logging as _lg

        _lg.getLogger(__name__).warning("[usage] 服务调用汇总失败", exc_info=True)
    return rows


@router.get("/usage/trend", response_model=UsageTrendOut)
def usage_trend(request: Request, days: int = 30) -> UsageTrendOut:
    """按天聚合的成本 / token 趋势（F10 成本趋势图数据源）。"""
    days = max(1, min(days, _MAX_TREND_DAYS))
    # SQLite 以文本比较 DATETIME，且落库时间为 naive UTC，故 cutoff 也用 naive UTC
    cutoff = (datetime.now(UTC) - timedelta(days=days - 1)).replace(tzinfo=None)
    session_svc = request.app.state.session_svc
    with session_svc.session_factory() as db:
        msgs = db.query(Message).filter(Message.created_at >= cutoff).all()
    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    )
    for m in msgs:
        if m.created_at is None:
            continue
        b = buckets[_local_date(m.created_at)]
        b["cost"] += m.cost or 0.0
        b["prompt_tokens"] += m.prompt_tokens or 0
        b["completion_tokens"] += m.completion_tokens or 0
    points = [
        UsageTrendPoint(
            date=d,
            cost=round(v["cost"], 6),
            prompt_tokens=int(v["prompt_tokens"]),
            completion_tokens=int(v["completion_tokens"]),
        )
        for d, v in sorted(buckets.items())
    ]
    return UsageTrendOut(days=days, points=points, total_cost=round(sum(p.cost for p in points), 6))


@router.get("/usage/sessions/{session_id}", response_model=SessionUsageDetailOut)
def session_usage_detail(session_id: int, request: Request) -> dict:
    """会话级用量明细：消息列表 + 工具调用 + Trace 轮次，供用量页展开查看（参考主流平台呈现）。"""
    session_svc = request.app.state.session_svc
    s = session_svc.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    with session_svc.session_factory() as db:
        messages = [
            {
                "id": m.id,
                "role": m.role,
                "content": (m.content[:200] + "…") if len(m.content or "") > 200 else (m.content or ""),
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "cost": round(m.cost or 0.0, 6),
                # 本轮**实际生效的模型**（2026-09-16，体检问题 7）：取自 meta_json.model，
                # 供用量页回答"这笔钱花在哪个模型上"。老消息（修复前落库）为 None —— 如实留空，
                # 不用会话当前模型倒填（那会伪造历史，与"该用什么"混淆）。
                "model": _message_model(m.meta_json),
            }
            for m in db.query(Message).filter(Message.session_id == session_id).order_by(Message.id).all()
        ]
        tool_runs = [
            {
                "id": t.id,
                "tool_name": t.tool_name,
                "args": (t.args_json or "")[:120],
                "result_summary": t.result_summary or "",
                "ok": bool(t.ok),
            }
            for t in db.query(ToolRun).filter(ToolRun.session_id == session_id).order_by(ToolRun.id).all()
        ]
        traces = [
            {
                "trace_id": t.trace_id,
                "total_steps": t.total_steps,
                "total_cost": round(t.total_cost or 0.0, 6),
                "total_latency_ms": t.total_latency_ms or 0,
                # 事件摘要（thinking/tool_call/message），避免全量 events 过大
                # ★ O-04：`events_json` 此前裸 json.loads（同列在 `routes_traces` 已有 `__safe_json`
                # 防护）⇒ 一条脏行让整个用量接口 500。统一走容错助手；
                # `safe_json_records` 额外过滤非 dict 元素（否则 `e.get` 又会 500）。
                "event_summary": [e.get("type") for e in safe_json_records(t.events_json)][:50],
            }
            for t in db.query(Trace).filter(Trace.session_id == session_id).order_by(Trace.id).all()
        ]
    u = request.app.state.chat_svc.count_usage(session_id)
    u["last_at"] = iso_utc(u.get("last_at"))
    cfg = safe_json_dict(s.config_json)
    u["model"] = cfg.get("model") or runtime_settings.get_state().get("default_model", "fake")
    return {
        "session_id": session_id,
        "title": s.title,
        "usage": u,
        "messages": messages,
        "tool_runs": tool_runs,
        "traces": traces,
    }
