"""PaperMate · Trace 路由（ER-19：推理轨迹可视化，docs/功能增强规格说明 ER-19）

- GET /traces?session_id=   ：列出该会话最近的若干 trace（含 trace_id / 步数 / 耗时）。
- GET /traces/{trace_id}    ：返回单条 trace 的完整事件流（前端渲染时间线用）。

Trace 事件已在持久化时附带 idx，前端按 idx 顺序渲染思维链 / 工具调用 / 结果 / 最终消息。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse

from ..models.db import Message, Trace, iso_utc
from ..schemas import EvalCaseOut, TraceListItemOut, TraceOut
from .json_safe import safe_json_list
from .pagination import DEFAULT_LIMIT, LimitParam, OffsetParam, paginate

router = APIRouter(tags=["traces"])


# ---------------------------------------------------------- E2（2026-09-21）：导出为评测样本
def _prev_user_content_from_list(
    user_msgs: list[Message], t: Trace,
) -> str:
    """从**预取**的该会话 user 消息（按 id 升序）里找轨迹之前最后一条（O-09 批量版）。

    语义与 `_prev_user_content` 的 SQL 完全一致：
    - `Message.created_at <= t.created_at`（SQL 里 NULL 不参与比较 ⇒ Python 侧显式排除 None）
    - `order_by(Message.id.desc()).first()`（取 id 最大的那条）
    """
    if t.created_at is None:
        return ""
    for m in reversed(user_msgs):
        if m.created_at is not None and m.created_at <= t.created_at:
            return m.content or ""
    return ""


def _prev_user_content(sess, t: Trace) -> str:
    """单条轨迹路径：SQL 找该轨迹之前最后一条 user 消息（导出单条时使用）。"""
    prev_user = (
        sess.query(Message)
        .filter(
            Message.session_id == t.session_id,
            Message.role == "user",
            Message.created_at <= t.created_at,
        )
        .order_by(Message.id.desc())
        .first()
    )
    return (prev_user.content or "") if prev_user is not None else ""


def _build_case(t: Trace, query: str) -> dict:
    """把一条轨迹组装成评测样本。

    ⚠️ **这里最关键的正确性**：`gold` **留空**，`candidates` 才是检索命中的块。
    把"检索命中的"当成标准答案 = 用系统自己的输出证明自己有效 —— **自证循环**，
    正是复检里点名的"用开发期样本自嗨"。所以导出的 case 必须**人工补 gold**后才能入评测集，
    这也是 `kind` 取 `candidate` 而非 `verbatim` 的原因。

    - `query`：取该轨迹**之前最后一条 user 消息**。轨迹事件里只有检索改写后的 query
      （`tool_call.arguments.query`），用改写 query 当评测输入，会掩盖
      「理解错了用户意图」这类失败。
    - `candidates`：从 `tool_result` 收集检索命中的 chunk id，供**人工标注**时参考。
    """
    events = safe_json_list(t.events_json)

    candidates: list[str] = []
    for e in events:
        if e.get("type") != "tool_result":
            continue
        out = (e.get("data") or {}).get("output") or {}
        for r in out.get("results") or []:
            cid = r.get("id")
            if cid and str(cid) not in candidates:
                candidates.append(str(cid))

    docs: dict[str, list[str]] = {}
    for cid in candidates:
        owner = cid.rsplit("-c", 1)[0] if "-c" in cid else cid
        docs.setdefault(owner, []).append(cid)

    return {
        "id": t.trace_id,
        "query": query,
        "kind": "candidate",
        "docs": docs,
        "gold": [],
        "candidates": candidates,
        "note": "gold 需人工确认后方可入评测集；candidates 仅为检索命中，直接当 gold 会构成自证循环",
        "meta": {
            "session_id": t.session_id,
            "trace_id": t.trace_id,
            "total_steps": t.total_steps,
            "total_cost": t.total_cost,
            "total_latency_ms": t.total_latency_ms,
            "created_at": iso_utc(t.created_at),
        },
    }


@router.get("/traces/export")
def export_traces(
    request: Request,
    session_id: int | None = Query(None),
    limit: int = Query(20, ge=1, le=200),
    fmt: str = Query("jsonl", pattern="^(jsonl|json)$"),
):
    """E2：批量导出轨迹为评测样本（jsonl 可直接被 `eval/run_eval.py` 消费）。

    ⚠️ **注册顺序**：本路由**必须排在 `/traces/{trace_id}` 之前**，
    否则字面量 `export` 会被 `/traces/{trace_id}` 抢先匹配成 trace_id（404）。
    """
    db = request.app.state.chat_svc._Session
    with db() as sess:
        q = sess.query(Trace)
        if session_id is not None:
            q = q.filter(Trace.session_id == session_id)
        rows = q.order_by(Trace.id.desc()).limit(limit).all()
        # O-09（2026-10-01）：批量预取涉及的 user 消息，替代每条 trace 各查一次 Message
        # （N+1）。涉及会话通常 ≤ 数个 ⇒ `in_()` 一次取回后按会话分组、内存挑选。
        sids = {t.session_id for t in rows}
        users_by_sess: dict[int, list[Message]] = {}
        if sids:
            msgs = (
                sess.query(Message)
                .filter(Message.session_id.in_(sids), Message.role == "user")
                .order_by(Message.id)
                .all()
            )
            for m in msgs:
                users_by_sess.setdefault(m.session_id, []).append(m)
        cases = [
            _build_case(t, _prev_user_content_from_list(users_by_sess.get(t.session_id, []), t))
            for t in rows
        ]

    if fmt == "json":
        return cases
    body = "\n".join(__json_line(c) for c in cases)
    return PlainTextResponse(
        content=body + ("\n" if body else ""),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="cases.jsonl"'},
    )


@router.get("/traces/{trace_id}/export", response_model=EvalCaseOut)
def export_trace(trace_id: str, request: Request) -> dict:
    """E2：导出单条轨迹为评测样本（同上，`gold` 留空待人工确认）。"""
    db = request.app.state.chat_svc._Session
    with db() as sess:
        t = sess.query(Trace).filter(Trace.trace_id == trace_id).first()
        if t is None:
            raise HTTPException(status_code=404, detail="轨迹不存在")
        return _build_case(t, _prev_user_content(sess, t))


@router.get("/traces", response_model=list[TraceListItemOut])
def list_traces(
    request: Request,
    response: Response,
    session_id: int | None = Query(None),
    limit: LimitParam = DEFAULT_LIMIT,
    offset: OffsetParam = 0,
) -> list[dict]:
    """轨迹列表（C4 2026-09-21：`limit` 由硬编码 50 改为可分页，总数走 `X-Total-Count`）。"""
    db = request.app.state.chat_svc._Session
    with db() as sess:
        q = sess.query(Trace)
        if session_id is not None:
            q = q.filter(Trace.session_id == session_id)
        rows = paginate(q.order_by(Trace.id.desc()), response, limit=limit, offset=offset)
        return [
            {
                "trace_id": t.trace_id,
                "session_id": t.session_id,
                "total_steps": t.total_steps,
                "total_cost": t.total_cost,
                "total_latency_ms": t.total_latency_ms,
                "created_at": iso_utc(t.created_at),
                "num_events": len(safe_json_list(t.events_json)),
            }
            for t in rows
        ]


@router.get("/traces/{trace_id}", response_model=TraceOut)
def get_trace(trace_id: str, request: Request) -> dict:
    db = request.app.state.chat_svc._Session
    with db() as sess:
        t = sess.query(Trace).filter(Trace.trace_id == trace_id).first()
        if t is None:
            raise HTTPException(status_code=404, detail="轨迹不存在")
        return {
            "trace_id": t.trace_id,
            "session_id": t.session_id,
            "total_steps": t.total_steps,
            "total_cost": t.total_cost,
            "total_latency_ms": t.total_latency_ms,
            "created_at": iso_utc(t.created_at),
            "events": safe_json_list(t.events_json),
        }


def __json_line(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
