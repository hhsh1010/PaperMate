"""PaperMate · 反馈路由（E1，2026-09-21）

- POST   /feedback              ：提交一条 👍/👎（同一消息可改分，取最后一次为准）
- GET    /feedback              ：列出反馈（可按会话过滤）
- GET    /feedback/summary      ：汇总（总数 / 好评 / 差评 / 每条消息的当前评分，供前端回填）

**为什么要有这张表（不只是"加个按钮"）**：
`docs/五阶段落地清单-复检-20260920.md` 的核心结论是「缺的不是功能，而是**真实证据**」——
评测集此前只能人工造（样本失真）。反馈表 + `GET /traces/{id}/export` 一起构成
**"真实问题 + 真实轨迹 → 评测样本"**的回流链路，是 A2（答案层评测）与 A3（判官）的数据前提。

**设计取舍**：
- `rating` 只接受 `+1 / -1`（见 `schemas.FeedbackCreate`）：聚合可求和/均值，脏数据无处藏。
- **不设唯一约束**：允许改分，"改主意"是反馈场景的高频行为，409 会让按钮点不动。
  查询时按 `id` 升序覆盖，得到"最后一次为准"的语义。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response

from ..models.db import ChatSession, Feedback, iso_utc
from ..schemas import FeedbackCreate, FeedbackOut, FeedbackSummaryOut
from .pagination import DEFAULT_LIMIT, LimitParam, OffsetParam, paginate

router = APIRouter(tags=["feedback"])


def _to_out(fb: Feedback) -> dict:
    return {
        "id": fb.id,
        "session_id": fb.session_id,
        "message_id": fb.message_id,
        "trace_id": fb.trace_id or "",
        "rating": fb.rating,
        "comment": fb.comment or "",
        "created_at": iso_utc(fb.created_at),
    }


@router.post("/feedback", response_model=FeedbackOut)
def create_feedback(payload: FeedbackCreate, request: Request) -> dict:
    """提交反馈。

    校验顺序：**先 rating 后 session** —— rating 是纯入参问题（422），
    session 是资源问题（404）；把入参校验放前面，用户能看到更直接的错误原因。
    """
    if payload.rating not in (1, -1):
        raise HTTPException(status_code=422, detail="rating 只能是 1（好评）或 -1（差评）")

    db = request.app.state.chat_svc._Session
    with db() as sess:
        if sess.get(ChatSession, payload.session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        fb = Feedback(
            session_id=payload.session_id,
            message_id=payload.message_id,
            trace_id=payload.trace_id or "",
            rating=payload.rating,
            comment=payload.comment or "",
        )
        sess.add(fb)
        sess.commit()
        sess.refresh(fb)
        return _to_out(fb)


@router.get("/feedback", response_model=list[FeedbackOut])
def list_feedback(
    request: Request,
    response: Response,
    session_id: int | None = Query(None),
    limit: LimitParam = DEFAULT_LIMIT,
    offset: OffsetParam = 0,
) -> list[dict]:
    """反馈列表（C4：与其它列表端点统一分页口径，总数走 `X-Total-Count`）。"""
    db = request.app.state.chat_svc._Session
    with db() as sess:
        q = sess.query(Feedback)
        if session_id is not None:
            q = q.filter(Feedback.session_id == session_id)
        rows = paginate(q.order_by(Feedback.id.desc()), response, limit=limit, offset=offset)
        return [_to_out(f) for f in rows]


@router.get("/feedback/summary", response_model=FeedbackSummaryOut)
def feedback_summary(request: Request, session_id: int | None = Query(None)) -> dict:
    """反馈汇总。

    `rated_message_ids` 返回**每条消息当前的评分**（后写的覆盖先写的 = 最后一次为准），
    前端据此回填按钮高亮 —— 否则刷新后按钮全灭，用户会以为"反馈没记上"。
    """
    db = request.app.state.chat_svc._Session
    with db() as sess:
        q = sess.query(Feedback)
        if session_id is not None:
            q = q.filter(Feedback.session_id == session_id)
        rows = q.order_by(Feedback.id.asc()).all()

    rated: dict[int, int] = {}
    positive = negative = 0
    for f in rows:
        if f.rating > 0:
            positive += 1
        elif f.rating < 0:
            negative += 1
        if f.message_id is not None:
            rated[f.message_id] = f.rating  # 升序遍历 → 最后一次覆盖
    return {
        "total": len(rows),
        "positive": positive,
        "negative": negative,
        "rated_message_ids": rated,
    }
