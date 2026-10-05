"""PaperMate · 会话路由（docs/04 §4.6.1）"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import or_

from ..models.db import ChatSession, Message, iso_utc
from ..schemas import (
    DocScopeOut,
    MessageOut,
    NetworkCapabilityOut,
    OkOut,
    SessionCreate,
    SessionDetailOut,
    SessionOut,
    SessionPin,
    SessionUpdate,
)
from .json_safe import safe_json_dict, safe_json_list
from .pagination import DEFAULT_LIMIT, LimitParam, OffsetParam

router = APIRouter(tags=["sessions"])


def _rule_title(text: str, max_chars: int = 16) -> str:
    """规则降级标题：取首行/首句截断 ≤max_chars 字（无 key 零成本回填历史会话标题）。"""
    for sep in ("\n", "。", "？", "！", ";", ". ", "? ", "! "):
        idx = text.find(sep)
        if 0 < idx <= max_chars:
            text = text[:idx]
            break
    return text[:max_chars].strip()


def _backfill_default_titles(svc) -> None:
    """懒回填：把标题仍为默认「新会话」的历史会话按首条 user 消息规则生成标题（幂等）。

    新会话的首轮标题由 LLM 生成（chat_service.auto_title）；此处覆盖存量/未对话的默认标题，
    保证侧栏不出现一屏「新会话」，且不覆盖人工标题。
    """
    from ..services.chat_service import _MM_PREFIX  # noqa: PLC0415 延迟导入避免循环依赖

    with svc._Session() as db:
        rows = db.query(ChatSession).filter(or_(ChatSession.title == "新会话", ChatSession.title == "")).all()
        changed = False
        for s in rows:
            first = (db.query(Message)
                     .filter(Message.session_id == s.id, Message.role == "user")
                     .order_by(Message.id).first())
            if first is None:
                continue
            t: str = first.content
            if t.startswith(_MM_PREFIX):  # 多模态块消息 → 取文本块
                try:
                    blocks = __import__("json").loads(t[len(_MM_PREFIX):])
                    t = "".join(b.get("text", "") for b in blocks
                                if isinstance(b, dict) and b.get("type") == "text")
                except Exception:  # noqa: BLE001 解析失败按纯文本处理
                    pass
            title = _rule_title(t)
            if title and title not in ("", "新会话"):
                s.title = title
                changed = True
        if changed:
            db.commit()


def _to_out(s) -> SessionOut:
    return SessionOut(
        id=s.id,
        title=s.title,
        # ★ O-04：`_to_out` 同时服务**会话列表**与详情 ⇒ 一条脏 `config_json` 会让整个列表 500。
        config=safe_json_dict(s.config_json),
        created_at=iso_utc(s.created_at),
        updated_at=iso_utc(s.updated_at),
        # 置顶（2026-10-02）：`bool()` 把 DB 的 0/1 转成 JSON 布尔。
        # ★ 用 `getattr` 兜底：迁移前的老对象/测试替身没有这一列时不应 500
        # （漏填则前端永远读到 False ⇒ 置顶"看起来没生效"）。
        pinned=bool(getattr(s, "pinned", 0)),
    )


@router.post("/sessions", response_model=SessionOut)
def create_session(body: SessionCreate, request: Request) -> SessionOut:
    svc = request.app.state.session_svc
    s = svc.create(title=body.title, config=body.config)
    return _to_out(s)


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(
    request: Request,
    response: Response,
    limit: LimitParam = DEFAULT_LIMIT,
    offset: OffsetParam = 0,
) -> list[SessionOut]:
    """会话列表（C4 2026-09-21：支持 `limit`/`offset`，总数走 `X-Total-Count`）。"""
    svc = request.app.state.session_svc
    # 懒回填默认标题（幂等），保证历史会话侧栏显示主题而非「新会话」
    _backfill_default_titles(svc)
    response.headers["X-Total-Count"] = str(svc.count())
    return [_to_out(s) for s in svc.list(limit=limit, offset=offset)]


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
def get_session(session_id: int, request: Request) -> SessionDetailOut:
    svc = request.app.state.session_svc
    s = svc.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    with svc._Session() as db:
        rows = db.query(Message).filter(Message.session_id == session_id).order_by(Message.id).all()
        messages = []
        for m in rows:
            content: str = m.content
            images: list[str] = []
            # 多模态：_MM_PREFIX 标记的块 JSON → content 取纯文本（text 块拼接），图片 URL 单独返回
            from ..services.chat_service import _MM_PREFIX  # noqa: PLC0415 延迟导入避免循环依赖
            if m.content.startswith(_MM_PREFIX):
                try:
                    blocks = json.loads(m.content[len(_MM_PREFIX):])
                    content = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
                    images = [b["image_url"]["url"] for b in blocks
                              if isinstance(b, dict) and b.get("type") == "image_url" and b.get("image_url")]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # 无法解析时按纯文本展示
            # B2-X1（2026-10-01）：引用从 meta 里**提升为独立字段**（见 MessageOut.citations），
            # 并从 meta 中移除 —— 否则同一份引用会在响应里出现两次（载荷翻倍）。
            # 顺带把 meta 解析做**容错**：损坏的 meta_json 不该让整页历史 500
            # （与该不变量在用量接口上的既有测试同口径，见 test_usage_persist::test_corrupt_meta_json_*）。
            raw_meta = getattr(m, "meta_json", None)
            meta: dict = {}
            if raw_meta:
                try:
                    parsed = json.loads(raw_meta)
                    meta = parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            msg_citations = meta.pop("citations", [])
            if not isinstance(msg_citations, list):
                msg_citations = []
            messages.append(MessageOut(
                id=m.id, role=m.role, content=content, images=images,
                # ★ O-04（2026-10-01）：`tool_calls_json` 此前**裸 json.loads** —— 而**同函数
                # 3 行前**的 `meta_json` 有 try/except（注释明写"损坏不该让整页历史 500"）。
                # 一条脏行 ⇒ 整个会话历史接口 500。统一走容错助手（同源化）。
                tool_calls=safe_json_list(m.tool_calls_json) if m.tool_calls_json else None,
                prompt_tokens=m.prompt_tokens, completion_tokens=m.completion_tokens, cost=m.cost,
                meta=meta,
                citations=msg_citations,
                created_at=iso_utc(m.created_at),
            ))
    usage = request.app.state.chat_svc.count_usage(session_id)
    out = _to_out(s)
    return SessionDetailOut(**out.model_dump(), messages=messages, usage=usage)


@router.get("/sessions/{session_id}/doc-scope", response_model=DocScopeOut)
def get_doc_scope(session_id: int, request: Request) -> dict:
    """当前会话的**检索范围**（2026-09-23，场景可见性）。

    三场景（无文档 / 单文档 / 多文档）下检索行为不同，但界面**看不出**自己处于哪个场景 ——
    用户会误判「约束没生效」（实测中最常见的是：会话其实**没绑文档**，于是走了通用对话）。

    只报**会话级**范围：本轮是否因 `@` 聚焦属每轮变化，由前端自行标注（它知道 @ 了什么）。
    """
    svc = request.app.state.session_svc
    if svc.get(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    chat_svc = request.app.state.chat_svc
    doc_ids = chat_svc._effective_doc_ids(session_id)

    # 取文件名用于展示（拿不到就退回 `#id`，不让展示失败拖垮接口）
    names: dict[int, str] = {}
    try:
        for d in request.app.state.doc_svc.list_documents(limit=None):
            if d.id in set(doc_ids):
                names[d.id] = d.filename
    except Exception:  # noqa: BLE001  纯展示用途，失败不阻断
        names = {}

    if not doc_ids:
        mode = "general"
    elif len(doc_ids) == 1:
        mode = "single"
    else:
        mode = "multi"
    return {
        "session_id": session_id,
        "doc_count": len(doc_ids),
        "doc_ids": doc_ids,
        "documents": [{"id": i, "filename": names.get(i, f"#{i}")} for i in doc_ids],
        "force_retrieval": chat_svc.should_force_tool_call(session_id),
        "mode": mode,
    }


@router.get("/sessions/{session_id}/network-capability", response_model=NetworkCapabilityOut)
def get_network_capability(session_id: int, request: Request) -> dict:
    """当前会话的**联网可用性 + 模型可用性**（2026-09-12 新增，供前端「可用性预警」）。

    为什么需要这个端点（实测结论）：联网是**模型原生能力**（`enable_search=True`），而
    - 仅部分模型支持：实测 qwen3-max / qwen-max / qwen3.6-flash / qwen3.7-flash / qwen-turbo 支持；
      qwen-plus、qwen3-vl-flash 与 DeepSeek 系列**不支持**；
    - 绑定文档（doc-bound）的会话若选择了「绑文档不可靠」的模型（如 qwen3-vl-flash），
      回答可能跳过检索（2026-09-16 起不再静默回落，见 `doc_bound_model_risk`）——
      前端应在此拿到 `model_risk` 后**常驻提示**「这轮回答请核对引用」，发送不受阻。

    因此「开关已开」≠「能力可用」。前端必须据此给出预警，否则用户会误以为
    「模型联网了却不去搜」（实际是**根本没有联网能力**）。
    """
    svc = request.app.state.session_svc
    if svc.get(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    chat_svc = request.app.state.chat_svc
    cfg = chat_svc._sessions.get_config(session_id)
    enabled = bool(cfg.get("enable_network", False))
    effective = chat_svc.effective_model(session_id)
    # 「不确定」必须与「确定不支持」区分（2026-09-17）：取实例失败时若一律当成"不支持"，
    # 前端会给出**错误归因**（"当前模型不支持"），而真实原因可能是配置读取失败。
    supports: bool | None
    try:
        supports = bool(chat_svc._llm_from_raw(effective).supports_native_web_search())
    except Exception:  # noqa: BLE001 取实例失败 → 能力**未知**（保守：仍按不可用处理）
        supports = None
    reason = ""
    if enabled and supports is False:
        reason = f"当前模型 {effective} 不支持原生联网 → 联网开关不生效"
    elif enabled and supports is None:
        reason = f"无法确认模型 {effective} 的联网能力（读取模型配置失败），联网开关可能不生效"
    # doc-bound 模型风险（2026-09-16 晚，策略 = 警示但放行）：不可靠组合在此**提前**暴露，
    # 让用户打字时就知道"这轮回答要核对引用"（若要可靠检索可自行换模型），发送不受阻。
    risk = chat_svc.doc_bound_model_risk(session_id)
    return {
        "session_id": session_id,
        "enabled": enabled,
        # ⚠️ 语义 = 「**当前是否真的可用**」= 开关已开 **且** 能力具备。
        # 注意它**不是**"能力是否具备"—— 开关关着时它同样为 False。前端判断
        # 「开关已开但能力不具备」必须用 `supports_native`，不能只看 `available`
        # （2026-09-17：前端曾用 `webOn && !available`，而 `webOn` 是乐观更新、
        #  `available` 来自陈旧响应 → 两者短暂矛盾 → 提示条闪现约 50ms 后消失）。
        "available": enabled and bool(supports),
        "supports_native": bool(supports),
        "effective_model": effective,
        "reason": reason,
        "model_risk": bool(risk),
        "model_risk_reason": risk,
    }


@router.patch("/sessions/{session_id}", response_model=SessionOut)
def update_session(session_id: int, body: SessionUpdate, request: Request) -> SessionOut:
    svc = request.app.state.session_svc
    s = svc.update(session_id, title=body.title, config=body.config)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _to_out(s)


@router.patch("/sessions/{session_id}/pin", response_model=SessionOut)
def pin_session(session_id: int, body: SessionPin, request: Request) -> SessionOut:
    """置顶/取消置顶（2026-10-02）。

    ★ 走**独立端点**而非复用 `PATCH /sessions/{id}`：
    置顶要保证 `updated_at` **不变**（否则"取消置顶"会让该项跳到非置顶区顶部，
    看起来像刚被改动过），而改标题/改配置**应该**刷新 `updated_at`
    ⇒ 两条路径的 `updated_at` 语义**相反**，无法共用一个 `if x is not None` 的合并更新。
    """
    svc = request.app.state.session_svc
    s = svc.set_pinned(session_id, body.pinned)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _to_out(s)


@router.delete("/sessions/{session_id}", response_model=OkOut)
def delete_session(session_id: int, request: Request) -> dict:
    svc = request.app.state.session_svc
    ok = svc.delete(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}
