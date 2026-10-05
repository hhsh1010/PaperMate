"""PaperMate · 会话服务（docs/07 M3.1）

会话 CRUD + 把持久化的消息重建为 LightCore 的 ChatMessage 历史（user/assistant/tool 回环），
供 AgentLoop 多轮上下文使用。
"""

from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from lightcore.llm.base import ChatMessage, Role, ToolCall

from ..models.db import ChatSession, Message
from .db_retry import write_with_retry


def _role_of(name: str) -> Role:
    return Role(name)


def _safe_loads(raw: str | None, fallback):
    """JSON 容错解析（2026-09-09 修复 M8）。

    DB 里的 JSON 是「历史数据」——可能来自旧版本格式、写入中断或手工修改。原实现
    `json.loads(...)` 直接抛异常会让**整个会话历史/配置加载失败**（会话彻底不可用），
    而非仅跳过损坏的那一条。逐条降级优于整体失败。
    """
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _safe_dict(raw: str | None) -> dict:
    """容错解析并保证 dict 类型（`.get` 调用方需要）。"""
    v = _safe_loads(raw, {})
    return v if isinstance(v, dict) else {}


class SessionService:
    def __init__(self, SessionLocal: sessionmaker) -> None:
        self._Session = SessionLocal

    @property
    def session_factory(self) -> sessionmaker:
        """公开的 DB 会话工厂，供路由等外部消费（P2-8：替代直接访问私有 _Session）。"""
        return self._Session

    def create(self, title: str | None = None, config: dict | None = None) -> ChatSession:
        def _create() -> ChatSession:
            with self._Session() as db:
                s = ChatSession(title=title or "新会话", config_json=json.dumps(config or {}, ensure_ascii=False))
                db.add(s)
                db.commit()
                db.refresh(s)
                return s

        return write_with_retry(_create)

    def count(self) -> int:
        """会话总数（C4 分页用），与 `list` 同源。"""
        with self._Session() as db:
            return db.query(ChatSession).count()

    def list(self, limit: int | None = None, offset: int = 0) -> list[ChatSession]:
        """会话列表：**置顶优先**，组内按最近活动倒序。

        C4（2026-09-21）：新增 `limit`/`offset`，**默认全量**以保持既有调用方行为不变。

        ★ 置顶排序（2026-10-02，`pinned` 列 · 迁移 5）：
        `ORDER BY pinned DESC, updated_at DESC` ——
        - `pinned DESC`：置顶（1）排在非置顶（0）之前；
        - `updated_at DESC`：组内仍按最近活动，**置顶项之间也按活动时间**，
          这样新点开的置顶项会浮到置顶区顶部（"我刚聊过的应该看得到"）。
        ⚠️ **不能**只按 `updated_at` 排、再在前端分组 —— 那样"置顶"只是**视觉分组**，
        用户新点开一个旧会话，它会因活动时间变化掉进非置顶区，**看起来像"置顶失效"**。
        """
        with self._Session() as db:
            q = db.query(ChatSession).order_by(
                ChatSession.pinned.desc(), ChatSession.updated_at.desc(),
            )
            if limit is not None:
                q = q.offset(offset).limit(limit)
            return q.all()

    def get(self, session_id: int) -> ChatSession | None:
        with self._Session() as db:
            return db.get(ChatSession, session_id)

    def get_config(self, session_id: int) -> dict:
        s = self.get(session_id)
        if s is None:
            return {}
        return _safe_dict(s.config_json)

    def update(self, session_id: int, title: str | None = None, config: dict | None = None) -> ChatSession | None:
        def _update() -> ChatSession | None:
            with self._Session() as db:
                s = db.get(ChatSession, session_id)
                if s is None:
                    return None
                if title is not None:
                    s.title = title
                if config is not None:
                    # 合并而非整体覆盖，避免丢失 doc_ids 等其它字段。
                    # 但支持「显式置 None 即删除该键」——用于前端取消关联（如 collection_id: null 解绑文献集）。
                    cur = _safe_dict(s.config_json)
                    for _k, _v in config.items():
                        if _v is None:
                            cur.pop(_k, None)
                        else:
                            cur[_k] = _v
                    s.config_json = json.dumps(cur, ensure_ascii=False)
                db.commit()
                db.refresh(s)
                return s

        return write_with_retry(_update)

    def set_pinned(self, session_id: int, pinned: bool) -> ChatSession | None:
        """置顶/取消置顶（2026-10-02）。

        ★ 刻意**独立于 `update`**：置顶是**列表排序键**，而 `update` 只碰
        `title` / `config_json`。混在一起会让"改标题"意外影响排序，
        而且前端只想改一个布尔值却必须**回传整个 config**（那份 config 很可能已过期，
        合并会覆盖掉别的设置——与重命名那次踩的是同一类坑）。

        ★★ **必须保持 `updated_at` 不变**（2026-10-02 实测纠正，试了两种写法）：
        - 我最初以为「不动updated_at」就行——**错**：`Column(onupdate=_now)`
          是**列/语句级**默认值，任何 UPDATE 都会用它**覆盖**我在 ORM 里回写的值；
        - 改成「先取 keep、赋值 `s.updated_at = keep`」**仍无效**（实测 True），
          因为 `onupdate` 在 flush 时**再次**赋值。

        ⇒ 唯一可靠做法：**用原生 SQL 显式回写** `updated_at`（实测生效）。
        真正要防的是「**取消**置顶时看起来像刚被改动过」：
        - 置顶 ⇒ 活动时间不变（只是排序变了）；
        - 取消置顶 ⇒ 活动时间不变（回到它本来的位置，而不是跳到非置顶区顶部）。
        """
        def _do() -> ChatSession | None:
            with self._Session() as db:
                s = db.get(ChatSession, session_id)
                if s is None:
                    return None
                keep = s.updated_at
                val = 1 if pinned else 0
                # ★ 原生 SQL：绕开 `onupdate=_now`（ORM 层无论怎么赋值都会被它覆盖）
                db.execute(
                    text("UPDATE sessions SET pinned=:p, updated_at=:u WHERE id=:i"),
                    {"p": val, "u": keep, "i": session_id},
                )
                db.commit()
                db.refresh(s)
                return s

        return write_with_retry(_do)

    def delete(self, session_id: int) -> bool:
        def _delete() -> bool:
            with self._Session() as db:
                s = db.get(ChatSession, session_id)
                if s is None:
                    return False
                db.delete(s)  # 级联删 messages/tool_runs/traces
                db.commit()
                return True

        return write_with_retry(_delete)

    def load_history(self, session_id: int) -> list[ChatMessage]:
        """把 DB 消息重建为 ChatMessage 历史（与 core/loop.py 期望格式一致）。

        多模态：user 消息若为 _MM_PREFIX 标记的 JSON 内容块，还原为 text/image_url 块列表，
        保证切换模型/刷新后图片上下文延续（数据一致性）。
        """
        from .chat_service import _MM_PREFIX, _decode_message_content

        with self._Session() as db:
            rows = db.query(Message).filter(Message.session_id == session_id).order_by(Message.id).all()
            history: list[ChatMessage] = []
            for m in rows:
                try:
                    role = _role_of(m.role)
                except ValueError:
                    continue  # 角色值损坏（历史脏数据）：跳过该条，不让整个会话历史加载失败（M8）
                content: str | list[dict] = m.content
                if m.content.startswith(_MM_PREFIX):
                    content = _decode_message_content(m.content)
                if role == Role.TOOL:
                    meta = _safe_dict(m.tool_calls_json)
                    history.append(ChatMessage(role=Role.TOOL, content=content,
                                               tool_call_id=str(meta.get("tool_call_id", ""))))
                elif role == Role.ASSISTANT and m.tool_calls_json:
                    # M8：tool_calls_json 逐条容错——单条缺 id/name 或整体损坏只丢该条/该消息，
                    # 不让整个会话历史加载失败。arguments 显式防 None。
                    raw_calls = _safe_loads(m.tool_calls_json, [])
                    calls = [
                        ToolCall(id=str(tc.get("id", "")), name=str(tc.get("name", "")),
                                 arguments=tc.get("arguments") or {})
                        for tc in (raw_calls if isinstance(raw_calls, list) else [])
                        if isinstance(tc, dict)
                    ]
                    history.append(ChatMessage(role=Role.ASSISTANT, content=content, tool_calls=calls))
                else:
                    history.append(ChatMessage(role=role, content=content))
            return history
