"""PaperMate · 工作间服务（A1：工作间后端基座）

工作间 = 一个特殊会话（sessions.config_json 含 workspace/collection_id/draft_doc_id/mode）
        + 一个底层文献集（collections）。**零 DDL**，全部存于 JSON 字段 + 既有表。

原子性：POST /workspaces 在一个 DB 事务内同时创建文献集（CollectionDocument 行）
与工作间会话（ChatSession.config_json），任一失败整体回滚，避免半截文献集 / 孤儿会话。
"""

from __future__ import annotations

import json

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ..models.db import ChatSession, Collection, CollectionDocument, Document
from .db_retry import write_with_retry

_WORKSPACE_MODES = ("research", "revision")
_DEFAULT_WS_NAME = "未命名工作间"


class WorkspaceError(Exception):
    """工作间业务错误基类；路由层按子类映射状态码。"""


class WorkspaceNotFound(WorkspaceError):
    """关联的文献集不存在（→ 404）。"""


class WorkspaceService:
    def __init__(self, SessionLocal: sessionmaker) -> None:
        self._Session = SessionLocal

    # ---------- 校验 / 读取 ----------
    @staticmethod
    def _existing_doc_ids(s: Session, doc_ids: list[int]) -> set[int]:
        """返回确实存在的文档 id 集合（写入前校验，避免 FK 完整性错误冒泡成 500）。"""
        if not doc_ids:
            return set()
        rows = s.query(Document.id).filter(Document.id.in_(doc_ids)).all()
        return {r[0] for r in rows}

    @staticmethod
    def _doc_ids_of(s: Session, cid: int) -> list[int]:
        rows = (
            s.query(CollectionDocument)
            .filter_by(collection_id=cid)
            .order_by(CollectionDocument.seq.asc())
            .all()
        )
        return [r.document_id for r in rows]

    # ---------- 原子创建 ----------
    def create(
        self,
        *,
        name: str | None = None,
        description: str = "",
        doc_ids: list[int] | None = None,
        collection_id: int | None = None,
        draft_doc_id: int | None = None,
        mode: str = "research",
    ) -> dict:
        if mode not in _WORKSPACE_MODES:
            raise ValueError(f"mode 必须为 {_WORKSPACE_MODES} 之一，收到：{mode!r}")

        def _create() -> dict:
            with self._Session() as db:
                # 1) 解析 / 创建底层文献集（同一事务，保证原子）
                if collection_id is not None:
                    coll = db.get(Collection, collection_id)
                    if coll is None:
                        raise WorkspaceNotFound(collection_id)
                    cid = collection_id
                    # 工作间名可独立于文献集名：显式传 name 时以其为准，否则沿用文献集名
                    cname = (name or "").strip()[:100] or coll.name
                    wanted = self._doc_ids_of(db, cid)  # 以文献集为准，忽略传入 doc_ids
                else:
                    cname = (name or "").strip()[:100] or _DEFAULT_WS_NAME
                    wanted = list(dict.fromkeys(doc_ids or []))
                    valid = self._existing_doc_ids(db, wanted)
                    missing = [d for d in wanted if d not in valid]
                    if missing:
                        raise ValueError(f"文档不存在或已删除：{missing}")
                    coll = Collection(name=cname, description=description or "")
                    db.add(coll)
                    db.flush()
                    cid = coll.id
                    for seq, did in enumerate(wanted):
                        db.add(CollectionDocument(collection_id=cid, document_id=did, seq=seq))

                # 2) 草稿校验（须在文献集内且存在）
                if draft_doc_id is not None:
                    if draft_doc_id not in self._existing_doc_ids(db, [draft_doc_id]):
                        raise ValueError(f"草稿文档不存在或已删除：{draft_doc_id}")
                    if draft_doc_id not in wanted:
                        raise ValueError(f"草稿文档 {draft_doc_id} 不在工作间文献集内")

                # 3) 创建工作间会话（config_json 承载工作间元信息）
                config = {
                    "workspace": True,
                    "collection_id": cid,
                    "draft_doc_id": draft_doc_id,
                    "mode": mode,
                }
                sess = ChatSession(
                    title=cname,
                    config_json=json.dumps(config, ensure_ascii=False),
                )
                db.add(sess)
                db.flush()
                sid = sess.id
                db.commit()
                return {
                    "session_id": sid,
                    "collection_id": cid,
                    "name": cname,
                    "mode": mode,
                    "draft_doc_id": draft_doc_id,
                    "doc_ids": wanted,
                }

        try:
            return write_with_retry(_create)
        except WorkspaceNotFound:
            raise  # 路由层转 404
        except IntegrityError as e:  # 竞态兜底：校验通过后、提交瞬间文档被删
            raise ValueError("文档不存在或已删除，请刷新后重试") from e

    # ---------- 加入文档（B1） ----------
    def add_documents(self, session_id: int, doc_ids: list[int]) -> dict:
        """往工作间文献集批量加入文档。

        - 幂等：已在集内的文档自动跳过，不重复写入、不改其原有 seq；
        - 新增文档 seq 接在当前最大值之后，保证排序稳定（删除留空洞也不会插队）；
        - 返回加入后的工作间投影（供前端直接刷新左栏/文件区）。
        """
        if not doc_ids:
            raise ValueError("未选择文档")

        def _add() -> dict:
            with self._Session() as db:
                s = db.get(ChatSession, session_id)
                if s is None:
                    raise WorkspaceNotFound(session_id)
                cfg = json.loads(s.config_json or "{}")
                if not cfg.get("workspace"):
                    raise WorkspaceNotFound(session_id)
                cid = cfg.get("collection_id")
                if not cid or db.get(Collection, cid) is None:
                    raise WorkspaceNotFound(session_id)

                wanted = list(dict.fromkeys(doc_ids))
                valid = self._existing_doc_ids(db, wanted)
                missing = [d for d in wanted if d not in valid]
                if missing:
                    raise ValueError(f"文档不存在或已删除：{missing}")

                current = self._doc_ids_of(db, cid)
                already = set(current)
                to_add = [d for d in wanted if d not in already]
                if to_add:
                    max_seq = (
                        db.query(func.coalesce(func.max(CollectionDocument.seq), -1))
                        .filter_by(collection_id=cid)
                        .scalar()
                    )
                    for i, did in enumerate(to_add):
                        db.add(
                            CollectionDocument(
                                collection_id=cid, document_id=did, seq=int(max_seq) + 1 + i
                            )
                        )
                    db.commit()
                return self.resolve(session_id)

        try:
            return write_with_retry(_add)
        except WorkspaceNotFound:
            raise
        except IntegrityError as e:  # 竞态：校验通过后、提交瞬间文档被删
            raise ValueError("文档不存在或已删除，请刷新后重试") from e

    # ---------- 移除文档（B1 反向：从工作间文献集移除关联） ----------
    def remove_document(self, session_id: int, doc_id: int) -> dict:
        """从工作间底层文献集移除单一文档关联（**不删除源 Document 本身**，仅解绑）。

        - 非工作间 / 不存在 → WorkspaceNotFound（→ 404）
        - 文档不在该文献集内 → 无可移除项，WorkspaceNotFound（→ 404）
        - 若被移除文档恰为主文献（draft_doc_id），同步清除该标记，避免悬空引用
        - 返回移除后的工作间投影（供前端直接刷新左栏/文件区）
        """

        def _remove() -> dict:
            with self._Session() as db:
                s = db.get(ChatSession, session_id)
                if s is None:
                    raise WorkspaceNotFound(session_id)
                cfg = json.loads(s.config_json or "{}")
                if not cfg.get("workspace"):
                    raise WorkspaceNotFound(session_id)
                cid = cfg.get("collection_id")
                if not cid or db.get(Collection, cid) is None:
                    raise WorkspaceNotFound(session_id)

                link = (
                    db.query(CollectionDocument)
                    .filter_by(collection_id=cid, document_id=doc_id)
                    .one_or_none()
                )
                if link is None:
                    raise WorkspaceNotFound(doc_id)  # 文档不在该工作间文献集内，无可移除项

                # 被移除文档恰为主文献 → 清除 draft_doc_id，防止悬空引用
                if cfg.get("draft_doc_id") == doc_id:
                    cfg["draft_doc_id"] = None
                    s.config_json = json.dumps(cfg, ensure_ascii=False)

                db.delete(link)
                db.commit()
                return self.resolve(session_id)

        try:
            return write_with_retry(_remove)
        except WorkspaceNotFound:
            raise

    # ---------- 更新配置（B2：主文献指定；C1 复用同端点切模式） ----------
    def update(
        self,
        session_id: int,
        *,
        set_mode: bool = False,
        mode: str | None = None,
        set_draft: bool = False,
        draft_doc_id: int | None = None,
    ) -> dict:
        """PATCH 语义更新工作间 config：只写显式传入的字段（set_* 标志区分「未传」与「清除」）。

        - draft_doc_id：非 None 时须存在且位于文献集内；None 表示清除主文献；
        - mode：须为 {research, revision}。
        """
        if set_mode and mode not in _WORKSPACE_MODES:
            raise ValueError(f"mode 必须为 {_WORKSPACE_MODES} 之一，收到：{mode!r}")

        def _update() -> dict:
            with self._Session() as db:
                s = db.get(ChatSession, session_id)
                if s is None:
                    raise WorkspaceNotFound(session_id)
                cfg = json.loads(s.config_json or "{}")
                if not cfg.get("workspace"):
                    raise WorkspaceNotFound(session_id)
                cid = cfg.get("collection_id")

                if set_draft and draft_doc_id is not None:
                    in_collection = self._doc_ids_of(db, cid) if cid else []
                    if draft_doc_id not in self._existing_doc_ids(db, [draft_doc_id]):
                        raise ValueError(f"草稿文档不存在或已删除：{draft_doc_id}")
                    if draft_doc_id not in in_collection:
                        raise ValueError(f"草稿文档 {draft_doc_id} 不在工作间文献集内")

                if set_mode:
                    cfg["mode"] = mode
                if set_draft:
                    cfg["draft_doc_id"] = draft_doc_id
                s.config_json = json.dumps(cfg, ensure_ascii=False)
                db.commit()
                return self.resolve(session_id)

        return write_with_retry(_update)

    # ---------- 读取 ----------
    def resolve(self, session_id: int) -> dict | None:
        """从会话 config_json 解析工作间配置（只读）。非工作间 / 不存在 → None。"""
        with self._Session() as db:
            s = db.get(ChatSession, session_id)
            if s is None:
                return None
            cfg = json.loads(s.config_json or "{}")
            if not cfg.get("workspace"):
                return None
            cid = cfg.get("collection_id")
            draft = cfg.get("draft_doc_id")
            doc_ids = self._doc_ids_of(db, cid) if cid else []
            return {
                "session_id": s.id,
                "collection_id": cid,
                "name": s.title,
                "mode": cfg.get("mode", "research"),
                "draft_doc_id": draft,
                "doc_ids": doc_ids,
            }
