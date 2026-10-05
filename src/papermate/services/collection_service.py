"""PaperMate · 文献集服务（F5：文献集 / 项目分组 Collections）

- 文献集（Collection）是按课题/项目组织文档的分组载体；会话可关联整个文献集
  （session.config.collection_id），查询时由 chat_service / routes_search 解析为有效文档
  （文献集文档 ∪ 会话手动勾选文档）。
- 纯 SQLAlchemy（与项目其余 service 一致）；DB 表由 models/db.py 的 create_all 自动建出。
- 删除文献集默认不级联删文档（仅解除关联），符合"删除集合 ≠ 删除论文"的心智。

健壮性约定（2026-08-30 审查后加固）：
- 名称统一走 normalize_name（strip → 截断 → 空回落默认），create 与 rename 共用，
  避免两处策略不一致导致出现空白/空名文献集。
- doc_ids 写入前先去重（dict.fromkeys）并校验文档存在；不存在则抛 ValueError，
  由路由层转 400，避免 IntegrityError 冒泡成 500。
- 所有写操作经 write_with_retry，与 session_service 一致地应对 SQLite locked。
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.db import Collection, CollectionDocument, Document, iso_utc
from .db_retry import write_with_retry

_MAX_NAME_LEN = 100
_DEFAULT_NAME = "未命名文献集"


class CollectionService:
    def __init__(self, SessionLocal) -> None:
        self._Session = SessionLocal

    # ---------- 校验 / 归一化 ----------
    @staticmethod
    def normalize_name(name: str | None) -> str:
        """名称归一化：strip → 截断 → 空则回落默认名。create / rename 共用。"""
        s = (name or "").strip()
        if not s:
            return _DEFAULT_NAME
        return s[:_MAX_NAME_LEN]

    @staticmethod
    def _existing_doc_ids(s: Session, doc_ids: list[int]) -> set[int]:
        """返回确实存在的文档 id 集合；用于写入前校验，避免 FK 完整性错误冒泡成 500。"""
        if not doc_ids:
            return set()
        rows = s.query(Document.id).filter(Document.id.in_(doc_ids)).all()
        return {r[0] for r in rows}

    def _doc_ids_of(self, s: Session, cid: int) -> list[int]:
        rows = (
            s.query(CollectionDocument)
            .filter_by(collection_id=cid)
            .order_by(CollectionDocument.seq.asc())
            .all()
        )
        return [r.document_id for r in rows]

    # ---------- CRUD ----------
    def create(self, name: str, description: str = "", doc_ids: list[int] | None = None) -> int:
        def _create() -> int:
            with self._Session() as s:
                c = Collection(name=self.normalize_name(name), description=description or "")
                s.add(c)
                s.flush()
                cid = c.id
                wanted = list(dict.fromkeys(doc_ids or []))
                if wanted:
                    valid = self._existing_doc_ids(s, wanted)
                    missing = [d for d in wanted if d not in valid]
                    if missing:
                        # 抛出即回滚本会话（with 退出时 close），不留下半截文献集
                        raise ValueError(f"文档不存在或已删除：{missing}")
                    for seq, did in enumerate(wanted):
                        s.add(CollectionDocument(collection_id=cid, document_id=did, seq=seq))
                s.commit()
                return cid

        return write_with_retry(_create)

    def list(self) -> list[dict]:
        """批量取回，避免对每个集合各查一次关联（N+1）。"""
        with self._Session() as s:
            rows = s.query(Collection).order_by(Collection.created_at.asc()).all()
            if not rows:
                return []
            ids = [c.id for c in rows]
            links = (
                s.query(CollectionDocument)
                .filter(CollectionDocument.collection_id.in_(ids))
                .order_by(CollectionDocument.collection_id.asc(), CollectionDocument.seq.asc())
                .all()
            )
            by_cid: dict[int, list[int]] = {i: [] for i in ids}
            for ln in links:
                by_cid[ln.collection_id].append(ln.document_id)
            return [self._out(c, by_cid.get(c.id, [])) for c in rows]

    def get(self, cid: int) -> dict | None:
        with self._Session() as s:
            c = s.get(Collection, cid)
            return None if c is None else self._out(c, self._doc_ids_of(s, cid))

    def rename(self, cid: int, name: str | None = None, description: str | None = None) -> dict | None:
        def _rename() -> dict | None:
            with self._Session() as s:
                c = s.get(Collection, cid)
                if c is None:
                    return None
                if name is not None:
                    c.name = self.normalize_name(name)
                if description is not None:
                    c.description = description
                s.commit()
                return self._out(c, self._doc_ids_of(s, cid))

        return write_with_retry(_rename)

    def delete(self, cid: int) -> bool:
        def _delete() -> bool:
            with self._Session() as s:
                c = s.get(Collection, cid)
                if c is None:
                    return False
                # 显式先删关联行，再删集合（避免依赖 PRAGMA foreign_keys 行为差异）
                s.query(CollectionDocument).filter_by(collection_id=cid).delete()
                s.delete(c)
                s.commit()
                return True

        return write_with_retry(_delete)

    # ---------- 文档增删 ----------
    def add_document(self, cid: int, doc_id: int) -> bool:
        def _add() -> bool:
            with self._Session() as s:
                c = s.get(Collection, cid)
                if c is None:
                    return False
                exists = (
                    s.query(CollectionDocument)
                    .filter_by(collection_id=cid, document_id=doc_id)
                    .first()
                    is not None
                )
                if not exists:
                    if doc_id not in self._existing_doc_ids(s, [doc_id]):
                        raise ValueError(f"文档不存在或已删除：{doc_id}")
                    # 2026-09-09 修复 K3：seq 取当前最大值 +1——原先用 count() 当 seq，
                    # remove_document 删除中间文档后不重排，count < max(seq)+1，
                    # 新增行 seq 与现存行撞号（_doc_ids_of 按 seq 排序 → 文献集顺序漂移）。
                    max_seq = s.query(func.max(CollectionDocument.seq)).filter_by(collection_id=cid).scalar()
                    next_seq = (max_seq + 1) if max_seq is not None else 0
                    s.add(CollectionDocument(collection_id=cid, document_id=doc_id, seq=next_seq))
                    s.commit()
                return True

        return write_with_retry(_add)

    def remove_document(self, cid: int, doc_id: int) -> bool:
        def _remove() -> bool:
            with self._Session() as s:
                row = s.query(CollectionDocument).filter_by(collection_id=cid, document_id=doc_id).first()
                if row is None:
                    return False
                s.delete(row)
                s.commit()
                return True

        return write_with_retry(_remove)

    def resolve_doc_ids(self, cid: int | None) -> list[int]:
        """解析文献集文档列表（按 seq 排序）。cid 为空/不存在返回 []。"""
        if not cid:
            return []
        with self._Session() as s:
            return self._doc_ids_of(s, cid)

    # ---------- 工作间草稿/参考解析（A1，供 C1/C2 改稿接入消费） ----------
    def draft_ids(self, cid: int | None, draft_doc_id: int | None) -> list[int]:
        """工作间草稿文档列表：draft_doc_id 在文献集内则返回 [draft_doc_id]，否则 []。"""
        if not cid or draft_doc_id is None:
            return []
        docs = set(self.resolve_doc_ids(cid))
        return [draft_doc_id] if draft_doc_id in docs else []

    def ref_ids(self, cid: int | None, draft_doc_id: int | None) -> list[int]:
        """工作间内除草稿外的常规可检索文档（按文献集 seq 顺序）。"""
        docs = self.resolve_doc_ids(cid)
        if draft_doc_id is None:
            return docs
        return [d for d in docs if d != draft_doc_id]

    # ---------- 内部 ----------
    @staticmethod
    def _out(c: Collection, doc_ids: list[int]) -> dict:
        return {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "created_at": iso_utc(c.created_at),
            "doc_ids": doc_ids,
        }
