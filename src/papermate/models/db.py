"""PaperMate · SQLAlchemy 数据层（docs/06 表结构：会话/消息/文档/块/记忆/工具记录/trace/向量映射）

- SQLite 单机零依赖；DAO 通过 services 隔离，便于日后迁 PostgreSQL。
- 会话删除 → 级联删 messages/tool_runs/traces；文档删除 → 级联删 chunks。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import declarative_base, relationship

from .migrations import current_version, run_migrations

Base = declarative_base()

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


def iso_utc(dt: datetime | None) -> str | None:
    """把（可能 naive 的）UTC 时间序列化为带时区标记的 ISO 字符串。

    SQLite 的 DateTime(timezone=True) 落库后读回多为 naive（语义 UTC）。若不加时区标记，
    前端 new Date() 会按浏览器本地时区解析，导致 GMT+8 下偏移 +8h（审查 P2-1）。统一补回
    UTC（naive 视为 UTC，aware 转 UTC），保证前后端对时间的理解一致。
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False, default="新会话")
    config_json = Column(String, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    # 侧栏置顶（2026-10-02，迁移 5）：0=否。★ 必须是**列**而非 config 项——
    # 列表排序在 SQL 层（`session_service.list` 的 `order_by`），排序键在列上。
    pinned = Column(Integer, nullable=False, default=0)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    tool_runs = relationship("ToolRun", back_populates="session", cascade="all, delete-orphan")
    traces = relationship("Trace", back_populates="session", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False, default="")
    tool_calls_json = Column(Text)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    cost = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), default=_now)
    # 消息级元数据（JSON）：承载结构化提示/警告（如无效回复 notice、invalid 标记），
    # 供前端刷新后从历史回填「打断提醒」卡片（见 routes_search._build_notice）。
    meta_json = Column(String, nullable=False, default="{}")

    session = relationship("ChatSession", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="uploaded")
    content_hash = Column(String, nullable=False, default="")  # sha256，重复上传检测（2026-08-20）
    meta_json = Column(String, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), default=_now)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    seq = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    page = Column(Integer)
    heading = Column(String)
    meta_json = Column(String, nullable=False, default="{}")

    document = relationship("Document", back_populates="chunks")


# F5：文献集（Collections）——按课题/项目组织文档；会话可关联整个文献集（config.collection_id），
# 查询时解析为 doc_ids 并集（文献集 ∪ 会话手动勾选）。新建表由 create_all 自动建出。
class Collection(Base):
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, default="未命名文献集")
    description = Column(String, nullable=False, default="")
    config_json = Column(String, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    documents = relationship(
        "CollectionDocument", back_populates="collection",
        cascade="all, delete-orphan", passive_deletes=True,
    )


class CollectionDocument(Base):
    __tablename__ = "collection_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_id = Column(Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    seq = Column(Integer, nullable=False, default=0)

    collection = relationship("Collection", back_populates="documents")

    # 唯一约束：同一文献集内同一文档只允许一条关联（旧库该索引非唯一，由 _migrate 清洗后重建）。
    # 写入侧已用 dict.fromkeys 去重，此处兜底防脏数据。
    __table_args__ = (Index("idx_coldoc", "collection_id", "document_id", unique=True),)


class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, default="")
    content = Column(Text, nullable=False)
    importance = Column(Float, nullable=False, default=0.0)
    access_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=_now)
    last_accessed_at = Column(DateTime(timezone=True), default=_now)


class ToolRun(Base):
    __tablename__ = "tool_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    tool_name = Column(String, nullable=False)
    args_json = Column(String, nullable=False, default="{}")
    result_summary = Column(String, nullable=False, default="")
    ok = Column(Boolean, nullable=False, default=True)
    latency_ms = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=_now)

    session = relationship("ChatSession", back_populates="tool_runs")


class Trace(Base):
    __tablename__ = "traces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String, nullable=False, default="")
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    events_json = Column(Text, nullable=False, default="[]")
    total_steps = Column(Integer, nullable=False, default=0)
    total_cost = Column(Float, nullable=False, default=0.0)
    total_latency_ms = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=_now)

    session = relationship("ChatSession", back_populates="traces")


class Feedback(Base):
    """E1（2026-09-21）：回答反馈 —— 👍/👎 回流，评测样本的**真实来源**。

    **为什么这张表是批次 1 的核心**：复检结论「缺的不是功能，而是真实证据」——
    此前评测集只能人工造（`docs/五阶段落地清单-复检-20260920.md` §0）。
    有了它，`GET /feedback/export` 才能把「真实问题 + 真实轨迹」导出成 eval case，
    从根上解决样本失真。

    设计取舍（都有记录价值）：
    - `rating` 用 **+1 / -1 整数**而非枚举字符串：聚合时可直接求和/均值，
      不必在 SQL 里比较字符串，也避免新增枚举时漏改聚合逻辑。
    - `message_id` / `trace_id` **可空**：用户可能在流式未结束、或轨迹未生成时打分，
      强制非空会把"想反馈却找不到锚点"直接变成 400。
    - **不设唯一约束**：允许同一条消息改分，取**最后一次**为准（见 `routes_feedback`）。
      设唯一约束会让"改主意"变成 409，反馈入口最怕的就是点不动。
    """

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(Integer, nullable=True)
    trace_id = Column(String, nullable=False, default="")
    rating = Column(Integer, nullable=False)
    comment = Column(String, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), default=_now)

    session = relationship("ChatSession", back_populates="feedbacks")


class FigureEnrichCache(Base):
    """T-06 P1（2026-09-30）：图片内容增强的**内容哈希缓存**。

    - 键 `cache_key` = `sha256(图片字节) + vlm_model + methods + kind`（由
      `FigureEnricher` 生成）⇒ 图片或增强配置任一变化自动 miss，**无需显式失效**。
    - 值 = 增强 triple（ocr_text / table_md / vlm_caption）。doc10 类图密集文档改稿
      重建时 VLM 占 75%（742s vs 无 VLM 187s）⇒ 未变图 0 次调用是改稿提速的主要来源。
    - 只增不改：命中即读、miss 即写（INSERT OR REPLACE 覆盖同键）。
    """

    __tablename__ = "figure_enrich_cache"

    cache_key = Column(String, primary_key=True)
    ocr_text = Column(Text, nullable=False, default="")
    table_md = Column(Text, nullable=False, default="")
    vlm_caption = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), default=_now)


class VectorMap(Base):
    __tablename__ = "vector_map"

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_type = Column(String, nullable=False)
    owner_id = Column(Integer, nullable=False)
    vector_id = Column(Integer, nullable=False)
    index_name = Column(String, nullable=False)


Index("idx_sessions_updated", ChatSession.updated_at.desc())
# ★ 置顶排序键（2026-10-02）：`list` 的 `order_by(pinned DESC, updated_at DESC)`
#   靠这一条索引。⚠️ **必须在这里声明**（而不只在迁移 5 里建）——
#   **新建库走 `create_all`，不会跑迁移** ⇒ 只写迁移会让新库**永远缺索引**
#   （实测：create_all 后只有 idx_sessions_updated；迁移在老库上才生效）。
#   两处都建是幂等的（`CREATE INDEX IF NOT EXISTS` / create_all 同名跳过）。
Index("idx_sessions_pinned", ChatSession.pinned)
Index("idx_messages_session", Message.session_id, Message.id)
Index("idx_messages_created", Message.created_at)
Index("idx_chunks_doc", Chunk.document_id, Chunk.seq)
Index("idx_memories_session", Memory.session_id)
Index("idx_tool_runs_session", ToolRun.session_id)
Index("idx_feedback_session", Feedback.session_id)
Index("idx_feedback_created", Feedback.created_at)
Index("idx_vector_map_owner", VectorMap.owner_type, VectorMap.owner_id)


def make_engine(db_url: str, echo: bool = False):
    engine = create_engine(db_url, echo=echo, future=True)

    # Windows/SQLite 健壮性（2026-08-19 事故复盘）：
    # 本机存在环境级文件监视进程，会不定期用「拒绝写共享」的句柄短暂锁定
    # 数据库文件（SQLite 报 "attempt to write a readonly database"）。
    # 对策：busy_timeout 让拿锁最长等待 5s，避开瞬态锁窗口；PRAGMA 失败不阻断启动。
    # 另：默认库已改为 papermate.db（config.py），实测该文件名不被监视进程锁定。
    # 注意：不要启用 WAL——WAL 会新增 -wal/-shm 两个副文件，反而扩大了监视进程
    # 可锁定的文件面，实测在 delete 日志模式下更稳（2026-08-19 验证）。
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA busy_timeout=5000")
        except Exception:  # noqa: BLE001  busy_timeout 失败不影响使用
            pass
        try:
            cur.execute("PRAGMA foreign_keys=ON")
        except Exception:  # noqa: BLE001
            pass
        finally:
            cur.close()

    return engine


def init_db(engine) -> None:
    # 迁移前快照（L0-1，2026-09-16）：手写 _migrate 没有版本号也无法回滚，
    # 一旦 ALTER/重建索引出错，唯一的退路就是"迁移前那一份数据"。
    # 判据：**调用 create_all 之前**文件已存在且有内容 = 老库（新库无需备份，避免噪音快照）。
    _preexisting = _db_file_has_data(engine)
    Base.metadata.create_all(engine)
    if _preexisting:
        from ..services.db_backup import snapshot_if_exists

        snapshot_if_exists(tag="pre_migration")
    _migrate(engine)


def _db_file_has_data(engine) -> bool:
    """判断数据库文件在本次启动前是否已有数据（只看 sessions/documents 是否非空）。"""
    try:
        with engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM sessions")).scalar()
            m = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        return bool(n or m)
    except Exception:  # noqa: BLE001  新库/表不存在 → 视为无需备份
        return False


def _migrate(engine) -> None:
    """执行数据库迁移（C3，2026-09-21 起委托给 `models/migrations.py`）。

    历史：本函数原为几段手写 DDL（无版本号、无记录、失败 `except: pass` 静默）。
    现由迁移框架接管 —— **版本可查**（`schema_migrations`）、**失败可见**（日志 + 记录 + 返回值）、
    **成功可跳过**（已应用的版本不再重放）。

    ⚠️ 这里**不再吞异常**：框架内部已保证"单个迁移失败不阻断启动"，
    但仍会记 ERROR 并把失败写进 registry —— 静默是此前最主要的问题。
    """
    report = run_migrations(engine)
    if report.applied or report.failed:
        logger.info(
            "%s（当前版本 %d）", report.summary(), current_version(engine)
        )




