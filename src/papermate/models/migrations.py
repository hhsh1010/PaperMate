"""轻量迁移框架（C3，2026-09-21）。

**要解决的问题**（复检 §3 C3）：此前的迁移是 `db._migrate()` 里几段手写 DDL ——
没有版本号、没有记录、失败 `except: pass` 静默吞掉，且**每段自己决定失败语义**
（有的 warning、有的静默跳过）。结果是：
- 无法回答"这个库现在处于哪一版"；
- 迁移失败**只有日志**，用户与后续代码都不知道；
- 新增迁移只能继续往那个函数里塞，越塞越难读。

## 为什么不引入 Alembic

本项目是**单用户本地应用**，零重依赖是明确风格（后端仅 FastAPI/SQLAlchemy/uvicorn）。
Alembic 的价值在于 autogenerate 与多人协作的版本树；这里是**单人、单库、表结构演进缓慢**，
手写规范化（版本号 + 记录 + 顺序执行）就够，且**没有额外依赖**。

## ⚠️ 引入框架时最容易踩的坑：迁移函数必须**幂等**

老库**没有** `schema_migrations` 记录 —— 框架首次运行时会认为"一个都没应用过"，
于是**重放全部迁移**。若某个迁移不是幂等的（比如裸 `ALTER TABLE ADD COLUMN`），
它会在老库上直接报错，而新库却一切正常（新库的列已由 `create_all` 建好，重放时才发现列已存在）。

→ 所以：**每个迁移的 `apply` 都必须能安全重复执行**（先查 `PRAGMA table_info` / 用
`IF NOT EXISTS`），与"是否已记录版本"无关。这是本模块的硬约束，由测试守护。

## 失败语义

- **失败不阻断启动**（启动路径上有 L0-1 迁移前快照兜底），但**绝不静默**：
  记 ERROR 日志 + 写入 `schema_migrations.detail` + 在返回值里体现。
- **`skipped` ≠ `failed`**：有些迁移**主动跳过**是正确的业务决策
  （如 `collection_documents` 检测到重复数据时的告警式跳过）——
  若把它记成 failed，就会每次启动都"报故障"，把真正的失败淹掉。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    """一条迁移。

    `apply(conn)` 返回 `None` 表示已应用；返回字符串表示**主动跳过**（字符串是跳过原因）。
    """

    version: int
    name: str
    apply: Callable[[Connection], str | None]


@dataclass
class MigrationResult:
    version: int
    name: str
    status: str  # applied / skipped / failed / already
    detail: str = ""


@dataclass
class MigrationReport:
    """一次 `run_migrations` 的结果汇总。"""

    results: list[MigrationResult] = field(default_factory=list)

    @property
    def applied(self) -> list[MigrationResult]:
        return [r for r in self.results if r.status == "applied"]

    @property
    def failed(self) -> list[MigrationResult]:
        return [r for r in self.results if r.status == "failed"]

    def summary(self) -> str:
        n_applied = len(self.applied)
        n_skipped = sum(1 for r in self.results if r.status == "skipped")
        n_failed = len(self.failed)
        return f"迁移：应用 {n_applied} · 跳过 {n_skipped} · 失败 {n_failed}"


# --------------------------------------------------------------------------- 各迁移实现
# 硬约束：每个 apply 必须幂等（见模块 docstring）。


def _add_documents_content_hash(conn: Connection) -> str | None:
    """documents.content_hash：上传去重依据。"""
    cols = {row[1] for row in conn.execute(text("PRAGMA table_info(documents)"))}
    if "content_hash" in cols:
        # ⚠️ 列已存在 → 返回 None（= applied），**不是 skipped**。
        # 迁移的目的是"达到目标结构"，列已在就是**目标已达成**；
        # 老库重放时若把这种正常情况记成 skipped，会让"跳过"这个信号失去意义
        # （每次启动一堆 skipped，真正的业务跳过反而被淹掉）—— 这是测试抓出来的。
        return None
    conn.execute(text("ALTER TABLE documents ADD COLUMN content_hash VARCHAR DEFAULT ''"))
    return None


def _add_messages_meta_json(conn: Connection) -> str | None:
    """messages.meta_json：消息级结构化元数据（提醒/无效标记），供刷新后回填。"""
    cols = {row[1] for row in conn.execute(text("PRAGMA table_info(messages)"))}
    if "meta_json" in cols:
        return None  # 同 `_add_documents_content_hash`：列已存在即目标达成
    conn.execute(text("ALTER TABLE messages ADD COLUMN meta_json VARCHAR DEFAULT '{}'"))
    return None


def _add_messages_created_index(conn: Connection) -> str | None:
    """messages.created_at 索引：避免趋势/用量查询全表扫描（审查 P2-4）。"""
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_created ON messages (created_at)"))
    return None


def _unique_collection_documents(conn: Connection) -> str | None:
    """collection_documents(collection_id, document_id) 唯一索引。

    **告警式**（2026-09-06 定）：检测到重复关联时**只跳过、不改数据**。
    旧版会静默 DELETE 重复行并按 MIN(id) 保留 —— 那是无备份地改写业务数据，
    且 `seq`（集内排序）语义可能被静默改变。去重交
    `scripts/dedup_collections.py`（带备份、按 MIN(seq) 保排序）人工确认后执行。
    """
    dup = conn.execute(
        text(
            "SELECT collection_id, document_id, COUNT(*) AS c FROM collection_documents "
            "GROUP BY collection_id, document_id HAVING c > 1"
        )
    ).fetchall()
    if dup:
        detail = f"检测到 {len(dup)} 组重复文献关联，跳过唯一索引创建（请运行 scripts/dedup_collections.py）"
        logger.warning("%s：%s", detail, dup[:5])
        return detail

    conn.execute(text("DROP INDEX IF EXISTS idx_coldoc"))
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_coldoc "
            "ON collection_documents (collection_id, document_id)"
        )
    )
    return None


#: 迁移清单 —— **只能追加，不得修改或删除已发布的条目**
#: （改已发布的迁移会让不同库处于"同版本但结构不同"的状态）。
def _add_sessions_pinned(conn: Connection) -> str | None:
    """sessions.pinned：侧栏置顶（工作间与对话共用）。

    ★ **为什么用独立列而不塞 `config_json`**：`session_service.list` 的排序是
    **SQL 层**的 `ORDER BY updated_at DESC`；若置顶只存在 config 里，就**只能在
    Python 内存里排** ⇒ ① 分页（`limit/offset`）语义会错（SQL 已分页后再排），
    ② 每次列表都要把全部行读进内存。**排序键必须在列上。**
    同理"未置顶内部按最近活动、置顶之间按最近置顶时间"需要**两个排序键**。
    """
    cols = {row[1] for row in conn.execute(text("PRAGMA table_info(sessions)"))}
    if "pinned" in cols:
        return None  # 同上：列已存在即目标达成
    conn.execute(text("ALTER TABLE sessions ADD COLUMN pinned INTEGER DEFAULT 0"))
    # ★ 同批建索引：SQLite**不支持** `ALTER TABLE ... ADD COLUMN` 后立即建索引，
    #   必须在**下一条语句**单独执行。索引名沿用 `idx_` 约定。
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_pinned ON sessions (pinned)"))
    return None


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "documents.content_hash", _add_documents_content_hash),
    Migration(2, "messages.meta_json", _add_messages_meta_json),
    Migration(3, "index:idx_messages_created", _add_messages_created_index),
    Migration(4, "unique:idx_coldoc", _unique_collection_documents),
    Migration(5, "sessions.pinned+index", _add_sessions_pinned),
)


# --------------------------------------------------------------------------- 框架


def _ensure_registry(conn: Connection) -> None:
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  version INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  applied_at TEXT NOT NULL,"
            "  status TEXT NOT NULL,"
            "  detail TEXT NOT NULL DEFAULT ''"
            ")"
        )
    )


def _applied_versions(conn: Connection) -> set[int]:
    rows = conn.execute(text("SELECT version FROM schema_migrations WHERE status = 'applied'"))
    return {int(r[0]) for r in rows}


def _record(conn: Connection, version: int, name: str, status: str, detail: str) -> None:
    conn.execute(
        text(
            "INSERT INTO schema_migrations (version, name, applied_at, status, detail) "
            "VALUES (:v, :n, :t, :s, :d) "
            "ON CONFLICT(version) DO UPDATE SET applied_at=:t, status=:s, detail=:d"
        ),
        {
            "v": version,
            "n": name,
            "t": datetime.now(UTC).isoformat(timespec="seconds"),
            "s": status,
            "d": detail,
        },
    )


def run_migrations(engine: Engine) -> MigrationReport:
    """按版本顺序执行未应用的迁移。

    **每个迁移独立事务**：一个失败不影响其它 —— 否则一个坏迁移会让后面的全部不执行，
    而它们之间通常是无关的（加列、建索引）。
    """
    report = MigrationReport()

    with engine.begin() as conn:
        _ensure_registry(conn)
        done = _applied_versions(conn)

    ordered = sorted(MIGRATIONS, key=lambda m: m.version)
    for mig in ordered:
        if mig.version in done:
            report.results.append(MigrationResult(mig.version, mig.name, "already"))
            continue
        try:
            with engine.begin() as conn:
                detail_or_none = mig.apply(conn)
                status = "skipped" if detail_or_none else "applied"
                _record(conn, mig.version, mig.name, status, detail_or_none or "")
            report.results.append(
                MigrationResult(mig.version, mig.name, status, detail_or_none or "")
            )
            if status == "applied":
                logger.info("迁移已应用 v%d %s", mig.version, mig.name)
            else:
                logger.info("迁移已跳过 v%d %s：%s", mig.version, mig.name, detail_or_none)
        except Exception as e:  # noqa: BLE001  失败不阻断启动，但必须可见
            detail = f"{type(e).__name__}: {e}"
            logger.error("迁移失败 v%d %s：%s", mig.version, mig.name, detail, exc_info=True)
            report.results.append(MigrationResult(mig.version, mig.name, "failed", detail))
            # 失败**不写** applied 记录 → 下次启动会重试（幂等所以安全）
            try:
                with engine.begin() as conn:
                    _record(conn, mig.version, mig.name, "failed", detail)
            except Exception:  # noqa: BLE001 记录失败也不能盖住原错误
                logger.debug("迁移失败状态写入 registry 失败", exc_info=True)

    if report.failed:
        logger.warning("%s（%d 项失败，服务继续启动）", report.summary(), len(report.failed))
    return report


def current_version(engine: Engine) -> int:
    """当前库的迁移版本（未记录 = 0）。用于诊断"这个库到哪一版了"。"""
    try:
        with engine.begin() as conn:
            _ensure_registry(conn)
            row = conn.execute(text("SELECT MAX(version) FROM schema_migrations WHERE status='applied'")).scalar()
        return int(row or 0)
    except Exception:  # noqa: BLE001 诊断函数不得抛错
        return 0
