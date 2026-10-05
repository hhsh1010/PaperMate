"""L0-1 数据库备份与恢复 · 回归测试（2026-09-16）

覆盖的真实风险（不是"函数被调用了"，而是"数据真的回来了"）：
  1. 快照内容与源库一致（行数 + integrity_check + 元数据）；
  2. **恢复后数据回到快照时刻**（写两份新数据 → 恢复 → 新数据消失、旧数据还在）；
  3. 恢复前自动留 `pre_restore` 快照（误恢复也能再退回来）；
  4. **恢复会清理 `-wal` / `-shm`**（残留旧 WAL 会在下次打开时回放到新库上 → 损坏）；
  5. 坏快照被拒且**不破坏现有库**（失败路径不碰现有数据）；
  6. prune 只删旧的、保留最新的；
  7. 启动前兜底快照失败不抛异常（不阻断服务）。
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from papermate.services import db_backup as bk


def _make_db(path: Path, sessions: int = 1, documents: int = 0) -> None:
    """造一个"最小可用"的库：只建 sessions / documents 两张表并按需填行。"""
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, title TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, name TEXT)")
        conn.executemany("INSERT INTO sessions (title) VALUES (?)", [(f"s{i}",) for i in range(sessions)])
        conn.executemany("INSERT INTO documents (name) VALUES (?)", [(f"d{i}",) for i in range(documents)])
        conn.commit()  # closing() 不提交事务，必须显式 commit


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "papermate.db"
    _make_db(p, sessions=1, documents=1)
    return p


def _counts(path: Path) -> dict[str, int]:
    return bk.table_row_counts(path)


def test_create_snapshot_writes_db_and_meta(db: Path) -> None:
    snap = bk.create_snapshot(db_path=db, keep=0)

    assert snap.exists() and snap.stat().st_size > 0
    meta_file = snap.with_suffix(".meta.json")
    assert meta_file.exists()

    import json

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["row_counts"] == {"documents": 1, "sessions": 1}
    assert meta["sha256"] and len(meta["sha256"]) == 64
    assert meta["source"] == str(db)
    assert bk.verify_snapshot(snap)["ok"] is True


def test_restore_brings_data_back(db: Path) -> None:
    snap = bk.create_snapshot(db_path=db, keep=0)

    with closing(sqlite3.connect(db)) as conn:
        conn.execute("INSERT INTO sessions (title) VALUES ('after-snapshot-1')")
        conn.execute("INSERT INTO sessions (title) VALUES ('after-snapshot-2')")
        conn.execute("DELETE FROM documents")
        conn.commit()
    assert _counts(db) == {"documents": 0, "sessions": 3}

    bk.restore_snapshot(snap.name, db_path=db, keep=0)

    assert _counts(db) == {"documents": 1, "sessions": 1}, "恢复后必须回到快照时刻的行数"
    with closing(sqlite3.connect(db)) as conn:
        titles = [r[0] for r in conn.execute("SELECT title FROM sessions").fetchall()]
    assert titles == ["s0"], "快照之后写入的数据必须消失"


def test_restore_leaves_pre_restore_safety_backup(db: Path) -> None:
    """恢复前必须先把"当前库"存一份 —— 否则误恢复到旧快照 = 又一次数据丢失。"""
    snap = bk.create_snapshot(db_path=db, keep=0)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("INSERT INTO sessions (title) VALUES ('only-in-live')")
        conn.commit()
    assert _counts(db)["sessions"] == 2

    r = bk.restore_snapshot(str(snap), db_path=db, keep=0)

    assert r["safety_backup"], "恢复前必须给当前库留一份快照"
    safety = Path(r["safety_backup"])
    assert safety.exists()
    assert safety.name.endswith("-pre_restore.db")
    assert _counts(safety)["sessions"] == 2, "安全快照里应当是'被覆盖前'的现场（含 only-in-live）"
    assert _counts(db)["sessions"] == 1, "恢复后当前库回到快照时刻"


def test_restore_removes_stale_wal_and_shm(db: Path) -> None:
    snap = bk.create_snapshot(db_path=db, keep=0)
    wal = Path(str(db) + "-wal")
    shm = Path(str(db) + "-shm")
    wal.write_bytes(b"stale-wal-content")
    shm.write_bytes(b"stale-shm")

    bk.restore_snapshot(snap.name, db_path=db, keep=0)

    assert not wal.exists(), "残留旧 WAL 会在下次打开时被回放到新库上 → 必须清理"
    assert not shm.exists()


def test_restore_rejects_corrupt_snapshot_without_touching_db(tmp_path: Path, db: Path) -> None:
    bad = bk.backup_dir(db) / "papermate-19700101-000000.db"
    bad.write_bytes(b"this is not a sqlite file")
    before = _counts(db)

    with pytest.raises(RuntimeError, match="完整性校验失败"):
        bk.restore_snapshot(bad.name, db_path=db, keep=0)

    assert _counts(db) == before, "失败路径绝不能改动现有库"


def test_prune_keeps_newest(tmp_path: Path, db: Path) -> None:
    names = [bk.create_snapshot(db_path=db, keep=0).name for _ in range(4)]
    assert len(bk.list_snapshots(db_path=db)) == 4

    removed = bk.prune(2, db_path=db)
    left = [s["name"] for s in bk.list_snapshots(db_path=db)]

    assert len(removed) == 2 and len(left) == 2
    assert names[-1] in left, "最新的必须保留"
    for name in left:
        meta = Path(bk.backup_dir(db)) / name
        assert meta.with_suffix(".meta.json").exists(), "元数据要跟着快照一起留存"


def test_list_snapshots_sorted_newest_first(db: Path) -> None:
    for _ in range(3):
        bk.create_snapshot(db_path=db, keep=0)
    names = [s["name"] for s in bk.list_snapshots(db_path=db)]
    assert names == sorted(names, reverse=True)


def test_snapshot_if_exists_never_raises(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(bk, "sqlite_path", lambda: db)
    monkeypatch.setattr(bk, "create_snapshot", _boom)

    assert bk.snapshot_if_exists() is None, "启动路径不允许因备份失败而抛异常"


def test_backup_includes_uncheckpointed_wal_data(tmp_path: Path, db: Path) -> None:
    """WAL 模式下、写连接**仍然打开**（数据尚未 checkpoint）时，快照必须仍拿到全部已提交行。

    这是"用 SQLite 在线备份 API、而不是直接 copy 数据库文件"的核心理由 ——
    直接 copy 主库文件会漏掉 `-wal` 里尚未 checkpoint 的数据，
    得到一个"看起来正常、实际缺数据"的快照（比没有备份更危险）。
    """
    writer = sqlite3.connect(db)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO sessions (title) VALUES ('wal-pending')")
        writer.commit()

        # 对照组：直接拷主库文件 → 会漏掉 WAL 里那行（证明"不留 WAL"的选择不是多余的）
        naive = tmp_path / "naive-copy.db"
        shutil.copyfile(db, naive)
        assert _counts(naive).get("sessions", 0) < 2, (
            "对照组失效：若直接拷贝也拿到了数据，本用例已无法证明在线备份的必要性"
        )

        snap = bk.create_snapshot(db_path=db, keep=0)
        assert _counts(snap)["sessions"] == 2, "WAL 中未 checkpoint 的行也必须进快照"
    finally:
        writer.close()


def test_sqlite_path_reads_db_url_not_empty_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """回归：`sqlite_path()` 必须读**派生**的 `db_url`，而不是可选覆盖字段 `database_url`。

    2026-09-16 真实库演习踩到：读错字段时 `db_url` 默认空串 → 路径解析成 `.` →
    `sqlite3.OperationalError: unable to open database file`；
    而当时的单测因为**都显式传了 `db_path`**，全部通过、完全没暴露这个问题。
    """
    from papermate.config import get_settings

    s = get_settings()
    p = bk.sqlite_path()

    assert p.name == "papermate.db", f"解析出的不是数据库文件而是 {p!r}"
    assert p.is_absolute(), "必须是绝对路径（相对路径会随 CWD 漂移）"
    assert p == Path(s.db_url.split("///", 1)[-1])
    assert p.parent == s.resolved_data_dir


# ---------------------------------------------- D4 留存（2026-09-21）


def test_default_keep_is_seven() -> None:
    """D4：保留份数由 10 调为 7（上限约 84MB）—— 用测试锁住这个决策。"""
    from papermate.services import db_backup

    assert db_backup.DEFAULT_KEEP == 7


# 注（2026-09-21）：原有一条 `_cleanup_stale_tmp` 的测试，随该函数一起移除了。
# 原因：把「删除」内联进快照关键路径是**危险设计** —— 本环境的 safe-delete 垫片
# 拦截批量删除时会**直接终止进程**（Python 捕获不到），导致**快照失败**。
# 结论：清理只能由 `scripts/retention.py` 这类**用户显式触发的工具**做，不得内联进业务路径。
