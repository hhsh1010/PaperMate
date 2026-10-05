"""C3（2026-09-21）：迁移框架 —— 版本记录 / 幂等 / 失败可见 / 跳过语义。

守护的是**框架不变量**，不是"某个迁移写对了"：
- ★ **幂等**：老库没有版本记录 → 首次会重放全部迁移，非幂等的迁移会当场炸
  （这是引入迁移框架时最容易踩的坑，新库反而发现不了）
- **失败可见**：此前 `except: pass` 静默吞掉，现在必须进 report + registry
- **跳过 ≠ 失败**：主动跳过（如检测到重复数据）不能记成故障，否则真故障会被淹掉
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy import create_engine, text

from papermate.models import migrations as mg
from papermate.models.db import Base, init_db


@pytest.fixture()
def engine(tmp_path):
    e = create_engine(f"sqlite:///{tmp_path/'m.db'}", future=True)
    Base.metadata.create_all(e)
    return e


def test_fresh_db_applies_all(engine):
    report = mg.run_migrations(engine)
    assert not report.failed, report.summary()
    assert len(report.applied) == len(mg.MIGRATIONS)
    assert mg.current_version(engine) == max(m.version for m in mg.MIGRATIONS)


def test_second_run_is_all_already(engine):
    """已应用的版本不得重放。"""
    mg.run_migrations(engine)
    report = mg.run_migrations(engine)
    assert not report.applied
    assert all(r.status == "already" for r in report.results)


def test_migrations_are_idempotent_on_legacy_db(tmp_path):
    """★ 关键：模拟"老库"——表结构已由 create_all 建好，但**没有版本记录**。

    框架首次运行会重放全部迁移。若非幂等（裸 ALTER ADD COLUMN），这里就会报
    "duplicate column name" —— 新库测不出来，只有这个场景能暴露。
    """
    e = create_engine(f"sqlite:///{tmp_path/'legacy.db'}", future=True)
    Base.metadata.create_all(e)  # 列已存在，且无 schema_migrations

    report = mg.run_migrations(e)
    assert not report.failed, f"老库重放迁移不应失败：{[r.detail for r in report.failed]}"

    # 再跑一次（此时已有记录）—— 仍不得失败
    again = mg.run_migrations(e)
    assert not again.failed


def test_init_db_runs_migrations(engine):
    """init_db 必须接上框架（否则迁移只在新代码路径生效）。"""
    init_db(engine)
    assert mg.current_version(engine) >= 1


def test_failure_is_visible_not_silent(engine, monkeypatch):
    """★ 失败必须进 report（可见），且**不抛出**（启动路径不能被迁移阻断）。"""

    def boom(conn):  # noqa: ANN001
        raise RuntimeError("模拟迁移失败")

    bad = mg.Migration(999, "boom", boom)
    monkeypatch.setattr(mg, "MIGRATIONS", mg.MIGRATIONS + (bad,))

    report = mg.run_migrations(engine)  # 不得抛异常
    assert len(report.failed) == 1
    assert report.failed[0].version == 999
    assert "模拟迁移失败" in report.failed[0].detail
    assert "失败 1" in report.summary()

    # 失败状态落 registry，便于事后查
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT status, detail FROM schema_migrations WHERE version = 999")
        ).fetchone()
    assert row[0] == "failed" and "模拟迁移失败" in row[1]

    # 失败**不写 applied** → 下次启动会重试
    report2 = mg.run_migrations(engine)
    assert any(r.version == 999 for r in report2.results)


def test_one_failure_does_not_block_others(engine, monkeypatch):
    """★ 每个迁移独立事务：一个坏迁移不能拦住后面的（它们通常互不相关）。"""

    def boom(conn):  # noqa: ANN001
        raise RuntimeError("坏迁移")

    bad = mg.Migration(0, "boom-first", boom)  # 版本最小 → 最先执行
    monkeypatch.setattr(mg, "MIGRATIONS", (bad,) + mg.MIGRATIONS)

    report = mg.run_migrations(engine)
    assert len(report.failed) == 1
    # 后面的好迁移仍然全部应用（MIGRATIONS 现在含那条坏迁移，故减 1）
    assert len(report.applied) == len(mg.MIGRATIONS) - 1, report.summary()


def test_skipped_is_not_failed(tmp_path):
    """★ 主动跳过（有重复关联）必须记为 skipped —— 记成 failed 会每次启动都"报故障"。"""
    e = create_engine(f"sqlite:///{tmp_path/'dup.db'}", future=True)
    with e.begin() as conn:
        conn.execute(text(
            "CREATE TABLE collection_documents (id INTEGER PRIMARY KEY, "
            "collection_id INTEGER, document_id INTEGER, seq INTEGER)"
        ))
        conn.execute(text(
            "INSERT INTO collection_documents (collection_id, document_id, seq) "
            "VALUES (1, 2, 5), (1, 2, 3)"
        ))
    # 其余表也要存在，否则其它迁移会失败（这里只关心被跳过的那条）
    Base.metadata.create_all(e)

    report = mg.run_migrations(e)
    by_version = {r.version: r for r in report.results}
    skipped = [r for r in report.results if r.status == "skipped"]
    assert skipped, "检测到重复时该迁移应记为 skipped"
    assert all("重复" in r.detail for r in skipped)
    assert by_version[4].status == "skipped"
    assert not report.failed


def test_current_version_on_empty_registry(engine):
    """诊断函数不得抛错：刚建库、还没跑迁移时应返回 0。"""
    assert mg.current_version(engine) == 0


def test_migration_versions_are_unique_and_ordered():
    """版本号唯一且严格递增 —— 重复版本会让 registry 主键冲突、顺序不确定。"""
    versions = [m.version for m in mg.MIGRATIONS]
    assert versions == sorted(versions)
    assert len(versions) == len(set(versions))


def test_every_migration_has_name_and_callable():
    for m in mg.MIGRATIONS:
        assert m.name and callable(m.apply)
        m2 = replace(m)
        assert m2.version == m.version
