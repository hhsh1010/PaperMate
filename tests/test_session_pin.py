"""侧栏**置顶**（2026-10-02）：`sessions.pinned` 列 + 独立端点 + 前端重排。

## 为什么置顶必须是**列**而不是 `config_json` 里的一项

`session_service.list` 的排序是 **SQL 层**的
`ORDER BY pinned DESC, updated_at DESC`。若置顶只存在 config 里：
- 排序只能在 Python 内存里做⇒ ① `limit/offset` **分页语义会错**
  （SQL 已分页，再在内存里排 ⇒ 跨页顺序不一致）；
- 每次列表都要把全部行读进内存。
⇒ **排序键必须在列上**（迁移 5，`ALTER TABLE sessions ADD COLUMN pinned`）。

## ★ 踩过的坑：`onupdate=_now` 会覆盖"不动 updated_at"的意图

`ChatSession.updated_at` 带 `onupdate=_now` ⇒ 任何 UPDATE 都会刷新它。
初版我以为"只改 pinned 就不会动 updated_at"，并写进注释——**实测是错的**
（改 pinned 也算 UPDATE）。改成"先取 keep 再 ORM 回写"**仍无效**（`onupdate`
在flush 时**再次**赋值）。**唯一可靠做法：原生 SQL 显式回写** `updated_at`。
⇒ 这条与"限长/减轮＝否决"同源：**注释里的断言必须实测**，否则它会骗你。
"""
from __future__ import annotations

import os
import tempfile
import time

import pytest
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def svc():
    """真实 SQLite（迁移是 DDL，用内存替身测不到）。

    ★ 返回一个**具名小对象**而不是裸`SessionService`：
    测试还需要 `engine` 去验索引/重跑迁移，而 `SessionService` **并不持有**
    `_engine` 属性（我曾臆想并写进测试 ⇒ AttributeError）。
    """
    os.environ["PAPERMATE_DATA_DIR"] = tempfile.mkdtemp(prefix="papermate_pin_")
    import papermate.config as cfg_mod

    cfg_mod._settings = None
    from papermate.models.db import init_db, make_engine
    from papermate.services.session_service import SessionService

    engine = make_engine(f"sqlite:///{tempfile.mkdtemp()}/pin.db")
    init_db(engine)
    service = SessionService(sessionmaker(bind=engine, expire_on_commit=False))

    class _Env:
        pass

    env = _Env()
    env.service = service
    env.engine = engine
    return env


class TestPinnedColumn:
    def test_migration_adds_column_and_index(self, svc):
        """★ 迁移 5 必须真的加列+索引（老库升级路径）。"""
        from sqlalchemy import text

        with svc.service._Session() as db:
            cols = {r[1] for r in db.execute(text("PRAGMA table_info(sessions)"))}
            assert "pinned" in cols, "迁移未加 pinned 列"
            idx = {r[1] for r in db.execute(text("PRAGMA index_list(sessions)"))}
            assert "idx_sessions_pinned" in idx, "缺排序索引"

    def test_migration_is_idempotent(self, svc):
        """★ 幂等：老库重放时列已存在应**静默跳过**（不报错、不重复 ALTER）。"""
        from papermate.models.migrations import _add_sessions_pinned, run_migrations

        engine = svc.engine  # ★ 取自 fixture 显式暴露的 engine（`SessionService` 并不持有它）
        with engine.connect() as conn:
            # 第二次调用：列已存在⇒ 返回 None（目标已达成）
            assert _add_sessions_pinned(conn) is None
        # 整框架重跑不应报错
        report = run_migrations(engine)
        assert not report.failed, f"迁移重放失败：{report.summary()}"

    def test_default_is_unpinned(self, svc):
        """老会话（DEFAULT 0）⇒ 列表里 `pinned` 为假，不是 None。"""
        s = svc.service.create(title="A")
        assert svc.service.get(s.id).pinned == 0


class TestPinnedOrdering:
    def test_pinned_first_then_recent(self, svc):
        """★ 置顶优先；组内按活动时间倒序（置顶项之间也按活动）。"""
        a = svc.service.create(title="A")
        time.sleep(0.02)
        svc.service.create(title="B")
        time.sleep(0.02)
        svc.service.create(title="C")

        svc.service.set_pinned(a.id, True)
        titles = [s.title for s in svc.service.list()]
        assert titles == ["A", "C", "B"], f"置顶项应在最前且组内按活动：{titles}"

    def test_pinned_sorts_before_updated_at(self, svc):
        """★★ 排序键**优先级**：`pinned` 必须压过 `updated_at`。
        置顶一个"很久没动"的旧会话，它仍要排在新会话之前。
        """
        old = svc.service.create(title="旧但已置顶")
        time.sleep(0.05)
        for _ in range(3):
            svc.service.create(title="新会话")
            time.sleep(0.02)
        svc.service.set_pinned(old.id, True)
        assert svc.service.list()[0].title == "旧但已置顶", (
            "置顶必须压过活动时间（否则用户会觉得置顶没生效）"
        )

    def test_unpin_returns_to_activity_order(self, svc):
        """取消置顶后**回到它本来的位置**（而不是跳到非置顶区顶部）。

        ★ 判据用「谁的活动时间更新」而不是写死顺序：
        置顶**不刷新** `updated_at`（见下一类），所以 A 取消置顶后应回到
        "活动时间较旧的那一档" —— 本例中 C 刚创建 ⇒ 顺序 `C, B, A`。
        """
        a = svc.service.create(title="A")
        time.sleep(0.05)
        svc.service.create(title="B")
        svc.service.set_pinned(a.id, True)
        assert [s.title for s in svc.service.list()][0] == "A", "置顶期间 A 应在首位"

        time.sleep(1.05)          # 让后续会话的活动时间明显更新
        svc.service.create(title="C")
        svc.service.set_pinned(a.id, False)
        titles = [s.title for s in svc.service.list()]
        assert titles == ["C", "B", "A"], (
            f"取消置顶后 A 应回到活动时间最旧的一档（末位），而不是被顶到前面：{titles}"
        )
        assert "A" not in titles[:1], f"取消置顶后 A 不应仍在首位：{titles}"
        assert titles[-1] == "A", f"A 的活动时间最旧⇒ 应回到末位：{titles}"


class TestPinnedDoesNotTouchUpdatedAt:
    def test_pin_keeps_updated_at(self, svc):
        """★★ 置顶**不得**刷新 `updated_at`。

        原因：`onupdate=_now` 会在任何 UPDATE 时触发；置顶不是"有新活动"，
        刷新它会让用户视角的"最近活跃"被**无意义的操作**污染。
        """
        s = svc.service.create(title="A")
        time.sleep(1.05)
        before = svc.service.get(s.id).updated_at
        svc.service.set_pinned(s.id, True)
        assert svc.service.get(s.id).updated_at == before, (
            "置顶刷新了 updated_at —— `onupdate=_now` 覆盖了 ORM 层的回写"
        )

    def test_unpin_also_keeps_updated_at(self, svc):
        """取消置顶同样不得刷新（否则它会跳到非置顶区顶部，像"刚被改动过"）。"""
        s = svc.service.create(title="A")
        svc.service.set_pinned(s.id, True)
        time.sleep(1.05)
        before = svc.service.get(s.id).updated_at
        svc.service.set_pinned(s.id, False)
        assert svc.service.get(s.id).updated_at == before

    def test_source_uses_raw_sql_to_preserve_updated_at(self):
        """★ 源码契约：必须用**原生 SQL** 回写 `updated_at`。

        ORM 层（`s.updated_at = keep`）**无效**（`onupdate` 在 flush 时再次赋值），
        这是实测踩过的坑⇒ 用源码断言把实现方式钉住，避免后人"简化"回ORM 写法。
        """
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / "src" / "papermate" / "services" / "session_service.py"
        ).read_text(encoding="utf-8")
        assert "UPDATE sessions SET pinned=:p, updated_at=:u" in src, (
            "置顶必须用原生 SQL 同时回写 updated_at（ORM 写法无效）"
        )

    def test_set_pinned_is_separate_from_update(self):
        """★ 置顶必须**独立于 `update`**：两条路径的 `updated_at` 语义相反
        （改标题要刷新、置顶不刷新）⇒ 无法共用一个合并更新。"""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / "src" / "papermate" / "services" / "session_service.py"
        ).read_text(encoding="utf-8")
        assert "def set_pinned(" in src
        # update 的签名里**不该**出现 pinned
        upd = src.split("def update(")[1].split("def set_pinned(")[0]
        assert "pinned" not in upd, "update 不该管置顶（语义相反，会刷新 updated_at）"


class TestPinApi:
    def test_pin_endpoint_roundtrip(self):
        """★ 端到端：置顶 → 列表顺序变化 → 取消 → 复原。"""
        import papermate.config as cfg_mod

        os.environ["PAPERMATE_DATA_DIR"] = tempfile.mkdtemp(prefix="papermate_pinapi_")
        cfg_mod._settings = None
        from fastapi.testclient import TestClient

        from papermate.api.app import create_app

        with TestClient(create_app()) as c:
            c.post("/api/v1/sessions", json={"title": "甲"}).json()
            time.sleep(0.02)
            b = c.post("/api/v1/sessions", json={"title": "乙"}).json()

            r = c.patch(f"/api/v1/sessions/{b['id']}/pin", json={"pinned": True})
            assert r.status_code == 200 and r.json()["pinned"] is True
            assert c.get("/api/v1/sessions").json()[0]["title"] == "乙"

            c.patch(f"/api/v1/sessions/{b['id']}/pin", json={"pinned": False})
            after = c.get("/api/v1/sessions").json()
            # ★ 取消后回到**活动顺序**：乙比甲晚创建 ⇒ 乙仍在前（这才是正确的期望；
            #   置顶/取消置顶都**不**刷新 updated_at，所以活动顺序没被改变过）。
            assert [x["title"] for x in after] == ["乙", "甲"], (
                f"取消置顶后应回到活动顺序（乙较新）：{[x['title'] for x in after]}"
            )
            assert all(x["pinned"] is False for x in after), "取消后不应残留 pinned=True"

    def test_pin_unknown_session_404(self):
        import papermate.config as cfg_mod

        os.environ["PAPERMATE_DATA_DIR"] = tempfile.mkdtemp(prefix="papermate_pin404_")
        cfg_mod._settings = None
        from fastapi.testclient import TestClient

        from papermate.api.app import create_app

        with TestClient(create_app()) as c:
            assert c.patch("/api/v1/sessions/999999/pin", json={"pinned": True}).status_code == 404

    def test_list_includes_pinned_field(self):
        """★ `SessionOut.pinned` 必须真的出现在响应里
        （漏填 `_to_out` ⇒ 前端永远读到 False ⇒ "点了没反应"）。"""
        import papermate.config as cfg_mod

        os.environ["PAPERMATE_DATA_DIR"] = tempfile.mkdtemp(prefix="papermate_pinfield_")
        cfg_mod._settings = None
        from fastapi.testclient import TestClient

        from papermate.api.app import create_app

        with TestClient(create_app()) as c:
            s = c.post("/api/v1/sessions", json={"title": "甲"}).json()
            assert "pinned" in s and s["pinned"] is False
            lst = c.get("/api/v1/sessions").json()
            assert lst[0]["pinned"] is False
