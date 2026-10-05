"""C4 列表分页 + C5 中断任务恢复（2026-09-21）。

两条都属于"数据量变化后才会暴露"的问题，所以测试重点在**边界**：
- C4：默认不改变现状（返回数组 + 全量）、`X-Total-Count` 与实际总数一致、offset 真的跳过
- C5：**中间态必须被恢复**，而**终态绝不能被误改**（误改会把可用文档标成失败）
"""

# ⚠️ 本文件**刻意不做** `from __future__ import annotations`：
# 分页端点定义在 fixture **函数体内**，注解若被延迟成字符串（ForwardRef），
# pydantic 解析 `limit: LimitParam` 时会在错误的命名空间里找该名字
# → 抛 `PydanticUserError: ... is not fully defined`，或表现为请求拿不到响应头。
# 真实路由不受影响（端点与注解同在模块顶层，已由端到端验证）。
import pytest
from sqlalchemy.orm import sessionmaker

from papermate.api.pagination import DEFAULT_LIMIT, LimitParam, OffsetParam
from papermate.models.db import Document, init_db, make_engine
from papermate.services.processing_service import (
    IN_FLIGHT_STATUSES,
    TERMINAL_STATUSES,
)

# ⚠️ 注意：本文件有 `from __future__ import annotations`（注解变字符串 / ForwardRef）。
# 因此上面 `LimitParam`/`OffsetParam` **必须在模块顶层 import** ——
# 放在 fixture 函数体内的话，pydantic 解析注解时找不到该名字，会抛
# `PydanticUserError: ... is not fully defined`（实测踩过）。

# ------------------------------------------------------------------ C5


@pytest.fixture()
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path/'r.db'}")
    init_db(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


class _StubDocSvc:
    pass


def _svc(session_factory):
    from papermate.config import get_settings
    from papermate.services.processing_service import ProcessingService

    return ProcessingService(_StubDocSvc(), get_settings(), session_factory)  # type: ignore[arg-type]


def _add_doc(sf, doc_id: int, status: str) -> None:
    with sf() as s:
        s.add(Document(id=doc_id, filename=f"d{doc_id}.pdf", storage_path=f"/x/{doc_id}.pdf", status=status))
        s.commit()


def test_interrupted_docs_are_marked_failed(session_factory):
    """★ 核心：中间态文档在启动时被标记 failed（否则永久卡在"处理中"）。"""
    for i, st in enumerate(IN_FLIGHT_STATUSES, start=1):
        _add_doc(session_factory, i, st)

    res = _svc(session_factory).recover_interrupted()

    assert res["recovered"] == len(IN_FLIGHT_STATUSES)
    with session_factory() as s:
        statuses = {d.id: d.status for d in s.query(Document).all()}
    assert set(statuses.values()) == {"failed"}


def test_terminal_docs_are_never_touched(session_factory):
    """★ 终态绝不能被误改 —— 误改会把 ready 的文档标成失败（用户得重新上传）。"""
    _add_doc(session_factory, 1, "ready")
    _add_doc(session_factory, 2, "failed")
    _add_doc(session_factory, 3, "cancelled")
    _add_doc(session_factory, 4, "parsing")

    res = _svc(session_factory).recover_interrupted()

    assert res["recovered"] == 1, "只有 parsing 那一条该被恢复"
    with session_factory() as s:
        assert s.get(Document, 1).status == "ready"
        assert s.get(Document, 3).status == "cancelled"
        assert s.get(Document, 4).status == "failed"


def test_recovery_is_idempotent(session_factory):
    """重复调用不得报错，也不得再"恢复"一次（第二次应为 0）。"""
    _add_doc(session_factory, 1, "embedding")
    svc = _svc(session_factory)
    assert svc.recover_interrupted()["recovered"] == 1
    assert svc.recover_interrupted()["recovered"] == 0


def test_recovery_on_empty_db_is_noop(session_factory):
    res = _svc(session_factory).recover_interrupted()
    assert res["recovered"] == 0 and res["documents"] == []


def test_status_sets_are_disjoint():
    """中间态与终态不得重叠 —— 重叠会让"哪些该恢复"变成矛盾定义。"""
    assert not (set(IN_FLIGHT_STATUSES) & set(TERMINAL_STATUSES))


# ------------------------------------------------------------------ C4


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    """最小 app：只挂 /items 列表端点，用于验证分页语义。

    ⚠️ **scope="module"**（而不是默认的函数级）：本文件多个测试都用到分页端点，
    若每个测试各建一个 app，FastAPI 会把**同一个模块级 `Annotated` 注解**重复注册多次 ——
    实测表现为"前几个测试通过、后面的请求拿不到 `X-Total-Count` 甚至返回错误体"。
    数据是只读的，整模块共用一个 app 即可，也更贴近"一个进程一个 app"的真实形态。
    """
    from fastapi import FastAPI, Response
    from fastapi.testclient import TestClient

    tmp_path = tmp_path_factory.mktemp("pag")
    engine = make_engine(f"sqlite:///{tmp_path/'p.db'}")
    init_db(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        for i in range(1, 8):  # 7 条记录
            s.add(Document(id=i, filename=f"d{i}.pdf", storage_path=f"/x/{i}", status="ready"))
        s.commit()

    app = FastAPI()

    @app.get("/items")
    def items(response: Response, limit: LimitParam = DEFAULT_LIMIT, offset: OffsetParam = 0) -> list[dict]:
        from papermate.api.pagination import paginate

        with Session() as s:
            q = s.query(Document).order_by(Document.id.asc())
            rows = paginate(q, response, limit=limit, offset=offset)
            return [{"id": d.id} for d in rows]

    return TestClient(app)


def test_default_returns_all_and_reports_total(api):
    """默认（不传参）行为与从前一致：返回全部。总数通过响应头给出。"""
    r = api.get("/items")
    assert r.status_code == 200
    assert isinstance(r.json(), list), "★ 返回体必须仍是数组（不破坏前端契约）"
    assert len(r.json()) == 7
    assert r.headers["X-Total-Count"] == "7"


def test_limit_and_offset_paginate(api):
    r = api.get("/items", params={"limit": 3, "offset": 0})
    assert [x["id"] for x in r.json()] == [1, 2, 3]
    assert r.headers["X-Total-Count"] == "7", "总数不随分页变化"

    r2 = api.get("/items", params={"limit": 3, "offset": 3})
    assert [x["id"] for x in r2.json()] == [4, 5, 6]

    r3 = api.get("/items", params={"limit": 3, "offset": 6})
    assert [x["id"] for x in r3.json()] == [7]


def test_offset_beyond_total_returns_empty_not_error(api):
    """越界 offset 应返回空列表（不是 404/500）—— 分页循环依赖这个语义。"""
    r = api.get("/items", params={"limit": 10, "offset": 100})
    assert r.status_code == 200 and r.json() == []


def test_limit_bounds_are_enforced(api):
    """上限必须拦住 `?limit=999999`（否则分页可被用来变相全量拉取）。"""
    assert api.get("/items", params={"limit": 0}).status_code == 422
    assert api.get("/items", params={"limit": 999999}).status_code == 422
    assert api.get("/items", params={"offset": -1}).status_code == 422


def test_total_count_matches_actual_rows(api):
    """★ 头里的总数必须等于真实行数 —— 不一致的话前端分页会多/少一页。"""
    total = int(api.get("/items").headers["X-Total-Count"])
    collected = []
    for off in range(0, total, 2):
        collected += [x["id"] for x in api.get("/items", params={"limit": 2, "offset": off}).json()]
    assert collected == list(range(1, 8))
    assert len(collected) == total


def test_expose_header_present(api):
    """跨域下要能读到 X-Total-Count（同源无所谓，但显式暴露不留坑）。"""
    assert "X-Total-Count" in api.get("/items").headers.get("access-control-expose-headers", "")
