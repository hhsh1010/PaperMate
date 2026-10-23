"""服务类调用计费（`service_usage` 表 + `/usage` 汇总）。

★ 背景（2026-10-23）：翻译/总结/提示词增强此前**完全不记账** ✗
⇒ `/usage` 与上下文面板的 ¥ 只含对话+技能，服务调用是黑账。
本文件钉住两条：**写得进**、**汇总得对**。
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from papermate.models.db import Base, ServiceUsage
from papermate.services.usage_service import record_service_usage


@pytest.fixture()
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'u.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


class _Resp:
    """模拟 LLMResponse 的用量字段（只测记账，不需要真实 LLM）。"""

    def __init__(self, p: int, c: int, cost: float) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c
        self.cost = cost


def test_record_writes_row(session_factory) -> None:
    record_service_usage(session_factory, source="translate_titles", model="deepseek-x",
                         response=_Resp(1200, 80, 0.0031))
    with session_factory() as db:
        rows = db.query(ServiceUsage).all()
    assert len(rows) == 1
    r = rows[0]
    assert (r.source, r.model, r.prompt_tokens, r.completion_tokens) == (
        "translate_titles", "deepseek-x", 1200, 80)
    assert abs(r.cost - 0.0031) < 1e-9


def test_multiple_sources_accumulate(session_factory) -> None:
    for src, cost in (("summary", 0.05), ("related_work", 0.02), ("prompt_enhance", 0.001)):
        record_service_usage(session_factory, source=src, model="m", response=_Resp(100, 10, cost))
    with session_factory() as db:
        assert db.query(ServiceUsage).count() == 3
        total = sum(r.cost for r in db.query(ServiceUsage).all())
    assert abs(total - 0.071) < 1e-9


def test_explicit_values_without_response(session_factory) -> None:
    """允许不传 response、直接给数字（自算成本场合）。"""
    record_service_usage(session_factory, source="summary", model="m",
                         prompt_tokens=500, completion_tokens=50, cost=0.001)
    with session_factory() as db:
        r = db.query(ServiceUsage).one()
    assert (r.prompt_tokens, r.completion_tokens) == (500, 50)


def test_broken_factory_never_raises() -> None:
    """★ 记账是旁路：工厂坏掉也不能把业务打断。"""

    class Boom:
        def __call__(self):
            raise RuntimeError("db down")

    record_service_usage(Boom(), source="summary", model="m", response=_Resp(1, 1, 0.0))
