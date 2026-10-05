"""样本时效护栏 · 单元测试（2026-10-05）

背景：看板取「最近 N 条非空 Trace」，但这份样本可能**又旧又少**，而它会被当成"现状"读。
实测踩到：窗口 = `10-03 14:28 ~ 10-04 13:13`（最新一条已 ≈19.5 小时前），
且 10 条里 8 条早于 10-03 的一次修复 ⇒ 看板在展示"修复前的世界"。
⇒ 后端算出 `stale` / `insufficient` / `age_hours`，前端据此出提示。

**本测试只测纯函数**（`now` 可注入，不依赖真实时钟），因为：
  - 判据必须与"当前时间"无关地可复现；
  - ⚠️ 落库时间是 **naive UTC**（`models/db.py:_now()` = `datetime.now(UTC)`），
    若按本地时区解释，GMT+8 会整体偏差 **8 小时** —— 这正是本次要钉死的点。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from papermate.skills.dev_eval import (
    _SAMPLE_MAX_AGE_HOURS,
    _sample_quality,
    _window_age_hours,
)

NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)


def _ago(hours: float) -> str:
    """构造一条"若干小时前"的落库时间串（naive UTC，与 SQLite 实际存法一致）。"""
    return (NOW - timedelta(hours=hours)).replace(tzinfo=None).isoformat(sep=" ")


# ---------------- _window_age_hours：naive UTC 口径 ----------------
def test_age_treats_naive_timestamp_as_utc() -> None:
    """★ 核心：naive 时间戳按 **UTC** 解释（不是本地时区）。"""
    assert _window_age_hours(_ago(19.5), now=NOW) == 19.5
    assert _window_age_hours(_ago(1), now=NOW) == 1.0
    assert _window_age_hours(_ago(0), now=NOW) == 0.0


def test_age_accepts_aware_timestamp() -> None:
    """带时区的串同样正确（落库可能是 aware，读回多为 naive，两种都要容错）。"""
    assert _window_age_hours(NOW.replace(tzinfo=None).isoformat() + "+00:00", now=NOW) == 0.0


def test_age_none_and_unparsable_do_not_fabricate() -> None:
    """无样本 / 时间损坏 ⇒ `None`（**绝不编造数字**，前端显示"时间未知"）。"""
    assert _window_age_hours(None, now=NOW) is None
    assert _window_age_hours("", now=NOW) is None
    assert _window_age_hours("不是时间", now=NOW) is None


def test_age_negative_when_timestamp_in_future() -> None:
    """未来时间戳 ⇒ 负数（如实返回，便于发现时钟/写入异常）。"""
    assert _window_age_hours(_ago(-2), now=NOW) == -2.0


# ---------------- _sample_quality：两个独立判据 ----------------
def _w(**over) -> dict:
    base = {"limit": 10, "sampled": 10, "table_total": 284, "excluded_empty": 55,
            "from": _ago(30), "to": _ago(19.5)}
    base.update(over)
    return base


def _q(**over) -> dict:
    """`_sample_quality` + 固定时钟（判据必须与真实时钟无关地可复现）。"""
    return _sample_quality(_w(**over), now=NOW)


def test_quality_healthy_window() -> None:
    q = _q()
    assert q["age_hours"] == 19.5
    assert q["stale"] is False, f"19.5h < {_SAMPLE_MAX_AGE_HOURS}h 不应判过时"
    assert q["insufficient"] is False
    assert q["max_age_hours"] == _SAMPLE_MAX_AGE_HOURS


def test_quality_stale_threshold_is_exclusive() -> None:
    """边界：**恰好等于**阈值不算过时（`>` 而非 `>=`），超过才算 —— 判据要可复现。"""
    assert _q(to=_ago(_SAMPLE_MAX_AGE_HOURS))["stale"] is False
    assert _q(to=_ago(_SAMPLE_MAX_AGE_HOURS + 0.1))["stale"] is True


def test_quality_insufficient_when_window_not_filled() -> None:
    """窗口没填满（表里非空样本 < limit）⇒ insufficient（比"n 小"更强的信号）。"""
    assert _q(sampled=6)["insufficient"] is True
    assert _q(sampled=9, limit=10)["insufficient"] is True
    assert _q(sampled=10, limit=10)["insufficient"] is False


def test_quality_no_samples_is_neither_stale_nor_insufficient_by_guess() -> None:
    """无样本：age 为 None ⇒ **不判过时**（没有样本 ≠ 样本过时）；insufficient 如实计算。"""
    q = _sample_quality({"limit": 10, "sampled": 0, "table_total": 284,
                         "excluded_empty": 284, "from": None, "to": None}, now=NOW)
    assert q["age_hours"] is None
    assert q["stale"] is False
    assert q["insufficient"] is True


def test_quality_survives_missing_keys() -> None:
    """键缺失（旧/异常数据）不得 KeyError —— 用 `.get` 兜底。"""
    q = _sample_quality({}, now=NOW)
    assert q["age_hours"] is None and q["stale"] is False
