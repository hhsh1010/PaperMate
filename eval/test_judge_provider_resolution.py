"""判官厂商解析的守护测试（2026-09-27）。

**为什么需要**：`_resolve_providers` 原先**一律拒绝**同厂商 —— 正常情况下这是对的
（同源自评会系统性偏乐观）。但当**异源厂商整体不可用**时（2026-09-27 实测：
qwen 免费额度耗尽、三个模型全 403），"强制跨厂商"会让基线**永远补不齐**
（两家配对里必然有一方是 qwen）。

于是加了**受限放宽** `--allow-same-provider-judge`。本文件钉死它的边界：

| 场景 | 期望 |
|---|---|
| 异源（默认路径） | 放行 —— 放宽不得影响它 |
| 同厂商、无开关 | **拒绝**（纪律不能被静默削弱） |
| 同厂商、同模型、有开关 | **拒绝**（同模型 = 真自评，任何情况都不接受） |
| 同厂商、不同模型、有开关 | 放行（受限放宽的唯一出口） |
"""

from __future__ import annotations

from types import SimpleNamespace

import judge as judge_mod
import pytest
import run_eval as R


def _args(**kw: object) -> SimpleNamespace:
    """构造 `_resolve_providers` 需要的最小 args（只含它读的字段）。"""
    base: dict[str, object] = {
        "answer_provider": "deepseek",
        "judge_provider": "qwen",
        "answer_model": None,
        "judge_model": None,
        "allow_same_provider_judge": False,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_cross_provider_passes() -> None:
    """异源（默认路径）必须正常放行 —— 放宽不能影响它。"""
    assert R._resolve_providers(_args()) == ("deepseek", "qwen")  # noqa: SLF001


def test_same_provider_rejected_by_default() -> None:
    """未加开关时同厂商仍必须被拒绝（纪律不能被静默削弱）。"""
    with pytest.raises(judge_mod.JudgeUnavailable, match="同厂商"):
        R._resolve_providers(_args(judge_provider="deepseek"))  # noqa: SLF001


def test_same_provider_same_model_rejected_even_with_flag() -> None:
    """同厂商 + 同模型 = 真自评 ⇒ 即便显式加开关也不接受。"""
    with pytest.raises(judge_mod.JudgeUnavailable, match="同一模型"):
        R._resolve_providers(  # noqa: SLF001
            _args(
                judge_provider="deepseek",
                answer_model="deepseek-v4-flash",
                judge_model="deepseek-v4-flash",
                allow_same_provider_judge=True,
            )
        )


def test_same_provider_diff_model_allowed_with_flag() -> None:
    """同厂商 + **不同模型** + 显式开关 ⇒ 放行（受限放宽的唯一出口）。"""
    got = R._resolve_providers(  # noqa: SLF001
        _args(
            judge_provider="deepseek",
            answer_model="deepseek-v4-flash",
            judge_model="deepseek-v4-flash-vision-exp",
            allow_same_provider_judge=True,
        )
    )
    assert got == ("deepseek", "deepseek")
