"""模型目录（catalog）一致性测试（2026-10-05）

**为什么需要它**：`router.py` 的 10 张派生表只要"漏改一张"，就会**静默降级** ——
忘填窗口 ⇒ 上下文预算悄悄回退；忘填定价 ⇒ 成本恒 0 ⇒ **¥1 熔断失效**。
两者都**不报错**，这正是本次"模型目录化"要根治的问题。

本测试守三件事：
1. **目录完整性** —— 每条 `ModelSpec` 的 provider 必须已在 `PRESET_ENDPOINTS` 接入；
2. **零行为变化**（重构回归基线）—— 10 张派生表与「模型目录化之前」的硬编码取值**逐项一致**。
   这是"重构不改行为"的证据；将来若**有意**变更某张表的取值，必须**同时改本测试**
   （改动会被 reviewer 看到，而不是悄悄发生）；
3. **成本护栏** —— 未知模型不再返回 0，而是按目录内最贵档保守估算。
"""

from __future__ import annotations

import pytest

from lightcore.llm import router as R
from lightcore.llm.catalog import (
    BUILTIN_MODELS,
    LONG_CONTEXT_MIN,
    MAX_PRICE_IN,
    MAX_PRICE_OUT,
    MODEL_CATALOG,
    ModelSpec,
    enabled_models_by_provider,
)

# ---------------------------------------------------------------- 1. 目录完整性


def test_every_provider_is_preset() -> None:
    """新加模型时最容易漏的一步：provider 未在 PRESET_ENDPOINTS ⇒ 无法构造端点。"""
    unknown = {m.provider for m in BUILTIN_MODELS} - set(R.PRESET_ENDPOINTS)
    assert not unknown, f"以下 provider 尚未在 PRESET_ENDPOINTS 接入：{sorted(unknown)}"


def test_window_and_price_are_required_fields() -> None:
    """★ 窗口 / 定价必须是**必填**：漏填时在构造期就抛错，而不是运行时静默回退。

    这条是本次改造的核心保证 —— 反向自检：故意少传两个参数，必须 `TypeError`。
    """
    with pytest.raises(TypeError):
        ModelSpec("x", "qwen", 1000)  # type: ignore[call-arg]   缺 price_in / price_out


def test_model_ids_unique_and_indexed() -> None:
    ids = [m.model_id for m in BUILTIN_MODELS]
    assert len(ids) == len(set(ids)), "model_id 重复"
    assert set(MODEL_CATALOG) == set(ids)


def test_long_context_is_derived_not_handwritten() -> None:
    """长文候选由窗口**派生** ⇒ 不可能与窗口表自相矛盾。"""
    expected = {m.model_id for m in BUILTIN_MODELS if m.context_window >= LONG_CONTEXT_MIN}
    assert R.LONG_CONTEXT_MODELS == expected


def test_enabled_models_by_provider_matches_provider_models() -> None:
    assert R.PROVIDER_MODELS == enabled_models_by_provider()


# ------------------------------------------------- 2. 零行为变化（重构回归基线）
# 期望值 = 「模型目录化之前」`router.py` 里的硬编码取值（逐项抄录）。
# ⚠️ 若某条断言失败，先问「是不是**有意**改了这个取值」——
#    有意则连同本测试一起更新（并在提交信息里说明）；无意则是回归。

_EXPECTED_PRICING = {
    "deepseek-chat": {"in": 0.001, "out": 0.002},
    "gpt-4o-mini": {"in": 0.0011, "out": 0.0043},
    "qwen-plus": {"in": 0.0008, "out": 0.002},
    "qwen3-max": {"in": 0.02, "out": 0.06},
    "deepseek-v4-flash-vision-exp": {"in": 0.001, "out": 0.002},
    "deepseek-v4-flash": {"in": 0.001, "out": 0.002},
    "qwen3.7-flash": {"in": 0.00024, "out": 0.00096},
    "qwen3.6-flash": {"in": 0.00144, "out": 0.00864},
    "qwen3-vl-flash": {"in": 0.00015, "out": 0.0015},
}

_EXPECTED_CONTEXT_WINDOW = {
    "qwen3-max": 262_144,
    "qwen-turbo": 1_000_000,
    "qwen-plus": 131_072,
    "qwen-long": 131_072,
    "qwen-max": 32_768,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "deepseek-v4-flash-vision-exp": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "qwen3.7-flash": 1_000_000,
    "qwen3.6-flash": 1_000_000,
    "qwen3-vl-flash": 262_144,
}

_EXPECTED_PROVIDER_MODELS = {
    "deepseek": ["deepseek-v4-flash-vision-exp", "deepseek-v4-flash"],
    "qwen": ["qwen3.7-flash", "qwen3.6-flash", "qwen3-vl-flash", "qwen3-max"],
}

_EXPECTED_WEB_SEARCH = frozenset({
    "qwen3-max", "qwen-max", "qwen3.6-flash", "qwen3.7-flash", "qwen-turbo",
})

_EXPECTED_HIDDEN = {
    "qwen-max": "qwen",
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4o-mini-2024-07-18": "openai",
    "qwen-plus": "qwen",
    "qwen-turbo": "qwen",
    "qwen-long": "qwen",
}


def test_derived_pricing_unchanged() -> None:
    assert R._PRICING == _EXPECTED_PRICING  # noqa: SLF001


def test_derived_context_window_unchanged() -> None:
    assert R.MODEL_CONTEXT_WINDOW == _EXPECTED_CONTEXT_WINDOW


def test_derived_provider_models_unchanged() -> None:
    assert R.PROVIDER_MODELS == _EXPECTED_PROVIDER_MODELS


def test_derived_web_search_set_unchanged() -> None:
    """★ 退役模型（qwen-max / qwen-turbo）的联网结论必须被 union 回来，否则行为回退。"""
    assert R.NATIVE_WEB_SEARCH_MODELS == _EXPECTED_WEB_SEARCH


def test_derived_hidden_models_unchanged() -> None:
    assert R._HIDDEN_MODELS == _EXPECTED_HIDDEN  # noqa: SLF001


def test_derived_model_to_provider_unchanged() -> None:
    expected = {m: p for p, ms in _EXPECTED_PROVIDER_MODELS.items() for m in ms}
    expected.update(_EXPECTED_HIDDEN)
    assert R.MODEL_TO_PROVIDER == expected


def test_derived_multimodal_and_doc_bound_unchanged() -> None:
    assert R.MULTIMODAL_MODELS == {"deepseek-v4-flash-vision-exp", "qwen3-vl-flash"}
    assert R.DOC_BOUND_UNRELIABLE_MODELS == {"qwen3-vl-flash"}


def test_derived_qwen3_thinking_set_unchanged() -> None:
    assert R._QWEN3_THINKING_MODELS == {  # noqa: SLF001
        "qwen3.7-flash", "qwen3.6-flash", "qwen3-vl-flash",
    }


# ---------------------------------------------------------------- 3. 成本护栏


def test_estimate_cost_known_model_unchanged() -> None:
    """已知模型：1k in + 1k out 应等于 price_in + price_out。"""
    m = MODEL_CATALOG["qwen3-max"]
    assert R.estimate_cost("qwen3-max", 1000, 1000) == pytest.approx(
        m.price_in + m.price_out
    )


def test_estimate_cost_unknown_model_is_conservative_not_zero() -> None:
    """★ 核心回归：未登记定价的模型**不得返回 0**（否则 ¥1 熔断静默失效）。"""
    cost = R.estimate_cost("no-such-model-xyz", 1000, 1000)
    assert cost > 0, "未知模型成本不得为 0 —— 否则成本熔断永不触发"
    assert cost == pytest.approx(MAX_PRICE_IN + MAX_PRICE_OUT), "应按目录内最贵档估算"


def test_unknown_model_cost_exceeds_cheapest_known() -> None:
    """保守性验证：未知模型的估算**不低于**任何已知模型（宁可高估）。"""
    cheapest = min(
        R.estimate_cost(m.model_id, 1000, 1000) for m in BUILTIN_MODELS
    )
    assert R.estimate_cost("unknown-xyz", 1000, 1000) >= cheapest
