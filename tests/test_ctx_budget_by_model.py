"""单次上下文预算：**按模型窗口自动推断**（2026-10-03）。

## 修的缺陷
`max_tokens_budget` 是**固定值 30000**（历史遗留：它当年还兼管"整轮累计"，
2026-09-17 改成只管单次后**数值没跟着调**）。而实测模型窗口差 4 倍
（`qwen3-vl-flash` 262k vs `qwen3.6-flash` 1M）⇒ 固定值对谁都不合适：
对大窗口**浪费 97%**，对将来接入的小窗口（表里有 32k 的 `qwen-max`）会**超窗**。

## ★ 本文件守的三条不变量
1. **按窗口分档**，不是"越大约好" —— 有 100k 硬上限（窗口只决定"不能超多少"）；
2. **显式设置优先**（`max_tokens_budget_auto=False` ⇒ 用显式值）；
3. ★★ **模型未登记时**不猜** ⇒ 回落显式值**（`MODEL_CONTEXT_WINDOW` 自注"近似值"，
   拿它做硬判据本身有风险）。
"""
from __future__ import annotations

import pytest

from papermate.runtime_settings import (
    TURN_BUDGET_CAP,
    TURN_BUDGET_FLOOR,
    TURN_BUDGET_RATIO,
    get_state,
    input_token_limit,
    resolve_budget_for,
    resolve_turn_budget_for,
    rules_token_limit,
)


@pytest.fixture(autouse=True)
def _auto_on(monkeypatch):
    """默认让"自动"处于开启态（与 DEFAULT 一致），并清掉显式值干扰。"""
    import papermate.runtime_settings as rs

    st = get_state()
    monkeypatch.setitem(st, "max_tokens_budget_auto", True)
    monkeypatch.setitem(st, "max_tokens_budget", 30000)
    yield
    _ = rs  # 保持引用，便于阅读


class TestByWindow:
    def test_1m_window_caps_at_100k(self):
        """1M 窗口 ⇒ 100k（**不是** 250k/700k）—— 硬上限保护成本与延迟。"""
        assert resolve_budget_for("qwen3.6-flash") == 100_000
        assert resolve_budget_for("deepseek-v4-flash") == 100_000

    def test_262k_window_scales_down(self):
        """262k 窗口 ⇒ 65536（262144 × 0.25），**小于** 1M 档 —— 这才叫"按模型适配"。"""
        assert resolve_budget_for("qwen3-vl-flash") == 65_536
        assert resolve_budget_for("qwen3-max") == 65_536

    def test_larger_window_never_smaller(self):
        """窗口更大 ⇒ 预算不小于小窗口（单调性；防止比例/夹取写反）。"""
        assert resolve_budget_for("qwen3.6-flash") >= resolve_budget_for("qwen3-vl-flash")

    def test_provider_prefixed_name_is_normalized(self):
        """★ 调用方可能传 `provider:model` ⇒ 必须能识别（否则静默退回显式值）。"""
        assert resolve_budget_for("qwen:qwen3.6-flash") == resolve_budget_for("qwen3.6-flash")
        assert resolve_budget_for("deepseek:deepseek-v4-flash") == 100_000


class TestFallback:
    def test_unknown_model_falls_back_to_explicit(self):
        """★★ **未登记的模型不猜** —— 回落显式值（宁可保守）。"""
        assert resolve_budget_for("某个不存在的模型") == 30_000
        assert resolve_budget_for("") == 30_000
        assert resolve_budget_for(None) == 30_000

    def test_explicit_wins_when_auto_off(self, monkeypatch):
        """★ 用户显式关闭自动 ⇒ 用他的值（自动只做兜底，不覆盖显式选择）。"""
        st = get_state()
        monkeypatch.setitem(st, "max_tokens_budget_auto", False)
        monkeypatch.setitem(st, "max_tokens_budget", 24_000)
        assert resolve_budget_for("qwen3.6-flash") == 24_000
        assert resolve_budget_for("qwen3-vl-flash") == 24_000

    def test_small_window_gets_floor_not_zero(self):
        """极小窗口 ⇒ 受下限保护（不得算出 0/负数）。

        表中没有 ≤48k 的**已接入**模型，这里直接用一个虚构的窗口验证夹取逻辑。
        """
        import papermate.runtime_settings as rs

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setitem(rs.MODEL_CONTEXT_WINDOW, "tiny-model", 8_000)
        try:
            # 8000 × 0.25 = 2000 < 下限 ⇒ 取下限
            assert resolve_budget_for("tiny-model") == rs._BUDGET_AUTO_FLOOR
        finally:
            monkeypatch.undo()


class TestSameSource:
    """★★ 派生值必须**同源** —— 否则输入上限按一个数算、实际预算按另一个数算（两把尺子）。"""

    def test_input_limit_follows_model(self):
        assert input_token_limit("qwen3.6-flash") == 50_000  # 100k × 0.5
        assert input_token_limit("qwen3-vl-flash") == 32_768  # 65536 × 0.5
        # 不传 model ⇒ 与显式值同源（旧调用方兼容）
        assert input_token_limit() == 15_000  # 30k × 0.5

    def test_rules_limit_follows_model(self):
        assert rules_token_limit("qwen3.6-flash") == 10_000  # 100k × 0.1
        assert rules_token_limit() == 3_000

    def test_input_limit_never_exceeds_budget(self):
        for m in ("qwen3.6-flash", "qwen3-vl-flash", None):
            assert input_token_limit(m) <= resolve_budget_for(m)


class TestDefaults:
    def test_auto_is_default_when_key_absent(self, monkeypatch):
        """★ 老用户的 json 里**没有** `max_tokens_budget_auto` 这个键（但有 30000）——
        此时必须**默认按自动**，否则本次改动对他们完全不生效。

        ⚠️ 本模块**没有**导出的 `DEFAULT_SETTINGS` 常量（默认值在 `_load()` 内构造）
        ⇒ 用"把键删掉"来模拟"老 json"，直接断言**行为**，而不是断言一个不存在的常量。
        """
        import papermate.runtime_settings as rs

        st = dict(get_state())
        st.pop("max_tokens_budget_auto", None)
        monkeypatch.setattr(rs, "_state", st)
        assert resolve_budget_for("qwen3.6-flash") == 100_000, (
            "缺键时未按自动 ⇒ 老用户仍被固定 30000 卡住"
        )

    def test_explicit_key_present_but_true_also_auto(self, monkeypatch):
        """显式写了 `auto: true` ⇒ 同样走自动。"""
        import papermate.runtime_settings as rs

        st = dict(get_state())
        st["max_tokens_budget_auto"] = True
        monkeypatch.setattr(rs, "_state", st)
        assert resolve_budget_for("qwen3.6-flash") == 100_000


class TestTurnBudgetSameSource:
    """★★ 整轮累计上限必须与**单次预算同源派生**（2026-10-03 事故）。

    事故：`max_tokens_budget` 由 30k 提到 100k 后，固定的 `max_tokens_per_turn=150000`
    **先于**单次被撞 ⇒ 报「本轮累计 token 超限（156085 > 150000）」。
    ⇒ 两个量必须同步，否则调了一个另一个就是错的。
    """

    def test_turn_is_derived_from_budget(self):
        for m in ("qwen3.6-flash", "qwen3-vl-flash", None):
            base = resolve_budget_for(m)
            turn = resolve_turn_budget_for(m)
            assert turn >= base, f"{m}: 整轮({turn}) 必须 ≥ 单次({base})"
            assert turn == max(TURN_BUDGET_FLOOR, min(base * TURN_BUDGET_RATIO, TURN_BUDGET_CAP))

    def test_1m_window_turn_covers_observed_abort(self):
        """★★ 锚定**两次**真实熔断（后者推翻了前者的取值依据）。

        - 2026-10-03：旧配置（单次 30k）撞整轮固定 150k，实测 156085；
        - 2026-10-04：新配置（单次 100k）撞按模型派生的 200k，实测 **214315**
          （trace 396，8 步、13 次工具调用、单次峰值仅 43k）。

        ⚠️ 后者暴露我第一版的 `×2` 是**跨配置推理**得出的（拿旧配置的单次峰值 24.8k
        去推新配置的倍数）⇒ 已改为同口径实测外推的 `×5`。
        """
        assert resolve_turn_budget_for("qwen3.6-flash") > 156_085
        assert resolve_turn_budget_for("qwen3.6-flash") > 214_315

    def test_turn_covers_full_iterations(self):
        """★ 满轮（`max_iterations=12`）不得中途被累计护栏拦下。

        trace 396 实测单次 prompt（估算×1.12 换算）：
        `19k→24k→30k→32k→33k→37k→39k→43k`，按末段 +3.2k/步 外推 12 步 ⇒ 累计 ≈ **463k**。
        ⇒ 1M 窗口的整轮上限必须高于它，否则**跑不满轮**就熔断。
        """
        turn = resolve_turn_budget_for("qwen3.6-flash")
        assert turn > 463_000, f"整轮上限 {turn} 覆盖不住 12 步外推的 463k"

    def test_short_window_scales_by_ratio_not_floor(self):
        """★ 短窗口模型（单次 30k）走 **ratio** 而非 floor：30k × 5 = 150k。

        ⚠️ 由此暴露一个事实：`TURN_BUDGET_FLOOR = 60_000` 恰好等于
        `resolve_budget_for` 的下限（12_000）× ratio（5）⇒ **auto 模式下它永不独立生效**，
        只在「关掉 auto 且把 budget 显式设得极小」时兜底。
        保留它是为了那条路径不至于写出 0 或负数。
        """
        assert resolve_turn_budget_for(None) == 150_000
        assert resolve_turn_budget_for(None) >= TURN_BUDGET_FLOOR

    def test_cap_prevents_runaway(self):
        """★ 上限必须存在（否则 1M 窗口时整轮可无限放大，失去"防跑飞"意义）。

        ⚠️ CAP 必须 **≥** `100k × ratio`，否则会把刚放大的值又夹回去
        （第一版 CAP=300k 而 ratio 提到 5 时，500k 会被夹到 300k ⇒ 修复失效）。
        """
        assert resolve_turn_budget_for("qwen3.6-flash") <= TURN_BUDGET_CAP
        assert TURN_BUDGET_CAP >= resolve_budget_for("qwen3.6-flash") * TURN_BUDGET_RATIO
