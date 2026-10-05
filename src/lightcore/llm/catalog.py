"""LightCore · 模型目录（**单一数据源**）

**为什么要有它**：接入一个模型原本要同步改 `router.py` 里 **12 张硬编码表**
（定价 / 窗口 / 联网 / 多模态 / 长文 / 模型→厂商 / 可选列表 / 退役映射 / thinking 适用集 / 风险名单）。
漏改任何一张都**不会报错**，只会静默降级：
- 忘填窗口 ⇒ 上下文预算悄悄回退到默认值（`runtime_settings.resolve_budget_for`）；
- 忘填定价 ⇒ `estimate_cost` 返回 0 ⇒ **¥1 成本熔断永不触发**。

⇒ 现在**只在本模块声明一次** `ModelSpec`，`router.py` 的 12 张表**全部派生**。

**新增模型（这就是"预留模型接口"）**：
    1. 在 `BUILTIN_MODELS` 追加一条 `ModelSpec`；
    2. 确认其 `provider` 已在 `router.PRESET_ENDPOINTS` 中（有测试守这条）。
**没有第 3 步。**

⚠️ `context_window` / `price_in` / `price_out` 是**必填参数**（无默认值）：
漏填在 **import 期**就抛 `TypeError`，而不是运行时静默回退 —— 这正是本模块的核心价值。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 长上下文阈值（tokens）。`ModelSpec.long_context` 由此**派生** ——
#: 不手填，避免「窗口写了 1M 却忘了加进长文候选」这类自相矛盾。
LONG_CONTEXT_MIN = 128_000


@dataclass(frozen=True)
class ModelSpec:
    """一个已接入模型所需的**全部**元数据。"""

    model_id: str
    provider: str
    context_window: int          # tokens；`resolve_budget_for` 依赖它
    price_in: float              # ¥ / 1k tokens（输入）；成本熔断依赖它
    price_out: float             # ¥ / 1k tokens（输出）
    supports_tools: bool = True  # 是否支持 function calling
    native_web_search: bool = False
    multimodal: bool = False
    thinking_style: str = "none"      # none | deepseek_v4 | qwen3
    doc_bound_unreliable: bool = False  # 绑文档场景下工具服从不可靠（风险提示用）
    enabled: bool = True               # False ⇒ 不进前端下拉（退役模型）

    @property
    def long_context(self) -> bool:
        """是否入选「长上下文」候选（**派生**，见 `LONG_CONTEXT_MIN`）。"""
        return self.context_window >= LONG_CONTEXT_MIN


#: ★★ **新增模型只改这里**（并确认 provider 已在 `PRESET_ENDPOINTS`）。
#: 声明顺序 = 前端下拉顺序；同一 provider 的**首条**即其默认模型。
BUILTIN_MODELS: tuple[ModelSpec, ...] = (
    # ---------------------------------------------------------------- DeepSeek
    # V4-Flash-Vision（实验性多模态）：1M 上下文 / 图片按 ≤384 tokens/张折算，计费与 V4-Flash 一致。
    ModelSpec(
        "deepseek-v4-flash-vision-exp", "deepseek", 1_000_000, 0.001, 0.002,
        multimodal=True, thinking_style="deepseek_v4",
    ),
    # V4-Flash（官方 2026-04 发布；deepseek-chat/reasoner 已退役）
    ModelSpec(
        "deepseek-v4-flash", "deepseek", 1_000_000, 0.001, 0.002,
        thinking_style="deepseek_v4",
    ),
    # ---------------------------------------------------------------- Qwen（百炼）
    # qwen3 flash 系列：1M 上下文；定价按 ≤32K 输入档取近似（¥/1k tokens，北京区）
    ModelSpec(
        "qwen3.7-flash", "qwen", 1_000_000, 0.00024, 0.00096,
        native_web_search=True, thinking_style="qwen3",
    ),
    ModelSpec(
        "qwen3.6-flash", "qwen", 1_000_000, 0.00144, 0.00864,
        native_web_search=True, thinking_style="qwen3",
    ),
    # VL（多模态文档理解）：262K；doc-bound 场景工具服从不可靠（2026-09-03 实测）
    ModelSpec(
        "qwen3-vl-flash", "qwen", 262_144, 0.00015, 0.0015,
        multimodal=True, thinking_style="qwen3", doc_bound_unreliable=True,
    ),
    # Qwen3-Max：262K，长文总结用；**不参与 thinking**（非 flash 系列）
    ModelSpec(
        "qwen3-max", "qwen", 262_144, 0.02, 0.06, native_web_search=True,
    ),
)

#: 按 id 索引（`router.py` 全部派生表的唯一来源）
MODEL_CATALOG: dict[str, ModelSpec] = {m.model_id: m for m in BUILTIN_MODELS}

#: 成本估算的**保守兜底档位**（未命中定价时使用）。
#: 取目录内最贵档 —— 宁可高估触发熔断，也不要「成本恒 0 ⇒ 护栏失效」。
MAX_PRICE_IN: float = max(m.price_in for m in BUILTIN_MODELS)
MAX_PRICE_OUT: float = max(m.price_out for m in BUILTIN_MODELS)


# ==================== 退役模型（保留解析兼容，不进下拉） ====================
# 为什么留在本模块：它们的元数据在旧实现里散落在 router.py 的 4 张表中
# （`_HIDDEN_MODELS` / `_PRICING` / `MODEL_CONTEXT_WINDOW` / `NATIVE_WEB_SEARCH_MODELS`）。
# 集中于此 ⇒ ①「哪些表该派生」一眼可见；② **保证 router 派生后取值与改动前逐项一致**。

#: 已下线但旧会话 config 仍可能引用 → 保留 provider 解析（不展示）
LEGACY_MODEL_TO_PROVIDER: dict[str, str] = {
    "qwen-max": "qwen",
    "deepseek-chat": "deepseek",       # 已退役（路由到 v4-flash），兼容旧会话 config
    "deepseek-reasoner": "deepseek",   # 同上
    "gpt-4o": "openai",                # openai 厂商已整体下线，仅保留解析
    "gpt-4o-mini": "openai",
    "gpt-4o-mini-2024-07-18": "openai",
    "qwen-plus": "qwen",
    "qwen-turbo": "qwen",
    "qwen-long": "qwen",
}

#: 退役模型仍保留的定价（改动前 `_PRICING` 中确有这三条）
LEGACY_MODEL_PRICING: dict[str, dict[str, float]] = {
    "deepseek-chat": {"in": 0.001, "out": 0.002},
    "gpt-4o-mini": {"in": 0.0011, "out": 0.0043},
    "qwen-plus": {"in": 0.0008, "out": 0.002},
}

#: 退役模型的窗口（改动前 `MODEL_CONTEXT_WINDOW` 中确有这些条目）
LEGACY_MODEL_CONTEXT_WINDOW: dict[str, int] = {
    "qwen-turbo": 1_000_000,
    "qwen-plus": 131_072,
    "qwen-long": 131_072,
    "qwen-max": 32_768,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
}

#: 退役模型里**实测支持**原生联网的两个 —— 保留判定，避免行为回退
LEGACY_NATIVE_WEB_SEARCH: frozenset[str] = frozenset({"qwen-max", "qwen-turbo"})


def enabled_models_by_provider() -> dict[str, list[str]]:
    """`provider -> [model_id, ...]`（仅 `enabled=True`），顺序 = 声明顺序。"""
    out: dict[str, list[str]] = {}
    for m in BUILTIN_MODELS:
        if m.enabled:
            out.setdefault(m.provider, []).append(m.model_id)
    return out
