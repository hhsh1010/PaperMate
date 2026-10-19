"""论文标题中译（**批量、单次请求、可校验**）

## 为什么这么设计
- **批量**：20 条标题逐条翻译 = 20 次 LLM 调用（贵、慢、更容易撞限流）；塞进**一个** prompt
  ⇒ 1 次调用、1 次计费，且能让模型看到全局上下文、译法更一致。
- **严格 JSON + 条数校验**：模型偶尔会多/少一条、或用 ```json 包起来。
  这里**宽松解析、严格校验**：能救的救（剥代码块），救不了的返回 None（前端保留原文），
  **不猜、不错位**（错位比不翻译严重得多）。
- **不点不花钱**：本模块不自动触发，只由前端「翻译标题」按钮显式调用。

## 三条硬约束（项目对 LLM 功能的约定）
1. 有明确入口（前端按钮）⇒ 用户不点就没有成本；
2. 有上限（一次最多 20 条；`max_tokens` 受限）；
3. 失败**可读**（模型未配置 / 返回非 JSON / 条数不符 ⇒ 都给人能懂的提示）。
"""

from __future__ import annotations

import json
import re

#: 一次最多翻译多少条（与检索单页条数一致）
MAX_TITLES = 20

#: 单次调用的输出上限（20 条中文标题 ≈ 20×30 字，留足余量）
MAX_OUTPUT_TOKENS = 2000

SYSTEM_PROMPT = (
    "你是学术论文标题翻译助手。把英文论文标题翻译成**简洁准确的中文**。"
    "要求：保留模型名、方法名、数据集名与通用缩写（如 RAG、ResNet、LiDAR、Transformer）不译；"
    "不要添加解释、不要扩写；只输出 JSON 数组，格式为 [{\"i\":1,\"zh\":\"…\"}]，i 与输入编号一致。"
)


def build_messages(titles: list[str]) -> list[dict[str, str]]:
    """构造一次批量翻译的 messages。编号从 1 开始，便于回填与校验。"""
    lines = [f"{i}. {t.strip()}" for i, t in enumerate(titles, 1)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "请翻译下列论文标题：\n" + "\n".join(lines)},
    ]


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _strip_fence(raw: str) -> str:
    m = _JSON_FENCE.search(raw or "")
    return (m.group(1) if m else raw or "").strip()


def parse_titles_json(raw: str, expected: int) -> list[str | None]:
    """把模型输出解析成**长度恰为 expected** 的列表；无法对应的位置为 None。

    宽松处：允许 ```json 代码块、允许对象/数组两种外层（`{"items":[...]}`）、允许 i 从 1 或 0 开始。
    严格处：**只按 i 回填**，不按顺序猜（顺序错一位会让标题张冠李戴）；条数不符不回填。
    """
    if expected <= 0:
        return []
    text = _strip_fence(raw)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        # 兜底：从文本里抠出第一个 [...] 再试一次（模型偶尔会在前后加话）
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return [None] * expected
        try:
            data = json.loads(text[start:end + 1])
        except (TypeError, ValueError):
            return [None] * expected

    if isinstance(data, dict):
        data = data.get("items") or data.get("titles") or []
    if not isinstance(data, list):
        return [None] * expected

    out: list[str | None] = [None] * expected
    for item in data:
        if not isinstance(item, dict):
            continue
        zh = str(item.get("zh") or item.get("translation") or "").strip()
        if not zh:
            continue
        try:
            idx = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        # 兼容 i 从 1 开始（约定）与从 0 开始（模型偶发）
        pos = idx - 1 if 1 <= idx <= expected else idx
        if 0 <= pos < expected and out[pos] is None:
            out[pos] = zh
    return out


def is_fake_model(model_id: str | None) -> bool:
    """是否是"没有真实模型"的退化状态（fake / 空）。

    ★ 必须显式拦住：`_llm_from_raw` 对**未知/缺失**会**静默退化成 ApiFakeLLM**，
    那种情况下模型会返回占位文本 ⇒ 用户看到一堆假标题，比报错更糟。
    """
    m = (model_id or "").strip().lower()
    return m in ("", "fake", "api-fake", "apifake")
