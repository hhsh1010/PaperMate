"""PaperMate · JSON 反序列化的**容错唯一入口**

### 为什么需要

`meta_json` / `events_json` / `config_json` / `tool_calls_json` 这些列都是**外部写入的
TEXT 列**，出现损坏/半截内容是迟早的事（本项目**已经发生过** `meta_json` 损坏）。
裸 `json.loads` 一旦抛 `JSONDecodeError`，会让**整个接口 500** ——

- 会话历史打不开（`routes_sessions`）
- 用量页打不开（`routes_usage`）

而**同一条数据在别的路由里早有防护**（`routes_traces` 的 `__safe_json`），
⇒ 形成"同一份数据、两种解析口径"的**同源化缺口**（本项目反复吃过这个亏）。

⇒ 把容错收敛为**唯一实现**：所有解析这类列的调用点统一走本模块。

### 语义（重要）

**坏数据 → 返回空容器，绝不抛异常**。调用方拿到的永远是"可用但可能为空"的值；
需要区分"真的空"与"解析失败"时，请显式打日志（本模块**不静默到无痕**——见下）。

⚠️ 本模块**有意不做日志**：它可能在**每次请求的热路径**上被调用，
逐次 warning 会淹没日志。需要可观测时由调用方按需记录。
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

_T = TypeVar("_T")


def _load(s: object) -> Any:
    if not s or not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def safe_json_list(s: object) -> list:
    """解析 JSON 列，**保证返回 list**（非 list 或解析失败 → 空列表）。"""
    data = _load(s)
    return data if isinstance(data, list) else []


def safe_json_dict(s: object) -> dict:
    """解析 JSON 列，**保证返回 dict**（非 dict 或解析失败 → 空字典）。"""
    data = _load(s)
    return data if isinstance(data, dict) else {}


def safe_json_records(s: object) -> list[dict]:
    """解析 JSON 列并**只保留 dict 元素**。

    用于"逐条取字段"的场景（如 `[e.get("type") for e in ...]`）——
    混入非 dict 元素时，直接取字段会抛 `AttributeError` ⇒ 又是 500；
    这里提前过滤，保持"坏数据不影响整页"的口径。
    """
    return [e for e in safe_json_list(s) if isinstance(e, dict)]
