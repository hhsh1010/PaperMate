"""服务类 LLM 调用的**统一计费记录**（2026-10-23）。

## 为什么需要它
`messages` 表只记"对话里的调用"，而下面这些**无会话**的服务此前**完全不记账** ✗：
- 标题翻译（`routes_papers.translate_paper_titles`）
- 一键总结 / 相关工作 / 对比（`summary_service`）
- 提示词增强（`routes_prompt`）

⇒ `/usage` 页面与上下文面板里的 ¥ 金额**只含对话+技能**，服务类调用是黑账 ✗
（而这些恰恰是"批量、容易烧钱"的地方 —— 一次翻译 20 条标题、一次总结 5 篇文档）

## 设计
**单点写入**：所有服务调用都走 `record_service_usage(...)`，避免各处各写一遍（口径漂移）。
失败**绝不影响主流程**：记账是旁路，异常只记日志不抛。
"""

from __future__ import annotations

import logging
from typing import Any

from ..models.db import ServiceUsage

logger = logging.getLogger(__name__)


def record_service_usage(
    session_factory: Any,
    *,
    source: str,
    model: str,
    response: Any = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost: float | None = None,
) -> None:
    """记一笔服务调用用量（旁路，失败不抛）。

    优先从 `LLMResponse` 上取（`prompt_tokens` / `completion_tokens` / `cost`）；
    也可显式传入（用于自算成本的场合）。

    ★ 只在**调用成功**时调用它 —— 失败（403/429）本就没有 token 消耗，不必记 0 值噪音。
    """
    try:
        if response is not None:
            prompt_tokens = prompt_tokens if prompt_tokens is not None else int(getattr(response, "prompt_tokens", 0) or 0)
            completion_tokens = (
                completion_tokens if completion_tokens is not None
                else int(getattr(response, "completion_tokens", 0) or 0)
            )
            cost = cost if cost is not None else float(getattr(response, "cost", 0.0) or 0.0)
        with session_factory() as db:
            db.add(ServiceUsage(
                source=source,
                model=model or "",
                prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(completion_tokens or 0),
                cost=float(cost or 0.0),
            ))
            db.commit()
    except Exception:  # noqa: BLE001  记账是旁路：失败只告警，绝不影响业务
        logger.warning("[usage] 服务调用记账失败 source=%s model=%s", source, model, exc_info=True)
