"""LightCore · 反思器（docs/05 §规划与反思 / Reflexion）

Reflector 实现 Reflexion 式自我批判：对一次执行的结果进行评价，若发现错误或不足，
产出"反思"与"改进计划"，最多迭代 `max_rounds=2` 轮（docs/05）。
由 LLM 驱动；测试用 TextFakeLLM 返回结构化评价。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import BaseLLM, ChatMessage, Role

# C7-lightcore（2026-09-23）：提示词与其**派生的解析标记**统一放在 `lightcore/prompts`。
# 关键点：提示词要求模型"第一行输出 SATISFIED"、写"反思: …/改进: …"，
# 而**本文件的解析代码找的就是这些词** —— 这是一对合约，两边必须同时改。
# 原先两边各写字面串（测试里还写了第三遍）→ 只改一边就**静默失效**。
from ..prompts import (
    REFLECT_CRITIQUE_LABEL,
    REFLECT_IMPROVEMENT_LABEL,
    REFLECT_PROMPT,
    REFLECT_SATISFIED,
)


@dataclass
class Reflection:
    satisfied: bool
    critique: str = ""
    improvement: str = ""


class Reflector:
    def __init__(self, llm: BaseLLM, max_rounds: int = 2) -> None:
        self.llm = llm
        self.max_rounds = max_rounds

    async def reflect(
        self, goal: str, result: str, *, max_tokens: int | None = None
    ) -> Reflection:
        """对一次执行结果做自评。

        `max_tokens`（2026-09-16 新增）：**调用方必须设上限**。自评属"元任务"，
        跑在用户等待路径上；不设上限时 thinking 模型会把思考链吃满数千 token，
        单次耗时失控且无可见输出（N11 那次事故的成因，见 `core/loop.py` 的护栏常量）。
        """
        prompt = f"{REFLECT_PROMPT}\n\n目标：{goal}\n\n执行结果：{result}"
        resp = await self.llm.chat(
            [ChatMessage(role=Role.USER, content=prompt)], max_tokens=max_tokens
        )
        lines = [ln.strip() for ln in resp.message.content.splitlines() if ln.strip()]
        if not lines:
            return Reflection(satisfied=True)
        satisfied = lines[0].upper().startswith(REFLECT_SATISFIED)
        critique = improvement = ""
        for ln in lines[1:]:
            if ln.startswith(REFLECT_CRITIQUE_LABEL):
                critique = ln.split(":", 1)[-1].strip()
            elif ln.startswith(REFLECT_IMPROVEMENT_LABEL):
                improvement = ln.split(":", 1)[-1].strip()
        return Reflection(satisfied=satisfied, critique=critique, improvement=improvement)
