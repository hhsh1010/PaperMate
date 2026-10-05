"""LightCore · 规划器（docs/05 §规划与反思 / Plan-then-Execute）

Planner 将用户目标分解为有序子任务（Plan-then-Execute），限制 ≤5 步（docs/05）。
输出 `Plan`：steps 有序、可含依赖标注。由 LLM 驱动；测试用 TextFakeLLM 返回结构化计划。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import BaseLLM, ChatMessage, Role

# C7-lightcore（2026-09-23）：提示词已集中到 `lightcore/prompts`，此处**只导入不再定义**。
# 保留模块级名字是为了不破坏既有引用方（`from ..planning.planner import PLANNER_PROMPT`）。
from ..prompts import PLANNER_PROMPT


@dataclass
class Step:
    index: int
    description: str
    depends_on: list[int] = field(default_factory=list)


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)


class Planner:
    max_steps: int = 5

    def __init__(self, llm: BaseLLM, max_steps: int = 5) -> None:
        self.llm = llm
        self.max_steps = max_steps

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\n目标：{goal}"
        resp = await self.llm.chat([ChatMessage(role=Role.USER, content=prompt)])
        steps: list[Step] = []
        for i, raw in enumerate(resp.message.content.splitlines()):
            raw = raw.strip().lstrip("0123456789.、-").strip()
            if not raw:
                continue
            deps: list[int] = []
            if "[" in raw and "]" in raw:
                seg = raw[raw.index("[") + 1: raw.index("]")]
                deps = [int(x) for x in seg.replace(",", " ").split() if x.isdigit()]
                raw = raw[: raw.index("[")].strip()
            steps.append(Step(index=i + 1, description=raw, depends_on=deps))
            if len(steps) >= self.max_steps:
                break
        return Plan(goal=goal, steps=steps)
