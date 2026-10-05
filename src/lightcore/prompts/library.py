"""LightCore 提示词**单一来源**（C7-lightcore，2026-09-23）。

**为什么要有这个文件**：
`lightcore` 的提示词此前散落在各模块里（`memory/short_term.py` / `planning/planner.py` /
`planning/reflector.py` 各定义一个常量，另有若干**内联**在函数体中）。
散落本身不会立刻出错，但会带来三个持续的成本：

1. **无人能一眼答出"内核里到底有几条提示词"** —— 本次梳理时我先按清单找"3 处"，
   实际 grep 出 9~10 处（含 6~7 处内联）。**范围失守是从"不知道有多少"开始的**。
2. **跨文件复制字面串** —— 测试里曾直接硬编码提示词片段
   （`tests/test_reflect_on_abort.py` 的 `REFLECT_PROMPT_HEAD` 与 `GUIDANCE_MARK`），
   提示词一改，测试的**判别条件静默失效**（仍会通过，但判的已不是原来那件事）。
3. **无版本可追溯** —— `papermate/prompts/` 有 `PROMPT_VERSION`，内核侧没有。

> **注：本包不合并 `papermate/prompts/`。** `lightcore` 是内核层、`papermate` 是业务层，
> 实测 `lightcore` **不 import `papermate`**（层方向干净）。把内核提示词搬到上层包
> = **层倒置**。两包**并列**，各自是所属层的单一来源。

**当前范围（最小集，2026-09-23）**：只收**模块级常量**这 3 条。
`core/loop.py`（收口指令 / 压缩摘要）、`memory/long_term.py`（记忆提炼）、
`rag/figure_content.py`（VLM 图表理解）里的**内联**提示词**暂不搬** ——
它们嵌在具体逻辑里，硬提出来会增加耦合，收益不抵改动面。
**新增提示词时**：与现有主题无关的，按主题新建模块（如 `planning.py`）并在 `__init__` 里
re-export；修提示词内容时**同步 bump 版本号**。
"""

from __future__ import annotations

#: 短期记忆压缩摘要（原 `lightcore/memory/short_term.py`，2026-09-23 搬运，逐字未改）
SUMMARY_PROMPT = (
    "你是记忆压缩器。请将以下对话片段压缩为一段简洁、保留关键事实与用户偏好的中文摘要，"
    "不超过 120 字，不要编造未提及的内容。"
)

#: 任务规划分解（原 `lightcore/planning/planner.py`，2026-09-23 搬运，逐字未改）
PLANNER_PROMPT = (
    "你是一个任务规划器。请将用户目标分解为不超过 5 个有序子任务。"
    "每行一个步骤，格式：<步骤描述>。如有依赖，用 [依赖步骤号] 标注。"
    "只输出步骤，不要解释。"
)

#: 执行结果反思（原 `lightcore/planning/reflector.py`，2026-09-23 搬运，逐字未改）
REFLECT_PROMPT = (
    "请评价以下任务执行结果是否满足目标。\n"
    "第一行输出 SATISFIED 或 NEEDS_IMPROVEMENT。\n"
    "若 NEEDS_IMPROVEMENT，第二行写'反思: ...'，第三行写'改进: ...'。"
)

#: ★ 由 `REFLECT_PROMPT` 契约派生的**输出标记**（供解析与测试共同引用，避免字面串复制）。
#:
#: `REFLECT_PROMPT` 要求第一行输出这两个词之一，调用方据此判定"是否满足目标"。
#: 此前**测试自己又写了一遍**这两个词，提示词一改就失去判别力 —— 故在此集中定义。
#:
#: ⚠️ 这构成**一对合约**：提示词说"输出 SATISFIED"，解析代码找 "SATISFIED"。
#: 只改一边 → 反思功能静默失效（解析永远判 satisfied / critique 为空），且不会报错。
#: `tests/test_prompt_single_source.py` 有守护断言把两者钉在一起。
REFLECT_SATISFIED = "SATISFIED"
REFLECT_NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"

#: 反思输出的两个**行首标签**（同受 `REFLECT_PROMPT` 规定，与解析代码成对）。
REFLECT_CRITIQUE_LABEL = "反思"
REFLECT_IMPROVEMENT_LABEL = "改进"

#: 收口引导的**分隔标记**（取自 `core/loop.py` 拼给模型的指令）。
#:
#: ⚠️ 该指令本身仍是**内联**的（未搬入本包，见模块 docstring 的范围说明）；
#: 这里只把**测试需要引用的那一小段标记**集中出来，消除跨文件字面串复制。
GUIDANCE_MARK = "上一版回答的不足"

__all__ = [
    "GUIDANCE_MARK",
    "PLANNER_PROMPT",
    "REFLECT_CRITIQUE_LABEL",
    "REFLECT_IMPROVEMENT_LABEL",
    "REFLECT_NEEDS_IMPROVEMENT",
    "REFLECT_PROMPT",
    "REFLECT_SATISFIED",
    "SUMMARY_PROMPT",
]
