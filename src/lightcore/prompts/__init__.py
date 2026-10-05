"""LightCore 提示词**单一来源**（C7-lightcore，2026-09-23）。

其它模块一律从这里导入，**不要在各自文件里再定义同名提示词或提示词里的字面标记** ——
"改一处漏一处"是静默不一致的根源，且没有任何机制能发现。

结构：`library.py` 放提示词与由提示词派生的标记；后续新增按主题分模块并在此 re-export。

⚠️ **本包不与 `papermate/prompts/` 合并**：`lightcore` 是内核层，实测**不 import `papermate`**；
把内核提示词搬进上层包会造成**层倒置**。两包并列，各自是所属层的单一来源。
"""

from .library import (
    GUIDANCE_MARK,
    PLANNER_PROMPT,
    REFLECT_CRITIQUE_LABEL,
    REFLECT_IMPROVEMENT_LABEL,
    REFLECT_NEEDS_IMPROVEMENT,
    REFLECT_PROMPT,
    REFLECT_SATISFIED,
    SUMMARY_PROMPT,
)

#: 提示词版本（改动内容时 bump）—— 让"某次回答用的是哪版策略"可回溯。
#: papermate 侧同名常量见 `papermate/prompts/system.py`；两包**各自独立版本**（不同层，不同节奏）。
PROMPT_VERSION = "2026-09-23.1"

__all__ = [
    "GUIDANCE_MARK",
    "PLANNER_PROMPT",
    "PROMPT_VERSION",
    "REFLECT_CRITIQUE_LABEL",
    "REFLECT_IMPROVEMENT_LABEL",
    "REFLECT_NEEDS_IMPROVEMENT",
    "REFLECT_PROMPT",
    "REFLECT_SATISFIED",
    "SUMMARY_PROMPT",
]
