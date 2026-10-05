"""PaperMate 提示词**单一来源**（C7，2026-09-21）。

其它模块一律从这里导入，不要在各自文件里再定义同名提示词 ——
"改一处漏一处"是静默不一致的根源，且没有任何机制能发现。

结构：`system.py` 放系统提示词与版本号，后续新增提示词按主题分模块。
"""

from .system import PROMPT_VERSION, SYSTEM_PROMPT

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT"]
