"""LightCore · 技能规格（docs/统一方案 阶段C / 设计文档 §5.2）

- SkillContext：技能运行时的依赖注入容器（query / 命令参数 / 关联文档 / 服务 / LLM / feature 开关）。
- SkillResult：技能产出（ok / content / error / citations）。
- SkillSpec：一个技能的声明（name / commands 触发词 / description / handler）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

SkillHandler = Callable[["SkillContext"], Awaitable["SkillResult"]]


@dataclass
class SkillContext:
    """技能运行时上下文（由路由层注入依赖）。"""

    query: str
    args: str
    doc_ids: list[int]
    summary_svc: Any = None
    doc_svc: Any = None
    llm: Any = None
    features: dict = field(default_factory=dict)
    db: Any = None  # 可选：DB Session 工厂（如 chat_svc._Session），供 dev_eval 等读取 Trace
    enable_network: bool = False  # 会话级「允许联网」开关：开启时联网类技能走大模型原生联网搜索（工具检索方式已移除，2026-08-27）
    discipline: str = "general"  # 学科视角（general / clinical / law / finance / bioscience），控制 LLM 解读框架注入
    # C1：工作间主文献（草稿）。工作间会话由路由层从 config.draft_doc_id 注入；
    # 普通会话恒为 None。/改稿 等草稿类技能在缺省与回落路径上优先采用它（显式参数仍可覆盖）。
    draft_doc_id: int | None = None
    # 结构化 @ 提及（2026-09-07）：前端在消息里 @ 文档时的精确 doc id（已过滤为有效文档）。
    # 技能应优先用它做范围限缩/选文——比 args 里的 "@文档N" 文本解析可靠（不受展示顺序影响）。
    mention_doc_ids: list[int] = field(default_factory=list)


@dataclass
class SkillResult:
    ok: bool
    content: str = ""
    error: str | None = None
    citations: list = field(default_factory=list)


class SkillSpec:
    """一个复合技能：由命令触发，handler 编排工具/服务并返回结构化结果。

    feature：可选，该技能额外依赖的功能开关键（如 "planning" / "dev_eval"）。
    路由时除受 master 开关 features.skills 控制外，还需该键为 True 才放行
    （满足「涉及大模型 API 的模块必须有后端开关」约束）。feature=None 的技能仅受 master 控制。
    icon / category：前端命令栏（技能 palette）渲染用的展示元数据，纯展示层、
    不影响路由逻辑；新增技能时一并声明即可统一出现在对话命令栏与设置页清单中（扩展点）。
    """

    def __init__(self, name: str, commands: list[str], description: str, handler: SkillHandler,
                 feature: str | None = None, icon: str = "⚡", category: str = "通用") -> None:
        self.name = name
        self.commands = commands  # 触发命令，如 ["/综述", "/literature"]
        self.description = description
        self.handler = handler
        self.feature = feature
        self.icon = icon          # 前端 palette 图标（emoji，无资源依赖）
        self.category = category  # 前端分组/配色用分类，如 "研究" / "规划" / "评测"

    def matches(self, text: str) -> str | None:
        """若 text 以本技能某命令开头，返回命令后的参数（已 strip）；否则 None。

        ## ★★ 2026-10-03 修正：分隔符放宽为**任意空白** + **大小写不敏感**

        **原先只认「命令 + 半角空格 / 全角空格」**，实测后果（`docs/诊断-命令功能代码审查-2026-10-03.md`）：
        - `/精读\\n参数`（**换行**）⇒ **不匹配** ← 输入框是**多行**的，
          "命令后换行写参数"是最自然的写法，却**静默回落普通 Agent**（无任何提示）；
        - `/精读\\t参数`（Tab）、`/DEEP_READ 参数`（大写）⇒ 同样不匹配。

        **修法**：改判 `text[len(cmd)]` 是否 `isspace()`（含空格/Tab/换行/全角空格），
        并用 `casefold()` 做大小写不敏感比较。

        ### ⚠️ 两处刻意的实现细节（都为防误判/错位）
        1. **索引基于原文 `t` 与 `cmd` 的原始长度**：
           `t[:len(cmd)].casefold() == cmd.casefold()`，
           **不是**先 `t.casefold()` 再 `startswith` ——
           某些字符 casefold 后**长度会变**（如 `"ß".casefold() == "ss"`），
           先整体 casefold 会让 `len(cmd)` 索引**错位**、切出错参数。
           本函数里的命令都是 ASCII/中文且 casefold 等长，但**不能依赖这个前提**。
        2. **必须要求分隔符存在**（`len(t) > len(cmd)` 且下一个字符是空白）：
           ⇒ `/精读论文` **仍然不匹配**（这是**正确**的：`/精读论文` 语义歧义，
           无条件吞掉会把它误判成"命令 `/精读` + 参数 `论文`"，比"不匹配"更危险）。
        """
        t = (text or "").strip()
        if not t:
            return None
        for cmd in self.commands:
            c = cmd.casefold()
            # ① 纯命令（无参数）—— 大小写不敏感
            if t.casefold() == c:
                return ""
            # ② 命令 + 任意空白分隔 + 参数（索引走原文，避免 casefold 长度变化导致错位）
            n = len(cmd)
            if len(t) > n and t[:n].casefold() == c and t[n].isspace():
                return t[n:].strip()
        return None

    async def run(self, ctx: SkillContext) -> SkillResult:
        return await self.handler(ctx)
