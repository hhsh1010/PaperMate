"""PaperMate · 技能注册中心（docs/统一方案 阶段C）

default_router() 构建包含全部已落地技能的 SkillRouter，供 routes_search 命令路由使用。
新增技能：在此 import 其 SPEC 并 register 即可。
"""

from __future__ import annotations

from lightcore.skills.router import SkillRouter

from .deep_read import SPEC as deep_read
from .dev_eval import SPEC as dev_eval
from .experiment_design import SPEC as experiment_design
from .find_code import SPEC as find_code
from .glossary import SPEC as glossary
from .literature_review import SPEC as literature_review
from .planning_skill import SPEC as planning
from .related_work_gen import SPEC as related_work_gen
from .reproduction_helper import SPEC as reproduction_helper
from .research_gap import SPEC as research_gap
from .review import SPEC as review
from .revision import SPEC as revision


def default_router() -> SkillRouter:
    r = SkillRouter()
    r.register(literature_review)
    r.register(related_work_gen)
    r.register(reproduction_helper)
    r.register(planning)      # S9 复杂规划（/规划，受 features.planning 门控）
    r.register(dev_eval)      # S10 开发评测（/eval，受 features.dev_eval 门控）
    r.register(deep_read)     # P0 单篇精读（/精读，受 features.deep_read 门控）
    r.register(glossary)      # P0 术语表（/术语表，受 features.glossary 门控）
    r.register(find_code)          # P1 找代码（/找代码，受 features.find_code 门控）
    r.register(research_gap)       # P1 研究缺口（/研究缺口，受 features.research_gap 门控）
    r.register(experiment_design)  # P2 实验设计（/实验设计，受 features.experiment_design 门控）
    r.register(review)             # P2 审稿（/审稿，受 features.review 门控）
    r.register(revision)           # 改稿（/改稿，受 features.revision 门控）
    return r
