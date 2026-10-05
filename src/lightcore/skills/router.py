"""LightCore · 技能路由（docs/统一方案 阶段C）

SkillRouter：命令 → 技能。受 features.skills 开关门控（关闭时所有命令回落到普通 Agent 循环）。
"""

from __future__ import annotations

from .spec import SkillSpec


class SkillRouter:
    def __init__(self, skills: list[SkillSpec] | None = None) -> None:
        self._skills: list[SkillSpec] = list(skills or [])

    def register(self, spec: SkillSpec) -> None:
        if any(s.name == spec.name for s in self._skills):
            raise ValueError(f"技能名冲突: {spec.name}")
        self._skills.append(spec)

    def route(self, text: str, features: dict | None = None) -> SkillSpec | None:
        """返回匹配的命令技能；features.skills 关闭、或技能依赖的 feature 关闭、或无匹配返回 None。"""
        features = features or {}
        if not features.get("skills", True):
            return None
        for s in self._skills:
            if s.matches(text) is not None:
                # per-skill 门控：技能若声明依赖某 feature 键，该键须为 True 才放行
                if s.feature and not features.get(s.feature, True):
                    continue
                return s
        return None

    def parse_args(self, text: str, spec: SkillSpec) -> str:
        """解析命令后的参数文本。"""
        return spec.matches(text) or ""
