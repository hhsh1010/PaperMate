"""LightCore · 技能层（docs/统一方案 阶段C / 设计文档 §5.2）

技能 = 复合工作流（编排多个原子工具 / 服务调用 + 后处理），由命令（/综述 等）触发。
与原子工具（ToolSpec）分层：工具可被 LLM function-calling 直调；技能由 SkillRouter 按命令路由，
在 Agent 循环之外（或之内）编排，产出结构化结果。
"""
