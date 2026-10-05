"""PaperMate · planning 技能（S9 复杂规划，docs/统一方案 阶段D）

命令：/规划 /plan
编排：run_planning（分解 ≤5 步 + 逐子任务检索取证）→ 结构化执行报告。
复用 domain.run_planning（与 plan_executor 工具同一核心逻辑），保证工具与命令入口一致。
无 key（fake）走规则分解，零成本；真实模型走 Planner。
受 features.planning 门控（SkillRouter 校验）。
"""

from __future__ import annotations

import logging

from lightcore.rag.pipeline import RetrievalUnavailable
from lightcore.skills.spec import SkillContext, SkillResult, SkillSpec
from lightcore.tools.builtin.domain import run_planning
from papermate.services.doc_labels import doc_label
from papermate.services.summary_service import _truncate_at_sentence
from papermate.skills._common import mention_order, resolve_mention

logger = logging.getLogger(__name__)


def _doc_label(index: int) -> str:
    """中文序号标签 —— ★ M3（2026-10-02）**委托 `services.doc_labels`**（单一来源）。

    此前此处独立抄了一份 `cn` 表（三份重复实现之一）⇒ 前端改样式会**静默**失配。
    """
    return doc_label(index)


def _scope_labels(scope_ids: list[int], doc_ids: list[int]) -> list[str]:
    """scope 内每个文档的展示标签（与前端「文档N」同口径）。

    U2 修复（2026-09-09）：原实现 `_doc_label(i) for i in range(len(scope_ids))` 传的是
    循环索引——无论 @ 哪篇文档，默认规划目标永远显示「文档一」。
    U3 修复（2026-09-10）：序号口径统一为 `mention_order`（**id 降序**，与前端文档列表
    及 `resolve_mention` 的 `@文档N` 解析同源）；不在关联集时安全回退「文档一」。
    """
    ordered = mention_order(list(doc_ids))
    out: list[str] = []
    for d in scope_ids:
        try:
            out.append(_doc_label(ordered.index(d)))
        except ValueError:
            out.append(_doc_label(0))
    return out


async def run(ctx: SkillContext) -> SkillResult:
    goal = (ctx.args or ctx.query or "").strip()
    # 去掉可能的前导命令词（如用户写 "/规划 帮我..." 已剥命令，但保险起见再清一次）
    for cmd in ("/规划", "/plan"):
        if goal.lower().startswith(cmd.lower()):
            goal = goal[len(cmd):].strip()
    # 2026-09-06：args 含 @文档N 提及 → 限缩规划取证范围到该篇，并从目标文本剥离提及
    # （否则 "@文档三" 会混进规划目标提示词，LLM 收到无意义 token）
    mentioned, goal_rest = resolve_mention(goal, list(ctx.doc_ids))
    scope_ids = [mentioned] if mentioned is not None else list(ctx.doc_ids)
    if mentioned is not None:
        goal = goal_rest
    if not goal:
        # 尝试从会话文档主题生成默认规划目标
        if scope_ids and ctx.doc_svc is not None:
            # F-回复文档名友好化：用「文档一/文档二」标注（与前端预览面板 docLabel 一致），
            # 不暴露真实文件名（源码名），避免规划目标句出现 2401.xxx.pdf 等原始文件名。
            # U2（2026-09-09）：标签按文档在关联集中的真实位置映射，不再用循环索引。
            names = _scope_labels(scope_ids, list(ctx.doc_ids))
            goal = f"基于 {'、'.join(names[:3])} 制定复现与研究计划"
        else:
            return SkillResult(ok=False, error="请在命令后给出要规划的目标，例如：/规划 调研 Transformer 并复现其实验")

    # 构建跨文档检索管线（与 chat_service.build_agent 一致）
    rag = None
    if ctx.doc_svc is not None and scope_ids:
        pipelines = []
        failures = 0
        for d in scope_ids:
            try:
                p = await ctx.doc_svc.get_pipeline(d)
                if p is not None:
                    pipelines.append(p)
            except Exception as e:  # noqa: BLE001
                failures += 1
                logger.warning("规划：文档 %s 索引加载失败：%s", d, e)
        # M2 修复（2026-09-10，与 F7/I1/K4 同口径）：全部文档索引故障 ≠ 无证据——
        # 原先静默 continue 会让规划在「零证据」下照常输出步骤，把系统故障伪装成正常结果。
        if failures == len(scope_ids):
            raise RetrievalUnavailable("规划取证失败：关联文档索引均不可用，请稍后重试")
        if pipelines:
            def _query(q, top_k=3):
                merged = []
                failed = 0
                for p in pipelines:
                    try:
                        merged.extend(p.query(q, top_k=top_k))
                    except Exception:  # noqa: BLE001  单管线故障降级继续
                        failed += 1
                if failed == len(pipelines):
                    # M2：全部管线检索失败 → 抛出而非返回空证据（run_planning 会据此
                    # 标记 retrieval_unavailable，见 domain.py K4 修复）
                    raise RetrievalUnavailable("规划取证失败：关联文档检索均不可用")
                merged.sort(key=lambda r: r.score, reverse=True)
                return merged[:top_k]
            rag = type("RAG", (), {"query": staticmethod(_query)})()

    out = await run_planning(goal, ctx.llm, rag, max_steps=5)
    if not out.get("ok"):
        return SkillResult(ok=False, error=out.get("error", "规划失败"))
    steps = out.get("plan", [])
    if not steps:
        return SkillResult(ok=False, error="未能分解出有效子任务，请用更具体的语言描述目标。")

    lines = [f"## 任务规划：{goal}", ""]
    lines.append(f"针对该目标，已将其拆解为 **{len(steps)}** 个前后有序的子任务。")
    lines.append("\n### 子任务清单")
    for s in steps:
        dep = f"（依赖步骤 {s['depends_on']}）" if s.get("depends_on") else ""
        lines.append(f"{s['index']}. **{s['description']}** {dep}")

    lines.append("\n### 执行建议与取证")
    execs = out.get("execution", [])
    if execs:
        for ex in execs:
            st = ex.get("status") or "ok"
            # C6（2026-09-21）：**跳过 ≠ 没检索到**，必须分开说 ——
            # 把"因前置失败被跳过"渲染成"未检索到直接对应段落"是误导：
            # 前者是这一步根本没执行，后者是执行了但文档中没有，读者据此得出的结论完全相反。
            if st == "skipped":
                lines.append(
                    f"- **步骤 {ex['step']}：{ex['description']}**\n"
                    f"  已跳过：{ex.get('reason', '前置步骤失败')}"
                )
                continue
            if st == "failed":
                lines.append(
                    f"- **步骤 {ex['step']}：{ex['description']}**\n"
                    "  执行失败（检索服务暂时不可用），建议稍后重试该步骤。"
                )
                continue
            ev = ex.get("evidence") or []
            if ev:
                ev_txt = "；".join(
                    f"第{(e.get('page') or '?')}页：{_truncate_at_sentence(e.get('text') or '', 160)}"
                    for e in ev
                )
                lines.append(f"- **步骤 {ex['step']}：{ex['description']}**\n  关联证据：{ev_txt}")
            else:
                lines.append(f"- **步骤 {ex['step']}：{ex['description']}**\n  关联证据：当前未检索到直接对应段落，可在执行该步骤时补充关键词再检索。")
    else:
        lines.append("- 当前未检索到直接对应段落，建议按子任务清单逐步执行，每一步完成后再检索补充证据。")

    lines.append(
        "\n### 如何使用这份规划\n"
        "1. 按子任务编号顺序执行，依赖关系已标注。\n"
        "2. 对证据不足的步骤，可在聊天中继续追问或让 Agent 调用检索工具补充。\n"
        "3. 若某一步需要数值验证，确保「设置」中已开启「代码解释器」能力。"
    )
    return SkillResult(ok=True, content="\n".join(lines), citations=[])


SPEC = SkillSpec(
    name="planning",
    commands=["/规划", "/plan"],
    description="将复杂目标分解为 ≤5 步子任务并逐子任务检索取证（S9 复杂规划）",
    handler=run,
    feature="planning",
    icon="🧭",
    category="规划",
)
