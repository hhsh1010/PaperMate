"""PaperMate · research_gap 技能（P1 研究缺口，docs/统一方案 阶段C）

命令：/研究缺口 /gap
编排：从关联文档分块确定性抽取局限/未来工作/未解决问题线索，
      配真实模型时由 LLM 生成「研究缺口分析」（已识别局限 + 研究空白与机会 + 建议方向 + 风险）；
      FakeLLM（无 key）回退到确定性分组输出；调用异常也回退。绝不臆造结论。
"""

from __future__ import annotations

import re

from lightcore.skills.spec import SkillContext, SkillResult, SkillSpec
from papermate.skills._common import (
    build_candidate_rows,
    citations_from,
    collect_candidates,
    gather_chunks,
    is_real_llm,
    render_grouped_section,
    resolve_doc_scope,
    synthesize,
)

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("局限与不足", re.compile(r"局限|不足|缺点|缺陷|弱点|limitation|shortcoming|drawback|weakness|未能|无法", re.I)),
    ("未来工作", re.compile(r"未来工作|future work|下一步|后续工作|有待|will investigate|plan to|展望", re.I)),
    ("未解决/挑战", re.compile(r"未解决|尚未|挑战|challenge|open problem|remains|unsolved|gap|空白|缺乏|缺少|lack|缺失|insufficient", re.I)),
    ("对比与差异", re.compile(r"然而|但是|不过|although|however|despite|whereas|in contrast|不同于|相比之下|相比之下", re.I)),
]
# 兜底：仅捕捉「机会/拓展」类强信号（机会/opportunity/potential/值得探索/可拓展），
# 去掉「研究/方向/可能/进一步」等高频泛词，避免把普通句子塞进「其他配置」稀释分组（F6 修复）。
_GENERIC = re.compile(r".*(?:机会|opportunity|potential|值得探索|可拓展|拓展空间|promising).*", re.I)
_CATEGORY_ORDER = [
    "局限与不足", "未来工作", "未解决/挑战", "对比与差异", "其他配置",
]


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.doc_svc is None:
        return SkillResult(ok=False, error="文档服务未就绪")
    doc_ids, _args = resolve_doc_scope(ctx)  # 2026-09-06：@文档N 提及 → 限缩到该篇
    if not doc_ids:
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /研究缺口。")

    items = await gather_chunks(ctx.doc_svc, doc_ids)
    if not items:
        return SkillResult(
            ok=True,
            content="## 研究缺口分析\n\n未在关联文档中检测到可分块文本，请确认已成功上传并解析。",
            citations=[],
        )

    grouped, flat = collect_candidates(items, _CATEGORY_PATTERNS, _GENERIC, discipline=ctx.discipline)
    citations = citations_from(flat)

    if is_real_llm(ctx.llm) and flat:
        rows = build_candidate_rows(flat, limit=40, max_len=240)
        sys_p = (
            "你是科研选题助手。基于下方【从论文抽取出的局限/未来工作/未解决问题（已标注类别与页码）】，"
            "生成一份研究缺口分析。\n要求：\n"
            "1. 按「已识别的局限（带证据页码）/ 研究空白与机会点 / 建议研究方向 / 风险与可行性」组织（Markdown）；\n"
            "2. 每个缺口标注证据页码（p.X），并说明它与现有工作的关系（继承 / 改进 / 差异）；\n"
            "3. 不编造结论；论文未明确的写「待核实」；使用中文，简洁。"
        )
        user_p = "【局限/未来工作/未解决问题线索】\n" + rows + "\n\n请输出研究缺口分析："
        content = await synthesize(ctx.llm, sys_p, user_p, temperature=0.3, discipline=ctx.discipline)
        if content:
            return SkillResult(
                ok=True,
                content="## 研究缺口分析\n\n" + content,
                citations=citations,
            )

    lines = [
        "## 研究缺口分析（确定性回退）",
        "\n本技能从关联文档分块中抽取局限、未来工作、未解决问题等线索，并标注页码。"
        "以下内容均来自原文；建议结合原文上下文与领域知识人工核实。",
        render_grouped_section(
            grouped, _CATEGORY_ORDER,
            header="\n### 候选缺口线索",
            note=(
                "\n> 提示：以上为从原文抽取的缺口线索。若需 LLM 生成的「研究空白与机会点 + 建议方向 + 风险」分析，"
                "可在「设置」配置并测试真实模型后重试。"
            ),
        ),
        (
            "\n### 下一步建议\n"
            "1. **定位证据**：每条缺口都标了页码，回到原文核对上下文与强度。\n"
            "2. **交叉对比**：用 /相关工作 或 /综述 看该缺口是否已被其他工作填补。\n"
            "3. **形成选题**：把「未解决/挑战」与你的资源结合，收敛为可执行的子问题。\n"
            "4. **可行性评估**：优先选择证据充分、资源可达的方向，避免凭空臆想。"
        ),
    ]
    return SkillResult(ok=True, content="\n".join(lines), citations=citations)


SPEC = SkillSpec(
    name="research_gap",
    commands=["/研究缺口", "/gap"],
    description="从关联文档抽取局限/未来工作/未解决问题并生成研究缺口分析",
    handler=run,
    feature="research_gap",
    icon="🧩",
    category="研究",
)
