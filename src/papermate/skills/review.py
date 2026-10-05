"""PaperMate · review 技能（P2 审稿，docs/统一方案 阶段C）

命令：/审稿 /review
编排：从关联文档分块确定性抽取贡献/方法/局限线索，
      配真实模型时由 LLM 生成「结构化审稿意见」（概要/贡献/实验充分性/局限/可复现性/问题/评分）；
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
    ("贡献与新颖性", re.compile(r"贡献|创新|新颖|novel|contribution|propose|提出|首次|state-of-the-art|sota", re.I)),
    ("方法与实验", re.compile(r"实验|方法|model|数据集|baseline|结果|accuracy|消融|训练|评估", re.I)),
    ("局限", re.compile(r"局限|不足|缺点|limitation|shortcoming|未能|无法|weakness", re.I)),
    ("相关工作", re.compile(r"related work|相关工作|previous|已有工作|compared to|不同于|相比之下", re.I)),
]
# 兜底：仅捕捉「结果/结论类」强信号（证明/demonstrate/达到/超越 等），避免「本文/我们」等
# 高频代词把普通句子塞进「其他配置」稀释分组质量（F6 修复）。
_GENERIC = re.compile(r".*(?:实验结果表明|结果显示|证明|demonstrate|show that|achieve|达到|超越|outperform|优于).*", re.I)
_CATEGORY_ORDER = [
    "贡献与新颖性", "方法与实验", "局限", "相关工作", "其他配置",
]


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.doc_svc is None:
        return SkillResult(ok=False, error="文档服务未就绪")
    doc_ids, _args = resolve_doc_scope(ctx)  # 2026-09-06：@文档N 提及 → 限缩到该篇
    if not doc_ids:
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /审稿。")

    items = await gather_chunks(ctx.doc_svc, doc_ids)
    if not items:
        return SkillResult(
            ok=True,
            content="## 审稿意见\n\n未在关联文档中检测到可分块文本，请确认已成功上传并解析。",
            citations=[],
        )

    grouped, flat = collect_candidates(items, _CATEGORY_PATTERNS, _GENERIC, discipline=ctx.discipline)
    citations = citations_from(flat)

    if is_real_llm(ctx.llm) and flat:
        rows = build_candidate_rows(flat, limit=40, max_len=240)
        sys_p = (
            "你是严谨的论文审稿人。基于下方【从论文抽取出的贡献/方法/局限线索（已标注页码）】，"
            "生成一份结构化审稿意见。\n要求：\n"
            "1. 按「论文概要 / 贡献与新颖性 / 方法与实验充分性 / 局限性 / 可复现性评估 / 给作者的问题 / 推荐评分」组织（Markdown）；\n"
            "2. 推荐评分从 {强接收, 接收, 弱接收, 边界, 拒稿} 中选一，并给出 1–2 句理由；"
            "可复现性评估结合论文给出的实验细节；\n"
            "3. 每条关键论断尽量标页码（p.X）；不编造；使用中文，简洁客观。"
        )
        user_p = "【论文线索】\n" + rows + "\n\n请输出审稿意见："
        content = await synthesize(ctx.llm, sys_p, user_p,  temperature=0.2, discipline=ctx.discipline)
        if content:
            return SkillResult(
                ok=True,
                content="## 审稿意见\n\n" + content,
                citations=citations,
            )

    lines = [
        "## 审稿意见（确定性回退）",
        "\n本技能从关联文档分块中抽取贡献、方法、局限等线索，并标注页码。"
        "以下内容均来自原文；建议结合原文上下文与领域标准人工核实。",
        render_grouped_section(
            grouped, _CATEGORY_ORDER,
            header="\n### 候选审稿线索",
            note=(
                "\n> 提示：以上为从原文抽取的审稿线索。若需 LLM 生成的「贡献/实验充分性/局限/可复现性/评分」"
                "结构化审稿意见，可在「设置」配置并测试真实模型后重试。"
            ),
        ),
        (
            "\n### 下一步建议\n"
            "1. **核对贡献**：回到原文确认每条贡献是否名副其实、是否有实验支撑。\n"
            "2. **评估充分性**：检查实验是否覆盖主要基线、是否做了充分消融。\n"
            "3. **可复现性**：确认超参、数据与代码是否充分披露。\n"
            "4. **形成评语**：把线索收敛为「概要 + 优点 + 缺点 + 评分 + 问题」。"
        ),
    ]
    return SkillResult(ok=True, content="\n".join(lines), citations=citations)


SPEC = SkillSpec(
    name="review",
    commands=["/审稿", "/review"],
    description="从关联文档抽取贡献/方法/局限线索并生成结构化审稿意见",
    handler=run,
    feature="review",
    icon="📝",
    category="评测",
)
