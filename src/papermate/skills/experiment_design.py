"""PaperMate · experiment_design 技能（P2 实验设计，docs/统一方案 阶段C）

命令：/实验设计 /exp
编排：从关联文档分块确定性抽取实验设置（数据集/指标/基线/协议/训练设置），
      配真实模型时由 LLM 生成「可复现实验设计方案」（数据集指标/基线/消融/设置清单/可扩展方向）；
      FakeLLM（无 key）回退到确定性分组输出；调用异常也回退。绝不臆造数值。
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
    ("数据集与基准", re.compile(r"dataset|数据集|benchmark|基准|imagenet|coco|cityscapes|数据|语料", re.I)),
    ("评价指标", re.compile(r"metric|指标|accuracy|mAP|f1|psnr|ssim|iou|误差|召回|精确率|得分|score", re.I)),
    ("基线与方法比较", re.compile(r"baseline|基线|sota|比较|对比|state-of-the-art|ablation|消融", re.I)),
    ("评估协议", re.compile(r"protocol|协议|划分|split|训练集|验证集|测试集|cross-validation|交叉验证|fold", re.I)),
    ("训练设置", re.compile(r"gpu|epoch|batch|学习率|optimizer|硬件|显存|hours|时长|训练时间|cuda", re.I)),
]
_CATEGORY_ORDER = [
    "数据集与基准", "评价指标", "基线与方法比较", "评估协议", "训练设置", "其他配置",
]


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.doc_svc is None:
        return SkillResult(ok=False, error="文档服务未就绪")
    doc_ids, _args = resolve_doc_scope(ctx)  # 2026-09-06：@文档N 提及 → 限缩到该篇
    if not doc_ids:
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /实验设计。")

    items = await gather_chunks(ctx.doc_svc, doc_ids)
    if not items:
        return SkillResult(
            ok=True,
            content="## 实验设计方案\n\n未在关联文档中检测到可分块文本，请确认已成功上传并解析。",
            citations=[],
        )

    grouped, flat = collect_candidates(items, _CATEGORY_PATTERNS, discipline=ctx.discipline)
    citations = citations_from(flat)

    if is_real_llm(ctx.llm) and flat:
        rows = build_candidate_rows(flat, limit=40, max_len=240)
        sys_p = (
            "你是实验设计助手。基于下方【从论文抽取出的实验设置（已标注类别与页码）】，"
            "生成一份可复现的实验设计方案。\n要求：\n"
            "1. 按「推荐数据集与指标（基于论文）/ 基线选择 / 消融实验建议 / 可复现设置清单 / 可扩展方向」组织（Markdown）；\n"
            "2. 推荐项尽量引用论文已有的数据集/指标/基线，并标页码；论文未给出的标「需补充」；\n"
            "3. 消融实验建议给出 2–3 个可操作的变量；使用中文，简洁，不编造数值。"
        )
        user_p = "【实验设置线索】\n" + rows + "\n\n请输出实验设计方案："
        content = await synthesize(ctx.llm, sys_p, user_p, temperature=0.3, discipline=ctx.discipline)
        if content:
            return SkillResult(
                ok=True,
                content="## 实验设计方案\n\n" + content,
                citations=citations,
            )

    lines = [
        "## 实验设计方案（确定性回退）",
        "\n本技能从关联文档分块中抽取实验设置（数据集/指标/基线/协议/训练设置），并标注页码。"
        "以下内容均来自原文；建议结合原文上下文与领域基线人工核实。",
        render_grouped_section(
            grouped, _CATEGORY_ORDER,
            header="\n### 候选实验设置",
            note=(
                "\n> 提示：以上为从原文抽取的实验设置。若需 LLM 生成的「数据集指标 + 基线 + 消融建议 + 设置清单」方案，"
                "可在「设置」配置并测试真实模型后重试。"
            ),
        ),
        (
            "\n### 下一步建议\n"
            "1. **对齐基准**：优先复用论文已用数据集与指标，保证可比性。\n"
            "2. **设计消融**：从方法模块中挑 2–3 个变量做消融，验证贡献。\n"
            "3. **固定协议**：明确划分比例、随机种子与评估脚本，提升可复现性。\n"
            "4. **扩展方向**：在论文局限基础上设计增量实验，避免重复造轮子。"
        ),
    ]
    return SkillResult(ok=True, content="\n".join(lines), citations=citations)


SPEC = SkillSpec(
    name="experiment_design",
    commands=["/实验设计", "/exp"],
    description="从关联文档抽取实验设置并生成可复现的实验设计方案",
    handler=run,
    feature="experiment_design",
    icon="🧪",
    category="规划",
)
