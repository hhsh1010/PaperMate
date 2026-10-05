"""PaperMate · reproduction_helper 技能（S8 复现检索，docs/统一方案 阶段C）

命令：/复现 /reproduce
编排：从关联文档分块确定性抽取候选超参/配置（按语义类别分组 + 页码溯源，来源=真实 chunk），
      配真实模型时由 LLM 生成「论文给出 / 待核实 / 需从代码补全」三态复现配置清单；
      FakeLLM（无 key）回退到确定性分组输出；调用异常也回退。绝不臆造数值、不编造仓库链接。
设计要点（2026-08-27 重构为 LLM 再加工 + 确定性回退，与 /术语表 /精读 一致）。
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
    synthesize_with_net,
)

# 候选配置类别（用于把抽取到的句子按语义分组，避免挤成一团无逻辑的段落）
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("优化器", re.compile(r"optimizer|优化器|adam|sgd|rmsprop|momentum", re.I)),
    ("学习率", re.compile(r"learning rate|学习率|lr\s*=|(?:^|\W)lr(?:$|\W)", re.I)),
    ("批次大小", re.compile(r"batch size|batch|批次", re.I)),
    ("训练轮数", re.compile(r"epoch|训练轮|轮数|迭代", re.I)),
    ("模型维度与层数", re.compile(r"hidden|dim|维度|层数|通道|head", re.I)),
    ("正则与丢弃", re.compile(r"dropout|正则|weight decay|weight_decay|l2|l1|衰减", re.I)),
    ("数据增强", re.compile(r"增强|augment|裁剪|翻转|抖动|归一化|normalize", re.I)),
    ("分辨率与尺寸", re.compile(r"分辨率|resolution|精度|image size|输入尺寸|输入分辨率", re.I)),
]
# 兜底：只要句子里出现下列通用信号也视为候选配置
_HP_GENERIC = re.compile(
    r".*(?:参数|步长|λ|α|β|lr|optimizer|learning rate|batch|epoch|hidden|dim|"
    r"dropout|分辨率|维度|层数|= ?[\d.]+).*", re.I,
)
# 展示顺序：先常见训练超参，再其他
_CATEGORY_ORDER = [
    "优化器", "学习率", "批次大小", "训练轮数", "模型维度与层数",
    "正则与丢弃", "数据增强", "分辨率与尺寸", "其他配置",
]
_MAX_ITEM_LEN = 400

_NEXT_STEPS = (
    "\n### 下一步建议\n"
    "1. **定位官方资源**：可在『联网』开启时通过大模型原生联网搜索检索论文标题 + GitHub，"
    "获取官方实现、数据集与预训练权重。\n"
    "2. **核对环境依赖**：从 README 或论文实验章节提取框架版本（PyTorch/TensorFlow）、CUDA 版本、关键包。\n"
    "3. **验证数值**：如需在沙箱中验证数值结果，先在「设置」开启「代码解释器」，"
    "再让 Agent 调用 `code_interpreter` 完成显存/指标换算等。\n"
    "4. **逐步复现**：按『环境 → 数据 → 训练脚本 → 评估脚本 → 结果对比』顺序执行，"
    "每一步与论文原文核对。"
)


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.doc_svc is None:
        return SkillResult(ok=False, error="文档服务未就绪")
    doc_ids, _args = resolve_doc_scope(ctx)  # 2026-09-06：@文档N 提及 → 限缩到该篇
    if not doc_ids:
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /复现。")

    items = await gather_chunks(ctx.doc_svc, doc_ids)
    if not items:
        return SkillResult(
            ok=True,
            content="## 复现检索辅助\n\n未在关联文档中检测到可分块文本，请确认已成功上传并解析。",
            citations=[],
        )

    grouped, flat = collect_candidates(items, _CATEGORY_PATTERNS, _HP_GENERIC, discipline=ctx.discipline)
    citations = citations_from(flat)

    # LLM 再加工（配真实模型）：生成三态复现配置清单
    if is_real_llm(ctx.llm) and flat:
        rows = build_candidate_rows(flat, limit=40, max_len=240)
        sys_p = (
            "你是论文复现助手。基于下方【从论文抽取出的候选超参/配置（已标注类别与页码）】，"
            "生成一份结构化复现配置清单。\n要求：\n"
            "1. 输出 Markdown 表格，列：配置项 | 论文给出值 | 状态 | 说明。"
            "状态仅取：论文给出 / 待核实 / 需从代码补全；\n"
            "2. 严格基于提供的候选，不要编造数值；论文给出的如实填值并标页码；"
            "论文未给出的标「待核实」或「需从代码补全」；\n"
            "3. 表格后给出「复现路径建议」（定位官方 repo、核对环境依赖、验证数值、逐步复现）；\n"
            "4. 使用中文，保持简洁，不编造仓库链接。"
        )
        user_p = "【候选配置】\n" + rows + "\n\n请输出复现配置清单："
        # 统一合成+联网入口：llm 模式走模型原生搜索、tools 模式走 enrich_external，均优雅降级
        content = await synthesize_with_net(ctx, sys_p, user_p, temperature=0.2)
        if content:
            llm_lines = ["## 复现检索辅助", content, _NEXT_STEPS]
            return SkillResult(ok=True, content="\n".join(llm_lines), citations=citations)

    # 确定性回退（无 key / LLM 不可用）：分组 + 页码溯源
    lines = [
        "## 复现检索辅助",
        "\n本技能从关联文档分块中抽取训练/实现相关的配置与超参数，并给出复现路径建议。"
        "以下内容均来自原文，未做臆造；建议结合原文上下文与页码人工核实。",
    ]
    note = (
        "\n> 提示：以上为从原文分块中抽取的候选句子，按类别与句子边界整理；"
        "若需更精确的结构化超参表（论文给出 / 待核实 / 需从代码补全 三态），"
        "可在「设置」配置并测试真实模型后重试。"
    )
    lines.append(
        render_grouped_section(grouped, _CATEGORY_ORDER, header="\n### 候选超参数与配置", note=note, max_len=_MAX_ITEM_LEN)
    )
    lines.append(_NEXT_STEPS)
    return SkillResult(ok=True, content="\n".join(lines), citations=citations)


SPEC = SkillSpec(
    name="reproduction_helper",
    commands=["/复现", "/reproduce"],
    description="从关联文档抽取候选超参并生成结构化复现清单（含三态：论文给出/待核实/需从代码补全）",
    handler=run,
    feature="reproduction",
    icon="🔁",
    category="研究",
)
