"""PaperMate · find_code 技能（P1 找代码，docs/统一方案 阶段C）

命令：/找代码 /find_code
编排：从关联文档分块确定性抽取代码相关线索（算法/方法、公式、伪代码、框架库、仓库、复杂度），
      配真实模型时由 LLM 生成「代码定位与实现指南」（核心模块映射 + 三态：论文描述/需查 repo/需自行实现）；
      FakeLLM（无 key）回退到确定性分组输出；调用异常也回退。绝不臆造代码、不编造仓库链接。
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

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("算法与方法", re.compile(r"algorithm|算法|method|方法|网络结构|架构|模型|module|模块", re.I)),
    ("公式与方程", re.compile(r"[=∑∫∮√±×÷αβγδλμθπσω≈≠≤≥]|公式|方程|推导|损失函数|objective|loss function|regression", re.I)),
    ("伪代码", re.compile(r"algorithm\s*\d|伪代码|require\s*:|ensure\s*:|^\s*input\s*:|^\s*output\s*:|procedure|function|def |class ", re.I)),
    ("框架与库", re.compile(r"pytorch|tensorflow|jax|sklearn|scikit|numpy|pandas|opencv|hugging ?face|框架|库|API", re.I)),
    ("仓库与开源", re.compile(r"github|gitlab|arxiv|开源|code available|代码已公开|repository|repo|源码", re.I)),
    ("复杂度", re.compile(r"o\(\s*[nN]|复杂度|时间复杂度|空间复杂度|computational cost|参数量|flops", re.I)),
]
_CATEGORY_ORDER = [
    "算法与方法", "公式与方程", "伪代码", "框架与库", "仓库与开源", "复杂度", "其他配置",
]


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.doc_svc is None:
        return SkillResult(ok=False, error="文档服务未就绪")
    doc_ids, _args = resolve_doc_scope(ctx)  # 2026-09-06：@文档N 提及 → 限缩到该篇
    if not doc_ids:
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /找代码。")

    items = await gather_chunks(ctx.doc_svc, doc_ids)
    if not items:
        return SkillResult(
            ok=True,
            content="## 代码定位与实现指南\n\n未在关联文档中检测到可分块文本，请确认已成功上传并解析。",
            citations=[],
        )

    grouped, flat = collect_candidates(items, _CATEGORY_PATTERNS, discipline=ctx.discipline)
    citations = citations_from(flat)

    if is_real_llm(ctx.llm) and flat:
        rows = build_candidate_rows(flat, limit=40, max_len=240)
        sys_p = (
            "你是论文代码实现助手。基于下方【从论文抽取出的代码相关线索（已标注类别与页码）】，"
            "生成一份「代码定位与实现指南」。\n要求：\n"
            "1. 按「核心模块映射 / 关键算法与公式 / 框架与环境 / 官方资源定位 / 三态清单」组织（Markdown）；\n"
            "2. 核心模块映射：把论文组件映射到建议的实现模块/文件（如 data/、models/、train.py）；\n"
            "3. 三态清单：每项标 论文描述 / 需查官方 repo / 需自行实现；\n"
            "4. 官方资源定位：建议开启『联网』通过大模型原生联网搜索检索标题 + GitHub 获取官方实现；\n"
            "5. 严格基于候选，不编造代码；使用中文，简洁。"
        )
        user_p = "【代码相关线索】\n" + rows + "\n\n请输出代码定位与实现指南："
        # 统一合成+联网入口：llm 模式走模型原生搜索、tools 模式走 enrich_external，均优雅降级
        content = await synthesize_with_net(ctx, sys_p, user_p, temperature=0.3)
        if content:
            return SkillResult(
                ok=True,
                content="## 代码定位与实现指南\n\n" + content,
                citations=citations,
            )

    lines = [
        "## 代码定位与实现指南（确定性回退）",
        "\n本技能从关联文档分块中抽取代码相关线索（算法/方法、公式、伪代码、框架库、仓库、复杂度），"
        "并标注页码。以下内容均来自原文；建议结合原文上下文与官方仓库人工核实。",
        render_grouped_section(
            grouped, _CATEGORY_ORDER,
            header="\n### 候选代码相关线索",
            note=(
                "\n> 提示：以上为从原文抽取的代码相关线索。若需 LLM 生成的「核心模块映射 + 三态实现清单」"
                "（论文描述 / 需查官方 repo / 需自行实现），可在「设置」配置并测试真实模型后重试。"
            ),
        ),
        (
            "\n### 下一步建议\n"
            "1. **定位官方资源**：让 Agent 调用 `arxiv_search` / `web_search` 检索论文标题 + GitHub，获取官方实现。\n"
            "2. **模块映射**：按「数据 → 模型 → 训练 → 评估」拆分论文方法到对应模块文件。\n"
            "3. **核对依赖**：从 README / 实验章节提取框架版本与关键包版本。\n"
            "4. **逐步落地**：优先实现论文明确给出的算法/公式，未明确的标注「需查 repo / 需自行实现」。"
        ),
    ]
    return SkillResult(ok=True, content="\n".join(lines), citations=citations)


SPEC = SkillSpec(
    name="find_code",
    commands=["/找代码", "/find_code"],
    description="从关联文档抽取代码相关线索（算法/公式/框架/仓库）并生成实现指南",
    handler=run,
    feature="find_code",
    icon="💻",
    category="研究",
)
