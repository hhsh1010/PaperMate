"""PaperMate · deep_read 技能（P0 单篇精读，docs/统一方案 阶段C）

命令：/精读 /deep_read [focus] [@文档N]
编排：按 focus 对目标文档发起多轮 retrieve → 真实 LLM 按模板生成结构化精读报告；
无 key / fake 模型下退化为「分主题摘录 + 页码标注」的确定性回退（绝不臆造）。
目标文档解析（2026-09-06）：args 含 `@文档N` 提及时优先按提及选文（label 口径与前端
docLabel 一致：按关联顺序 文档一/二/…），多篇关联下也可精读指定篇目；无提及则维持
「恰好关联 1 篇」的旧约束。依赖：ctx.doc_svc（检索）+ ctx.llm（生成）；feature=deep_read 门控。
"""

from __future__ import annotations

from lightcore.llm.base import ChatMessage, Role
from lightcore.rag.invoke import call_query
from lightcore.rag.pipeline import RetrievalUnavailable
from lightcore.skills.spec import SkillContext, SkillResult, SkillSpec
from papermate.skills._common import is_real_llm, strip_mentions
from papermate.skills._common import resolve_mention as _resolve_mention
from papermate.skills.disciplines import discipline_fragment

# 各 focus 对应的检索主题词（retrieve 按主题匹配，不用任务短语稀释相关性）
_FOCUS_QUERIES: dict[str, str] = {
    "方法": "method approach 方法 模型 架构 算法 公式",
    "实验": "experiment evaluation 实验 数据集 指标 结果 baseline 消融",
    "贡献": "contribution novelty 贡献 创新 动机 改进",
    "局限": "limitation future work 局限 不足 展望 缺陷",
}
# 默认（全部）：覆盖核心维度，按论文阅读顺序组织
_ALL_SECTIONS: list[tuple[str, str]] = [
    ("研究问题 / 背景", "research question motivation background 摘要 引言 问题 背景"),
    ("核心方法", "method approach model architecture 方法 模型 架构 算法"),
    ("实验设置", "experiment setup dataset metric baseline 实验 数据集 指标 基线"),
    ("主要结果", "results findings 结果 发现 准确率 提升"),
    ("创新点", "contribution novelty 贡献 创新 改进"),
    ("局限与展望", "limitation future work 局限 不足 展望"),
]

_TOP_K = 4  # 每个维度召回的段落数

async def _retrieve_sections(pipe, sections: list[tuple[str, str]], top_k: int = _TOP_K) -> list[dict]:
    out: list[dict] = []
    for title, q in sections:
        try:
            # 形态适配（同步 / 异步管线）+ **同步 GPU 管线下放线程池**：统一走 `call_query`
            # （N4/N9 的唯一实现）。精读在事件循环里逐维度检索，同步管线不 to_thread
            # 会阻塞整个循环（N9 实测一次工具调用 262.82s ⇒「停止」无效）。
            hits = await call_query(pipe, q, top_k=top_k)
        except RetrievalUnavailable:
            # 2026-09-09 修复 K4（F7/I1 口径）：检索**服务故障** ≠ 无相关段落——
            # 显式上抛由技能层转成可读失败，而非伪装成"未检索到段落"的误导性结论。
            raise
        except Exception:  # noqa: BLE001 单维度取证失败不拖垮整体
            hits = []
        chunks = [
            {"id": getattr(h, "id", None), "page": getattr(h, "page", None), "text": getattr(h, "text", "")}
            for h in hits
        ]
        out.append({"title": title, "query": q, "chunks": chunks})
    return out


def _fallback(sections: list[dict], focus_hint: str | None) -> str:
    lines = ["## 单篇精读（确定性回退）", ""]
    if focus_hint:
        lines.append(f"> 未识别的 focus 参数「{focus_hint}」，已按「全部」维度精读。")
        lines.append("")
    lines.append("> 当前为演示模型，无法生成解读；以下为各主题检索到的原文摘录（标注页码），供你直接参考。")
    for sec in sections:
        lines.append(f"\n### {sec['title']}")
        if not sec["chunks"]:
            lines.append("- （未检索到相关段落）")
            continue
        for c in sec["chunks"][:_TOP_K]:
            txt = (c["text"] or "").strip().replace("\n", " ")
            lines.append(f"- 第 {c['page']} 页：{txt[:200]}")
    return "\n".join(lines)


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.doc_svc is None:
        return SkillResult(ok=False, error="文档服务未就绪")
    doc_ids = ctx.doc_ids
    args = (ctx.args or "").strip()
    # 2026-09-07：目标解析优先级——ctx.mention_doc_ids（前端结构化提及，精确 id）
    # > args 里的 `@文档N` 文本解析（多篇关联也可精读指定篇目）> 单篇关联
    if ctx.mention_doc_ids:
        target = ctx.mention_doc_ids[0]
        focus_arg = strip_mentions(args)
    else:
        mentioned, rest_args = _resolve_mention(args, doc_ids)
        if mentioned is not None:
            target, focus_arg = mentioned, rest_args
        elif len(doc_ids) == 1:
            # ★ M3（2026-10-02）：此处 `doc_ids[0]` **安全** —— 前面已断言只有 1 篇，
            # 「第一篇」= 「唯一篇」，与标签顺序无关。
            # ⚠️ 但**多篇场景下的 `doc_ids[0]` 是陷阱**：标签顺序 = `mention_order`（id 降序），
            # 而 `ctx.doc_ids` 是**关联顺序**（实测 session18：`doc_ids=[14,6,16,19]`，
            # 但「文档一」= 19 ⇒ `doc_ids[0]` 是**文档三**！）。
            # 要"第一篇"请用 `mention_order(doc_ids)[0]`，别直接下标。
            target, focus_arg = doc_ids[0], args
        else:
            return SkillResult(
                ok=False,
                error="「/精读」用于单篇深度研读：请先在右栏恰好关联 1 篇文档"
                f"（当前已关联 {len(doc_ids)} 篇），或用 @文档N 指定要精读的篇目。",
            )
    pipe = await ctx.doc_svc.get_pipeline(target)
    if pipe is None:
        return SkillResult(ok=False, error="关联的文档管线未就绪，请确认该文档已完成解析。")

    focus = focus_arg
    if focus in _FOCUS_QUERIES:
        sections_def = [(focus, _FOCUS_QUERIES[focus])]
        focus_hint = None
    elif focus:
        # 未识别的 focus（如「/精读 全部」的口语）→ 退回全部维度并提示
        sections_def = _ALL_SECTIONS
        focus_hint = focus
    else:
        sections_def = _ALL_SECTIONS
        focus_hint = None

    sections = await _retrieve_sections(pipe, sections_def, top_k=_TOP_K)

    llm = ctx.llm
    if not is_real_llm(llm):
        # 无 key / 演示模型：确定性回退，绝不臆造
        return SkillResult(ok=True, content=_fallback(sections, focus_hint), citations=[])

    # 真实 LLM：拼装上下文 + 模板化提示
    context_parts: list[str] = []
    citations: list[dict] = []
    for sec in sections:
        if not sec["chunks"]:
            continue
        block = f"【{sec['title']}】"
        for c in sec["chunks"][:_TOP_K]:
            txt = (c["text"] or "").strip().replace("\n", " ")
            block += f"\n- 第 {c['page']} 页：{txt}"
            citations.append({"id": c["id"], "page": c["page"], "text": txt[:160]})
        context_parts.append(block)
    if not context_parts:
        return SkillResult(ok=False, error="该文档检索不到相关段落，可能尚未完成解析或内容为空。")
    context = "\n\n".join(context_parts)
    prompt = (
        "你是论文精读助手。基于下方【检索到的论文原文片段】（均已标注页码），"
        "为读者生成一份结构化精读报告。\n"
        "要求：\n"
        "1. 严格基于原文，区分「论文实际陈述」与「你的合理推断」，推断需注明『（推断）』；\n"
        "2. 每项关键结论标注对应页码（如『第 3 页』），用缩写引用原文要点；\n"
        "3. 原文不足以支撑处写『待核实』，严禁编造页码、数据、DOI 或参考文献；\n"
        "4. 输出 Markdown，按以下小节组织：研究问题 / 核心方法 / 实验设置 / 主要结果 / 创新点 / 局限与可改进方向。"
        + (f"\n\n（本次 focus：{focus}，可侧重该维度，但仍需覆盖报告整体结构。）" if focus and focus in _FOCUS_QUERIES else "")
        + f"\n\n【检索到的论文原文片段】\n{context}\n\n请输出精读报告："
    )
    # 学科视角：仅当非 general 时追加解读框架（向后兼容：general 零注入）
    frag = discipline_fragment(ctx.discipline)
    if frag:
        prompt = f"{prompt}\n\n{frag}"
    try:
        resp = await llm.chat([ChatMessage(role=Role.USER, content=prompt)], tools=None, temperature=0.3)
    except Exception as e:  # noqa: BLE001
        return SkillResult(ok=False, error=f"精读生成失败: {e}")
    content = (resp.message.content or "").strip()
    if not content:
        return SkillResult(ok=False, error="精读生成失败：模型返回为空。")
    return SkillResult(ok=True, content=content, citations=citations)


SPEC = SkillSpec(
    name="deep_read",
    commands=["/精读", "/deep_read"],
    description="单篇论文结构化深读（研究问题/方法/实验/结果/创新/局限）",
    handler=run,
    feature="deep_read",
    icon="🔍",
    category="研究",
)
