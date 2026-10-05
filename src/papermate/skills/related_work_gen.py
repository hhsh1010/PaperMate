"""PaperMate · related_work_gen 技能（S6 相关工作，docs/统一方案 阶段C）

命令：/相关工作 /related
编排：summary_svc.related_work → 相关工作草稿 + 跨文档共引表。
"""

from __future__ import annotations

from lightcore.rag.page_utils import normalize_page
from lightcore.skills.spec import SkillContext, SkillResult, SkillSpec
from papermate.skills._common import resolve_doc_scope


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.summary_svc is None:
        return SkillResult(ok=False, error="总结服务未就绪")
    doc_ids, _args = resolve_doc_scope(ctx)  # 2026-09-06：@文档N 提及 → 限缩到该篇
    if not doc_ids:
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /相关工作。")
    try:
        out = await ctx.summary_svc.related_work(doc_ids)
    except Exception as e:  # noqa: BLE001
        return SkillResult(ok=False, error=f"相关工作生成失败: {e}")

    draft = (out.get("draft") or "（暂无相关工作草稿）").strip()
    lines = ["## 相关工作综述\n", draft]
    co = (out.get("co_citation") or {}).get("pairs", [])
    if co:
        lines.append("\n**跨文档共引（基于分块关键词共现）**：")
        for p in co[:5]:
            terms = "、".join(p.get("shared_terms", [])[:5])
            lines.append(f"- 文档 {p['doc_a']} ↔ 文档 {p['doc_b']}：{terms}（重合度 {p['jaccard']}）")
    # 区分 /综述：相关工作专注「方法-实验-结论」的联系，不重复 /综述 的完整五段式
    lines.append(
        "\n> 说明：/相关工作 侧重梳理各篇在方法、实验与结论上的关联；"
        "如需完整背景-方法-实验-结论-局限概览，请使用 /综述。"
    )
    citations = [
        # 2026-09-09 修复 M3：页码统一经 normalize_page 清洗（I6 同口径）。
        {"id": c.get("chunk_id"), "page": normalize_page(c.get("page")), "text": c.get("text")}
        for c in out.get("citations", [])
    ]
    return SkillResult(ok=True, content="\n".join(lines), citations=citations)


SPEC = SkillSpec(
    name="related_work_gen",
    commands=["/相关工作", "/related"],
    description="生成相关工作综述草稿 + 跨文档共引",
    handler=run,
    feature="summary",
    icon="🔗",
    category="研究",
)
