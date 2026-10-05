"""PaperMate · literature_review 技能（S5 文献调研，docs/统一方案 阶段C）

命令：/综述 /literature
编排：summary_svc.related_work + summary_svc.summarize → 文献综述（背景+方法+相关工作+共引）。
无 key 时走确定性回退，绝不中断（满足 Always degrade gracefully）。
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
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /综述。")
    try:
        rw = await ctx.summary_svc.related_work(doc_ids)
        sm = await ctx.summary_svc.summarize(doc_ids)
    except Exception as e:  # noqa: BLE001
        return SkillResult(ok=False, error=f"文献综述生成失败: {e}")

    lines = ["## 文献综述"]
    narrative = (sm.get("narrative") or "").strip()
    if narrative:
        # 真实 LLM 生成连贯叙述时优先使用
        lines.append("\n### 研究背景与方法概览")
        lines.append(narrative)
    else:
        lines.append("\n### 研究背景与方法概览（离线回退）")
        for sec in sm.get("sections", []):
            content = (sec.get("content") or "").strip()
            if content and content != "（未在分块中检测到相关内容）":
                lines.append(f"\n**{sec['title']}**\n\n{content}")
    lines.append("\n### 相关工作与联系")
    lines.append(rw.get("draft", "（暂无相关工作草稿）"))
    co = (rw.get("co_citation") or {}).get("pairs", [])
    if co:
        lines.append("\n**跨文档共引（基于分块关键词共现）**：")
        for p in co[:5]:
            terms = "、".join(p.get("shared_terms", [])[:5])
            lines.append(f"- 文档 {p['doc_a']} ↔ 文档 {p['doc_b']}：{terms}（重合度 {p['jaccard']}）")
    citations = [
        # 2026-09-09 修复 M3：页码统一经 normalize_page 清洗（I6 同口径），
        # 杜绝 0/负数等伪页码进入引用卡。
        {"id": c.get("chunk_id"), "page": normalize_page(c.get("page")), "text": c.get("text")}
        for c in rw.get("citations", [])
    ]
    return SkillResult(ok=True, content="\n".join(lines), citations=citations)


SPEC = SkillSpec(
    name="literature_review",
    commands=["/综述", "/literature"],
    description="生成文献综述（背景+方法+相关工作+跨文档共引）",
    handler=run,
    feature="summary",
    icon="📚",
    category="研究",
)
