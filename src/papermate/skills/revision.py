"""PaperMate · revision 技能（改稿：对照参考文献修改用户自己的论文）

命令：/改稿 /revision

定位：把会话关联的文档分成两类——**你的论文（草稿）** 与 **参考文献**。
技能先抽取参考文献的共性标准（方法 / 实验 / 写作 / 相关工作），再对照这些标准
逐节诊断草稿，产出「按草稿段落给出修改建议」+「整体结构化改稿报告」。

草稿指定（ctx.args，按优先级）：
  1. 文件名（精确匹配或唯一子串）——预览面板「改稿分析」按钮发送此形式，无歧义；
  2. 「文档N」「第N篇」「N」序号——按 ctx.doc_ids 顺序（注意：后端为 sorted 并集，
     与前端面板按勾选顺序的编号可能不同，故报告头部会显式标注实际采用的草稿文件名，
     错配立即可见、可用文件名重跑）；
  3. 缺省 → 第一篇。

管线（与同族技能一致）：
    分块抽取（确定性） → LLM 合成（配真模型时） → 确定性回退（无 key / 异常） → citations 溯源
绝不臆造：所有建议必须可溯源到给定线索或明确标注为通用写作规范。
"""

from __future__ import annotations

import re
from typing import Any

from lightcore.skills.spec import SkillContext, SkillResult, SkillSpec
from papermate.skills._common import (
    build_candidate_rows,
    citations_from,
    collect_candidates,
    gather_chunks,
    is_real_llm,
    render_grouped_section,
    synthesize,
)

# 改稿维度：既用于抽取草稿自身段落，也用于抽取参考文献的对应标准
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("研究问题与动机", re.compile(
        r"研究问题|动机|旨在|目标|挑战|不足|gap|problem|motivat|challenge|aim to|objective", re.I)),
    ("方法与创新点", re.compile(
        r"方法|模型|算法|框架|提出|创新|贡献|method|approach|model|framework|propose|novel|contribution", re.I)),
    ("实验与结果", re.compile(
        r"实验|数据集|基线|baseline|指标|accuracy|结果|消融|评估|对比|experiment|dataset|ablation|evaluat", re.I)),
    ("写作与结构", re.compile(
        r"综上|然而|因此|首先|其次|本章|本节|结构|表述|however|therefore|furthermore|section|structure", re.I)),
    ("局限与未来工作", re.compile(
        r"局限|不足|未能|未来|展望|limitation|future work|drawback|weakness", re.I)),
]
# 兜底：仅捕捉强信号句，避免把普通陈述塞进「其他」稀释分组（同 review.py 的 F6 经验）
_GENERIC = re.compile(
    r".*(?:结果显示|实验表明|证明|contribute|achieve|达到|优于|outperform|首次|state-of-the-art).*", re.I)
_CATEGORY_ORDER = [
    "研究问题与动机", "方法与创新点", "实验与结果", "写作与结构", "局限与未来工作", "其他配置",
]

_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_ORD_PAT = re.compile(r"^(?:文档|第)\s*([一二三四五六七八九十\d]+)\s*(?:篇|个|份)?$")


def _doc_filename(doc_svc: Any, doc_id: int) -> str | None:
    """取文档原始文件名（失败返回 None，不阻断主流程）。"""
    try:
        doc = doc_svc.get_document(doc_id)
    except Exception:  # noqa: BLE001
        return None
    name = getattr(doc, "filename", None)
    return str(name) if name else None


def _parse_ordinal(args: str) -> int | None:
    """从「文档二 / 第3篇 / 2」解析 1-based 序号；非序号返回 None。"""
    t = (args or "").strip()
    if not t:
        return None
    m = _ORD_PAT.match(t)
    if m:
        raw = m.group(1)
        if raw.isdigit():
            return int(raw)
        return _CN_NUM.get(raw)
    if t.isdigit():
        return int(t)
    return None


def _ws_fallback(doc_ids: list[int], default_draft: int | None) -> tuple[int, str] | None:
    """C1：工作间主文献回落目标（须在检索范围内）；非工作间/不在范围 → None（走原默认）。"""
    if default_draft is not None and default_draft in doc_ids:
        return default_draft, "按工作间主文献"
    return None


def resolve_draft(
    doc_svc: Any, doc_ids: list[int], args: str, default_draft: int | None = None
) -> tuple[int, str]:
    """解析草稿文档。(draft_id, 说明) —— 文件名优先，其次序号，最后缺省。

    C1：default_draft（工作间主文献）参与缺省与**一切回落路径**；
    显式文件名/序号参数仍可覆盖它（工作间主文献是「默认草稿」，不是强制锁定）。
    """
    t = (args or "").strip()
    # 2026-09-06：剥离 @ 前缀——用户在输入框用 @ 提及（"@文档三" / "@审阅.docx"）时，
    # 该前缀属于提及语法而非文件名的一部分，不剥离会导致文件名/序号匹配全部失败而回落。
    if t.startswith("@"):
        t = t[1:].strip()
    # ⚠️ M3（2026-10-02，**刻意不改**）：本技能的「第 N 篇」**按 ctx.doc_ids 顺序**
    #   （见本函数 docstring 的既定契约），与面板标签的 mention_order（id 降序）
    #   是**两套口径**。曾尝试统一为 mention_order，但 8 条既有测试正确地拦住了 ——
    #   那是**行为变更**而非缺陷修复，会改变用户既有习惯。
    #   ⇒ 保持原样；引用卡片走 mention_order（M3 已统一），两者互不引用。
    names: dict[int, str] = {}
    if t:
        for d in doc_ids:
            fn = _doc_filename(doc_svc, d)
            if fn:
                names[d] = fn
        low = t.lower()
        # 1) 精确匹配（忽略大小写）
        for d, fn in names.items():
            if fn.lower() == low:
                return d, f"按文件名精确匹配：{fn}"
        # 2) 唯一子串匹配
        hits = [d for d, fn in names.items() if low in fn.lower()]
        if len(hits) == 1:
            return hits[0], f"按文件名子串匹配：{names[hits[0]]}"
        if len(hits) > 1:
            # 歧义：回落（工作间优先主文献），并在说明里提示，避免静默选错
            fb = _ws_fallback(doc_ids, default_draft)
            if fb:
                return fb[0], (
                    f"参数「{t}」匹配到多篇文档（{len(hits)} 篇），已回落——{fb[1]}；"
                    "可用更完整的文件名重试。"
                )
            return doc_ids[0], (
                f"参数「{t}」匹配到多篇文档（{len(hits)} 篇），已回落第一篇"
                f"「{names.get(doc_ids[0], doc_ids[0])}」；可用更完整的文件名重试。"
            )
    # 3) 序号
    n = _parse_ordinal(t)
    if n is not None:
        if 1 <= n <= len(doc_ids):
            return doc_ids[n - 1], f"按序号匹配：第 {n} 篇"
        fb = _ws_fallback(doc_ids, default_draft)
        if fb:
            return fb[0], f"序号 {n} 超出范围（当前共 {len(doc_ids)} 篇），已回落——{fb[1]}"
        return doc_ids[0], (
            f"序号 {n} 超出范围（当前共 {len(doc_ids)} 篇），已回落第一篇；"
            "若与面板编号不一致，请改用文件名（预览面板「改稿分析」会自动带文件名）。"
        )
    # 4) 缺省：工作间主文献优先，其次第一篇
    fb = _ws_fallback(doc_ids, default_draft)
    if fb:
        if t:
            return fb[0], f"参数「{t}」未匹配到文档，已回落——{fb[1]}"
        return fb
    if t:
        return doc_ids[0], f"参数「{t}」未匹配到文档，已回落第一篇"
    return doc_ids[0], "未指定草稿，默认取第一篇"


def _sort_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 (页码, chunk id) 升序，保证喂给 LLM 的草稿片段符合阅读顺序。"""
    return sorted(items, key=lambda it: ((it.get("page") or 0), (it.get("id") or 0)))


def _header(draft_name: str, draft_pages: int, ref_count: int, note: str) -> str:
    return (
        f"> **草稿**：{draft_name}（{draft_pages} 个分块）　｜　**参考文献**：{ref_count} 篇　｜　{note}\n\n"
        "> 若草稿文件不是你的论文，请改用完整文件名重跑，例如 `/改稿 my_paper.pdf`。"
    )


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.doc_svc is None:
        return SkillResult(ok=False, error="文档服务未就绪")
    doc_ids = list(ctx.doc_ids or [])
    if not doc_ids:
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /改稿。")

    # C1：工作间主文献（ctx.draft_doc_id）作为缺省草稿；显式文件名/序号参数仍可覆盖
    draft_id, note = resolve_draft(
        ctx.doc_svc, doc_ids, ctx.args or "", default_draft=ctx.draft_doc_id
    )
    ref_ids = [d for d in doc_ids if d != draft_id]
    draft_name = _doc_filename(ctx.doc_svc, draft_id) or f"文档 #{draft_id}"

    draft_items = _sort_items(await gather_chunks(ctx.doc_svc, [draft_id]))
    if not draft_items:
        return SkillResult(
            ok=True,
            content=f"## 改稿报告\n\n{_header(draft_name, 0, len(ref_ids), note)}\n\n"
            "未在草稿文档中检测到可分块文本，请确认该文档已成功上传并解析。",
            citations=[],
        )

    ref_items = _sort_items(await gather_chunks(ctx.doc_svc, ref_ids)) if ref_ids else []

    # 确定性抽取：草稿症状 + 参考文献标准（始终执行，既是回退也是证据装配）
    draft_grouped, draft_flat = collect_candidates(
        draft_items, _CATEGORY_PATTERNS, _GENERIC, max_per_category=10, discipline=ctx.discipline)
    ref_grouped, ref_flat = (collect_candidates(
        ref_items, _CATEGORY_PATTERNS, _GENERIC, max_per_category=6, discipline=ctx.discipline)
        if ref_items else ({}, []))

    citations = citations_from((draft_flat + ref_flat)[:60])

    header = _header(draft_name, len(draft_items), len(ref_ids), note)

    # ---- LLM 主路径 ----
    if is_real_llm(ctx.llm) and draft_flat:
        draft_rows = build_candidate_rows(draft_flat, limit=50, max_len=200)
        ref_rows = build_candidate_rows(ref_flat, limit=40, max_len=200) if ref_flat else "（本次未提供参考文献）"
        sys_p = (
            "你是一位严谨的论文修改导师。请基于【草稿片段（你的论文，已标注页码）】"
            "与【参考文献标准（已标注页码）】，帮助用户修改他自己的论文。\n要求：\n"
            "1. 先输出「### 一、逐节修改建议」：按草稿自身结构（或下面片段的顺序）分段，"
            "每段包含：**原文摘录**（≤50 字并标 p.X）、**问题**、**修改建议**（具体可操作）、"
            "**参考依据**（参考文献 p.Y；无对应依据则写「通用写作规范」）；\n"
            "2. 再输出「### 二、整体问题清单」：按严重度分「必须改 / 建议改 / 可选」三档；\n"
            "3. 最后输出「### 三、优先行动清单」：3–5 条可立即执行的动作；\n"
            "4. 只依据给定材料，不编造页码与事实；参考文献未覆盖之处标注「通用写作规范」；使用中文，简洁。"
        )
        user_p = (
            f"【草稿：{draft_name}】\n{draft_rows}\n\n"
            f"【参考文献标准】\n{ref_rows}\n\n请输出改稿报告："
        )
        content = await synthesize(ctx.llm, sys_p, user_p, temperature=0.3, discipline=ctx.discipline)
        if content:
            return SkillResult(
                ok=True,
                content=f"## 改稿报告\n\n{header}\n\n{content}",
                citations=citations,
            )

    # ---- 确定性回退 ----
    lines = [
        "## 改稿报告（确定性回退）",
        "",
        header,
        "",
        "以下内容均从原文抽取：先列出草稿中值得修改的段落，再列出参考文献可对照的标准。",
        render_grouped_section(
            draft_grouped, _CATEGORY_ORDER,
            header="\n### 一、草稿中待修改的段落",
            note=None,
        ),
    ]
    if ref_grouped:
        lines.append(render_grouped_section(
            ref_grouped, _CATEGORY_ORDER,
            header="\n### 二、参考文献可对照的标准",
            note=None,
        ))
    else:
        lines.append(
            "\n### 二、参考文献可对照的标准\n\n"
            "（当前会话除草稿外没有其他关联文档，无参考文献可对照。"
            "建议关联 2–3 篇领域内范文后再执行 /改稿。）"
        )
    lines.append(
        "\n### 三、下一步建议\n"
        "1. **逐节核对**：按上面「草稿中待修改的段落」回到原文确认问题是否成立。\n"
        "2. **对照标准**：把参考文献的做法与你草稿逐条比对，补齐缺失的方法细节或实验设置。\n"
        "3. **优先修高严重度**：先解决「方法与创新点」「实验与结果」类问题，再打磨写作。\n"
        "4. 如需 LLM 生成的「逐节 原文→修改建议」完整改稿报告，"
        "可在「设置」配置并测试真实模型后重试。"
    )
    return SkillResult(ok=True, content="\n".join(lines), citations=citations)


SPEC = SkillSpec(
    name="revision",
    commands=["/改稿", "/revision"],
    description="以关联文档为参考，逐节诊断并给出你自己论文的修改建议",
    handler=run,
    feature="revision",
    icon="✍️",
    category="写作",
)
