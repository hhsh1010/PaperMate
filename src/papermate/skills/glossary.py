"""PaperMate · glossary 技能（P0 术语表，docs/统一方案 阶段C）

命令：/术语表 /glossary
编排：从关联文档所有 chunk 文本抽取缩写/术语候选（正则），定位每个术语首次出现的上下文，
真实 LLM 生成术语表；fake 下退化为「候选术语 + 首次出现页 + 上下文」确定性表（绝不臆造）。
依赖：ctx.doc_svc（检索）+ ctx.llm（生成）；feature=glossary 门控（满足 LLM 模块开关硬约束）。
"""

from __future__ import annotations

import re

from lightcore.llm.base import ChatMessage, Role
from lightcore.skills.spec import SkillContext, SkillResult, SkillSpec
from papermate.skills._common import is_real_llm, resolve_doc_scope

# 缩写候选：2–6 个连续大写字母（排除常见非术语连词/机构名）
_ABBR_PAT = re.compile(r"\b[A-Z]{2,6}\b")
#: 常见非术语缩写（连词 / 机构 / **文献题录元数据**）。
# ★★ P-02（2026-10-09）：原先只有连词与机构，实测截图中术语表混进了**刊物题录**类候选：
#   ACTA / SINICA（刊名）、ISSN（国际刊号）、CN（国内刊号）、TN（中图分类）、OL、DOI、DZXB（刊名缩写）。
#   ⇒ 按"宁可少列、不列垃圾"（与该技能"严禁编造"的自述一致）扩容白名单。
_COMMON_ABBR = {
    "AND", "OR", "NOT", "THE", "FOR", "IEEE", "ACM", "USA", "UK", "ETC",
    # 文献题录（本次实测混入的那批）
    "ISSN", "ISBN", "CN", "TN", "OL", "DOI", "URL", "EID", "PII",
    "ACTA", "SINICA", "DZXB", "SPIE", "CVPR", "ICCV", "ECCV",
    "PDF", "HTML", "XML", "JSON", "API",
}
#: **占位符**（文章编号里的 `XXXX` / `XX`）：全 X 的短串必不是术语。
_PLACEHOLDER_PAT = re.compile(r"^X{2,}$")
#: **题录元数据行**特征：命中即**整行**不作候选来源。
#  ★ 为什么要按"行"判：作者拼音（JIANG / CHENG / YANG）形态上就是普通大写词，
#    孤立地看无法与真缩写区分；但它们**总是出现在题录行**（与刊名/刊号/DOI/作者标签同行或邻近）⇒
#    按行剔除既精准又不误伤正文（正文里不会同时出现这些标签）。
_META_LINE_PAT = re.compile(
    r"ISSN|ISBN|CN\s*\d|DOI|文章编号|稿件编号|中图分类号|文献标识码|"
    r"收稿日期|修回日期|基金项目|作者简介|通信作者|E-?mail|"
    r"\bVol\.|\bNo\.|第\s*\d+\s*卷|第\s*\d+\s*期|"
    r"ACTA|SINICA|学报|期刊|作者[:：]|单位[:：]",
    re.IGNORECASE,
)
# 括号定义：全称（中文/英文）(Abbreviation) 形式，如「卷积神经网络（CNN）」
_PAREN_DEF_PAT = re.compile(
    r"([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9\-]{1,20})[（(]([A-Za-z][A-Za-z0-9\- ]{1,30})[)）]"
)
# 中文专业术语：以常见技术后缀结尾的中文名词短语（4–20 字），如「旋转框遥感图像目标检测模型」
_CN_TECH_SUFFIXES = [
    "模型", "方法", "算法", "网络", "神经网络", "框架", "机制", "模块",
    "损失", "注意力", "检测", "分类", "回归", "池化", "标签", "边界框",
    "目标检测", "图像", "遥感", "特征提取", "卷积", "学习", "优化",
    "采样", "融合", "注意力机制", "特征金字塔", "锚框", "激活函数",
    "数据增强", "轻量化", "旋转框", "水平框", "数据集",
]
_CN_TERM_PAT = re.compile(
    r"([\u4e00-\u9fa5]{2,18}(?:" + "|".join(re.escape(s) for s in sorted(_CN_TECH_SUFFIXES, key=len, reverse=True)) + r"))"
)
_CN_STOPWORDS = {
    "本文提出", "本文方法", "实验结果", "结果表明", "结果显示", "综上所述",
    "因此", "所以", "由于", "但是", "然而", "此外", "同时", "随着", "因为",
    "为了", "可以", "进行", "使用", "通过", "基于", "采用", "提出", "设计",
    "实现", "完成", "取得", "达到", "具有", "存在", "能够", "需要", "应该",
    "性能", "精度", "效果", "实验", "研究", "工作", "领域", "任务", "问题",
    "技术", "系统", "应用", "过程", "部分", "方面", "情况", "水平", "不同",
}

_MAX_TERMS_FALLBACK = 25  # 确定性回退表最多展示条数
_MAX_TERMS_LLM = 30       # 喂给真实 LLM 的术语上限


def _collect_chunks(pipe) -> list[dict]:
    try:
        chunks = pipe.get_chunks()
    except Exception:  # noqa: BLE001
        return []
    return [
        {"id": getattr(c, "id", None), "page": getattr(c, "page", None), "text": getattr(c, "text", "")}
        for c in chunks
    ]


def _strip_meta_lines(text: str) -> str:
    """剔除**题录元数据行**（刊名/刊号/DOI/稿件编号/作者与单位…），其余原样保留。

    ★★ P-02（2026-10-09）：为什么按**整行**剔而不是按词过滤 ——
      作者姓名拼音（JIANG / CHENG / YANG）形态上就是普通大写词，孤立地看**无法**与真缩写区分；
      但它们总是出现在题录行（与刊名/刊号/DOI/作者标签同行）⇒ 整行剔除既精准又不误伤正文
      （正文里不会同时出现这些标签词）。
    """
    lines = (text or "").splitlines()
    return "\n".join(ln for ln in lines if not _META_LINE_PAT.search(ln))


def _extract_terms(chunk_texts: list[str]) -> list[dict]:
    """抽取候选术语：括号定义对 + 纯缩写 + 中文专业术语。返回去重列表（按出现顺序）。

    ★ P-02：扫描前先 `_strip_meta_lines()` —— 题录区的刊号/刊名缩写/作者拼音不再进入候选。
    """
    seen: set[str] = set()
    terms: list[dict] = []
    covered_fulls: set[str] = set()  # 已在括号定义中作为「全称」出现的中文短语

    # 第一轮：括号定义（中文全称 + 英文缩写）与纯英文缩写
    for t in chunk_texts:
        t = _strip_meta_lines(t)
        for m in _PAREN_DEF_PAT.finditer(t):
            full, abbr = m.group(1).strip(), m.group(2).strip()
            if len(abbr) < 2 or abbr in seen or abbr in _COMMON_ABBR:
                continue
            seen.add(abbr)
            covered_fulls.add(full)
            start = max(0, m.start() - 40)
            terms.append({"term": abbr, "full": full, "context": t[start:m.end()].replace("\n", " ").strip(), "kind": "abbr"})
        for ab in _ABBR_PAT.findall(t):
            if ab in seen or ab in _COMMON_ABBR or _PLACEHOLDER_PAT.match(ab):
                continue
            seen.add(ab)
            terms.append({"term": ab, "full": "", "context": "", "kind": "abbr"})

    # 第二轮：中文专业术语（未被括号定义覆盖的才加入，避免重复）
    for t in chunk_texts:
        t = _strip_meta_lines(t)
        for m in _CN_TERM_PAT.finditer(t):
            cn = m.group(1).strip()
            if len(cn) < 4 or cn in seen or cn in _CN_STOPWORDS:
                continue
            # 若该中文短语已作为某英文缩写的「全称」出现，则跳过
            if any(cn in f or f in cn for f in covered_fulls):
                continue
            seen.add(cn)
            start = max(0, m.start() - 30)
            terms.append({"term": cn, "full": "", "context": t[start:m.end()].replace("\n", " ").strip(), "kind": "cn"})
    return terms


async def run(ctx: SkillContext) -> SkillResult:
    if ctx.doc_svc is None:
        return SkillResult(ok=False, error="文档服务未就绪")
    doc_ids, _args = resolve_doc_scope(ctx)  # 2026-09-06：@文档N 提及 → 限缩到该篇
    if not doc_ids:
        return SkillResult(ok=False, error="请先在右栏关联至少一篇文档，再执行 /术语表。")

    all_chunks: list[dict] = []
    for d in doc_ids:
        pipe = await ctx.doc_svc.get_pipeline(d)
        if pipe is None:
            continue
        all_chunks.extend(_collect_chunks(pipe))
    if not all_chunks:
        return SkillResult(ok=False, error="关联文档尚无可解析的文本分块，请确认已成功上传并解析。")

    terms = _extract_terms([c["text"] for c in all_chunks])
    if not terms:
        return SkillResult(
            ok=True,
            content="## 术语表\n\n未在关联文档中检测到明显缩写或术语（如 `BERT（Bidirectional ...）` 形式）。"
                    "可直接向我询问某术语的含义，或换更具体的问法。",
            citations=[],
        )

    # 定位每个术语首次出现的 chunk（用于上下文/页码/引用溯源）
    first_seen: dict[str, dict] = {}
    for c in all_chunks:
        for term in terms:
            if term["term"] in (c["text"] or ""):
                first_seen.setdefault(term["term"], c)

    llm = ctx.llm
    if not is_real_llm(llm):
        # 无 key / 演示模型：确定性回退，绝不臆造释义
        lines = [
            "## 术语表（确定性回退）", "",
            "> 当前为演示模型，无法生成释义；下表列出从原文抽取的候选术语（含中文专业术语与英文缩写）及其首次出现位置，供你参考。",
            "", "| 术语 | 全称/英文 | 首次出现页 | 上下文 |", "|---|---|---|---|",
        ]
        for term in terms[:_MAX_TERMS_FALLBACK]:
            c = first_seen.get(term["term"], {})
            page = c.get("page")
            ctx_text = (term.get("context") or (c.get("text") or ""))[:80].replace("\n", " ")
            lines.append(f"| {term['term']} | {term.get('full') or '—'} | 第 {page} 页 | {ctx_text} |")
        return SkillResult(ok=True, content="\n".join(lines), citations=[])

    # 真实 LLM：喂候选术语 + 首次出现上下文，生成术语表
    rows: list[str] = []
    citations: list[dict] = []
    for term in terms[:_MAX_TERMS_LLM]:
        c = first_seen.get(term["term"], {})
        page = c.get("page")
        ctx_text = (term.get("context") or (c.get("text") or ""))[:160].replace("\n", " ")
        rows.append(
            f"- 术语：{term['term']}（全称：{term.get('full') or '未知'}）；"
            f"首次出现：第 {page} 页；上下文：{ctx_text}"
        )
        if c:
            citations.append({"id": c.get("id"), "page": page, "text": ctx_text[:160]})
    prompt = (
        "你是学术术语助手。基于下方【论文中抽出的术语及其首次出现上下文】，生成一份术语表。\n"
        "要求：\n"
        "1. **只能使用下方给出的上下文**；区分「论文自定义术语」与「领域通用术语」，"
        "但两者都只按上下文陈述；\n"
        "2. 术语既可能是英文缩写（如 YOLO），也可能是中文专业术语（如 旋转框遥感图像目标检测模型）；\n"
        "3. ★★ **禁止用先验知识补全**：上下文没给出某缩写的全称/含义时，对应列一律写 `—`"
        "（如上下文只出现 `YOLO`，就写 `—`，**不得**写“You Only Look Once”）——"
        "本题只做「按上下文整理」，**不做**知识问答；\n"
        "4. 对一词多义或需结合上下文的缩写注明『需结合上下文』；\n"
        "5. 无法从上下文确定的写『待核实』，严禁编造；\n"
        "6. 输出 Markdown 表格：| 术语 | 缩写/英文 | 中文释义 | 首次出现页 | 相关概念 |。\n\n"
        "【术语与上下文】\n" + "\n".join(rows) + "\n\n请输出术语表："
    )
    try:
        resp = await llm.chat([ChatMessage(role=Role.USER, content=prompt)], tools=None, temperature=0.3)
    except Exception as e:  # noqa: BLE001
        return SkillResult(ok=False, error=f"术语表生成失败: {e}")
    content = (resp.message.content or "").strip()
    if not content:
        return SkillResult(ok=False, error="术语表生成失败：模型返回为空。")
    return SkillResult(ok=True, content=content, citations=citations)


SPEC = SkillSpec(
    name="glossary",
    commands=["/术语表", "/glossary"],
    description="从关联文档抽取术语/缩写并生成术语表（含释义与首次出现页）",
    handler=run,
    feature="glossary",
    icon="📖",
    category="研究",
)
