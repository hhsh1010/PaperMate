"""PaperMate · 技能共享工具（阶段C / P1 / P2 技能复用）

提供：从关联文档管线收集分块、按句切分、按类别正则抽取候选、统一 LLM 合成封装、
以及确定性分组渲染。所有技能遵循同一管线：

    检索/抽取（确定性） → LLM 合成（配真模型时） → 确定性回退（无 key / 调用异常） → 装配 citations

LLM 是主生成器（"基于大模型结果再加工"），确定性层既是离线回退也是始终执行的装配
（citations 溯源 / 分页码 / 按类别分组），满足「绝不臆造、Always degrade gracefully」。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import FakeLLM
from lightcore.rag.page_utils import normalize_page
from lightcore.rag.pipeline import RetrievalUnavailable
from lightcore.skills.spec import SkillContext
from papermate.services.summary_service import _truncate_at_sentence
from papermate.skills.disciplines import discipline_boost_patterns, discipline_fragment

logger = logging.getLogger(__name__)

# 以句末标点切分，保证抽取候选都是完整句子（避免 PDF 行断开产生半截句）。
# 中文句末标点直接切；英文句点仅在「前为字母/右括号、后跟空白+大写字母」时才切
# ——保护小数（0.001）、版本号（v1.2）、缩写（e.g.）不被误切。
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=[a-zA-Z\)\]]\.)(?=\s+[A-Z](?![a-z]*\.?\d))")

# ---------- @文档N 提及解析（2026-09-06，供全部技能共享） ----------
# label 口径与前端 docLabel 一致：按会话关联顺序 文档一/二/…（>10 为「文档 11」）。
_CN_IDX = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_MENTION_RE = re.compile(r"@文档\s*([一二三四五六七八九十]|\d+)")


def mention_order(doc_ids: list[int]) -> list[int]:
    """「文档N」的唯一排序口径（U3 修复 2026-09-10）：**id 降序**，与前端文档列表一致。

    前端 `list_documents` 按 id 降序返回，「文档一」= 最大 id；而后端原先按
    `ctx.doc_ids` 的**关联顺序**解析 `@文档N`——两者在「关联顺序 ≠ id 降序」时
    会指到不同文档（同一句 `@文档二` 前端高亮 A 篇、后端检索 B 篇）。
    此后所有「文档N」的解析与展示（`resolve_mention`、规划目标标签）都走本函数。
    """
    return sorted(doc_ids, reverse=True)


def resolve_mention(args: str, doc_ids: list[int]) -> tuple[int | None, str]:
    """从 args 解析 `@文档N` 提及，返回 (命中的 doc_id | None, 剥离提及后的剩余 args)。

    序号越界或无匹配 → (None, 原 args)。
    U3（2026-09-10）：N 按 `mention_order`（id 降序）解释，与前端标签同源。
    """
    m = _MENTION_RE.search(args)
    if not m:
        return None, args
    token = m.group(1)
    idx = _CN_IDX.get(token) or (int(token) if token.isdigit() else None)
    ordered = mention_order(list(doc_ids))
    if idx is None or idx < 1 or idx > len(ordered):
        return None, args
    rest = (args[:m.start()] + args[m.end():]).strip()
    return ordered[idx - 1], rest


def resolve_mention_in_scope(
    args: str, valid_ids: list[int]
) -> tuple[list[int], str, str | None]:
    """★M1（2026-10-02）：在**会话有效集合**内解析 `@文档N`（普通对话路径专用）。

    ## 为什么需要它（实测事故）

    `resolve_mention` 原先**只被技能层调用**（`deep_read` / `planning_skill`），
    而**普通对话的 `retrieve` 路径完全没有"@文档N → doc_id"这一步** ——
    `routes_search` 只过滤**结构化** `mention_doc_ids`：

    ```python
    mention_ids = [d for d in body.mention_doc_ids if d in set(_base_ids)]
    ```

    ⇒ 用户**手打**「@文档二」（不点 `@` 弹窗 ⇒ 不发 `mention_doc_ids`）
    ⇒ `mention_ids = []` ⇒ `_scope` 无限制 ⇒ **在全部关联文档里盲搜**。
    实测铁证：某会话 10 次 `retrieve` 的 args **0 次**带 `doc_ids`，
    模型直接答「未能取到你所指『文档二』的实验步骤」。

    ## 口径

    1. **以 `valid_ids` 为标签空间**（= 会话有效文档集），而不是全库 ——
       否则「文档七」会指到一篇**根本没关联**的论文，检索必然落空。
    2. **序号越界**（如关联 2 篇却写「文档五」）⇒ 返回空 + 越界标签，
       调用方**据实提示**，**绝不静默降级成"全库盲搜"**（那会让用户以为限定生效了）。
    3. 命中时**同时返回剩余文本**，调用方据此**剥离提及**，
       避免模型把「@文档二」当成检索关键词。

    返回 `(命中的 id 列表, 剥离后的文本, 越界标签或 None)`。
    """
    m = _MENTION_RE.search(args)
    if not m:
        return [], args, None
    token = m.group(1)
    idx = _CN_IDX.get(token) or (int(token) if token.isdigit() else None)
    ordered = mention_order(list(valid_ids))
    rest = (args[:m.start()] + args[m.end():]).strip()
    if idx is None or idx < 1 or idx > len(ordered):
        # 越界：**不返回任何 id**（宁可无限定 + 明确提示，也不猜）
        return [], rest, f"@文档{token}"
    return [ordered[idx - 1]], rest, None


def resolve_doc_scope(ctx: SkillContext) -> tuple[list[int], str]:
    """技能通用：@ 提及 → doc_ids 限缩、args 剥离提及；无提及 → (原关联集, 原 args)。

    优先级（2026-09-07）：ctx.mention_doc_ids（前端结构化提及，精确 id）>
    args 里的 "@文档N" 文本解析（依赖展示顺序，仅作 CLI/兜底）。
    供多文档技能统一接入（@ 提及从「只加关联」升级为「选中」）。"""
    raw_args = (ctx.args or "").strip()
    if ctx.mention_doc_ids:
        rest = _MENTION_RE.sub("", raw_args).strip()
        return list(ctx.mention_doc_ids), rest
    mentioned, rest = resolve_mention(raw_args, list(ctx.doc_ids))
    if mentioned is not None:
        return [mentioned], rest
    return list(ctx.doc_ids), raw_args


def strip_mentions(args: str) -> str:
    """剥离 args 中的 @文档N 提及 token（结构化提及路径下清理 focus 文本用）。"""
    return _MENTION_RE.sub("", args).strip()


async def gather_chunks(doc_svc, doc_ids: list[int]) -> list[dict[str, Any]]:
    """收集关联文档所有分块的 (id, page, doc_id, text)。

    2026-09-08 修复（与 F7 口径对齐）：原实现对 `get_pipeline` 的异常一律吞掉并
    `continue`，使**索引服务故障**与**文档确无内容**两种情况都表现为「候选为空」，
    20 个技能统一输出「未在关联文档中检测到相关段落」——把系统故障伪装成检索结论，
    与 `retrieve` 工具已区分 `RetrievalUnavailable` 的口径直接冲突（同一故障两种说法）。

    现统计失败数：**全部**文档均因异常取不到管线时抛 `RetrievalUnavailable`，由路由转成
    「检索故障」提示；仅部分文档失败仍降级继续（记 warning），不因单篇故障否定其余文档。
    """
    items: list[dict[str, Any]] = []
    failures = 0
    for d in doc_ids:
        try:
            p = await doc_svc.get_pipeline(d)
        except Exception as e:  # noqa: BLE001  单篇故障先计数，全失败才升级为「检索故障」
            failures += 1
            logger.warning("技能管线：文档 %s 索引加载失败：%s", d, e)
            continue
        if p is None:
            continue
        for c in p.get_chunks():
            items.append({
                "id": getattr(c, "id", None),
                "page": getattr(c, "page", None),
                "doc_id": d,
                "text": getattr(c, "text", "") or "",
            })
    if doc_ids and failures == len(doc_ids):
        raise RetrievalUnavailable(
            f"全部 {failures}/{len(doc_ids)} 篇文档的索引均加载失败",
            failed=failures, total=len(doc_ids),
        )
    return items


def split_sentences(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", s).strip() for s in _SENT_SPLIT.split(text)]


def _classify(sent: str, category_patterns: list[tuple[str, re.Pattern[str]]]) -> str | None:
    for name, pat in category_patterns:
        if pat.search(sent):
            return name
    return None


def collect_candidates(
    items: list[dict[str, Any]],
    category_patterns: list[tuple[str, re.Pattern[str]]],
    generic_pat: re.Pattern[str] | None = None,
    min_len: int = 8,
    max_per_category: int = 8,
    discipline: str = "general",
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """按类别正则从分块抽取候选句子。

    候选判定：generic_pat 命中 或 任一 category 命中；category 取首个命中类别，
    仅 generic 命中则归为 "其他配置"。返回 (分组 dict, 去重扁平列表)。
    候选 sent 保留完整句子，截断统一在展示/喂 LLM 时进行，避免半截句。

    discipline（P3 检索加权）：命中学科 boost 模式的候选在呈现中稳定优先；
    仅作软重排，不剔除任何候选；general（默认）零加权、原序返回，保证向后兼容。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    flat: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, str]] = set()
    for it in items:
        text = it["text"]
        for s in split_sentences(text):
            if len(s) < min_len:
                continue
            hit_cat = _classify(s, category_patterns)
            hit_generic = bool(generic_pat.search(s)) if generic_pat else False
            if hit_cat is None and not hit_generic:
                continue
            cat = hit_cat or "其他配置"
            key = (it["doc_id"], it["page"], s)
            if key in seen:
                continue
            seen.add(key)
            cand = {
                "sent": s,
                "page": it["page"],
                "doc_id": it["doc_id"],
                "chunk_id": it["id"],
                "category": cat,
            }
            grouped.setdefault(cat, []).append(cand)
            flat.append(cand)
    # P3 检索加权：命中学科 boost 模式的候选稳定前置（截断前应用，避免优先项被每类上限丢失）
    flat = boost_candidates(flat, discipline)
    for cat in grouped:
        grouped[cat] = boost_candidates(grouped[cat], discipline)
    for cat in grouped:
        grouped[cat] = grouped[cat][:max_per_category]
    return grouped, flat


def _compile_boost_patterns(discipline: str) -> list[re.Pattern[str]]:
    """编译学科检索加权模式为正则（IGNORECASE）；无模式返回空列表（零加权）。"""
    return [re.compile(p, re.IGNORECASE) for p in discipline_boost_patterns(discipline)]


def _discipline_boost_key(cand: dict[str, Any], pats: list[re.Pattern[str]]) -> tuple[bool, int]:
    """加权排序键：(是否命中, 命中数)；命中者前置、命中越多越前，同分保持原序（稳定）。"""
    s = cand.get("sent", "")
    hits = sum(1 for p in pats if p.search(s))
    return (hits > 0, hits)


def boost_candidates(
    flat: list[dict[str, Any]],
    discipline: str = "general",
) -> list[dict[str, Any]]:
    """按学科检索加权对候选做稳定优先排序（P3）。

    命中学科 boost 正则的候选前置；命中数越多越靠前；同分组内保持原文相对顺序（Python sorted 稳定）。
    仅作软重排，不剔除任何候选。general / 无模式：零加权，原序返回（向后兼容、既有测试零回归）。
    """
    if not flat:
        return flat
    pats = _compile_boost_patterns(discipline)
    if not pats:
        return flat
    return sorted(flat, key=lambda c: _discipline_boost_key(c, pats), reverse=True)


def build_candidate_rows(flat: list[dict[str, Any]], limit: int = 40, max_len: int = 240) -> str:
    """把候选拼成喂给 LLM 的行（带类别与页码），用于生成再加工内容。"""
    rows: list[str] = []
    for c in flat[:limit]:
        page = c.get("page")
        tag = f"第 {page} 页" if page else "页码未知"
        rows.append(f"- [{c.get('category', '其他')} | {tag}] {c['sent'][:max_len]}")
    return "\n".join(rows)


def citations_from(flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从候选扁平列表回填引用溯源（与 /综述 等技能一致的字段：id/page/text）。

    page 经 normalize_page 统一清洗：仅 ≥1 整数视为有效页码，0/负数/None/非法 → None，
    保证引用卡页码语义与 API 边界（routes_search 同样 normalize）一致，杜绝 0 当真实页码。
    """
    return [
        {
            "id": c["chunk_id"],
            "page": normalize_page(c["page"]),
            "doc_id": c.get("doc_id"),
            "text": c["sent"],
        }
        for c in flat
    ]


def render_grouped_section(
    grouped: dict[str, list[dict[str, Any]]],
    order: list[str],
    header: str = "### 候选要点",
    note: str | None = None,
    max_len: int = 400,
) -> str:
    """确定性回退渲染：按类别分组 + 每条例句截断 + 页码溯源。"""
    lines = [header]
    any_cat = False
    for cat in order:
        its = grouped.get(cat)
        if not its:
            continue
        any_cat = True
        lines.append(f"\n**{cat}**")
        for it in its:
            # 2026-09-09 修复 I6：与引用卡（`citations_from`）口径一致——原先直接用原始 page，
            # 负页码/0 会渲染出「（p.-1）」「（p.0）」这类伪页码，而同一条内容在引用卡里
            # page 已被 normalize 成 None，正文与引用卡前后矛盾。统一经 normalize_page 清洗。
            pg = normalize_page(it.get("page"))
            page_tag = f"（p.{pg}）" if pg else ""
            lines.append(f"- {_truncate_at_sentence(it['sent'], max_len)}{page_tag}")
    if not any_cat:
        lines.append("\n（未在关联文档中检测到相关段落）")
    # note 是「以上为从原文抽取…」式提示，仅在有实际内容时有意义；空分组时不冗余展示
    if note and any_cat:
        lines.append(note)
    return "\n".join(lines)


async def synthesize(llm, system: str, user: str, temperature: float = 0.3, web_search: bool = False,
                     discipline: str = "general") -> str | None:
    """调真实 LLM 合成；异常或空返回时返回 None（调用方据此回退确定性输出）。

    web_search=True 时优先走模型原生联网搜索（llm.chat_web_search，仅支持的厂商/模型有效）；
    不支持则回退普通 chat。是否真的联网由调用方依据 enable_network 与能力检测决定。
    discipline：学科视角（general 时零注入；非 general 追加学科解读框架 + 证据铁律片段）。
    """
    system = _with_discipline(system, discipline)
    try:
        if web_search and hasattr(llm, "chat_web_search"):
            resp = await llm.chat_web_search(
                [ChatMessage(role=Role.SYSTEM, content=system),
                 ChatMessage(role=Role.USER, content=user)],
                temperature=temperature,
            )
        else:
            resp = await llm.chat(
                [ChatMessage(role=Role.SYSTEM, content=system),
                 ChatMessage(role=Role.USER, content=user)],
                temperature=temperature,
            )
        return (resp.message.content or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def is_real_llm(llm) -> bool:
    """是否可调用真实大模型（非演示/占位 FakeLLM）。"""
    return not isinstance(llm, FakeLLM)


# ---------------- 联网扩展（大模型原生联网搜索） ----------------
# 仅当会话开启「允许联网」（ctx.enable_network）且为支持原生的真实模型时，
# 调用模型自带联网检索（chat_web_search）做内容扩展；否则返回 None，
# 由调用方回退确定性输出，绝不阻断主流程（满足「绝不臆造 / Always degrade gracefully」）。
def _with_discipline(system: str, discipline: str | None) -> str:
    """在 system prompt 末尾追加学科片段（general 时为空串，零注入、零回归）。"""
    frag = discipline_fragment(discipline)
    if frag:
        return f"{system}\n\n{frag}"
    return system


async def synthesize_with_net(ctx: SkillContext, system: str, user: str, *, temperature: float = 0.3,
                             discipline: str | None = None) -> str | None:
    """统一合成+联网入口：联网开启时调用大模型原生联网搜索做内容扩展（仅 Qwen / DeepSeek 支持）。

    - enable_network=True 且为真实模型且 supports_native_web_search：走 llm.chat_web_search 直出
      （模型自带检索+引用），返回文本内容。
    - 2026-09-09 修复 M1：真实模型但**未走联网路径**时（未开联网 / 模型不支持原生联网 /
      联网调用失败 / 联网返回空），回退**普通 chat 合成**——与 review / research_gap /
      experiment_design 等技能同口径。原先直接 return None，导致只调本函数的
      /复现、/找代码 在未开联网时永远只能出确定性回退，核心「三态清单」功能完全失效。
    - FakeLLM（is_real_llm=False）仍返回 None：离线/测试模式保持确定性输出，绝不阻断主流程。
    discipline：学科视角；未显式传参时回落 ctx.discipline（来自会话/全局配置）。
    """
    discipline = discipline if discipline is not None else getattr(ctx, "discipline", "general")
    if (
        getattr(ctx, "enable_network", False)
        and is_real_llm(ctx.llm)
        and getattr(ctx.llm, "supports_native_web_search", lambda: False)()
    ):
        try:
            resp = await ctx.llm.chat_web_search(
                [ChatMessage(role=Role.SYSTEM, content=_with_discipline(system, discipline)),
                 ChatMessage(role=Role.USER, content=user)],
                temperature=temperature,
            )
            content = (resp.message.content or "").strip()
            if content:
                return content
            # 联网返回空 → 继续走下方普通 chat 兜底，而非直接放弃
        except Exception:  # noqa: BLE001
            pass  # 联网失败 → 回退普通 chat（M1），不再直接放弃
    if is_real_llm(ctx.llm):
        # 传原始 system：synthesize 内部会做学科注入，避免双重注入
        return await synthesize(ctx.llm, system, user, temperature=temperature, discipline=discipline)
    return None
