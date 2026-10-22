"""PaperMate · 应用服务（A-02：结构化总结 + 相关工作，docs/功能增强规格说明 A-02）

两个高频应用级能力：
- summarize(doc_ids)        ：把指定文档归纳为「背景 / 方法 / 实验 / 结论 / 局限」五段式，并附引用块。
- related_work(doc_ids)     ：产出「相关工作」草稿 + 引文来源。

设计要点：
- **离线优先**：即便没有配置真实 LLM（fake），也能从分块文本做确定性抽取，保证本地可用。
- **LLM 增强**：当配置了真实模型时，用一次性 chat 生成更连贯的叙述；解析失败自动回退到确定性结果，
  绝不因为模型不可用而中断功能（符合「Always degrade gracefully」）。
- 引用来源一律基于真实分块的 doc_id / page，确保可溯源（与 ER-22 引用跳转一致）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.orm import sessionmaker

from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import FakeLLM
from lightcore.llm.router import (
    MODEL_TO_PROVIDER,
    PRESET_ENDPOINTS,
    PROVIDER_MODELS,
    ModelEndpoint,
    OAICompatibleLLM,
)
from lightcore.rag.page_utils import normalize_page

from .. import runtime_settings
from ..config import Settings
from ..models.db import Chunk, Document
from .doc_labels import doc_label
from .document_service import DocumentService


def _doc_label(index: int) -> str:
    """中文序号标签 —— ★ M3（2026-10-02）**委托 `services.doc_labels`**（单一来源）。

    此前此处**独立抄了一份** `cn` 表（与前端 `docLabel`、技能 `_doc_label` 共三份）。
    三份各自演化 ⇒ 一旦前端改样式（如「文档 11」→「文档十一」），这里会**静默**产出
    不同标签 ⇒ 技能回复里的「文档三」与面板上的「文档三」不是同一篇。
    """
    return doc_label(index)

# 段落抽取的启发式关键词（中英文），用于离线确定性归纳
_LABEL_HINTS: dict[str, list[str]] = {
    "background": ["背景", "引言", "介绍", "Related Work", "Introduction", "动机", "问题", "贡献"],
    "method": ["方法", "模型", "算法", "框架", "架构", "Method", "Approach", "提出的", "我们提出"],
    "experiments": ["实验", "评估", "数据集", "基准", "Experiment", "Evaluation", "结果", "指标", "消融"],
    "conclusions": ["结论", "总结", "发现", "Conclusion", "未来", "展望"],
    "limitations": ["局限", "限制", "不足", "未来工作", "Limitation", "Future Work", "局限性与"],
}

_SECTION_TITLES = {
    "background": "研究背景",
    "method": "方法",
    "experiments": "实验与评估",
    "conclusions": "主要结论",
    "limitations": "局限与展望",
}


# ---------- 共享：句子级截断（避免硬字符截断导致断句） ----------
#: 常见英文缩写：末词命中则其后的 "." 不视为句末（et al. / Fig. / e.g. 等，2026-08-29）
_EN_ABBREV_WORDS = {
    "al", "eg", "e.g", "ie", "i.e", "etc", "vs", "fig", "eq", "eqs", "sec",
    "no", "nos", "pp", "p", "vol", "ref", "refs", "dept", "dr", "mr", "mrs",
    "ms", "prof", "cf", "approx", "ext", "ed", "eds", "trans", "resp", "viz",
}


def _is_sentence_end(text: str, idx: int) -> bool:
    """判断 text[idx] 是否是真正的句末标点（小数点 0.001 不算、英文缩写点不算）。"""
    ch = text[idx]
    if ch in "。！？!?；;\n":
        return True
    if ch == ".":
        # 小数点：前后都是数字则不是句末
        prev_digit = idx > 0 and text[idx - 1].isdigit()
        next_digit = idx + 1 < len(text) and text[idx + 1].isdigit()
        if prev_digit and next_digit:
            return False
        # 英文缩写点：句点前的词是常见缩写（et al. / Fig. / e.g.）→ 不是句末
        m = re.search(r"([A-Za-z.]+)$", text[:idx])
        if m and m.group(1).lower() in _EN_ABBREV_WORDS:
            return False
        # 句点后（跳过空格）紧跟小写字母 → 多为缩写/行内引用，不是句末
        j = idx + 1
        while j < len(text) and text[j] in " \t":
            j += 1
        if j < len(text) and text[j].islower():
            return False
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    """按句末标点把文本切成完整句子列表（缩写点、小数点不切断）。"""
    sents: list[str] = []
    start = 0
    for i in range(len(text)):
        if _is_sentence_end(text, i):
            sents.append(text[start:i + 1])
            start = i + 1
    if start < len(text):
        sents.append(text[start:])
    return [s.strip() for s in sents if s.strip()]


def _flow_text(text: str) -> str:
    """把句中折行（PDF 行换行）折叠掉，消除『句子中间莫名断行』的观感（2026-08-29）。

    规则：非空行并入上一行（bullet 列表行、段首行除外）；CJK-CJK 直接相接，
    其余以单空格相接；空行保留为段落分隔。
    """
    if not text or "\n" not in text:
        return text
    bullet = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
    lines = text.split("\n")
    out: list[str] = []
    for i, ln in enumerate(lines):
        if i == 0:
            out.append(ln.rstrip())
            continue
        prev = out[-1] if out else ""
        if (
            ln.strip()
            and prev.strip()
            and not bullet.match(ln)
            and not prev.rstrip().endswith(("。", "！", "？", "；", ".", "!", "?", ";"))
        ):
            # 上一行未到句末 + 当前行不是新段落/新 bullet → 句中折行，合并
            sep = "" if (prev and re.match(r"[\u4e00-\u9fff，、；：]", prev[-1]) and ln.lstrip()[:1] and re.match(r"[\u4e00-\u9fff，、；：]", ln.lstrip()[0])) else " "
            out[-1] = prev + sep + ln.strip()
        else:
            out.append(ln.rstrip())
    return "\n".join(out)


def _first_sentences(text: str, max_sentences: int = 2, max_len: int = 200) -> str:
    """取前 N 个完整句子（总长受 max_len 约束），保证输出是完整表述而非半截句。"""
    text = (text or "").strip()
    if not text:
        return text
    sents = _split_sentences(text)
    out: list[str] = []
    total = 0
    for s in sents[:max_sentences]:
        if out and total + len(s) > max_len:
            break
        out.append(s)
        total += len(s)
    if not out:
        return _truncate_at_sentence(text, max_len)
    joined = ""
    for i, s in enumerate(out):
        if i == 0:
            joined = s
            continue
        sep = "" if re.search(r"[\u4e00-\u9fff。！？；…]$", joined) else " "
        joined += sep + s
    if len(joined) > max_len:
        joined = _truncate_at_sentence(joined, max_len)
    return joined


def _truncate_at_sentence(text: str, max_len: int = 600) -> str:
    """在句子边界截断，避免切断单词或语义；若找不到边界则回退到最近空白。

    2026-08-27 改进：新增中文省略号兜底，处理 chunk 内部没有句末标点的极端情况，
    并排除小数点被误判为句末标点的场景。
    """
    text = text.strip()
    if len(text) <= max_len:
        return text
    # 优先在 max_len 之前找最后一个真正句末标点
    cut = -1
    for i in range(max_len, -1, -1):
        if _is_sentence_end(text, i):
            cut = i + 1
            break
    if cut == -1:
        # 回退：找 max_len 附近最近的空白（避免切断英文单词）
        cut = text.rfind(" ", max_len - 40, max_len + 1)
    if cut == -1:
        cut = max_len
    truncated = text[:cut].strip()
    return f"{truncated}…" if cut < len(text) - 1 else truncated


def _build_llm(raw: str, settings: Settings):
    """根据模型描述符构建 LLM（FakeLLM 或 OpenAI 兼容端点）。

    逻辑与 ChatService.build_agent 一致，独立抽出便于复用；返回 (llm, provider)。
    未知 provider 或缺失 key 时退化到 FakeLLM，避免崩溃。
    """
    if raw in PRESET_ENDPOINTS:
        provider = raw
    elif raw in MODEL_TO_PROVIDER:
        provider = MODEL_TO_PROVIDER[raw]
    else:
        provider = "fake"
    if provider == "fake":
        return FakeLLM(), "fake"
    ep = PRESET_ENDPOINTS.get(provider)
    if ep is None:
        return FakeLLM(), "fake"
    pset = runtime_settings.get_state()["providers"].get(provider, {}) or {}
    base_url = pset.get("base_url") or ep.base_url
    concrete = raw if raw in PROVIDER_MODELS.get(provider, []) else (pset.get("model") or ep.default_model)
    try:
        return OAICompatibleLLM(
            ModelEndpoint(ep.name, base_url, ep.api_key_env, concrete),
            api_key=pset.get("api_key") or None,
        ), provider
    except Exception:  # noqa: BLE001  任何构造失败都退化 fake
        return FakeLLM(), "fake"


# S4 长上下文总结：优先用支持 ≥128K 上下文的模型，避免长文档被截断（docs/统一方案 §4）
_LONG_CONTEXT_MODELS = {"qwen3-max", "qwen-long", "gpt-4o"}


def _long_context_llm(settings: Settings):
    """在所有已配置 key 的厂商中，挑一个支持长上下文的模型做总结。

    无可用长上下文端点时返回 None（退回默认模型 / 确定性结果）。仅在厂商配置了真实 key 时才有意义。
    """
    for provider, ep in PRESET_ENDPOINTS.items():
        pset = runtime_settings.get_state()["providers"].get(provider, {}) or {}
        if not pset.get("api_key"):
            continue
        models = PROVIDER_MODELS.get(provider, [])
        chosen = next((m for m in models if m in _LONG_CONTEXT_MODELS), None)
        if chosen:
            return OAICompatibleLLM(
                ModelEndpoint(ep.name, pset.get("base_url") or ep.base_url, ep.api_key_env, chosen),
                api_key=pset["api_key"],
            )
    return None


class SummaryService:
    """结构化总结与相关工作（应用层，docs/功能增强规格说明 A-02）。"""

    def __init__(self, doc_svc: DocumentService, settings: Settings, SessionLocal: sessionmaker) -> None:
        self._docs = doc_svc
        self._settings = settings
        self._Session = SessionLocal

    # ---------- 数据收集 ----------
    def _gather_chunks(self, doc_ids: list[int]) -> list[dict[str, Any]]:
        """按 doc + seq 顺序返回分块 [{doc_id, filename, seq, page, text}]。"""
        rows: list[dict[str, Any]] = []
        with self._Session() as db:
            for d in doc_ids:
                doc = db.get(Document, d)
                if doc is None:
                    continue
                chunks = db.query(Chunk).filter(Chunk.document_id == d).order_by(Chunk.seq).all()
                for c in chunks:
                    rows.append({
                        "doc_id": d,
                        "filename": doc.filename,
                        "seq": c.seq,
                        "page": c.page,
                        "text": c.content,
                    })
        return rows

    # ---------- PDF 页数（供对比视图「原文对照」并排展示）----------
    def _pdf_page_count(self, doc_id: int) -> int:
        """读取文档原始 PDF 的页数（用于前端并排展示原文页）。

        非 PDF / 文件缺失 / 解析失败均返回 0（前端据此跳过该文档的原文展示）。
        """
        from pathlib import Path

        from ..models.db import Document

        try:
            if self._Session is None:
                return 0
            with self._Session() as db:
                doc = db.get(Document, doc_id)
                if doc is None or not doc.storage_path:
                    return 0
                p = Path(doc.storage_path)
            if not p.exists() or p.suffix.lower() != ".pdf":
                return 0
            import pymupdf

            with pymupdf.open(str(p)) as pdf:
                return pdf.page_count
        except Exception:  # noqa: BLE001  任何异常（无 session / 文件缺失 / 解析失败）均不阻塞对比
            return 0

    # ---------- 引文共现（基于真实分块关键词共现，近似共引）----------
    _CJK = re.compile(r"[\u4e00-\u9fff]+")
    _EN = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}")

    # PDF 文本清洗（修复 pymupdf 提取 CJK+Latin 混排时的空格异常）
    # 模式 1：字符级空格拉丁词  "d i l a t e d   c o n v o l u t i o n" → "dilated convolution"
    _SPACED_LATIN = re.compile(r"(?:(?:[a-zA-Z]\s+){3,}[a-zA-Z])(?:\s+(?:[a-zA-Z]\s+)*[a-zA-Z])*")
    # 模式 2：CJK 与 Latin 之间多余空格  "文本 word" → "文本word"（保留一个）
    _CJK_LATIN_SPACE = re.compile(r"([\u4e00-\u9fff])\s+([a-zA-Z0-9])")
    _LATIN_CJK_SPACE = re.compile(r"([a-zA-Z0-9])\s+([\u4e00-\u9fff])")
    # 模式 3：连字符后多余空格  "shape- aware" → "shape-aware"
    _HYPHEN_SPACE = re.compile(r"([a-zA-Z0-9])-\s+([a-zA-Z])")

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗 pymupdf 提取的 PDF 文本，修复显示异常。

        处理：
        1. 字符级空格拉丁词合并（CID 字体常见问题）
        2. CJK/Latin 边界多余空格去除
        3. 多重空白归一化为单空格
        4. 控制字符剔除
        """
        if not text:
            return text
        t = text
        # 1) 字符级空格拉丁：连续的 "字母+空格" 序列 → 去空格合并为单词
        def _join_spaced(m: re.Match) -> str:
            return "".join(m.group(0).split())
        t = SummaryService._SPACED_LATIN.sub(_join_spaced, t)
        # 2) CJK-Latin 边界去多余空格（保留一个即可阅读）
        t = SummaryService._CJK_LATIN_SPACE.sub(r"\1\2", t)
        t = SummaryService._LATIN_CJK_SPACE.sub(r"\1\2", t)
        # 2b) 连字符后空格合并
        t = SummaryService._HYPHEN_SPACE.sub(r"\1-\2", t)
        # 3) 多重空白归一化
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n\s*\n", "\n\n", t)
        # 4) 控制字符剔除（保留换行和常用标点区间）
        t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", t)
        # 5) 句中折行流动化：PDF 行换行并入所在句子，消除『句子中间莫名断行』（2026-08-29）
        t = _flow_text(t)
        return t.strip()

    def _doc_terms(self, texts: list[str]) -> set[str]:
        """从一组分块文本抽出高频特征词（CJK 2–4 字片段 + ASCII 词），用于跨文档共现比较。"""
        freq: dict[str, int] = {}
        for t in texts:
            for run in self._CJK.findall(t):
                for n in (2, 3, 4):
                    for i in range(len(run) - n + 1):
                        freq[run[i:i + n]] = freq.get(run[i:i + n], 0) + 1
            for w in self._EN.findall(t):
                freq[w.lower()] = freq.get(w.lower(), 0) + 1
        top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:20]
        return {w for w, _ in top}

    def _co_citation(self, chunks: list[dict[str, Any]], doc_ids: list[int]) -> dict[str, Any]:
        """跨文档共引分析：基于分块关键词 Jaccard 共现近似。

        诚实约束（docs/统一方案 风险预案）：不编造 DOI/页码，来源一律为真实 chunk 文本。
        仅关联单篇时返回空 pairs + 提示；关联 ≥2 篇时返回共现文档对（shared_terms + jaccard）。
        """
        if len(doc_ids) < 2:
            return {"pairs": [], "note": "仅关联单篇文档，共引分析需 ≥2 篇；可上传更多论文后重试。"}
        by_doc: dict[int, list[str]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(c["text"])
        doc_terms = {d: self._doc_terms(txts) for d, txts in by_doc.items()}
        pairs: list[dict[str, Any]] = []
        ds = list(doc_terms.keys())
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = ds[i], ds[j]
                sa, sb = doc_terms.get(a, set()), doc_terms.get(b, set())
                if not sa or not sb:
                    continue
                inter = sa & sb
                union = sa | sb
                score = round(len(inter) / len(union), 3) if union else 0.0
                if inter:
                    pairs.append({
                        "doc_a": a, "doc_b": b,
                        "shared_terms": sorted(inter)[:10],
                        "jaccard": score,
                    })
        pairs.sort(key=lambda p: p["jaccard"], reverse=True)
        return {
            "pairs": pairs,
            "note": "基于分块关键词共现近似共引（来源=真实 chunk 文本）；非真实参考文献图，避免编造 DOI。",
        }

    # ---------- 离线确定性归纳（2026-08-27 改进：完整句子、段落化、避免断句）----------
    _MAX_CHUNKS_PER_SECTION = 5  # 每段最多引用 5 个分块

    def _deterministic(self, chunks: list[dict[str, Any]], kind: str) -> dict[str, str]:
        """无 LLM 时，从分块文本做确定性抽取（关键词命中 + 清洗 + 句子级截断 + 兜底）。

        2026-08-27 改进：
        - 所有文本经 _clean_text() 清洗
        - 按完整句子截断，避免半截输出
        - 相关工作输出连贯段落而非碎片化 bullet。
        """
        sections: dict[str, list[str]] = {k: [] for k in _LABEL_HINTS}
        seen_hashes: dict[str, set[int]] = {k: set() for k in _LABEL_HINTS}  # 按 hash 去重

        for c in chunks:
            raw = self._clean_text(c["text"])
            if len(raw) < 20:  # 过短的分块跳过
                continue
            for key, hints in _LABEL_HINTS.items():
                if any(h in raw for h in hints):
                    # 用前 80 字符做粗粒度去重（避免同一段落被不同分块重复收录）
                    h = hash(raw[:80])
                    if h in seen_hashes[key]:
                        continue
                    seen_hashes[key].add(h)
                    if len(sections[key]) < self._MAX_CHUNKS_PER_SECTION:
                        sections[key].append(raw)

        # 兜底：背景至少首块，结论至少末块
        if not sections["background"] and chunks:
            sections["background"] = [self._clean_text(chunks[0]["text"])]
        if not sections["conclusions"] and chunks:
            sections["conclusions"] = [self._clean_text(chunks[-1]["text"])]

        if kind == "related_work":
            seeds = sections["method"] + sections["experiments"] + sections["conclusions"]
            if not seeds:
                return {"draft": "（暂无足够分块生成相关工作草稿）"}
            # 取前若干块并按句子截断，组成连贯段落
            fragments: list[str] = []
            for s in seeds[:6]:
                frag = _truncate_at_sentence(s, 280)
                if frag and frag not in fragments:
                    fragments.append(frag)
            draft = (
                "以下从文档分块中抽取出的方法、实验与结论要点可作为相关工作草稿：\n\n"
                + "\n\n".join(fragments)
                + "\n\n> 注：当前为离线确定性回退，内容直接来自原文分块，建议配置真实模型后获得更连贯的叙述。"
            )
            return {"draft": draft}

        out: dict[str, str] = {}
        for k, v in sections.items():
            if v:
                # 每个分块取一个完整句子段落，避免 bullet 碎片
                paragraphs: list[str] = []
                for s in v:
                    paragraph = _truncate_at_sentence(s, 500)
                    if paragraph and paragraph not in paragraphs:
                        paragraphs.append(paragraph)
                merged = "\n\n".join(paragraphs)
                out[k] = merged
            else:
                out[k] = "（未在分块中检测到相关内容）"
        return out

    # ---------- 有效模型解析（与 ChatService 一致：runtime_settings 优先于 config 默认）----------
    def _effective_model(self) -> str:
        """解析实际使用的模型描述符。

        必须读 runtime_settings.get_state()['default_model']（设置页持久化的那份），
        回退到 settings.default_model（仅受 PAPERMATE_DEFAULT_MODEL 环境变量影响）。
        否则在设置页配了 DeepSeek 也被忽略，技能永远走离线确定性回退。
        """
        return runtime_settings.get_state().get("default_model") or self._settings.default_model

    # ---------- LLM 增强（可选）----------
    async def _via_llm(self, chunks: list[dict[str, Any]], kind: str, lang: str = "zh") -> dict[str, Any] | None:
        llm, provider = _build_llm(self._effective_model(), self._settings)
        if provider == "fake":
            return None  # 离线环境退化到确定性结果
        # S4：总结（相对相关工作更吃上下文）优先路由到长上下文模型（如 Qwen3-Max 262K）
        if kind == "summary":
            lc = _long_context_llm(self._settings)
            if lc is not None:
                llm = lc
        context = "\n\n".join(
            f"[{c['filename']} | doc-{c['doc_id']}-c{c['seq']} | p.{c['page']}]\n{self._clean_text(c['text'])}"
            for c in chunks
        )
        if kind == "summary":
            sys = ("你是论文研读助手。请基于所给分块，按以下五段式输出（使用中文，保持简洁）：\n"
                   "【研究背景】【方法】【实验与评估】【主要结论】【局限与展望】。\n"
                   "要求：1) 每个小节写成连贯段落，不要写成半截的 bullet；2) 句与句之间逻辑衔接自然；"
                   "3) 只依据原文，不要编造；4) 引用用 '(doc-X-cY, p.Z)' 标注。")
        else:
            sys = ("你是论文研读助手。请基于所给分块，撰写一段『相关工作』(Related Work) 综述草稿，"
                   "归纳方法、实验与结论，并用 '(doc-X-cY, p.Z)' 标注每条出处；使用中文，保持简洁。\n"
                   "要求：1) 输出连贯段落，不要碎片化 bullet；2) 明确不同方法/工作之间的关系（继承、改进、差异）；"
                   "3) 只依据原文，不要编造。")
        try:
            resp = await llm.chat([
                ChatMessage(role=Role.SYSTEM, content=sys),
                ChatMessage(role=Role.USER, content=f"文档分块如下：\n{context}"),
            ], temperature=0.2)
            return resp.message.content
        except Exception as e:  # noqa: BLE001  LLM 不可用 → 回退确定性
            # ★ 必须留原因：否则用户只看到正文里的「（离线回退）」，无法判断是
            #   额度耗尽（403）/ 限流（429）/ 超时 / 网络，还是模型根本不可用。
            #   同一类"哑失败"在本项目已出现过三次（翻译字段、4xx detail、这里）。
            import logging as _lg

            _lg.getLogger(__name__).warning(
                "[summary] narrative 生成失败，回退确定性结果：%s: %s", type(e).__name__, e
            )
            return None

    # ---------- 对外接口 ----------
    async def summarize(self, doc_ids: list[int], lang: str = "zh") -> dict[str, Any]:
        chunks = self._gather_chunks(doc_ids)
        if not chunks:
            raise ValueError("未找到可总结的文档块；请先上传并完成处理的文档。")
        det = self._deterministic(chunks, "summary")

        # 引用：取每段命中块（去重），附 page 以便 ER-22 跳转
        citations: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in chunks:
            key = f"{c['doc_id']}-c{c['seq']}"
            if key in seen:
                continue
            seen.add(key)
            citations.append({"doc_id": c["doc_id"], "filename": c["filename"],
                              "chunk_id": key, "page": normalize_page(c.get("page")),
                              "text": _truncate_at_sentence(self._clean_text(c["text"]), 240)})

        narrative = await self._via_llm(chunks, "summary", lang)
        out: dict[str, Any] = {
            "doc_ids": doc_ids,
            "sections": [{"key": k, "title": _SECTION_TITLES[k], "content": det[k]} for k in _SECTION_TITLES],
            "citations": citations,
            "narrative": narrative or "",
            "model": "llm" if narrative else "deterministic",
        }
        return out

    async def related_work(self, doc_ids: list[int], lang: str = "zh") -> dict[str, Any]:
        chunks = self._gather_chunks(doc_ids)
        if not chunks:
            raise ValueError("未找到可分析的文档块；请先上传并完成处理的文档。")
        det = self._deterministic(chunks, "related_work")
        citations: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in chunks:
            key = f"{c['doc_id']}-c{c['seq']}"
            if key in seen:
                continue
            seen.add(key)
            citations.append({"doc_id": c["doc_id"], "filename": c["filename"],
                              "chunk_id": key, "page": normalize_page(c.get("page")),
                              "text": _truncate_at_sentence(self._clean_text(c["text"]), 240)})
        narrative = await self._via_llm(chunks, "related_work", lang)
        out: dict[str, Any] = {
            "doc_ids": doc_ids,
            "draft": narrative or det["draft"],
            "citations": citations,
            "co_citation": self._co_citation(chunks, doc_ids),
            "model": "llm" if narrative else "deterministic",
        }
        return out

    # ---------- S5：多篇文档对比 ----------
    async def compare(self, doc_ids: list[int], lang: str = "zh") -> dict[str, Any]:
        """多篇文档对比：逐篇五段式归纳 + 按维度并列对比表 + 共性/差异叙述 + 共引分析。

        确定性回退保证无 key 可用；LLM 增强生成对比叙述（差异/共性），解析失败自动回退。
        """
        if len(doc_ids) < 2:
            raise ValueError("对比至少需要 2 篇文档")
        # 文档友好名：按 doc_ids 顺序映射为「文档一/文档二」（与前端预览面板 docLabel 一致），
        # 后续 brief / 结果均以标签指代文档，不暴露真实文件名（源码名）。
        doc_labels = {d: _doc_label(i) for i, d in enumerate(doc_ids)}
        per_doc: list[dict[str, Any]] = []
        all_chunks: list[dict[str, Any]] = []
        for d in doc_ids:
            chunks = [c for c in self._gather_chunks([d])]
            if not chunks:
                continue
            all_chunks.extend(chunks)
            det = self._deterministic(chunks, "summary")
            per_doc.append({
                "doc_id": d,
                "label": doc_labels[d],
                "filename": chunks[0]["filename"],
                "sections": {k: det[k] for k in _SECTION_TITLES},
                "num_pages": self._pdf_page_count(d),
            })
        if not per_doc:
            raise ValueError("未找到可对比的文档块；请先上传并完成处理的文档。")

        # 对比表单元格兜底：确定性「完整句子」简述（每篇每维度 ≤2 句、≤200 字）。
        # 2026-08-29 重梳理：此前直接放 5 chunk×500 字大段并 pre-wrap 渲染，
        # chunk 内句中折行/半截句在表格里形成大量不明所以的断句。
        def _cell_brief(full: str) -> str:
            if not full or full == "（未在分块中检测到相关内容）":
                return full
            return _first_sentences(full, max_sentences=2, max_len=200)

        # LLM 增强：一次调用同时生成 ①逐维度逐篇的对比单元格（完整句概述）
        # ②【共性】【差异】【结论建议】叙述。JSON 解析失败自动全量回退确定性简述
        # （绝不中断、绝不臆造内容之外的数据，2026-08-29）。
        narrative: str | None = None
        llm_cells: dict[str, dict[str, str]] = {}
        llm, provider = _build_llm(self._effective_model(), self._settings)
        if provider != "fake":
            brief = "\n\n".join(
                f"### {d['label']}\n"
                + "\n".join(
                    f"- {k}({_SECTION_TITLES[k]})：{_first_sentences(v, max_sentences=2, max_len=200)}"
                    for k, v in d["sections"].items()
                )
                for d in per_doc
            )
            dims_desc = "、".join(f"{k}={_SECTION_TITLES[k]}" for k in _SECTION_TITLES)
            sys_prompt = (
                "你是论文研读助手。基于多篇论文的分维度归纳，为对比表生成内容。"
                f"维度编号：{dims_desc}。只输出如下结构的 JSON，不要输出 JSON 以外的任何文字：\n"
                '{"cells": {"<维度key>": {"<文档标签如文档一>": "<该篇该维度的 1~2 个完整句子概述>"}}, '
                '"narrative": "【共性】…\\n【差异】…\\n【结论建议】…"}\n'
                "文档标签必须与上方各篇「### 文档一 / 文档二 …」严格一致，不要使用文件名或编号。\n"
                "硬约束：1) 每个概述必须是语法完整、以句号结尾的 1~2 个句子，严禁半截短语或词组罗列；\n"
                "2) 句子中间不得换行；\n"
                "3) narrative 中【共性】2~3 句概述共同点、【差异】每篇 1~2 句（点出该篇主题/方法名）、"
                "【结论建议】1~2 句给出综合判断或选文建议；\n"
                f"4) 使用{('中文' if lang == 'zh' else '英文')}；5) 严禁编造输入之外的数据。"
            )
            try:
                resp = await llm.chat([
                    ChatMessage(role=Role.SYSTEM, content=sys_prompt),
                    ChatMessage(role=Role.USER, content=f"各篇归纳如下：\n{brief}"),
                ], temperature=0.2)
                raw = (resp.message.content or "").strip()
                if raw.startswith("```"):  # 剥离 Markdown 代码栅栏
                    raw = raw.strip("`").strip()
                    if raw.lower().startswith("json"):
                        raw = raw[4:].strip()
                data = json.loads(raw)
                cells_raw = data.get("cells") or {}
                if isinstance(cells_raw, dict):
                    for k, by_doc in cells_raw.items():
                        if k in _SECTION_TITLES and isinstance(by_doc, dict):
                            # by_doc 的键为「文档一/文档二」标签（与 brief 一致），直接以标签索引
                            llm_cells[k] = {
                                str(label): str(txt).strip()
                                for label, txt in by_doc.items()
                                if txt and str(txt).strip()
                            }
                nar = str(data.get("narrative") or "").strip()
                narrative = _flow_text(nar) if nar else None
            except Exception:  # noqa: BLE001  模型/JSON 失败 → 全量回退确定性
                llm_cells = {}
                narrative = None

        dimensions = []
        for k in _SECTION_TITLES:
            by_doc = llm_cells.get(k) or {}
            cells = []
            for d in per_doc:
                content = by_doc.get(d["label"]) or _cell_brief(d["sections"][k])
                # 兜底清洗：LLM 文本同样合并句中折行，消除断句观感
                content = _flow_text(content)
                cells.append({
                    "doc_id": d["doc_id"],
                    "label": d["label"],
                    # ★ O-03（2026-10-01）：维度单元格此前**只给 `label`**，而前端
                    # `CompareDimensionCell` 读的是 `filename`（同一响应的 `per_doc` 两个键都给了）
                    # ⇒ `c.filename` 恒 undefined ⇒ 对比表每行渲染成「：内容」（文档名消失）。
                    # 这里与 `per_doc` 的命名对齐（**同源**），而不是改前端。
                    "filename": d.get("filename"),
                    "content": content,
                })
            dimensions.append({"key": k, "title": _SECTION_TITLES[k], "cells": cells})

        return {
            "doc_ids": doc_ids,
            "docs": per_doc,
            "dimensions": dimensions,
            "co_citation": self._co_citation(all_chunks, doc_ids),
            "narrative": narrative or "",
            "model": "llm" if (narrative or llm_cells) else "deterministic",
        }
