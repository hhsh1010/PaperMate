"""PaperMate · 论文检索数据源客户端（阶段 1：**只做搜索**，不做下载/导入）

## 为什么「只做搜索」
2026-10-18 实测（本机 + 沙箱出网）：
- arXiv **搜索**接口：HTTP 200 · **2654 ms** · 结构稳定 ⇒ 完全可用；
- arXiv **PDF 下载**：5.81 MB 花了 **228 秒**（≈25 KB/s）⇒ 慢是现实。
因此本模块**只拿元数据**（标题/作者/摘要/PDF 直链），**从不下载 PDF** ——
全文下载必须交给上层的"异步任务 + 进度 + 可取消"（否则搜索页会被拖死）。

## arXiv 礼貌策略（必须遵守，否则会被封）
- 请求间隔 **≥ 3 秒**（`min_interval`，进程内按实例节流）；
- 带**可识别的 User-Agent**（附联系方式）；
- 429/503 时**指数退避**，并尊重 `Retry-After`。

## 输出契约
一律归一化为 `PaperHit`（数据源无关）—— 上层（HTTP 接口 / 前端 / Agent 工具）不感知 arXiv 的 Atom 细节。
`source_id` **去掉版本号**（`2401.00001v2` → `2401.00001`）：同一论文的多个版本在业务上是同一篇，
便于与 `documents.content_hash` 一起做去重。
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

#: arXiv 默认端点（export 子域专用于 API；不要用网页端）
ARXIV_API = "http://export.arxiv.org/api/query"

#: 可识别的 UA（arXiv 要求）
USER_AGENT = "PaperMate/0.1 (academic paper reader; contact: local-user)"

#: 礼貌间隔（秒）—— 官方建议 3s
MIN_INTERVAL_S = 3.0

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

#: arXiv id 形如 2401.00001v2（新式）或 hep-th/9901001v1（老式，**含斜杠**）
_ID_RE = re.compile(r"^(?P<id>.+?)(?P<ver>v\d+)?$")


class PaperSearchError(RuntimeError):
    """检索失败（**面向用户可读**：上层可直接把 message 展示给用户）。"""


@dataclass(frozen=True)
class PaperHit:
    """归一化后的检索结果（**不含 PDF 内容**，只有直链）。"""

    source: str
    source_id: str            # 已去版本号，如 2401.00001
    version: str              # 如 v2；无则空串
    title: str
    abstract: str
    authors: tuple[str, ...] = field(default_factory=tuple)
    published: str = ""       # YYYY-MM-DD
    updated: str = ""
    pdf_url: str = ""         # 直链：**默认不下载**
    abs_url: str = ""         # 摘要页
    categories: tuple[str, ...] = field(default_factory=tuple)
    primary_category: str = ""
    comment: str = ""         # 例如 "Accepted at CVPR 2024"
    doi: str = ""             # 部分条目没有

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "version": self.version,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "published": self.published,
            "updated": self.updated,
            "pdf_url": self.pdf_url,
            "abs_url": self.abs_url,
            "categories": list(self.categories),
            "primary_category": self.primary_category,
            "comment": self.comment,
            "doi": self.doi,
        }


#: 支持的检索范围（arXiv 字段前缀）
SCOPES: dict[str, str] = {"all": "all", "ti": "ti", "au": "au", "abs": "abs"}

#: arXiv 分类形如 cs.CL / math.OC / stat.ML（小写主类 + 点 + 大写子类）
_CATEGORY_RE = re.compile(r"^[a-z][a-z-]*\.[A-Z]{2}$")

#: 用户自己写了字段前缀或布尔运算符时，**不要再替他包引号**（会改变语义）
_HAS_FIELD_RE = re.compile(r"\b(all|ti|au|abs|cat|co|jr|rn|id)\s*:")
_HAS_OP_RE = re.compile(r"\b(AND|OR|ANDNOT)\b")


def normalize_categories(raw: object) -> tuple[str, ...]:
    """校验并规范化分类列表（去重、保序）；非法分类**直接拒绝**（不静默丢弃）。"""
    if raw is None:
        return ()
    items = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    out: list[str] = []
    for c in items:
        v = str(c).strip()
        if not v:
            continue
        if not _CATEGORY_RE.match(v):
            raise PaperSearchError(f"分类格式不正确：{v}（应形如 cs.CL）")
        if v not in out:
            out.append(v)
    return tuple(out)


def build_query(q: str, *, scope: str = "all", categories: tuple[str, ...] = ()) -> str:
    """把「关键词 + 范围 + 分类」拼成 arXiv 的 search_query。

    ★ 两个容易踩的点：
    1. **范围 + 多词**：`ti:machine learning` 里第二个词会**脱离范围**（arXiv 只把前缀绑到紧邻的
       一个词）⇒ 所以当用户没写运算符时，把整串**包成精确词组**：`ti:"machine learning"`。
       若用户自己写了 `AND`/`ti:` 等，**保持原样**（他已经表达了意图，替他加引号会改变语义）。
    2. **分类与主查询**之间用 `AND`，并用括号包住两边 —— 否则 `A OR B AND cat:C` 的结合次序不可控。
    """
    text = (q or "").strip()
    if not text:
        raise PaperSearchError("检索关键词为空")
    sc = (scope or "all").strip().lower()
    if sc not in SCOPES:
        raise PaperSearchError(f"不支持的检索范围：{scope}（可选：{'/'.join(SCOPES)}）")

    user_has_syntax = bool(_HAS_FIELD_RE.search(text) or _HAS_OP_RE.search(text))
    if sc == "all":
        core = text
    elif " " in text and not user_has_syntax:
        core = f'{sc}:"{text}"'          # 多词且无运算符 ⇒ 词组，保证范围作用于整串
    else:
        core = f"{sc}:{text}"

    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        return f"({core}) AND ({cats})"
    return core


#: ★ 中→英**短语级**术语表（不是逐字翻译）
#
# 为什么要它：arXiv 是英文库，中文关键词几乎必 0 命中（实测「深度学习」= 0 条）。
# 但**逐字翻译更糟** —— "大模型微调" 直译成 "large model fine-tuning" 不通，
# 正确说法是 "LLM fine-tuning"。所以这里只收**成句成词的高频术语**，做最长匹配替换；
# 未命中的部分保持原样，由调用方提示用户改用英文（**不替用户猜**）。
#
# 覆盖范围刻意保守：宁可不翻译，也不要产出"半通不通"的查询。
ZH_TERMS: dict[str, str] = {
    "检索增强生成": "retrieval augmented generation",
    "人类反馈强化学习": "reinforcement learning from human feedback",
    "提示词工程": "prompt engineering",
    "知识蒸馏": "knowledge distillation",
    "自监督学习": "self-supervised learning",
    "半监督学习": "semi-supervised learning",
    "无监督学习": "unsupervised learning",
    "对比学习": "contrastive learning",
    "迁移学习": "transfer learning",
    "强化学习": "reinforcement learning",
    "监督学习": "supervised learning",
    "深度学习": "deep learning",
    "机器学习": "machine learning",
    "少样本学习": "few-shot learning",
    "零样本学习": "zero-shot learning",
    "上下文学习": "in-context learning",
    "大语言模型": "large language model",
    "大型语言模型": "large language model",
    "语言模型": "language model",
    "大模型": "large language model",
    "多模态大模型": "multimodal large language model",
    "多智能体": "multi-agent",
    "智能体": "agent",
    "注意力机制": "attention mechanism",
    "多头注意力": "multi-head attention",
    "自注意力": "self-attention",
    "知识图谱": "knowledge graph",
    "思维链": "chain of thought",
    "推理链": "reasoning chain",
    "工具调用": "tool use",
    "函数调用": "function calling",
    "语义检索": "semantic retrieval",
    "向量检索": "vector retrieval",
    "稠密检索": "dense retrieval",
    "稀疏检索": "sparse retrieval",
    "混合检索": "hybrid retrieval",
    "检索增强": "retrieval augmentation",
    "向量数据库": "vector database",
    "向量化": "embedding",
    "嵌入": "embedding",
    "重排序": "reranking",
    "重排": "reranking",
    "微调": "fine-tuning",
    "提示词": "prompt",
    "提示工程": "prompt engineering",
    "多模态": "multimodal",
    "视觉问答": "visual question answering",
    "目标检测": "object detection",
    "语义分割": "semantic segmentation",
    "图像分割": "image segmentation",
    "文本分类": "text classification",
    "命名实体识别": "named entity recognition",
    "信息抽取": "information extraction",
    "机器翻译": "machine translation",
    "文本摘要": "text summarization",
    "问答系统": "question answering",
    "幻觉": "hallucination",
    "对齐": "alignment",
    "基准测试": "benchmark",
    "评估指标": "evaluation metric",
    "检索器": "retriever",
    "生成器": "generator",
}

_HAN_RE = re.compile(r"[\u4e00-\u9fa5]")


def has_han(text: str) -> bool:
    """是否还含汉字（用于提示用户"没翻干净"）。"""
    return bool(_HAN_RE.search(text or ""))


def translate_zh_query(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """把中文关键词里的**已知术语**换成英文短语。

    返回 ``(新查询, 命中的 (中文, 英文) 列表)``。

    三条原则：
    1. **最长优先**：先替换长短语（「检索增强生成」不会被「检索增强」截断）；
    2. **只替换认识的**：剩下的中文保持原样，绝不逐字硬翻（会产出不通的查询）；
    3. ★ **不隐瞒**：调用方必须把替换列表返回给用户并在界面明示 ——
       悄悄换词的后果是"用户以为搜的是中文、结果莫名其妙"。
    """
    src = (text or "").strip()
    if not src or not has_han(src):
        return src, ()
    out = src
    applied: list[tuple[str, str]] = []
    for zh, en in sorted(ZH_TERMS.items(), key=lambda kv: -len(kv[0])):
        if zh in out:
            out = out.replace(zh, f" {en} ")
            applied.append((zh, en))
    out = re.sub(r"\s+", " ", out).strip()
    return out, tuple(applied)


def split_arxiv_id(raw: str) -> tuple[str, str]:
    """从 `http://arxiv.org/abs/2401.00001v2` 或 `2401.00001v2` 拆出 (id, version)。

    ★ 去版本号是关键：同一论文 v1/v2 在业务上是同一篇。
    """
    s = raw.strip().rstrip("/")
    if "/abs/" in s:
        s = s.split("/abs/", 1)[1]
    m = _ID_RE.search(s)
    if not m:
        return s, ""
    return m.group("id"), (m.group("ver") or "")


def parse_arxiv_feed(xml_text: str) -> list[PaperHit]:
    """把 arXiv 的 Atom 响应解析为 `PaperHit` 列表。

    对缺失字段**宽容**（真实样本里 `arxiv:doi` 就常常不存在），
    但对**结构性损坏**严格（抛 `PaperSearchError`，不要静默返回空）。
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise PaperSearchError(f"检索结果解析失败（返回内容不是合法 XML）：{e}") from e

    hits: list[PaperHit] = []
    for entry in root.findall(f"{_ATOM}entry"):
        # id 取 abs 页 URL，拆出去版本号的 id 与版本
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        sid, ver = split_arxiv_id(raw_id)

        # PDF 直链：link[rel=related][type=application/pdf]；缺失则按 id 兜底拼
        pdf_url = ""
        abs_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            rel = (link.get("rel") or "").strip()
            typ = (link.get("type") or "").strip()
            href = (link.get("href") or "").strip()
            if rel == "related" and typ == "application/pdf":
                pdf_url = href
            elif rel == "alternate":
                abs_url = href
        if not pdf_url and sid:
            pdf_url = f"https://arxiv.org/pdf/{sid}{ver}"

        hits.append(
            PaperHit(
                source="arxiv",
                source_id=sid,
                version=ver,
                title=" ".join((entry.findtext(f"{_ATOM}title") or "").split()),
                abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
                authors=tuple(
                    " ".join((a.findtext(f"{_ATOM}name") or "").split())
                    for a in entry.findall(f"{_ATOM}author")
                    if (a.findtext(f"{_ATOM}name") or "").strip()
                ),
                published=(entry.findtext(f"{_ATOM}published") or "")[:10],
                updated=(entry.findtext(f"{_ATOM}updated") or "")[:10],
                pdf_url=pdf_url,
                abs_url=abs_url or raw_id,
                categories=tuple(
                    (c.get("term") or "").strip()
                    for c in entry.findall(f"{_ATOM}category")
                    if (c.get("term") or "").strip()
                ),
                # ⚠️ primary_category 的值在**属性** term 上（元素文本为空）
                primary_category=((entry.find(f"{_ARXIV}primary_category").get("term") or "").strip()
                                  if entry.find(f"{_ARXIV}primary_category") is not None else ""),
                comment=" ".join((entry.findtext(f"{_ARXIV}comment") or "").split()),
                doi=(entry.findtext(f"{_ARXIV}doi") or "").strip(),
            )
        )
    return hits


class ArxivClient:
    """arXiv 搜索客户端（**只搜索，不下载**）。

    用法：
        client = ArxivClient()
        hits = await client.search("retrieval augmented generation", limit=10)

    参数（都有合理默认，测试时可调小）：
        min_interval: 两次请求的最小间隔（秒），默认 3.0（arXiv 礼貌策略）
        max_retries:  429/5xx/网络错误的重试次数
        backoff_base: 指数退避基数（秒）
    """

    def __init__(
        self,
        *,
        timeout_s: float = 25.0,
        min_interval: float = MIN_INTERVAL_S,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout_s
        self._min_interval = max(0.0, min_interval)
        self._max_retries = max(0, max_retries)
        self._backoff_base = max(0.0, backoff_base)
        self._transport = transport          # 测试注入 MockTransport；生产为 None
        self._lock = asyncio.Lock()          # 同实例串行，保证间隔
        self._last_call = 0.0

    # ---------------- 内部 ----------------
    async def _throttle(self) -> None:
        """保证两次请求间隔 ≥ min_interval（进程内、按实例）。"""
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            wait = self._min_interval - (loop.time() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = loop.time()

    @staticmethod
    def _retry_after_s(resp: httpx.Response, default: float) -> float:
        raw = resp.headers.get("retry-after", "")
        try:
            return min(float(raw), 30.0)     # 别被超长的 Retry-After 拖死
        except (TypeError, ValueError):
            return default

    async def search(self, query: str, *, limit: int = 10, start: int = 0,
                     sort: str = "relevance", scope: str = "all",
                     categories: object = ()) -> list[PaperHit]:
        """检索 arXiv。

        - `limit`：返回条数（arXiv 单次上限 2000，业务上用 10~50 即可）
        - `sort`：`relevance` | `submitted`
        - `scope`：`all` | `ti` | `au` | `abs`（见 `build_query` 的两条拼接口径）
        - `categories`：`cs.CL` 之类的分类，多个用 `AND (cat:A OR cat:B)` 组合
        - 无结果 ⇒ **返回空列表**（不是异常 —— arXiv 对无结果同样返回 200）
        """
        q = (query or "").strip()
        if not q:
            raise PaperSearchError("检索关键词为空")
        limit = max(1, min(int(limit), 2000))
        sort_by = "relevance" if sort == "relevance" else "submitted"
        params = {
            "search_query": build_query(q, scope=scope,
                                        categories=normalize_categories(categories)),
            "start": str(max(0, int(start))),
            "max_results": str(limit),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        headers = {"User-Agent": USER_AGENT, "Accept": "application/atom+xml"}
        last_err = ""

        for attempt in range(self._max_retries + 1):
            await self._throttle()
            try:
                async with httpx.AsyncClient(timeout=self._timeout,
                                             transport=self._transport,
                                             follow_redirects=True) as client:
                    resp = await client.get(ARXIV_API, params=params, headers=headers)
            except httpx.TransportError as e:       # 超时 / 连接失败 / DNS
                last_err = f"{type(e).__name__}: {e}"
                if attempt >= self._max_retries:
                    raise PaperSearchError(f"网络请求失败（已重试 {self._max_retries} 次）：{last_err}") from e
                await asyncio.sleep(self._backoff_base * (2**attempt))
                continue

            if resp.status_code == 200:
                return parse_arxiv_feed(resp.text)

            # 429 限流 / 5xx 服务端错误 ⇒ 退避重试
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_err = f"HTTP {resp.status_code}"
                if attempt >= self._max_retries:
                    break
                wait = self._retry_after_s(resp, self._backoff_base * (2**attempt))
                logger.warning("[paper_search] arXiv 返回 %s，%.1fs 后重试（第 %d 次）",
                               resp.status_code, wait, attempt + 1)
                await asyncio.sleep(wait)
                continue

            # 其他 4xx：重试没意义，直接给出可读错误
            raise PaperSearchError(
                f"检索服务返回 HTTP {resp.status_code}（{resp.reason_phrase}）"
                + ("：请求过于频繁，请稍后再试" if resp.status_code == 429 else "")
            )

        raise PaperSearchError(
            f"检索服务暂时不可用（{last_err}）；请稍后重试，或检查网络能否访问 export.arxiv.org"
        )
