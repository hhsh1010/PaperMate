"""LightCore · 检索查询改写（ER-12 简化版 / query_rewrite 节点）

口语提问 → 检索式：去掉**零信息量**的语气词/引导语/指示代词/结构助词，保留语义核心词，
提升 BM25/向量检索的命中率（规格 ER-12：Rewrite 小节）。

- 纯规则实现（零依赖、确定性、可单测）；相同 query 走 lru_cache 缓存（ER-12 成本控制）。
- 降级原则：改写结果为空（纯语气词/停用词）时回退原始 query，绝不阻断检索。
- 中英文分别处理：
  - **中文按 jieba 分词「整词」删除**（2026-09-26 ④ 加固，见 `_ZH_REMOVABLE_TOKENS`）；
  - 英文按词过滤停用词（`\\b` 边界断言，本就安全）。

★★ **疑问词（什么/怎么/多少…）有意不删** —— 它们兼作**信号词**，删除会造成语义反转
（`受什么影响` → `受影响`）。取证见 `docs/query_rewrite-缺陷取证.md`。
"""

from __future__ import annotations

import re
from functools import lru_cache

#: 中文**可安全删除**的词（按 jieba 分词后的**整词**匹配，绝不做子串替换）。
#:
#: ## 为什么必须按「词」而不是子串（2026-09-26 T-04+ ④ 加固）
#: 旧实现用裸 `str.replace()` 逐词子串替换，存在**顺序耦合**缺陷 ——
#: 删掉一个词后，**相邻字符会重新组合成词表里的另一个词**，被下一轮误删：
#:
#: | 原 query | 第一轮误删 | 拼接结果 | 第二轮误删 | 最终 |
#: |---|---|---|---|---|
#: | `失败原因为什么必须显式记录` | 删「什么」 | `失败原因为必须显式记录` | 删「因为」 | **`失败原必须显式记录`** |
#: | `目的是什么` | 删「什么」 | `目的是` | 删「的」 | **`目是`** |
#: | `了解这个概念` | 删「这个」 | `了解概念` | 删「了」 | **`解概念`** |
#:
#: 实测 221 例评测集中 **200 例（90.5%）** 的 query 被改写、部分语义被破坏。
#: 改用 jieba 分词后，`['目的','是','什么']` 中「目的」是**独立 token**，
#: 「的」再也不会误伤它 —— 这是根治而非绕过。
_ZH_REMOVABLE_TOKENS = frozenset({
    # 纯语气词（零信息量）
    "吗", "呢", "啊", "吧", "呀", "么", "哦", "嗯", "啦",
    # 引导语（纯交际功能）
    "请问", "请帮我", "帮我", "告诉我", "我想要", "我想知道", "麻烦", "帮忙",
    # 指示代词（检索中无区分度）
    "这个", "那个", "这些", "那些", "这篇", "那篇", "这里", "那里",
    # 结构助词 —— jieba 把它们切成独立 token 时才是它们本身，
    # 因此不会误伤「目的 / 了解 / 标的 / 定义」这类含同形字的实词。
    "的", "了", "着", "过",
})

#: ★★ **有意保留**：中文疑问词 —— 它们兼作**信号词**。
#:
#: 删除代价（实测）：
#: - `重排的开销主要受什么影响` → `重排开销受影响`（**语义反转**：询问影响源 → 陈述被影响）
#: - `取消应该被记成什么状态` → `取消应该被记成状态`（丢失限定，gold 块排名 1 → 4）
#:
#: ①-b 对照实测：保留疑问词下 multi_doc recall@5 **0.9024 → 0.9268**。
#: 本常量不参与过滤逻辑，只作为**声明式契约**（供测试与审计引用）。
_ZH_KEPT_QUESTION_WORDS = frozenset({
    "什么", "怎么", "如何", "为什么", "为啥", "哪些", "多少", "是否", "有没有", "是不是",
})

#: 英文疑问词（词级，仅匹配整词/前缀 + 空格）
_EN_QUESTION_WORDS = (
    "what", "how", "why", "which", "who", "where", "when",
    "please", "tell", "explain", "describe", "summarize", "is", "are", "do", "does",
    "can", "could", "would", "should",
)
#: 英文停用词
_EN_STOP_WORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "with", "on", "at", "by",
    "be", "been", "being", "it", "its", "that", "this", "these", "those", "you", "your",
    "me", "my", "we", "our", "from", "about", "into", "over", "under", "i", "not",
}

#: 章节定向展开（2026-08-25 修复「翻译/总结某章节只召回尾部」）：
#: 用户说"翻译摘要 / 总结实验 / 复述方法"时，检索应命中该章节；学术论文的章节标题
#: 多为英文（Abstract / Experiment / Method / Conclusion…），纯中文 query 经词法（BM25
#: 字符级）几乎匹配不到，语义检索也易被章节内的『高密度片段』（如结论数字 92.4%）带偏。
#: 故在改写结果中追加规范英文词（同时保留中文关键词），显著提升跨语言命中与整章覆盖。
#: 该展开是确定性的、与具体 LLM 无关——即便弱模型生成的 retrieve query 不够好，
#: 系统层也能保证章节检索命中，从而根治「只召回尾部→模型以为内容不全→叫用户上传」。
_SECTION_CANON = (
    ("摘要", "abstract"),
    ("引言", "introduction"),
    ("介绍", "introduction"),
    ("背景", "background"),
    ("相关工作", "related work"),
    ("方法", "method methodology"),
    ("实验", "experiment experimental"),
    ("结果", "result results"),
    ("结论", "conclusion"),
    ("讨论", "discussion"),
    ("附录", "appendix"),
    ("参考文献", "reference references"),
)

_HAS_ZH = re.compile(r"[\u4e00-\u9fff]")
_EN_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")


@lru_cache(maxsize=512)
def rewrite_query(query: str) -> str:
    """口语 → 检索式（**2026-09-26 ④ 加固版**）。输入为空/改写后为空 → 回退原 query。

    与旧版的差别只在**中文路径**（`_rewrite_zh`）：从「裸 `str.replace` 子串替换」
    改为「jieba 分词 + 整词删除」，并把疑问词从删除集移出（它们是信号词）。

    需要复现**加固前**行为的对照，用 `rewrite_query_legacy`（仅评测用）。
    """
    q = (query or "").strip()
    if not q:
        return q
    # 1) 去掉 /命令 前缀（/总结 xx → xx）
    q = re.sub(r"^\s*/[^\s]*\s*", "", q)
    # M11 修复（2026-09-10）：剥离命令后为空（用户只发了 `/总结`）时返回空串——
    # 原先会一路回退到**含斜杠的原 query**，把 `/总结` 标记当检索词送进 BM25 污染匹配。
    if not q:
        return ""
    if _HAS_ZH.search(q):
        rewritten = _rewrite_zh(q)
    else:
        rewritten = _rewrite_en(q)
    # 2) 章节定向展开：命中章节关键词时追加规范英文词，提升跨语言检索命中
    extra: list[str] = []
    for key, canon in _SECTION_CANON:
        if key in q:  # 用原始 q 判断，避免被停用词清洗误删
            extra.append(canon)
    if extra:
        rewritten = (rewritten + " " + " ".join(extra)).strip()
    rewritten = rewritten.strip()
    return rewritten if rewritten else query.strip()


# ---------------------------------------------------------------------------
# 加固前实现（**仅供评测对照与迁移参考，生产禁用**）
# ---------------------------------------------------------------------------
#
# ## 为什么保留而不是删掉
# 评测需要一个「改动前基线」来回答「④ 到底修好了什么」。若删掉旧实现，
# 就只能与**别人跑的历史报告**比 —— 无法同代码、同环境复现，对照不成立
# （这条纪律来自 T-04+ ①-b：加 `--query-rewrite full` 才让基线可复现）。
#
# ## 它有什么缺陷（详细取证：`docs/query_rewrite-缺陷取证.md`）
# 中文用裸 `str.replace()` 做**子串**替换，存在**顺序耦合**：
# 删掉一个词后相邻字符重新组合成词表里的另一个词，被下一轮误删。
# 实测 221 例中 200 例（90.5%）被改写，其中 `目的是什么`→`目是`、
# `了解这个概念`→`解概念`、`受什么影响`→`受影响`（语义反转）。

#: ⚠️ **旧词表**（加固前的 `_ZH_QUESTION_WORDS`）。仅 `rewrite_query_legacy` 使用。
_ZH_QUESTION_WORDS_LEGACY = (
    "请问", "帮我", "请帮我", "我想要", "我想知道", "告诉我",
    "什么", "怎么", "如何", "为什么", "为啥", "哪些", "多少", "是否", "有没有", "是不是",
    "吗", "呢", "啊", "吧", "么", "呀",
)
#: ⚠️ **旧词表**（加固前的 `_ZH_STOP_WORDS`）。仅 `rewrite_query_legacy` 使用。
_ZH_STOP_WORDS_LEGACY = (
    "这篇文章", "这篇论文", "这篇文章里面", "论文里面", "文档里面", "里面",
    "关于", "对于", "对于这篇", "主要是", "主要", "大概", "一下", "一个", "这个", "那个",
    "这些", "那些", "以及", "并且", "然后", "那么", "所以", "因为", "但是", "然而",
    "的", "了",
)


@lru_cache(maxsize=512)
def rewrite_query_legacy(query: str) -> str:
    """**加固前**的 `rewrite_query`（2026-09-26 ④ 之前）—— 仅供评测对照，**生产禁用**。

    ⚠️ 有已知缺陷（子串替换 + 顺序耦合），会破坏 query。保留原因见上方说明。
    """
    q = (query or "").strip()
    if not q:
        return q
    q = re.sub(r"^\s*/[^\s]*\s*", "", q)
    if not q:
        return ""
    if _HAS_ZH.search(q):
        # ⚠️ 这里刻意保留旧的「裸子串替换」写法 —— 它就是被对照的**缺陷本体**。
        # 注意：删词作用在临时变量 `_t` 上，**不能**污染 `q`（下面章节展开要用 `q`）。
        _t = q
        for w in _ZH_QUESTION_WORDS_LEGACY:
            _t = _t.replace(w, "")
        for w in _ZH_STOP_WORDS_LEGACY:
            _t = _t.replace(w, "")
        rewritten = re.sub(r"\s+", "", _t)
    else:
        rewritten = _rewrite_en(q)
    extra = [canon for key, canon in _SECTION_CANON if key in q]
    if extra:
        rewritten = (rewritten + " " + " ".join(extra)).strip()
    rewritten = rewritten.strip()
    return rewritten if rewritten else (query or "").strip()


@lru_cache(maxsize=512)
def _jieba_lcut(q: str) -> tuple[str, ...]:
    """jieba 按词切分（**为停用词删除提供词边界**）。

    为什么用 `lru_cache`：`rewrite_query` 本身已缓存，但 `_rewrite_en` 路径不经过分词；
    这里再缓存一层是为了让同一 query 的重复调用（含测试）零成本。
    jieba 首次调用会加载词典（~0.5s），之后单次切分毫秒级。

    降级：jieba 或其词典不可用 → 返回空元组，调用方**不做任何删除**
    （保守原则：宁可少洗，不可洗坏）。
    """
    try:
        import jieba

        return tuple(jieba.lcut(q))
    except Exception:  # noqa: BLE001  可选依赖缺失不应阻断检索
        return ()


def _rewrite_zh(q: str) -> str:
    """中文改写：**按词**删除零信息量词（jieba 提供词边界）。

    ★ 与旧实现的本质区别：旧版用裸 `str.replace()` 做**子串**替换，删掉一个词后
    相邻字符会重新组合成词表里的另一个词而被误删（`目的是什么`→`目是`）。
    按文档 `_ZH_REMOVABLE_TOKENS` 的说明，用分词才是根治。

    ★ **疑问词不删**（`_ZH_KEPT_QUESTION_WORDS`）：它们是信号词，删除会造成语义反转。

    降级：jieba 不可用 → 只做空白压缩、**不删任何词**。
    """
    tokens = _jieba_lcut(q)
    if not tokens:
        return re.sub(r"\s+", "", q)
    kept = [t for t in tokens if t and t not in _ZH_REMOVABLE_TOKENS]
    return re.sub(r"\s+", "", "".join(kept))


def _rewrite_en(q: str) -> str:
    """英文改写：`\\b` 边界断言保证整词匹配（本身不存在「重新拼接」缺陷）。

    ★ 与中文的差别：英文有天然词边界，`re.sub(rf"\\b{w}\\b", " ", low)` 删掉 `is`
    不会把 `this` 变成 `th`；且最后按 `_EN_WORD` 重新提取，天然去重空格。
    2026-09-26 ④ 加固只改了中文路径，英文路径**逐字未动**（无缺陷，改它只会引入风险）。
    """
    low = q.lower()
    # 去疑问词（整词）
    for w in _EN_QUESTION_WORDS:
        low = re.sub(rf"\b{w}\b", " ", low)
    # 去停用词
    words = [w for w in _EN_WORD.findall(low) if w not in _EN_STOP_WORDS and len(w) > 1]
    return " ".join(words)


@lru_cache(maxsize=512)
def expand_sections(query: str) -> str:
    """**只做「/命令 剥离 + 章节定向展开」**，不做疑问词 / 停用词清洗。

    T-04+（2026-09-26）新增：`rewrite_query` 一度有**系统性缺陷**（裸 `str.replace()`
    子串替换 → `目的是什么`→`目是`、`受什么影响`→`受影响`），本函数是当时的**止血版**。

    ### 现状（2026-09-26 ④ 加固之后）
    `rewrite_query` 已改用 **jieba 分词 + 整词删除**，缺陷已根治。本函数与它的分工：
    - `expand_sections` = **最保守**：一个词都不删（只做命令剥离 + 章节展开）；
    - `rewrite_query` = **会删零信息量词**（语气词/引导语/指示代词/结构助词），保留疑问词。
    实测 221 例中两者**结论 219/221 相同**（差异用例见 `docs/T04plus-①b-止血验证报告.md`）；
    评测可用 `--query-rewrite {sections,v2,legacy}` 三臂对照。

    ### 保留的能力（与 `rewrite_query` 一致）
    1. `/命令` 前缀剥离（纯格式处理，`/总结 xx` → `xx`）；
    2. `_SECTION_CANON` 章节定向展开（2026-08-25 为修「翻译/总结某章节只召回尾部」加的
       **跨语言检索**能力：中文题面 → 追加规范英文词）。

    ### 为什么保留它（而不是删掉直接用 `rewrite_query`）
    "不删任何词"是一条**更弱的假设**：任何删除规则都会在某个 query 上出错，
    而这条路径保证**至少不会比原 query 更差**。它还是一次实测的产物（①-b 对照），
    删掉就失去了那个可复现的对照点。
    """
    q = (query or "").strip()
    if not q:
        return q
    # 1) 去掉 /命令 前缀（与 `rewrite_query` 同一规则，M11 语义不变）
    q = re.sub(r"^\s*/[^\s]*\s*", "", q)
    # 剥离命令后为空（用户只发了 `/总结`）→ 返回空串，由调用方决定是否回退。
    if not q:
        return ""
    # 2) 章节定向展开：命中章节关键词时追加规范英文词（用原始 q 判断）
    extra = [canon for key, canon in _SECTION_CANON if key in q]
    return (q + " " + " ".join(extra)).strip() if extra else q
