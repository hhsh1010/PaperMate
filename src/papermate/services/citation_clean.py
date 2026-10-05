"""引用片段清洗 —— **全项目唯一实现**（B2-X2，2026-10-01）。

## 为什么要有这个模块

此前"哪些引用能进列表、文本怎么裁"有**三份独立实现**：

| # | 位置 | 规则 |
|---|---|---|
| ① | `api/routes_search.py`（原 `_clean_citations` / `_dedup_citations`） | 空/＜16→丢 · **>600 截断** · 噪声词 · 去重 |
| ② | `skills/_common.citations_from` + `literature_review`/`related_work_gen` 内联 | **只做 `normalize_page`**（长度/噪声/去重全无） |
| ③ | `web/src/components/CitationCard.tsx::cleanCitations` | 空/＜16→丢 · **>600 直接丢弃** · 噪声词（**比后端多一个 `"Citation"`**） |

三份各自演化 ⇒ **行为分歧**，且分歧方向正是本项目最忌讳的一类：

- 分歧 1：**技能路径完全绕过清洗**（`/综述`、`/相关工作` 的引用文本是**整块 chunk**）；
- 分歧 2：**前端比后端更严**（>600 丢弃 vs 截断；噪声词多一个）
  ⇒ 引用**在 UI 里静默消失，但落库与 `citation_rate` 仍然算它** = 「**指标有数、用户看不到**」。

**量化依据**（真实库只读，2026-10-01）：`chunks` 共 658 条，**58.2% 超过 600 字符**
（p50 = 921、最长 2748）⇒ 分歧 2 对"整块文本"的引用**过半概率触发**，不是理论风险。

## 收敛后的口径（单一事实来源）

- 后端产出即**最终形态**；前端**不再做任何清洗/过滤**，只保留展示层截断（条数 / 预览字符）。
- 本文的 `NOISE_KEYWORDS` 是**唯一**一份词表。

### ★ 关于 `"Citation"` 一词的取舍（有意决定，勿顺手改回）

前端那份词表比后端多一个 `"Citation"`。收敛时**未采纳**该词，理由：

1. `"Citation"` 是**常见正文词**（如 "Citation counts have grown …"）—— 把它整体当噪声，
   会**静默丢弃合法证据**，正是 B3（2026-09-29）判定为 **bug** 的那类行为
   （丢弃 → 无法溯源 + 可能误触发"零引用/疑似幻觉"提示）；
2. 该词**只在**前端过滤过 ⇒ 指标口径里从来没有它 ⇒ **不采纳 = 指标零漂移**；
3. 代价仅"少数含该词的片段由不显示变为显示"，方向与 B3 一致（**宁可显示噪声，不可静默丢证据**）。

⚠️ 若将来确认它在本场景确实是页眉噪声（而不是正文用词），加回是一个词的改动 —— 但**必须同时**
更新本模块与 `tests/test_p3_minor_fixes.py` 的相关断言，并重跑评测指纹（它决定"算不算引用"）。

## 与相邻概念的区别（别混）

- `RETRIEVE_MAX_TEXT=200`（`retrieve` 注入给模型的压缩）—— 那管**模型看到多少**；
- 本模块管**给用户看的引用卡片**（存储/展示形态）；
- `lightcore.model_injection_truncates` 给出的 `model_truncated` 标记 —— 说明"模型只看到片段开头"，
  其判据刻意保持**单一来源**（lightcore），本模块不重算。
"""

from __future__ import annotations

import re

#: 太短的片段视为噪声（页眉残留、单行编号等），不作为引用展示。
MIN_TEXT = 16

#: 单条引用正文的上限（B3，2026-09-29）：超过则**截断 + 省略号**，而不是整条丢弃。
#: 截断成**总长 = MAX_TEXT**（含省略号），即 `text[: MAX_TEXT - 1] + "…"`。
MAX_TEXT = 600

#: 噪声词（页眉/页脚/元数据字段名）。**全项目唯一一份**（B2-X2）。
#: 只收"基本可判定为版面噪声"的词；**不收** `"Citation"`（理由见模块 docstring）。
NOISE_KEYWORDS = (
    "重点实验", "作者简介", "基金项目", "收稿日期", "E-mail", "通讯作者",
    "联系电话", "版权声明", "DOI", "ISBN", "卷号", "期号", "总第",
)


def clean_citations(results: list[dict]) -> list[dict]:
    """清洗引用片段：去空、去太短、超长**截断**（B3）、去噪声、单次内去重（B4）。

    ### B3（2026-09-29）：超长片段**截断**而非**丢弃**
    原实现 `len(text) > 600: continue` 会让长 chunk 的引用**整条消失**（不是截断），两个后果：
    ① **无法溯源**——模型引用了某段，用户引用列表里却找不到它（完整性损失）；
    ② ★ **误触发"零引用/疑似幻觉"提示**——若本轮命中的 chunk **全部**超过 600 字，
      `citations` 会意外为空 ⇒ 下游 `zero_citation_notice` 据"零引用"判定"可能是幻觉"，
      而真相恰恰相反（证据很充分，只是被清洗函数丢掉了）。这是**判据与事实不同源**。

    ### B4（2026-09-29）：`str.replace(r"\\s+", " ")` 是**字面量**，不是正则
    `r"\\s+"` 传给 `str.replace` 只会匹配字面文本 `\\s+`（几乎不可能出现）⇒ 本意的
    "把空白归一化"**从未生效**。此处用 `re.sub`。这正是本项目反复吃过的亏：
    **"看起来在做 A、实际做的是 B"，且无任何测试报警**。
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in results:
        text = (r.get("text") or "").strip()
        if not text or len(text) < MIN_TEXT:
            continue
        if len(text) > MAX_TEXT:  # B3：截断 + 标记，而不是丢弃
            text = text[: MAX_TEXT - 1] + "…"
        if any(k in text for k in NOISE_KEYWORDS):
            continue
        key = re.sub(r"\s+", " ", text[:80])  # B4：真正的空白归一化
        if key in seen:
            continue
        seen.add(key)
        # 用**清洗后的文本**（已 strip / 可能已截断），并复制字典：
        # 直接用原 `r` 会①把未清洗的长文带出去、②与调用方共享可变对象（潜在串改）。
        out.append({**r, "text": text})
    return out


def dedup_citations(
    results: list[dict], keys: set[str], limit: int,
    *, capped_out: list[bool] | None = None,
) -> list[dict]:
    """清洗 + **跨工具调用**去重 + 总数截断（N1 修复 2026-09-10）。

    `clean_citations` 只在单次调用内去重（每次新建 seen），而 agent 一轮里会多次检索，
    各次结果直接 extend 会导致同一片段反复出现、总量无上限（用户实测「引用来源几千条」）。
    本函数用**调用方持有的** `keys` 集合做跨调用去重，并按 `limit` 截断。

    去重键：优先 chunk id（精确），退化为文本前 80 字符（无 id 的兜底）。

    ### B5（2026-09-29）：`capped_out` —— 让"被截断"这件事**可观测**
    达 `limit` 后此前是**静默 `break`**：用户在"引用很多"时不知道自己少看了若干条，
    trace 里也无痕迹。传入一个 list 时，若**确有新增引用因达上限被丢弃**，就 `append(True)`。
    ⚠️ 是**可选**参数（默认 `None`）⇒ 不改既有调用的签名与返回类型；
    ⚠️ 只在"**该条是新的**（非重复）却因上限被丢"时置位 ⇒ 不会因"恰好填满 limit"而误报。
    """
    out: list[dict] = []
    for c in clean_citations(results):
        ck = str(c.get("id") or (c.get("text") or "")[:80])
        if ck in keys:
            continue
        if len(keys) >= limit:
            if capped_out is not None:
                capped_out.append(True)
            break
        keys.add(ck)
        out.append(c)
    return out
