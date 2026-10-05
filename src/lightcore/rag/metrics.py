"""LightCore · 评测指标（运行期与离线评测共用的单一来源）

背景（2026-08-30 修复）：
`tool_success_rate` / `citation_rate` 原先只存在于 `eval/metrics.py`（离线评测脚本目录，
不在 src 下，运行期不可导入），而 `papermate.skills.dev_eval` 却 import
`lightcore.rag.metrics` —— 该模块当时并不存在，导致有 DB 时 `/eval` 命令直接抛
`ModuleNotFoundError`，Agent 指标（工具成功率 / 平均成本 / 平均步数）全部拿不到。

现统一置于此处：
- 运行期（`dev_eval` 技能、`GET /api/v1/eval/summary`）直接 import；
- 离线评测（`eval/metrics.py`）改为 re-export，保证两套口径不会漂移。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def tool_success_rate(events: Sequence[dict]) -> float:
    """从 Agent 事件流计算工具成功率（tool_call 与 tool_result.ok）。

    无任何 tool_call 时视为 1.0（无失败即成功），避免空样本把指标拉成 0。
    """
    calls = [e for e in events if e.get("type") == "tool_call"]
    results = [e for e in events if e.get("type") == "tool_result"]
    if not calls:
        return 1.0
    ok = sum(1 for r in results if r.get("data", {}).get("ok", True))
    return ok / len(calls)


def citation_rate(answers: Sequence[dict]) -> float:
    """给定若干**回答样本**（含 `citations` 字段），返回『带引用』样本占比（溯源率）。

    ⚠️ 契约（2026-09-16 明确，两处都踩过 —— 见 docs/用量与评测功能体检报告.md 问题 3）：
    1. **样本必须是"完整回答"**，即 `message` 事件中带 `content` 的那一条；
       流式 `delta` 片段不是回答（实测 11279/11297 都是片段），混进来会把指标稀释到约 0；
    2. `citations` 必须由**生产者真的写入**：`chat_service.persist_turn(..., citations=...)`
       会把它附加到最终回答事件的 `data` 上。此前没有任何生产者写这个字段 →
       指标恒为 0，用户看到"溯源率 0%"会误判"引用功能坏了"。

    ★ 2026-10-05 第三条契约（新增）：**同一条回答可能在样本里出现多次**，必须去重。
    实测（10 条真实轨迹）：15 个 content 事件实际只对应 **8 条唯一回答**（7 对重复），
    且重复的那两条**逐字符相同**、只差 `citations` 键（一条有、一条没有）。
    根因是"发出回答"有两个生产者（`loop.py:1544` 补发 + `:1945` 再发），
    `persist_turn` 只给**最后一条**附加引用 ⇒ 每条回答恒定贡献「1 无引用 + 1 有引用」
    ⇒ **无论实际行为如何，比率都被钉在 ~50%**（实测显示 53.33%，去重后为 **8/8 = 100%**）。

    去重规则：按 `content.strip()` 归组，**组内任一成员带引用**即该条回答记为"带引用"
    （缺引用的那份是重复产物的残影，不代表真实回答不带引用）；
    无 `content` 的样本（历史调用方手工构造的裸 dict，如 `{"citations": [1]}`）
    **不做去重**，各自独立计一个样本 —— 保持既有契约不变。
    """
    if not answers:
        return 0.0
    cited: dict[str, bool] = {}
    for i, a in enumerate(answers):
        # 无 content 的样本无法判定"是不是同一条回答" ⇒ 不猜、各自独立（键含序号）
        text = str(a.get("content") or "").strip()
        key = f"content:{text}" if text else f"blank:{i}"
        cited[key] = cited.get(key, False) or bool(a.get("citations"))
    return sum(1 for v in cited.values() if v) / len(cited)


# ==========================================================================
# 检索层（IR）指标
#
# 2026-09-15：原先只存在于 `eval/metrics.py`（离线脚本目录），本次按既有架构决策
# **统一上移到此处**（与 Agent 层指标同源），`eval/metrics.py` 改为 re-export。
# 理由与 2026-08-30 那次相同：两份实现必然漂移，修一处漏一处。
# ==========================================================================


def _as_ids(retrieved: Sequence[Any]) -> list[str]:
    """把检索结果规整成 id 列表；兼容 RetrievedChunk / dict / 裸字符串。"""
    ids: list[str] = []
    for r in retrieved:
        if isinstance(r, dict):
            ids.append(str(r.get("id")))
        elif hasattr(r, "id"):
            ids.append(str(r.id))
        else:
            ids.append(str(r))
    return ids


def recall_at_k(retrieved: Sequence[Any], gold: Sequence[str], k: int) -> float:
    """Recall@k：前 k 个结果覆盖了 gold 集合的**比例**（gold 为空时 0.0）。

    回答「该找的**都**找到了吗」—— 对「多段证据缺一不可」的问题（如综述类）敏感。
    """
    if not gold:
        return 0.0
    top = set(_as_ids(retrieved)[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def precision_at_k(retrieved: Sequence[Any], gold: Sequence[str], k: int) -> float:
    """Precision@k：前 k 个结果中**相关**的占比（分母取 `min(k, 实际返回数)`）。

    回答「塞进上下文的内容**干不干净**」—— 直接对应「注入给大模型的噪声有多少」。
    """
    top = _as_ids(retrieved)[:k]
    if not top:
        return 0.0
    gold_set = set(gold)
    return sum(1 for r in top if r in gold_set) / len(top)


def mrr_at_k(retrieved: Sequence[Any], gold: Sequence[str], k: int) -> float:
    """MRR@k（单 query 贡献）：**首个**命中的排名倒数；未命中记 0.0。多 query 取均值即 MRR。

    只关心「第一个相关的结果排得多靠前」—— 对应「模型第一眼看到的是不是有用的」。
    """
    gold_set = set(gold)
    for rank, rid in enumerate(_as_ids(retrieved)[:k], start=1):
        if rid in gold_set:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved: Sequence[Any], gold: Sequence[str], k: int) -> float:
    """Hit@k：前 k 是否**至少命中一个**（1.0 / 0.0）。

    最粗的判据 —— 「有没有找到」，不受 gold 规模影响，适合作为改动后的冒烟指标。
    """
    if not gold:
        return 0.0
    return 1.0 if set(_as_ids(retrieved)[:k]) & set(gold) else 0.0


def average_precision(retrieved: Sequence[Any], gold: Sequence[str], k: int | None = None) -> float:
    """AP（单 query 平均精度）；多 query 取均值即 **MAP**。

    `AP = (Σ 命中位置 r 的 precision@r) / |gold|` —— 同时反映「命中多少」与「命中多靠前」，
    比 P@k 更稳健（不会被 k 的取值人为拉高/拉低）。
    """
    if not gold:
        return 0.0
    gold_set = set(gold)
    ids = _as_ids(retrieved)
    if k is not None:
        ids = ids[:k]
    hits = 0
    acc = 0.0
    for rank, rid in enumerate(ids, start=1):
        if rid in gold_set:
            hits += 1
            acc += hits / rank
    return acc / len(gold_set)


def ndcg_at_k(
    retrieved: Sequence[Any],
    gold: Sequence[str],
    k: int,
    *,
    gains: dict[str, float] | None = None,
) -> float:
    """nDCG@k：归一化折损累计增益（取值 [0,1]）。

    - **默认二值相关性**（命中=1，否则 0）：`DCG = Σ rel_i / log2(i+1)`
    - 传 `gains` 可做**分级相关性**（如 `{"d-c1": 3, "d-c2": 1}`）：
      `DCG = Σ (2^rel_i − 1) / log2(i+1)`
    - IDCG 取「理想排序」下的最大 DCG 用于归一化；IDCG 为 0（无任何相关项）时返回 0.0

    **`gains` 应覆盖「所有相关项」，而不只是 `gold`** —— 分级标注里通常有 gold（核心证据）
    之外的次级相关段落。IDCG 的候选池取 `gold ∪ gains` 中 rel>0 的并集：
    若只按 gold 算 IDCG，当 gains 含非 gold 的相关项时 **IDCG 偏小 → nDCG 会算出 >1 的荒谬值**。
    （2026-09-15 修复：原实现只对 gold 集合取相关性。）

    **与 MRR 的区别**：MRR 只看第一个命中；nDCG 考虑**所有**命中的位置与（可选）相关性等级，
    是排序质量的综合指标。⚠️ 没有分级 `gains` 时它与 MRR 高度相关，别把两者当独立证据。
    """
    import math

    if not gold and not gains:
        return 0.0
    gold_set = set(gold)

    def rel(rid: str) -> float:
        if gains is not None:
            return float(gains.get(rid, 0.0))
        return 1.0 if rid in gold_set else 0.0

    def gain(r: float) -> float:
        return (2.0**r - 1.0) if gains is not None else r

    ids = _as_ids(retrieved)[:k]
    dcg = sum(gain(rel(rid)) / math.log2(i + 1) for i, rid in enumerate(ids, start=1))
    # 理想排序池 = 所有相关项（gold ∪ gains 中 rel>0 者），保证 IDCG ≥ DCG → nDCG ∈ [0,1]
    pool = [cid for cid in (gold_set | set(gains or {})) if rel(cid) > 0]
    ideal = sorted((rel(cid) for cid in pool), reverse=True)[:k]
    idcg = sum(gain(r) / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
