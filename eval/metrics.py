"""PaperMate · 评测指标（A-01，docs/功能增强规格说明 A-01）

本模块是**离线评测的对外入口**；指标实现已统一到 `lightcore.rag.metrics`（单一来源），
此处 re-export 以兼容既有 `from metrics import ...` 调用（run_eval.py / test_eval.py）。

指标分层：

- **检索层（IR）**：`recall_at_k` / `precision_at_k` / `mrr_at_k` / `hit_at_k` /
  `average_precision`（多 query 均值即 MAP）/ `ndcg_at_k`（支持分级相关性 `gains`）
- **Agent 层**：`citation_rate`（溯源率）/ `tool_success_rate`（工具成功率）
- **评测专用聚合**：`aggregate_events` / `summarize_results`（依赖事件流口径，留在此处）

⚠️ 说明（读指标前必看）：
Recall/MRR 等**依赖检索器质量** —— `FakeHashEmbedder` 是哈希占位向量，仅当 query 与目标块
文本高度一致时才稳定命中。因此 `run_eval.py` 的指标**只能证明「链路与指标计算正确」，
不代表真实语义检索质量**；要拿真实质量必须走 `retrieval_eval.py --embedder real`（BGE-M3）。
faithfulness（忠实度）需跨厂商 LLM + temperature=0 判定，prompt 见 `judge_prompts/faithfulness.md`
（**有意不在线执行**，避免评测依赖外部大模型）。
"""

from __future__ import annotations

from collections.abc import Sequence

# 指标实现统一在 lightcore.rag.metrics：
# - 2026-08-30 收敛 Agent 层（citation_rate / tool_success_rate）
# - 2026-09-15 收敛检索层（recall/precision/mrr/hit/average_precision/ndcg）
# 两次收敛的动机相同：两份实现必然漂移，修一处漏一处。
from lightcore.rag.metrics import (  # noqa: F401
    average_precision,
    citation_rate,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    tool_success_rate,
)

#: 带 `@k` 后缀的指标族（k 可变，故按前缀识别）
_METRIC_PREFIXES: tuple[str, ...] = ("recall@", "precision@", "mrr@", "ndcg@", "hit@")

#: 无后缀的固定指标键
_FIXED_METRIC_KEYS: tuple[str, ...] = ("map", "citation_rate", "tool_success_rate", "latency_ms", "cost")


def _is_metric_key(key: str) -> bool:
    """指标键判定：族名前缀（`recall@5` / `ndcg@10` …）或固定名（`map` / `latency_ms` …）。"""
    return key.startswith(_METRIC_PREFIXES) or key in _FIXED_METRIC_KEYS


def aggregate_events(events: Sequence[dict]) -> dict:
    """从单次运行的事件流聚合延迟与成本（与 chat_service 的 done 事件口径一致）。"""
    steps = sum(1 for e in events if e.get("type") == "thinking")
    # 优先读 done 事件里的精确聚合值；否则从工具结果/消息粗略估算
    done = next((e for e in events if e.get("type") == "done"), None)
    if done:
        d = done.get("data", {})
        return {
            "steps": d.get("steps", steps),
            "total_tokens": d.get("total_tokens", 0),
            "cost": d.get("cost", 0.0),
            "latency_ms": d.get("latency_ms", 0),
        }
    return {"steps": steps, "total_tokens": 0, "cost": 0.0, "latency_ms": 0}


def summarize_results(rows: Sequence[dict]) -> dict:
    """对多 case 的指标求均值，输出评测报告摘要。

    - **只输出各行实际存在的指标键**（缺失的键不出现在结果里）——
      避免把「本次没测的项」按 0 计入，让报告出现「忠实度 = 0.0」这类误导性数字；
    - 键名支持带 `@k` 后缀的族（`recall@5` / `ndcg@10` …），k 变化时无需改本函数。
    """
    if not rows:
        return {}
    keys: list[str] = []
    for r in rows:
        for k, v in r.items():
            if k not in keys and _is_metric_key(k) and isinstance(v, int | float):
                keys.append(k)
    out: dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), int | float)]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    return out
