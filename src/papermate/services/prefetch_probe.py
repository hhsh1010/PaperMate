"""O-12 档 2 · P0 探针（2026-10-02）：**命中率**测量，零行为变更、零 GPU 成本。

## 它测什么（以及刻意不测什么）

档 2 的设计（`docs/O-12-档2-并行预取-影响面与去重口径设计.md`）里，唯一不确定的量是
**命中率** —— 即模型自己 `retrieve` 用的 query 是否**等于**用户原 query（严格相等，
因为"不匹配就复用"等于偷换证据）。

P0 **不真的发起预取检索**，因此：
- **零 GPU 成本**（不白跑一次编码/重排）—— 数据靠"比较两次 query"就能得到；
- **零行为变更**（只写一行日志），因此可以放心地开着跑真实流量。

真正的"预取能否赶在模型提问之前完成"由两个量推断（都已有数据）：
`t_first_retrieve_ms`（本探针测）vs 该轮检索耗时（既有 `[工具] retrieve … 耗时=…ms` 日志）。

## 判据

| 观测 | 含义 |
|---|---|
| `命中=是` 占比 | 档 2 的**复用率上限**（≥50% 才值得做 P1） |
| `t_first_retrieve` 分布 | 预取窗口（实测 ① LLM1 决策中位 1748ms）；小于检索耗时则预取大概率来不及 |

⚠️ 探针日志里会打印 query 文本（本地单用户环境可接受）；**不要**把它接到外部日志系统。
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

#: 探针开关（默认关）。为什么默认关：它是**实验性观测**，且会在日志里打印 query 文本。
PROBE_ENV = "PAPERMATE_PREFETCH_PROBE"

_WS = re.compile(r"\s+")


def probe_enabled() -> bool:
    """每次调用都读环境变量（便于测试与运行时切换，不缓存）。"""
    return os.environ.get(PROBE_ENV, "") == "1"


def prefetch_key(query: str) -> str:
    """P0/P1 **共用**的规范化判据（单一来源，避免"两把尺子"）。

    只做去空白 + ASCII 小写：
    - 去空白：模型经常把多空格/换行压平，语义不变；
    - 小写：仅对 ASCII 生效（中文无大小写），避免 `Retrieve` / `retrieve` 这类差异造成假未命中。
    ⚠️ **刻意不做**去标点/去疑问词/词干化 —— 那些会引入"看起来相同其实不同"的假命中，
    而假命中的代价是**偷换证据**（比低命中率严重得多）。
    """
    return _WS.sub("", (query or "")).strip().lower()


def log_probe(*, user_query: str, model_query: str | None, t_first_retrieve_ms: float | None) -> None:
    """打一行探针日志（命中/未命中 + 首次 retrieve 时刻 + 两侧 query，便于人工核对）。"""
    if model_query is None:
        # 该轮**没有** retrieve（例如模型没检索）——也要记：它计入分母（"不会命中"）。
        logger.info("[预取探针] 命中=否 t_first_retrieve=nanms（本轮无 retrieve）| user_q=%r",
                    user_query[:80])
        return
    hit = prefetch_key(user_query) == prefetch_key(model_query)
    logger.info(
        "[预取探针] 命中=%s t_first_retrieve=%.0fms | user_q=%r | model_q=%r",
        "是" if hit else "否", t_first_retrieve_ms or 0.0, user_query[:80], model_query[:80],
    )
