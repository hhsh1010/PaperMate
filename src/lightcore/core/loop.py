"""LightCore · Agent循环（docs/05 §Agent 循环 / ReAct + Function Calling 双协议）

状态机：THINK → (ACT → OBSERVE)* → FINISH；异常走 ERROR_HANDLE → (重试/ABORT)。
配置 LoopConfig（★ 下列为**代码默认值**；`data/provider_settings.json` 可覆盖，
实测该文件当前就设了前两项，**别只改代码就以为生效了**）：
- max_iterations=12        : 最大思考-行动轮次（NFR 防失控）
- max_tool_failures=3      : 工具连续失败上限，超出则 ABORT
- max_steps_no_progress=3  : 连续"未获得新证据"步数上限 ⇒ 跨越时**注入换策略注记**（不中止；阈值按真实分布定）
- max_tokens_budget=30000  : **单次请求**的上下文上限，超则熔断（NFR: 成本熔断）
- max_tokens_per_turn=150000 : **整轮累计**token（所有迭代加起来），防跑飞的粗护栏
- max_cost_per_turn=1.0    : 单轮成本上限（¥）——**真正的钱袋子护栏**

⚠️⚠️ **两处最易误读的口径（都已踩坑，勿再改反）**：
1. **`max_tokens_budget` 量的是「单次请求」不是「整段对话」**。历史可以远超它
   （超了会先压缩、再闸门裁剪）；受约束的只是**每一次 LLM 调用的 prompt**。
2. **`max_tokens_budget` ≠ `max_tokens_per_turn`**：前者单次、后者整轮累加。
   ⚠️ `messages.prompt_tokens` 记的是**整轮累计**（已踩坑：看到 8 万会误以为
   单次上下文 8 万）。**单次上下文只能看日志的 `[LLM] ... prompt=`**。
派生刻度（同一件事的两个刻度，不另设配置）：
- `compact_trigger_tokens` = budget × **0.7** = **21000** → 触发历史压缩 + 上下文闸门
- 压缩执行线cap = trigger × 0.7 = **14699** → 压到这条以下

对外产出 SSE 风格事件（thinking/tool_call/tool_result/message/done/error），
供 docs/04 的 SSE 协议与 S3 验证线直接消费。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..llm.base import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    Role,
    ToolCall,
    ToolSpecForLLM,
    _rough_tokens,
)
from ..llm.fake import FakeLLM
from ..llm.router import OAICompatibleLLM, estimate_cost

# C7-lightcore（2026-09-23）：收口引导的标记取自单一来源（见下方使用处的说明）
from ..prompts import GUIDANCE_MARK
from ..tools.spec import Permission, ToolExecutor, ToolResult
from . import failure

logger = logging.getLogger(__name__)

#: 历史摘要（`_summarize_dropped`）的 max_tokens 上限。
#: N11（2026-09-11）：摘要只需 ≤150 字，但原先未限制 —— thinking 模型下思考链吃满数千 token，
#: 单次耗时 70+ 秒且无任何可见输出，用户以为卡死并点「停止」→「多轮上下文无法生成有效回答」。
# ★ 2026-10-23：150 字的摘要丢信息太多（具体结论/数值/被否定的方向全丢），
#   改为「6 段式 ≤800 字」⇒ 上限 600 → 1400（800 汉字约 1200~1400 tokens）。
SUMMARY_MAX_TOKENS = 1400

#: 历史摘要调用的超时（秒）。超时即回退滚动截断，绝不阻断主流程。
SUMMARY_TIMEOUT_S = 20.0

#: 失败自评（`_reflect_on_abort`）的 max_tokens 上限与超时。
#: 与摘要同源的护栏（N11 教训：**元任务必须限 max_tokens + 设超时**，否则 thinking 模型
#: 会吃满思考链、耗时失控且无可见输出）。自评只需「一行判定 + 两行建议」，400 足够。
REFLECT_MAX_TOKENS = 400
REFLECT_TIMEOUT_S = 15.0


class LoopState(StrEnum):
    THINK = "THINK"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    ERROR_HANDLE = "ERROR_HANDLE"
    ABORT = "ABORT"
    FINISH = "FINISH"


@dataclass
class LoopConfig:
    max_iterations: int = 12
    max_tool_failures: int = 3
    # 单次请求上下文上限（防上下文爆炸）：本次调用的 **prompt** 规模超过它即中止。
    # 2026-09-17 语义修正：它原先同时承担「整轮累计熔断」，导致 30000 只够跑 2-3 轮迭代
    # （实测单轮 prompt 已达 24883 —— 绑多篇文档 + 累积检索结果），而 max_iterations=12，
    # 等于"允许迭代 12 轮、却只给 2 轮的额度"。现在它只管单轮上下文，累计交给下一项。
    max_tokens_budget: int = 30000
    # 整轮累计 token 上限（含所有迭代）：防"跑飞"的粗护栏，与 max_cost_per_turn（¥）互补。
    # 实测参考：单轮 prompt 5k~25k、正常 3-5 轮迭代 → 50k~125k；默认 150000 正常用不到，
    # 但能拦住"失控循环十几轮"。真正的钱袋子护栏是 max_cost_per_turn。
    max_tokens_per_turn: int = 150000
    # C1 观察期收口（2026-10-12）：**阈值按真实分布定**，而不是沿用拍脑袋的 6。
    # 依据（真实日志）：触发过 6 次，连续步数**全部是 1**（1/6），跨阈值告警 **0 次**
    # ⇒ 阈值 6 实际永远不会被跨过 ⇒ 护栏形同虚设。
    # 取 3 的理由：容得下"重复提问/重复抽页"这类**正常**的 1~2 步，又能在真正原地打转时出手。
    max_steps_no_progress: int = 3
    max_cost_per_turn: float = 1.0  # 单轮成本熔断上限（¥），默认 ¥1/轮（NFR: 成本熔断）
    thinking: bool = False  # V4 思考模式（features.thinking）：开启时展示推理链且不强制 tool_choice（官方不兼容）
    temperature: float = 0.3
    web_search: bool = False  # 会话『联网』开关：开启时把大模型原生联网搜索注入 LLM 调用（qwen enable_search / deepseek 托管工具）
    # 失败轮次自评（features.reflect_on_abort，2026-09-16 首次接线）：
    # 仅在**中止路径**（迭代耗尽 / 连续无进展）触发 —— 那一轮本来就要失败，
    # 多花 1~2 次调用换一个更好的收尾是划算的；正常收敛的轮次**零额外调用**。
    reflect_on_abort: bool = False
    # 单次工具执行超时（秒，2026-09-16）。此前**循环层完全没有工具超时**：
    # 某个工具挂死（网络连接不返回、外部服务僵住）会让整轮无限等待，用户只能点「停止」脱身。
    # 取值刻意**宽松**（2 分钟）：目的是兜住"挂死"，而不是给正常慢调用设卡 ——
    # LLM 驱动的工具（summarize / plan_executor）本身可能跑十几秒甚至更久，
    # 卡太紧会把它误判为失败（那是比"挂死"更常见的伤害）。
    tool_timeout_s: float = 120.0

    @property
    def compact_trigger_tokens(self) -> int:
        """历史压缩的触发线（**真实 token** 口径）= 单次请求上限的 70%。

        语义：**整个单次请求的 prompt 规模**（不是"历史那部分"的估算）。
        留 30% 余量给本轮新增（用户输入 + 检索注入 + 工具结果）—— 压缩发生在发出之前，
        必须为"还没发出去的那部分"留空间。

        为什么派生而不是新加一项配置：它与 `max_tokens_budget` 是**同一件事的两个刻度**
        （上限 / 提前量），分开配只会产生"上限调了、触发线没调"的不一致。
        """
        return int(self.max_tokens_budget * 0.7)

    @property
    def compact_keep_capacity(self) -> int:
        """压缩的 **整 prompt 目标**（★ 与触发线**同量纲**）= 触发线的 70%。

        ## ★★ 2026-10-03 P0 修正：原先**量纲不一致**，实测导致压缩空转（保留原文脉）

        - **2026-09-25 T-01-B（第一次修）**：旧实现写死 `max_tokens_budget // 2`（15000），
          且拿**估算值**去比 ⇒ 改成"由触发线派生"。
          **但只统一了「数值来源」，未统一「量纲」** ⇒ 问题仍在（见下）。
        - **2026-09-25 收尾实验（T-01-F，`scripts/exp_t01_compact_sim.py`）** 的措辞：
          「压缩延迟一轮 + 单次压得不够」，而不是"一条都不丢"。
        - **2026-10-02 线上日志（实测）**：「触发了但**未丢弃任何消息**：历史估算=11427 ≤ 阈值=14699」。

        **根因（本次定位）**：本属性**数值**由触发线派生（21000×0.7=14700），但**比较对象**是
        「**仅历史**的估算值」；而触发线的比较对象是「**整 prompt** 的真实值」⇒
        数值看似同源、**量纲却是两回事**：
        `整 prompt = 固定开销 + 历史 + 本轮新增`（实测触发时历史仅 11427，差额近 1 万）
        ⇒ `11427 < 14700` ⇒ `while` 不成立 ⇒ **一条不丢**。

        **现在**：本属性 = 「**整 prompt 要压到的目标**」，由 `_history_cap_for()`
        换算成本次该保留多少历史：
        `hist_cap = hist_est − max(0, 整prompt基线 − 本目标)`。

        **为什么盯整 prompt 而不是历史**：压缩是**给本轮新增腾空间**，而"本轮新增"
        （用户输入 + 工具结果）与固定开销**都不在历史里**。只盯历史 ⇒ 会得出
        "历史还没满、不用压"的错误结论，而整 prompt 已经要撞上限。

        ⚠️ `max(1000, ...)` 下限保留：极小预算的测试/演示场景避免 cap 归零。
        """
        return max(1000, int(self.compact_trigger_tokens * 0.7))

    def _history_cap_for(self, hist_est: int, prompt_baseline: int) -> int:
        """把「整 prompt 目标」换算成「**本次历史保留量**」。P0（2026-10-03）。

        ### 公式
        ```
        need_cut = max(0, prompt_baseline − compact_keep_capacity)  # 整 prompt 超目标多少
        hist_cap = max(0, hist_est − need_cut)                      # 只能从历史里削
        ```
        - **有真实基线**（跨轮）：`prompt_baseline` = 上一轮首步真实 prompt，
          即"本轮整 prompt 的预估"（实测代理误差 +2%，见 `_prepare_history`）。
          超出目标的量**必须从历史削**（固定开销不可动、本轮新增尚未发）⇒ 上式。
          **案例验算**：`hist_est=11427`、`baseline=21000`、目标 `14700`
          ⇒ `need_cut=6300` ⇒ `hist_cap=5127` ⇒ **真的会丢 6300 的历史**（修复空转 ✓）。

        ★ **无真实基线**（首轮 / 进程重启后首次）：退回"历史估算 vs 目标"的**旧判据**
        （`hist_cap = 目标`）⇒ **首轮行为逐字不变**。
        理由：首轮无从得知"整 prompt 从哪来"；若此时硬套公式
        （`baseline=0 ⇒ need_cut=0 ⇒ hist_cap=hist_est`）会**永不压缩**，
        等于把"首轮兜底"整个删掉 ⇒ **回归**（既有测试会红）。
        """
        # ⚠️ 本方法定义在 **LoopConfig** 上 ⇒ `self` 就是 config，**不要**写 `self.config.`
        #   （写了会 AttributeError：'LoopConfig' object has no attribute 'config'）。
        target = self.compact_keep_capacity
        if not prompt_baseline:
            return target
        need_cut = max(0, prompt_baseline - target)
        return max(0, hist_est - need_cut)


@dataclass
class LoopEvent:
    """SSE 事件载体（docs/04 API 一览）。"""

    type: str  # thinking | tool_call | tool_result | message | done | error
    data: dict[str, Any] = field(default_factory=dict)
    state: LoopState = LoopState.THINK

    def to_sse(self) -> str:
        import json

        payload = {"type": self.type, "state": self.state.value, **self.data}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---- 检索注入截断（P0-2）：压缩 retrieve 结果，避免撑爆 8000 token 预算 ----
_RETRIEVE_MAX_TEXT = 200

#: `pdf_extract` 单次注入的**总量**上限（字符）。P1-b（2026-10-03）。
#:
#: ### 取值依据（实测，不拍脑袋）
#: `data/logs/app.log` 实测两工具的注入量：
#: - `pdf_extract`：**原始 = 注入（1:1 直灌，无任何压缩）**，样本 1836 / 2659 / 2859 /
#:   3014 / 3089 / 3129 字，**极端值 11057 与 13498 字**（≈4k~4.9k tokens）；
#: - `retrieve`（对照）：原始 4402~13263 字 → 注入 **1112~3105 字**（压缩 3~6×）。
#: ⇒ 4000 既**保住了正常 page 查询**（实测正常上限 3129 < 4000，逐字不变），
#:   又能压住极端值（11057 / 13498 ⇒ 削掉约 60~70%）。
#:
#: ### 为什么是"总量"而不是像 `retrieve` 那样的"每段 200 字"
#: `retrieve` 的语义是"找最相关的**片段**"，逐段截断合理；
#: 而 `pdf_extract` 的语义是"**取这一段/这一页的全文**"（工具描述与离线兜底指引都这么写），
#: 逐段砍成 200 字会让"整节翻译"彻底失效。
#: ⇒ 按**总量**设卡：尽量保住**完整段落**，只在整体过大时削。
_PDF_EXTRACT_MAX_TOTAL = 4000

#: 截断提示的**幂等标记**：`_prepare_tool_output` 会被调用两次（调用点 + `_format_tool_result`），
#: 靠这个标记避免把 note 重复拼两遍。
_PDF_EXTRACT_TRUNCATED_FLAG = "【本页/本段内容过长，已按注入上限截断】"


def _compact_pdf_extract_output(output: Any) -> Any:
    """压缩 `pdf_extract` 的注入量（P1-b，2026-10-03）。

    ## 为什么需要
    `pdf_extract` 此前是**唯一完全不压缩**的注入源：`retrieve` 每段压到 200 字，
    而它 `text` **原样直灌**（实测「原始=注入」）。而上下文里它的占比很高
    （前序实测：占注入 token 的 46%）⇒ 单次 13498 字（≈4.9k tokens）就能吃掉
    单轮预算（30k）的 16%。

    ## 与 `_compact_retrieve_output` 的两处**刻意不同**
    1. **按总量而非逐段**（理由见 `_PDF_EXTRACT_MAX_TOTAL` 注释）；
    2. **加了"如何取全文"的指引**——被截断时若不告诉模型"可以分页取"，
       它会误以为"文档就这么多"（静默降准）；给了指引才能自救。

    ## 幂等（硬要求）
    调用点与 `_format_tool_result` 各调一次 ⇒
    ① 已 ≤ 上限时**直接原样返回**（不重复截）；
    ② note 靠 `_PDF_EXTRACT_TRUNCATED_FLAG` 去重（不重复拼）。
    """
    if not isinstance(output, dict):
        return output

    changed = False
    result = dict(output)

    # ---- 形态一：page 模式（`text` 是整页全文）----
    text = result.get("text")
    if isinstance(text, str) and len(text) > _PDF_EXTRACT_MAX_TOTAL:
        result["text"] = text[:_PDF_EXTRACT_MAX_TOTAL] + "…"
        changed = True

    # ---- 形态二：query 模式（`results[]` 每条的 text）----
    raw_results = result.get("results")
    if isinstance(raw_results, list) and raw_results:
        budget = _PDF_EXTRACT_MAX_TOTAL
        used = 0
        kept: list[Any] = []
        for item in raw_results:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            t = item.get("text") or ""
            remain = budget - used
            if remain <= 0:
                # 额度用尽：**保留条目结构**（id/page 供引用），只清空正文并标记。
                # ⚠️ 直接丢弃整条会让模型看不到"这条存在"（无法判断是否漏了内容），
                #    保留 id/page 则至少能引用、也能据此再精确取。
                kept.append({**item, "text": "（因注入上限未展开）"})
                changed = True
                continue
            if len(t) > remain:
                kept.append({**item, "text": t[:remain] + "…"})
                used = budget
                changed = True
            else:
                kept.append(item)
                used += len(t)
        if changed:
            result["results"] = kept

    if changed:
        base = (result.get("note") or "").strip()
        if _PDF_EXTRACT_TRUNCATED_FLAG not in base:
            tip = (
                f"{_PDF_EXTRACT_TRUNCATED_FLAG} 如需完整内容，请改用更精确的方式："
                "`pdf_extract(page=N)` 取指定页，或用 `retrieve(query=主题词)` 定位片段。"
                "**不要**因此认为『文档里没有这些内容』。"
            )
            result["note"] = f"{tip} {base}".strip()
    return result


def _compact_retrieve_output(output: Any) -> Any:
    """压缩 retrieve 工具结果：保留 id/page/score 供引用溯源，将每段 text 截断到 ≤200 字。

    检索块全文经此压缩后再注入 LLM 上下文，可大幅降低 token 占用，
    避免 AgentLoop 在 loop.py:38 的 8000 token 预算（成本熔断）被单次检索撑爆。
    """
    if not isinstance(output, dict) or "results" not in output:
        return output
    new_results: list[dict] = []
    for r in output.get("results", []):
        if not isinstance(r, dict):
            new_results.append(r)
            continue
        text = r.get("text", "") or ""
        if len(text) > _RETRIEVE_MAX_TEXT:
            text = text[:_RETRIEVE_MAX_TEXT] + "…"
        new_results.append({
            "id": r.get("id"),
            "page": r.get("page"),
            "text": text,
            "score": r.get("score"),
        })
    return {**output, "results": new_results}


def _prepare_tool_output(name: str, output: Any) -> Any:
    """注入前的**内容准备**（当前只有 `retrieve` 走 200 字/段压缩）。

    ★ 结果级去重必须跑在**这一步之后**（2026-09-28 实现中纠正）：
    `retrieve` 注入的是**截断到 200 字**的片段，而 `pdf_extract` 注入的是**全文**。
    若去重在压缩前比对"原文长度"，同一条 chunk 在两边长度相同 ⇒ 会被误判为重复，
    把模型**正要拿到的全文**换成占位 —— 那是"改内容"（丢证据），不是"改分布"。
    去重必须在"将要注入的字符数"这个口径上比较。

    ⚠️ **幂等**：对已压缩过的 `retrieve` 输出重复调用不会二次截断
    （`200 字 + "…"` 再截一次结果完全相同），故调用点与 `_format_tool_result` 里各调一次是安全的。
    """
    if name == "retrieve":
        return _compact_retrieve_output(output)
    if name == "pdf_extract":
        return _compact_pdf_extract_output(output)
    return output


#: 每个 tool_call 的协议开销（角色/字段名等）—— 与 `[上下文构成]` 既有口径保持一致（勿单独改）。
_TOOL_CALL_OVERHEAD = 20


def message_tokens(m: Any) -> int:
    """**单条消息**进上下文要花的 token（估算）—— **全项目唯一入口**。

    ### 为什么必须三样都算（2026-10-02，实测事故）
    此前 `_gate_context` 的增量只算 `sum(_rough_tokens(m.content))`，而实测某一步：
    闸门增量 **1436**（= 两次工具结果的估算 982 + 454），真实增量 **9809** ——
    差额里正包含 assistant 那条消息的 **`tool_calls[].arguments`** 与 **`reasoning`**：
    - `tool_calls` 会随请求回传（OpenAI 协议要求），而 `ChatMessage.content` 不含它；
    - `reasoning_content` 在**带工具调用的轮次必须回传**（`llm/router.py:40`，否则 400）。
    两者都被漏算 ⇒ **闸门这把尺子偏小** ⇒ 该裁时不裁 ⇒ 下一步直接越过上限
    ⇒ 用户侧表现为"**突然熔断**"。而 `[上下文构成]` 日志**早已**计入 `tool_calls`
    ⇒ 同一项目两把尺子 ⇒ 本函数把它们收成一把。

    ⚠️ 全部用 `getattr` 兜底：测试替身（`object.__new__(AgentLoop)` / 精简 ChatMessage）
    可能没有 `tool_calls` / `reasoning`，直接访问会抛 AttributeError 把**本来通过的用例弄红**
    （2026-09-28 实测踩过同类）。
    """
    total = _rough_tokens(getattr(m, "content", "") or "")
    for tc in (getattr(m, "tool_calls", None) or []):
        total += _rough_tokens(str(getattr(tc, "arguments", None) or "")) + _TOOL_CALL_OVERHEAD
    total += _rough_tokens(getattr(m, "reasoning", None) or "")
    return total


def model_injection_truncates(tool: str, text: str | None) -> bool:
    """B2（2026-09-29）：该工具输出的这段文本，**注入模型时会不会被截断**？

    用途：让"引用卡片"能如实告知用户 —— **模型当时可能只看到了片段开头**。
    引用文本取自工具的**完整**输出，而注入的是 `_prepare_tool_output` 处理后的版本 ⇒
    两者**不同源**（"模型可能'引用'它并未看全的段落"）。

    ★ **单一来源**：判据与 `_prepare_tool_output` **完全一致**（同一个工具名分派、同一个
    `_RETRIEVE_MAX_TEXT` 阈值），只问"会不会"、不真截。若在这里另写一份长度判断
    （如按 `output["source"] == "rag"`），就又多了一把尺子 —— 本项目栽过多次的坑。

    当前只有 `retrieve` 走压缩（`pdf_extract` 全文注入 ⇒ 恒 False）。
    """
    return tool == "retrieve" and len(text or "") > _RETRIEVE_MAX_TEXT


def _format_tool_result(name: str, output: Any) -> str:
    """把工具结果序列化并注入 LLM 上下文；保持 `RESULT: {dict}` 形式以便 ast.literal_eval 解析。"""
    return f"RESULT: {_prepare_tool_output(name, output)}"


# ---- 结果级去重（D2，2026-09-28）----
#: 重复命中时保留的"首句"字符数（**不删信息**：仍给出 id / 页码 / 开头片段）
_DEDUP_HEAD_CHARS = 60

def _body_key_from_result(item: dict) -> str:
    """`results` 型输出的**正文键**：优先用 chunk id（跨工具可比），缺失时退化为内容指纹。"""
    rid = item.get("id")
    if rid:
        return f"chunk:{rid}"
    return "text:" + hashlib.sha1(str(item.get("text", "")).encode("utf-8")).hexdigest()[:16]


def _is_key_trackable(output: Any) -> bool:
    """该工具输出能否产出**证据键**（= 是否被结果级去重追踪）。C1（2026-09-28）。

    用途：无进展检测只能建立在"能产出证据键"的工具上。
    - `retrieve` → `{"results": [...]}` ⇒ True（键 = chunk id / 内容指纹）
    - `pdf_extract(page=N)` → `{"text": ..., "page": N}` ⇒ True（键 = 内容指纹）
    - 其它工具（计算器 / 文本处理 / 联网检索等）返回的结构里**没有**这两类字段 ⇒ False

    ⚠️ 返回 False 的**不是"没进展"，而是"判不了"**：调用方必须保守处理（视为有进展），
    否则会把"换用了非检索类工具"误报成原地打转 —— 那是**假阳性**，
    而假阳性一旦接上"中止/换策略"就会伤到正常轮次。
    """
    if not isinstance(output, dict):
        return False
    if isinstance(output.get("results"), list):
        return True
    return "text" in output and output.get("page") is not None


def _dedup_placeholder(kind: str, head: str) -> str:
    """重复命中的占位正文：**明确说明"已在上文提供"**，并保留开头片段供模型辨认。"""
    return (f"【重复命中：该{kind}正文已在本轮上文提供过，此处不再重复注入；"
            f"开头为「{head}…」】")


#: D3（2026-10-02）**允许结果缓存**的工具白名单 —— **只读检索类**。
#: ⚠️ **绝不扩到有副作用的工具**（写长期记忆、改稿写 docx、plan_executor、arxiv 下载等）：
#: 它们第二次调用的"副作用"本身就是意图（如"再记一条"），缓存会让它**静默不生效**。
_CACHEABLE_TOOLS: frozenset[str] = frozenset({"retrieve", "pdf_extract", "citation_lookup"})


def _tool_cache_key(name: str, arguments: Any) -> str:
    """同 args 缓存的键：工具名 + **规范化** args（键序无关，中文不转义）。

    为什么必须规范化：模型每次生成的 JSON 键序可能不同（`{"query":…,"top_k":3}` 与
    `{"top_k":3,"query":…}` 语义相同），不排序就会漏命中 ⇒ 缓存形同虚设。
    """
    try:
        payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = repr(arguments)
    return f"{name}|{payload}"


def _dedup_tool_output(name: str, output: Any, seen: dict[str, int]) -> tuple[Any, dict[str, int], int]:
    """本轮内**结果级去重**：同一份正文只注入一次，重复命中降级为「id/页码/首句」占位。

    ### 为什么做（实测依据）
    单请求的原始命中量可达**语料的 136%**（`docs/去重换策略-影响面评估.md`）——
    `pdf_extract` 12 次调用占 46% 注入量、压缩比 100%。峰值 prompt 实测到 **29,357 / 30,000**，
    中途触发 3 次闸门裁剪（`_gate_context`）⇒ **裁剪即静默丢证据**。
    去重的真实价值**不是省延迟**（省 4.5k tokens ≈ 66ms，可忽略），而是**减少触发裁剪的次数**。

    ### ★ `seen` 的口径：键 → **已注入的正文长度**（不是布尔"见过"）
    这一点是**实现过程中纠正的真缺陷**：同一条 chunk 在 `retrieve` 里被截到 **200 字**、
    在 `pdf_extract` 里是**全文**。若只记"见过一次"，第二次的**全文**会被判成重复而换成占位
    ⇒ 模型失去它**正要取回**的内容 —— 那是"改内容（丢证据）"，不是"改分布"。
    规则：**只有当新出现的正文不比已注入的更长时，才算重复**；更长 ⇒ 照常注入并抬高记录。

    ### 另三条硬约束（缺一即"用质量换 token"）
    ① **集合必须与上下文同步**：闸门 `_gate_context` 会**替换**早期工具结果的正文
       （内容已不在上下文里）。若去重记录仍留着这些键，就会对模型说"已在上文提供"
       —— 而模型看不到 ⇒ **静默降准**。故调用方须在裁剪时撤销对应键。
    ② **去重不删信息**：重复命中仍返回 `id` / `page` / `score` / **首句**。
    ③ **只在本轮内去重**：跨轮历史可能已被 `_compact_history` 压缩过。
    ⚠️ 调用方必须先过 `_prepare_tool_output`（`retrieve` 的 200 字压缩），再调本函数 ——
    否则长度口径与"实际注入的字符数"不一致，①里的误判就会复现。

    ### 返回
    `(可能被改写的 output, 本次**新注入**的 {键: 正文字符数}, 被去重的条数)`。
    重复命中**不登记**（否则撤销语义会错乱）。
    """
    if not isinstance(output, dict):
        return output, {}, 0

    results = output.get("results")
    if isinstance(results, list) and any(isinstance(r, dict) for r in results):
        new_results: list[dict] = []
        new_keys: dict[str, int] = {}
        dropped = 0
        # 本轮已见 ∪ 本次已见：同一份输出内重复出现也应去重
        local_seen = dict(seen)
        for r in results:
            if not isinstance(r, dict) or "text" not in r:
                new_results.append(r)
                continue
            key = _body_key_from_result(r)
            text = r.get("text") or ""
            if key in local_seen and len(text) <= local_seen[key]:
                head = " ".join(text.split())[:_DEDUP_HEAD_CHARS]
                new_results.append({**r, "text": _dedup_placeholder("段落", head)})
                dropped += 1
                continue
            local_seen[key] = len(text)
            new_keys[key] = len(text)
            new_results.append(r)
        if dropped == 0:
            return output, new_keys, 0
        return {**output, "results": new_results}, new_keys, dropped

    # 页模式（`pdf_extract(page=N)`）：返回的是"整页拼接正文"，没有逐块 id。
    # 用**内容指纹**作键 —— 同一页重复取回时指纹一致；不同文档的同号页内容不同 ⇒ 不会误判。
    # 键含内容哈希 ⇒ 同键即同内容，长度必然相同，无需再做长度比较。
    if "text" in output and output.get("page") is not None:
        text = output.get("text") or ""
        if not text.strip():
            return output, {}, 0
        key = "page:" + hashlib.sha1(f"{output.get('page')}|{text}".encode()).hexdigest()[:16]
        if key in seen:
            head = " ".join(text.split())[:_DEDUP_HEAD_CHARS]
            return ({**output, "text": _dedup_placeholder("整页", head)}, {}, 1)
        return output, {key: len(text)}, 0

    return output, {}, 0


# 厂商托管的「服务端执行」工具：由模型厂商在云端完成检索并把结果并入回复正文，
# AgentLoop 不本地执行（否则会因找不到注册工具而失败 / 死循环）。
# 当前仅 web_search 一种；若后续新增托管工具（如 code_interpreter 云端版）在此补充。
HOSTED_TOOLS = {"web_search"}


class AgentLoop:
    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolExecutor,
        config: LoopConfig | None = None,
        system_prompt: str = "",
        last_prompt_tokens: int = 0,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config or LoopConfig()
        self.system_prompt = system_prompt
        self._tool_specs: list[ToolSpecForLLM] = []  # 延迟从 executor.registry 取
        # 成本/trace 精确聚合（由每轮 LLMResponse.cost 累加，供 done 事件回传）
        self._acc_cost: float = 0.0
        self._acc_tokens: int = 0
        # A7（2026-09-28）：隐藏 LLM 调用（历史摘要 / 兜底收尾 / 失败自评）的观测计数。
        # ⚠️ **必须在 `__init__` 里就存在**，不能只靠 `_run_stream_impl` 开头重置：
        # `_summarize_dropped` 可以被**不经 `run_stream`** 的路径直接调用（单测直调、
        # 以及将来的任何内部调用）⇒ 属性不存在会抛 `AttributeError`，
        # 而它被 `except Exception: pass` 吞掉后表现为**"摘要神秘地没生成"**
        # —— 实测就是这样弄红了 `test_history_summary_guard` 与 `test_phase_a` 两条既有用例。
        # 这正是本项目"**观测代码必须能被替身/直调路径容下**"的老教训（替身缺字段 ⇒
        # 弄坏本来通过的用例）。
        self._hidden_llm_calls: int = 0
        self._hidden_llm_wall_ms: float = 0.0
        # A7 统计同源（2026-09-28 第二步）：隐藏调用**自身的**用量（与主调用同一套字段）。
        # 与主调用分开存放是**刻意的**：`_acc_*` 是 `max_cost_per_turn` 熔断的判据，
        # 本次**不动熔断**（隐藏调用多发生在收尾时刻，此刻中止损失的是最终回答）；
        # 两者只在 `_done_event` **汇报时相加** ⇒ 对外数字完整，对内判据不变。
        self._hidden_prompt_tokens: int = 0
        self._hidden_completion_tokens: int = 0
        self._hidden_tokens: int = 0
        self._hidden_cost: float = 0.0
        self._hidden_latency_ms: int = 0
        # ---- C4/C6（2026-09-29）：上下文"被动刀"的可观测计数 ----
        # 此前 `_gate_context`（循环内闸门裁剪）与 `_compact_history`（跨轮压缩）
        # **只有 logger.info，没有任何事件/落库痕迹** ⇒ 用户与 trace 都无法知道
        # "这轮证据被裁过"。而 judge 只测 faithfulness/abstention，**裁剪造成的完整性损失测不出来**
        # （判据盲区）⇒ 必须让它进 trace，成为可核对的事实（只加观测，不改行为）。
        #: 本轮循环内闸门**实际替换**掉的早期工具结果条数（累计）。
        self._gate_hits: int = 0
        #: 本轮跨轮压缩**实际丢弃**的历史消息条数（累计）。
        self._compacted_dropped: int = 0
        #: 最近一次 `_log_context_composition` 的估算值（用于"实测 vs 估算"偏差对照日志）
        self._last_ctx_estimate: int = 0
        #: ★ 2026-10-03（P2）：最近一次的**分项构成**（按来源：system/notes/messages/
        #: tool_results/tool_defs）—— 供「上下文用量」面板显示"哪一块最大"。
        #: 与 `_last_ctx_estimate` 同源同刻（都来自 `_log_context_composition`）。
        self._ctx_breakdown: dict[str, int] = {}
        # ---- 真实 token（2026-09-17 方案 D′）----
        #: 上一次请求 API 返回的**真实** prompt_tokens（跨轮由调用方注入，见 chat_service）。
        #: 用途：**压缩的触发判据** + 循环内闸门的初始真实基线。
        #: ⚠️ 注入值应当是**上一轮首次调用**的真实 prompt（"进入那轮时的上下文规模"），
        #: 而不是上一轮的**峰值** —— 见 `_first_prompt_tokens` 的说明。
        #: 首轮为 0（无历史真实值）→ 退回估算兜底（见 `_prepare_history`）。
        self._last_real_prompt_tokens: int = last_prompt_tokens
        #: 本轮**首次调用**的真实 prompt（每轮重置）。这是"本轮进入时的上下文规模"，
        #: 也是**下一轮跨轮基线的最佳代理**（实测：15,411 → 下一轮首步 15,700，误差 +2%）。
        #: ★ 为什么不能用峰值做跨轮基线（2026-09-28 实测修正）：峰值 = max(各步 prompt)，
        #: 而本轮的工具结果**不会进入下一轮历史** ⇒ 峰值系统性高估。
        #: 实测事故：某一轮峰值 29,357 被回写后，之后每轮都以它为基线（实际首步仅 15,411，
        #: **高估 87%**），导致（a）跨轮压缩每轮都误触发（b）闸门在 step 0 投影必然越线、
        #: **每轮都去裁历史里的长工具结果** —— 由过期数字驱动的静态证据损失。
        self._first_prompt_tokens: int = 0
        #: 本轮所有 LLM 调用中**最大**的单次 prompt（done 事件回传，供展示/落库）。
        #: ⚠️ 口径是**本轮**（每轮重置），**不再**继承注入值 —— 继承会让 `[本轮]` 的
        #: "峰值prompt"显示成上一轮的数字（实测踩到：本轮真实峰值 19,177 却显示 29,357）。
        #: 它**不参与**跨轮压缩判据（那是 `_first_prompt_tokens` 的职责）。
        self._peak_prompt_tokens: int = 0
        # ---- 循环内上下文闸门（2026-09-25 T-01-C）----
        #: `messages` 的"增量起点"：下标之前的内容已被 `_last_real_prompt_tokens` 覆盖。
        #: 在 `_build_messages` 之后、以及每次拿到响应之后更新 —— 这样闸门只对本步**新增**
        #: 的消息做估算（真实基线 + 估算增量），既避免"整段估算"的漂移，也避免把历史重复计入。
        self._gate_mark: int = 0
        # ---- 结果级去重（2026-09-28 D2）----
        #: 本轮**已注入过正文**的键 → **已注入的正文长度**（不是布尔集合，见 `_dedup_tool_output`）。
        #: 每轮起点清空。
        self._text_keys: dict[str, int] = {}
        #: 反向索引 `消息对象 id → 该消息贡献的键`。
        #: 唯一用途：闸门把某条工具结果**替换**成裁剪标记时，同步撤销它贡献的键 ——
        #: 否则去重集合会撒谎（声称"已在上文提供"，而内容已被裁掉）⇒ 静默降准。
        #: 用 `id(msg)` 作键是安全的：消息在列表存活期间对象不会被回收，id 不会被复用。
        self._text_keys_by_msg: dict[int, list[str]] = {}
        # ---- 同 args 结果缓存（2026-10-02 D3）----
        #: `(工具名, 规范化 args) → ToolResult`（**只读检索类工具**，见 `_CACHEABLE_TOOLS`）。
        #: 动机：重复执行要再付一次检索编码 / PDF 解析的时间；内容既已相同，直接复用上一次的结果。
        #:
        #: ★★ **2026-10-03 实测修正：本缓存「实际从未命中」**。
        #: 近 200 条 trace 的 `tool_result` 事件（475 次调用）里 `cached=True` 的数量为 **0**，
        #: 且该字段在落库时**原样保留**（`events_json` 无字段过滤）⇒ 0 命中是**真实空转**，
        #: 不是观测缺失。
        #: ⇒ 原因：**重复只发生在「跨轮」**（同 args 的 `pdf_extract(page=9)` 在 40 条 trace 里出现 5 次，
        #:   但分散在不同轮次），而缓存**每轮重置**（见下方 `_tool_cache = {}`）⇒ 同轮自然撞不上。
        #: ⇒ ⚠️ **不要**因此改成跨轮缓存：跨轮缓存会与「`_compact_history` 压缩」+「结果级去重」
        #:   **三方冲突** ⇒ 出现"缓存说给过、但历史已被压缩掉"⇒ 模型看不到却被告知名场面 ⇒ **静默降准**。
        #: ⇒ 结论：**这条路没有可行收益**，真正该动的是 `retrieve` 的**段数上限**（p50=8 / p90=10）
        #:   与首轮量纲（P1-2），而非缓存。保留本缓存因其零成本，但**别再按"6% 收益"规划**。
        #: ★ **只缓存只读工具**：写记忆 / 改稿 / plan_executor 等有副作用的工具绝不能缓存
        #:   （否则"第二次调用"会静默不生效）。跨轮同样不缓存（见轮起点注释）。
        self._tool_cache: dict[str, ToolResult] = {}
        self._acc_latency: int = 0
        # ---- 耗时分解（2026-09-26 T-05+）----
        #: 本轮**所有 LLM 调用**的墙钟耗时合计（含网络/排队，由 `_think` 与补偿重试累加）。
        #: 与 `_acc_latency` 的区别：后者是**厂商自报的 latency_ms**（不含 SDK 重试、连接建立、
        #: 排队等待），二者之差才是"我们等在了什么上面"。诊断"慢在哪"必须先有这个差值。
        self._acc_llm_wall_ms: float = 0.0
        #: 本轮**所有工具调用**的墙钟耗时合计（由 `_run_one` 累加，含超时等待）。
        #: 与 `_acc_llm_wall_ms` 相加即可与整轮墙钟相减 → 得到"本地开销"（jieba/索引/SSE）。
        self._acc_tool_ms: float = 0.0
        #: 本轮墙钟起点（`run_stream` 开头设置）—— 存为实例属性而非局部变量，是为了让
        #: **中止路径**（`_abort_with_fallback`，它拿不到 `run_stream` 的局部变量）
        #: 也能打出同一条 `[本轮]` 汇总。否则"失败轮次有多慢"仍然不可观测。
        self._turn_t0: float = 0.0
        #: `[本轮]` 汇总是否已打过。存在原因：汇总现在有**两个**触发点
        #: （循环内的正常/中止出口 + `run_stream` 的 `finally`），必须幂等，否则一轮打两条。
        self._summary_logged: bool = False
        # 输入/输出分项（2026-09-16）：用量页需要 pt/ct 分列，而落库的
        # `total_tokens_override` 只能承载一个总数（会把 ct 写成 0）→ 这里分开累计，
        # 由 done 事件一并回传，落库端即可写入精确的 prompt/completion tokens。
        # ⚠️⚠️ **语义是「整轮累计」不是「单次」**：每调用一次 LLM 就累加一次
        # （多步迭代各发一遍累积上下文 → 4 步之和可能是单步的 3~5 倍）。
        # 落库后即 `messages.prompt_tokens` —— 把它读成"单次 prompt"会严重高估上下文规模
        # （2026-09-17 就因此误判过一次）。要看**单次**规模：
        # `grep "上下文构成" data/logs/app.log`。
        self._acc_prompt_tokens: int = 0
        self._acc_completion_tokens: int = 0
        self._steps: int = 0

    async def _build_messages(self, user_input: str, history: list[ChatMessage]) -> list[ChatMessage]:
        """拼装 LLM 输入消息。

        顺序（与多数 chat 模型对 SYSTEM 消息权重的经验一致）：
        history（含「【长期记忆】」SYSTEM 注记） → system_prompt → user_input。

        把 system_prompt 放到 user_input 的紧前位置，是为了压制先前已注入的 SYSTEM 注记
        （例如被毒化的长期记忆把『无文档』当成硬规则），让最新的人类约束生效。
        并对 history 做预算感知压缩（P0-3），避免多轮(S2)丢前文或撑爆 token 预算。
        """
        msgs: list[ChatMessage] = await self._prepare_history(list(history))
        if self.system_prompt:
            msgs.append(ChatMessage(role=Role.SYSTEM, content=self.system_prompt))
        msgs.append(ChatMessage(role=Role.USER, content=user_input))
        return msgs

    async def _prepare_history(self, history: list[ChatMessage]) -> list[ChatMessage]:
        """决定本轮是否压缩历史 —— **优先用真实 token，估算仅作首轮兜底**（方案 D′，2026-09-17）。

        **为什么不用估算做触发判据**：~~`_rough_tokens` 实测偏低约 2 倍**且会漂移**~~
        （~~消息/工具调用越多偏差越大~~；**2026-09-25 T-01 已更正**：实测偏差**按内容类型分档、
        方向相反**——中文偏低 52%、英文偏高 54%，不是"系统性偏低 2 倍"；估算器已按字符类重标，
        相对误差 p50 5.4%）→ 即便如此，拿估算判断"要不要压缩"仍会**引入误差且需自我校准**，
        而压缩的时机错过就补不回来（上下文已经发出去了）。
        反过来，"上一轮 API 返回的真实 prompt_tokens"既准确又**自动适配厂商 tokenizer**。

        ★★ **但"上一轮的哪个 prompt"至关重要**（2026-09-28 实测修正）：
        必须是**上一轮首次调用**的真实 prompt（= "进入那一轮时的上下文规模"），
        **不能是上一轮的峰值**。原因：本轮的工具结果**不会进入下一轮历史**，
        峰值 = max(各步 prompt) 因而系统性高估；而且 `max()` 会让基线**只增不减（棘轮）** ——
        实测事故：某轮峰值 29,357 被回写后，之后每轮首步实际仅 15,411，**高估 87%**，
        于是压缩**每轮都误触发**，闸门也**每轮都去裁历史里的长工具结果**。
        实测最优代理：上一轮首步 15,411 → 下一轮首步 15,700（**误差 +2%**）。

        两条路径：
        - **有真实值**（跨轮注入）：低于触发线 → **直接原样返回**（省掉估算与摘要开销）；
          达到/超过触发线 → 压缩。
        - **无真实值**（首轮，如新会话/进程重启后第一次）：退回 `_compact_history` 的估算判据 ——
          不能因为"没有真实值"就完全不设防。
        """
        if self._last_real_prompt_tokens:
            trigger = self.config.compact_trigger_tokens
            if self._last_real_prompt_tokens < trigger:
                return history
            logger.info(
                "[上下文压缩] 触发（真实口径）：跨轮基线 prompt=%d ≥ 触发线 %d（单次上限 %d 的 70%%）",
                self._last_real_prompt_tokens, trigger, self.config.max_tokens_budget,
            )
            # A5：`triggered=True` 让 `_compact_history` 在"触发了却一条都没丢"时
            # 也打出一条日志 —— 那种情况此前**只能靠日志缺失来推断**。
            # ★ P0（2026-10-03）：同时把**整 prompt 真实基线**传下去，供换算"本次该保留多少历史"。
            #   不传的话 `_compact_history` 只能拿"整 prompt 目标"去比"仅历史估算"
            #   —— 那正是"量纲不一致 ⇒ 永远不丢"的根因。
            return await self._compact_history(
                history, triggered=True, prompt_baseline=self._last_real_prompt_tokens,
            )
        # ★ P0（2026-10-03）首轮兜底：无真实基线 ⇒ `prompt_baseline=0` ⇒
        #  `_history_cap_for` 退回旧判据（历史估算 vs 目标），**首轮行为逐字不变**。
        return await self._compact_history(history, prompt_baseline=0)

    #: 循环内闸门保留最近 N 条消息不动（含当前的 assistant + tool 配对）
    _GATE_KEEP_TAIL = 6
    #: 太短的工具有结果裁了没收益、还会丢信息 → 只裁剪超过此长度的
    _GATE_MIN_ELIDE_CHARS = 400
    #: ③（2026-10-02）**第二阶段**阈值：早期**长回答/技能产物**（assistant）超过此长度才裁。
    #: 刻意高于工具结果（400）：普通问答的中等长度回答不该被裁，这一阶段只针对明显很长的既有产物
    #: （`/精读`·`/综述` 之类动辄数千字符的产物正是本类会话撑爆上下文的主因）。
    _GATE_ELIDE_ANSWER_MIN_CHARS = 800
    #: 占位里保留的回答开头片段长度（字符）—— 让模型能认出"这是哪一轮的产物"。
    _GATE_ANSWER_HEAD_CHARS = 60

    def _gate_context(self, messages: list[ChatMessage]) -> None:
        """**循环内**上下文闸门（T-01-C，2026-09-25）：把"撞上限就中止"提前成"先腾空间再继续"。

        **为什么需要**：`_prepare_history` 只在 `run()` 开头跑一次，而循环内（step 2~12）
        每一步都会追加工具结果 → 此前循环内**没有任何闸门**，唯一防线是
        `resp.prompt_tokens > max_tokens_budget` 时**中止报错**（用户侧就是「本次没有生成有效回答」）。

        **预算口径：真实基线 + 估算增量**
        - 基线用 `self._last_real_prompt_tokens`（厂商上一步返回的**真实** prompt_tokens）
          —— 这是"能不能不等估算"的答案：**整段不估**；
        - 只对本步**新增**消息（`messages[self._gate_mark:]`）用估算。估算器已按字符类标定
          （相对误差 p50≈5%），增量误差远小于"整段估算"。

        **动作：替换内容，不删消息**
        把最旧的工具结果压成一行标记。**删消息会拆散 `assistant.tool_calls` 与 `tool` 结果的配对
        （厂商直接 400）**；替换内容则天然保持配对 —— 这是本设计的关键取舍。
        替换的同时**撤销该内容在结果级去重集合里的键**（见下方 `_text_keys_by_msg`）：
        内容已被裁掉却仍被去重记为"已提供" ⇒ 模型看不到 ⇒ 静默降准。

        已知边界：首轮（`_last_real_prompt_tokens == 0`）不在此动手，交给 `_compact_history` 的估算路径；
        且真实基线来自**上一轮**，与"本步将要发的内容"之间隔着一个用户输入，属保守估计。
        """
        if not self._last_real_prompt_tokens:
            return
        # ★ 2026-10-02：增量走**唯一入口** `message_tokens`（content + tool_calls + reasoning）。
        # 此前只算 `m.content` ⇒ 实测某步增量少算 8373 tokens（预估 1436 / 真实 9809），
        # 其中 assistant 那条消息的 `tool_calls.arguments` 与 `reasoning` 完全没进估算
        # ⇒ 投影偏小 ⇒ 该裁时不裁 ⇒ 下一步直接越过上限（用户侧 = "突然熔断"）。
        delta = sum(message_tokens(m) for m in messages[self._gate_mark:])
        projected = self._last_real_prompt_tokens + delta
        trigger = self.config.compact_trigger_tokens
        if projected < trigger:
            return

        # 第一阶段：早期**工具结果**（既有行为，逐字不变）⇒ 工具结果够用时，第二轮不执行，
        # 既有测试与线上行为**零回归**。
        keep_from = max(0, len(messages) - self._GATE_KEEP_TAIL)
        need = projected - trigger
        elided = saved = 0
        for i in range(min(self._gate_mark, keep_from)):
            m = messages[i]
            if m.role != Role.TOOL or not isinstance(m.content, str):
                continue
            if len(m.content) <= self._GATE_MIN_ELIDE_CHARS:
                continue
            stub = f"[早期工具结果已裁剪：原 {len(m.content)} 字符；如需该内容请重新调用对应工具]"
            saved += max(0, _rough_tokens(m.content) - _rough_tokens(stub))
            # ★ 与结果级去重同步（D2 硬约束①）：本条的正文被替换掉 ⇒ 它贡献的正文键
            # 必须一并撤销。不撤销的话，后面再取回同一条时去重会答"已在上文提供"，
            # 而模型其实看不到内容 —— 那是**静默降准**（指标测不出、用户感知得到）。
            #
            # ⚠️ 全程 `getattr` 兜底（2026-09-28 实测踩到）：`tests/test_token_gate.py` 用
            # `object.__new__(AgentLoop)` 只塞了闸门所需的最小字段，直接访问 `_text_keys*`
            # 会抛 AttributeError，把**本来通过的用例弄红**。
            # **闸门没资格要求替身具备去重状态** —— 替身没有该状态时，撤销本就无事可做。
            _by_msg = getattr(self, "_text_keys_by_msg", None)
            _keys = getattr(self, "_text_keys", None)
            if isinstance(_by_msg, dict) and isinstance(_keys, dict):
                for _k in _by_msg.pop(id(m), ()):
                    _keys.pop(_k, None)
            messages[i] = ChatMessage(role=m.role, content=stub, tool_call_id=m.tool_call_id)
            elided += 1
            if saved >= need:
                break
        # 第二阶段（③，2026-10-02）：工具结果**不够**时，动早期**长回答/技能产物**。
        #
        # ### 为什么需要（实测，见 docs/诊断-离题回答与熔断预警-2026-10-02.md §3.3）
        # 真实会话（session 18）里历史的大头是 `/精读`·`/综述`·`/术语表` 的**长回答**
        # （assistant 消息），而闸门原先**只认工具结果** ⇒ 两次日志都是
        # 「没有「值得裁剪」的早期工具结果」或只省 1269 tokens ⇒ 对本类会话**近乎无效**，
        # 最后表现为"突然熔断"。
        #
        # ### 三条硬约束（与工具结果同款）
        # ① **只替换内容，绝不删消息** —— 删消息会拆散 `assistant.tool_calls ↔ tool` 配对（厂商 400）；
        # ② **保留 `tool_calls` 与 `reasoning`** —— 带工具调用的轮次必须回传 `reasoning_content`
        #    （`llm/router.py:40`），丢掉它同样会 400；
        # ③ **占位里保留开头片段** —— 让模型能认出"这是哪一轮的产物"，而不是一段无来由的省略号。
        #
        # 阈值刻意高于工具结果（800 字符）：普通问答的中等长度回答不该被裁，这一阶段只针对
        # "明显很长的既有产物"。
        if saved < need:
            for i in range(min(self._gate_mark, keep_from)):
                m = messages[i]
                if m.role != Role.ASSISTANT or not isinstance(m.content, str):
                    continue
                if len(m.content) <= self._GATE_ELIDE_ANSWER_MIN_CHARS:
                    continue
                head = " ".join(m.content.split())[:self._GATE_ANSWER_HEAD_CHARS]
                stub = (f"[早期长回答已裁剪：原 {len(m.content)} 字符；"
                        f"开头为「{head}…」。如需细节请重新执行对应命令或检索]")
                saved += max(0, _rough_tokens(m.content) - _rough_tokens(stub))
                messages[i] = ChatMessage(
                    role=m.role, content=stub,
                    tool_calls=getattr(m, "tool_calls", None),
                    reasoning=getattr(m, "reasoning", None),
                )
                elided += 1
                if saved >= need:
                    break
        if elided:
            # C4（2026-09-29）：累计本轮"被裁过"的条数，供 `done` 落 trace。
            # ⚠️ `getattr` 兜底（同下方 `_text_keys*`）：`tests/test_token_gate.py` 用
            # `object.__new__(AgentLoop)` 只塞最小字段 ⇒ 直接访问会抛 AttributeError、
            # 把本来通过的用例弄红。**闸门没资格要求替身具备新计数**。
            self._gate_hits = getattr(self, "_gate_hits", 0) + elided
            logger.info(
                "[上下文闸门] 循环内裁剪 %d 条早期内容（工具结果 >%d 字符 / 长回答 >%d 字符）："
                "投影 %d（真实基线 %d + 增量估 %d）≥ 触发线 %d，省 ≈%d tokens | step=%d",
                elided, self._GATE_MIN_ELIDE_CHARS, self._GATE_ELIDE_ANSWER_MIN_CHARS,
                projected, self._last_real_prompt_tokens, delta, trigger, saved, self._steps,
            )
        else:
            logger.info(
                "[上下文闸门] 投影 %d ≥ 触发线 %d，但没有「值得裁剪」的早期内容"
                "（保留最近 %d 条 / 工具结果需 >%d 字符、回答需 >%d 字符）→ 本轮不动作",
                projected, trigger, self._GATE_KEEP_TAIL,
                self._GATE_MIN_ELIDE_CHARS, self._GATE_ELIDE_ANSWER_MIN_CHARS,
            )

    async def _finalize_without_tools(
        self, messages: list[ChatMessage], guidance: str | None = None
    ) -> str:
        """循环中止/耗尽时的**兜底收尾生成**（2026-09-10 修复「无法生成有效回答」）。

        原先 max_iterations 耗尽或连续无进展时直接发 error 事件返回，**不产出任何回答**——
        上层 `final_text` 为空 → 用户只看到「抱歉，本次没有生成有效回答」。
        用户实测（改稿工作间，主文献 146 chunks、检索已验证可用）连续两次触发该路径，
        表现为「改稿工作间无法生成有效回答」= 功能不可用。

        此处用**已积累的工具结果**强制生成一次最终回答：不提供 tools → 模型无法再调工具
        （从机制上保证收敛），从而任何情况下都给出基于已有证据的实质回复。
        生成失败返回空串，交由上层继续走原有兜底链路（绝不改变既有语义）。

        `guidance`（2026-09-16 新增）：失败自评给出的**改进建议**。传入时作为额外约束
        并入收尾指令，用于「自评判定 NEEDS_IMPROVEMENT 后重生成一次」——
        仍然**不给 tools**，因此即便带着建议重生成也必然收敛。
        """
        instruction = (
            "请基于以上已获得的信息，直接给出最终回答。"
            "不要再调用任何工具，也不要重复检索；若信息不足请说明已找到什么、还缺什么。"
        )
        if guidance:
            # C7-lightcore（2026-09-23）：标记改为引用单一来源 —— 此前测试**自己又写了一遍**
            # 这个字面串（`tests/test_reflect_on_abort.py` 的 `GUIDANCE_MARK`）并用它给假模型
            # 路由返回；改这个词就会让测试**静默走错分支**（仍通过，但判的不是原来那件事）。
            # 注意：本指令的其余文本仍是**内联**的（属"内联提示词暂不搬"的范围，见 prompts 包说明）。
            instruction += f"\n\n{GUIDANCE_MARK}（请针对性修正）：{guidance}"
        try:
            _wall_t0 = time.perf_counter()  # T-05+：收尾生成也要计入耗时分解（它是真实调用）
            resp = await self.llm.chat(
                [*messages, ChatMessage(role=Role.USER, content=instruction)],
                tools=None,
                temperature=0.3,
            )
            _wall_ms = (time.perf_counter() - _wall_t0) * 1000
            # A7：兜底收尾生成**不计入** `_acc_*`（那会改熔断）⇒ 单独记入隐藏累计，
            # 只在 `done` 汇报时相加 ⇒ 用户看到的用量完整、熔断判据不变。
            self._account_hidden(resp, _wall_ms)
            self._acc_llm_wall_ms += _wall_ms
            self._log_llm_timing("finalize", _wall_ms, resp)
            return (resp.message.content or "").strip()
        except Exception:  # noqa: BLE001  兜底生成失败不阻断中止流程
            self._acc_llm_wall_ms += (time.perf_counter() - _wall_t0) * 1000
            logger.warning("兜底收尾生成失败（将继续走空回复兜底）", exc_info=True)
            return ""

    async def _reflect_on_abort(self, messages: list[ChatMessage], draft: str) -> tuple[str, dict | None]:
        """失败轮次的**自评 + 一次改进重生成**（Reflexion 的最便宜形态，2026-09-16 首次接线）。

        `Reflector`（`lightcore/planning/reflector.py`）此前是**写了没用的死代码**（全库无引用）。
        这里给它接上唯一一个"划算"的触发面：**只在循环中止路径**（迭代耗尽 / 连续无进展）。
        那一轮本来就要失败，多花 1~2 次调用换一个更好的收尾是值得的；
        **正常收敛的轮次完全不受影响**（零额外调用）。

        三重护栏（对齐 N11「元任务必须限 max_tokens + 设超时 + 打日志」）：
        ① `max_tokens` 上限 + `asyncio.wait_for` 超时 —— 不让失败轮次变得更慢；
        ② 全程 try/except —— **任何异常都返回原草稿**，绝不阻断收尾；
        ③ 无需额外查成本：循环级熔断在**每轮开头**已保证 `_acc_cost <= max_cost_per_turn`
          （超支那一轮会直接以「成本熔断」返回、走不到中止路径），所以这里不可能超支。

        返回 `(最终文本, 自评摘要或 None)`；摘要用于结构化日志（`satisfied/critique/improvement`）。
        """
        if not self.config.reflect_on_abort or not draft.strip():
            return draft, None
        goal = ""
        for m in reversed(messages):
            if m.role == Role.USER:
                raw = m.content
                goal = raw if isinstance(raw, str) else "（多模态提问）"
                break
        _wall_t0 = time.perf_counter()  # T-05+：自评调用也计入耗时分解（中止轮次常有 1~2 次额外调用）
        try:
            from ..planning.reflector import Reflector

            ref = await asyncio.wait_for(
                Reflector(self.llm).reflect(goal, draft, max_tokens=REFLECT_MAX_TOKENS),
                timeout=REFLECT_TIMEOUT_S,
            )
            _wall_ms = (time.perf_counter() - _wall_t0) * 1000
            # A7：失败自评同理 —— 墙钟计入、tokens/成本不计入 ⇒ 标为隐藏调用。
            # ⚠️ 不记 tokens：`ReflectResult` **没有** token 字段，不为它伪造
            #（返回体不是 `LLMResponse`，与 `_log_llm_timing` 的契约不符）。
            self._hidden_llm_calls += 1
            self._hidden_llm_wall_ms += _wall_ms
            self._acc_llm_wall_ms += _wall_ms
            # 返回体是 `ReflectResult` 而非 `LLMResponse`（没有 tokens/latency 字段）→
            # 只打墙钟，**不复用** `_log_llm_timing`（不为它伪造字段，口径不能靠猜）。
            logger.info("[LLM] mode=reflect 墙钟=%.0fms | satisfied=%s", _wall_ms, ref.satisfied)
        except TimeoutError:
            self._acc_llm_wall_ms += (time.perf_counter() - _wall_t0) * 1000
            logger.warning("失败自评超时（%.0fs），沿用兜底草稿", REFLECT_TIMEOUT_S)
            return draft, None
        except Exception:  # noqa: BLE001  自评失败绝不阻断收尾
            self._acc_llm_wall_ms += (time.perf_counter() - _wall_t0) * 1000
            logger.warning("失败自评异常，沿用兜底草稿", exc_info=True)
            return draft, None

        summary = {"satisfied": ref.satisfied, "critique": ref.critique, "improvement": ref.improvement}
        if ref.satisfied or not ref.improvement.strip():
            logger.info("失败自评：SATISFIED（不重生成）")
            return draft, summary
        logger.info("失败自评：NEEDS_IMPROVEMENT → 带建议重生成一次 | 反思=%s", ref.critique or "-")
        improved = await self._finalize_without_tools(messages, guidance=ref.improvement)
        return (improved or draft), summary

    async def _abort_with_fallback(
        self, messages: list[ChatMessage], reason: str, kind: str = failure.UNKNOWN
    ) -> AsyncIterator[LoopEvent]:
        """中止前的统一收尾：兜底生成 → 失败自评 → 下发 message + done + error。

        「耗尽 ≠ 无回答」的落点：**先产出实质回答，再报中止原因**。
        `reflect_on_abort` 关闭时，行为与 2026-09-10 版**完全一致**（不新增任何调用）。

        A4（2026-09-21）：新增 `kind` —— **由调用方显式标注失败分类**，随 error 事件落库供归因聚合。
        刻意**不用 `reason` 文本反推**：文案是给人看的，改一个字就会让统计静默失配
        （见 `core/failure.py` 的设计说明）。

        C1（2026-09-28）：调用点从 2 个减为 **1 个**（仅"达到最大迭代"）。
        "连续无进展"不再走中止 —— 它的判据原先是死代码，现改为**纯日志观察**，
        待真实触发率明确后再决定动作（中止 vs 换策略）。因此 `failure.NO_PROGRESS`
        当前**没有发出点**（与 `RETRIEVAL_EMPTY` 同属"已定义未使用"，是显式的打点缺口）。
        """
        draft = await self._finalize_without_tools(messages)
        text, reflection = await self._reflect_on_abort(messages, draft)
        if text:
            yield LoopEvent(type="message", state=LoopState.THINK, data={"content": text})
        if reflection is not None and not reflection.get("satisfied"):
            logger.info("失败自评已应用改进建议 | 改进=%s", reflection.get("improvement", "-"))
        # T-05+：中止路径也打耗时汇总 —— "这轮为什么这么慢"在失败场景同样需要答案，
        # 且此处已包含兜底生成/自评的额外调用耗时（它们也计入 `_acc_llm_wall_ms`）。
        # ⚠️ 必须在 `yield` **之前**打：这是 async generator，消费方收到 error 事件后
        # 可能立即 `break`/`aclose()` → yield 之后的代码**不会执行** → 汇总静默丢失。
        self._log_turn_summary()
        # C2（2026-09-28）：中止路径**补发 done**（本轮口径的真实用量），顺序
        # message → done → error。`done` 放在 `error` **之前**是为了让任何消费方都能拿到它
        # —— `error` 是终止信号，消费方可能立即 break/aclose，排在它后面的 yield 会丢失。
        # （路由层另有"保序"设计：它捕获 done 并最后统一发出 ⇒ 前端看到的 done 仍最后一个。）
        # 不补这条的后果：路由层落进"未收到 done"分支 → 用量改用 `count_usage()`（**整段会话**），
        # 与 `[本轮]` 日志、落库估算**三处口径互不相同**（详见 `_done_event`）。
        yield self._done_event(aborted=True, kind=kind)
        yield LoopEvent(
            type="error",
            state=LoopState.ABORT,
            data={"reason": reason, "failure_kind": failure.normalize(kind)},
        )

    # ---- 上下文预算管理（P0-3）----
    async def _compact_history(
        self,
        history: list[ChatMessage],
        *,
        triggered: bool = False,
        prompt_baseline: int = 0,
    ) -> list[ChatMessage]:
        """多轮(S2)上下文压缩：当历史 token 超阈值时，保留置顶 SYSTEM 注记（长期记忆/意图识别），
        丢弃最旧的对话消息；若 LLM 为真实模型，则把被丢弃部分用 LLM 摘要成一条注记，避免丢前文。
        Fake 模型无 key，回退纯滚动截断（fake 摘要无意义，见 model_routing_solution.md）。

        `triggered`（A5，2026-09-28）：本次是否由"**真实 prompt 触发线**"调进来的。
        用于区分两种"没丢消息"：根本没触发 / **触发了但历史其实不大**（空转）。

        ★★ `prompt_baseline`（P0，2026-10-03）：**整 prompt** 的真实基线
        （= 上一轮首步真实 prompt）。用于把"整 prompt 目标"换算成"本次历史保留量"
        （见 `_history_cap_for`）——**这是修复"压缩空转"的关键**：
        此前直接用"整 prompt 目标"（14700）去比"仅历史估算"（11427）⇒ 永远不丢。
        `0` 表示无可用的真实基线（首轮）⇒ 退回旧判据，**首轮行为逐字不变**。
        """
        # ★★ P0-1（2026-10-03 审查修复）：改用**唯一入口** `message_tokens`
        # （content + tool_calls[].arguments + reasoning_content）。
        # 此前只算 `m.content` ⇒ **漏掉工具调用的 arguments 与 reasoning**
        # ⇒ 压缩判据**系统性低估** ⇒ 该丢的不丢 ⇒ 反复靠有损的上下文闸门兜底
        #   （实测：闸门裁剪 30 次 vs 压缩触发 16 次，而 `[历史压缩]` 只有 1 条且是空转）。
        # ⚠️ 2026-10-02 已把 `_gate_context` 改成同一入口，**这次补上压缩侧**（当时漏改）。
        est = sum(message_tokens(m) for m in history)
        # ★ P0：cap 现在是**换算后的历史保留量**（不是配置里那个"整 prompt 目标"）。
        # ⚠️ `_history_cap_for` 定义在 **LoopConfig** 上（口径换算属配置语义），
        #   本方法在 AgentLoop 里 ⇒ 必须经 `self.config` 访问（直写 `self.` 会 AttributeError）。
        cap = self.config._history_cap_for(est, prompt_baseline)
        if est <= cap:
            if triggered:
                # ★ P0（2026-10-03）：口径已统一到**整 prompt**，故这里**不再是"必然空转"**，
                # 而是"整 prompt 基线本身就不高"（或历史确实很小）⇒ 属**正常**结果，
                # 但仍要可见：没有这条日志，事后**只能靠"日志里没有 [历史压缩] 行"来推断**。
                # ⚠️ 历史（A5，2026-09-28）本行曾把此现象记为"两把尺子 ⇒ 审查报告 A5"；
                # 那把尺子**已于 2026-10-03 修掉**，现在的判据是同量纲的。
                logger.info(
                    "[历史压缩] 触发了但未丢弃：历史估算=%d ≤ 保留量=%d"
                    "（整 prompt 基线=%d · 整 prompt 目标=%d；%s）",
                    est, cap, prompt_baseline, self.config.compact_keep_capacity,
                    "同量纲换算" if prompt_baseline else "首轮兜底口径（无真实基线）",
                )
            return history
        # A6（2026-09-28）：排除置顶注记改用 **id 集合**。
        # 原写法 `m not in pinned` 走 dataclass 的 `__eq__` —— 判的是"**是否相等**"，
        # 而不是"**是不是同一条消息**"。若将来出现"内容与注记完全相同"的普通消息会被误排除。
        # （当前 notes 唯一，影响低；但排除判据应保持"同一条"这个语义。）
        pinned_ids = {
            id(m) for m in history
            if m.role == Role.SYSTEM and self._is_pinned_note(m.content)
        }
        pinned = [m for m in history if id(m) in pinned_ids]
        rest = [m for m in history if id(m) not in pinned_ids]
        dropped: list[ChatMessage] = []
        # ★ 同上：口径必须与上面的 `est` **完全一致**，否则"算出来多少"与"丢多少"两套尺子。
        # 保持原有 while 结构（逐条丢到满足 cap 为止），只换估算入口。
        while rest and sum(message_tokens(m) for m in rest) > cap:
            dropped.append(rest.pop(0))
        _summarized = False
        if dropped and not isinstance(self.llm, FakeLLM):
            try:
                summary = await self._summarize_dropped(dropped)
                if summary:
                    pinned.append(ChatMessage(
                        role=Role.SYSTEM,
                        content=f"【历史摘要】以下为较早对话的压缩摘要：{summary}",
                    ))
                    _summarized = True
            except Exception:  # noqa: BLE001 — 摘要失败不应阻断主流程，回退滚动截断
                pass
        # A5（2026-09-28）：**压缩结果必须可见**（丢了几条）。
        # ★ P0（2026-10-03）：口径已统一到**整 prompt** ⇒ 本行改为同时打出
        #   「整 prompt 基线 / 整 prompt 目标 / 换算后的历史保留量」三个数，
        #   这样"为什么只丢了这么几条"可直接从日志读出，不必再心算量纲。
        logger.info(
            "[历史压缩] 历史估算=%d > 保留量=%d ⇒ 丢弃 %d 条 · 摘要=%s · 剩余 %d 条"
            "（整 prompt 基线=%d · 整 prompt 目标=%d，同量纲）",
            est, cap, len(dropped), "是" if _summarized else "否", len(rest),
            prompt_baseline, self.config.compact_keep_capacity,
        )
        # C4（2026-09-29）：累计本轮"被压缩丢掉"的条数，供 `done` 落 trace。
        # ⚠️ `getattr` 兜底：直调本函数（单测 / 内部）的路径可能没有该属性。
        self._compacted_dropped = getattr(self, "_compacted_dropped", 0) + len(dropped)
        return pinned + rest

    def _account_hidden(self, resp: object, wall_ms: float) -> None:
        """把一次**隐藏 LLM 调用**记入隐藏累计（A7，2026-09-28）。

        字段与主路径（`:1370` 附近）**同源同口径**：`prompt/completion/total_tokens`、
        `cost`、`latency_ms`（HTTP 实测）。单独一个入口，避免三个调用点各写一份、
        将来又出现"这个记了那个没记"的缺口。

        ⚠️ 全部用 `getattr` 兜底：响应体不一定是 `LLMResponse`
        （例如某些替身只给 `message`）⇒ **观测代码不能因为缺字段就把主流程弄崩**
        （本轮已因此弄红过两条既有用例，见 P2 实施报告）。
        """
        self._hidden_llm_calls += 1
        self._hidden_llm_wall_ms += wall_ms
        pt = int(getattr(resp, "prompt_tokens", 0) or 0)
        ct = int(getattr(resp, "completion_tokens", 0) or 0)
        self._hidden_prompt_tokens += pt
        self._hidden_completion_tokens += ct
        self._hidden_tokens += int(getattr(resp, "total_tokens", 0) or (pt + ct))
        self._hidden_cost += float(getattr(resp, "cost", 0.0) or 0.0)
        self._hidden_latency_ms += int(getattr(resp, "latency_ms", 0) or 0)

    @staticmethod
    def _is_pinned_note(content: str | None) -> bool:
        # 多模态块消息（list）不参与注记 pin 判断
        return bool(content) and isinstance(content, str) and ("长期记忆" in content or "意图识别" in content)

    async def _summarize_dropped(self, dropped: list[ChatMessage]) -> str:
        """用真实 LLM 把被丢弃的历史压缩为 ≤150 字摘要。

        N11 修复（2026-09-11）：**加 max_tokens 上限 + 超时 + 日志**。
        故障背景（用户实测「多轮上下文无法生成有效回答」）：多轮对话使历史超过压缩阈值后，
        本方法是 `run()` 开头**第一件事**，而它是一次**完整 LLM 调用**且**未限制 max_tokens** ——
        在 thinking 模型（deepseek-v4-flash-vision-exp）下思考链可吃满数千 token、耗时 **70+ 秒**，
        期间**零日志、零用户可见输出** → 用户以为卡死 → 点「停止」→ 表现为「无法生成有效回答」。
        摘要只需 ≤150 字，无需长推理：限制 max_tokens 并设超时兜底（超时即回退滚动截断，
        绝不阻断主流程）。
        """
        text = "\n".join(f"{m.role.value}: {m.text_only()}" for m in dropped)
        # ★★ 2026-10-23：改用**结构化摘要**（借鉴 WorkBuddy 的做法）。
        #   原实现只要求"≤150 字"，实测一个 30 轮的研读对话压完只剩一句话，
        #   具体结论 / 实验数值 / 用户否定过的方向**全丢** ⇒ 下一轮只能重新检索（成本反而更高）✗
        #   现在按 6 段输出，并**保留用户原话**（意图不能改写）。
        prompt = (
            "请把以下多轮对话压缩成**结构化摘要**（中文，≤800 字），严格按下面 6 段输出，"
            "每段没有内容就写「（无）」：\n"
            "## 1. 用户的诉求与约束\n"
            "★ 逐条列出用户**原话**中的要求与限制（不要改写、不要概括成一句）\n"
            "## 2. 已确认的结论\n"
            "含具体数值/公式/页码/文件名；可验证的写清出处\n"
            "## 3. 已被否定 / 排除的方向\n"
            "用户明确不接受的做法，避免之后重复走弯路\n"
            "## 4. 涉及的文件与位置\n"
            "路径 + 关键函数/章节\n"
            "## 5. 未解决的问题\n"
            "## 6. 下一步\n"
            "仅当用户明确说过时填写\n\n"
            "只依据对话内容，不臆造；宁缺勿错。\n\n"
            "=== 对话 ===\n" + text
        )
        _t0 = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                self.llm.chat(
                    [ChatMessage(role=Role.USER, content=prompt)],
                    tools=None,
                    temperature=0.2,
                    max_tokens=SUMMARY_MAX_TOKENS,
                ),
                timeout=SUMMARY_TIMEOUT_S,
            )
        except TimeoutError:
            logger.warning(
                "历史摘要超时（>%.0fs），回退滚动截断（丢弃 %d 条，本次未生成摘要）",
                SUMMARY_TIMEOUT_S, len(dropped),
            )
            return ""
        # A7：历史摘要是**隐藏 LLM 调用** —— 此前它的墙钟**哪里都不记**
        #（既不进 `_acc_llm_wall_ms`，也没有自己的出口）⇒ 只能被"其它"吸收，
        # 而"其它 = 本地开销"这条已写进文档的口径就此失真（摘要实测可达 20s）。
        # ⚠️ 刻意**不**把它加进 `_acc_llm_wall_ms`：那会改变 `[本轮]` 分解里
        # "LLM 墙钟"的既有含义（且 `_summarize_dropped` 可被直调、该属性未必已初始化）。
        # 摘要耗时仍落在"其它"里 —— 但 `[隐藏调用]` 那行会给出它的数值，可自行扣除。
        self._account_hidden(resp, (time.perf_counter() - _t0) * 1000)
        out = (resp.message.content or "").strip()
        logger.info(
            "历史摘要完成：丢弃 %d 条 → 摘要 %d 字，耗时 %.2fs",
            len(dropped), len(out), time.perf_counter() - _t0,
        )
        return out

    async def run(
        self,
        user_input: str,
        history: list[ChatMessage] | None = None,
        *,
        emit: AsyncIterator[LoopEvent] | None = None,  # 仅类型占位，真正流式见 run_stream
    ) -> tuple[str, list[ChatMessage]]:
        """同步式（收集全部事件后返回最终答案）。返回 (final_text, updated_history)。"""
        events: list[LoopEvent] = []
        async for ev in self.run_stream(user_input, history):
            events.append(ev)
        final = next((e for e in reversed(events) if e.type == "message"), None)
        final_text = final.data.get("content", "") if final else ""
        # 重建 history（user + assistant + tool 回环）
        new_history = list(history or [])
        new_history.append(ChatMessage(role=Role.USER, content=user_input))
        for e in events:
            if e.type == "tool_call":
                new_history.append(ChatMessage(
                    role=Role.ASSISTANT, content=e.data.get("thought", ""),
                    tool_calls=[ToolCall(id=e.data["call_id"], name=e.data["name"], arguments=e.data["arguments"])]))
            elif e.type == "tool_result":
                new_history.append(ChatMessage(role=Role.TOOL, content=f"RESULT: {e.data.get('output')}",
                                               tool_call_id=e.data["call_id"]))
            elif e.type == "message" and e.data.get("content"):
                new_history.append(ChatMessage(role=Role.ASSISTANT, content=e.data.get("content", "")))
        return final_text, new_history

    #: 上下文构成日志的打印阈值（**估算** tokens）。低于它不打印 —— 正常小轮次不打，保持信噪比。
    #: 2026-09-17 从 8000 降到 3000：~~`_rough_tokens` 实测偏低约 2 倍~~（**2026-09-25 T-01 更正：
    #: 不是"偏低 2 倍"，而是按内容类型分档、方向相反；估算器已按字符类重标，误差 p50 5.4%**），
    #: 当时 8000 的估算门槛等效于 ~16000 真实 token → 真实 1.5 万以下的轮次全都打不出来，
    #: 门控形同虚设（首次上线时 4 步实测 9292~15015 却一条日志都没有，就是这么来的）。
    #: ⚠️ **T-01 的副作用（未处理，属行为决策）**：估算器校准后 `估算 ≈ 真实`，
    #: 于是 3000 的含义由"真实 ≈6000 起打印"变成"真实 ≈3000 起打印" → **日志比改动前更频繁**。
    #: 若要保留原信噪比，应把阈值调到 ≈6000；**未改，等产品决定**。
    _CTX_LOG_THRESHOLD = 3000

    def _log_context_composition(self, messages: list[ChatMessage]) -> None:
        """按角色统计本轮 messages 规模并落盘（2026-09-17）。

        **为什么需要**：`traces.events_json` 里的 `tool_result` 是截断摘要，
        真实注入 messages 的内容不在任何日志里 → "prompt 为什么这么大"只能靠反推，
        而反推需要正确的量纲，恰恰最容易错（见下）。

        **首次上线即证明了价值**（yolo 工作间，4 步）：实测 prompt 9292 / 10382 / 12602 / 15015，
        而本地估算 4951 / 5514 / 6456 / 7561 —— 偏差稳定在 **1.88~1.99 倍**。
        这条日志把"prompt 为什么这么大"变成可 grep 的事实。

        ⚠️ **口径提醒（曾据此误判两次）**：
        ① `messages.prompt_tokens` 落库的是**整轮累计**（逐次调用累加），不是单次 —— 别拿它当"单次 prompt"；
        ② 本地估算与 API 真实值**不是同一把尺子**，两者相减得到的"缺口"是假象。

        只在合计估算 ≥ `_CTX_LOG_THRESHOLD` 时打印：正常小轮次不打。
        """
        buckets: dict[str, int] = {}
        # ★★ 2026-10-03（P2）：**按「来源」分类**（原先只按角色，无法回答"这轮 prompt 里
        # 哪一块最大"）。分类口径与前端「上下文用量」面板一一对应：
        #   - `system`  系统提示词（**精确匹配** `self.system_prompt`，避免与注记混算）
        #   - `notes`   记忆/意图等 SYSTEM **注记**（其余 system 消息）
        #   - `messages` 对话消息（user + assistant）
        #   - `tool_results` 工具**结果**（role=tool）
        #   - `tool_defs` 工具**定义**（tools schema）—— ⚠️ **不在 messages 里**，
        #     必须单独算；否则用户会看到"工具占了 0%"，而它其实是每一轮都付的固定成本。
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            if role == "system":
                content = getattr(m, "content", "") or ""
                key = "system" if (self.system_prompt and content == self.system_prompt) else "notes"
            elif role in ("tool", "function"):
                key = "tool_results"
            elif role in ("user", "assistant"):
                key = "messages"
            else:
                key = "notes"
            # 单条消息的上下文成本走**唯一入口** `message_tokens`（content + tool_calls + reasoning）。
            # 2026-10-02 归一化：此前本方法单独算 tool_calls、而闸门只算 content ⇒ 两把尺子。
            buckets[key] = buckets.get(key, 0) + message_tokens(m)
        # 工具定义（每轮固定成本）：`specs_for_llm()` 的序列化规模。
        # ⚠️ `getattr` 兜底：替身/直调路径可能没有 registry（见 `message_tokens` 的同类理由）。
        try:
            _specs = self.tools.registry.specs_for_llm() if hasattr(self.tools, "registry") else []
            if _specs:
                import json as _json

                buckets["tool_defs"] = _rough_tokens(
                    _json.dumps(_specs, ensure_ascii=False)
                )
        except Exception:  # noqa: BLE001 — 观测代码不得影响主流程
            pass
        total = sum(buckets.values())
        self._last_ctx_estimate = total
        # ★ 分项**无条件**留存（面板要显示小轮次）；日志仍按阈值打（保持信噪比）。
        self._ctx_breakdown = dict(buckets)
        if total < self._CTX_LOG_THRESHOLD:
            return
        parts = " ".join(f"{k}={v}" for k, v in sorted(buckets.items(), key=lambda x: -x[1]))
        logger.info(
            "[上下文构成] 估算≈%d tokens | %s | 消息 %d 条 | step=%d",
            total, parts, len(messages), self._steps,
        )

    def _log_llm_timing(self, mode: str, wall_ms: float, resp: Any, ttft_ms: int = 0) -> None:
        """LLM 单次调用耗时打点（2026-09-26 T-05+）。**只观测，不改变任何行为**。

        ### 为什么必须有这条日志（实测踩到的观测盲区）
        `loop.py` 此前**完全没有 LLM 耗时打点**：`[上下文构成]` 只记 token 规模，
        而 `_acc_latency` 累加的是 `LLMResponse.latency_ms`，既不打日志也不区分单次。
        后果：一次 20s 的观察里，"慢在哪"**无法从日志回答**（只能靠时序空档推断）。

        ### ⚠️⚠️ 口径（2026-09-26 查证 `router.py:424/541/633` 后**更正过**，勿再写错）
        `resp.latency_ms` **不是厂商回传值**，而是 `OAICompatibleLLM` 自己用
        `time.perf_counter()` 实测的 **HTTP 请求耗时**（源码注释原文："覆盖建连 + 收流全程"）。
        因此它与 `wall` 是**嵌套区间**（`wall ⊇ latency`），不是两个独立来源：
        - `wall`：`_think` 这一侧从"即将调用"到"收完"的全过程；
        - `HTTP 实测`：其中**纯 HTTP 请求**那一段；
        - ⇒ `wall − HTTP` = **LLM 层外层开销**（httpx 客户端构造/复用、payload 组装、
          密钥读取、`_think` 与调用之间的衔接），**不是"网络+排队"**（那个说法是错的）。

        ### 由此：单靠这两个数**分不开"网络"与"模型生成"**，必须看 `ttft`：
        - `ttft`（仅流式有）= 建连 → 服务端**吐出第一个增量**，含建连 + 服务端排队 + prefill；
        - `HTTP 实测 − ttft` = **生成/收流阶段**。
        ⇒ 判据：`ttft` 占大头 ⇒ 排队/限流/服务端忙；`HTTP−ttft` 占大头 ⇒ 模型生成慢或下行带宽。

        ⚠️ **但这条判据只有在"真流式"下才成立**（2026-09-27）。
        若传输把响应**整段缓冲**（`OAICompatibleLLM.STREAM_BUFFERED=True`），
        `ttft_ms` 会恒 ≈ `latency_ms`（实测 12/12 样本，差 1~21ms），"生成段"只是**解析耗时**；
        此时本方法**不打 ttft**，改标「整段缓冲」并只给 `HTTP实测`（那才是用户的真实等待）
        —— **在缓冲传输下"首 token vs 生成"根本不可分，别用日志编造一个分法**。
        传输已改真流式（`client.stream()`），故现在 ttft 是真 TTFT、判据可用；
        只要有人把传输改回缓冲，本方法会自动降级为如实标注（靠 `STREAM_BUFFERED` 声明，不靠数值反推）。

        ⚠️ `prompt`/`completion` 是**单次**（本方法在单次调用后立即记录），
        与落库的 `messages.prompt_tokens`（**整轮累计**）严格不同 —— 别再混。
        """
        http_ms = int(getattr(resp, "latency_ms", 0) or 0)
        # 外层开销：两者是嵌套区间，理论上恒 ≥0；取 max 规避浮点/替身造成的负数假象。
        overhead = f"{max(0.0, wall_ms - http_ms):.0f}ms" if http_ms > 0 else "n/a（未实测）"
        # 是否整段缓冲：由 LLM 后端**声明**（单一来源），不用 `ttft≈latency` 反推 ——
        # 反推在"恰好生成极快"时会把缓冲误判成真流式。
        buffered = bool(getattr(self.llm, "STREAM_BUFFERED", False))
        if mode != "stream":
            ttft = "n/a（非流式无法观测首 token）"
        elif buffered:
            ttft = "n/a（响应整段缓冲 ⇒ 首 token 不可观测；上列 HTTP实测即用户真实等待）"
        elif ttft_ms > 0:
            ttft = f"{ttft_ms}ms（生成段 {max(0, http_ms - ttft_ms)}ms）"
        else:
            ttft = "n/a（流中未出现增量块）"
        # ⚠️ 全程 `getattr` 兜底：**观测代码绝不能反过来弄坏被观测对象**。
        # 实测踩到：`tests/test_lightcore.py` 的 `_RecordingLLM` 替身没有 `model_name` 属性，
        # 直接取 `self.llm.model_name` 会让本来通过的用例变成 AttributeError。
        # LLM 替身只需实现 `chat`/`stream`，日志没有资格要求它更多字段。
        n_calls = len(getattr(getattr(resp, "message", None), "tool_calls", None) or [])
        logger.info(
            "[LLM] mode=%s step=%d 墙钟=%.0fms HTTP实测=%dms 外层开销=%s | ttft=%s | "
            "prompt=%d completion=%d | 工具调用=%d | model=%s",
            mode, self._steps, wall_ms, http_ms, overhead, ttft,
            int(getattr(resp, "prompt_tokens", 0) or 0),
            int(getattr(resp, "completion_tokens", 0) or 0),
            n_calls, getattr(self.llm, "model_name", type(self.llm).__name__),
        )

    def _log_turn_summary(self) -> None:
        """整轮耗时汇总（2026-09-26 T-05+）。**只观测，不改变任何行为**。

        ### 为什么需要这条（真实触发场景）
        用户报「一次提问约 20 秒，而且只检索了两篇文档，是不是太慢了」。
        但当时日志里**只有 token 规模、没有耗时**，只能靠"两个时间戳之间的空档"反推，
        得出"本地 1.3s / LLM 17.3s"这种**推算值**而非实测值。这条汇总把它变成实测。

        分解等式（**可直接用日志验算**）：
            墙钟 ≈ LLM 墙钟合计 + 工具墙钟合计 + 其它
        其中「其它」= jieba 首次初始化 + 索引 `load_state` + SSE 编码 + DB 落库等本地开销。
        ⚠️ 它还包含"事件被消费方（SSE 客户端）读走"的时间 —— 故「其它」偏大时，
        需结合 `data/logs/app.log` 里 `jieba`/`索引缓存` 两条日志（若有）进一步区分。

        两个衍生判据：
        - **LLM 墙钟 ≫ `HTTP实测`合计** ⇒ 开销在 **LLM 层外层**（客户端构造/payload/密钥），
          不在网络也不在模型 —— 这两个数**分不开**网络与生成，那要看每行的 `ttft`；
        - **其它 ≫ 0** ⇒ 本地/流式链路是瓶颈（可能含 `load_state` 或客户端消费慢）。

        ⚠️ 汇总里的 `LLM HTTP实测合计` 来自 `_acc_latency` —— 与 `[LLM]` 行的 `HTTP实测`
        同源同口径（都是 `router.py` 本地实测），**不是厂商回传值**。

        调用点（**三条路径都要有**，否则最需要数字的场景没有数字）：
        ① 正常收敛（`done` 事件前）；② 中止收尾（`_abort_with_fallback`，打在 `yield error` **之前**）；
        ③ **客户端中断** —— 由 `run_stream` 的 `finally` 兜底（用户等不及就换问题时，
        两个出口都到不了，只有 `finally` 会执行）。本方法**幂等**，重复调用只打第一条。
        """
        if not self._turn_t0 or self._summary_logged:
            return  # 未经过 run_stream（如测试直调内部方法）/ 本轮已打过 → 不打
        self._summary_logged = True
        wall_ms = (time.perf_counter() - self._turn_t0) * 1000
        other_ms = wall_ms - self._acc_llm_wall_ms - self._acc_tool_ms
        logger.info(
            "[本轮] steps=%d 墙钟=%.0fms = LLM %.0fms + 工具 %.0fms + 其它 %.0fms | "
            "LLM HTTP实测合计=%dms · 首步prompt=%d（跨轮基线）· 本轮峰值prompt=%d · 累计tokens=%d · 成本=¥%.4f",
            self._steps, wall_ms, self._acc_llm_wall_ms, self._acc_tool_ms, other_ms,
            self._acc_latency, self._first_prompt_tokens, self._peak_prompt_tokens,
            self._acc_tokens, self._acc_cost,
        )
        # A7（2026-09-28）：隐藏 LLM 调用**单独打一行**，而不是并进上一行。
        # 为什么单列：并进去会让已有那行的字段集合变化（下游 grep/脚本靠字段顺序），
        # 且"有/没有隐藏调用"是两种结构完全不同的轮次，混在一行反而看不清。
        # 只在**真的发生过**时打印 —— 正常轮次一次都没有（保持既有信噪比）。
        if self._hidden_llm_calls:
            logger.info(
                "[隐藏调用] %d 次 · 耗时 %.0fms · tokens=%d（prompt %d + completion %d）· 成本=¥%.4f"
                " ⇒ 已计入本轮 `done`（统计同源）；⚠️ **不计入** `max_cost_per_turn` 熔断判据。"
                "注：历史摘要的耗时不进 `LLM 墙钟`，仍落在「其它」里 —— 需扣除时看本行。"
                "`ReflectResult` 无 token 字段 ⇒ 失败自评只计次数与耗时，不为它伪造",
                self._hidden_llm_calls, self._hidden_llm_wall_ms, self._hidden_tokens,
                self._hidden_prompt_tokens, self._hidden_completion_tokens, self._hidden_cost,
            )

    def _done_event(self, *, aborted: bool = False, kind: str | None = None) -> LoopEvent:
        """构造本轮的 `done` 事件 —— **唯一入口**（正常收敛 / 各类中止共用）。

        ### 为什么要有这个入口（C2，2026-09-28）

        修复前 `type="done"` 全文件**只有一处**（正常收敛分支），其余**每一条**提前
        `return` 的路径都不发 `done`：
        ① LLM 调用出错 ② 单次请求上下文超限 ③ 整轮累计 token 超限 ④ 成本熔断
        ⑤ 工具连续失败 ⑥ 兜底收尾（`_abort_with_fallback`）。

        后果是这些轮次在路由层落进"未收到 done"的兜底分支，把用量换成
        `chat_svc.count_usage()`（**整段会话累计**）—— 于是**同一轮的用量三处互不相同**：

        | 位置 | 中止轮显示的值 | 口径 |
        |---|---|---|
        | `[本轮]` 日志 | 真实 | ✅ 本轮 |
        | SSE `done` | `count_usage()` | ❌ 整段会话 |
        | 落库 `messages.*_tokens` | 估算（`pt = user + Σhistory`） | ⚠️ 第三种 |

        而"失败轮次花了多少钱"恰恰是运维最需要准的数字。现在所有出口都走这里
        ⇒ 三处同源（都是本轮的 `_acc_*`），且 `_write_back_prompt_baseline` 也能拿到真实
        `first_prompt_tokens`（此前中止轮永远写不回基线）。

        ### 事件顺序（重要）
        中止路径的产出顺序固定为 **message → done → error**。
        `done` 刻意放在 `error` **之前**：`error` 通常被消费方当作**终止信号**
        （可能立即 `break()` / `aclose()`），若 `done` 排在它之后就会被静默丢失。

        ⚠️ 别把这条与**路由层**的 SSE 顺序混为一谈（曾误判）：`routes_search.gen`
        对 `done` 有**保序**设计 —— 它把 done **捕获**下来，在 `message` / `verify` 之后
        **最后统一发出**（`error` 反而是即时透传的）。所以前端最终看到的
        `done` 仍是最后一个事件，与下面这条顺序不冲突。

        `aborted=True` 时额外带 `aborted` / `failure_kind`，供路由层识别中止轮并写结构化 notice。
        """
        # ---- A7 统计同源（2026-09-28）：`done` 汇报**完整**用量（主调用 + 隐藏调用）----
        # 修前：隐藏调用（历史摘要 / 兜底收尾 / 失败自评）的 tokens 与成本
        # **完全不出现在任何对外的数字里** ⇒ 用量页、落库、`[本轮]` 的 `成本=` 全都系统性偏低。
        # 这里**只在汇报时相加**，不动 `self._acc_*` ⇒ `max_cost_per_turn` 熔断判据**不变**
        # （为什么不动：隐藏调用多发生在**收尾时刻**，此刻中止损失的是最终回答 ——
        #  为"数字正确"牺牲产出，不该默认接受；另见 A7 影响面评估）。
        data: dict[str, Any] = {
            "steps": self._steps,
            "total_tokens": self._acc_tokens + self._hidden_tokens,
            # 分项用量（2026-09-16）：供落库写入精确的 prompt/completion tokens，
            # 避免用量页只能拿到一个总数（旧版 `total_tokens_override` 会把 ct 写成 0）。
            "prompt_tokens": self._acc_prompt_tokens + self._hidden_prompt_tokens,
            "completion_tokens": self._acc_completion_tokens + self._hidden_completion_tokens,
            # 本轮**最大单次** prompt（方案 D′）：与上面 `prompt_tokens`（整轮**累计**）
            # 严格区分。**仅用于展示/落库**，不作为下一轮的压缩判据
            # （`last_prompt_tokens` 由 `first_prompt_tokens` 回写，见 2026-09-28 修正）。
            "peak_prompt_tokens": self._peak_prompt_tokens,
            # ★ P1-a（2026-10-03）：**分母**。`peak_prompt_tokens` 单看没有意义
            #（"12,537 tokens" 是多还是少？用户无从判断），必须配上单次上限才知道**离熔断多远**。
            # 市面 Claude Code / Cursor 均提供 percent-used 指示器；本项目此前只有
            # **后端日志**（`[上下文构成]`），用户侧完全看不到 ⇒ 真实事故过「**突然熔断**」。
            # ⚠️ 稳定 schema：前端缺该字段时应能优雅降级（不画条），不得报错。
            "max_tokens_budget": self.config.max_tokens_budget,
            # ★ P1-a：**警示线**也一并下发（而不是让前端自己乘 0.7）——
            # 前端复制派生逻辑 = 再造一把尺子（后端改了前端不改 ⇒ 警示时机错位）。
            # 语义同 `compact_trigger_tokens`：超过它即"该压缩了"，用户应看到警示色。
            "compact_trigger_tokens": self.config.compact_trigger_tokens,
            # ★ 2026-10-03（P2）：**分项构成**（按来源：system/notes/messages/
            # tool_results/tool_defs）—— 供前端「上下文用量」面板显示"哪一块最大"
            # （对齐 WorkBuddy 的分项列表）。⚠️ 稳定 schema：未跑过 step 时为空 dict，
            # 前端据此不渲染面板（优雅降级）。
            "context_breakdown": getattr(self, "_ctx_breakdown", None) or {},
            # 本轮**首次调用**的真实 prompt：**跨轮基线的唯一来源**（落 session config
            # → 下一轮注入 → 压缩触发 + 闸门初始基线）。用首步而不用峰值，是因为
            # 本轮的工具结果不会进入下一轮历史 ⇒ 峰值系统性高估（实测 29,357 vs 15,411）。
            "first_prompt_tokens": self._first_prompt_tokens,
            "cost": round(self._acc_cost + self._hidden_cost, 6),
            "latency_ms": self._acc_latency + self._hidden_latency_ms,
            # A7 分项（**稳定 schema**：正常轮次为 0，便于"这轮有没有隐藏调用"可核对）:
            "_hidden_calls": self._hidden_llm_calls,
            "_hidden_tokens": self._hidden_tokens,
            "_hidden_cost": round(self._hidden_cost, 6),
            # ---- C4/C6（2026-09-29）：上下文"被动刀"的可观测 ----
            # 闸门裁剪 / 历史压缩此前**只写日志、不进事件与 trace** ⇒ 用户与事后分析都无法知道
            # "这轮证据被裁过"。而 judge 只测 faithfulness/abstention，**裁剪造成的完整性损失
            # 测不出来**（判据盲区）⇒ 让它进 `done`（随之落 trace），成为可核对的事实。
            # ⚠️ 同样**稳定 schema**：未发生裁剪时为 0；`getattr` 兜底以容下替身/直调。
            "_gate_hits": getattr(self, "_gate_hits", 0),
            "_compacted_dropped": getattr(self, "_compacted_dropped", 0),
        }
        if aborted:
            data["aborted"] = True
            data["failure_kind"] = failure.normalize(kind)
        return LoopEvent(
            type="done",
            state=LoopState.ABORT if aborted else LoopState.FINISH,
            data=data,
        )

    async def _think(
        self,
        messages: list[ChatMessage],
        tool_specs: list,
        stream_tokens: bool,
        tool_choice: str | dict | None = None,
    ) -> tuple[Any, list[LoopEvent]]:
        """一轮 THINK。返回 (LLMResponse | None, 本轮回显事件列表)。

        - 非流式：调用 llm.chat，若无工具调用则补发完整 message 事件。
        - 流式：调用 llm.stream，逐 delta 发增量 message 事件；无工具时再补发完整
          message 事件供 history 重建；有工具时仅回传 tool_calls 给 ACT 阶段。
        - LLM 调用失败（key 无效/网络/限流 401·429·5xx）时返回 (None, [error 事件])，
          由 run_stream 透传为 SSE error 帧，避免流断裂导致前端只看到「回复失败」。
        - tool_choice：可选 'auto' / 'required' / {"type":"function","function":{"name":"..."}}
          用于强制 LLM 至少一次工具调用；当 LLM 因上下文偏见拒绝调工具时尤其有用。
        """
        events: list[LoopEvent] = []
        # 仅对真实 OAICompatibleLLM 透传 web_search（FakeLLM 等本地替身不接收该形参，否则 TypeError）。
        # 托管联网搜索（web_search）由厂商服务端执行，不进入本地工具循环。
        ws_kwargs: dict = {}
        if isinstance(self.llm, OAICompatibleLLM) and self.config.web_search:
            ws_kwargs["web_search"] = True
        # 结构化上下文日志（2026-09-17）：把"prompt 为何这么大"变成可 grep 的事实。
        # 放在**调用前**，这样即使调用失败（超限/402/超时）也能看到当时的上下文构成。
        self._log_context_composition(messages)
        # T-05+：墙钟起点放在**调用前**（紧接上下文日志），覆盖连接建立 + SDK 重试 + 排队，
        # 这才是用户实际等待的时间；与响应里的厂商 latency_ms 之差 = 网络/排队开销。
        _wall_t0 = time.perf_counter()
        _llm_mode = "stream" if stream_tokens else "chat"
        _ttft_ms = 0  # T-05+：仅流式路径可观测首 token 延迟（非流式保持 0）
        try:
            if stream_tokens:
                content_parts: list[str] = []
                reasoning_parts: list[str] = []  # V4 思考模式：reasoning_content 增量
                tool_acc: dict[str, dict[str, Any]] = {}
                pt = ct = 0
                stream_latency = 0
                stream_cost = 0.0
                async for chunk in self.llm.stream(
                    messages, tool_specs, temperature=self.config.temperature,
                    tool_choice=tool_choice, **ws_kwargs,
                ):
                    if chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            tool_acc[tc.id] = {"name": tc.name, "arguments": tc.arguments}
                    if chunk.reasoning:
                        # 思考增量 → thinking 事件（前端折叠展示推理链）
                        reasoning_parts.append(chunk.reasoning)
                        events.append(LoopEvent(type="thinking", state=LoopState.THINK, data={"reasoning": chunk.reasoning}))
                    if chunk.delta:
                        content_parts.append(chunk.delta)
                        events.append(LoopEvent(type="message", state=LoopState.THINK, data={"delta": chunk.delta}))
                    if chunk.finished:
                        pt, ct = chunk.prompt_tokens, chunk.completion_tokens
                        stream_latency = chunk.latency_ms
                        stream_cost = chunk.cost
                        # T-05+：首 token 延迟（仅流式有）。用 getattr 兜底 —— 替身
                        # （FakeLLM / 测试 double）产出的 chunk 可能没有该字段。
                        _ttft_ms = int(getattr(chunk, "ttft_ms", 0) or 0)
                reasoning = "".join(reasoning_parts)
                if pt == 0 and ct == 0:
                    # 厂商未下发 usage（端点不支持 include_usage 等）→ **本地估算兜底**。
                    # 为什么必须兜底而不是留 0：`_acc_tokens`/`_acc_cost` 为 0 会让
                    # token 预算熔断与单轮成本熔断**永远不触发**（护栏变装饰）。
                    # 估算口径与 `chat_service._rough_tokens` 同源（base._rough_tokens），
                    # 仅用于"防失控"，不作为账单依据；厂商给出真实 usage 时优先用真实值。
                    pt = sum(_rough_tokens(m.content) for m in messages)
                    ct = _rough_tokens("".join(content_parts))
                    stream_cost = estimate_cost(self.llm.model_name, pt, ct)
                    logger.warning(
                        "流式响应未含 usage，已按本地估算计入用量（model=%s, pt=%d, ct=%d）"
                        "—— 估算仅用于成本熔断，不作为账单依据",
                        self.llm.model_name, pt, ct,
                    )
                if tool_acc:
                    msg = ChatMessage(
                        role=Role.ASSISTANT, content="".join(content_parts),
                        tool_calls=[ToolCall(id=k, name=v["name"], arguments=v["arguments"]) for k, v in tool_acc.items()],
                        reasoning=reasoning,
                    )
                else:
                    msg = ChatMessage(role=Role.ASSISTANT, content="".join(content_parts), reasoning=reasoning)
                    events.append(LoopEvent(type="message", state=LoopState.FINISH, data={"content": msg.content}))
                resp = LLMResponse(
                    message=msg, prompt_tokens=pt, completion_tokens=ct, model=self.llm.model_name,
                    # 2026-09-16：流式路径此前**不传** cost/latency_ms → `_acc_cost`/`_acc_latency`
                    # 恒 0，Trace 的耗时与成本几乎全为 0、成本熔断不生效（见体检报告）。
                    latency_ms=stream_latency, cost=stream_cost,
                )
            else:
                resp = await self.llm.chat(
                    messages, tool_specs, temperature=self.config.temperature,
                    tool_choice=tool_choice, **ws_kwargs,
                )
                if not resp.message.tool_calls:
                    events.append(LoopEvent(type="message", state=LoopState.FINISH, data={"content": resp.message.content}))
        except Exception as e:  # noqa: BLE001  # LLM 调用失败：透传为 error 事件，避免 SSE 流断裂
            # T-05+：失败路径同样记墙钟 —— "请求发了多久才失败"是区分
            # 「超时/限流（等很久才报错）」与「鉴权/参数错（立即报错）」的关键，
            # 而这条信息此前完全缺失（只看到 error 文案，看不到耗时）。
            _fail_ms = (time.perf_counter() - _wall_t0) * 1000
            self._acc_llm_wall_ms += _fail_ms
            logger.warning(
                "[LLM] mode=%s step=%d 墙钟=%.0fms **调用失败**：%s：%s",
                _llm_mode, self._steps, _fail_ms, type(e).__name__, str(e)[:200],
            )
            return None, [LoopEvent(
                type="error", state=LoopState.ERROR_HANDLE,
                data={
                    "reason": f"LLM 调用失败：{type(e).__name__}：{str(e)[:300]}",
                    "failure_kind": failure.LLM_ERROR,  # A4
                },
            )]
        _wall_ms = (time.perf_counter() - _wall_t0) * 1000
        self._acc_llm_wall_ms += _wall_ms
        self._log_llm_timing(_llm_mode, _wall_ms, resp, _ttft_ms)
        return resp, events

    # ---- 工具权限查询（P1-5 并行分组用）----
    def _permission_of(self, name: str):
        try:
            return self.tools.registry.get(name).permission
        except KeyError:
            return Permission.ADMIN  # 未知工具按最高权限，走串行

    async def _run_one(self, tc: Any, thought: str) -> tuple[ToolResult, list[LoopEvent], float]:
        """执行单个工具并返回 (结果, 待回显事件列表, 墙钟毫秒)。不在此处 yield，交给调用方按序回显。

        T-05+（2026-09-26）：返回值增加**墙钟耗时**（原来只有 (结果, 事件)）。
        为什么不让本方法自己打日志：工具日志最有价值的一列是「**实际注入上下文多少字符**」，
        而注入发生在调用方（`_format_tool_result` + `messages.append`）——
        在这里拿不到，分开打会变成两行、难 grep。故把耗时透传给调用方，**一次调用一行**。

        **单次执行超时**（2026-09-16）：`config.tool_timeout_s`（默认 120s）兜住"工具挂死"。
        此前循环层完全没有超时 —— 某个工具不返回会让整轮无限等待，用户只能点「停止」脱身。

        超时的语义选择：**算作一次工具失败**（`ok=False`），与项目既有"系统故障 ≠ 空结果"口径一致 ——
        超时是系统侧异常，若伪装成"空结果"会让模型以为"文档里没有"。
        会计入 `max_tool_failures`（连续 3 次则中止整轮，与其它工具失败同一阈值），
        但**错误文案里带工具名与上限**，模型可据此换参数/换工具重试，用户也能一眼看出卡在哪。

        ⚠️ 已知局限：handler 若把阻塞工作交给 `asyncio.to_thread`，超时只能让**循环不再等待**，
        线程本身无法被强制杀死（Python 无此能力）—— 但它已不再阻塞对话。
        ⚠️ 只捕获 `TimeoutError`：`CancelledError` 属 `BaseException`，必须原样向上传播，
        否则「客户端停止」会被误记成"工具失败"。
        """
        events: list[LoopEvent] = [LoopEvent(
            type="tool_call", state=LoopState.ACT,
            data={"call_id": tc.id, "name": tc.name, "arguments": tc.arguments, "thought": thought},
        )]
        # T-05+：墙钟覆盖"等待工具返回"的完整时间。⚠️ 起点必须在 `wait_for` **之前** ——
        # 若放在里面，超时路径（最需要被观测的那条）就拿不到耗时。
        _wall_t0 = time.perf_counter()
        # T-09+（2026-09-30）：工具失败也要**显式归因** —— 此前失败事件只有 `error` 文本、
        # 没有 `failure_kind` ⇒ metrics 无法分桶（"超时"与"参数错"混在同一个桶里，
        # 而二者的自救方向完全不同）。★ 分类在**产生处**标注，不靠事后解析文案。
        _fail_kind = failure.TOOL_ERROR
        # D3（2026-10-02）：同 args 结果复用 —— **在执行之前**判断，命中则完全跳过执行
        # （省下一次检索编码 / PDF 解析；内容与上次逐字相同，故对模型无差异）。
        # ⚠️ 只在白名单内、且只复用**成功**结果；失败结果一律重试（失败可能瞬时）。
        _cache_key = _tool_cache_key(tc.name, tc.arguments) if tc.name in _CACHEABLE_TOOLS else ""
        _cached = self._tool_cache.get(_cache_key) if _cache_key else None
        if _cached is not None:
            result = _cached
            _wall_ms = 0.0
            events.append(LoopEvent(
                type="tool_result", state=LoopState.ACT,
                data={"call_id": tc.id, "name": tc.name, "ok": True,
                      "output": result.output, "cached": True},
            ))
            return result, events, _wall_ms
        try:
            result = await asyncio.wait_for(
                self.tools.execute(tc.name, tc.arguments),
                timeout=self.config.tool_timeout_s,
            )
        except TimeoutError:
            limit = self.config.tool_timeout_s
            logger.warning("工具 %s 执行超过 %.0fs 未返回，已中止本次调用", tc.name, limit)
            _fail_kind = failure.TIMEOUT
            result = ToolResult(
                ok=False, output=None,
                error=f"tool_timeout: 工具 '{tc.name}' 执行超过 {limit:.0f}s 未返回，本次调用已中止",
            )
        _wall_ms = (time.perf_counter() - _wall_t0) * 1000
        # ⚠️ **不在这里累加 `_acc_tool_ms`**（2026-09-27 实测修正）：并行分组下多个 `_run_one`
        # 由 `asyncio.gather` **同时**执行，各自把墙钟累加会让同一段时间被计 N 次
        # ⇒ 汇总出现 `其它` 为**负数**（实测 `墙钟=12714ms = LLM 9537 + 工具 5250 + 其它 -2073`）。
        # 正确做法：由调用方测量**整个 ACT 阶段**的墙钟（见 `_run_stream_impl`），
        # 本方法的返回值只用于 `[工具]` 行的**单次**耗时展示。
        if not result.ok:
            events.append(LoopEvent(
                type="tool_result", state=LoopState.ERROR_HANDLE,
                # ★ F7 修复（2026-09-28）：**失败事件也必须带 `output`**。
                #
                # 原先只放 `{call_id, ok, error}` —— 于是所有"靠 `output` 判定失败种类"的
                # 消费方**全部够不到**，其中两处是实打实的缺陷（实测证据见
                # `docs/上下文预算-C3实施报告.md` §5）：
                #   ① **路由层 F7「检索服务暂时不可用」探测**读的是 `output["error"]`
                #      ⇒ `_retrieval_error` 恒 False ⇒ 索引损坏时用户看到的是通用兜底
                #      「可能是模型调用失败…」，**指向了错误的自救方向**
                #      （该去重建索引，却被引导去查余额/换模型）；
                #   ② `run()`（非流式，CLI 用）**从事件重建 history** 时用 `e.data["output"]`
                #      ⇒ 失败工具在历史里变成 `RESULT: None`；而 `run_stream` 恰恰是注入
                #      真实 output 的 ⇒ **两个 API 对同一件事给出了不同的上下文**。
                #
                # ⚠️ 与成功事件**同形**（都带 `output` 键）：形状不一致正是下一处"够不到"的温床。
                #
                # ### 影响面（已逐个核对，见本轮实施报告）
                # - `run_stream`（生产路径）**不改**：它本来就把 `output` 注入 messages
                #   （`_format_tool_result` 对 ok / not-ok **一视同仁**），此处只是把它也放进事件；
                # - `run()`（CLI 路径）**会改**：失败工具从 `RESULT: None` 变为 `RESULT: {output}`
                #   ⇒ 这是**与 run_stream 对齐**的一致性修复，不是新行为；
                # - 前端 `ToolCallCard` 渲染 `result ?? error`：`output` 为 `None` 时
                #   `??` 只对 null/undefined 生效 ⇒ **仍回落到 `error`**，无回归；
                # - `chat_service` 工具记录 / `routes_traces` 候选收集：均已用 `.get(...) or ...` 兜底。
                data={"call_id": tc.id, "name": tc.name, "ok": False, "error": result.error,
                      "output": result.output,
                      # ★ T-09+：失败事件也带归因分类（与成功事件**同形**，避免"够不到"）
                      "failure_kind": _fail_kind},
            ))
        else:
            events.append(LoopEvent(
                type="tool_result", state=LoopState.OBSERVE,
                # B2（2026-09-29）：带上**工具名**。消费方（引用卡片）需要它来问
                # `model_injection_truncates(name, text)` —— 若不给名字，消费方只能改用
                # `output["source"]=="rag"` 之类的**近似判据**，那就与 `_prepare_tool_output`
                # 的判据（按 `name == "retrieve"`）**不同源**（"两把尺子"）。
                # ⚠️ 与失败分支**同形**（两边都带 `name`）。
                data={"call_id": tc.id, "name": tc.name, "ok": True, "output": result.output},
            ))
            # D3（2026-10-02）：成功结果入缓存（供本轮内**同 args** 的后续调用复用）。
            # 只缓存白名单工具；失败/超时**不入缓存**（瞬时故障重试应真的重试）。
            if _cache_key:
                self._tool_cache[_cache_key] = result
        return result, events, _wall_ms

    async def run_stream(
        self,
        user_input: str,
        history: list[ChatMessage] | None = None,
        *,
        stream_tokens: bool = False,
        force_tool_choice: str | dict | None = None,
    ) -> AsyncIterator[LoopEvent]:
        """运行 AgentLoop 并产出 SSE 事件。

        force_tool_choice：可选，若设置则在「第一轮 THINK」强制 LLM 调用工具（'required' / 强制工具名）。
        设计动机：即便已清理 prompt 系统，LLM 仍可能因历史上下文中大量『无文档』拒接记录
        而拒绝调 retrieve，导致用户拿不到真实 PDF 内容。把『必须先用 retrieve』变成
        协议级硬约束（OpenAI 兼容 tool_choice），从架构上消除该类幻觉拒接
        （见 2026-08-19 三续）。

        T-05+（2026-09-27）：本方法现在是**薄包装**，只为加一个 `finally` ——
        真正的循环在 `_run_stream_impl`。原因：原来 `[本轮]` 汇总只写在「正常收敛」与
        「`_abort_with_fallback`」两个出口，**客户端中断（用户等不及就换问题）会直接
        `aclose()` 本生成器，两个出口都到不了** ⇒ 汇总静默丢失。
        实测踩到：一次 **9 步 / 23 秒**的请求被中断，全程**没有一行 `[本轮]`** ——
        而"被中断的那次有多慢"恰恰是最需要数字的场景。
        """
        impl = self._run_stream_impl(
            user_input, history, stream_tokens=stream_tokens, force_tool_choice=force_tool_choice,
        )
        try:
            async for ev in impl:
                yield ev
        finally:
            # 显式关闭内层生成器：否则中断时内层要等 GC 才收尾，还可能留下
            # "async generator ignored GeneratorExit" 噪声。`aclose()` 幂等，重复调用安全。
            await impl.aclose()
            # 汇总自带幂等（`_summary_logged`），故正常收敛时不会重复打。
            self._log_turn_summary()

    async def _run_stream_impl(
        self,
        user_input: str,
        history: list[ChatMessage] | None = None,
        *,
        stream_tokens: bool = False,
        force_tool_choice: str | dict | None = None,
    ) -> AsyncIterator[LoopEvent]:
        """`run_stream` 的实际实现（**不要直接调用** —— 会绕过耗时汇总的 `finally`）。"""
        history = history or []
        # T-05+（2026-09-26）：本轮耗时分解的起点。分解口径：
        #   整轮墙钟 = LLM 墙钟合计 + 工具墙钟合计 + 「其它」（jieba/索引加载/SSE 编码/DB 落库）
        # 「其它」大 ⇒ 本地开销；LLM 墙钟 ≫ 厂商延迟 ⇒ 网络/排队。
        # 这里**重置**累计量：`AgentLoop` 实例虽多为每请求新建，但重置能保证被复用时汇总仍是"本轮"。
        _turn_t0 = time.perf_counter()
        self._turn_t0 = _turn_t0
        self._acc_llm_wall_ms = 0.0
        self._acc_tool_ms = 0.0
        # A7（2026-09-28）：**隐藏 LLM 调用**的观测计数。
        # "隐藏" = 发生在循环保护与 `done` 口径之外的调用：
        #   `_summarize_dropped`（历史摘要）· `_finalize_without_tools`（中止兜底生成）·
        #   `_reflect_on_abort`（失败自评，经 Reflector）· 以及 RAG 侧的 multi_query 改写。
        # 它们**不计入 `_acc_tokens`/`_acc_cost`** ⇒ `max_cost_per_turn` 熔断被**系统性低估**。
        # 本轮只做**观测**（不并入累计 —— 那会改变熔断触发点，属行为变更，另议）；
        # 只记**次数与墙钟**，**不记 tokens**：`Reflector` 返回体没有 token 字段，
        # 不为它伪造（本项目"口径不能靠猜"的一贯做法）。
        self._hidden_llm_calls = 0
        self._hidden_llm_wall_ms = 0.0
        self._hidden_prompt_tokens = 0
        self._hidden_completion_tokens = 0
        self._hidden_tokens = 0
        self._hidden_cost = 0.0
        self._hidden_latency_ms = 0
        # C4/C6（2026-09-29）：本轮口径重置（同 `_hidden_*` 的理由——语义都是"本轮"）。
        self._gate_hits = 0
        self._compacted_dropped = 0
        self._summary_logged = False
        # 本轮口径重置（2026-09-28）：`峰值prompt` 与 `首步prompt` 都只描述**本轮**。
        # 此前 `_peak_prompt_tokens` 继承注入值，使 `[本轮]` 打出上一轮的数字（实测误导）。
        self._peak_prompt_tokens = 0
        self._first_prompt_tokens = 0
        # D2（2026-09-28）：去重范围 = **本轮**，故在本轮起点清空。
        # 为什么不能跨轮：跨轮历史可能已被 `_compact_history` 压缩过，拿它当"已在上下文"
        # 会撒谎（与闸门撤销要解决的是同一类问题）。这里早于 `_build_messages`，
        # 因此历史压缩不会往集合里塞任何键。
        self._text_keys = {}
        self._text_keys_by_msg = {}
        # D3（2026-10-02）：**同 args 结果缓存**的范围同样是"本轮"。
        # 为什么不能跨轮：跨轮历史可能被 `_compact_history` 压缩、且"模型换角度重取"本身是
        # 有效行为，跨轮复用会把它掩盖掉；只读工具的结果在本轮内才是稳定可复用的。
        self._tool_cache = {}
        messages = await self._build_messages(user_input, history)
        # T-01-C：增量起点 = 当前长度。此刻 `messages` 的内容已大致被上一轮的真实峰值覆盖
        # （D′ 的既有近似），故闸门只对本步之后**新增**的消息做估算。
        self._gate_mark = len(messages)
        tool_specs = self.tools.registry.specs_for_llm() if hasattr(self.tools, "registry") else []

        tokens_used = 0
        tool_failures = 0
        # C1（2026-09-28 修判据；2026-10-12 观察期收口）：连续"未获得新证据"的步数。
        # 判据已从"TOOL 消息条数"改为"新证据键"；跨越阈值时**注入换策略注记**（仍不中止）。
        # （原 `last_tool_count` 已删除：它是那份永不触发的判据的载体。）
        no_progress_steps = 0

        for step in range(self.config.max_iterations):
            # ---- THINK ----
            yield LoopEvent(type="thinking", state=LoopState.THINK, data={"step": step})
            # 仅第一轮强制工具调用；之后的轮次让 LLM 自主判断（如 summarize retrieve 结果）
            # V4 思考模式：官方不支持 tool_choice（400）→ 不强制（模型仍可自主调用工具）
            choice_now = None if self.config.thinking else (force_tool_choice if step == 0 else None)
            # 2026-09-10：接近迭代上限时给出**收敛提醒**（软着陆）。
            # 实测（改稿工作间）模型在 doc-bound 场景会持续调用 retrieve 而不主动作答，
            # 12 轮耗尽后原实现直接 ABORT 且无回答 → 用户侧表现为「无法生成有效回答」。
            # 提前两轮提示"证据已足够、请作答"，把硬中止降级为正常收尾（配合
            # _finalize_without_tools 的双保险）。
            if (
                self.config.max_iterations >= 4
                and step == self.config.max_iterations - 2
                and not any("请立即基于现有证据给出最终回答" in m.content for m in messages)
            ):
                messages.append(ChatMessage(
                    role=Role.USER,
                    content="提示：你已获得多轮检索结果。请立即基于现有证据给出最终回答，不要再调用工具。",
                ))
            # T-01-C：发请求前的上下文闸门（真实基线 + 估算增量 → 超线则裁剪早期工具结果）
            self._gate_context(messages)
            resp, think_events = await self._think(messages, tool_specs, stream_tokens, tool_choice=choice_now)
            for ev in think_events:
                yield ev
            # ★ 2026-10-05：本步**是否已发出「完整回答」事件**。
            # `_think` 在"无（含托管）工具调用"时会补发一条 `message{content}`（见 `_think` 内
            # `if tool_acc: ... else: events.append(...)`）；而下方 FINISH 分支此前**无条件再发一条**，
            # 于是同一条回答在事件流里出现两次（content **逐字符相同**、仅差 `citations`）。
            # 后果（实测 10 条轨迹）：`persist_turn` 只给**最后一条**附加引用 ⇒ 每条回答恒定贡献
            # 「1 无引用 + 1 有引用」⇒ `citation_rate` **无论真实行为如何都被钉在 ~50%**
            # （显示 53.33%，去重后真值 8/8 = 100%）；同时 Trace 体积翻倍。
            _content_emitted = any(
                ev.type == "message" and (ev.data or {}).get("content") for ev in think_events
            )
            if resp is None:
                # C2（2026-09-28）：LLM 调用失败也要发 done —— 否则路由层拿不到本轮用量，
                # 会退化成"整段会话累计"（见 `_done_event` 的口径表）。分类由 `_think` 已标注。
                yield self._done_event(aborted=True, kind=failure.LLM_ERROR)
                return  # LLM 出错，已通过 error 事件透传
            # 成本/trace 精确聚合（来自 LLMResponse.cost，替换原先 token 粗算）
            self._steps += 1
            self._acc_tokens += resp.total_tokens
            # 实测 vs 估算对照（2026-09-17）：偏差显著时补一条 ——
            # 若本地估算长期严重偏低，说明"历史压缩阈值"与"上下文熔断"都在用错误的尺子量上下文。
            if self._last_ctx_estimate and resp.prompt_tokens > self._last_ctx_estimate * 1.5:
                logger.info(
                    "[上下文构成] 实测 prompt=%d 是估算 %d 的 %.1f 倍（估算口径偏低，压缩/熔断阈值受影响）| step=%d",
                    resp.prompt_tokens, self._last_ctx_estimate,
                    resp.prompt_tokens / max(1, self._last_ctx_estimate), self._steps,
                )
            # ★ A0（2026-10-02，② 的归因前置）：**每次调用**都打一行「实测 / 估算 / 比值 / 模型」。
            # 为什么与上面那行并存：上面只在偏差 >1.5 倍时才打（省日志），而 ② 要的是
            # **分布**（p50/离散度 ⇒ 决定该不该上校正系数 k），"只在异常时打"给不出分布。
            # 为什么带 model：比值可能**随厂商/模型而变**（T-01 的密度标定是特定模型做的），
            # 混在一起平均会得出无意义的数（同"跨模型混合污染 RMSE"那次教训）。
            # ⚠️ 只观测、**不改任何裁剪行为**（`estimate_ratio` 目前仅用于日志与离线分析）。
            _est_now = int(self._last_ctx_estimate or 0)
            if _est_now > 0 and resp.prompt_tokens > 0:
                logger.info(
                    "[估算比值] model=%s step=%d prompt=%d est=%d ratio=%.2f",
                    getattr(resp, "model", "?") or "?", self._steps,
                    resp.prompt_tokens, _est_now, resp.prompt_tokens / _est_now,
                )
            self._acc_prompt_tokens += resp.prompt_tokens
            self._acc_completion_tokens += resp.completion_tokens
            # 方案 D′（2026-09-17；口径 2026-09-28 修正）：记下**真实** prompt 规模 ——
            # `_last_real_prompt_tokens` 供循环内闸门与下一步估算（单次覆盖）；
            # `_first_prompt_tokens` 供 **done 事件回传并落 session config**（下一轮跨轮基线）；
            # `_peak_prompt_tokens` **仅供展示/落库**（本轮最大单次，不再参与跨轮判据）。
            self._last_real_prompt_tokens = resp.prompt_tokens
            self._peak_prompt_tokens = max(self._peak_prompt_tokens, resp.prompt_tokens)
            # 首步 = **本轮第一次**调用的真实 prompt（后续调用不覆盖）。
            # 它是"本轮进入时的上下文规模"，也是跨轮基线的最佳代理 —— 见 `__init__` 里的说明。
            # 只在本轮尚未记录时赋值：同一 step 的补偿重试不改变"进入本轮时的规模"这一语义。
            if not self._first_prompt_tokens:
                self._first_prompt_tokens = resp.prompt_tokens
            # T-01-C：响应已覆盖当前 `messages` → 闸门的增量起点前移，避免把历史重复计入
            self._gate_mark = len(messages)
            self._acc_cost += getattr(resp, "cost", 0.0) or 0.0
            self._acc_latency += getattr(resp, "latency_ms", 0) or 0
            tokens_used += resp.total_tokens
            # ① 单次请求上下文超限（2026-09-17 从"累计"改为"单轮 prompt"）：
            #    判据口径与历史压缩阈值（max_tokens_budget // 2）同源，都是"上下文规模"。
            if resp.prompt_tokens > self.config.max_tokens_budget:
                yield LoopEvent(type="error", state=LoopState.ABORT,
                                data={"reason": f"单次请求上下文超限（prompt {resp.prompt_tokens} > "
                                                f"上限 {self.config.max_tokens_budget} tokens）",
                                      "tokens": resp.prompt_tokens,
                                      # C2：本路径原先**漏标**分类 ⇒ 归因里落成 unknown。
                                      # 它与下面两条同属"预算熔断"，口径统一。
                                      "failure_kind": failure.BUDGET_EXCEEDED})
                yield self._done_event(aborted=True, kind=failure.BUDGET_EXCEEDED)
                return
            # ② 整轮累计 token 超限（含所有迭代）：防"跑飞"
            if self._acc_tokens > self.config.max_tokens_per_turn:
                yield LoopEvent(type="error", state=LoopState.ABORT,
                                data={"reason": f"本轮累计 token 超限（{self._acc_tokens} > "
                                                f"上限 {self.config.max_tokens_per_turn} tokens）",
                                      "tokens": self._acc_tokens,
                                      "failure_kind": failure.BUDGET_EXCEEDED})  # A4
                yield self._done_event(aborted=True, kind=failure.BUDGET_EXCEEDED)
                return
            # 成本熔断（NFR）：单轮成本超过上限即中止，避免失控计费
            if self._acc_cost > self.config.max_cost_per_turn:
                yield LoopEvent(type="error", state=LoopState.ABORT,
                                data={"reason": f"单轮成本超 ¥{self.config.max_cost_per_turn:.2f} 熔断",
                                      "cost": round(self._acc_cost, 4),
                                      "failure_kind": failure.BUDGET_EXCEEDED})  # A4
                yield self._done_event(aborted=True, kind=failure.BUDGET_EXCEEDED)
                return

            # ---- FINISH（无「真实」工具调用 → 直接回答）----
            # 注意：厂商托管的 web_search 工具由服务端执行并已在 content 并入检索结果，
            # 不计入「需要本地执行」的工具调用，避免 AgentLoop 误把它当本地工具去执行。
            real_calls = [tc for tc in (resp.message.tool_calls or []) if tc.name not in HOSTED_TOOLS]
            if not real_calls:
                # 2026-09-03 可靠性补偿：首轮被强制调用工具但模型未出 tool_call——
                # 部分厂商/模型对流式 tool_choice='required' 服从不稳定（实测 qwen3-vl-flash
                # 时遵守时无视），跳过检索会自由发挥（幻觉页码/拒答）。补一次**非流式**强制调用：
                # 同一约束下非流式 chat 返回 tool_call 的概率显著更高（实测稳定命中）。
                # 仅首轮强制场景补一次，成本可控；thinking 模式不兼容 tool_choice 故跳过。
                if step == 0 and force_tool_choice is not None and not self.config.thinking:
                    _retry_t0 = time.perf_counter()  # T-05+：补偿重试也要计入耗时分解（它是真实调用）
                    try:
                        retry = await self.llm.chat(messages, tool_specs, tool_choice=force_tool_choice)
                        self._steps += 1
                        self._acc_tokens += retry.total_tokens
                        self._acc_prompt_tokens += retry.prompt_tokens
                        self._acc_completion_tokens += retry.completion_tokens
                        # 与主调用同源：重试也是真实调用，同样刷新真实 prompt 与峰值
                        self._last_real_prompt_tokens = retry.prompt_tokens
                        self._peak_prompt_tokens = max(self._peak_prompt_tokens, retry.prompt_tokens)
                        self._acc_cost += getattr(retry, "cost", 0.0) or 0.0
                        self._acc_latency += getattr(retry, "latency_ms", 0) or 0
                        tokens_used += retry.total_tokens
                        # T-05+：与主调用同一口径打点。放在**判据之前** —— 即使这次结果被
                        # 判据否决（不采用），它也真实消耗了用户的时间与 token，必须可见。
                        _retry_ms = (time.perf_counter() - _retry_t0) * 1000
                        self._acc_llm_wall_ms += _retry_ms
                        self._log_llm_timing("chat-retry", _retry_ms, retry)
                        # 与主调用同一套判据（单轮上下文 + 整轮累计 + 成本）：任一超限都不采用该次重试结果
                        if retry.prompt_tokens <= self.config.max_tokens_budget \
                                and self._acc_tokens <= self.config.max_tokens_per_turn \
                                and self._acc_cost <= self.config.max_cost_per_turn:
                            real_calls = [tc for tc in (retry.message.tool_calls or [])
                                          if tc.name not in HOSTED_TOOLS]
                            if real_calls:
                                resp = retry
                    except Exception:  # noqa: BLE001  补偿失败不阻断：沿用流式结果继续
                        self._acc_llm_wall_ms += (time.perf_counter() - _retry_t0) * 1000
                        logger.warning("强制工具调用补偿失败（非流式），沿用流式结果", exc_info=True)
                if not real_calls:
                    if resp.message.content and not _content_emitted:
                        # 仅托管工具且已有正文（厂商已把检索结果并入 content）：必须以 message 事件回传正文，
                        # 否则调用方（routes_search.gen）拿不到最终回答（final_text 来自 message 事件）。
                        # ⚠️ 2026-10-05：**仅当本步尚未发过完整回答事件时才补发**（`_content_emitted`）。
                        # 此前无条件补发 ⇒ 与 `_think` 补发的那条重复（content 逐字符相同）⇒
                        # `citation_rate` 被钉在 ~50%、Trace 体积翻倍（详见 `_content_emitted` 处注释）。
                        yield LoopEvent(type="message", state=LoopState.FINISH,
                                        data={"content": resp.message.content})
                    # T-05+（2026-09-26）：本轮汇总 —— 一条日志回答"这 20 秒花在哪了"。
                    # 此前完全没有这条：`[上下文构成]` 只说规模、说不了耗时，用户报"慢"时
                    # 只能靠日志时序空档反推。分解口径：
                    #   墙钟 = LLM 墙钟 + 工具墙钟 + 其它（jieba/索引加载/SSE 编码/DB）
                    #   LLM 墙钟 − 厂商延迟合计 = 网络+排队
                    self._log_turn_summary()
                    # C2（2026-09-28）：done 改为**统一构造入口**（见 `_done_event`），
                    # 与各类中止路径共用同一份口径，避免"只有这一条路径有 done"。
                    yield self._done_event()
                    return

            # ---- ACT（P1-5 工具并行：SAFE/NETWORK 幂等工具并发，SANDBOX/ADMIN 串行）----
            calls = real_calls
            parallel: list = []
            serial: list = []
            for tc in calls:
                if self._permission_of(tc.name) in (Permission.SAFE, Permission.NETWORK):
                    parallel.append(tc)
                else:
                    serial.append(tc)
            # T-05+（2026-09-27 修正）：**在这里测整个 ACT 阶段的墙钟**，而不是在 `_run_one`
            # 里逐个累加。原因：并行分组下同一段时间会被计 N 次 → 汇总算出负的「其它」。
            # 起点放在执行前、终点放在**回显之前** ⇒ 本段语义 = "等工具返回"，不含 SSE 回显。
            _act_t0 = time.perf_counter()
            results: dict[str, tuple[ToolResult, list[LoopEvent], float]] = {}
            if parallel:
                ran = await asyncio.gather(*(self._run_one(tc, resp.message.content) for tc in parallel))
                results.update(dict(zip((tc.id for tc in parallel), ran, strict=False)))
            for tc in serial:
                results[tc.id] = await self._run_one(tc, resp.message.content)
            self._acc_tool_ms += (time.perf_counter() - _act_t0) * 1000
            # 按原始调用序回显 + 回填消息 + 失败计数（保证消息顺序稳定，避免污染 LLM 上下文）
            # C1（2026-09-28）：本步的"新证据"累计 —— 供下方的**无进展检测**使用。
            _step_new_keys = 0        # 本步新增的正文键（chunk id / 页指纹）条数
            _step_unjudgeable = False  # 本步含"无证据键"的工具 ⇒ 无法判断，保守视为有进展
            for tc in calls:
                result, evs, tool_ms = results[tc.id]
                for ev in evs:
                    yield ev
                if not result.ok:
                    tool_failures += 1
                    if tool_failures >= self.config.max_tool_failures:
                        yield LoopEvent(type="error", state=LoopState.ABORT,
                                        data={"reason": f"工具连续失败 {tool_failures} 次",
                                              "last_error": result.error,
                                              "failure_kind": failure.TOOL_ERROR})  # A4
                        yield self._done_event(aborted=True, kind=failure.TOOL_ERROR)
                        return
                else:
                    tool_failures = 0
                messages.append(ChatMessage(role=Role.ASSISTANT, content=resp.message.content,
                                             tool_calls=[tc], reasoning=resp.message.reasoning))
                # 2026-09-06 修复：此前直接拼 f"RESULT: {result.output}"，使 _format_tool_result
                # （及其内部 retrieve 的 200 字压缩）成为死代码 —— 单次检索 6 段全文直灌上下文，
                # 与 8000 token 预算设计冲突，表现为长对话过早压缩历史（"AI 忘了前面"）。
                #
                # 2026-09-28 D2：注入前先做**本轮内结果级去重** —— 同一份正文只注入一次，
                # 重复命中降级为「id/页码/首句」占位（不删信息）。收益不是延迟，而是
                # **少触发闸门裁剪**（裁剪 = 静默丢证据）。`原始/注入` 两列会直接体现它。
                # ★ 顺序要紧：**先** `_prepare_tool_output`（`retrieve` 的 200 字压缩），
                # **再**去重 —— 去重按"将要注入的字符数"比较，否则"先 retrieve（截断）
                # 后 pdf_extract（全文）"会被误判成重复，把模型正要拿到的全文换成占位。
                _prepared = _prepare_tool_output(tc.name, result.output)
                deduped, new_keys, dropped = _dedup_tool_output(tc.name, _prepared, self._text_keys)
                # C1：本步"是否获得新证据"的累计（判定见下方无进展检测）
                if new_keys:
                    _step_new_keys += len(new_keys)
                elif not _is_key_trackable(_prepared):
                    # 该工具的输出**不被结果级去重追踪**（产不出证据键）⇒ 无法判断是否有新信息。
                    # 保守记为"可能有进展"，避免把"用了非检索类工具"误报成原地打转。
                    _step_unjudgeable = True
                # 用统一的序列化入口（内部会再调一次 `_prepare_tool_output`，**幂等**：
                # 对已截断到 200 字的 retrieve 结果重复处理不产生二次截断）
                injected = _format_tool_result(tc.name, deduped)
                _tool_msg = ChatMessage(role=Role.TOOL, content=injected, tool_call_id=tc.id)
                messages.append(_tool_msg)
                if new_keys:
                    self._text_keys.update(new_keys)
                    # 登记"这条消息贡献了哪些正文" —— 闸门裁掉它时要按此撤销（硬约束①）
                    self._text_keys_by_msg[id(_tool_msg)] = list(new_keys)
                # T-05+（2026-09-26）：**工具调用统一日志** —— 一次调用一行，含"实际注入了多少"。
                # 为什么必须有：此前只有 `retrieve` 自己在工具层打耗时日志，其它工具（尤其
                # `pdf_extract`）**连是哪个工具都得靠上下文推断**。而"原始 → 注入"两列直接回答
                # 「哪个工具把上下文灌爆」：`retrieve` 走 200 字/段压缩、`pdf_extract` 不压缩，
                # 差距在这次日志里会一眼可见（实测 2026-09-26 那次请求 tool 估算 13,283 tokens）。
                # 用 `_rough_tokens`（与 `[上下文构成]` 同一把尺子）而非精确 tokenizer，
                # 口径一致才可互相验算。
                logger.info(
                    "[工具] %s %s 耗时=%.0fms 原始=%d字 → 注入=%d字（≈%d tokens）| 去重=%d%s | args=%s",
                    tc.name, "ok" if result.ok else "FAILED", tool_ms,
                    len(str(result.output)), len(injected), _rough_tokens(injected),
                    dropped,
                    # D3（2026-10-02）：缓存命中显式标注（以**事件里的 cached 标记**为准，
                    # 不用"耗时==0"推断 —— 极快的真实执行也会接近 0ms）。
                    # 未命中时不加任何字 ⇒ 既有 `[工具]` 行格式不变（既有测试按正则取值）。
                    " | 缓存=命中" if any(e.data.get("cached") for e in evs) else "",
                    str(tc.arguments)[:160],
                )

            # ---- 进度检测（C1，2026-09-28 修判据；2026-10-12 观察期收口：加「换策略」动作）----
            #
            # 原实现（`new_tool_count == last_tool_count`）**永不触发**：它位于 ACT 之后，
            # 而能走到这里的每一步都至少追加了一条 TOOL 消息 ⇒ 计数**单调递增** ⇒
            # `no_progress_steps >= 6` 不可达 ⇒ 唯一的"无进展"预警是**死代码**。
            # 已付代价（实测）：2026-09-27 一次 **9 轮 / 23 秒**反复读 page 8/9/10
            # （同一批内容读 2~3 遍），**全程零预警**，最终靠用户点「停止」脱身。
            #
            # 新判据 = **内容级**："本步是否获得新的证据键（chunk id / 页指纹）"，
            # 对齐语义（"有没有拿到新信息"）而不是计数（"追加了几条消息"）。
            #
            # ⚠️ 动作 = **换策略（补救），不是中止（放弃）**：项目既定原则是质量优先
            # （见 `docs/上下文预算-全面审查报告.md` P0-1）。观察期（09-28~10-12）的真实分布：
            # 连续无新证据**最长为 1 步**（6 次触发全为 1/6）、跨阈值告警 **0 次**
            # ⇒ 阈值由 6 收到 3，跨越时**注入一条换策略注记**（只注入一次），仍不中止。
            # 契约测试：`tests/test_no_progress_detection.py`（"可干预但绝不中止"）。
            if _step_new_keys > 0:
                no_progress_steps = 0
            elif _step_unjudgeable:
                no_progress_steps = 0  # 本步有无法判定的工具 → 保守归零
            else:
                no_progress_steps += 1
                logger.info(
                    "[无进展] 连续 %d/%d 步未获得新证据 | step=%d 工具=%s",
                    no_progress_steps, self.config.max_steps_no_progress, step,
                    ",".join(tc.name for tc in calls),
                )
                if no_progress_steps == self.config.max_steps_no_progress:
                    # 只干预一次（`==` 而非 `>=`）：跨越阈值那一刻最值得出手。
                    # ⚠️ 动作是**换策略（补救）**：注入一条系统注记要求换关键词/换工具/换参数；
                    #    **仍然不中止**（中止是放弃，见上方取舍）。
                    # 注记追加在 messages **末尾** ⇒ 历史压缩只从最前面丢（`rest.pop(0)`）
                    # ⇒ 本轮内不会被裁掉，无需改 `_is_pinned_note` 的 pin 规则。
                    messages.append(
                        ChatMessage(
                            role=Role.SYSTEM,
                            content=(
                                f"【无进展提醒】已连续 {no_progress_steps} 步未获得新的证据。"
                                "请**更换策略**：改写检索关键词、换用其它工具或调整参数，"
                                "不要重复上一轮完全相同的调用。"
                            ),
                        )
                    )
                    logger.warning(
                        "[无进展] 已达阈值 %d 步无新证据（疑似原地打转）| step=%d 工具=%s "
                        "⇒ 已注入「换策略」注记（补救；仍不中止）",
                        no_progress_steps, step, ",".join(tc.name for tc in calls),
                    )

        # 2026-09-10：达到最大迭代同样先兜底收尾（同上），保证「耗尽 ≠ 无回答」。
        async for ev in self._abort_with_fallback(
            messages,
            f"达到最大迭代 {self.config.max_iterations} 仍未收敛",
            failure.MAX_ITERATIONS,  # A4：显式标注分类
        ):
            yield ev
