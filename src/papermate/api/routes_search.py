"""PaperMate · 检索路由（docs/04 §4.6.1 / §4.6.2）

- POST /sessions/{id}/messages/stream：核心 SSE 流式对话接口（thinking/tool_call/tool_result/
  message_delta/message[带引用]/done/error）。
- POST /search：调试/评测用纯检索接口，直接返回带引用块（不走 Agent 循环）。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from lightcore.core import failure as m_failure
from lightcore.core.loop import LoopEvent, LoopState, model_injection_truncates
from lightcore.llm.base import ChatMessage, LLMResponse, Role, _rough_tokens
from lightcore.llm.fake import FakeLLM
from lightcore.llm.router import estimate_cost
from lightcore.rag.invoke import call_query
from lightcore.rag.page_utils import normalize_page
from lightcore.rag.pipeline import IndexingCancelled

from .. import runtime_settings
from ..schemas import ChatRequest, RetrievedChunkOut, SearchRequest
from ..services import failure_notice, prefetch_probe
from ..services.citation_clean import clean_citations, dedup_citations
from ..services.doc_labels import doc_label, mention_order
from ..services.intent import classify_intent
from ..skills._common import resolve_mention_in_scope
from .json_safe import safe_json_dict
from .sse import sse_event, sse_frame

router = APIRouter(tags=["search"])

logger = logging.getLogger(__name__)

# 流式中断哨兵：watcher 检测到客户端断开时投递，消费端据此取消 Agent/LLM 生成
_STREAM_STOP = object()

# 流式失败哨兵（P1-11，2026-09-15）：pump 内**非取消类异常**经此投递到消费端。
#
# 为什么必须有它：两条 pump（`_pump_direct` / `_pump`）原先都只捕获 `asyncio.CancelledError`，
# 其它异常会被 `finally: await _q.put(None)` 掩盖 —— 消费端把 `None` 当作「流正常结束」：
#   ① 照常落库并发出 `done` → **用户看到半截回答却以为已完成**（静默失败，比报错更难查）；
#   ② 异常从未被 `await`（清理处只有 `if not pump.done(): pump.cancel()`）→
#      仅在 GC 时打一行 "Task exception was never retrieved"，从日志里几乎无从定位。
# 载体用独立类型而非复用 `_STREAM_STOP`：后者语义是「客户端断开」（不该报错给已离开的客户端），
# 二者处理方式不同，混用会把"失败"降级成"中断"，再次丢信息。
class _StreamFailure:
    """承载 pump 内未捕获异常的最小载体（区别于「客户端断开」的 `_STREAM_STOP`）。"""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _UsageRecorder:
    """包装 LLM：把技能路径用到的**真实** prompt 记下来（T-01-D，2026-09-25）。

    **为什么需要这一层**：跨轮链是
    `loop 真实 prompt → done.first_prompt_tokens → session.config.last_prompt_tokens → 下一轮注入`，
    但**技能在 `skill.run(ctx)` 里直接调 LLM、不经 AgentLoop** → 没有 done 事件可回传，
    该会话下一轮的压缩判据只能退回估算（D′ 在技能路径上**断了一环**）。

    ⚠️ **回写的是"首次调用"的 prompt，不是峰值**（2026-09-28 修正）：
    峰值 = max(各次 prompt)，而技能往 prompt 里拼的检索 chunk **不会进入下一轮历史**
    ⇒ 峰值系统性高估，且 `max()` 会让跨轮基线**只增不减**。两个字段都留，由调用方选。

    **为什么只包装真实 LLM**：`is_real_llm(llm)` 的实现是 `not isinstance(llm, FakeLLM)` ——
    若给 FakeLLM 套壳，会让技能误判"有真实模型"而改变行为；而且 Fake 的 token 是**估算**，
    本来就不该回写。故 Fake 不包装（等价于"技能路径不回写"，与既有语义一致）。

    其余属性/方法一律透传（`__getattr__`），技能不需要知道这层壳存在。
    """

    __slots__ = ("_inner", "peak_prompt_tokens", "first_prompt_tokens")

    def __init__(self, inner) -> None:
        self._inner = inner
        self.peak_prompt_tokens = 0
        #: **首次**调用的真实 prompt —— 回写 `last_prompt_tokens` 用它，**不用峰值**。
        #: 理由同 `AgentLoop._first_prompt_tokens`：技能会把检索到的 chunk 拼进
        #: `synthesize` 的 prompt，而那部分**不会进入下一轮历史** ⇒ 峰值会系统性高估。
        self.first_prompt_tokens = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _note(self, n) -> None:
        if not isinstance(n, int):
            return
        if n > self.peak_prompt_tokens:
            self.peak_prompt_tokens = n
        if n > 0 and not self.first_prompt_tokens:
            self.first_prompt_tokens = n

    async def chat(self, *a, **kw):
        resp = await self._inner.chat(*a, **kw)
        self._note(getattr(resp, "prompt_tokens", 0))
        return resp

    async def stream(self, *a, **kw):
        # 流式的 usage 在 finished 块里（鸭子类型：用 getattr 取，别直接取新字段）
        async for ev in self._inner.stream(*a, **kw):
            if getattr(ev, "finished", False):
                self._note(getattr(ev, "prompt_tokens", 0))
            yield ev


class _StreamingChatProxy:
    """把技能的 `chat()`（一次性返回）转成「逐 delta 上报」，供 SSE 实时下发（2026-09-28）。

    ### 为什么需要在路由层加这一层
    技能内部统一走 `await llm.chat(...)`（**非流式**，一次拿到全文），路由层再
    `yield sse_event("message", {...})` 一次性发完 ⇒ 用户体感是「整段突然刷出」；
    而问答路径（`AgentLoop` / 直答）都是逐 `message_delta` 的 ⇒ **同一产品里两种观感**。

    修法**不改 12 个技能的代码**：技能调到的 `ctx.llm` 换成这个代理，
    代理的 `chat()` 内部改走底层 `stream()`，每个增量推入队列由路由层取走并逐块下发。

    ### 四条硬约束（都对着既有事故/约定来的）
    1. **只包真实 LLM**：`is_real_llm(llm)` 的实现是 `not isinstance(llm, FakeLLM)`。
       给 Fake 套壳会让技能误判「有真实模型」从而改变行为（同 `_UsageRecorder` 的理由）。
    2. **返回值必须与 `chat()` 完全一致**：技能拿到的仍是完整 `LLMResponse`，
       只是流路上多了上报 ⇒ 技能代码零感知、零改动。
    3. **异常照常抛出**：不在代理里吞异常，否则 `synthesize()` 的
       `try/except → return None → 确定性回退` 这条既有降级链会被破坏。
    4. **只上报、不决策**：delta 进队列纯粹为了展示，不参与任何流程判断。
       队列满/无人消费也不会阻塞技能（`put` 只会等待，不会丢内容）。
    """

    __slots__ = ("_inner", "_q")

    def __init__(self, inner, q: asyncio.Queue) -> None:
        self._inner = inner
        self._q = q

    def __getattr__(self, name):
        # 其余属性/方法（chat_web_search / supports_native_web_search / model_name …）
        # 一律透传 —— 技能与 `_common` 里的能力检测不需要知道这层壳存在。
        return getattr(self._inner, name)

    async def chat(self, messages, *args, **kwargs) -> LLMResponse:
        """非流式接口的流式实现：边收 delta 边上报，最后拼成完整的 `LLMResponse` 返回。"""
        parts: list[str] = []
        prompt_tokens = completion_tokens = latency_ms = 0
        cost = 0.0
        reasoning = ""
        tool_calls: list = []
        async for ev in self._inner.stream(messages, *args, **kwargs):
            delta = getattr(ev, "delta", "")
            if delta:
                parts.append(delta)
                await self._q.put(ev)   # 仅上报：消费端据此逐块 yield message_delta
            if getattr(ev, "finished", False):
                # 鸭子类型取值：替身/部分端点可能缺字段（同 `_UsageRecorder.stream` 的教训）
                prompt_tokens = getattr(ev, "prompt_tokens", 0) or 0
                completion_tokens = getattr(ev, "completion_tokens", 0) or 0
                latency_ms = getattr(ev, "latency_ms", 0) or 0
                cost = getattr(ev, "cost", 0.0) or 0.0
                reasoning = getattr(ev, "reasoning", "") or ""
                tool_calls = list(getattr(ev, "tool_calls", None) or [])
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="".join(parts),
                                reasoning=reasoning, tool_calls=tool_calls),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=getattr(self._inner, "model_name", ""),
            latency_ms=latency_ms,
            cost=cost,
        )


# 空回复兜底：Agent 出错或无实质回答时，必须给用户默认输出，绝不出现空白回复
# ④（2026-10-02）：文案本体移入 `services/failure_notice.py`（**单一来源**），
# 这里只保留别名，避免"同一句话两处各写一遍"必然漂移。
_FALLBACK_REPLY = failure_notice.FALLBACK_REPLY

# F7（2026-09-07）：检索故障专用兜底文案——与「模型调用失败」区分开。
# 明确告知是检索侧故障、不是文档没有内容，避免诱导用户重复上传文档。
_RETRIEVAL_ERROR_REPLY = (
    "抱歉，本轮检索服务暂时不可用（文档已关联，但检索全部失败，"
    "可能是索引损坏或嵌入/重排服务异常）。这不是文档中没有相关内容，"
    "请不要重新上传文档。建议：① 稍后重试；② 若持续失败，"
    "到「文献」对该文档重新解析以重建索引。"
)

#: 反思上下文窗口：反思时带上最近 N 条历史消息 + 本轮 user/assistant（2026-08-29 C 档）。
#: 此前只喂本轮两条，模型看不到上下文，提炼出的事实偏浅且易与已有记忆重复。
_REFLECT_HISTORY_WINDOW = 8


def _reflect_messages(history: list, user_text: str, final_text: str) -> list:
    """组装反思输入：最近若干条历史 + 本轮问答，末尾两条固定为 user / assistant。"""
    tail = history[-_REFLECT_HISTORY_WINDOW:] if history else []
    return [
        *tail,
        ChatMessage(role=Role.USER, content=user_text),
        ChatMessage(role=Role.ASSISTANT, content=final_text),
    ]

# 通用问答 / 寒暄走直接 LLM 路径，不再经过 Agent 工具循环（避免把通用问题强行走 retrieve 后
# 给出「当前会话未关联任何文档」的误导性回答，也避免展示工具推理过程）。
_GENERAL_QA_SYSTEM = (
    "你是 PaperMate（论文智友）的通用对话助手。用户本轮问题与已上传论文无关，"
    "请像普通 AI 助手一样直接、自然地回答，不要调用检索工具。\n"
    "【回复格式——必须遵守】\n"
    "1. 直接给答案，结论先行：开门见山回答问题本身。\n"
    "2. 严禁输出『过程』板块：不要出现『过程：』『分析：』『思路：』等流程叙述，"
    "不要描述你做了什么判断、调用了什么工具、检索到什么（如『意图识别：…』『文档检索：…』『结果分析：…』）；"
    "即使历史消息里出现过这类段落，也不要模仿。\n"
    "3. 简洁、有要点、排版清晰（可用 Markdown）。\n"
    "4. 思考过程也必须用简体中文（2026-09-28）：推理链会**直接展示在用户界面**的「推理过程」面板里，"
    "不得整段使用英文；原文术语与代码标识符可保留原文。\n"
    "【输出过滤——必须遵守】\n"
    "严禁在回复中提及系统内部状态：不要出现「当前会话」「关联的文档」「已关联文档」"
    "「已上传的文档」「知识库」「检索结果」「通用问答」等字样；直接给答案，就像你在和一个普通用户聊天。"
)
# 寒暄同理：它的推理链也会展示，一并约束（与 `_GENERAL_QA_SYSTEM` / `SYSTEM_PROMPT` 同口径）
_CHITCHAT_SYSTEM = (
    "你是 PaperMate（论文智友）。请简短、友好地回应用户的寒暄或闲聊。"
    "思考过程用简体中文（推理链会展示在用户界面）。"
)


# 引用清洗（噪声词表 / 长度上限 / 去重 / 跨调用去重 + 总数上限）**已收敛到单一实现**：
# `papermate.services.citation_clean`（B2-X2，2026-10-01）—— 模块顶部已 import。
# 历史：此前本文件与 `skills/_common.citations_from`、前端 `CitationCard.cleanCitations`
# 各有一份 ⇒ 行为分歧（技能路径绕过清洗；前端超长直接丢弃、噪声词多一个）⇒
# 引用在 UI 里静默消失但指标仍计入。现在**只保留这一个入口**，前端不再清洗。

#: 单轮引用的**总条数上限**（N1 2026-09-10）：跨工具调用去重后再截断，
#: 防止 agent 多轮检索累加出数百上千条引用（用户实测「引用来源显示几千条」）。
#: ⚠️ 刻意留在本模块：它是 **agent 循环**的跨调用配额（`dedup_citations(..., limit=)` 由调用方传入），
#: 不是"引用清洗"的口径 ⇒ 测试对它的 monkeypatch（`routes_search._MAX_CITATIONS`）必须继续生效。
_MAX_CITATIONS = 60


def is_empty_retrieval(output: object) -> bool:
    """该工具输出是否 = 「**已关联文档、确实检索了、但一条都没命中**」（C3，2026-09-28）。

    ### 为什么需要它
    `failure.RETRIEVAL_EMPTY` 定义了却**从未发出**（全库无发出点）⇒ 轨迹统计里
    永远没有"有文档但检索没命中"这一类 —— 而它正是「**文档在库里却答不出**」的主因，
    也是最能定位检索链路的信号。

    ### 为什么是"窄口径"（这是本函数的全部价值）
    下面每一种都**不是**本信号，漏判任何一条都会让这个指标变成噪声：

    | 情形 | 输出特征 | 为什么不算 |
    |---|---|---|
    | 未关联文档 | `source="none"` | 根本没检索（是"缺文档"，路由层另有拦截） |
    | 检索服务故障 | `ok=False` + `error="retrieval_unavailable"` | 是**系统故障**，不是"没命中"（F7 专门区分过） |
    | `pdf_extract(page=N)` 页不存在 | 无 `results` 键 | 是**页码非法**，不是检索结果 |
    | `pdf_extract()` 两个参数都没给 | `query` 为空 | **没有发生检索**（工具返回的是"该怎么用"的引导文案） |

    ⇒ 只有「source 是 rag/pdf **且** 请求里带了 query **且** results 是空列表 **且** 非故障」才算。

    ### ⚠️ 关于"检索故障"那一条：当前是**防御性**的（2026-09-28 实测）
    实测（`tests/test_retrieval_empty_attribution.py` 的故障用例 + 事件形状探针）：
    工具**失败**时 `loop._run_one` 产出的事件只有 `{call_id, ok: False, error}`，
    **不含 `output`** ⇒ 调用方拿到的 `output` 是 `None`，走不到本函数的这一条判断。
    保留它的理由：① 本函数是**纯判定**，契约应自洽（任何人直接传该 output 都应得到 False）；
    ② 一旦未来把失败事件的 output 补回去，这里就是现成的正确行为。

    ★ 同一探针还查出一个**独立缺陷**（不属 C3，已单独记录）：
    路由层依据 `output` 做的 **F7「检索服务暂时不可用」探测同样够不到** ⇒
    `_retrieval_error` 恒为 False、用户看到的是通用兜底文案而不是"该重建索引"。
    """
    if not isinstance(output, dict):
        return False
    if output.get("source") not in ("rag", "pdf"):
        return False
    if output.get("error") == "retrieval_unavailable":
        return False
    if output.get("source") == "pdf" and not output.get("query"):
        # `pdf_extract` 的三条出口里，只有 query 模式会真的去检索
        return False
    results = output.get("results")
    return isinstance(results, list) and len(results) == 0


def _effective_doc_scope(
    coll_doc_ids: list[int], cfg: dict, mention_ids: list[int]
) -> list[int]:
    """本轮检索的文档范围（N3 修复 2026-09-10）。

    基础范围 = 文献集文档 ∪ 会话 config.doc_ids（id 降序，与前端「文档N」标签同口径）。
    **存在主文献（draft_doc_id）时默认限缩到主文献**——判据与**前端 UI 完全对齐**：
    前端只要 `draft_doc_id != null` 就渲染改稿工作台（ChatPage: `isWorkbench && workbenchDraftId != null`），
    用户据此认为自己"在改稿模式"，后端就必须按改稿语义检索。

    ⚠️ 判据演进（第一版修错了）：
      - v1 用 `cfg["mode"] == "revision"` → **实测失效**：工作间创建时 mode 默认 "research"
        （`web/src/api/client.ts: mode: body.mode ?? "research"`），而设置主文献
        （`workspace_service.update`: `set_mode` / `set_draft` 相互独立）**不会**把 mode 改成 revision
        → session 15 是 `workspace + draft_doc_id=9 + mode="research"` → 限缩从未生效 →
        回复混入文献集里另外两篇（用户报「回复中还是出现了未添加文档的内容」）。
      - v2（当前）只看 `draft_doc_id`：与 UI 同源，无模式不同步的缺口。
    用户显式 `@` 提及其他文档时**不限缩**（尊重用户意图，也是跨篇对比的逃生通道）。
    """
    ids = sorted(set(coll_doc_ids) | set(cfg.get("doc_ids") or []), reverse=True)

    # ★ 2026-09-23 语义变更（用户决策）：用户显式 `@` 指定文档 → **限缩到被指定的文档**。
    #
    # 原语义是「`@` 时**不限缩**」，注释理由是「跨篇对比的逃生通道」。那个取向与本次需求冲突：
    # 用户要求「**指定了某篇文档时，只检索对应文档内容并回答**」——即 `@` 是**聚焦**语义，
    # 不是**追加**语义。按用户需求改为限缩。
    #
    # 与另一条链的口径统一：`_scope = mention_ids or (...)`（agent 检索范围）本来就是
    # 「@ 了就用 mention」，本函数却返回全量 → 两条链不一致（技能路径 @指定不生效）。
    # 现在两条链同口径：**mention > 主文献 > 全量**。
    if mention_ids:
        mentioned = [i for i in ids if i in set(mention_ids)]
        if mentioned:
            return mentioned
        # 提及的 id 均不在本会话关联范围内 → 视为无效提及，退回默认范围（不静默变空）
    draft = cfg.get("draft_doc_id")
    if isinstance(draft, int) and draft in set(ids):
        return [draft]
    return ids


# 多模态（DeepSeek-V4-Flash-Vision）：允许的图片 MIME 前缀与数量上限（官方支持 JPEG/PNG/GIF/WebP、
# 每请求 ≤600 张；本地保守限 10 张防滥用，单张 ≤32MiB 由前端校验）
_IMAGE_MIME_PREFIXES = ("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/gif;base64,", "data:image/webp;base64,")
_MAX_IMAGES_PER_REQUEST = 10


def _build_user_content(text: str, images: list[str] | None) -> str | list[dict]:
    """组装本轮 user 消息内容：纯文本直接返回 str；带图则返回 OpenAI 兼容内容块列表
    （text + image_url，DeepSeek-V4-Flash-Vision 的接口规范）。图片仅限 user 消息。"""
    imgs = images or []
    if not imgs:
        return text
    blocks: list[dict] = [{"type": "text", "text": text or ""}]
    for url in imgs:
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks


async def _stream_skill_output(
    skill_task: asyncio.Task,
    q: asyncio.Queue,
    request: Request,
    cancel_flag: threading.Event,
    session_id: int,
    *,
    poll_s: float = 0.3,
) -> AsyncIterator[str]:
    """边消费技能产生的流式增量、边等技能跑完；逐条 `yield` 出增量文本。

    返回时技能**已结束**（正常完成 / 出错 / 被取消），由调用方用
    `skill_task.result()` 或 `skill_task.cancelled()` 取最终结果 —— 本函数不做决策，
    也不吞异常（异常留在 task 上，由调用方按既有分支处理）。

    ### 结束判据为什么是 `done() + empty()` 而不是队列哨兵
    技能可能**多次**调用 LLM（如 `/规划` 走 `run_planning` 的多步循环），
    若用「一次调用结束就投哨兵」的写法，会在第一次调用后就提前终止消费、丢掉后续增量。
    ⇒ 以 **task 状态**为准：`done()` 且队列已排空才算结束。

    ### ★★ 循环体内必须有 `await`（2026-09-27 P0 事故）
    原先这段是 `while True:  if skill_task.done(): ... break` ——
    **循环体内零 `await`**（那句 `sleep` 误写在循环**之外**、只有技能跑完之后才执行得到的分支里）。
    后果链：`while True` 独占事件循环 → 被它 `create_task` 出来的 task **永远得不到调度**
    → `done()` 恒为 False → **永久死锁 + 单核 100%**：后端无法处理任何请求、
    连接堆积 CLOSE_WAIT、日志停止。当时的取证：`py-spy dump` 连续 5 次采样**全部**停在
    同一行且从不出现 sleep 帧；后端 PID 积压 **199 个 CLOSE_WAIT**；日志静止 23 分钟。

    ⇒ 抽成模块级函数是为了**可被单元测试直接覆盖**：走路由层时「有文档 / 无文档」
    两条路径行为不同，无文档会在更早处提前返回，导致集成测试**假通过**、对死锁毫无分辨力。
    """
    while True:
        if skill_task.done() and q.empty():
            return
        try:
            ev = await asyncio.wait_for(q.get(), timeout=poll_s)
        except TimeoutError:
            # 超时分支同时充当「周期性检查点」：这里必须先 await（wait_for 自己就是 await）
            # —— 它既是让出点，也是断连/停止的唯一检测位。
            if await request.is_disconnected() or cancel_flag.is_set():
                skill_task.cancel()
                logger.info("技能执行已被客户端中断 session=%s", session_id)
                return
            continue
        if ev is _STREAM_STOP:
            # 兼容「由外部 watcher 投递停止哨兵」的用法（与直答路径同款设施）
            skill_task.cancel()
            logger.info("技能执行已被客户端中断 session=%s", session_id)
            return
        delta = getattr(ev, "delta", "")
        if delta:
            yield delta


@router.post("/sessions/{session_id}/stop")
async def stop_session_stream(session_id: int, request: Request,
                              stream_id: str | None = None) -> dict:
    """停止当前流式生成：置位取消事件，流式端点在下一检查点终止。

    前端「停止」按钮在 abort 连接之外同时调用本端点——不依赖 ASGI 断连语义，
    确定性终止 LLM 生成与计费。无进行中的流 → ok: true（幂等 no-op）。

    F8（2026-09-07）：升级为**请求级**取消令牌。
    - 带 `stream_id`：**只**停止这一条流，同会话其它流不受影响。
      （旧版按 session_id 单槽存储，并发流会互相覆盖 → 先前的流永远收不到停止信号。）
    - 不带 `stream_id`：停止该会话的全部进行中流（兼容旧客户端 / 兜底语义）。
    """
    cancels = request.app.state.stream_cancels
    if stream_id:
        ev = cancels.get((session_id, stream_id))
        if ev is not None:
            ev.set()
        return {"ok": True}
    for (sid, _stream), ev in list(cancels.items()):
        if sid == session_id:
            ev.set()
    return {"ok": True}


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(session_id: int, body: ChatRequest, request: Request) -> StreamingResponse:
    # ★ O-12（2026-10-02）：TTFT 计时起点 = 进入处理器的那一刻。
    # 口径声明：这是**服务端**首字延迟（请求进入 → 首个正文增量），**不含网络往返**；
    # 与 `[LLM] ttft=`（单次调用的首 token）不是一回事，报告里必须写清用的是哪个。
    _t_request = time.perf_counter()
    # 首个正文增量的时刻（由 `_with_stream_cleanup` 单点埋点写入；一个请求只记一次）
    _ttft: dict[str, float] = {}

    def _ttft_ms() -> int | None:
        """已采到首个正文增量则返回其毫秒数（服务端 TTFT），否则 None（未产出即结束）。"""
        v = _ttft.get("first_delta_ms")
        return None if v is None else int(round(v))

    chat_svc = request.app.state.chat_svc
    session_svc = request.app.state.session_svc
    doc_svc = request.app.state.doc_svc
    session = session_svc.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 2026-09-07：线程级取消事件——贯穿「编码线程（add_chunks 批次边界检查）」与
    # 「LLM 流式 await（watcher 投哨兵）」。asyncio 取消传不进 to_thread 线程，
    # 必须用 threading.Event 让编码在批次间止损；/stop 端点与断连检测都置位它。
    # F8（2026-09-07）：请求级取消令牌——键为 (session_id, stream_id)。
    # 旧版按 session_id 单槽存储，同会话并发流（多标签页 / 上一轮未结束就重生成）
    # 会互相覆盖，导致先前的流永远收不到停止信号。前端不传时退化为会话级（"" 键）。
    cancel_flag = threading.Event()
    _stream_key = (session_id, body.stream_id or "")
    # M4 修复（2026-09-10）：**注册延后**到全部请求体校验通过、确定要开流时（见下方
    # 「注册取消令牌」处）——原先在此处注册，而其后的 regenerate / 图片 / 模型校验失败
    # 会抛 HTTPException，`_with_stream_cleanup` 尚未创建 → 令牌永不注销，
    # `stream_cancels` 随失败请求无上限增长（F8 修复的残留边角）。

    def _unregister_stream() -> None:
        """流结束（正常 / 异常 / 客户端断开）时注销本请求的令牌，避免条目残留。

        仅当字典里仍是自己时才移除——防止误删同会话后启动的流的令牌。
        """
        if request.app.state.stream_cancels.get(_stream_key) is cancel_flag:
            request.app.state.stream_cancels.pop(_stream_key, None)

    async def _with_stream_cleanup(agen):
        """包装 SSE 生成器：无论正常结束、异常还是客户端断开，都注销本请求的令牌。

        ★ O-12（2026-10-02）：TTFT 埋点也在这里 —— 这是**所有路径的唯一出口**
        （doc-bound / 直答 / 技能 / 各种兜底流都经此下发）⇒ 只埋一处，四条路径都覆盖，
        且**不侵入任何业务分支**（不影响 partial_text、落库、引用）。
        """
        try:
            async for chunk in agen:
                # 首个 `message_delta`（正文增量）= 用户看到第一个字的时刻。
                # 用 `in`（不是 startswith）：单个 chunk 可能是多事件拼接。
                if "first_delta_ms" not in _ttft and "event: message_delta" in chunk:
                    _dt = (time.perf_counter() - _t_request) * 1000
                    _ttft["first_delta_ms"] = _dt
                    logger.info(
                        "[TTFT] session=%s 服务端首字=%.0fms（请求进入→首个正文增量；不含网络）",
                        session_id, _dt,
                    )
                yield chunk
        except Exception:
            # ★ T-09+：SSE 流异常终止此前**完全无痕**（原本只有 `finally` 注销令牌）。
            # ⚠️ 此处**够不到** `gen()` 内的 events（局部变量）⇒ 只落日志 + 原样抛出；
            #   能落盘的那部分由 gen() 内各失败分支负责（`_failed` / `loop_cancelled`，
            #   二者均已补 `failure_kind`）。所以这里是"最后一道可观测性"，不是补偿逻辑。
            logger.exception("[SSE] 流式响应异常终止（gen 未自行收敛）")
            raise
        finally:
            _unregister_stream()

    # 「重新生成」：裁掉上一轮 assistant 回复，复用最后一条用户消息（content/images 从 DB 取）重跑，
    # 避免重复落库产生重复提问。无用户消息则无法重生成。
    if body.regenerate:
        turn = chat_svc.trim_last_assistant(session_id)
        if not turn:
            raise HTTPException(status_code=400, detail="当前会话没有可重新生成的回复")
        # F8：重建 body 时必须带上原 stream_id，否则注销与停止会退化为会话级
        body = ChatRequest(content=turn[0], images=turn[1] or None, regenerate=True,
                           stream_id=body.stream_id)

    # 多模态入参校验：格式（data URL + 允许的 MIME）与数量；非法则 400 明示，不静默
    if body.images:
        if len(body.images) > _MAX_IMAGES_PER_REQUEST:
            raise HTTPException(status_code=400, detail=f"单次最多 {_MAX_IMAGES_PER_REQUEST} 张图片")
        for i, url in enumerate(body.images):
            if not isinstance(url, str) or not url.startswith(_IMAGE_MIME_PREFIXES):
                raise HTTPException(status_code=400, detail=f"第 {i + 1} 张图片不是受支持的 data URL（仅 JPEG/PNG/GIF/WebP）")
    # ★ O-04：会话 `config_json` 同为外部写入的 TEXT 列，裸 json.loads ⇒ 一条脏行让对话入口 500。
    cfg = safe_json_dict(session.config_json or "{}")
    # F6：学科自动识别反馈——统一在入口计算 resolved_discipline / label / is_auto，
    # 供所有回答路径的 done 事件回传，使 chat 顶栏可显示「已识别为：X」并一键改选。
    from ..skills.disciplines import detect_discipline, discipline_label, get_profile
    _raw_d = cfg.get("discipline")
    _is_auto = (_raw_d == "auto")
    if _is_auto:
        _raw_d = detect_discipline(body.content or "")
    elif _raw_d in (None, "", "general"):
        _raw_d = runtime_settings.get_state().get("default_discipline", "general")
    resolved_discipline = get_profile(_raw_d).key
    resolved_discipline_label = discipline_label(resolved_discipline)
    _done_extra = {
        "discipline": resolved_discipline,
        "discipline_label": resolved_discipline_label,
        "discipline_auto": _is_auto,
    }
    # F5：会话有效文档 = 关联文献集文档 ∪ 会话手动勾选文档（并集，去重）。
    # 统一在此解析，供下方所有 doc_ids 读取点共用，避免漏改导致"关联文献集却报缺文档"。
    # 用 getattr 兜底：未装配文献集服务的 app（如精简测试/CLI）不应因此 500，
    # 退化为「仅用手动勾选文档」，与未关联文献集的行为一致。
    _collection_svc = getattr(request.app.state, "collection_svc", None)
    _coll_id = cfg.get("collection_id")
    _coll_doc_ids = _collection_svc.resolve_doc_ids(_coll_id) if (_collection_svc and _coll_id) else []
    # 2026-09-07：id **降序**，与前端 allDocs（GET /documents 为 id desc）展示顺序一致——
    # @文档N 序号解析依赖此顺序对齐（此前升序导致 @文档三 解析成另一篇文档）。
    # ★ 口径单一来源：`mention_order` 定义在 `services/doc_labels.py`（M3，2026-10-02）。
    _base_ids = sorted(set(_coll_doc_ids) | set(cfg.get("doc_ids") or []), reverse=True)
    mention_ids = [d for d in body.mention_doc_ids if d in set(_base_ids)]
    # ★★ M1（2026-10-02）：**手打「@文档N」的回退解析**。
    # 事故实测：用户手打「@文档二」⇒ 前端不发 `mention_doc_ids` ⇒ 上一行得到 `[]`
    #⇒ `_scope` 无限制 ⇒ `retrieve` 在全部关联文档里盲搜（10 次调用 0 次带 `doc_ids`）
    #   ⇒ 模型直接答「未能取到你所指『文档二』…」。
    # 优先级与技能层`resolve_doc_scope` 一致：**结构化字段 > 文本回退**。
    # ⚠️ 标签空间是 `_base_ids`（会话有效集），不是全库 —— 否则「文档七」会指到
    #    一篇**根本没关联**的论文，检索必然落空。
    mention_invalid_label: str | None = None
    if not mention_ids:
        _text_ids, _stripped, mention_invalid_label = resolve_mention_in_scope(
            body.content, _base_ids
        )
        mention_ids = [d for d in _text_ids if d in set(_base_ids)]
    # ⚠️ 越界（关联 2 篇却写「文档五」）**不静默降级为全库盲搜**：
    # 保持 `mention_ids = []`（行为=不限定），但由 notice/日志**据实告知**，
    # 否则用户会以为限定生效了（实测里这正是"看似 @ 了、实际没限定"的来源）。
    # N3（2026-09-10）：改稿模式限缩到主文献——口径见 _effective_doc_scope（可单测）
    effective_doc_ids = _effective_doc_scope(_coll_doc_ids, cfg, mention_ids)

    _REFUSAL_PATTERNS = (
        "系统中无任何", "无任何文档", "无任何上传", "当前系统中", "当前未上传任何",
        "无可供分析", "无可总结", "向量库为空", "所有检索返回空", "所有工具调用返回",
        "未触发任何文档", "前提不成立", "静默跳过", "无效指令", "未成功上传",
        "未上传 PDF", "未上传文档", "未提供任何文件", "不存在待总结", "不存在任何文档",
        "未关联任何文档", "未关联文档", "没有关联文档", "无文档关联",
        "无法检索", "无法基于论文", "无法执行对比", "无法执行检索",
        "请在右栏勾选", "请到右栏勾选", "右栏勾选",
        "没有生成有效回答", "模型调用失败或返回内容为空",
    )

    def _is_refusal(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return any(p in t for p in _REFUSAL_PATTERNS)

    # 「打断提醒」机制（2026-09-03，会话 14「YOLO改动差异对比」死循环复盘）：
    # 无效回复（兜底/拒答/空回复）若被当作普通回答输出、无结构化区分、无「该怎么做」指引，
    # 用户只能反复重问 → 对话既不推进又无清晰阻断。机制目标 = 让无效回复在
    # 结构（notice）+ 视觉（前端警示卡）+ 行为（循环检测升级）三层跳脱正常回答流。
    # 连续 _LOOP_THRESHOLD 次无效回复升级为 loop_risk 强打断，彻底打断重复提问死循环。
    _LOOP_THRESHOLD = 2

    def _is_invalid_reply(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        if _is_refusal(t):
            return True
        if _FALLBACK_REPLY in t:
            return True
        return False

    def _count_trailing_invalid(hist) -> int:
        """统计历史尾部连续无效助手消息数（含拒答/兜底/空回复）。"""
        n = 0
        for m in reversed(hist):
            if getattr(m, "role", None) != Role.ASSISTANT:
                break
            if _is_invalid_reply(getattr(m, "content", "") or ""):
                n += 1
            else:
                break
        return n

    def _persist_meta(
        notice: dict | None,
        *,
        model_risk: str = "",
        ctx: dict | None = None,
        extra: dict | None = None,
    ) -> dict | None:
        """落库 meta：在 `invalid/notice` 之外附加本轮的「模型风险」标记（若有）。

        ⚠️ `model_risk` 刻意做成**独立字段**：不挂 notice、不计入「连续无效」——
        否则一次"模型不可靠"就会把历史轮次升级成 `loop_risk` 强打断（口径污染）。
        它只是**提示性**标记：告诉用户"这轮回答要核对引用"。

        ★ P1-a2（2026-10-03）：`ctx` = 上下文占用（`ctx_peak`/`ctx_budget`/`ctx_warn_at`），
        用于**刷新后回填**占用率（解析口径见 `_ctx_usage_from_done`；技能/流式两条路径
        现在都用它，见 P-03 2026-10-09）。

        ★ P0-2（2026-10-03）：`extra` = **其它中性观测**（当前是 `unrecognized_command`）。
        ⚠️ 与 `ctx` 一样**只做普通合并**，**绝不夹带 `invalid`** ——
        凡"用户/系统的操作事实"（打错命令、上下文多大）都不该算作"模型给出无效回答"，
        否则会污染「连续无效」计数（进而误触发 `loop_risk` 强打断）。
        """
        meta = {"invalid": True, "notice": notice} if notice else None
        if model_risk:
            meta = {**(meta or {}), "model_risk": True}
        if ctx:
            meta = {**(meta or {}), **ctx}
        if extra:
            meta = {**(meta or {}), **extra}
        return meta

    def _ctx_usage_from_done(data: dict | None) -> dict:
        """从 `done` 事件的 data 里抽出「上下文占用」三数（P1-a2，2026-10-03）。

        ⚠️ 全部做 `isinstance` 校验：`done` 帧来自网络/鸭子类型替身，
        缺字段或类型不对时**必须降级**（返回缺项）而不是抛出 ——
        否则一条脏 meta 会让**整页历史 500**（该风险在 `routes_sessions` 已有先例）。
        """
        d = data or {}
        out: dict = {}
        peak, budget = d.get("peak_prompt_tokens"), d.get("max_tokens_budget")
        if isinstance(peak, int) and peak > 0:
            out["ctx_peak"] = peak
        if isinstance(budget, int) and budget > 0:
            out["ctx_budget"] = budget
        warn_at = d.get("compact_trigger_tokens")
        if isinstance(warn_at, int) and warn_at > 0:
            out["ctx_warn_at"] = warn_at
        # ★ P2（2026-10-03）：分项构成（供「上下文用量」面板显示"哪块最大"）。
        # ⚠️ 逐值校验并**丢弃非正数**：脏 meta 不得让历史接口 500，
        #   也不该在面板上画出"0% 的段"（0 值在 `buildCtxPanel` 里本就会被过滤）。
        bd = d.get("context_breakdown")
        if isinstance(bd, dict):
            clean = {
                k: v for k, v in bd.items() if isinstance(k, str) and isinstance(v, int) and v > 0
            }
            if clean:
                out["ctx_breakdown"] = clean
        return out

    # ★ P-03（2026-10-09）：原 `_ctx_usage_from_events(events)` **已删除**。
    #   原因：它是"技能路径落库占用率"的唯一调用方，而技能分支的 `events` 收集器**恒为空**
    #   （技能输出直接走 SSE、不进 events）⇒ 它对该路径**永远返回 `{}`**：接上了但没数据，
    #   当轮 `done` 又不带数 ⇒ 实时与历史**双双缺失**（用户实测：「只使用命令时不计入上下文用量」）。
    #   现改为直接用 `_UsageRecorder` 捕获的 `_ctx_payload`（与实时 done 同源），
    #   `_ctx_usage_from_done` 仍是**唯一解析口径**。
    #   ★ 删除而非保留：避免留下"看似可用、实则恒返回空"的死代码/死接线
    #   （与本项目对"死配置项"的一贯纪律一致）。

    def _build_notice(kind: str, title: str, message: str, suggestion: str) -> dict:
        return {"kind": kind, "title": title, "message": message, "suggestion": suggestion}

    def _write_back_prompt_baseline(baseline: int) -> None:
        """把本轮的**跨轮基线**回写 session config，供下一轮做压缩触发判据（方案 D′）。

        **该写哪个数**：本轮的 **首次调用真实 prompt**（= "进入本轮时的上下文规模"）。
        ⚠️ **不是峰值**（2026-09-28 修正）：峰值 = max(各步 prompt)，而本轮的工具结果
        **不会进入下一轮历史** ⇒ 峰值系统性高估；更糟的是 `max()` 会让基线**只增不减**。
        实测事故：某轮峰值 29,357 被回写后，之后每轮首步实际仅 15,411（**高估 87%**），
        导致压缩每轮误触发、闸门每轮都去裁历史里的长工具结果（**过期数字驱动的证据损失**）。
        实测最优代理是首步：15,411 → 下一轮首步 15,700（**误差 +2%**）。

        AgentLoop **每轮新建**（无跨轮缓存）→ 真实值必须跨轮传递，否则第 1 步
        （历史最长、最该压缩的那一步）永远拿不到真实值，D′ 会静默退回"靠估算"。

        ⚠️ 只接受**真实**采集到的 usage：估算值写进来会让下一轮的判据失去意义
        （那正是方案 D′ 要摆脱的东西）→ 因此 `baseline <= 0` 时直接跳过。
        纯优化项：写失败只记日志，绝不因此影响本轮回答。
        """
        if baseline <= 0:
            return
        try:
            session_svc.update(session_id, config={"last_prompt_tokens": int(baseline)})
        except Exception:  # noqa: BLE001
            logger.warning("last_prompt_tokens 写回失败（不影响本轮回答）", exc_info=True)

    def _invalid_notice(kind: str, trailing_invalid: int) -> dict:
        """构造无效回复 notice；尾部已连续无效达阈值时升级为 loop_risk 强打断。"""
        if trailing_invalid >= _LOOP_THRESHOLD:
            return _build_notice(
                "loop_risk",
                "⚠️ 连续多次未能回答",
                f"最近 {trailing_invalid + 1} 次回复均无效，问题可能卡在同一处。",
                "请先排查：① 会话是否已关联相关文档（右栏「关联文档」）；② 模型是否可用"
                "（设置页「测试连接」/账户余额）；③ 问题是否需拆分为更小、具体的子问题。"
                "排除任一后再继续提问。",
            )
        if kind == "missing_doc":
            return _build_notice(
                "missing_doc",
                "⚠️ 缺少关联文档",
                "当前会话未关联任何文档，无法基于论文内容作答。",
                "请在右侧「关联文档」勾选要研读的论文；或改问与文档无关的问题"
                "（将基于通用知识直接回答，不读取任何文档）。",
            )
        return _build_notice(
            "fallback",
            "⚠️ 未生成有效回答",
            "本次模型调用失败或返回内容为空。",
            "① 检查设置页该厂商的「测试连接」与账户余额；② 换个模型重试；③ 稍后再试。",
        )

    def _retrieval_error_notice() -> dict:
        """F7（2026-09-07）：检索故障专用提示——与「文档无相关内容」严格区分。

        检索全部失败时若沿用 fallback 文案，用户会以为「文档里没有」而去重复上传；
        本提示明确指向检索侧故障，并给出「重建索引」这类可操作建议。
        """
        return _build_notice(
            "retrieval_error",
            "⚠️ 检索服务暂时不可用",
            "文档已关联，但本轮检索全部失败（可能是索引损坏或嵌入/重排服务异常），"
            "并非文档中没有相关内容。",
            "① 稍后重试本问题；② 若持续失败，到「文献」对该文档重新解析以重建索引；"
            "③ 检查后端日志中「检索管线失败」的具体异常。",
        )

    def _stream_error_notice(reason: str = "", kind: str = "") -> dict:
        """P1-11（2026-09-15）：流式生成**中途异常**专用提示。

        与相邻两类严格区分（口吻差异决定用户下一步动作）：

        | 场景 | 提示 | 用户该做什么 |
        |---|---|---|
        | 模型调用失败/返回空（`fallback`） | 「未生成有效回答」 | 查余额/换模型 |
        | 检索全部失败（`retrieval_error`） | 「检索服务不可用」 | 重建索引 |
        | **生成中途异常**（本项） | 「回答生成中断，内容可能不完整」 | **直接重试**；反复出现再查设置 |

        落库时写入 `meta.notice` → 前端从历史回填警示卡（`historyToThread`），
        因此**刷新页面后仍能看到"这次回答是残缺的"**，不会被误当成完整结论。

        2026-09-16：新增 `reason` —— 把 AgentLoop 的**具体失败原因**（余额不足 / token 熔断 /
        迭代上限…）透给用户。此前前端只看到"未生成有效回答"，完全无法自救
        （实测：用户因 DeepSeek 余额不足连续失败，却以为是"模型连接"问题）。
        安全考虑：reason 可能含厂商响应体片段 → **截断 160 字符**且不透传 traceback。
        """
        # ④（2026-10-02）：文案与分流全部移到 `services/failure_notice`（**单一来源**）。
        # 本函数只做转发，保留名字与调用点不动。分流判据从"按中文子串猜"改为
        # **`failure_kind` 枚举**（老轨迹 kind 缺失时，`notice_for` 内部才退回旧启发式）。
        return failure_notice.notice_for(kind, reason)

    def _user_stop_notice() -> dict:
        """「用户主动停止」的**中性回执**（2026-09-30 新增）。

        ★ 为什么是**提示**而不是告警：用户点「停止」是**正常操作**。
          用红色告警会把正常行为渲染成"出问题了" —— 这与 `unfinished_rate`
          （刻意不叫 error_rate）那次命名教训同源。
        ★ 为什么要有它：此前停止后**没有任何回执**，刷新页面只看到一段半截内容，
          用户分不清"是我自己停的"还是"系统断了"（这正是 T-09 里 `interrupted`
          长期无法与崩溃区分的用户侧表现）。
        """
        return _build_notice(
            "user_stop", "已停止本轮",
            "你停止了本轮生成。以下内容是停止前产出的部分结果，可能不完整。",
            "如需完整回答，可重新提问，或把问题拆得更具体后再问。",
        )

    def _stream_error_payload(reason: str = "", kind: str = m_failure.CRASH) -> dict:
        """发给前端 `error` 事件的内容。

        2026-09-16：携带**具体失败原因**（截断 160 字符）。此前只给固定文案，
        用户无法区分"余额不足"和"服务异常" —— 而两者的自救动作完全不同。
        仍不透传 traceback 原文（可能含路径/密钥等敏感信息）。

        2026-09-30（T-09+）：补 `failure_kind`。此前**只有 SSE 事件带原因、落库 trace 不带**
        ⇒ `/metrics` 按 `failure_kind` 分桶时这些轮次全落 `unknown`（真实库实测 45/51）。
        ★ 分类由**产生处**给出，不靠事后解析 reason 文案（A4 的既有纪律）。
        """
        detail = (reason or "").strip()
        if len(detail) > 160:
            detail = detail[:160] + "…"
        return {
            "reason": detail or "回答生成中断：服务端在生成过程中出现异常，内容可能不完整。",
            "code": "stream_failed",
            "recoverable": True,
            "failure_kind": m_failure.normalize(kind),
        }


    # 「打断提醒」循环检测：提前加载历史，统计尾部连续无效助手消息数，
    # 供下方各无效回复点（缺文档拦截 / 兜底）升级为 loop_risk 强打断。
    history_raw = session_svc.load_history(session_id)
    _trailing_invalid = _count_trailing_invalid(history_raw)
    # 本轮意图提前分类：协议级拦截与后续路径复用，避免重复规则判断。
    # ⑤（2026-10-02）：把**结构化 @ 提及**（mention_ids，已做有效性过滤）喂进分类 ——
    # 否则「@文档四 实验对象是什么」被判 general_qa ⇒ 注记说与具体文档无关，
    # 而 doc_bound 又强制检索 ⇒ 两个信号互相矛盾（实测日志 intent=general_qa doc_bound=True）。
    _has_doc_mention = bool(mention_ids)
    it = classify_intent(body.content, has_doc_mention=_has_doc_mention)
    # 会话实际生效模型（2026-09-16 起**恒等于用户所选**——doc-bound 静默回落已取消，
    # 计费/用量记录以「实际跑的模型」为准）：
    model = chat_svc.effective_model(session_id)
    # doc-bound 模型风险（2026-09-16 晚，策略 = **「警示但放行」**）：所选模型在该场景可能跳过检索，
    # **不拦截**（用户有权使用自己付费的模型），但风险必须可见：
    #   ① 输入框上方常驻横幅（network-capability 的 model_risk，前端每轮都能看到）；
    #   ② 本轮回答落库时打 `meta.model_risk` → 刷新后该条回答仍带警示。
    model_risk = chat_svc.doc_bound_model_risk(session_id)
    # ---- A8/C8（2026-09-28）：**超长输入前置拦截** ----
    #
    # ### 为什么必须拦（缺陷本体）
    # `max_tokens_budget` 的判据在**响应返回之后**才执行（`loop._run_stream_impl`）
    # ⇒ 一旦输入本身就把上下文占满，用户的体验是「等一会儿 → 本次没有生成有效回答」：
    # **钱已经花了、却完全不知道原因**（兜底文案里没有一句提到"输入太长"）。
    #
    # ### 做法
    # 与下面「缺文档」拦截同款：**不调 LLM**、直接给可操作文案 + 结构化 notice，
    # 并照常落库（用户提问不能丢、刷新后警示卡仍在）。
    #
    # ### 位置说明
    # 放在这里是因为它需要 `_build_notice`（:560）/ `_done_extra`（:482）/ `model`（:690）
    # 三个**上面才定义好**的东西；其上方只有纯本地校验与意图分类（**无 LLM 调用**），
    # 相对它要拦下的"整轮白跑 + 真实计费"可忽略。
    #
    # ### 口径
    # 用 `_rough_tokens`（与 `[上下文构成]`/闸门**同一把尺子**）而不是 `len()` ——
    # 换字符数就是又一个"两把尺子"，而本项目在这上面吃过大亏。
    # 多模态也一并计入（`_rough_tokens` 对 image_url 块按单图 384 tokens 计）。
    _input_tokens = _rough_tokens(_build_user_content(body.content, body.images))
    # ⚠️ 必须传 model：预算按模型自动推断后，输入上限也得同源（否则又一把尺子）
    _input_limit = runtime_settings.input_token_limit(model)
    if _input_tokens > _input_limit:
        # ★ 同源：与 build_agent 用同一个 resolve（超长提示里的额度才与实际一致）
        _budget = runtime_settings.resolve_budget_for(model)
        _too_long_text = (
            f"这条消息太长了，本次没有执行。\n\n"
            f"- 它的规模约 **{_input_tokens:,} tokens**（{len(body.content):,} 个字符）；\n"
            f"- 单条消息上限 **{_input_limit:,} tokens**"
            f"（= 单次请求上下文上限 {_budget:,} 的一半）。\n\n"
            "若直接发出，它会在检索之前就把上下文占满并导致本轮中止（**并且已经产生费用**）——"
            "所以这里提前拦下，不消耗任何额度。"
        )
        _too_long_notice = _build_notice(
            "input_too_long",
            "⚠️ 这条消息太长了",
            f"约 {_input_tokens:,} tokens（{len(body.content):,} 字符），"
            f"超过单条上限 {_input_limit:,} tokens。已提前拦下，未产生任何费用。",
            "① 把长文**拆成几段**分次提问；"
            "② 若这是一整篇论文/整章内容：把它作为文档上传到「文献」并关联到本会话，"
            "之后直接针对它提问（我可检索全文，比粘贴更省上下文、也更快）；"
            "③ 若只需改写其中某一段，只粘贴那一段即可。",
        )

        async def _too_long_stream():
            yield sse_event("message", {"text": _too_long_text, "citations": []})
            yield sse_event("done", {
                "steps": 0, "total_tokens": 0, "cost": 0.0, "latency_ms": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "notice": _too_long_notice, **_done_extra,
            })

        # 仍持久化本轮 user 消息（含多模态块），保证历史/标题回填可用，不丢失用户输入；
        # 同时写入 notice，供前端刷新后回填警示卡（与「缺文档」拦截同一契约）。
        chat_svc.persist_turn(
            session_id,
            _build_user_content(body.content, body.images),
            [],
            [],
            _too_long_text,
            model,
            regenerate=body.regenerate,
            meta={"invalid": True, "notice": _too_long_notice},
        )
        logger.info(
            "输入超长已拦截：约 %d tokens（字符 %d）> 上限 %d | session=%s",
            _input_tokens, len(body.content), _input_limit, session_id,
        )
        return StreamingResponse(_with_stream_cleanup(_too_long_stream()), media_type="text/event-stream")
    # 协议级拦截：未关联任何文档，却提出明确指代文档的问题（或文档类命令）→
    # 直接提醒「缺少文档」，不调 LLM、不让弱模型用自身知识编造「文档内容」。
    # 仅当问题/命令确实指向文档时才拦截；纯通用疑问（如「什么是注意力机制」）放行，
    # 由模型基于通用知识作答。根因：should_force_tool_call 仅在有文档时强制检索，
    # 未关联文档时 force_choice=None，模型按提示词「若会话已关联文档才检索」选择不检索、
    # 直接编答案，导致「缺失文档提醒」路径（依赖 retrieve→source=none）永远走不到。
    if not effective_doc_ids:
        from ..services.intent import has_document_reference

        if it.name == "command" or (it.name == "doc_qa" and has_document_reference(body.content)):
            # 区分「已上传到文献库但未关联到本会话」与「完全无文档」：
            # 前者要让用户知道「文档已在库里，只是没勾选进本会话」，避免"上传过却显示未上传"的困惑。
            library_count = 0
            try:
                library_count = len(doc_svc.list_documents())
            except Exception:  # noqa: BLE001  查库失败不阻断，退化为"完全无文档"提示
                library_count = 0
            _lib_hint = (
                f"提示：你在「文献」中已上传了 {library_count} 篇文档，但它们未关联到本会话。"
                "请在右侧「关联文档」中勾选要研读的论文。"
            ) if library_count > 0 else ""
            if it.name == "command":
                _reminder = (
                    "当前会话尚未关联任何文档，命令类操作（如 /总结、/相关工作、/对比）"
                    "需要先关联文档才能执行。\n\n"
                    + (_lib_hint + "\n\n" if _lib_hint else "请在右侧「关联文档」中勾选要研读的论文，再向我发送命令。")
                )
            else:
                _reminder = (
                    f"当前会话尚未关联任何文档，我无法基于论文内容回答「{body.content.strip()}」。\n\n"
                    + (_lib_hint + "\n\n" if _lib_hint else "")
                    + "请先在右侧「关联文档」中勾选要研读的论文，再向我提问；"
                    "或换一个与文档无关的问题（我可基于通用知识直接回答，不读取任何文档）。"
                )

            notice = _invalid_notice("missing_doc", _trailing_invalid)

            async def _missing_doc_stream():
                yield sse_event("message", {"text": _reminder, "citations": []})
                yield sse_event("done", {"steps": 0, "total_tokens": 0, "cost": 0.0, "latency_ms": 0, "notice": notice, **_done_extra})

            # 仍持久化本轮 user 消息（含多模态块），保证历史/标题回填可用，不丢失用户输入；
            # 同时把 invalid+notice 写入 meta，供前端刷新后回填「打断提醒」卡片
            chat_svc.persist_turn(
                session_id,
                _build_user_content(body.content, body.images),
                [],
                [],
                _reminder,
                model,
                regenerate=body.regenerate,
                meta={"invalid": True, "notice": notice},
            )
            return StreamingResponse(_with_stream_cleanup(_missing_doc_stream()), media_type="text/event-stream")
    # history_raw 已在上方提前加载（供「打断提醒」循环检测复用），此处直接进入过滤
    # 过滤「空 assistant 噪声」（之前 LLM 错误后留下的 content="" 占位，避免误导新模型）
    history = [
        m for m in history_raw
        if not (m.role == Role.ASSISTANT and not (m.content or "").strip() and not m.tool_calls)
    ]
    # 过滤「拒接型/状态观察型」历史 assistant 消息：早期 reflect() 反复把 Qwen 的拒接/观察
    # （如『系统中无任何文档』『对重复指令静默跳过』）写进了会话历史，这些是幻觉而非事实，
    # 会让 LLM 在每轮推理时把它们当成"刚刚发生的真事"，跳过 retrieve 工具调用
    # （见 docs/实施记录.md 2026-08-19 三续）。
    # 2026-09-03 拓宽：检索/重排故障期产出的「未关联任何文档」「无法检索」「请右栏勾选」
    # 「没有生成有效回答」等拒答/兜底也被误当事实写回历史并自证循环（词表原只覆盖
    # 『无任何文档』写法，匹配不到『未关联任何文档』）。补齐变体让被污染会话自动自愈。
    history = [
        m for m in history
        if not (m.role == Role.ASSISTANT and _is_refusal(m.content))
    ]
    # 「重新生成」：trim 仅删了上一轮 assistant，history 仍含最后一条 user 消息；
    # 但 Agent 循环会在末尾再追加本次 user_content，若不剔除则用户消息在上下文里出现两次。
    # drop 尾部那条 user 消息即可（其紧跟的 assistant 已被 trim，故末尾必为 user）。
    if body.regenerate and history and history[-1].role == Role.USER:
        history = history[:-1]
    # 短期记忆滚动窗口：只保留最近 N 条（NFR: max_tokens_budget 8k 配合）
    _MAX_HISTORY = 20
    # A1（2026-09-28）：截断**必须留痕**。
    # 原先 `history[-20:]` 静默执行：40 条短消息（约 2k tokens）在预算极其宽裕时被砍到 20 条
    # —— 用户会发现"前面说过的话被忘了"，而日志里**没有任何线索**。
    # ⚠️ 口径说明（易踩）：这里按**条数**，而其余预算机制全是 **token 口径**
    #（21,000 / 14,700 / 30,000）—— 它是 8k 预算时代的遗留，预算扩容后未同步。
    # 本次**只加日志、不改判据**（改成 token 口径会改每轮的上下文，属行为变更，另议）。
    if len(history) > _MAX_HISTORY:
        logger.info(
            "[历史窗口] 会话=%s 历史 %d 条 > 上限 %d ⇒ 丢弃最旧 %d 条"
            "（⚠️ 按【条数】截断，与其余 token 口径不同源 —— 审查报告 A1）",
            session_id, len(history), _MAX_HISTORY, len(history) - _MAX_HISTORY,
        )
    history = history[-_MAX_HISTORY:]
    # 2026-09-07：build_agent 编码阶段的早期 watcher——此阶段 LLM 流 watcher 尚未建立，
    # 若不监视断连，用户关页/停止后编码线程会空转到结束（asyncio 取消传不进线程，
    # 必须置位 threading.Event 让 add_chunks 在批次边界止损）。
    async def _watch_build() -> None:
        while not (await request.is_disconnected() or cancel_flag.is_set()):
            await asyncio.sleep(0.3)
        cancel_flag.set()  # 断连视同停止：编码在批次边界退出

    build_watch = asyncio.create_task(_watch_build())
    try:
        # N3-v2 修复（2026-09-10）：把「本轮检索范围」**真正接到 agent**。
        # 原先只传 `mention_ids`（@ 提及）——`_effective_doc_scope` 算出的**主文献限缩**
        # 仅作用于技能路径（SkillContext.doc_ids），agent 的 retrieve 仍按会话全部关联文档
        # 检索 → 改稿工作间问「文档中的创新点」会把文献集里其他论文一并作答（用户实测）。
        # 语义：①@ 提及优先；②无提及但有主文献 → `[draft_doc_id]`；
        #      ③两者皆无 → None（**全量且不过滤**，保持既有契约：仅在真正限缩时才传 restrict）。
        _scope_restricted = len(effective_doc_ids) < len(_base_ids)
        _scope = mention_ids or (effective_doc_ids if _scope_restricted else None)
        agent = await chat_svc.build_agent(
            session_id, cancel_event=cancel_flag,
            restrict_doc_ids=_scope or None,
        )
    except IndexingCancelled:
        # 用户在管线编码阶段点停止：编码线程已在批次边界退出，无生成内容可持久化。
        # 2026-09-07 修复：原为裸 `return`（返回 None）→ 端点声明 -> StreamingResponse
        # 却输出 JSONResponse(null)（200 / application/json），与 text/event-stream 契约不符，
        # 前端 SSE 流解析异常。改为返回一条标记 cancelled 的空 SSE 流。
        # N13 修复（2026-09-12）：**此路径同样必须保住用户提问**——用户在编码等待期间点停止
        # 同样会导致整轮（含提问）消失，与另两条中断路径口径统一。
        logger.info("会话 %s 的管线编码已被客户端中断", session_id)
        if not body.regenerate:
            # 注意：此处早于 `user_content` 的解析（:564），故用同一构造函数现场编码，
            # 口径与下方各路径一致（含多模态图片块）。
            chat_svc.persist_user_message(
                session_id, _build_user_content(body.content, body.images)
            )

        async def _cancelled_stream():
            yield sse_event("done", {"steps": 0, "total_tokens": 0, "cost": 0.0,
                                     "latency_ms": 0, "cancelled": True, **_done_extra})

        return StreamingResponse(_with_stream_cleanup(_cancelled_stream()), media_type="text/event-stream")
    finally:
        build_watch.cancel()
    # 多模态校验（2026-08-25）：带图消息仅多模态模型可处理；纯文本模型直接给明确错误，
    # 避免图片块被静默忽略或厂商 400。多模态集合见 lightcore.llm.router.MULTIMODAL_MODELS。
    # FakeLLM 等本地替身不会真正把图片发给厂商，放行（本地演示/测试下带图全链路不崩）。
    if body.images:
        from lightcore.llm.router import MULTIMODAL_MODELS

        if not isinstance(agent.llm, FakeLLM):
            llm_model = getattr(agent.llm, "model_name", "") or ""
            if llm_model not in MULTIMODAL_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"当前模型 {llm_model or '（未配置）'} 不支持图片输入；"
                        "请切换到 qwen3-vl-flash 或 deepseek-v4-flash-vision-exp 后再发送图片。"
                    ),
                )
    # M4 修复（2026-09-10）：请求体校验已全部通过，此处注册本请求的取消令牌
    # （延后注册的原因见上方 cancel_flag 处注释）。此后的路径要么返回 SSE 流
    # （由 _with_stream_cleanup 负责注销），要么抛异常——都不再会留下残留条目。
    request.app.state.stream_cancels[_stream_key] = cancel_flag

    # 若会话绑了文档 → 强制第一轮 THINK 必须调工具，避免 LLM 因历史上下文拒接
    force_choice: str | dict | None = "required" if chat_svc.should_force_tool_call(session_id) else None

    # Phase B：跨会话记忆功能开关（features.memory）。关闭时既不注入长期记忆注记，
    # 也不在轮末反思写入——记忆模块整体停用（满足"后端启用开关"约束）。
    _feat = runtime_settings.get_state().get("features", dict(runtime_settings.DEFAULT_FEATURES))
    memory_enabled = _feat.get("memory", True)

    # P1-1：显式意图分类，作为 SYSTEM 注记注入本轮（帮助 LLM 稳定理解意图）
    from ..services.intent import intent_system_note

    intent_note = ChatMessage(role=Role.SYSTEM,
                              # ⑤：与上面的 classify_intent **同一实参**，防路由与注记分裂
                              content=intent_system_note(body.content,
                                                         has_doc_mention=_has_doc_mention))

    # P1-2.1：回复逐条核验（受 features.verify_reply 开关控制，默认关）。
    # L1 引用文本↔页码确定性校验零 LLM；L2 LLM 语义核验超时/失败自动降级 skipped。
    # 只产出只读的 verify 事件（前端折叠展示），绝不改写 final_text，失败静默不阻断。
    async def _run_verify(user_text: str, final: str, cites: list, llm) -> dict | None:
        """P1-2.1 回复核验。**2026-09-16 调整：L1 常开、L2 随开关。**

        - **L1（确定性、零 LLM、零延迟）**：只要本轮有引用就**必跑** —— 它抓的是最伤研究场景的错
          （页码标错 / 引用文本不在该页 / 引用无法定位）。此前它被 `verify_reply=False` **连坐关闭**，
          等于把最便宜的一道防线跟着最贵的（L2）一起关掉了。
        - **L2（+1 次 LLM 调用）**：仍由 `features.verify_reply` 控制（默认关，省成本省延迟）。
          实现上通过**传 `llm=None`** 跳过（`verifier.verify_reply` 按 `llm` 是否为空决定跑不跑 L2）。
        - **下发策略**：开关关闭时只下发 `suspect`（`only_suspects` 过滤），`warn` 级静音 ——
          否则 docx/md 这类无物理页码的文档每轮都会弹「无法核验页码」，导致警示疲劳；
          开关打开时维持原行为（L1 + L2 全量下发）。
        """
        l2_on = bool(_feat.get("verify_reply", False))
        if not cites:
            # ★★ 2026-09-23：doc-bound 会话里的「零引用」本身就是一个信号。
            #
            # 此前这里直接 return None（理由：没有引用就没得核验）—— 但这个理由有个致命盲区：
            # **先验知识幻觉的特征恰恰是「回答里没有任何引用」**（文档里没有该内容，自然引不出来）。
            # 于是「最需要核验的场景」反而静默通过，用户看到一个通顺、专业、无引用的回答，
            # 无从察觉它来自模型的训练先验而非他的文档。
            #
            # 这就是「强制检索 ≠ 强制采信」的落点：协议能强制模型**调用** retrieve，
            # 但没有任何机制能强制它**采信**结果；零引用检测是**事后**唯一能兜住的一层。
            #
            # 两个限定，避免警示疲劳：
            #   ① 只在 doc-bound 会话（未绑文档时本就不该要求引用）；
            #   ② 只在回答有实质长度时（"好的""收到"这类短答不提示）。
            # ⚠️ 这里**重算** doc-bound 而不用外层 `_doc_bound`：直答路径的调用点（861 行）
            #    在 `_doc_bound` 赋值（899 行）**之前**，直接引用会 NameError。
            from ..services.verifier import zero_citation_notice

            _notice = zero_citation_notice(
                doc_bound=chat_svc.should_force_tool_call(session_id),
                final_text=final,
                has_cites=bool(cites),
            )
            if _notice is not None:
                return _notice
            if not l2_on:
                # 非 doc-bound（或回答过短）→ 保持原行为：不发空的 verify 事件，保持事件流干净
                return None
        try:
            from ..services.verifier import only_suspects, verify_reply

            result = await verify_reply(
                user_query=user_text,
                final_text=final,
                citations=cites or [],
                db=chat_svc._Session,
                llm=(llm if l2_on else None),
            )
            return result if l2_on else only_suspects(result)
        except Exception:  # noqa: BLE001  核验失败静默降级，不阻断对话
            return None

    # 组装本轮 user 消息（带图时为 OpenAI 兼容内容块列表），供通用问答快速路径与 Agent 循环共用。
    user_content: str | list[dict] = _build_user_content(body.content, body.images)

    cancel_flag.clear()  # 新流开始：清掉上一轮残留的停止信号

    async def _watch_disconnect(q: asyncio.Queue) -> None:
        """每 0.3s 轮询「客户端断连 OR 显式停止事件」；命中 → 投递停止哨兵（消费端取消生成）。"""
        while not (await request.is_disconnected() or cancel_flag.is_set()):
            await asyncio.sleep(0.3)
        q.put_nowait(_STREAM_STOP)

    async def _direct_answer_stream(intent_name: str):
        """通用问答 / 寒暄：直接走 LLM 流式，不进入 Agent 工具循环，不展示推理过程。"""
        system = _CHITCHAT_SYSTEM if intent_name == "chitchat" else _GENERAL_QA_SYSTEM
        messages: list[ChatMessage] = [ChatMessage(role=Role.SYSTEM, content=system)]
        note = chat_svc.recall_note(session_id, body.content) if memory_enabled else None
        if note is not None:
            messages.append(note)
        messages.extend(history)
        messages.append(ChatMessage(role=Role.USER, content=user_content))

        final_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        # 是否采到**真实** usage（方案 D′）：FakeLLM 剧本分支的 token 是 `_rough_tokens` 估算，
        # 不能回写成 `last_prompt_tokens` —— 那会让下一轮的压缩判据重新依赖估算。
        real_usage = False
        # 流式实测延迟（2026-09-16，流式用量采集）：来自 finished 块。
        # 此前直答路径把 latency_ms 硬编码为 0，导致该路径产生的 Trace 耗时恒 0。
        stream_latency = 0

        # FakeLLM 默认剧本会触发 retrieve；通用问答路径必须给出确定性文本回答，避免把工具提示带回。
        # 文案本身也要遵守「无过程叙述」规范：不描述判断/工具/检索行为。
        if isinstance(agent.llm, FakeLLM):
            final_text = (
                "你好！很高兴为你服务，有什么可以帮你的吗？"
                if intent_name == "chitchat"
                else f"关于「{body.content.strip()[:80]}」：这是基于通用知识的直接回答（演示模式，未接入真实大模型）。"
            )
            for seg in FakeLLM._chunk_text(final_text):
                yield sse_event("message_delta", {"text": seg})
            prompt_tokens = sum(_rough_tokens(m.content) for m in messages)
            completion_tokens = _rough_tokens(final_text)
        else:
            web_search = bool(cfg.get("enable_network")) and bool(
                getattr(agent.llm, "supports_native_web_search", lambda: False)()
            )
            # 2026-09-07：与 Agent 路径同款断连取消——LLM 流式期间客户端停止/关页，
            # watcher 投递哨兵 → 消费端中断 → pump 取消（CancelledError 打断 LLM await）。
            _q: asyncio.Queue = asyncio.Queue()

            async def _pump_direct() -> None:
                try:
                    async for ev in agent.llm.stream(messages, temperature=0.7, web_search=web_search):
                        await _q.put(ev)
                except asyncio.CancelledError:
                    logger.info("直答流已被客户端中断，LLM 生成已取消 session=%s", session_id)
                    raise
                except Exception as e:  # noqa: BLE001  中途失败必须显式投递，不得被 finally 掩盖（见 _StreamFailure 的说明）
                    # 在 pump 内就记日志：即使消费端已随客户端断开，traceback 也一定落盘。
                    logger.exception("直答流生成中途失败 session=%s model=%s", session_id, model)
                    await _q.put(_StreamFailure(e))
                finally:
                    await _q.put(None)

            _watcher = asyncio.create_task(_watch_disconnect(_q))
            _pump = asyncio.create_task(_pump_direct())
            _cancelled = False
            _failed: BaseException | None = None
            try:
                while True:
                    ev = await _q.get()
                    if ev is _STREAM_STOP:
                        _cancelled = True
                        break
                    if isinstance(ev, _StreamFailure):  # P1-11：必须先于 `None` 判定（两者都在队列里）
                        _failed = ev.exc
                        break
                    if ev is None:
                        break
                    if ev.delta:
                        yield sse_event("message_delta", {"text": ev.delta})
                        final_text += ev.delta
                    if ev.finished:
                        prompt_tokens = ev.prompt_tokens or 0
                        completion_tokens = ev.completion_tokens or 0
                        real_usage = True  # 来自厂商的 finished 块 → 可作方案 D′ 的跨轮真值
                        # 用 getattr：流式块在测试/替换实现里是**鸭子类型**的（只保证必要字段），
                        # 直接取新字段会让这类替身当场 AttributeError（2026-09-16 回归过一例）。
                        stream_latency = getattr(ev, "latency_ms", 0) or 0
            finally:
                _watcher.cancel()
                if not _pump.done():
                    _pump.cancel()
            if _failed is not None:
                # P1-11（2026-09-15）：**生成中途异常必须显式告知失败**。
                # 原实现下该异常被 `finally` 的哨兵掩盖成「正常结束」——有部分输出时既无 notice
                # 也无 error 事件，用户看到半截回答却以为已完成；且异常从未被 await 取回。
                # 现口径（对齐 N13「关键数据优先落库」）：
                #   ① 无任何输出 → 落既有兜底文案（历史里不留空白）；
                #   ② 有部分输出 → 连同 notice 落库，刷新后仍能看到「本次回答残缺」；
                #   ③ 无论哪种，都发 `error` 事件（前端立即转错误态），并**跳过后续 LLM 调用**
                #      （reflect / 自动标题 / 逐条核验），避免在失败轮次上继续产生费用。
                partial = (final_text or "").strip()
                if not partial:
                    final_text = _FALLBACK_REPLY
                # 直答失败的具体原因：来自 pump 捕获的异常（原样截断由 notice/payload 内处理）
                _failed_reason = str(_failed) if _failed is not None else ""
                # `invalid` 仅在**完全无输出**时置位：部分输出不算「无效回复」，
                # 不参与「连续无效 → loop_risk 强打断」计数（避免偶发截断被误升级为死循环风险）；
                # 前端仍会因 meta.notice 存在而渲染警示卡（见 historyToThread）。
                chat_svc.persist_turn(
                    session_id, user_content, history,
                    # ★ T-09+：直答路径此前**传空 events** ⇒ 失败在 trace 里毫无痕迹
                    #   （只能靠"没有 done"推断，与用户主动停止混为一谈）。
                    #   现在落一条**带归因**的 error 事件。
                    # ⚠️ 必须是 `LoopEvent` 对象：下游（`chat_service`）按 `ev.type` **属性**
                    #    访问，传裸 dict 会 AttributeError（2026-09-30 实测踩到）。
                    [LoopEvent(type="error", state=LoopState.ERROR_HANDLE,
                               data={"reason": _failed_reason,
                                     "failure_kind": m_failure.CRASH})],
                    final_text, model,
                    cost_override=estimate_cost(model, prompt_tokens, completion_tokens),
                    # 直答路径本来就有真实分项用量（来自 finished 块）→ 直接落库，
                    # 不再让 persist_turn 退回字符粗算（2026-09-16 问题 2 修复）。
                    prompt_tokens_override=prompt_tokens,
                    completion_tokens_override=completion_tokens,
                    # 直答路径不检索 → 本轮无引用（如实落空，而不是留字段缺失）
                    citations=[],
                    latency_ms=stream_latency,
                    regenerate=body.regenerate,
                    meta={"invalid": not partial, "notice": _stream_error_notice(_failed_reason, m_failure.CRASH)},
                )
                yield sse_event("error", _stream_error_payload(_failed_reason, m_failure.CRASH))
                return
            if _cancelled:
                # 客户端已断开：跳过 reflect/自动标题/核验等额外 LLM 调用；已生成的部分内容照常持久化。
                # N13 修复（2026-09-12）：**中断也必须保住用户提问**。
                # 原先「无已生成内容则不落库」→ 用户发问后模型尚未产出任何 delta 就点「停止」时，
                # 整轮（**含提问本身**）全部消失，用户被迫重复输入（用户实测反馈）。
                # 现改为：
                #   ① 有部分内容 → 照旧落 user + 部分 assistant（meta.interrupted）；
                #   ② 无任何内容 → **只落 user 一条**（不写空 assistant），提问不再丢失。
                partial = (final_text or "").strip()
                cost = estimate_cost(model, prompt_tokens, completion_tokens)
                if partial:
                    chat_svc.persist_turn(
                        session_id, user_content, history, [], partial, model,
                        cost_override=cost,
                        regenerate=body.regenerate,
                        # ★ T-09+：停止也给**中性回执**（不是告警），刷新后仍能看到"这是我停的"
                    meta={"interrupted": True, "notice": _user_stop_notice()},
                    )
                elif not body.regenerate:
                    # 「重新生成」场景该 user 消息已存在，跳过以免重复
                    chat_svc.persist_user_message(session_id, user_content)
                return

        # 「打断提醒」：兜底回复携带结构化 notice（循环检测可能升级为 loop_risk）。
        # 2026-09-07 修复：必须**先**判定原回复是否为空，再写入兜底文案——
        # 此前先赋 _FALLBACK_REPLY（非空）再判空，条件恒 False → notice 恒 None，
        # 导致 loop_risk 强打断在兜底场景从未触发（死循环复盘机制形同虚设）。
        _was_empty = not final_text.strip()
        if _was_empty:
            final_text = _FALLBACK_REPLY
        notice = _invalid_notice("fallback", _trailing_invalid) if _was_empty else None
        yield sse_event("message", {"text": final_text, "citations": []})
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        chat_svc.persist_turn(
            session_id, user_content, history, [], final_text, model,
            cost_override=cost,
            # 分项用量同样落精确值（见上）
            prompt_tokens_override=prompt_tokens,
            completion_tokens_override=completion_tokens,
            citations=[],  # 直答路径不检索 → 无引用
            latency_ms=stream_latency,
            regenerate=body.regenerate,
            meta=_persist_meta(notice, model_risk=model_risk),
        )
        if final_text and memory_enabled:
            await chat_svc.reflect(
                session_id, agent.llm,
                _reflect_messages(history, body.content, final_text),
            )
        await chat_svc.auto_title(session_id, agent.llm, body.content)
        # P1-2.1：直接回答路径同样支持逐条核验（开关默认关，只读不改写）
        v = await _run_verify(body.content, final_text, [], agent.llm)
        if v:
            yield sse_event("verify", v)
        # 方案 D′：**直答路径不经过 AgentLoop**，必须在这里回写跨轮基线 ——
        # 否则所有非 doc-bound 会话（走直答）永远没有真实值，D′ 对它们完全失效。
        # 仅真实 usage 才写：FakeLLM 剧本分支的 token 是估算，写进去会让判据重新依赖估算。
        # 直答路径**不执行本地工具** ⇒ prompt 里没有"本轮专属"的检索结果，
        # 故这里的 `prompt_tokens`（该次调用）与首步口径等价，可直接作为基线。
        _write_back_prompt_baseline(prompt_tokens if real_usage else 0)
        yield sse_event("done", {
            "steps": 0,
            # 2026-09-16：改为**本轮**用量。此前这里给的是 count_usage 的**会话累计**，
            # 而前端 ChatMessageView 是按"本条消息"渲染 tokens/成本 → 每条直答都显示累计值，
            # 与 Agent 路径（本轮）口径不一致。两条路径现在统一为"本轮"。
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
            # 实测延迟（此前硬编码 0 → 该路径 Trace 耗时恒为 0）
            "latency_ms": stream_latency,
            # O-12：服务端首字延迟（未产出即结束则为 None，不伪造）
            "ttft_ms": _ttft_ms(),
            **({"notice": notice} if notice else {}),
            **_done_extra,
        })

    async def gen():
        # B8（2026-09-29）：**首个 SSE 字节之前**先发一条保活注释帧。
        # 该窗口内真实可能发生：**管线加载**（历史实测单篇可达 **33s**）+ **历史摘要**（≤20s）
        # ⇒ 期间**零字节下行**，本地单用户风险低，但**代理/网关易判超时并断开**。
        #
        # 为什么是 SSE **注释帧**（`: ` 开头）：
        #   ① 前端 `web/src/lib/sse.ts:55` 已有 `if (line.startsWith(":")) continue;`，
        #      且 `parseFrame` 在"无 event 且无 data"时返回 `null` ⇒ **被直接丢弃、不产生任何 UI 事件**；
        #   ② 它不是 `LoopEvent`、**不进 `events`** ⇒ 不影响 trace / 用量 / 引用任何口径。
        #
        # ⚠️ 放在 `gen()` 的**第一条语句**：覆盖**所有**分支（agent / 技能 / 直答），
        # 而不只是"管线加载"那一条 —— 代价 ~20 字节，收益是"任何慢启动都不会静默"。
        # ⚠️ 三个**快速返回**路径（超长输入 / 缺文档 / 已取消）用的是**各自的**生成器
        # （`_too_long_stream` 等），不经过这里；它们本就立即返回、无静默窗口，故无需保活。
        yield ": keepalive\n\n"
        events: list = []
        citations: list = []
        # N1 修复（2026-09-10）：引用**跨工具调用**去重 + 总数上限。
        # 原先每次 tool_result 都 `citations.extend(clean_citations(...))`，而它
        # 只在**单次调用内**去重（每次新建 seen）且无总数限制——agent 多轮 × 多次检索 × TopK
        # 累加后引用可达数百至上千条（用户实测「引用来源显示几千条」），且大量重复。
        citation_keys: set[str] = set()
        # B5（2026-09-29）：本轮引用是否**因达上限被截断**（`dedup_citations` 在确有
        # 新增引用被丢时 append(True)）。仅作**观测**：落 trace / done 载荷，不改任何行为。
        _citations_capped: list[bool] = []
        final_text = ""
        # F7（2026-09-07）：本轮是否发生「检索故障」（全部管线失败）——
        # 与「检索到但无命中」区分，用于产出独立的 retrieval_error 提示。
        _retrieval_error = False
        # 通用问答 / 寒暄直接走 LLM 快速路径：不强制检索、不输出工具推理过程，像普通 AI 一样回答。
        # 2026-09-03 修正：doc_qa 之外，若会话已绑定文档（should_force=True）且只是 general_qa，
        # 也**不**走直答——否则模型会基于历史自由发挥并编造「原文第N页」式幻觉引用（实测 E-YOLO
        # 查询产出『第 632–634 页』，实际文档仅 1–11 页）。doc-bound 会话一律进 Agent 循环，
        # 由首轮强制检索提供接地内容；仅纯寒暄仍直答（避免无谓检索）。
        # ★ ⑤（2026-10-02）：**复用上方已算好的 `it`**，不再第二次 `classify_intent`。
        # 为什么：同一次请求里算两遍就是**两把尺子** —— 守卫（缺文档拦截）与路由/日志可能
        # 看到**不同**的意图；实测反向自检时我只改第一处，日志里的 intent 纹丝不动，
        # 才发现这两处是独立计算。上方注释本就写着"后续路径复用，避免重复规则判断"，
        # 这里此前并未真正复用。
        _doc_bound = chat_svc.should_force_tool_call(session_id)
        # ★ 2026-09-23：意图分类结果必须**可观测**。
        # 此前它只参与分支判断、不留任何痕迹 —— 于是「这轮为什么走了直答」事后无法回答，
        # 排查「文档约束为何失效」时只能靠翻代码推断，定位不到具体某一轮。
        # 一行日志换来「可回溯路径」，这正是那次诊断里最缺的东西。
        _route = (
            "direct"
            if (it.name == "chitchat" or (it.name == "general_qa" and not _doc_bound))
            else "agent"
        )
        logger.info(
            "意图路由 session=%s intent=%s doc_bound=%s → %s",
            session_id, it.name, _doc_bound, _route,
        )
        # ★★ M1（2026-10-02）：`@文档N` 的**范围**也必须进日志 —— 事故排查时
        # 唯一能证明"限定到底生效没有"的地方（此前 retrieve 的 args 里**从不**出现
        # doc_ids，正是"看似 @ 了、实际全库盲搜"的唯一证据）。
        # `source=text` 表示由**手打文字**回退解析而来（前端弹窗才会发结构化字段）。
        if mention_ids or mention_invalid_label:
            logger.info(
                "[文档限定] session=%s 生效 | 标签=%s doc_ids=%s | 来源=%s%s",
                session_id,
                doc_label(mention_order(_base_ids).index(mention_ids[0]))
                if len(mention_ids) == 1 and mention_ids[0] in _base_ids else "多篇",
                mention_ids,
                "结构化" if body.mention_doc_ids else "手打文本回退",
                f" | ⚠️{mention_invalid_label} 越界（本会话仅 {len(_base_ids)} 篇）"
                if mention_invalid_label else "",
            )
        if it.name == "chitchat" or (it.name == "general_qa" and not _doc_bound):
            async for ev in _direct_answer_stream(it.name):
                yield ev
            return
        # Phase C：命令触发的复合技能（受 features.skills 门控）。命中则在 Agent 循环之外直接编排，
        # 失败明示、不静默；未命中（或技能关闭）回落到普通 Agent 循环。
        from lightcore.skills.spec import SkillContext

        from ..skills import default_router

        skill_router = default_router()
        skill = skill_router.route(body.content, _feat)
        # ★★ P0-2（2026-10-03）：**用户敲了 `/` 开头却没命中** ⇒ 必须告知，不得静默。
        #
        # 原先这里**没有 else**：未命中就直接回落普通 Agent 循环，
        # 用户看到一段普通回答，**无法区分**"技能没触发"与"技能触发了但答得不好"
        # （见 `docs/诊断-命令功能代码审查-2026-10-03.md` §1.2）。
        # ⚠️ **不对称**：同一文件对**技能执行失败**是明示的（下游 `yield sse_event("error")`），
        # 唯独"命令没命中"什么都不说。
        #
        # ⚠️ **判据要保守**：只在"看起来像命令"时才提示
        # （`/` 后紧跟非空白字符），避免把 `/` 开头的普通文本误报。
        # ⚠️ **不放进 `notice`**：notice 会被落库成 `invalid=True`（见 `_persist_meta`），
        # 而"打错命令"不该让该轮计入**连续无效**（那是给"模型给出无效回答"用的口径）
        # ⇒ 用**独立字段** `unrecognized_command`。
        unrecognized_cmd: str | None = None
        if skill is None:
            _text = (body.content or "").lstrip()
            if _text.startswith("/") and len(_text) > 1 and not _text[1].isspace():
                _tok = _text.split()[0] if _text.split() else _text
                # 可用命令：只列**当前启用**的（列了禁用的会误导用户去用）
                _avail = [
                    _s.commands[0]
                    for _s in skill_router._skills
                    if _s.commands
                    and (not _s.feature or _feat.get(_s.feature, True))
                ]
                unrecognized_cmd = (
                    f"未识别为技能命令「{_tok}」，已按**普通问答**处理。"
                    f"可用命令：{' '.join(_avail)}"
                )
                logger.info(
                    "命令未命中 session=%s token=%s（按普通问答处理，已提示用户）", session_id, _tok,
                )
        if skill is not None:
            args = skill_router.parse_args(body.content, skill)
            # A3（2026-09-28）：补上"**实际走的是技能**"这一条日志。
            # `_route` 是**在技能路由之前**算的，只会给出 `direct | agent`；
            # 于是走 `/总结` `/术语表` 这类命令时，日志写的是 `intent=command ... → agent`，
            # 而**实际执行的是技能**（技能在 `skill.run(ctx)` 里直接调 LLM、不经 AgentLoop）
            # ⇒ 日志把真实路径**盖住了**。
            # 实测代价：本次审查排查 `/术语表` 时，我正是被 "→ agent" 这行误导，
            # 以为它在走检索循环，花在错误方向上的时间比定位本身还多。
            # 修法刻意只补日志、不改 `_route` 的取值（改取值会影响它对下游分支的语义）。
            logger.info(
                "意图路由（技能覆盖）session=%s intent=%s → skill:%s（原判定=%s，技能不经 AgentLoop）",
                session_id, it.name, getattr(skill, "name", type(skill).__name__), _route,
            )
            # 会话级『联网』开关：开启时联网类技能走大模型原生联网搜索（工具检索方式已移除）
            enable_network = bool(cfg.get("enable_network", False))
            # 学科视角：会话级 discipline 优先于全局默认（runtime_settings.default_discipline），
            # 详见 docs/技能学科适配器设计方案.md。非法/缺失经 get_profile 回落 general（零注入）。
            # "auto"：按本次查询内容启发式自动识别学科（零 LLM 调用，仅启发式关键词）。
            # 学科已在入口统一解析（resolved_discipline），此处直接复用，避免重复检测
            discipline = resolved_discipline
            # T-01-D：技能直接调 LLM、不经 AgentLoop → 用录制壳捕获真实 prompt 峰值，跑完回写。
            # 惰性导入 is_real_llm：与下面 `from ..skills import default_router` 同一风格，
            # 避免在模块加载期就把 skills 包拉进来（潜在环形导入）。
            from ..skills._common import is_real_llm

            recorder = _UsageRecorder(agent.llm) if is_real_llm(agent.llm) else None
            # 技能流式化（2026-09-28）：技能内部统一走 `llm.chat()`（一次性返回），路由层此前
            # 也是一次性 `yield message` ⇒ 用户看到「整段突然刷出」，与问答路径（逐
            # `message_delta`）观感不一致。这里在**路由层**包一层代理：`chat()` 内部改走底层
            # `stream()`，增量实时进队列、由消费端逐块下发 —— **12 个技能代码零改动**。
            # FakeLLM 不包装（`is_real_llm` 为假）：输出确定性、无增量可言；且套壳会让技能
            # 误判「有真实模型」而改变行为（同 `_UsageRecorder` 的理由）。
            skill_q: asyncio.Queue = asyncio.Queue()
            skill_llm = recorder or agent.llm
            if is_real_llm(agent.llm):
                skill_llm = _StreamingChatProxy(skill_llm, skill_q)
            ctx = SkillContext(
                query=  body.content, args=args,
                doc_ids=effective_doc_ids,
                summary_svc=chat_svc._summary, doc_svc=chat_svc._docs,
                llm=skill_llm, features=_feat,
                db=chat_svc._Session,
                enable_network=enable_network,
                discipline=discipline,
                # C1：工作间会话注入主文献（草稿），/改稿 缺省与回落路径优先采用
                draft_doc_id=cfg.get("draft_doc_id") if isinstance(cfg.get("draft_doc_id"), int) else None,
                mention_doc_ids=mention_ids,
            )
            # 2026-09-07：技能执行也接入中断（单次 LLM await 期间点停止 → 立即取消）
            # ⚠️ 2026-09-27 **P0 修复**（死锁详情见 `_stream_skill_output` 的 docstring）：
            # 此处原为 `while True: if skill_task.done(): ... break` —— **循环体内零 `await`**，
            # 独占事件循环 ⇒ `skill_task` 永不调度 ⇒ `done()` 恒 False ⇒ 永久死锁 + 单核 100%
            # （后端僵住、199 个 CLOSE_WAIT、日志静止 23 分钟）。修复要点两条：
            # ① 轮询**必须 `await`**；② 「断连 → 取消技能」必须**在循环内**（原实现写在外面，
            # 从未生效过）。现在这两点都在 `_stream_skill_output` 内保证，并由单元测试钉住。
            skill_task = asyncio.create_task(skill.run(ctx))
            # ★ 2026-10-23：技能是**一次性 return**（`run() -> SkillResult`），内部多步不产增量
            #   ⇒ 用户点命令后长时间空白、最后整段刷出（`message_delta` 管道在、但技能不往里写）。
            #   这里先推一条 **progress** 让前端立刻显示"正在执行 /综述…"，至少不是空白。
            #   真正的"逐段进度"需要把 12 个技能改成生成器（另立任务）。
            yield sse_event("progress", {
                # ★ 前端 reducer 读的是 `phase` + `text`（`web/src/lib/reducer.ts:31`，O-12 首字前进度）
                #   ⇒ 字段名必须对齐，否则只会显示空 phase。
                "phase": "running_skill",
                "skill": getattr(skill, "name", type(skill).__name__),
                "text": f"正在执行 /{getattr(skill, 'name', '技能')}…",
            })
            try:
                async for _delta in _stream_skill_output(
                        skill_task, skill_q, request, cancel_flag, session_id):
                    yield sse_event("message_delta", {"text": _delta})
            finally:
                # 消费端提前退出（断连 / 停止 / 下游异常）时必须收掉技能 task：
                # 否则它会在后台继续把 LLM 调用跑完，白耗额度。
                if not skill_task.done():
                    skill_task.cancel()
            if skill_task.cancelled() or await request.is_disconnected() or cancel_flag.is_set():
                return   # 已被客户端中断：不发 message、不落库（与改造前 `return` 语义一致）
            # 2026-09-07 修复（保留）：技能内部异常若不捕获，会随 .result() 重新抛出，
            # 冲垮 SSE 生成器 → 客户端收到截断流且无 error 事件（只看到空白）。
            # 转为 error 事件明示原因，与 result.ok=False 分支行为一致。
            try:
                result = skill_task.result()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  技能实现异常：转 error 事件，不中断流
                logger.warning("技能执行失败 session=%s", session_id, exc_info=True)
                yield sse_event("error", {"reason": f"技能执行失败：{e}"})
                return
            # T-01-D：技能跑完回写**真实**基线（Fake 未包装 → recorder 为 None → 自动跳过，
            # 与"估算值绝不回写"的既有约定一致）。
            # 用 `first_prompt_tokens` 而非峰值：技能把检索 chunk 拼进 prompt，
            # 那部分不会进入下一轮历史 ⇒ 峰值会系统性高估（见 `_UsageRecorder` 说明）。
            if recorder is not None:
                _write_back_prompt_baseline(recorder.first_prompt_tokens)
            yield sse_event("tool_call", {"call_id": f"skill_{skill.name}", "name": skill.name,
                                          "arguments": {"args": args}})
            # 无文档警示：会话未关联任何文档时，技能无法基于论文内容，
            # 只能靠大模型自身知识作答（可能不准确、无溯源）。按需求在输出前加警示提醒。
            no_doc_warn = (
                "⚠️ 警示：当前会话未关联任何文档，以下结果仅由大模型基于自身知识生成，"
                "未引用任何论文，准确性与时效性无法保证。建议先到右栏「关联文档」勾选文档后再使用技能。"
            ) if not effective_doc_ids else None
            if result.ok:
                final_text = result.content
                # ★ B2-X2（2026-10-01）：技能引用此前**完全绕过清洗** ——
                #   `skills/_common.citations_from` 只做 `normalize_page`（长度/噪声/去重全无），
                #   `literature_review` / `related_work_gen` 更是直接塞**整块 chunk 文本**。
                #   后果：超长片段进前端后被 `>600` 静默丢弃，而落库与 `citation_rate` 仍计入
                #   ⇒ 「指标有数、用户看不到」。现在统一走**唯一清洗入口**（与 agent 路径同一实现）。
                citations = clean_citations(result.citations or [])
            else:
                yield sse_event("error", {"reason": result.error})
                # 用技能返回的具体错误原因作为可见文案，避免叠加误导性通用兜底
                # （如「请先关联文档」比「模型调用失败」更准确，error 事件仍承载结构原因）
                final_text = (result.error or "").strip()
            # ★ O-02（2026-10-01）：技能路径此前**只发 `tool_call`、从不发配对的 `tool_result`**
            #   （全文件唯一一处 `tool_call`，零处 `tool_result`）⇒ 前端 reducer 只有
            #   `tool_result` 才会把卡片从「执行中」收敛（`done` 分支**不处理** tool 状态）
            #   ⇒ 每次使用技能都留一张**永久转圈**的工具卡（必现）。
            #   ⚠️ `call_id` 必须与上面的 `tool_call` **严格一致**（前端靠它匹配）；
            #   ⚠️ 上方"技能实现异常"分支发生在 tool_call **发出之前**（error + return），
            #      本就没有发过 tool_call ⇒ 不需要配对，不会产生"孤儿 result"。
            #   ⚠️ payload 形态与 agent 路径的 `tool_result` 一致（`ok` + `output`/`error`），
            #      卡片内容刻意用一句简短说明 —— 技能的正文已由 `message` 事件承载。
            yield sse_event("tool_result", {
                "call_id": f"skill_{skill.name}",
                "name": skill.name,
                "ok": bool(result.ok),
                **({"output": {"note": "技能执行完成"}} if result.ok
                   else {"error": (result.error or "")[:200]}),
            })
            # 无文档时把警示前缀拼到可见文案最前（仅当有实质内容，避免与兜底文案叠加）
            if no_doc_warn and final_text.strip():
                final_text = f"{no_doc_warn}\n\n{final_text}"
            # 空回复兜底：仅当技能既无内容、又无错误原因时，给默认文案（绝不空白）
            # 2026-09-07 修复：先记录 _was_empty 再赋兜底文案（否则判空恒 False、notice 恒 None）
            _was_empty = not final_text.strip()
            if _was_empty:
                final_text = _FALLBACK_REPLY
            # 「打断提醒」：技能兜底回复携带结构化 notice（循环检测可能升级为 loop_risk）；
            # 技能返回具体 error 文案（有实质内容）不算无效回复，不挂 notice。
            notice = _invalid_notice("fallback", _trailing_invalid) if _was_empty else None
            yield sse_event("message", {"text": final_text,
                "citations": [{"id": c.get("id"), "page": normalize_page(c.get("page")), "doc_id": c.get("doc_id"), "text": c.get("text"), "model_truncated": c.get("model_truncated", False)} for c in citations]})
            # ★ P-03（2026-10-09）：**技能轮也上报上下文占用**。
            #   背景：技能在 `skill.run` 里直接调 LLM、不经 AgentLoop ⇒ 原先 done 事件里没有
            #   `peak_prompt_tokens`/`max_tokens_budget`，`meta.ctx` 又依赖恒为空的 `events`
            #   ⇒ 前端「上下文用量」面板**对命令轮完全不计数**（用户实测反馈）。
            #   数据其实早已捕获：`_UsageRecorder`（上方 `recorder`）在包装 LLM 时记录了
            #   `peak_prompt_tokens`/`first_prompt_tokens`，只是**没往外发**。
            #   ⚠️ 双尺子警告：技能轮的 prompt **不含历史/全量文档**（只发候选+上下文），
            #      所以这里的数值天然**小于**普通问答轮 —— 这是**真实值**，不要折算、也不要跨轮比较。
            #   ⚠️ 只有 `peak > 0` 才注入：若写成 `peak_prompt_tokens: 0`，前端 `??` 会短路
            #      掉历史回填（`ctx.peak`），把"无数据"显示成 0%。
            _ctx_payload: dict = {}
            if recorder is not None and recorder.peak_prompt_tokens > 0:
                _ctx_payload = {
                    "peak_prompt_tokens": recorder.peak_prompt_tokens,
                    # ★★ 2026-10-23：这里原来直接取 `get_state()["max_tokens_budget"]`（原始 30000），
                    #   而普通路径的 AgentLoop 用的是**解析后**的预算（`resolve_budget_for(model)`，
                    #   按模型窗口 auto 推算，deepseek 1M ⇒ 100000）。
                    #   ⇒ 同一份"预算"在两条路径上取值不同 = **第三把尺子** ✗
                    #   表现：技能轮里面板显示 `2.7K / 30K = 9.0%`，而真实是 `2.7K / 100K = 2.7%`
                    #   （用户会误以为上下文快满了）。
                    #   现在统一走 resolve_budget_for，未登记模型时它自己回落原始值，口径一致。
                    "max_tokens_budget": runtime_settings.resolve_budget_for(
                        runtime_settings.get_state().get("default_model")
                    ),
                }
            # 2026-09-07 修复：持久化用 user_content（含多模态图片块）而非 body.content 纯文本，
            # 否则走技能路径（/总结 /对比 等）时用户随消息发送的图片不会写入历史，
            # 刷新后图片上下文丢失（与"多模态历史不丢失"目标矛盾，Agent/直答路径均用 user_content）。
            chat_svc.persist_turn(session_id, user_content, history, events, final_text, model,
                                  regenerate=body.regenerate,
                                  # ★ B2-X1（2026-10-01）：技能路径此前**没传 citations** ——
                                  #   它有 `citations` 变量（上面的 `result.citations`）却从未落库
                                  #   ⇒ 刷新后引用卡片消失（本次修复的目标）。
                                  #   ⚠️ **边界（如实记录，勿当成缺陷）**：技能分支的 `events`
                                  #   收集器始终为空（技能输出直接走 SSE，不进 events）⇒ trace 里
                                  #   没有"完整回答事件"可供挂引用 ⇒ `persist_turn` 如实**不注入**。
                                  #   推论：`citation_rate`（样本 = 完整回答事件）**本就不计技能轮**
                                  #   ⇒ 本参数不会扭曲该指标；要让技能轮进指标需另立任务。
                                  citations=citations,
                                  # ★ P1-a2：把占用率一并落进 meta，供**刷新后回填**
                                  # （否则"当轮看得到、刷新即消失"，与 B2-X1 同类缺陷）
                                  # ★ P-03（2026-10-09）：技能轮改用 `_ctx_payload`（recorder 捕获）——
                                  #   原先传 `_ctx_usage_from_events(events)`，而技能分支的 events
                                  #   收集器**恒为空** ⇒ 技能轮的历史回填永远是 `{}`。
                                  meta=_persist_meta(notice, model_risk=model_risk,
                                                     ctx=_ctx_usage_from_done(_ctx_payload)))
            if final_text and memory_enabled:
                await chat_svc.reflect(
                    session_id, agent.llm,
                    _reflect_messages(history, body.content, final_text),
                )
            # 会话自动标题：默认标题按首轮主题生成（LLM / 规则降级）
            await chat_svc.auto_title(session_id, agent.llm, body.content)
            # P1-2.1：技能路径同样核验（开关默认关）
            v = await _run_verify(body.content, final_text, citations, agent.llm)
            if v:
                yield sse_event("verify", v)
            usage = chat_svc.count_usage(session_id)
            yield sse_event("done", {"steps": 1,
                                     "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                                     "cost": usage["cost"], "latency_ms": 0,
                                     # O-12：技能路径同口径上报服务端首字延迟
                                     "ttft_ms": _ttft_ms(),
                                     # ★ P-03：技能轮的上下文占用（空 dict = 无数据，前端保持不确定态）
                                     **_ctx_payload,
                                     **({"notice": notice} if notice else {}),
                                     **_done_extra})
            return
        # ---- 普通 Agent 循环路径 ----
        # 跨会话记忆：把相关长期记忆作为 SYSTEM 注记前缀注入本轮（受 memory 开关控制）
        note = chat_svc.recall_note(session_id, body.content) if memory_enabled else None
        prefix: list[ChatMessage] = []
        if note is not None:
            prefix.append(note)
        prefix.append(intent_note)
        messages = prefix + history
        done_data: dict | None = None
        # 2026-09-07 修复「模型思考中点停止，后端却还在跑」：
        # 旧版只在事件之间检查 is_disconnected——而 thinking 阶段长时间不产出事件，
        # 检查永不执行；uvicorn 对 POST SSE 也不会主动取消生成器 → LLM 调用与计费
        # 继续到自然结束。现改为：pump 任务承载 Agent 流；watcher 每 0.4s 轮询断连，
        # 断开即向队列投递停止哨兵 → 消费端中断循环并取消 pump（CancelledError 传入
        # run_stream 的 LLM 流式 await，生成与计费立即停止）。
        _q: asyncio.Queue = asyncio.Queue()
        # 2026-09-11 修复：**不要**在此新建局部哨兵——`_watch_disconnect` 投递的是模块级
        # `_STREAM_STOP`（定义见文件顶部，注释见技能路径的 `if ev is _STREAM_STOP`）。
        # 原实现新建局部 `_STOP = object()`（**另一个对象**），`is` 永不成立，导致两处后果：
        #   ① 哨兵落进 `ev = item` → `ev.type` → `AttributeError: 'object' object has no
        #      attribute 'type'`（实测日志 2026-09-11 00:58:13 的 ASGI 500）；
        #   ② **中断语义完全失效**——`loop_cancelled` 永不置位，已生成内容的落盘与
        #      pump 取消等中断处理从不执行（点「停止」后端仍跑到自然结束）。

        # ★ O-12 档 1（2026-10-02）：**首字之前的进度提示**。
        # 为什么需要：doc-bound 首轮的首字延迟实测中位 **≈4.07s**（LLM1 决策 1748ms + 检索 1488ms
        # + LLM2 首 token 832ms，见 docs/O-12-TTFT探针-首字构成实测.md）——这段窗口内**零字节正文**，
        # 用户只看到一个转圈的空气泡。
        # 诚实性：只在 `force_choice` 为真（会话已关联文档 ⇒ 协议级要求必须调工具）时发出 ——
        # 此时"正在检索"是**必然发生的事实**，不是猜测。
        # ⚠️ 为什么**不**用 `message_delta`：那段文字会进 `partial_text` → 落库 → 污染回答正文与引用；
        # 也**不**用 `thinking`：那会虚增"推理过程（N 步）"的计数。故新增独立 `progress` 事件，
        # 前端在流式期间展示、首个正文增量到达即清除（不进正文、不进 trace 语义、不进任何计数）。
        if force_choice:
            yield sse_event("progress", {"phase": "retrieving", "text": "正在检索关联文档…"})

        async def _pump() -> None:
            try:
                async for ev in agent.run_stream(user_content, messages, stream_tokens=True,
                                                 force_tool_choice=force_choice):
                    await _q.put(ev)
            except asyncio.CancelledError:
                logger.info("Agent 流已被客户端中断，LLM 生成已取消 session=%s", session_id)
                raise
            except Exception as e:  # noqa: BLE001  中途失败必须显式投递，不得被 finally 掩盖（见 _StreamFailure 的说明）
                # 注：AgentLoop 自身已把「LLM 调用失败」转为 error 事件（loop.py 的 THINK 段），
                # 走到这里的是**循环保护之外**的异常（消息装配、历史压缩、落库前的处理等）。
                # 在 pump 内记日志：即使消费端已随客户端断开，traceback 也一定落盘。
                logger.exception("Agent 流生成中途失败 session=%s model=%s", session_id, model)
                await _q.put(_StreamFailure(e))
            finally:
                await _q.put(None)

        async def _watch_agent() -> None:
            await _watch_disconnect(_q)

        pump = asyncio.create_task(_pump())
        watcher = asyncio.create_task(_watch_agent())
        loop_cancelled = False
        loop_failed: BaseException | None = None
        partial_text = ""  # 流式增量累计（中断时用于持久化已生成部分）
        # AgentLoop 的**具体失败原因**（余额不足 / token 熔断 / 迭代上限…）——
        # 来自 error 事件，用于 notice 与收尾 error 事件（2026-09-16：此前前端只看到
        # "未生成有效回答"，无法区分"该充值"还是"该调预算"，完全无法自救）。
        stream_error_reason = ""
        # ④（2026-10-02）：**结构化**的失败归因（`core/failure.py` 的枚举）。
        # 为什么必须留下它：原先只有 `stream_error_reason` 这个**自由文本**，
        # 下游只能"按中文子串猜原因"（"熔断"两个字出现在无关文本里就会假匹配）。
        # error 事件本来就带 `failure_kind` ⇒ 记下来，让文案按**枚举**分流。
        stream_error_kind = ""
        # O-12 档 2 · P0 探针：记录**首次** retrieve 的 query 与时刻（只写日志，零行为变更）
        _probe_first_retrieve_ms: float | None = None
        _probe_model_query: str | None = None
        try:
            while True:
                item = await _q.get()
                if item is _STREAM_STOP:
                    loop_cancelled = True
                    break
                if isinstance(item, _StreamFailure):  # P1-11：必须先于 `None` 判定（两者都在队列里）
                    loop_failed = item.exc
                    break
                if item is None:
                    break
                ev = item
                events.append(ev)
                if ev.type == "message" and ev.data.get("delta") is not None:
                    partial_text += ev.data["delta"]
                    yield sse_event("message_delta", {"text": ev.data["delta"]})
                elif ev.type == "message" and ev.data.get("content"):
                    final_text = ev.data["content"]  # 末尾统一带引用发出
                elif ev.type == "done":
                    # 捕获 AgentLoop 回传的精确 done（含真实成本/token），稍后统一发出（保序）
                    done_data = ev.data
                    # ★ P0-2："命令没命中"随 done 下发（前端渲染一条轻提示）。
                    # ⚠️ 用**独立字段**而不是塞进 `notice` —— notice 会连带 `invalid=True`，
                    #   而"打错命令"不该让该轮计入「连续无效」（那是模型无效回答的口径）。
                    #   而"打错命令"不该让该轮计入「连续无效」（那是模型无效回答的口径）。
                    if unrecognized_cmd:
                        done_data["unrecognized_command"] = unrecognized_cmd
                    # B5（2026-09-29）：把"引用是否因达上限被截断"并入 done 载荷。
                    # ⚠️ `done_data is ev.data` 且 `ev` 已在 `events` 里 ⇒ 该字段**随之落 trace**
                    # （`/metrics`、用量页可见），同时经 `**done_data` 出现在 SSE 的 done 里。
                    # 只加字段、不改任何既有字段 ⇒ 非行为变更。
                    done_data["_citations_capped"] = bool(_citations_capped)
                    # ★ O-12（2026-10-02）：把**服务端首字延迟**写进 done 载荷。
                    # ⚠️ `done_data is ev.data` 且 `ev` 已在 `events` 里 ⇒ 该字段**随 trace 落库**
                    # （`/metrics` 与用量页可见），同时经 `**done_data` 出现在 SSE 的 done 里。
                    # 只加字段、不改既有字段 ⇒ 非行为变更（与 `_citations_capped` 同模式）。
                    _ttft_value = _ttft_ms()
                    if _ttft_value is not None:
                        done_data["ttft_ms"] = _ttft_value
                    # 方案 D′：把本轮**跨轮基线**（= 首步真实 prompt）回写 session config，
                    # 供下一轮构造 AgentLoop 时注入，作为压缩触发判据（取代漂移的字符估算）。
                    # ⚠️ 用 `first_prompt_tokens` 而非 `peak_prompt_tokens`（2026-09-28 修正）：
                    # 峰值含本轮工具结果，而那不会进入下一轮历史 ⇒ 高估且只增不减（棘轮）。
                    # 无任何成功调用时 `first_prompt_tokens == 0` ⇒ 跳过，保留上一轮基线。
                    _write_back_prompt_baseline(
                        int((done_data or {}).get("first_prompt_tokens") or 0))
                elif ev.type in ("thinking", "tool_call", "tool_result", "error"):
                    # ★ O-12 档 2 · P0 探针（2026-10-02）：**只在 doc-bound 且开关开启时**记录
                    # 「模型的首次 retrieve」的 query 与它距请求开始的时刻。
                    # 为什么在这里测：这是**唯一**能在真实流量上零成本拿到命中率的点 ——
                    #   · 不真的发起预取检索 ⇒ **零 GPU 成本**（P0 不白跑）；
                    #   · 只写一行日志 ⇒ **零行为变更**（不发事件、不落 ToolRun、不改任何口径）。
                    # 判据：`命中=是` 占比 ≥50% 才值得进入 P1（真做预取复用）。
                    if (
                        ev.type == "tool_call" and force_choice and prefetch_probe.probe_enabled()
                        and str((ev.data or {}).get("name") or "").strip().lower() == "retrieve"
                        and _probe_first_retrieve_ms is None
                    ):
                        _probe_first_retrieve_ms = (time.perf_counter() - _t_request) * 1000
                        _probe_model_query = str(((ev.data or {}).get("arguments") or {}).get("query") or "")
                    yield sse_frame(ev)
                    if ev.type == "error":
                        stream_error_reason = str((ev.data or {}).get("reason") or "")
                        # ④：结构化归因（枚举，可为空 = 老轨迹/未标注）；文案按它分流
                        stream_error_kind = str((ev.data or {}).get("failure_kind") or "")
                    if ev.type == "tool_result":
                        out = ev.data.get("output")
                        # ---- C3（2026-09-28）：检索空命中打点 ----
                        # 标记写在**事件 data 这一层**，而不是塞进 `output` 里：
                        # `output` 会被序列化进下一轮 prompt（`loop._format_tool_result`），
                        # 而事件 data 只进 trace/SSE ⇒ 加在这里**不改变任何提示词内容**。
                        # ⚠️ 也不能发 `error` 事件：那不是错误（本轮照常继续），
                        # 且会污染 `stream_error_reason` → 让中止轮的用量/notice 口径失真。
                        # 它落在 trace 里，由 `/metrics` 按轨迹计数（见 routes_metrics）。
                        if is_empty_retrieval(out):
                            ev.data["retrieval_empty"] = True
                        if isinstance(out, dict) and out.get("source") == "rag":
                            # F7：检索故障标记（区别于「已检索但无命中」的空结果）
                            # ⚠️ **本探测依赖一条契约**：失败工具的 `tool_result` 事件
                            # **必须带 `output`**（`loop._run_one`，2026-09-28 F7 修复）。
                            # 它此前只带 `{call_id, ok, error}` ⇒ 这里恒拿不到 `output`
                            # ⇒ `_retrieval_error` 恒 False ⇒ 索引损坏时用户看到的是通用兜底
                            # 「可能是模型调用失败…」，**指向错误的自救方向**。
                            # 再改 `_run_one` 的事件形状请同步本处，并跑
                            # `tests/test_retrieval_fault_surfaced.py`。
                            if out.get("error") == "retrieval_unavailable":
                                _retrieval_error = True
                            # N1：跨调用去重 + 总数上限（口径见 services/citation_clean.dedup_citations，可单测）
                            # B5：把"是否因达上限被截断"记下来（观测，不改行为）
                            _added = dedup_citations(
                                out.get("results", []), citation_keys, _MAX_CITATIONS,
                                capped_out=_citations_capped,
                            )
                            # ---- B2（2026-09-29）：标注「模型所见是否已截断」----
                            # **问题**：引用文本取自工具的**完整**输出，而注入模型的是
                            # `retrieve` 的 **200 字/段**压缩稿（`loop._prepare_tool_output`）
                            # ⇒ 两者**不同源**：模型可能"引用"它**并未看全**的段落，
                            # 而用户看到引用卡片会以为模型是基于全文作答的。
                            # **修**：如实标注，让卡片能提示"模型仅见片段开头"。
                            # ★ 判据复用 `model_injection_truncates`（与注入逻辑**同一来源**），
                            # 不在这里另写长度判断 —— 否则又是一把尺子。
                            _tool_name = str(ev.data.get("name") or "")
                            for _c in _added:
                                _c["model_truncated"] = model_injection_truncates(
                                    _tool_name, _c.get("text"))
                            citations.extend(_added)
        finally:
            # 消费端无论正常结束/中断/被外层取消，都要收掉两个后台任务；
            # pump.cancel() 会把 CancelledError 传入 run_stream 的 LLM await，立即止损。
            watcher.cancel()
            if not pump.done():
                pump.cancel()
        # O-12 档 2 · P0 探针收尾（零行为变更）：把「首次 retrieve 的 query vs 用户原 query」
        # 以及它距请求开始的时刻写成一行日志 —— 命中率是档 2 唯一的不确定量。
        if force_choice and prefetch_probe.probe_enabled():
            prefetch_probe.log_probe(
                user_query=body.content,
                model_query=_probe_model_query,
                t_first_retrieve_ms=_probe_first_retrieve_ms,
            )

        if loop_failed is not None:
            # P1-11（2026-09-15）：见 `_pump_direct` 同款说明。Agent 路径多两层考量：
            #   ① 失败前已产出的 tool_call/tool_result 事件**仍随 events 落库**——保留执行留痕，
            #      便于事后定位"失败发生在哪一步"；
            #   ② reflect / 自动标题 / 逐条核验全部跳过——失败轮次不再发起任何额外 LLM 调用。
            generated = (final_text or partial_text or "").strip()
            text_to_persist = generated or (
                _RETRIEVAL_ERROR_REPLY if _retrieval_error else _FALLBACK_REPLY
            )
            # ★ T-09+：把"生成失败"**写进 trace**（此前只发 SSE `error`、不落 events）
            # ⇒ 落库后 `/metrics` 才能按 `failure_kind` 分桶，而不是全落 `unknown`
            #   （真实库实测：51 条有 error 的轨迹里 45 条 kind 缺失）。
            events.append(LoopEvent(
                type="error", state=LoopState.ERROR_HANDLE,
                data={"reason": stream_error_reason, "failure_kind": m_failure.CRASH},
            ))
            chat_svc.persist_turn(
                session_id, user_content, history, events, text_to_persist, model,
                cost_override=(done_data or {}).get("cost"),
                prompt_tokens_override=(done_data or {}).get("prompt_tokens"),
                completion_tokens_override=(done_data or {}).get("completion_tokens"),
                total_tokens_override=(done_data or {}).get("total_tokens"),
                citations=citations,  # 引用落库（问题 3 修复）
                steps=(done_data or {}).get("steps"),
                latency_ms=(done_data or {}).get("latency_ms"),
                regenerate=body.regenerate,
                # `invalid` 仅在完全无输出时置位（口径同 `_pump_direct`）
                meta={"invalid": not generated, "notice": _stream_error_notice(stream_error_reason, stream_error_kind or m_failure.CRASH)},
            )
            yield sse_event("error", _stream_error_payload(stream_error_reason, m_failure.CRASH))
            return
        if loop_cancelled:
            # 客户端已断开：不再产生任何额外 LLM 调用（reflect/自动标题/核验全部跳过），
            # 仅持久化已生成的部分内容（meta.interrupted 标记）。
            # N13 修复（2026-09-12）：**无部分内容时也必须落用户提问**，否则整轮（含提问）
            # 消失、用户被迫重复输入（同 `_pump_direct` 路径的修复，两处口径一致）。
            partial = partial_text.strip()
            # ★ T-09+：补「用户主动停止」的**独立事件**（此前 events 里没有任何标记，
            #   于是"我主动停的"与"系统崩了"在归因上完全混在一起）。
            #   为什么不是 `error`：前端 reducer 的 `case "error"` 会把该消息染成**错误态**，
            #   而停止是正常行为；新类型由前端 `default: return items` 安全忽略。
            events.append(LoopEvent(
                type="cancelled", state=LoopState.OBSERVE,
                data={"reason": "user_stop"},
            ))
            if partial:
                chat_svc.persist_turn(
                    session_id, user_content, history, events, partial, model,
                    cost_override=(done_data or {}).get("cost"),
                    prompt_tokens_override=(done_data or {}).get("prompt_tokens"),
                    completion_tokens_override=(done_data or {}).get("completion_tokens"),
                    total_tokens_override=(done_data or {}).get("total_tokens"),
                    citations=citations,  # 引用落库（问题 3 修复）
                    steps=(done_data or {}).get("steps"),
                    latency_ms=(done_data or {}).get("latency_ms"),
                    regenerate=body.regenerate,
                    # ★ T-09+：停止也给**中性回执**（不是告警），刷新后仍能看到"这是我停的"
                    meta={"interrupted": True, "notice": _user_stop_notice()},
                )
            elif not body.regenerate:
                # 「重新生成」场景该 user 消息已存在，跳过以免重复落库
                chat_svc.persist_user_message(session_id, user_content)
            return

        # 空回复兜底：模型失败/无实质输出时给默认文案（绝不空白回复）
        # 2026-09-07 修复：先记录 _was_empty 再赋兜底文案（否则判空恒 False、notice 恒 None）
        _was_empty = not (final_text or "").strip()
        if _was_empty:
            # ④（2026-10-02）：**先按结构化 kind 给正确归因**，再退到检索故障/通用兜底。
            # 为什么顺序如此：熔断中止时通用兜底会写「可能是模型调用失败或返回内容为空」，
            # 把用户引向"查余额/换模型"—— 而对"上下文超预算"这两件事毫无作用（误导）。
            # `_retrieval_error` 属于**更具体且已确认**的现场，仍优先于 kind 文案。
            _kind_text = failure_notice.fallback_text(stream_error_kind, stream_error_reason)
            final_text = (
                _RETRIEVAL_ERROR_REPLY if _retrieval_error
                else (_kind_text if stream_error_kind else _FALLBACK_REPLY)
            )
        # 提示优先级：检索故障（更具体、可操作）> **结构化中止归因** > 通用兜底 > 无。
        # 即便模型仍产出了部分内容，检索故障也必须显式告知，不能被静默吞掉。
        if _retrieval_error:
            notice = _retrieval_error_notice()
        elif _was_empty and stream_error_kind:
            # ④（2026-10-02）：**结构化归因优先于通用兜底**
            # 原先 `_was_empty`（空回复）无条件走 `_invalid_notice("fallback")` ⇒ 熔断被渲染成
            # 「未生成有效回答：本次模型调用失败或返回内容为空」+「查余额/换模型」——
            # 用户被引向**错误方向**（实测端到端复现：熔断轮 notice.kind == "fallback"）。
            # 现在：已标注 kind 的中止 ⇒ 用它自己的归因；但**保留"连续无效"的强打断升级**
            # （`loop_risk` 是行为层的招呼，不能被一个更具体的文案吃掉）。
            notice = (
                _invalid_notice("loop_risk", _trailing_invalid)
                if _trailing_invalid >= _LOOP_THRESHOLD
                else _stream_error_notice(stream_error_reason, stream_error_kind)
            )
        elif _was_empty:
            notice = _invalid_notice("fallback", _trailing_invalid)
        elif stream_error_reason:
            # C2（2026-09-28）：**中止轮**（迭代耗尽 / 预算熔断 / 工具连续失败 / LLM 出错）
            # 此前 `notice` 恒为 None —— 中止原因只存在于即时的 error 事件里，
            # **刷新页面后历史里查不到**（而 missing_doc / fallback / retrieval_error
            # 三条路径都写 notice，此处不一致）。而"这轮为什么失败"恰恰是用户最需要的。
            # 复用既有 `_stream_error_notice`：按原因给**可操作**建议
            # （余额不足 → 充值；熔断 → 调预算或减文档；迭代上限 → 拆小问题）。
            notice = _stream_error_notice(stream_error_reason, stream_error_kind)
        else:
            notice = None
        yield sse_event("message", {
            "text": final_text,
            "citations": [{"id": c.get("id"), "page": normalize_page(c.get("page")), "doc_id": c.get("doc_id"), "text": c.get("text"), "model_truncated": c.get("model_truncated", False)} for c in citations],
        })
        # 持久化：优先用 AgentLoop 回传的精确成本/令牌，否则回退 token 粗算（保证"用量页成本与实际一致"）。
        # 多模态：user_content 为内容块（text/image_url）时随块持久化，历史加载后图片上下文不丢失。
        chat_svc.persist_turn(
            session_id, user_content, history, events, final_text, model,
            cost_override=(done_data or {}).get("cost"),
            prompt_tokens_override=(done_data or {}).get("prompt_tokens"),
            completion_tokens_override=(done_data or {}).get("completion_tokens"),
            total_tokens_override=(done_data or {}).get("total_tokens"),
            citations=citations,  # 引用落库（问题 3 修复）
            # 透传 AgentLoop 真实迭代步数 / 累计延迟，使用量页 Trace 不再恒显示 0
            steps=(done_data or {}).get("steps"),
            latency_ms=(done_data or {}).get("latency_ms"),
            regenerate=body.regenerate,
            # ★ P1-a2：同上（本路径直接有 done_data，比 events 更直接）
            meta=_persist_meta(notice, model_risk=model_risk,
                               ctx=_ctx_usage_from_done(done_data),
                               # ★ P0-2：中性观测（不夹带 invalid），供刷新后仍能看到提示
                               extra={"unrecognized_command": unrecognized_cmd}
                               if unrecognized_cmd else None),
        )
        # 跨会话记忆：轮末反思写入（仅当有实质回答，且 memory 功能开启）
        if final_text and memory_enabled:
            await chat_svc.reflect(
                session_id, agent.llm,
                _reflect_messages(history, body.content, final_text),
            )
        # 会话自动标题：默认标题按本轮主题生成（LLM 生成 / fake 规则截断降级）
        await chat_svc.auto_title(session_id, agent.llm, body.content)
        # P1-2.1：回复逐条核验（verify_reply 开启时，message 之后、done 之前发出；只读不改写）
        v = await _run_verify(
            body.content,
            final_text,
            [{"id": c.get("id"), "page": normalize_page(c.get("page")), "doc_id": c.get("doc_id"), "text": c.get("text")} for c in citations],
            agent.llm,
        )
        if v:
            yield sse_event("verify", v)
        if done_data is not None:
            # 精确 done（真实成本/token）已捕获，补入学科反馈字段后转发
            yield sse_event("done", {**({"notice": notice} if notice else {}), **done_data, **_done_extra})
        else:
            # 客户端中途断开导致未收到 loop 的 done：补一个兜底 done（token 粗算）
            usage = chat_svc.count_usage(session_id)
            yield sse_event("done", {
                "steps": sum(1 for e in events if e.type == "thinking"),
                "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                "cost": usage["cost"],
                "latency_ms": 0,
                # O-12：即使走了兜底 done，也如实上报已采到的首字延迟（未采到则 None）
                "ttft_ms": _ttft_ms(),
                **({"notice": notice} if notice else {}),
                **_done_extra,
            })

    return StreamingResponse(_with_stream_cleanup(gen()), media_type="text/event-stream")


@router.post("/search", response_model=list[RetrievedChunkOut])
async def search(body: SearchRequest, request: Request) -> list[RetrievedChunkOut]:
    doc_svc = request.app.state.doc_svc
    pipelines = []
    for d in (body.doc_ids or []):
        p = await doc_svc.get_pipeline(d)
        if p is not None:
            pipelines.append(p)
    if not pipelines:
        raise HTTPException(status_code=400, detail="未指定有效文档或未上传文档")
    from ..services.chat_service import CompositeRAGPipeline

    # 形态适配（同步/异步）+ **同步 GPU 管线下放线程池**：统一走 `call_query`
    # （N4/N9 唯一实现）。本路由是 `async def`，直接调用同步管线（BGE-M3 编码 +
    # 混合检索 + CrossEncoder 重排）会阻塞整个事件循环。
    results = await call_query(CompositeRAGPipeline(pipelines), body.query, top_k=body.top_k)
    return [RetrievedChunkOut(id=r.id, page=normalize_page(r.page), text=r.text, score=round(r.score, 4)) for r in results]
