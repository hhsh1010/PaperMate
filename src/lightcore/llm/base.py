"""LightCore · LLM 抽象层（docs/05 §LLM 抽象）

定义：
- Role: 消息角色
- ChatMessage: 单条对话消息（支持可选 tool_calls / tool_call_id）
- ToolCall: 模型决策的一次工具调用（name + arguments）
- ToolChoice / 工具描述: 喂给模型的 JSON Schema
- BaseLLM: 所有模型后端的统一抽象（chat / stream）
- ModelRouter: 按名路由到不同供应商（OpenAI 兼容端点，key 走环境变量，不入库）
- FakeLLM: 确定性本地替身，无网络/无 key 即可跑通 Agent 循环与测试（docs/08 FakeLLM 替身）
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """模型产出的一次工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


# 多模态：单张图片的本地 token 估算上限（与 DeepSeek-V4-Flash-Vision 官方计费口径一致：
# 任意尺寸图片折算 ≤384 tokens/张；供成本估算/熔断与上下文预算使用）
_IMAGE_TOKENS = 384

# 多模态 content 块类型（OpenAI 兼容）：{"type":"text","text":...} / {"type":"image_url","image_url":{"url":...}}
ContentBlock = dict[str, Any]
MessageContent = str | list[ContentBlock]


@dataclass
class ChatMessage:
    role: Role
    content: MessageContent = ""  # 文本字符串，或 OpenAI 兼容内容块列表（text/image_url，多模态）
    tool_calls: list[ToolCall] = field(default_factory=list)
    # 当 role=TOOL 时，回填被调用工具的 id，便于模型对齐
    tool_call_id: str | None = None
    # 思考模式（DeepSeek V4）：assistant 消息携带的链式思考文本。
    # to_openai 时映射为 reasoning_content 回传——V4 官方要求带工具调用的轮次必须回传，
    # 否则后续请求 400（docs/实施记录 2026-08-22）。
    reasoning: str = ""

    def to_openai(self) -> dict[str, Any]:
        # OpenAI 兼容标准：content 可为 null，但部分厂商（如 DashScope/Qwen）强制要求字符串
        # 出现 Assistant/TOOL 历史空白消息时为 ""，且不报错；user/system 必须有内容（上游已保证）
        # 多模态：content 为块列表时原样透传（text + image_url），供 vision 模型消费。
        if isinstance(self.content, list):
            content: Any = self.content
        else:
            content = self.content or ""
        msg: dict[str, Any] = {"role": self.role.value, "content": content}
        if self.reasoning:
            # DeepSeek V4 思考链：仅对 assistant 消息回传；其他模型/角色忽略（OpenAI 兼容字段）
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def text_only(self) -> str:
        """提取纯文本内容（多模态块消息取全部 text 块拼接），供意图识别/记忆检索等文本侧逻辑使用。"""
        if isinstance(self.content, str):
            return self.content
        parts = [b.get("text", "") for b in self.content if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(parts)


@dataclass
class ToolSpecForLLM:
    """喂给模型的函数声明（JSON Schema 子集）。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class LLMResponse:
    """一次 chat 的返回。"""

    message: ChatMessage
    # 观测/计费字段（NFR: 成本熔断、trace 可观测）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_ms: int = 0
    cost: float = 0.0  # 估算成本（¥），由 OAICompatibleLLM 按定价表计算；FakeLLM 恒为 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMStreamChunk:
    """流式增量。delta 为内容增量；finished 时为最终元数据。"""

    delta: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finished: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 思考模式：本轮累计的 reasoning_content 增量（DeepSeek V4，流式逐块追加）
    reasoning: str = ""
    #: 本次流式调用的实测耗时（ms），仅 `finished` 块上有意义（2026-09-16 流式用量采集）。
    #: 修复前流式路径完全不测延迟 → `Trace.total_latency_ms` 与循环层 `_acc_latency` 恒 0。
    latency_ms: int = 0
    #: 本次流式调用的估算成本（¥），仅 `finished` 块上有意义。
    #: 由厂商返回的 usage + 定价表算出；厂商未下发 usage 时按本地粗算兜底。
    cost: float = 0.0
    #: **首个增量块出现的耗时（ms）**，仅 `finished` 块上有意义（2026-09-26 T-05+ 新增）。
    #: 口径：从 `OAICompatibleLLM.stream` 的 `t0`（**建连之前**）到**第一个增量块**
    #: （reasoning / content / tool_call 任一）被解析出来。
    #:
    #: ⚠️ **它的含义取决于传输是否缓冲**（2026-09-27 实测澄清）：
    #: - **真流式（当前）**：`_stream_impl` 用 `async with client.stream("POST", ...)`
    #:   ⇒ 本值是**真 TTFT**（建连 + 服务端排队 + prefill），
    #:   `latency_ms − ttft_ms` = **生成/收流阶段**。
    #: - **若传输退回整段缓冲**（如 `client.post()`）：本值会恒 ≈ `latency_ms`
    #:   （实测 12/12 样本差 1~21ms），**不再是首 token 延迟**，差值只是解析耗时。
    #: 判定依据：`OAICompatibleLLM.STREAM_BUFFERED`（单一来源，**勿用"ttft≈latency"反推**）。
    #: 非流式路径（`chat`）无此概念，恒 0。
    ttft_ms: int = 0


class BaseLLM(ABC):
    """统一模型后端抽象。同步 chat 与异步流式 stream 二选一实现即可。"""

    model_name: str = "base"

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpecForLLM] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpecForLLM] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """默认实现：退化成一次性 chat 然后吐一个 finished chunk。"""
        resp = await self.chat(messages, tools, temperature=temperature, max_tokens=max_tokens, tool_choice=tool_choice)
        if resp.message.content:
            yield LLMStreamChunk(delta=resp.message.content)
        if resp.message.tool_calls:
            yield LLMStreamChunk(tool_calls=resp.message.tool_calls)
        yield LLMStreamChunk(
            finished=True,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            reasoning=resp.message.reasoning,  # 思考模式：退化路径一次性回传完整 reasoning
        )


#: 字符类 → **字符/token 密度**（2026-09-25 T-01 实测标定，真值 = 厂商返回的 `prompt_tokens`）。
#:
#: **为什么不是单一系数**：旧实现是 `len/2.5`。标定（21 条受控样本，模型 `qwen3.6-flash`，
#: 见 `scripts/probe_token_calibration.py` 与 `eval/t01_calibration.txt`）发现偏差**按内容类型分档、
#: 且方向相反** ——
#:
#:   | 内容 | 旧口径偏差 | 实测渐近密度 |
#:   |---|---|---|
#:   | 中文正文 | **低估 52%** | 1.64 字符/token |
#:   | 英文正文 | **高估 54%** | 5.4（含空格后按字符类拆解 → 字母 11.9 / 空白标点 1.36）|
#:   | 代码 | 高估 29% | 3.5 |
#:   | JSON | 偏低 12% | 2.2 |
#:
#: 于是**任何单一系数都补不平**（中英混合会话里误差方向相反）。改用"按字符类分别计"后：
#: 相对误差 **p50 34.2% → 5.4%**（留一法；`≤±10%` 的比例 2/21 → 16/21）。
#:
#: ⚠️ **标定边界（改这些常数前必读）**
#: 1. 密度是在 **Qwen `qwen3.6-flash`** 上量的。换厂商（DeepSeek 等）分词器不同，
#:    数值会有差异 —— 但"按字符类分档"这个**结构**是通用的，换厂商只需重跑探针重标系数。
#: 2. 只覆盖**文本**：`image_url` 块仍按官方单图上限 384 tokens 计（与旧实现一致）。
#: 3. **未建模固定开销**：实测每条请求还有约 **+14 tokens** 的 chat 模板开销
#:    （`_say` 系统标记等）。它是**请求级**而非消息级，故不放进本函数 —— 否则按消息累加会重复计。
#:    需要"整段 prompt 估算"时请自行加一次。
#: 4. **D′ 架构下本函数只做兜底**：整段 prompt 优先用厂商真实 usage（见 `AgentLoop._prepare_history`），
#:    估算只用于「首轮无真值」与「本步新增增量」。
_TOKEN_DENSITY_CJK = 1.64      # 中日韩文字 + 全角标点（token 最贵）
_TOKEN_DENSITY_ALNUM = 11.87   # 拉丁字母与数字（BPE 压缩率最高）
_TOKEN_DENSITY_OTHER = 1.36    # 空白、半角标点、代码符号（很贵：一次一个 token 很常见）

_CJK_RE = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def _rough_tokens(content: MessageContent) -> int:
    """本地 token 估算（**按字符类分档**），用于计费/熔断兜底与上下文构成日志。

    **2026-09-25 重写（T-01）**：旧实现 `len(content)/2.5` 是**单一系数**，实测相对误差
    p50 **34.2%**，且偏差**按内容类型分档、方向相反**（中文低估 52%、英文高估 54%）——
    详见 `_TOKEN_DENSITY_*` 常量的注释与 `eval/t01_calibration.txt`。
    改为按字符类（CJK / 字母数字 / 其它）分别除以实测密度后，留一法相对误差 **p50 5.4%**。

    影响面**不止于日志**：`AgentLoop._compact_history` 的压缩阈值与熔断的估算兜底用的都是这把尺子
    （旧口径下"该压缩时几乎不触发"）。**同时它也是"两把尺子"问题的来源之一** ——
    触发线用真实值、cap 用估算值，旧估算偏低 2 倍会让 cap 变成"等效 2 倍大"从而空转。

    多模态：文本块按文本计，image_url 块按官方单图上限 384 tokens 计（不变）。

    成本：旧实现是 O(1)，现在是**两次正则扫描**（O(n)）。调用点是"每条消息一次"，
    单次 ≤ 数十 KB 量级，实测开销可忽略；但**不要放进逐 token 的热循环**。
    """
    if isinstance(content, list):
        total = 0
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "image_url":
                total += _IMAGE_TOKENS
            else:
                total += _rough_tokens(str(b.get("text", "")))
        return total
    if not content:
        return 0
    text = str(content)
    cjk = len(_CJK_RE.findall(text))
    alnum = len(_ALNUM_RE.findall(text))
    other = len(text) - cjk - alnum
    est = cjk / _TOKEN_DENSITY_CJK + alnum / _TOKEN_DENSITY_ALNUM + other / _TOKEN_DENSITY_OTHER
    return max(1, round(est))
