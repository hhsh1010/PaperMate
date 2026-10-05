"""T-05+（2026-09-26）耗时观测的三条日志守护测试。

## 为什么需要这些测试

用户报「一次提问约 20 秒，只检索了两篇文档，为什么这么慢」。当时 `loop.py`
**完全没有耗时打点** —— `[上下文构成]` 只说 token 规模，`_acc_latency` 只有个数
且不打日志。于是"慢在哪"只能靠 app.log 里两个时间戳之间的**空档反推**，
得出的是**推算值而非实测值**（本地 1.3s / LLM 17.3s 就是推出来的）。

这三条日志把"慢在哪"变成可 grep 的事实：
- `[LLM]`   —— 每次调用的**墙钟 / HTTP实测 / 外层开销**，流式另有 **ttft 与生成段**
- `[工具]`  —— 每次工具调用的**原始字符数 → 实际注入字符数**（哪个工具灌爆上下文）
- `[本轮]`  —— 整轮分解 `墙钟 = LLM + 工具 + 其它`（可当场验算）

## 这些测试在守什么（易复发点）

1. **日志不能反过来弄坏被观测对象**：实测踩到 —— 直接用 `self.llm.model_name` 让
   `test_lightcore.py` 里没有该属性的 `_RecordingLLM` 替身抛 AttributeError，
   把**本来通过的用例**弄红。观测代码必须用 `getattr` 兜底。
2. **汇总必须在 `yield error` 之前打**：`_abort_with_fallback` 是 async generator，
   消费方收到 error 事件后可能立即 break → yield 之后的代码**不执行** → 汇总静默丢失。
3. **口径不能编错**：`LLMResponse.latency_ms` 是 `router.py` **本地实测的 HTTP 耗时**，
   不是厂商回传值 ⇒ 不能写成"厂商延迟"，也不能把 `墙钟−HTTP` 说成"网络+排队"。
4. **失败路径也要有耗时**：工具失败/超时时，"等了多久才失败"正是定位关键。
5. **分解等式可验算**：`墙钟 = LLM + 工具 + 其它` 的差必须落在舍入噪声内，
   否则日志是骗人的（三种口径混用）。
"""

from __future__ import annotations

import asyncio
import logging
import re

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, LLMResponse, LLMStreamChunk, Role, ToolCall
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec

LOGGER_NAME = "lightcore.core.loop"
_LONG_TEXT = "检索段落正文" * 100  # 600 字，远超 retrieve 的 200 字/段压缩上限


def _registry(handler, name: str = "retrieve") -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name=name, description="测试用工具",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission=Permission.SAFE, handler=handler,
    ))
    return reg


def _retrieve_ok(**_kw):
    return ToolResult(ok=True, output={
        "results": [{"id": f"doc-1-c{i}", "page": i + 1, "score": 0.9, "text": _LONG_TEXT}
                    for i in range(6)],
    })


def _retrieve_boom(**_kw):
    raise RuntimeError("模拟检索崩溃")


class _TwoStepLLM:
    """首轮调 retrieve，次轮给最终答案（非流式）。带完整响应字段（模拟真实厂商）。"""

    def __init__(self, *, model_name: str = "test-model", latency_ms: int = 11) -> None:
        self.model_name = model_name
        self._lat = latency_ms
        self.n = 0

    async def chat(self, messages, *args, **kw):  # noqa: ANN001, ANN002
        self.n += 1
        if self.n == 1:
            return LLMResponse(
                message=ChatMessage(role=Role.ASSISTANT, content="",
                                    tool_calls=[ToolCall(id="c1", name="retrieve",
                                                         arguments={"query": "q"})]),
                prompt_tokens=100, completion_tokens=10,
                latency_ms=self._lat, model=self.model_name,
            )
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="最终答案"),
            prompt_tokens=500, completion_tokens=20,
            latency_ms=self._lat, model=self.model_name,
        )


class _TwoStepStreamLLM:
    """流式替身：带 `ttft_ms`，且声称**真流式**（`STREAM_BUFFERED=False`），
    用于验证「生成段 = HTTP实测 − ttft」可算出来。

    ⚠️ 只有把 `STREAM_BUFFERED` 置 False 时才该出现"生成段" —— 那是"真流式"的前提。
    """

    STREAM_BUFFERED = False

    def __init__(self) -> None:
        self.model_name = "test-stream-model"
        self.n = 0

    async def stream(self, messages, *args, **kw):  # noqa: ANN001, ANN002
        self.n += 1
        if self.n == 1:
            yield LLMStreamChunk(tool_calls=[ToolCall(id="c1", name="retrieve",
                                                      arguments={"query": "q"})])
            yield LLMStreamChunk(finished=True, prompt_tokens=100, completion_tokens=5,
                                 latency_ms=800, ttft_ms=300)
        else:
            yield LLMStreamChunk(delta="最终答案")
            yield LLMStreamChunk(finished=True, prompt_tokens=500, completion_tokens=20,
                                 latency_ms=900, ttft_ms=400)


class _BufferedStreamLLM(_TwoStepStreamLLM):
    """**整段缓冲**的流式替身（= 当前 `OAICompatibleLLM` 的真实形态）。

    带 `STREAM_BUFFERED=True` 时，`ttft_ms ≈ latency_ms` 是**必然**（整段 body 先到齐），
    日志**不得**把它说成首 token 延迟、也不得算出"生成段"（那只是解析耗时）。
    实测：1742/1743、1014/1015、8919/8940（ttft/latency，ms）。
    """

    STREAM_BUFFERED = True


async def _retrieve_slow(**_kw):
    """并行计时的探针：每次调用睡 60ms。"""
    await asyncio.sleep(0.06)
    return ToolResult(ok=True, output={
        "results": [{"id": "doc-1-c0", "page": 1, "score": 0.9, "text": "段落正文"}],
    })


def _registry_many(handler, *names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for n in names:
        reg.register(ToolSpec(
            name=n, description="测试用工具",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            permission=Permission.SAFE, handler=handler,
        ))
    return reg


class _TwoToolLLM:
    """首轮一次发起 **2 个并行** 工具调用，次轮给答案。用于验证 ACT 墙钟不重复计。"""

    def __init__(self) -> None:
        self.model_name = "test-model"
        self.n = 0

    async def chat(self, messages, *args, **kw):  # noqa: ANN001, ANN002
        self.n += 1
        if self.n == 1:
            return LLMResponse(
                message=ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                    ToolCall(id="c1", name="retrieve", arguments={"query": "a"}),
                    ToolCall(id="c2", name="retrieve", arguments={"query": "b"}),
                ]),
                prompt_tokens=100, completion_tokens=10, latency_ms=5, model=self.model_name,
            )
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="最终答案"),
            prompt_tokens=500, completion_tokens=20, latency_ms=5, model=self.model_name,
        )


class _AlwaysToolLLM:
    """永远返回工具调用 → 迭代耗尽，走 `_abort_with_fallback`。"""

    def __init__(self) -> None:
        self.n = 0

    async def chat(self, messages, *args, **kw):  # noqa: ANN001, ANN002
        self.n += 1
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="",
                                tool_calls=[ToolCall(id=f"c{self.n}", name="retrieve",
                                                     arguments={"query": "q"})]),
            prompt_tokens=50, completion_tokens=5, latency_ms=3, model="test-model",
        )


class _BareLLM:
    """**最小替身**：只有 `chat`，没有 `model_name`、响应里也没有 latency_ms。

    这就是实测把 `test_lightcore.py` 弄红的那种替身。观测代码必须容得下它。
    """

    def __init__(self) -> None:
        self.n = 0

    async def chat(self, messages, *args, **kw):  # noqa: ANN001, ANN002
        self.n += 1
        if self.n == 1:
            return LLMResponse(message=ChatMessage(
                role=Role.ASSISTANT, content="",
                tool_calls=[ToolCall(id="c1", name="retrieve", arguments={"query": "q"})]))
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="最终答案"))


def _run(llm, handler, *, stream_tokens: bool = False, **cfg):  # noqa: ANN001
    async def _go():
        loop = AgentLoop(llm, ToolExecutor(_registry(handler)), LoopConfig(**cfg))
        return [ev async for ev in loop.run_stream("测试问题", stream_tokens=stream_tokens)]

    return asyncio.run(_go())


def _run_many(llm, handler, names: tuple[str, ...], **cfg):  # noqa: ANN001
    async def _go():
        loop = AgentLoop(llm, ToolExecutor(_registry_many(handler, *names)), LoopConfig(**cfg))
        return [ev async for ev in loop.run_stream("测试问题", stream_tokens=False)]

    return asyncio.run(_go())


# ============ ① `[LLM]` 每次调用打点 ============

class TestLlmTimingLog:
    def test_each_call_logged_with_wall_and_http(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        _run(_TwoStepLLM(latency_ms=11), _retrieve_ok)
        lines = [r.getMessage() for r in caplog.records if "[LLM]" in r.getMessage()]
        assert len(lines) == 2, f"两次 LLM 调用应各打一条，实际 {len(lines)}：{lines}"

        assert "mode=chat step=0" in lines[0]
        assert "mode=chat step=1" in lines[1]
        # ★ 关键：单次 prompt 必须分别可见（落库的 prompt_tokens 是整轮累计，看不了单次）
        assert "prompt=100" in lines[0], lines[0]
        assert "prompt=500" in lines[1], lines[1]
        # ★ 口径：必须是 HTTP实测（本地测的），不得写成"厂商延迟"
        assert "HTTP实测=11ms" in lines[0], lines[0]
        assert "厂商" not in lines[0], f"口径写错：latency_ms 不是厂商回传值 | {lines[0]}"
        assert "网络+排队" not in lines[0], f"口径写错：wall−HTTP 是外层开销 | {lines[0]}"
        assert "外层开销=" in lines[0], lines[0]
        assert "model=test-model" in lines[0], lines[0]
        # 非流式无法观测首 token → 必须明确写 n/a，而不是留 0 让人误读
        assert "ttft=n/a（非流式无法观测首 token）" in lines[0], lines[0]

    def test_stream_logs_ttft_and_generation_segment(self, caplog):
        """流式：必须能算出「等首字」与「生成」两段 —— 这是判"排队"还是"生成慢"的唯一依据。"""
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        _run(_TwoStepStreamLLM(), _retrieve_ok, stream_tokens=True)
        lines = [r.getMessage() for r in caplog.records if "[LLM]" in r.getMessage()]
        assert len(lines) == 2, lines
        assert "mode=stream step=0" in lines[0], lines[0]
        # 800 - 300 = 500ms 生成段（HTTP实测 − ttft）
        assert "ttft=300ms（生成段 500ms）" in lines[0], lines[0]
        assert "HTTP实测=800ms" in lines[0], lines[0]

    def test_buffered_stream_must_not_claim_ttft(self, caplog):
        """★ 回归守卫（2026-09-27 实测修正）：整段缓冲的传输**不得**报 ttft / 生成段。

        真实后端 `stream()` 用 `client.post()`，httpx 会把整个 body 缓冲完再返回 ⇒
        `ttft_ms ≈ latency_ms`（实测 12/12 样本，差 1~21ms），"生成段"只是**解析耗时**。
        若日志照报，会让人误以为"模型生成只花 1ms、时间全在等服务端开口"——
        **恰好把结论指反**。判定用 LLM 后端**声明**的 `STREAM_BUFFERED`，不靠数值反推。
        """
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        _run(_BufferedStreamLLM(), _retrieve_ok, stream_tokens=True)
        lines = [r.getMessage() for r in caplog.records if "[LLM]" in r.getMessage()]
        assert lines, "缓冲流式仍必须有日志"
        line = lines[0]
        assert "整段缓冲" in line, line
        assert "生成段" not in line, f"缓冲传输下不得报生成段：{line}"
        # 真实等待仍在 HTTP实测 里，必须保留（那才是用户等的时间）
        assert "HTTP实测=800ms" in line, line

    def test_logged_even_when_double_lacks_model_name_and_latency(self, caplog):
        """回归守卫：观测代码**不得**因替身字段缺失而抛异常。

        实测踩到：直接取 `self.llm.model_name` 让 `test_lightcore.py` 的
        `_RecordingLLM` 抛 AttributeError，把本来通过的用例弄红。
        """
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        _run(_BareLLM(), _retrieve_ok)
        lines = [r.getMessage() for r in caplog.records if "[LLM]" in r.getMessage()]
        assert lines, "替身缺字段时仍必须打出日志"
        assert "_BareLLM" in lines[0], f"应回退到类名：{lines[0]}"
        assert "n/a（未实测）" in lines[0], f"缺 latency 时不得编造差值：{lines[0]}"

    def test_failure_path_records_wall_time(self, caplog):
        """LLM 调用失败时也要有墙钟 —— 区分「超时/限流（等很久才报错）」与「立即报错」。"""
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)

        class _BoomLLM:
            async def chat(self, *a, **kw):  # noqa: ANN002, ANN003
                raise ValueError("模拟 401")

        loop = AgentLoop(_BoomLLM(), ToolExecutor(_registry(_retrieve_ok)), LoopConfig())

        async def _go():
            return [ev async for ev in loop.run_stream("q", stream_tokens=False)]

        events = asyncio.run(_go())
        assert any(ev.type == "error" for ev in events)
        warn = [r.getMessage() for r in caplog.records if "[LLM]" in r.getMessage()]
        assert warn and "调用失败" in warn[0], warn
        assert "墙钟=" in warn[0] and "ValueError" in warn[0], warn[0]


# ============ ② `[工具]` 原始 → 注入 ============

class TestToolLog:
    def test_logs_raw_and_injected_size(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        _run(_TwoStepLLM(), _retrieve_ok)
        lines = [r.getMessage() for r in caplog.records if "[工具]" in r.getMessage()]
        assert len(lines) == 1, lines
        line = lines[0]
        assert "retrieve ok" in line, line
        assert "耗时=" in line, line

        m = re.search(r"原始=(\d+)字 → 注入=(\d+)字（≈(\d+) tokens）", line)
        assert m, f"缺原始/注入字符数：{line}"
        raw, injected, toks = (int(x) for x in m.groups())
        # retrieve 走 200 字/段压缩 → 注入必须显著小于原始
        assert raw > injected * 2, f"注入未压缩？raw={raw} injected={injected}"
        assert toks > 0, line

    def test_logs_failed_tool(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        _run(_TwoStepLLM(), _retrieve_boom)
        lines = [r.getMessage() for r in caplog.records if "[工具]" in r.getMessage()]
        assert lines, "失败的调用同样必须有一行日志"
        assert "FAILED" in lines[0], lines[0]
        assert "耗时=" in lines[0], lines[0]


# ============ ③ `[本轮]` 汇总分解 ============

_SUMMARY_RE = re.compile(
    r"墙钟=(\d+)ms = LLM (\d+)ms \+ 工具 (\d+)ms \+ 其它 (-?\d+)ms"
    r".*LLM HTTP实测合计=(\d+)ms.*峰值prompt=(\d+)"
)


class TestTurnSummary:
    def test_summary_adds_up(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        _run(_TwoStepLLM(), _retrieve_ok)
        lines = [r.getMessage() for r in caplog.records if "[本轮]" in r.getMessage()]
        assert len(lines) == 1, lines
        m = _SUMMARY_RE.search(lines[0])
        assert m, f"汇总格式不符合可验算契约：{lines[0]}"
        wall, llm_ms, tool_ms, other_ms, http_sum, peak = (int(x) for x in m.groups())
        # ★ 分解等式必须闭合（容差 2ms 覆盖 %.0f 的三处独立舍入）
        assert abs(wall - (llm_ms + tool_ms + other_ms)) <= 2, (
            f"分解等式不闭合：{wall} != {llm_ms}+{tool_ms}+{other_ms} | {lines[0]}"
        )
        assert peak >= 500, f"峰值 prompt 应取本轮最大值 500，实际 {peak}"
        # HTTP实测合计 = 2 次 × 11ms（同源同口径：都是 router.py 本地实测）
        assert http_sum == 22, f"HTTP实测合计应为 22，实际 {http_sum}"
        # ⚠️ 替身瞬时返回 → 墙钟/LLM/工具可能为 0ms；不该断言 >0（那是环境噪声，不是契约）
        assert other_ms >= -2, lines[0]

    def test_summary_on_abort_path(self, caplog):
        """中止路径也必须打汇总 —— 否则「失败轮次有多慢」不可观测。

        ⚠️ 这里同时守一个**极易复发**的点：汇总必须在 `yield error` **之前**执行。
        `_abort_with_fallback` 是 async generator，若消费方收到 error 后立即 break，
        yield 之后的语句永不执行 → 汇总静默丢失。故本测试**故意不消费完事件流**。
        """
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        loop = AgentLoop(_AlwaysToolLLM(), ToolExecutor(_registry(_retrieve_ok)),
                         LoopConfig(max_iterations=1, max_steps_no_progress=99))

        async def consume_until_error():
            async for ev in loop.run_stream("测试问题", stream_tokens=False):
                if ev.type == "error":
                    break  # ★ 故意提前退出：模拟消费方收到终止事件就走

        asyncio.run(consume_until_error())
        lines = [r.getMessage() for r in caplog.records if "[本轮]" in r.getMessage()]
        assert lines, "中止路径未打汇总（很可能被 yield 之后的代码位置吞掉了）"
        assert _SUMMARY_RE.search(lines[0]), lines[0]

    def test_no_summary_without_run_stream(self, caplog):
        """未经 `run_stream` 的直调不应打无意义汇总（`_turn_t0` 未设置 → 直接返回）。"""
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        loop = AgentLoop(_TwoStepLLM(), ToolExecutor(_registry(_retrieve_ok)), LoopConfig())
        loop._log_turn_summary()  # noqa: SLF001  直调私有方法，验证守卫
        assert not [r for r in caplog.records if "[本轮]" in r.getMessage()]

    def test_summary_on_client_interrupt(self, caplog):
        """★ 客户端中断（用户等不及就换问题）也必须留下汇总。

        实测踩到：一次 **9 步 / 23 秒**的请求被中断，全程**没有一行 `[本轮]`** ——
        因为原实现只在「正常收敛」与「`_abort_with_fallback`」两个出口打，
        而中断会直接把生成器 `aclose()`，**两个出口都到不了**。
        修复方式：`run_stream` 变薄包装 + `finally` 兜底。本测试钉死它。
        """
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        loop = AgentLoop(_TwoStepLLM(), ToolExecutor(_registry(_retrieve_ok)), LoopConfig())

        async def _interrupt_midway():
            agen = loop.run_stream("测试问题", stream_tokens=False)
            async for ev in agen:
                if ev.type == "tool_call":   # ★ 工具刚发起就中断（模拟用户关掉/发新消息）
                    break
            await agen.aclose()

        asyncio.run(_interrupt_midway())
        lines = [r.getMessage() for r in caplog.records if "[本轮]" in r.getMessage()]
        assert lines, "客户端中断后未打汇总（`finally` 兜底失效）"
        assert _SUMMARY_RE.search(lines[0]), lines[0]


# ============ ④ 并行工具：ACT 墙钟不得重复计 ============

class TestParallelToolAccounting:
    def test_parallel_tools_counted_once(self, caplog):
        """★ 回归守卫（2026-09-27 实测修正）：并行分组下**同一段时间不得被计 N 次**。

        实测踩到：3 个 retrieve 并行，各自把墙钟累加 ⇒ 汇总出现
        `墙钟=12714ms = LLM 9537 + 工具 5250 + 其它 **-2073**`（负数，物理上不可能）。
        修复：改由调用方测**整个 ACT 阶段**的墙钟。本用例让 2 个工具各睡 60ms：
        并行 ⇒ ≈60ms；若退化为逐个累加 ⇒ ≈120ms。阈值取 100ms 区分两者。
        """
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        _run_many(_TwoToolLLM(), _retrieve_slow, ("retrieve",))
        lines = [r.getMessage() for r in caplog.records if "[本轮]" in r.getMessage()]
        assert len(lines) == 1, lines
        m = _SUMMARY_RE.search(lines[0])
        assert m, lines[0]
        wall, llm_ms, tool_ms, other_ms, _http, _peak = (int(x) for x in m.groups())

        assert 40 <= tool_ms <= 100, (
            f"两个并行工具各 60ms：并行应 ≈60ms，逐个累加才会 ≈120ms，实测 {tool_ms}ms | {lines[0]}"
        )
        assert abs(wall - (llm_ms + tool_ms + other_ms)) <= 2, lines[0]
        # 最关键的一条：分解不得出现负的「其它」
        assert other_ms >= 0, f"「其它」为负说明仍存在重复计：{lines[0]}"
        # 单次工具的耗时仍须各自可见（60ms 左右），用于区分"哪个工具慢"
        tool_lines = [r.getMessage() for r in caplog.records if "[工具]" in r.getMessage()]
        assert len(tool_lines) == 2, tool_lines
