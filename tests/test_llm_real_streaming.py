"""真流式改造（2026-09-27）的守护测试。

## 背景：为什么必须改，以及改完要守住什么

原实现 `_stream_impl` 用 `resp = await client.post(...)`。httpx 的 `post()` 走 `stream=False`
⇒ **把整个 body 缓冲完才返回** ⇒ 后面的 `aiter_lines()` 遍历的是**内存里的完整缓冲**
⇒ 所有 delta 在模型**生成完之后一次性**下发。

实测取证（12/12 次真实调用一致）：`ttft_ms ≈ latency_ms`（1742/1743、1014/1015、8919/8940…），
"生成段"只有 1~21ms —— 那是**解析耗时**，不是生成耗时。
用户侧体感：**等满全程 → 整段突然出现**（总时长不变，感知延迟=全程）。

改成 `async with client.stream("POST", ...)` 后 `aiter_lines()` **边收边解析**。

## 这些测试在守什么（易复发点）

1. **必须真的是增量下发** —— 不是"改了写法但没改行为"。用**慢速 mock 流**度量：
   首块到达时间必须显著小于总时长；一旦有人改回 `post()`，`ttft` 会立刻塌回 ≈ `latency`。
2. **`STREAM_BUFFERED` 必须与实现一致** —— 它是耗时日志的**单一来源**，
   实现变了而常量没变 ⇒ **日志会撒谎**。
3. ★ **已产出增量后不得重试** —— 真流式的新约束：已下发的 delta **无法回滚**，
   重试会让用户看到**重复内容**。这条是改造引入的，必须钉死。
4. **未产出增量时仍要重试** —— 不能因为上一条就把重试能力弄丢（429/5xx/连接失败照旧）。
5. `finished` 块仍要带 usage / `latency_ms` / `ttft_ms` / cost（用量与成本熔断依赖它）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time

import httpx

from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.router import ModelEndpoint, OAICompatibleLLM

_ENDPOINT = ModelEndpoint("mock", "https://example.com/v1", "MOCK_KEY", "mock-model")


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


def _delta(content: str = "", *, reasoning: str = "", tool_calls: list | None = None) -> bytes:
    d: dict = {}
    if content:
        d["content"] = content
    if reasoning:
        d["reasoning_content"] = reasoning
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    return _sse({"choices": [{"delta": d}]})


def _usage(pt: int, ct: int) -> bytes:
    return _sse({"choices": [{"delta": {}}], "usage": {"prompt_tokens": pt, "completion_tokens": ct}})


def _make(handler, *, max_retries: int = 3) -> OAICompatibleLLM:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OAICompatibleLLM(_ENDPOINT, client=client, api_key="k", max_retries=max_retries)


async def _drain(llm, **kw):
    """消费 `stream()`，同时记录每个 delta 的到达时刻（用于验证"边收边下发"）。"""
    t0 = time.perf_counter()
    arrivals: list[tuple[str, float]] = []
    finished = None
    async for chunk in llm.stream([ChatMessage(role=Role.USER, content="hi")], **kw):
        if chunk.delta or chunk.reasoning:
            arrivals.append((chunk.delta or f"[r]{chunk.reasoning}", (time.perf_counter() - t0) * 1000))
        if chunk.finished:
            finished = chunk
    return arrivals, finished, (time.perf_counter() - t0) * 1000


# ============ ① 真的是增量下发（核心回归） ============

class TestRealStreaming:
    def test_deltas_arrive_incrementally_not_all_at_once(self):
        """★ 核心守卫：首块必须**远早于**末块到达。

        慢速 mock：每块之间睡 60ms，共 4 块 + usage。
        - 真流式：首块 ≈60ms 到，总时长 ≈300ms ⇒ 首块只在全程的 ~20%
        - 退回 `post()`（整段缓冲）：首块 ≈300ms（全部收完才解析）⇒ 首块 ≈ 全程
        ⇒ 用"首块 / 全程"这个比值即可分辨两者，且不依赖绝对耗时。
        """
        async def body():
            for i in range(4):
                await asyncio.sleep(0.06)
                yield _delta(f"c{i}")
            await asyncio.sleep(0.06)
            yield _usage(11, 22)
            yield b"data: [DONE]\n\n"

        llm = _make(lambda req: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()))

        arrivals, finished, total_ms = asyncio.run(_drain(llm))

        assert [t for t, _ in arrivals] == ["c0", "c1", "c2", "c3"], arrivals
        first_ms = arrivals[0][1]
        assert first_ms < total_ms * 0.5, (
            f"首块到达 {first_ms:.0f}ms / 全程 {total_ms:.0f}ms ⇒ 疑似仍为整段缓冲（delta 一次性下发）"
        )
        assert finished is not None and finished.prompt_tokens == 11 and finished.completion_tokens == 22

    def test_ttft_much_smaller_than_latency(self):
        """`ttft_ms`（首块）必须显著小于 `latency_ms`（收完流）—— 这两个数的差才是"生成段"。"""
        async def body():
            yield _delta("a")
            await asyncio.sleep(0.20)   # 纯"生成"时间
            yield _delta("b")
            yield _usage(1, 2)
            yield b"data: [DONE]\n\n"

        llm = _make(lambda req: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()))
        _arrivals, finished, _total = asyncio.run(_drain(llm))

        assert finished is not None
        assert finished.ttft_ms > 0, "真流式下 ttft 必须被记录"
        assert finished.latency_ms > finished.ttft_ms + 150, (
            f"ttft={finished.ttft_ms}ms latency={finished.latency_ms}ms ⇒ "
            "两者几乎相等说明响应被整段缓冲（ttft 不是真 TTFT）"
        )

    def test_stream_buffered_flag_matches_implementation(self):
        """★ `STREAM_BUFFERED` 是日志的**单一来源**，必须与实现一致（否则日志撒谎）。"""
        assert OAICompatibleLLM.STREAM_BUFFERED is False, (
            "实现已改真流式，此常量若仍为 True 会让耗时日志谎称「整段缓冲」"
        )
        # 反向一致性：从**方法源码**直接看（比整文件匹配精确）。
        # ⚠️ 必须先剔除注释行 —— 注释里引用了旧写法（`resp = await client.post(...)`）做对比，
        # 不剔除会把"引用"误判成"实现"（本用例第一次就踩到自己身上）。
        src = inspect.getsource(OAICompatibleLLM._stream_impl)
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert "client.stream(" in code, f"真流式应使用 client.stream(...)：\n{code[:300]}"
        assert "await client.post(" not in code, (
            "`_stream_impl` 里不应再有整段缓冲的 client.post()（会让 delta 一次性下发）"
        )


# ============ ② 重试语义（真流式引入的新约束） ============

class TestRetrySemantics:
    def test_no_retry_after_delta_emitted(self):
        """★ 已产出增量后出错**不得重试** —— 重放会让用户看到重复内容。"""
        calls = {"n": 0}

        async def body():
            yield _delta("部分内容")
            raise httpx.ReadError("模拟中途断流")

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

        llm = _make(handler)
        seen: list[str] = []

        async def run():
            async for chunk in llm.stream([ChatMessage(role=Role.USER, content="hi")]):
                if chunk.delta:
                    seen.append(chunk.delta)

        try:
            asyncio.run(run())
        except httpx.ReadError:
            pass  # 预期：如实向上抛，不吞
        else:
            raise AssertionError("中途断流应向上抛出（不能被静默吞掉）")

        assert seen == ["部分内容"], f"内容被重复下发（疑似重试）：{seen}"
        assert calls["n"] == 1, f"已产出增量后不应重试，实际请求 {calls['n']} 次"

    def test_retry_before_any_delta(self):
        """未产出增量时必须**保留重试能力**（5xx/429 照旧退避重试）。"""
        calls = {"n": 0}

        async def good_body():
            yield _delta("ok")
            yield _usage(1, 1)
            yield b"data: [DONE]\n\n"

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500, text="server boom")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=good_body())

        llm = _make(handler)
        arrivals, finished, _ = asyncio.run(_drain(llm))
        assert calls["n"] == 2, f"应重试一次，实际 {calls['n']} 次"
        assert [c for c, _ in arrivals] == ["ok"], arrivals
        assert finished is not None

    def test_stream_options_rejection_falls_back(self):
        """端点 400 拒绝 `stream_options` 时去掉该字段重试（不能为观测把对话打挂）。"""
        calls = {"n": 0}
        payloads: list[dict] = []

        async def good_body():
            yield _delta("done")
            yield b"data: [DONE]\n\n"

        def handler(request):
            calls["n"] += 1
            body = json.loads(request.content.decode("utf-8"))
            payloads.append(body)
            if "stream_options" in body:
                return httpx.Response(400, text="unsupported field: stream_options")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=good_body())

        llm = _make(handler)
        arrivals, _finished, _ = asyncio.run(_drain(llm))
        assert calls["n"] == 2, f"应重试一次，实际 {calls['n']} 次"
        assert "stream_options" in payloads[0] and "stream_options" not in payloads[1]
        assert [c for c, _ in arrivals] == ["done"], arrivals
        assert llm._stream_usage_enabled is False  # noqa: SLF001  记住端点不支持，后续不再请求


# ============ ③ 解析细节不回归 ============

class TestParsingUnchanged:
    def test_reasoning_and_tool_calls_and_usage(self):
        """思考增量 / 工具调用分片拼接 / usage 三项解析不能因改造而丢。"""
        async def body():
            yield _delta(reasoning="想")
            yield _delta(reasoning="一下")
            yield _delta(tool_calls=[{"index": 0, "id": "call_1",
                                      "function": {"name": "retrieve", "arguments": '{"q"'}}])
            yield _delta(tool_calls=[{"index": 0, "function": {"arguments": ': "x"}'}}])
            yield _usage(7, 9)
            yield b"data: [DONE]\n\n"

        llm = _make(lambda req: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()))

        async def run():
            got_reasoning: list[str] = []
            calls = []
            finished = None
            async for chunk in llm.stream([ChatMessage(role=Role.USER, content="hi")]):
                # ⚠️ `finished` 块也带**完整** reasoning（既有行为，供 history 重建用），
                # 故这里只收增量块，避免把"最终拼接值"误当成多产出一个增量。
                if not chunk.finished and chunk.reasoning:
                    got_reasoning.append(chunk.reasoning)
                calls.extend(chunk.tool_calls)
                if chunk.finished:
                    finished = chunk
            return got_reasoning, calls, finished

        got_reasoning, calls, finished = asyncio.run(run())
        assert got_reasoning == ["想", "一下"], got_reasoning
        assert len(calls) == 1 and calls[0].name == "retrieve", calls
        assert calls[0].arguments == {"q": "x"}, f"工具参数分片未正确拼接：{calls[0].arguments}"
        assert finished is not None
        assert (finished.prompt_tokens, finished.completion_tokens) == (7, 9)
        assert finished.reasoning == "想一下", finished.reasoning
        assert finished.cost >= 0


# ============ ④ 客户端中断应中止上游（真流式带来的额外收益） ============

class TestClientDisconnect:
    def test_closing_consumer_stops_upstream_iteration(self):
        """消费方提前关闭 → 上游生成器不应被读完（省 token/带宽）。

        这是真流式的**额外收益**：原来整段缓冲时，用户点了"停止"上游早已收完；
        现在连接会被关闭，厂商侧也能停。
        """
        produced = {"n": 0}

        async def body():
            for i in range(20):
                produced["n"] += 1
                yield _delta(f"c{i}")
                await asyncio.sleep(0.02)
            yield _usage(1, 1)
            yield b"data: [DONE]\n\n"

        llm = _make(lambda req: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()))

        async def run():
            agen = llm.stream([ChatMessage(role=Role.USER, content="hi")])
            async for chunk in agen:
                if chunk.delta == "c2":
                    break
            await agen.aclose()

        asyncio.run(run())
        assert produced["n"] < 20, f"上游把 20 块都产完了（{produced['n']}）⇒ 中断未传导到上游"
