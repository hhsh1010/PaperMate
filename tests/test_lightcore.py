"""LightCore 单元测试（docs/08 测试金字塔 · 单元层）

覆盖：LLM 抽象（FakeLLM 闭环）、工具系统（权限/执行/失败计数）、Agent 循环（ReAct 状态机）。
使用 FakeLLM 替身，无需网络与 API key。
"""

from __future__ import annotations

import asyncio

from lightcore.core.loop import AgentLoop, LoopState
from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import FakeLLM
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec


def test_fake_llm_emits_tool_call_then_answer():
    async def run():
        llm = FakeLLM()
        # 第一轮应返回工具调用
        r1 = await llm.chat([ChatMessage(role=Role.USER, content="hi")])
        assert r1.message.tool_calls, "FakeLLM 首轮应决策调用工具"
        # 第二轮（带工具结果）应返回最终答案
        r2 = await llm.chat([ChatMessage(role=Role.TOOL, content="RESULT: x", tool_call_id="call_1")])
        assert not r2.message.tool_calls
        assert r2.message.content
        return r1, r2

    asyncio.run(run())


def test_tool_registry_and_permission():
    reg = ToolRegistry()
    register_builtin_tools(reg)
    assert "retrieve" in reg.names()
    assert "calculator" in reg.names()
    # calculator 安全计算
    exe = ToolExecutor(reg)
    res = asyncio.run(exe.execute("calculator", {"expression": "(1+2)*3"}))
    assert res.ok and res.output["result"] == 9.0
    # 网络权限默认关闭 → retrieve 是 safe 应可跑；未关联文档时明确告知（不回退假数据）
    res2 = asyncio.run(exe.execute("retrieve", {"query": "LightCore", "top_k": 1}))
    assert res2.ok and res2.output["source"] == "none"
    assert res2.output["results"] == []
    assert res2.output.get("note")


def test_code_interpreter_sandbox_rejected_by_default():
    reg = ToolRegistry()
    register_builtin_tools(reg)
    exe = ToolExecutor(reg)
    res = asyncio.run(exe.execute("code_interpreter", {"code": "print(1)", "lang": "python"}))
    # NFR-07：SANDBOX 权限默认关闭，未显式开启则拒绝执行
    assert not res.ok
    assert "权限" in res.error or "sandbox" in res.error


def test_agent_loop_react_closure():
    async def run():
        reg = ToolRegistry()
        register_builtin_tools(reg)
        loop = AgentLoop(FakeLLM(), ToolExecutor(reg))
        events = [ev async for ev in loop.run_stream("LightCore 的核心贡献？")]
        types = [e.type for e in events]
        assert "thinking" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "message" in types
        final = next(e for e in reversed(events) if e.type == "message")
        assert final.data["content"]
        return types

    asyncio.run(run())


def test_agent_loop_max_iterations_abort():
    async def run():
        reg = ToolRegistry()
        # 注册一个永远返回空结果、诱导循环的工具
        reg.register(ToolSpec(
            name="noop", description="noop", parameters={"type": "object", "properties": {}},
            permission=Permission.SAFE,
            handler=lambda: ToolResult(ok=True, output="empty"),
        ))
        # 自定义 FakeLLM：永远调用 noop
        from lightcore.llm.fake import FakeLLM

        class LoopFake(FakeLLM):
            async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
                from lightcore.llm.base import ToolCall
                r = await super().chat(messages, tools, temperature=temperature, max_tokens=max_tokens, tool_choice=tool_choice)
                # 强制每次都调用 noop（覆盖默认决策）
                r.message.tool_calls = [ToolCall(id="c", name="noop", arguments={})]
                r.message.content = "calling"
                return r

        loop = AgentLoop(LoopFake(), ToolExecutor(reg), config=__import__("lightcore.core.loop", fromlist=["LoopConfig"]).LoopConfig(max_iterations=3))
        events = [ev async for ev in loop.run_stream("loop me")]
        assert any(e.type == "error" and e.state == LoopState.ABORT for e in events)

    asyncio.run(run())


def test_agent_loop_injects_compacted_retrieve_result():
    """回归守卫（2026-09-06）：AgentLoop 必须把工具结果经 `_format_tool_result` 注入上下文。

    历史缺陷：loop 曾直接拼 `f"RESULT: {result.output}"`，使 _format_tool_result（含 retrieve
    的 200 字压缩）成为**死代码**——单次检索 6 段全文直灌，与 8000 token 预算冲突，
    表现为长对话过早压缩历史（"AI 忘了前面说过什么"）。改动仅一行却极易复发，故加测试锁定。
    """
    long_text = "检索段落正文" * 100  # 600 字，远超 200 字上限

    class _RecordingLLM:
        """最小 LLM 替身：记录 THINK 收到的 messages；首轮调 retrieve、次轮给最终答案。"""

        def __init__(self) -> None:
            self.seen: list[list[ChatMessage]] = []
            self.n = 0

        async def chat(self, messages, *args, **kw):  # noqa: ANN001, ANN002
            from lightcore.llm.base import LLMResponse, ToolCall

            self.seen.append(list(messages))
            self.n += 1
            if self.n == 1:
                return LLMResponse(message=ChatMessage(
                    role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="c1", name="retrieve", arguments={"query": "q"})]))
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="最终答案"))

    def _retrieve_stub(**_kw):  # noqa: ANN001
        return ToolResult(ok=True, output={
            "results": [{"id": f"doc-1-c{i}", "page": i + 1, "score": 0.9, "text": long_text}
                        for i in range(6)],
        })

    async def run():
        reg = ToolRegistry()
        reg.register(ToolSpec(
            name="retrieve", description="检索",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            permission=Permission.SAFE, handler=_retrieve_stub,
        ))
        llm = _RecordingLLM()
        loop = AgentLoop(llm, ToolExecutor(reg))  # type: ignore[arg-type]
        [ev async for ev in loop.run_stream("测试问题", stream_tokens=False)]  # type: ignore[call-arg]
        # 第二轮 THINK 的输入里应含被压缩后的 TOOL 消息
        tool_msgs = [m for m in llm.seen[-1] if m.role == Role.TOOL]
        assert tool_msgs, "第二轮 THINK 应带上工具结果"
        injected = tool_msgs[0].content
        assert "检索段落正文" in injected
        # 6 段 × (200 字 + 省略号) ≈ 1200 字，远小于未压缩的 3600 字
        assert len(injected) < len(long_text) * 6 / 2, f"工具结果未被压缩：len={len(injected)}"
        assert injected.count("…") == 6, "每段都应被截断并带省略号"

    asyncio.run(run())


def test_oai_compatible_accepts_dict_tools_from_registry():
    """回归：ToolRegistry.specs_for_llm() 返回 list[dict]（来自 ToolSpec.to_llm()），
    OAICompatibleLLM.chat/stream 必须能直接消费 dict 而不报 AttributeError。
    
    这一回归在生产中表现为 'LLM 调用失败：dict' object has no attribute 'to_openai''，
    前端只能看到 thinking(0)+error+done，没有 message_delta，文本完全缺失。
    """
    import json

    import httpx

    from lightcore.llm.router import ModelEndpoint, OAICompatibleLLM

    captured = {}

    def fake_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["payload"] = body
        # 验证 tools 字段已正确塞入（即使是 dict 也能正常 OpenAI 兼容）
        assert "tools" in body, "OAI 兼容端点必须收到 tools 字段"
        assert isinstance(body["tools"][0], dict)
        assert body["tools"][0]["function"]["name"] in {"calc"}
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        )

    transport = httpx.MockTransport(fake_handler)
    client = httpx.AsyncClient(transport=transport)
    ep = ModelEndpoint("mock", "https://example.com/v1", "MOCK_KEY", "mock-model")
    llm = OAICompatibleLLM(ep, client=client, api_key="sk-test")

    # 模拟 ToolRegistry.specs_for_llm() 返回的 dict 列表
    tool_dicts = [
        {"type": "function", "function": {"name": "calc", "description": "calc",
                                          "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}}},
    ]

    async def run():
        # chat 路径
        r = await llm.chat([ChatMessage(role=Role.USER, content="hi")], tool_dicts)
        assert r.message.content == "ok"
        # stream 路径（生产出错的那条）
        chunks = []
        async for c in llm.stream([ChatMessage(role=Role.USER, content="hi")], tool_dicts):
            chunks.append(c)
        assert any(c.finished for c in chunks)
        await client.aclose()
        return r

    asyncio.run(run())
    assert "tools" in captured["payload"]
