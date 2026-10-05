"""联网开关（enable_network）端到端接线测试（2026-08-27 晚修复）。

覆盖：
- OAICompatibleLLM.chat/stream 的 web_search 标志：qwen→enable_search；deepseek 不注入（避免 400）。
- AgentLoop 把 LoopConfig.web_search 透传给 LLM（仅 OAICompatibleLLM，不传给 FakeLLM）。
- AgentLoop 把 web_search 视为托管工具：不本地执行，直接以 content 收尾（不崩溃、不误执行）。
均用 httpx.MockTransport 离线验证，不触网。
"""

from __future__ import annotations

import json

import httpx

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, Role, ToolCall
from lightcore.llm.fake import FakeLLM
from lightcore.llm.router import ModelEndpoint, OAICompatibleLLM
from lightcore.tools.spec import ToolExecutor, ToolRegistry


def _qwen_endpoint(model: str = "qwen3.7-flash") -> ModelEndpoint:
    return ModelEndpoint("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "DASHSCOPE_API_KEY", model)


def _deepseek_endpoint(model: str = "deepseek-v4-flash") -> ModelEndpoint:
    return ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", model)


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------- 低层：chat/stream 的 web_search 标志 ----------------
async def test_chat_web_search_flag_qwen_enable_search():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    llm = OAICompatibleLLM(_qwen_endpoint(), api_key="sk", client=_client(handler))
    await llm.chat([ChatMessage(role=Role.USER, content="hi")], web_search=True)
    assert captured["body"].get("enable_search") is True


async def test_chat_web_search_flag_deepseek_no_tool_injection():
    """DeepSeek 不支持原生联网：即便 web_search=True 也不得注入非法 web_search 工具（避免 400）。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    llm = OAICompatibleLLM(_deepseek_endpoint(), api_key="sk", client=_client(handler))
    await llm.chat(
        [ChatMessage(role=Role.USER, content="hi")],
        tools=[{"type": "function", "function": {"name": "retrieve", "description": "d", "parameters": {}}}],
        web_search=True,
    )
    tools = captured["body"].get("tools", [])
    assert not any(t.get("type") == "web_search" for t in tools), "deepseek 不应注入 web_search 工具"
    assert any(t.get("type") == "function" and t["function"]["name"] == "retrieve" for t in tools), "原工具应保留"


# ---------------- AgentLoop 透传 web_search 给 OAICompatibleLLM ----------------
async def test_agentloop_threads_web_search_to_llm():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "答"}}], "usage": {}})

    llm = OAICompatibleLLM(_qwen_endpoint(), api_key="sk", client=_client(handler))
    agent = AgentLoop(
        llm=llm,
        tools=ToolExecutor(ToolRegistry()),
        config=LoopConfig(web_search=True),
        system_prompt="sys",
    )
    events = [ev async for ev in agent.run_stream("问题？", [], stream_tokens=False)]
    assert any(ev.type == "done" for ev in events)
    assert captured["body"].get("enable_search") is True, "web_search 应透传到 LLM 请求"


async def test_agentloop_no_web_search_when_disabled():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "答"}}], "usage": {}})

    llm = OAICompatibleLLM(_qwen_endpoint(), api_key="sk", client=_client(handler))
    agent = AgentLoop(
        llm=llm,
        tools=ToolExecutor(ToolRegistry()),
        config=LoopConfig(web_search=False),
        system_prompt="sys",
    )
    list_of_ev = [ev async for ev in agent.run_stream("问题？", [], stream_tokens=False)]
    assert any(ev.type == "done" for ev in list_of_ev)
    assert "enable_search" not in captured["body"], "未开启联网不应注入 enable_search"


# ---------------- AgentLoop 把 web_search 当托管工具：不本地执行 ----------------
class _WebSearchFakeLLM(FakeLLM):
    """返回一条带 web_search 工具调用、但 content 已有联网答案的消息（模拟厂商托管工具已服务端执行）。"""

    def _decide(self, last_content, messages):
        return ChatMessage(
            role=Role.ASSISTANT,
            content="根据联网检索：2026 年 AI 领域最新进展是……（来源：example.com）",
            tool_calls=[ToolCall(id="w1", name="web_search", arguments={})],
        )


async def test_agentloop_hosted_web_search_not_executed():
    """web_search 是托管工具，AgentLoop 不应尝试本地执行（否则会因未注册而崩溃）。"""
    llm = _WebSearchFakeLLM()
    agent = AgentLoop(
        llm=llm,
        tools=ToolExecutor(ToolRegistry()),
        config=LoopConfig(web_search=True),
        system_prompt="sys",
    )
    events = [ev async for ev in agent.run_stream("最新 AI 新闻？", [], stream_tokens=False)]
    done = [ev for ev in events if ev.type == "done"]
    errors = [ev for ev in events if ev.type == "error"]
    assert done, "应正常收尾"
    assert not errors, f"不应出现 error 事件：{errors}"
    # 最终内容即为厂商回传的联网答案（未被本地执行改写）
    contents = [ev.data.get("content") for ev in events if ev.type == "message" and ev.data.get("content")]
    assert any("联网检索" in c for c in contents), "应保留联网答案正文"


def test_loopconfig_default_web_search_false():
    assert LoopConfig().web_search is False
