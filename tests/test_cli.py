"""CLI 最小对话冒烟测试（fake 模式，无需 key）。"""
from __future__ import annotations

import asyncio

from papermate.cli import build_agent


def test_cli_fake_single_turn_runs_react_loop():
    agent = build_agent(model="fake")
    final_text, history = asyncio.run(agent.run("LightCore 的核心贡献是什么？"))

    # 最终回答非空
    assert final_text.strip(), "FakeLLM 应产出非空总结"
    # 轨迹包含一次工具调用（retrieve）与对应的工具结果
    tool_calls = [m for m in history if m.role.value == "assistant" and m.tool_calls]
    tool_results = [m for m in history if m.role.value == "tool"]
    assert tool_calls, "history 应包含 ASSISTANT 的工具调用"
    assert tool_results, "history 应包含 TOOL 结果（retrieve 被执行）"
    assert tool_calls[0].tool_calls[0].name == "retrieve"


def test_cli_fake_multi_turn_independent_retrieve():
    agent = build_agent(model="fake")

    t1, h1 = asyncio.run(agent.run("什么是双层记忆？"))
    t2, h2 = asyncio.run(agent.run("RAG 用了什么检索策略？", h1))

    assert t1.strip() and t2.strip()
    # 两轮各自触发一次 retrieve（每轮 history 末尾 USER 收尾，FakeLLM 重新检索）
    assert any(m.role.value == "tool" for m in h2)
