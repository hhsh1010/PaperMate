"""Spike S1 · LLM Function Calling 验证线（docs/13 第 0 周 Spike S1）

目标：确认 LightCore 能在「无 API key」环境下跑通「模型决策→工具调用→OBSERVE→FINISH」闭环。
本地用 FakeLLM（确定性替身）驱动 AgentLoop；真实端点（DeepSeek/OAI）脚本另见下方注释，
仅在配置环境变量后可用。
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")

from lightcore.core.loop import AgentLoop
from lightcore.llm.fake import FakeLLM
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry


async def main() -> None:
    # 1) 构造内核
    registry = ToolRegistry()
    register_builtin_tools(registry)
    executor = ToolExecutor(registry)
    llm = FakeLLM()
    loop = AgentLoop(llm, executor, system_prompt="你是论文研读助手 PaperMate。")

    # 2) 跑通一轮（含一次工具调用）
    user_q = "LightCore 的核心贡献是什么？请检索相关段落后回答。"
    print(f"[S1] 用户: {user_q}\n")
    final_text = ""
    async for ev in loop.run_stream(user_q):
        if ev.type == "thinking":
            print(f"  · THINK (step={ev.data['step']})")
        elif ev.type == "tool_call":
            print(f"  · ACT   → 调用 {ev.data['name']}({ev.data['arguments']})")
        elif ev.type == "tool_result":
            ok = ev.data["ok"]
            body = ev.data.get("output") if ok else ev.data.get("error")
            print(f"  · OBSERVE → {'OK' if ok else 'FAIL'} {body}")
        elif ev.type == "message":
            final_text = ev.data["content"]
            print(f"  · FINISH → {final_text}")
        elif ev.type == "error":
            print(f"  · ERROR/ABORT → {ev.data}")

    print("\n[S1] 结论:", "PASS ✅ — FakeLLM 闭环跑通" if final_text else "FAIL ❌")


"""
真实端点验证（需 key，默认不执行）：
  import os
  os.environ["DEEPSEEK_API_KEY"] = "sk-..."
  r = ModelRouter.from_presets(prefer="deepseek", fake=False)
  loop = AgentLoop(r.get("deepseek"), executor)
  # 同上加 async for ev in loop.run_stream(user_q)
"""


if __name__ == "__main__":
    asyncio.run(main())
