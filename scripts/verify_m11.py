"""M1.1 验收脚本（docs/07 §M1.1 验收标准，有证据版）

逐条验证：
  A. 终端多轮对话 + --stream 逐字渲染
  B. 断网/限流触发重试日志 + 成本统计正确（含 latency）
  C. 换供应商（DeepSeek ↔ Qwen）只改配置不改代码

运行：python scripts/verify_m11.py
退出码 0 = 全部通过；非 0 = 有未达标项。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import httpx

# 确保 src 在 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lightcore.core.loop import AgentLoop, LoopConfig  # noqa: E402
from lightcore.llm.base import ChatMessage, Role  # noqa: E402
from lightcore.llm.fake import FakeLLM  # noqa: E402
from lightcore.llm.router import (  # noqa: E402
    PRESET_ENDPOINTS,
    ModelRouter,
    OAICompatibleLLM,
    estimate_cost,
)
from lightcore.tools.spec import ToolExecutor, ToolRegistry  # noqa: E402

# 真实端点测试需要 key（仅用于本地成本/延迟字段；不联网，client 被注入桩）
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-for-local-verify")


# ---- 验收辅助桩 ----
class FakeResp:
    def __init__(self, status: int, data: dict):
        self.status_code = status
        self._data = data
        self.request = httpx.Request("POST", "http://x")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=self.request, response=self)

    def json(self):
        return self._data


class FakeHttpClient:
    def __init__(self, behavior):
        self.behavior = behavior
        self.post_calls = 0

    async def post(self, url, json=None, headers=None):
        self.post_calls += 1
        return self.behavior()


def _ok_payload(content="hi", pt=10, ct=5):
    return FakeResp(200, {
        "choices": [{"message": {"content": content, "tool_calls": []}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
    })


def behavior_retry_ok():
    state = {"n": 0}

    def f():
        state["n"] += 1
        if state["n"] <= 2:
            raise httpx.ConnectError(f"conn-{state['n']}")  # 前两次断网
        return _ok_payload()

    return f


def behavior_always_conn():
    def f():
        raise httpx.ConnectError("boom")

    return f


def behavior_429():
    def f():
        return FakeResp(429, {})

    return f


class ChatOnlyFake(FakeLLM):
    """纯对话 Fake（不触发工具），专用于验收逐字渲染与多轮。"""

    def _decide(self, last_content, messages):
        return ChatMessage(role=Role.ASSISTANT,
                           content="这是一条逐字渲染的示例回答，用于验收 M1.1 的流式输出能力。")


# ---- 验收主体 ----
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"  [{mark}] {name}\n        {detail}")


async def verify_A() -> None:
    print("\n=== A. 多轮对话 + 逐字渲染 ===")
    registry = ToolRegistry()
    agent = AgentLoop(llm=ChatOnlyFake(), tools=ToolExecutor(registry),
                      system_prompt="", config=LoopConfig(temperature=0.3))
    # 多轮
    h: list[ChatMessage] = []
    ok_multi = True
    for q in ["问题一", "问题二"]:
        ft, h = await agent.run(q, h)
        if not ft:
            ok_multi = False
    record("A1 多轮对话", ok_multi and len(h) >= 4,
           f"两轮后 history 消息数={len(h)}，各轮均有非空回答" if ok_multi else "存在空回答")

    # 逐字：流式模式下 message delta 事件应 > 1
    h = []
    deltas = 0
    async for ev in agent.run_stream("逐字测试", h, stream_tokens=True):
        if ev.type == "message" and ev.data.get("delta") is not None:
            deltas += 1
    record("A2 --stream 逐字渲染", deltas > 1,
           f"流式产出增量 message(delta) 事件数={deltas}（>1 即逐字）")


async def verify_B() -> None:
    print("\n=== B. 重试日志 + 成本统计 ===")
    # 捕获 lightcore.llm 的日志
    cap: list[logging.LogRecord] = []

    class Cap(logging.Handler):
        def emit(self, rec):
            cap.append(rec)

    log = logging.getLogger("lightcore.llm")
    handler = Cap()
    log.addHandler(handler)
    log.setLevel(logging.WARNING)

    # B1 断网重试成功
    client = FakeHttpClient(behavior_retry_ok())
    llm = OAICompatibleLLM(PRESET_ENDPOINTS["deepseek"], client=client, max_retries=3)
    resp = await llm.chat([ChatMessage(role=Role.USER, content="hi")])
    retried = client.post_calls
    logged_retry = any("重试" in r.getMessage() or "退避" in r.getMessage() for r in cap)
    record("B1 断网重试 + 重试日志",
           retried == 3 and resp.message.content == "hi" and logged_retry,
           f"POST 调用次数={retried}（前2次断网第3次成功），重试日志已输出={logged_retry}")

    # B2 持续失败 → 抛错且尝试 max_retries 次
    cap.clear()
    client2 = FakeHttpClient(behavior_always_conn())
    llm2 = OAICompatibleLLM(PRESET_ENDPOINTS["deepseek"], client=client2, max_retries=3)
    raised = False
    try:
        await llm2.chat([ChatMessage(role=Role.USER, content="x")])
    except httpx.ConnectError:
        raised = True
    record("B2 持续失败按最大次数重试", raised and client2.post_calls == 3,
           f"持续断网下 POST 调用次数={client2.post_calls}（==max_retries=3），最终抛出 ConnectError={raised}")

    # B3 限流 429 可重试
    cap.clear()
    client3 = FakeHttpClient(behavior_429())
    llm3 = OAICompatibleLLM(PRESET_ENDPOINTS["deepseek"], client=client3, max_retries=3)
    raised429 = False
    try:
        await llm3.chat([ChatMessage(role=Role.USER, content="x")])
    except httpx.HTTPStatusError:
        raised429 = True
    record("B3 限流(429)触发重试", raised429 and client3.post_calls == 3,
           f"429 下 POST 调用次数={client3.post_calls}（==3），最终抛出 HTTPStatusError={raised429}")

    # B4 成本 + 延迟统计正确
    cap.clear()
    client4 = FakeHttpClient(lambda: _ok_payload(pt=10, ct=5))
    llm4 = OAICompatibleLLM(PRESET_ENDPOINTS["deepseek"], client=client4, max_retries=3)
    resp4 = await llm4.chat([ChatMessage(role=Role.USER, content="hi")])
    expect_cost = estimate_cost("deepseek-chat", 10, 5)
    cost_ok = abs(resp4.cost - expect_cost) < 1e-9
    lat_ok = isinstance(resp4.latency_ms, int) and resp4.latency_ms >= 0
    record("B4 成本/延迟统计正确",
           cost_ok and lat_ok and resp4.prompt_tokens == 10 and resp4.completion_tokens == 5,
           f"prompt={resp4.prompt_tokens} completion={resp4.completion_tokens} "
           f"cost=¥{resp4.cost:.6f}(期望¥{expect_cost:.6f}) latency={resp4.latency_ms}ms")

    log.removeHandler(handler)


async def verify_C() -> None:
    print("\n=== C. 换供应商只改配置不改代码 ===")
    r_q = ModelRouter.from_presets(prefer="qwen")
    r_d = ModelRouter.from_presets(prefer="deepseek")
    q_ok = (r_q.get("qwen").endpoint.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
            and r_q.get("qwen").model_name == "qwen-plus")
    d_ok = (r_d.get("deepseek").endpoint.base_url == "https://api.deepseek.com/v1"
            and r_d.get("deepseek").model_name == "deepseek-chat")
    record("C 供应商切换（Qwen↔DeepSeek）", q_ok and d_ok,
           f"qwen→{r_q.get('qwen').endpoint.base_url} ; deepseek→{r_d.get('deepseek').endpoint.base_url} "
           f"（仅改 prefer 配置，无代码改动）")


async def main() -> int:
    print("M1.1 验收（docs/07 §M1.1）")
    await verify_A()
    await verify_B()
    await verify_C()
    print("\n=== 验收汇总 ===")
    ok = all(ok for _, ok, _ in RESULTS)
    for name, passed, _ in RESULTS:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n结论：{'全部通过' if ok else '存在未达标项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
