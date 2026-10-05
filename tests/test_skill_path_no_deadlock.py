"""技能路径的两条底线：**不许忙等死锁** + **必须逐块流式**。

### 一、2026-09-27 P0 忙等死锁（本文件的存在原因）
`routes_search.py` 的技能等待循环原为：

```python
skill_task = asyncio.create_task(skill.run(ctx))
while True:                       # ← 循环体内零 await
    if skill_task.done():
        ...
        break
```

那句 `await asyncio.sleep(0.3)` 误写在循环**之外**（`if recorder is not None:` 分支里），
只有技能**跑完之后**才执行得到。

**后果链**：`while True` 独占事件循环 ⇒ 被它 `create_task` 的 task **永远得不到调度**
⇒ `done()` 恒为 False ⇒ **永久死锁 + 单核 100%**：后端无法处理任何请求、
连接堆积 CLOSE_WAIT、日志停止。

**用户可观测现象**：`/实验设计 @文档一`（**有**关联文档）发出去后前端**迟迟无回复**
——不是报错，是永远转圈。

**实测取证（2026-09-27 22:53 起）**：
- `py-spy dump` 连续 5 次采样**全部**停在 `routes_search.py:1037`，从不出现 sleep 帧；
- 后端 PID 44780 积压 **199 个 CLOSE_WAIT**；日志静止 **23 分钟**。

### 二、2026-09-28 技能流式化
技能内部走 `llm.chat()`（一次性返回）⇒ 用户看到「整段突然刷出」，与问答路径
（逐 `message_delta`）不一致。修法是在**路由层**包 `_StreamingChatProxy`
（`chat()` 内部改走 `stream()`，增量进队列由路由层逐块下发），**12 个技能零改动**。

### ★ 为什么分两层测（踩过才知道）
集成层（走 HTTP）**不足以**覆盖这个循环：会话「无文档」会在更早处提前返回
（响应是"请先关联文档再执行命令"），于是集成测试**假通过** —— 反向自检时
还原成 `while True` 仍然 `passed`，就是这个原因。
⇒ 精确覆盖必须靠**单元层**直接调 `_stream_skill_output` / `_StreamingChatProxy`；
集成层只负责兜住「命令路径不会把整个后端卡死」这条用户可观测的底线。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.llm.base import ChatMessage, LLMStreamChunk, Role
from papermate.api.routes_search import _stream_skill_output, _StreamingChatProxy

#: 修复前是**永久**忙等，超时只是兜底（单元层另用 `asyncio.wait_for` 保护）。
_DEADLOCK_TIMEOUT_S = 15


class _FakeRequest:
    """只需 `is_disconnected`；`disconnected` 可切换以模拟断连。"""

    def __init__(self, disconnected: bool = False) -> None:
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return bool(self.disconnected)


class _StreamOnlyLLM:
    """只有 `stream()` 的替身 —— 用来验证代理把非流式接口「架」在流式实现之上。

    刻意**不实现 `chat`**：一旦代理偷偷回落到 `chat()`，本替身会当场 AttributeError。
    """

    model_name = "stub-stream-only"

    def __init__(self, pieces: tuple[str, ...] = ("甲", "乙")) -> None:
        self.pieces = pieces

    async def stream(self, messages, *a, **kw):  # noqa: ANN001, ANN002
        for s in self.pieces:
            yield LLMStreamChunk(delta=s)
        yield LLMStreamChunk(
            finished=True, prompt_tokens=11, completion_tokens=2,
            latency_ms=7, cost=0.0012, reasoning="想了",
        )


# ------------------------- 一、单元层：等待/消费循环 -------------------------


def test_consume_returns_when_task_completes():
    """消费循环必须让出事件循环 —— 否则被等的 task 永远跑不到（2026-09-27 P0）。"""

    async def _go() -> list[str]:
        task = asyncio.create_task(asyncio.sleep(0.05, result="ok"))
        return [d async for d in _stream_skill_output(
            task, asyncio.Queue(), _FakeRequest(), threading.Event(), 1, poll_s=0.01
        )]

    # ⚠️ `wait_for` 只是**正常路径**下的保护：一旦真的退化成忙等，**超时回调本身也拿不到
    # 事件循环调度** ⇒ 测试会**挂死**（实测：反向自检时 pytest 被外层 `timeout 150` 杀掉，
    # exit=124）而不是干净地失败。所以 CI 侧仍需**外层超时**兜底 —— 这也正是"忙等"
    # 比普通死锁更危险的地方：任何基于事件的超时都形同虚设。
    assert asyncio.run(asyncio.wait_for(_go(), timeout=5)) == []


def test_consume_cancels_on_disconnect():
    """客户端断连 ⇒ 取消技能（原实现这段逻辑写在循环外，从未生效过）。"""

    async def _go() -> bool:
        task = asyncio.create_task(asyncio.sleep(30))
        async for _ in _stream_skill_output(
            task, asyncio.Queue(), _FakeRequest(disconnected=True), threading.Event(), 1, poll_s=0.01
        ):
            pass
        # `cancel()` 只是**请求**取消；要再让调度器跑一轮，`cancelled()` 才为 True。
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return task.cancelled()

    assert asyncio.run(asyncio.wait_for(_go(), timeout=5)) is True


def test_consume_cancels_on_stop_flag():
    """显式「停止」事件（`cancel_flag`）与断连同效。"""

    async def _go() -> bool:
        task = asyncio.create_task(asyncio.sleep(30))
        flag = threading.Event()
        flag.set()
        async for _ in _stream_skill_output(task, asyncio.Queue(), _FakeRequest(), flag, 1, poll_s=0.01):
            pass
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return task.cancelled()

    assert asyncio.run(asyncio.wait_for(_go(), timeout=5)) is True


def test_consume_yields_deltas_until_task_done():
    """增量必须**全部**取出后再结束（结束判据是 `done() + empty()`，不是首个哨兵）。

    技能可能多次调用 LLM（如 `/规划`），用"一次调用一个哨兵"的写法会提前终止、丢内容。
    """

    async def _go() -> list[str]:
        q: asyncio.Queue = asyncio.Queue()

        async def _producer() -> None:
            for s in ("一", "二", "三"):
                await q.put(LLMStreamChunk(delta=s))

        task = asyncio.create_task(_producer())
        return [d async for d in _stream_skill_output(
            task, q, _FakeRequest(), threading.Event(), 1, poll_s=0.01
        )]

    assert asyncio.run(asyncio.wait_for(_go(), timeout=5)) == ["一", "二", "三"]


# ------------------------- 二、单元层：流式代理 -------------------------


def test_proxy_turns_chat_into_stream_and_keeps_response_shape():
    """`chat()` 返回值必须与真实 `chat()` **同构**，否则技能侧会出现隐性差异。"""
    inner = _StreamOnlyLLM(("甲", "乙"))

    async def _go():
        q: asyncio.Queue = asyncio.Queue()
        proxy = _StreamingChatProxy(inner, q)
        resp = await proxy.chat([ChatMessage(role=Role.USER, content="hi")], temperature=0.3)
        pushed = []
        while not q.empty():
            pushed.append(q.get_nowait().delta)
        return resp, pushed

    resp, pushed = asyncio.run(asyncio.wait_for(_go(), timeout=5))
    assert resp.message.content == "甲乙"
    assert resp.message.reasoning == "想了"
    assert resp.message.role == Role.ASSISTANT
    assert (resp.prompt_tokens, resp.completion_tokens) == (11, 2)
    assert (resp.latency_ms, resp.cost) == (7, 0.0012)
    assert resp.model == "stub-stream-only"
    # 上报与返回值同源：推给 SSE 的正是拼进 message 的那些增量
    assert pushed == ["甲", "乙"]


def test_proxy_forwards_non_chat_attributes():
    """`model_name` / `chat_web_search` 等必须透传 —— 技能的能力检测依赖它们。"""
    q: asyncio.Queue = asyncio.Queue()
    proxy = _StreamingChatProxy(_StreamOnlyLLM(), q)
    assert proxy.model_name == "stub-stream-only"
    with pytest.raises(AttributeError):
        _ = proxy.not_exist_attr   # 透传后仍按底层语义报错，不吞掉


def test_proxy_does_not_swallow_llm_failure():
    """底层流失败必须**照常抛出** —— `synthesize()` 靠 `except` 回退到确定性输出。

    若代理在此吞掉异常，技能会拿到空内容并当作"模型说没有" ⇒ 静默降级、无可观测告警。
    """

    class _BoomLLM:
        model_name = "boom"

        async def stream(self, *a, **kw):  # noqa: ANN002, ANN003
            yield LLMStreamChunk(delta="半")
            raise RuntimeError("上游 500")

    async def _go():
        proxy = _StreamingChatProxy(_BoomLLM(), asyncio.Queue())
        with pytest.raises(RuntimeError, match="上游 500"):
            await proxy.chat([ChatMessage(role=Role.USER, content="hi")])

    asyncio.run(asyncio.wait_for(_go(), timeout=5))


# ------------------------- 三、集成层：命令路径不会卡死后端 -------------------------


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_skill_deadlock_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    rs_mod._state = None
    cfg_mod._settings = None


def test_skill_command_does_not_hang_the_backend(client):
    """命令类请求必须能结束，且其后普通请求仍拿得到响应。

    ⚠️ 本层**不保证**覆盖到技能等待循环（无文档时会提前返回，见模块 docstring）；
    它的职责是兜住「整条命令路径不会把后端卡死」这个**用户可观测**的底线。
    """
    sid = client.post("/api/v1/sessions", json={"title": "命令路径"}).json()["id"]
    box: dict = {}

    def _run() -> None:
        try:
            box["resp"] = client.post(
                f"/api/v1/sessions/{sid}/messages/stream",
                json={"content": "/实验设计", "stream_id": "skill-hang"},
            )
        except Exception as e:  # noqa: BLE001  把异常带回主线程判定
            box["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    after: dict = {}

    def _probe() -> None:
        try:
            after["resp"] = client.get("/api/v1/sessions")
        except Exception as e:  # noqa: BLE001
            after["error"] = f"{type(e).__name__}: {e}"

    probe = threading.Thread(target=_probe, daemon=True)
    probe.start()

    for t in (thread, probe):
        t.join(timeout=_DEADLOCK_TIMEOUT_S)

    assert not thread.is_alive(), "命令请求未结束 ⇒ 疑似事件循环被独占（忙等死锁）"
    assert not probe.is_alive(), "命令执行期间普通请求拿不到响应 ⇒ 事件循环被饿死"
    assert "error" not in box, box.get("error")
    assert box["resp"].status_code == 200, box["resp"].text
    assert after.get("resp") is not None, after.get("error")
    assert after["resp"].status_code == 200
