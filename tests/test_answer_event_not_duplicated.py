"""回答事件被重复发送 → 溯源率被稀释 · 回归测试（2026-10-05）

## 背景（实测，10 条真实轨迹）

15 个 `content` 事件实际只对应 **8 条唯一回答**（7 对重复），且重复的两条**逐字符相同**
（1744 == 1744），只差 `citations` 键（一条有一条没有）：

```
trace_id            ans  uniq  dup
s14-1791119589        2     1    1   ... （10 条轨迹中 7 条如此）
------------------------------------------------
样本(事件)数 = 15；去重后唯一回答 = 8；重复对 = 7
按事件算（现状 citation_rate） = 53.33%     去重后（真实） = 8/8 = 100.00%
```

⇒ 溯源率**无论真实行为如何都被钉在 ~50%**（每条回答恒定贡献「1 无引用 + 1 有引用」）。

## 根因

末步「无（含托管）工具调用」时，回答被**两个生产者各发一次**：

| 生产者 | 位置 |
|---|---|
| `_think` 内补发（供 history 重建） | `loop.py` `if tool_acc: ... else: events.append(message{content})` |
| `_run_stream_impl` FINISH 分支 | `loop.py` `if not real_calls: ... yield message{content}` |

两条都经路由层 `events.append(ev)` 落库，而 `chat_service.persist_turn` 只给**最后一条**
附加 `citations` ⇒ 前一条永远"无引用"。

## 修复（两半，缺一不可）

- **治本**（`loop.py`）：FINISH 分支**仅在本步尚未发过完整回答时**才补发（`_content_emitted`）。
  顺带让 Trace 体积减半（回答正文不再存两份）。
- **治标**（`lightcore/rag/metrics.py`）：`citation_rate` 按 `content` 去重
  （组内任一成员带引用即该条记为带引用）—— 它**同时修正已落库的历史数据**
  （旧轨迹仍是重复形态，见本文件最后一组测试）。

数据层纯本地，不触发大模型 / 网络。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.core.loop import AgentLoop, LoopConfig, LoopEvent, LoopState
from lightcore.llm.base import ChatMessage, Role, ToolCall
from lightcore.llm.fake import FakeLLM
from lightcore.rag.metrics import citation_rate
from lightcore.tools.spec import (
    Permission,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from papermate.skills.dev_eval import collect_metrics

ANSWER = "根据检索结果，本文的核心贡献是提出了一种轻量级 Agent 内核。引用：[chunk-1]。"
CITES = [{"id": "doc-1-c1", "page": 1, "doc_id": 1, "text": "证据段落"}]


# ==================================================================
# A. 治本：循环层不得重复发「完整回答」事件
# ==================================================================
class _PlainLLM(FakeLLM):
    """始终直接给出最终答案（**不调用任何工具**）—— 复现"末步无工具调用"的真实路径。"""

    def _decide(self, last_content: str, messages) -> ChatMessage:  # noqa: ARG002
        return ChatMessage(role=Role.ASSISTANT, content=ANSWER)


class _HostedToolLLM(FakeLLM):
    """只调用**厂商托管工具**（web_search），正文已由厂商并入 content。"""

    def _decide(self, last_content: str, messages) -> ChatMessage:  # noqa: ARG002
        return ChatMessage(
            role=Role.ASSISTANT,
            content=ANSWER,
            tool_calls=[ToolCall(id="w1", name="web_search", arguments={"query": "q"})],
        )


class _ToolThenAnswerLLM(FakeLLM):
    """第一步调用真实工具（retrieve），第二步直接作答 —— 常见 Agent 轨迹。"""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    def _decide(self, last_content: str, messages) -> ChatMessage:  # noqa: ARG002
        if last_content.startswith("RESULT:"):
            return ChatMessage(role=Role.ASSISTANT, content=ANSWER)
        return ChatMessage(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(id="c1", name="retrieve", arguments={"query": "q"})],
        )


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="retrieve", description="t",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission=Permission.SAFE,
        handler=lambda **_kw: ToolResult(ok=True, output={"results": [{"id": "c1", "text": "证据"}]}),
    ))
    return reg


def _answers(events) -> list[str]:
    """所有「完整回答」事件（`content` 非空）的内容 —— 正是 `citation_rate` 的样本口径。"""
    return [
        e.data.get("content", "")
        for e in events
        if e.type == "message" and (e.data or {}).get("content")
    ]


async def _run(llm, *, stream_tokens: bool, web_search: bool = False):
    loop = AgentLoop(llm, ToolExecutor(_registry()), config=LoopConfig(web_search=web_search))
    return [ev async for ev in loop.run_stream("问题", stream_tokens=stream_tokens)]


@pytest.mark.parametrize("stream_tokens", [True, False])
async def test_plain_answer_emitted_exactly_once(stream_tokens: bool) -> None:
    """★ 核心回归：无工具调用的直接作答，事件流里**只能有一条**完整回答。"""
    events = await _run(_PlainLLM(), stream_tokens=stream_tokens)
    got = _answers(events)

    assert len(got) == 1, f"完整回答出现了 {len(got)} 次（修复前为 2 次，content 逐字符相同）：{got}"
    assert got[0] == ANSWER
    assert any(ev.type == "done" for ev in events), "应正常收尾"


async def test_hosted_tool_content_still_emitted_exactly_once() -> None:
    """★ 反向自检：仅托管工具时 `_think` 不会补发 content ⇒ 这里**必须**仍发 1 条，不能变 0。"""
    events = await _run(_HostedToolLLM(), stream_tokens=False, web_search=True)
    got = _answers(events)

    assert len(got) == 1, f"仅托管工具时正文必须回传一次（否则调用方拿不到 final_text）：{got}"
    assert got[0] == ANSWER


@pytest.mark.parametrize("stream_tokens", [True, False])
async def test_tool_then_answer_emitted_exactly_once(stream_tokens: bool) -> None:
    """真实工具轨迹（retrieve → 作答）：最终回答同样只能出现一次。"""
    events = await _run(_ToolThenAnswerLLM(), stream_tokens=stream_tokens)
    got = _answers(events)

    assert len(got) == 1, f"最终回答出现 {len(got)} 次：{got}"
    assert got[0] == ANSWER
    assert any(ev.type == "tool_result" for ev in events), "应真的执行过工具"


# ==================================================================
# B. 治标：citation_rate 按 content 去重
# ==================================================================
def test_duplicate_answer_counted_once() -> None:
    """同一条回答出现两次（一条带引用、一条没有）⇒ 记为**带引用**，比率 1.0（修复前 0.5）。"""
    assert citation_rate([{"content": "答"}, {"content": "答", "citations": CITES}]) == 1.0


def test_duplicate_answer_both_uncited_stays_zero() -> None:
    """反向自检：两次都不带引用 ⇒ 仍应如实记 0，去重不得把"没引用"洗成"有引用"。"""
    assert citation_rate([{"content": "答"}, {"content": "答"}]) == 0.0


def test_distinct_answers_unchanged() -> None:
    """两条**不同**回答（一带有、一无）⇒ 0.5 —— 去重不得误合并不同回答。"""
    assert citation_rate([{"content": "回答一", "citations": CITES}, {"content": "回答二"}]) == pytest.approx(0.5)


def test_blank_samples_keep_legacy_contract() -> None:
    """既有契约：无 `content` 的裸样本（历史调用方手工构造）**不做去重**，各自计一个样本。"""
    assert citation_rate([{"citations": [1]}, {}]) == pytest.approx(0.5)
    assert citation_rate([]) == 0.0


def test_whitespace_only_difference_is_deduped() -> None:
    """首尾空白差异不应被当成两条不同回答。"""
    assert citation_rate([{"content": "答\n"}, {"content": " 答 ", "citations": CITES}]) == 1.0


# ==================================================================
# C. 端到端：治标修复**已落库的旧轨迹**（它们仍是重复形态）
# ==================================================================
@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_dup_answer_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _persist_duplicated(client, content: str = "这是回答") -> None:
    """按**修复前**的真实形态落库：两条相同 content 的事件，最后一条带引用。"""
    sid = client.post("/api/v1/sessions", json={"title": "重复回答"}).json()["id"]
    events = [
        LoopEvent(type="message", state=LoopState.THINK, data={"delta": "片1"}),
        LoopEvent(type="message", state=LoopState.FINISH, data={"content": content}),
        LoopEvent(type="message", state=LoopState.FINISH, data={"content": content}),
    ]
    client.app.state.chat_svc.persist_turn(sid, "问题", [], events, content, "fake", citations=CITES)


def test_legacy_duplicated_trace_now_reports_true_rate(client) -> None:
    """★ 旧轨迹仍是重复形态 ⇒ 治标必须把它**修正为 100%**，而不是继续显示 53%。"""
    _persist_duplicated(client)
    m = collect_metrics(client.app.state.chat_svc._Session)

    assert m["agent"]["citation_rate"] == 1.0, f"旧重复轨迹未被修正：{m['agent']}"


def test_legacy_duplication_is_real_in_stored_events(client) -> None:
    """先确认"重复"确实落进了 events（否则上面的测试是空转）。"""
    _persist_duplicated(client)
    with client.app.state.chat_svc._Session() as db:
        from papermate.models.db import Trace

        t = db.query(Trace).order_by(Trace.id.desc()).first()
        events = json.loads(t.events_json or "[]")

    answers = [e for e in events if e["type"] == "message" and (e["data"] or {}).get("content")]
    assert len(answers) == 2, "本测试要复现的就是「同一回答落库两次」的形态"
    cited = [a for a in answers if (a["data"] or {}).get("citations")]
    assert len(cited) == 1, "只有最后一条被附加引用（这正是缺陷的一半）"
