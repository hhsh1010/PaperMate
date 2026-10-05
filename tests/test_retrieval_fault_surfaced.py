"""F7（2026-09-28 修复）：**检索服务暂时不可用**必须真的告诉用户。

## 被修的缺陷（实测发现，非推测）

C3 实施时做反向自检，放宽了一个谓词后**端到端用例仍然通过** —— 不合常理。
加探针实测得到：

```
[PROBE] tool_result 事件： [{'call_id': 'call_1', 'ok': False, 'error': None}]
[PROBE] SSE 是否含 retrieval_error notice： False
```

**根因**：`loop._run_one` 对失败工具产出的事件只放 `{call_id, ok, error}`、**丢弃 `output`**；
而路由层的 F7 探测读的正是 `output`：

```python
out = ev.data.get("output")                      # ← 失败时恒为 None
if isinstance(out, dict) and out.get("source") == "rag":
    if out.get("error") == "retrieval_unavailable":
        _retrieval_error = True                  # ← 永不执行（死代码）
```

⇒ `_retrieval_error` 恒为 False ⇒ `_RETRIEVAL_ERROR_REPLY` 与 `_retrieval_error_notice()`
**两套文案都到不了用户**。索引损坏时用户看到的是通用兜底
「本次没有生成有效回答（可能是模型调用失败或返回内容为空）」——
**指向了错误的自救方向**（该去重建索引，却被引导去查余额 / 换模型）。

## 本文件在守什么

1. ★ **失败事件必须带 `output`**（`loop._run_one` 的契约）—— 这是 F7 探测的**唯一数据来源**。
   这条契约极易被下一次"顺手精简事件字段"抹掉，且抹掉后**不报错、只是静默失效**
   （本项目最典型的一类教训）。
2. ★ **用户可见的最终结果**：`retrieval_error` notice 真的出现在 SSE 与落库 meta 里
   （刷新后警示卡仍在）。
3. **两个 API 一致**：`run()`（CLI，非流式）从事件重建 history 时，
   失败工具不得退化成 `RESULT: None`（那与 `run_stream` 注入真实 output 的做法不一致）。
4. **不得误伤**：普通工具失败（非检索故障）**不得**被报成"检索服务不可用"。
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.fake import FakeLLM
from lightcore.rag.pipeline import RetrievalUnavailable, RetrievedChunk
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec

DOC = "量子纠缠是量子力学中一种非局域关联现象。对其中一个粒子测量会瞬间决定另一个的状态。"


class _BoomPipe:
    """检索必然抛 `RetrievalUnavailable`（模拟索引损坏 / 嵌入或重排服务异常）。"""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RetrievalUnavailable("模拟索引损坏")

    def get_chunks(self):  # noqa: ANN201
        return [RetrievedChunk(id="doc-9-c0", text="正文", page=1, score=0.5)]

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        raise self._exc


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_f7_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _prepare(client, monkeypatch, pipe) -> int:
    doc_svc = client.app.state.doc_svc
    sid = client.post("/api/v1/sessions", json={"title": "f7"}).json()["id"]
    files = {"file": ("doc.md", DOC.encode(), "text/markdown")}
    assert client.post(f"/api/v1/documents?session_id={sid}", files=files).status_code == 200
    client.patch(f"/api/v1/sessions/{sid}", json={"config": {"model": "fake"}})

    async def _fake_get_pipeline(_doc_id, **_kw):
        return pipe

    monkeypatch.setattr(doc_svc, "get_pipeline", _fake_get_pipeline)
    return sid


def _ask(client, sid: int) -> str:
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "文档里的量子纠缠是什么？"})
    assert r.status_code == 200, r.text
    return r.text


def _events(client, sid: int) -> list[dict]:
    from papermate.models.db import Trace

    with client.app.state.chat_svc._Session() as s:  # noqa: SLF001  与 routes_metrics 同一取库方式
        t = s.query(Trace).filter(Trace.session_id == sid).order_by(Trace.id.desc()).first()
        assert t is not None
        return json.loads(t.events_json or "[]")


# --------------------------------------- ① 契约：失败事件必须带 output（F7 的数据来源）


def _loop_with_failing_tool(exc: Exception) -> AgentLoop:
    reg = ToolRegistry()

    def _boom(**_kw):
        return ToolResult(ok=False, output={"source": "rag", "results": [],
                                            "error": "retrieval_unavailable"},
                          error=None)

    reg.register(ToolSpec(name="retrieve", description="测试用",
                          parameters={"type": "object", "properties": {}},
                          permission=Permission.SAFE, handler=_boom))
    return AgentLoop(FakeLLM(), ToolExecutor(reg), config=LoopConfig(max_iterations=3))


def test_failed_tool_event_carries_output(caplog):
    """★ 契约测试：`ok=False` 的 tool_result 事件**必须带 `output`**。

    它是路由层 F7 探测的**唯一数据来源** —— 少了它，探测会**静默失效**
    （不报错、只是用户看到错误的失败原因）。
    """
    loop = _loop_with_failing_tool(RetrievalUnavailable("x"))

    events = asyncio.run(_collect(loop))
    trs = [e for e in events if e.type == "tool_result"]
    assert trs, "没有 tool_result 事件"
    for e in trs:
        assert e.data.get("ok") is False
        assert "output" in e.data, f"失败事件丢了 output ⇒ F7 探测会变死代码：{e.data}"
        assert isinstance(e.data["output"], dict)
        assert e.data["output"].get("error") == "retrieval_unavailable"


async def _collect(loop: AgentLoop) -> list:
    return [ev async for ev in loop.run_stream("测试问题")]


def test_run_rebuilds_history_with_failure_output():
    """★ 两个 API 一致：`run()`（CLI）重建的 history 里，失败工具不得退化成 `RESULT: None`。

    `run_stream` 本来就把 output 注入 messages（ok/not-ok 一视同仁）；
    `run()` 是**从事件**重建 history 的 ⇒ 事件丢了 output，它就与前者不一致
    （同一次失败，两个 API 给模型的上下文不同）。
    """
    loop = _loop_with_failing_tool(RetrievalUnavailable("x"))

    _final, history = asyncio.run(loop.run("测试问题"))

    tool_msgs = [m for m in history if getattr(m.role, "value", m.role) == "tool"]
    assert tool_msgs, f"history 里没有 tool 消息：{[m.role for m in history]}"
    for m in tool_msgs:
        assert "None" not in m.content, f"失败工具在 history 里退化成 RESULT: None：{m.content}"
        assert "retrieval_unavailable" in m.content, m.content


# --------------------------------------- ② 用户可见：notice 真的出现


def test_retrieval_fault_reaches_the_user_as_retrieval_error_notice(client, monkeypatch):
    """★ 核心：检索故障 → 用户看到「检索服务暂时不可用」而不是通用的「可能是模型调用失败」。"""
    sid = _prepare(client, monkeypatch, _BoomPipe())

    body = _ask(client, sid)

    assert "retrieval_error" in body, (
        "SSE 里没有 retrieval_error notice ⇒ 用户被引导去查余额/换模型，"
        f"而真正该做的是重建索引。SSE 片段：{body[-600:]}"
    )
    assert "重建索引" in body, "F7 文案必须给出可操作建议（重建索引）"

    # 落库留痕：刷新后警示卡仍在
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    last = [m for m in detail["messages"] if m["role"] == "assistant"][-1]
    notice = (last.get("meta") or {}).get("notice") or {}
    assert notice.get("kind") == "retrieval_error", last.get("meta")


def test_fault_marker_is_present_on_the_event(client, monkeypatch):
    """同一件事的另一半证据：事件里确实带上了 `retrieval_unavailable`（探测的原料）。"""
    sid = _prepare(client, monkeypatch, _BoomPipe())

    _ask(client, sid)

    outs = [(e.get("data") or {}).get("output") for e in _events(client, sid)
            if e.get("type") == "tool_result" and (e.get("data") or {}).get("ok") is False]
    assert outs, "没有失败的 tool_result 事件"
    assert any(isinstance(o, dict) and o.get("error") == "retrieval_unavailable" for o in outs), outs


# --------------------------------------- ③ 不得误伤


def test_empty_hit_does_not_claim_retrieval_fault(client, monkeypatch):
    """★ 反向自检：**检索成功但 0 命中**不得被报成"检索服务暂时不可用"。

    两者对用户的意义**完全相反**：
    - 空命中 ⇒「文档里可能真没有」→ 该换问法（**不要**去重建索引，白折腾且可能丢索引）；
    - 检索故障 ⇒「检索链路坏了」→ 该重建索引。

    混为一谈会让用户在**索引完全正常**时被引导去重建索引。
    （这条与 C3 的 `retrieval_empty` 是同一枚硬币的两面：统计上分开，提示也要分开。）

    ⚠️ 为什么不用"普通工具失败（RuntimeError）"来构造这条：
    `CompositeRAGPipeline.query` 会把**任何**单篇管线异常统一转成
    `RetrievalUnavailable`（"全部 1/1 个检索管线均失败"）—— 见
    `chat_service.py` 的全失败分支。也就是说在检索路径上，
    "非 RetrievalUnavailable 的管线失败"**不可能出现**（读码 + 实测双重确认）。
    """

    class _Empty:
        def get_chunks(self):  # noqa: ANN201
            return [RetrievedChunk(id="doc-9-c0", text="正文", page=1, score=0.5)]

        def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
            return []

    sid = _prepare(client, monkeypatch, _Empty())

    body = _ask(client, sid)

    assert "retrieval_error" not in body, "空命中被误报成检索故障（会把用户引向重建索引）"
