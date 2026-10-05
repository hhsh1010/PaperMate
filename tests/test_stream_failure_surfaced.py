"""P1-11 回归（2026-09-15）：流式生成**中途异常**必须显式告知失败。

原缺陷分两层（均已核实）：

1. **Starlette 1.6.0 的设计限制**：`ServerErrorMiddleware.__call__` 只在
   `not response_started` 时才发送异常处理器构造的响应 —— SSE 已 flush 首帧后，
   全局异常处理器**发不出** JSON 500（读 `starlette/middleware/errors.py` 确认）。
   → 流式端点必须**自己**兜底，不能指望全局处理器。
2. **两条 pump 只捕获 `CancelledError`**：其它异常触发 `finally: await _q.put(None)`，
   消费端把 `None` 当作「流正常结束」→ 有部分输出时既无 notice 也无 error 事件，
   **用户看到半截回答却以为已完成**（静默失败）；异常本身也从未被 `await` 取回
   （清理处只有 `if not pump.done(): pump.cancel()`），仅在 GC 时打一行
   "Task exception was never retrieved"。

本文件从两个角度锁住修复：

- **行为级**（直答路径可稳定复现）：错误被转成 `event: error`、**不再发 `event: done`**、
  已生成部分与用户提问仍落库且带 `meta.notice`（刷新后警示卡不丢）。
- **结构级**（Agent 路径难以构造确定性触发条件）：用 AST 判定两条 pump 都接了
  `except Exception` 分支，且消费端优先判定 `_StreamFailure`。
"""

from __future__ import annotations

import ast as _ast
import inspect
import json

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.llm.base import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    Role,
    ToolCall,
)


class _BoomLLM(BaseLLM):
    """产出 `deltas` 个增量后抛错，模拟「生成中途失败」。

    注意**不能继承 FakeLLM**：直答路径对 FakeLLM 走剧本文案分支，根本不会调用 `stream`
    （同 `tests/test_api.py::_SlowLLM` 的注释）。
    """

    def __init__(self, deltas: list[str] | None = None) -> None:
        super().__init__()
        self.deltas = deltas if deltas is not None else ["这是已经生成的部分"]
        self.stream_called = False

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="部分内容"))

    async def stream(self, messages, tools=None, **kw):  # noqa: ANN001, ANN003
        self.stream_called = True
        for d in self.deltas:
            yield LLMStreamChunk(delta=d)
        raise RuntimeError("simulated mid-stream failure")


@pytest.fixture
def client():
    import os
    import tempfile

    tmp = tempfile.mkdtemp(prefix="pm_p1_11_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _install_boom(client, monkeypatch, boom: _BoomLLM) -> None:
    """把该 app 的 LLM 换成 `boom`（沿用 `tests/test_api.py` 的注入方式）。"""
    real_build = client.app.state.chat_svc.build_agent

    async def _patched(session_id, cancel_event=None, restrict_doc_ids=None):
        agent = await real_build(
            session_id, cancel_event=cancel_event, restrict_doc_ids=restrict_doc_ids
        )
        agent.llm = boom
        return agent

    monkeypatch.setattr(client.app.state.chat_svc, "build_agent", _patched)


def _new_session(client, title: str) -> int:
    return client.post("/api/v1/sessions", json={"title": title}).json()["id"]


def _assistant_messages(client, sid: int) -> list[dict]:
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    return [m for m in detail["messages"] if m["role"] == "assistant"]


# ---------- 行为级 ----------


def test_mid_stream_failure_is_surfaced_not_silent(client, monkeypatch):
    """首个 delta 之后失败：必须发 `error` 事件、**不得发 `done`**，且内容与警示卡落库。"""
    boom = _BoomLLM(["这是已经生成的部分"])
    _install_boom(client, monkeypatch, boom)
    # 「你好」→ chitchat → 直答路径（不走 Agent 工具循环），行为确定
    sid = _new_session(client, "stream-failure")

    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    body = r.text

    assert boom.stream_called, "直答路径应真的调用 LLM stream（否则本用例没测到目标分支）"
    assert "event: error" in body, "中途失败必须发 error 事件（否则用户只看到半截回答）"
    assert "stream_failed" in body, "error 事件应带机器可读 code，供前端分支与埋点"
    assert "event: done" not in body, "失败轮次不得发 done —— 否则前端会把残缺回答当成功完成"
    # 已生成的部分照常推给前端（用户至少看到已有内容）
    assert "这是已经生成的部分" in body

    # 落库：用户提问与已生成部分都要保住（N13 口径：关键数据优先落库）
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert any(m["role"] == "user" for m in detail["messages"]), "用户提问必须落库"
    assistant = _assistant_messages(client, sid)
    assert assistant, "失败轮次也要落 assistant 记录（否则历史里只剩提问、看不出发生了什么）"
    last = assistant[-1]
    assert "这是已经生成的部分" in last["content"]
    notice = (last.get("meta") or {}).get("notice") or {}
    assert notice.get("kind") == "stream_failed", "落库须带 notice，刷新后警示卡才不丢"
    assert last["meta"].get("invalid") is False, "有部分输出不算「无效回复」，不参与 loop_risk 计数"


def test_mid_stream_failure_without_output_persists_fallback(client, monkeypatch):
    """任何 delta 都没有就失败：历史里仍要有**可读的兜底文案**（不留空白）+ 警示卡。"""
    boom = _BoomLLM([])  # 直接抛错
    _install_boom(client, monkeypatch, boom)
    sid = _new_session(client, "stream-failure-empty")

    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    body = r.text

    assert "event: error" in body
    assert "event: done" not in body

    assistant = _assistant_messages(client, sid)
    assert assistant, "无输出时也要落库（否则用户以为提问丢了）"
    last = assistant[-1]
    assert last["content"].strip(), "不得落空白内容 —— 历史里要能看到「本次未生成有效回答」"
    assert (last.get("meta") or {}).get("notice", {}).get("kind") == "stream_failed"
    assert last["meta"].get("invalid") is True, "完全无输出应计入「无效回复」（参与 loop 风险计数）"


def test_success_path_still_emits_done(client):
    """对照用例：正常路径不得被本次改动影响（仍发 done、不发 error）。"""
    sid = _new_session(client, "stream-ok")
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    assert "event: done" in r.text
    assert "event: error" not in r.text


# ---------- 结构级 ----------


def test_both_pumps_guard_non_cancel_exceptions():
    """两条 pump 都必须有 `except Exception` 分支，且消费端优先判定 `_StreamFailure`。

    用 AST 判定（而非字符串包含）——注释里同样会出现 `except Exception` 字样，
    字符串匹配会把"只写在注释里的修复"误判为已修复。
    """
    from papermate.api import routes_search as rs

    tree = _ast.parse(inspect.getsource(rs))
    pumps = {
        node.name: node
        for node in _ast.walk(tree)
        if isinstance(node, _ast.AsyncFunctionDef) and node.name in {"_pump", "_pump_direct"}
    }
    assert set(pumps) == {"_pump", "_pump_direct"}, f"未找到两条 pump：{sorted(pumps)}"

    for name, fn in pumps.items():
        handled: set[str] = set()
        for node in _ast.walk(fn):
            if isinstance(node, _ast.Try):
                for h in node.handlers:
                    if h.type is None:
                        handled.add("bare")
                    elif isinstance(h.type, _ast.Name):
                        handled.add(h.type.id)          # except Exception:
                    elif isinstance(h.type, _ast.Attribute):
                        handled.add(h.type.attr)        # except asyncio.CancelledError:
        assert "Exception" in handled, f"{name} 未捕获 Exception —— 中途失败会被 finally 的哨兵掩盖"
        assert "CancelledError" in handled, f"{name} 必须保留取消语义（CancelledError 需原样抛出）"

    src = inspect.getsource(rs)
    assert "class _StreamFailure" in src, "失败哨兵类型必须存在（与「客户端断开」语义区分）"
    assert "isinstance(item, _StreamFailure)" in src, "Agent 路径消费端必须判定失败哨兵"
    assert "isinstance(ev, _StreamFailure)" in src, "直答路径消费端必须判定失败哨兵"
    # 判定顺序：失败哨兵先于 `None`（pump 的 finally 会把 None 也排进队列）
    assert src.index("isinstance(ev, _StreamFailure)") < src.index(
        "if ev is None:"
    ), "必须先判失败哨兵再判 None，否则异常又被当成「流正常结束」"


def test_error_reason_is_surfaced_to_notice_and_event(client, monkeypatch):
    """2026-09-16：失败原因必须**透给用户** —— 事件与落库 notice 都带具体原因。

    背景：用户连续失败却只看到「未生成有效回答」，无法区分"该充值（402 余额）"
    还是"该调预算（token 熔断）"，完全无法自救（见当日对话失败诊断）。
    """
    class _ReasonBoom(_BoomLLM):
        async def stream(self, messages, tools=None, **kw):  # noqa: ANN001, ANN003
            self.stream_called = True
            # 必须是 async generator（调用方 `async for` 消费）—— 普通 async def 会得到
            # coroutine，报 "'async for' requires an object with __aiter__"（本用例踩过）。
            yield LLMStreamChunk(delta="")
            raise RuntimeError("API 账户余额不足（Insufficient Balance）：请充值后重试")

    boom = _ReasonBoom()
    _install_boom(client, monkeypatch, boom)
    sid = _new_session(client, "stream-failure-reason")

    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    body = r.text

    # ① error 事件带具体原因（含"余额"关键词，用户能看懂该去充值）
    assert "Insufficient Balance" in body, "error 事件应携带具体失败原因"
    # ② 落库 notice 的 message 带原因 + suggestion 给出**可操作**建议
    last = _assistant_messages(client, sid)[-1]
    notice = (last.get("meta") or {}).get("notice") or {}
    assert "Insufficient Balance" in notice.get("message", ""), "刷新后警示卡仍要能看到原因"
    assert "充值" in notice.get("suggestion", ""), "建议必须可操作（充值/换模型），不能只说重试"
    # ③ 防泄漏：超长原因被截断（异常原文可能含厂商响应体）
    assert len(notice["message"]) < 400


# ---------- C2（2026-09-28）：Agent 中止路径的 done 与用量口径 ----------


class _AlwaysToolStreamLLM:
    """流式**永远只产一次工具调用** → 迭代耗尽 → 走 `_abort_with_fallback`。

    用**固定 usage**（每次调用 pt/ct 恒定），使"本轮累计"可以被精确预测 ——
    这正是 C2 断言的支点：中止轮的用量必须是**本轮**的真实值，而不是整段会话累计。
    """

    model_name = "fake"
    thinking_mode = False

    def __init__(self, pt: int = 1000, ct: int = 10) -> None:
        self.pt = pt
        self.ct = ct
        self.n = 0  # stream 调用次数（= 本轮 LLM 调用次数）

    async def stream(self, messages, *args, **kw):  # noqa: ANN001, ANN002, ANN003
        self.n += 1
        yield LLMStreamChunk(
            tool_calls=[ToolCall(id=f"c{self.n}", name="retrieve",
                                 arguments={"query": "量子纠缠"})],
            finished=True, prompt_tokens=self.pt, completion_tokens=self.ct,
            latency_ms=5, cost=0.0,
        )

    async def chat(self, messages, *args, **kw):  # noqa: ANN001, ANN002, ANN003
        """兜底收尾（`_finalize_without_tools`）走非流式：必须给一段可读文案。"""
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="基于现有证据的兜底回答"),
            prompt_tokens=self.pt, completion_tokens=self.ct,
            model=self.model_name, latency_ms=5, cost=0.0,
        )


def _sse_named(text: str) -> list[tuple[str, dict]]:
    """解析 `event:` / `data:` 配对（保留事件名，供断言**顺序**）。"""
    out: list[tuple[str, dict]] = []
    name: str | None = None
    for ln in text.splitlines():
        if ln.startswith("event:"):
            name = ln.split(":", 1)[1].strip()
        elif ln.startswith("data:") and name:
            try:
                out.append((name, json.loads(ln[5:].strip())))
            except json.JSONDecodeError:
                pass
    return out


def test_agent_abort_path_emits_done_with_this_turn_usage(client, monkeypatch):
    """★ C2 核心：**中止轮**必须发 done，且用量三处同源（SSE == 落库 == 本轮真值）。

    修复前的表现：loop 只有"正常收敛"分支发 done；中止轮不发 ⇒ 路由层落进
    "未收到 done"兜底分支，把用量换成 `count_usage()`（**整段会话累计**），
    而落库走的是**估算** ⇒ "用户看到的 / 落库的 / 日志里的"三处互不相同，
    而"失败轮次花了多少钱"恰恰是运维最需要准的数字。
    另：中止原因此前只存在于即时 error 事件里，**刷新后历史里查不到**（notice 恒 None）。
    """
    import papermate.services.chat_service as cs

    async def _noop_reflect(self, session_id, llm, conversation):  # noqa: ANN001
        return None

    async def _noop_title(self, session_id, llm, text):  # noqa: ANN001
        return None

    monkeypatch.setattr(cs.ChatService, "reflect", _noop_reflect)
    monkeypatch.setattr(cs.ChatService, "auto_title", _noop_title)

    llm = _AlwaysToolStreamLLM(pt=1000, ct=10)
    _install_boom(client, monkeypatch, llm)

    sid = _new_session(client, "agent-abort")
    # doc-bound 才会走 Agent 循环（否则被路由到直答路径 —— 会变成"假通过"）
    files = {"file": ("doc.md", "量子纠缠是非局域关联现象。对其中一个粒子测量会决定另一个的状态。".encode(),
                      "text/markdown")}
    assert client.post(f"/api/v1/documents?session_id={sid}", files=files).status_code == 200
    client.patch(f"/api/v1/sessions/{sid}", json={"config": {"model": "fake"}})

    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "文档里量子纠缠是什么？"})
    assert r.status_code == 200, r.text
    pairs = _sse_named(r.text)
    names = [n for n, _ in pairs]

    # 前置条件：必须真的走了 Agent 循环并调用过工具（否则本用例测不到目标分支）
    assert "tool_call" in names, f"没走 Agent 路径 ⇒ 本用例无效：{names}"
    assert llm.n > 1, "只调用了一次 ⇒ 没跑到迭代耗尽"

    # ① 中止轮**必须有 done**
    assert "done" in names, f"中止轮缺 done（C2 的缺陷本体）：{names}"
    assert "error" in names, "中止原因仍要即时告知"
    # ② 路由层对 done 有**保序**设计：它在 message/verify 之后**最后**统一发出
    #    （error 反而是即时透传的 ⇒ SSE 里 error 在 done 之前，这是既有的、正确的水位）
    assert names[-1] == "done", f"done 必须是最后一个事件（保序契约）：{names[-6:]}"
    assert "message" in names, "中止轮仍要产出可读回答（「耗尽 ≠ 无回答」）"

    done = next(d for n, d in pairs if n == "done")
    assert done.get("aborted") is True, "中止轮必须自带标记"
    # 取证留痕（反向自检时实测）：把 loop 的 done 拆掉后，本断言报出的是
    # `total_tokens: 14`（兜底分支的会话累计/估算），而真实值是 **12,120**（12×1010）
    # —— 差三个数量级。这就是"用量三处互不相同"在数字上的样子。
    assert done.get("failure_kind") == "max_iterations", (
        "C1 交叉验证：本轮反复取到同一批证据（无进展判据会记日志），"
        "但**不得**成为中止原因 —— 观察期该判据没有中止能力"
    )
    # ③ 用量 = **本轮**真实值（每次 1000/10，调用 llm.n 次）
    assert done["steps"] == llm.n
    # ⚠️ A7（2026-09-28）**修正了本断言**：`llm.n` 只统计 `stream()` 调用，
    # 而兜底收尾（`_finalize_without_tools`）走 **`chat()`** ⇒ **不计入 `n`**。
    # ⇒ 本轮真实调用是 `n + 1` 次。此前 `done` 只报主调用（n 次），
    #   把收尾那次**漏掉了** —— 那正是 A7 所说的"统计缺口"（用量页/落库系统性偏低）。
    # 现在 `done` 报**完整**值，并用 `_hidden_calls` 显式暴露隐藏部分，便于核对。
    assert done["_hidden_calls"] == 1, "兜底收尾 = 1 次隐藏调用（A7 统计同源）"
    assert done["prompt_tokens"] == (llm.n + done["_hidden_calls"]) * 1000
    assert done["completion_tokens"] == (llm.n + done["_hidden_calls"]) * 10
    # ★ 熔断判据**不受影响**：隐藏调用只进统计、不进 `max_cost_per_turn` 的判据
    # （为什么：隐藏调用多发生在收尾时刻，此刻中止损失的是最终回答 —— A7 影响面评估）。
    assert done["prompt_tokens"] > llm.n * 1000, "隐藏用量必须被加进统计（否则缺口依旧）"

    # ④ 三处口径统一：SSE done == 落库（C2 的核心断言）
    usage = client.get(f"/api/v1/usage/sessions/{sid}").json()
    last = [m for m in usage["messages"] if m["role"] == "assistant"][-1]
    assert last["prompt_tokens"] == done["prompt_tokens"], (
        "落库用量与 SSE done 不同源（修复前落库是估算、SSE 是会话累计）"
    )
    assert last["completion_tokens"] == done["completion_tokens"]

    # ⑤ 中止原因**落库**为结构化 notice（刷新后警示卡仍在）
    #    注：notice 要从会话详情读（`/usage/...` 只回用量字段，不含 meta）。
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    last_assistant = [m for m in detail["messages"] if m["role"] == "assistant"][-1]
    notice = (last_assistant.get("meta") or {}).get("notice") or {}
    assert notice.get("kind") == "stream_failed", (
        f"中止轮 notice 不得为 None（刷新后原因就查不到了）：{last_assistant.get('meta')}"
    )
    assert "最大迭代" in notice.get("message", ""), notice
