"""技能路径**必须逐块流式**返回（2026-09-28）。

### 背景：同一产品里的两种观感
- 问答路径（`AgentLoop` / 直答）：逐 `message_delta` ⇒ 用户看到「逐字出现」；
- 技能路径（`/总结` `/对比` `/实验设计` …）：技能内部走 `llm.chat()`（一次性拿全文），
  路由层再 `yield sse_event("message", ...)` 一次性发完 ⇒ 用户看到「整段突然刷出」。

### 修法
**不改 12 个技能的任何代码**，而是在路由层把 `ctx.llm` 换成一个代理
（`_StreamingChatProxy`）：它的 `chat()` 内部改走底层 `stream()`，
每个增量推入队列、由路由层逐块 `yield sse_event("message_delta", ...)`；
技能结束后的最终 `message` 事件照旧（承载 citations 与拼好的最终文本）。

### 本文件测的是「接线」而不是「零件」
零件（代理转换 / 增量上报 / 异常传播 / 死锁防护）由
`test_skill_path_no_deadlock.py` 的单元层覆盖。这里补的是**只有端到端才看得见**的那一层：
事件真的从 HTTP 流里逐块出来了。
⇒ 若哪天有人把 `async for ... yield sse_event("message_delta")` 改回一次性 `message`，
本文件会红。

### 为什么用假技能 + monkeypatch，而不是真技能
真技能（如 `/实验设计`）在**无文档**会话下不会调用 LLM（走确定性回退），
拿不到任何增量 ⇒ 测不出流式。这里用假技能直接调 `ctx.llm.chat()`，
把「有没有逐块下发」从「技能有没有 LLM 输出」里解耦出来。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.llm.base import ChatMessage, Role
from lightcore.skills.spec import SkillResult

#: FakeLLM 会按此切分（>8 字且带句末标点 ⇒ 必然切成多块），用来验证"逐块"而非"一次"。
_REPLY_TEXT = "第一句验证增量。第二句验证顺序。第三句验证结尾。"


class _EchoSkill:
    """最小技能：直接调 `ctx.llm.chat()` 并把结果作为回答返回（模拟真实技能的做法）。"""

    name = "echo"

    async def run(self, ctx) -> SkillResult:  # noqa: ANN001
        resp = await ctx.llm.chat([
            ChatMessage(role=Role.SYSTEM, content="system"),
            ChatMessage(role=Role.USER, content="user"),
        ])
        return SkillResult(ok=True, content=resp.message.content or "", citations=[])


class _StubRouter:
    """始终命中 `_EchoSkill` —— 把测试与真实技能清单解耦（新增技能也不会影响本文件）。"""

    def __init__(self, skill: _EchoSkill) -> None:
        self._skill = skill

    def route(self, content: str, feats) -> _EchoSkill:  # noqa: ANN001
        return self._skill

    def parse_args(self, content: str, skill) -> list:  # noqa: ANN001
        return []


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    """把 SSE 文本拆成 (event, data) 序列。"""
    out: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        name = None
        data: dict = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    data = {}
        if name:
            out.append((name, data))
    return out


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_skill_stream_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    rs_mod._state = None
    cfg_mod._settings = None


def _patch(monkeypatch, *, real_llm: bool = True, skill=None) -> None:  # noqa: ANN001
    """让 app 走「技能路径 + 代理生效」的组合。

    - `ApiFakeLLM._decide`：**app 实际用的是这个子类**（它覆盖了基类的 `_decide`），
      默认剧本返回 `tool_calls`（那样 `stream()` 只吐工具调用、没有 delta），
      这里改成纯文本，才能验证增量；
    - `is_real_llm` → True：让路由层真的把 `ctx.llm` 包上流式代理；
    - `default_router` → 固定命中假技能（`skill=` 可换成别的假技能，如带引用的）。
    """
    import papermate.skills as skills_pkg
    import papermate.skills._common as common_mod
    from papermate.services.chat_service import ApiFakeLLM

    monkeypatch.setattr(ApiFakeLLM, "_decide", lambda self, last, msgs: ChatMessage(  # noqa: ARG005
        role=Role.ASSISTANT, content=_REPLY_TEXT,
    ))
    monkeypatch.setattr(common_mod, "is_real_llm", lambda _llm: real_llm)
    monkeypatch.setattr(skills_pkg, "default_router", lambda: _StubRouter(skill or _EchoSkill()))


def _new_session(client, title: str) -> int:  # noqa: ANN001
    """建会话并**关联一篇文档**。

    ⚠️ 命令路径有前置检查：`effective_doc_ids` 为空时**直接返回提示、根本不执行技能**
    （"当前会话尚未关联任何文档，命令类操作…需要先关联文档才能执行"）。
    这正是上一轮集成层测试**假通过**的原因 —— 不关联文档就永远走不到技能里。
    文档 id 不存在也无妨：假技能不做检索。
    `config` 是**合并**语义（`session_service.update`），不会覆盖 `model` 等其他键。
    """
    sid = client.post("/api/v1/sessions", json={"title": title}).json()["id"]
    client.patch(f"/api/v1/sessions/{sid}", json={"config": {"doc_ids": [1]}})
    return sid


def _stream(client, sid: int) -> list[tuple[str, dict]]:  # noqa: ANN001
    resp = client.post(
        f"/api/v1/sessions/{sid}/messages/stream",
        json={"content": "/echo", "stream_id": "skill-stream"},
    )
    assert resp.status_code == 200, resp.text
    return _parse_sse(resp.text)


def test_skill_path_emits_message_delta_chunks(client, monkeypatch):
    """技能路径必须产生**多个** `message_delta`（证明是逐块下发，而非一次性整段）。"""
    _patch(monkeypatch)
    sid = _new_session(client, "流式")
    events = _stream(client, sid)

    deltas = [d.get("text", "") for name, d in events if name == "message_delta"]
    assert len(deltas) >= 2, f"应逐块下发，实际只有 {len(deltas)} 块：{events}"
    assert "".join(deltas) == _REPLY_TEXT


def test_skill_path_final_message_matches_streamed_text(client, monkeypatch):
    """最终 `message` 事件必须与流式拼接结果一致（并承载 citations）。

    这是「流式展示」与「落库/定稿」的一致性契约：若两者能对不上，
    用户就会看到"流出来的内容"与"最后定稿的内容"不同。
    """
    _patch(monkeypatch)
    sid = _new_session(client, "定稿")
    events = _stream(client, sid)

    deltas = [d.get("text", "") for name, d in events if name == "message_delta"]
    finals = [d for name, d in events if name == "message"]
    assert finals, f"缺少最终 message 事件：{events}"
    final_text = finals[-1].get("text", "")
    streamed = "".join(deltas)

    # 无文档会话会在定稿文本前拼「⚠️ 警示」前缀（既有行为）⇒ 用包含关系而非相等
    assert streamed and streamed in final_text, f"流式内容 {streamed!r} 未出现在定稿 {final_text!r} 中"
    assert "citations" in finals[-1]
    # 事件顺序：增量必须在定稿之前（否则前端会先看到完整文本、再被追加一遍）
    names = [n for n, _ in events]
    assert names.index("message_delta") < names.index("message")


def test_fake_llm_path_stays_unstreamed_and_consistent(client, monkeypatch):
    """FakeLLM 不被代理包装时（`is_real_llm` 为假）仍须给出**完整**回答。

    这是"不包装 Fake"的契约：代理只在真实 LLM 上生效，
    离线/演示模式下行为与改造前完全一致（不得因为缺少增量而丢内容）。
    """
    _patch(monkeypatch, real_llm=False)
    sid = _new_session(client, "Fake")
    events = _stream(client, sid)

    finals = [d for name, d in events if name == "message"]
    assert finals, f"缺少最终 message 事件：{events}"
    assert _REPLY_TEXT in finals[-1].get("text", "")


def test_skill_path_persists_citations(client, monkeypatch):
    """★ B2-X1（2026-10-01）：技能路径的引用必须落库（否则用户"刷新即消失"）。

    修复前：技能路径**有** `citations` 变量（`result.citations`，下游 `_run_verify` 还在用）
    却**没传给 `persist_turn`** ⇒ 引用从未落库 ⇒ 刷新后引用卡片消失。
    """
    cite = {"id": "doc1-c0", "page": 1, "doc_id": 1, "text": "技能引用的证据段落，长度须超过清洗下限十六个字。"}

    class _CiteSkill:
        """返回**非空 citations** 的最小技能。"""

        name = "cite"

        async def run(self, ctx) -> SkillResult:  # noqa: ANN001
            return SkillResult(ok=True, content="带引用的回答", citations=[cite])

    _patch(monkeypatch, skill=_CiteSkill())
    sid = _new_session(client, "技能引用")
    _stream(client, sid)

    # ① 历史接口回传 ⇒ 前端刷新/切换会话后能回填引用卡片
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assistant = [m for m in detail["messages"] if m["role"] == "assistant"]
    assert assistant, detail["messages"]
    assert assistant[-1]["citations"] == [cite], "技能轮次的引用未随历史返回 ⇒ 刷新后仍不显示"

    # ② trace 侧：**如实记录的边界**，不是缺陷，也不算本次目标。
    #    技能分支的 `events` 收集器始终为空（技能输出直接走 SSE，不进 events）⇒ trace 里没有
    #    "完整回答事件" ⇒ 引用无处可挂，`persist_turn` 如实不注入。
    #    推论：`citation_rate`（样本 = 完整回答事件）**本就不计技能轮** ⇒ 本改动不扭曲该指标。
    #    （要让技能轮进指标需另立任务：把技能轮事件也收进 events。）
    SL = client.app.state.chat_svc._Session
    with SL() as db:
        from papermate.models.db import Trace

        t = db.query(Trace).order_by(Trace.id.desc()).first()
        events = json.loads(t.events_json or "[]")
    assert all("citations" not in (e.get("data") or {}) for e in events), (
        "技能路径不得为挂引用而**伪造**事件（宁可如实为空，也不要造一个假的'完整回答'）"
    )


def test_skill_path_citations_go_through_the_single_cleaner(client, monkeypatch):
    """★ B2-X2（2026-10-01）：技能引用必须经**唯一清洗入口**（此前完全绕过）。

    修复前：技能侧 `citations_from` 只做 `normalize_page`（长度/噪声/去重全无），
    `literature_review` / `related_work_gen` 更是直接塞**整块 chunk 文本** ⇒
    超长片段进前端后被静默丢弃，而落库与 `citation_rate` 仍计入。
    真实库实测：658 个 chunk 里 **58.2% 超过 600 字符**（p50 921 / max 2748）⇒ 不是理论风险。
    """
    long_text = "论文正文内容。" * 300  # 2100 字，远超单条上限
    cite = {"id": "doc1-c0", "page": 1, "doc_id": 1, "text": long_text}
    noisy = {"id": "doc1-c1", "page": 2, "doc_id": 1, "text": "收稿日期：2025-01-01　通讯作者：张三"}

    class _LongCiteSkill:
        """返回"一条超长 + 一条噪声"的技能：验证清洗真的作用在技能出口。"""

        name = "longcite"

        async def run(self, ctx) -> SkillResult:  # noqa: ANN001
            return SkillResult(ok=True, content="带长引用的回答", citations=[cite, noisy])

    _patch(monkeypatch, skill=_LongCiteSkill())
    sid = _new_session(client, "技能长引用")
    _stream(client, sid)

    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assistant = [m for m in detail["messages"] if m["role"] == "assistant"]
    assert assistant, detail["messages"]
    cits = assistant[-1]["citations"]

    assert len(cits) == 1, f"超长引用被整条丢弃、或噪声未被过滤：{cits}"
    t = cits[0]["text"]
    assert t.endswith("…"), f"技能引用未被截断（仍是 {len(t)} 字原文）"
    assert len(t) <= 600, f"截断后仍 {len(t)} 字"
    assert "收稿日期" not in t, "噪声片段未被过滤 ⇒ 技能路径仍未走唯一清洗入口"


def test_skill_path_emits_paired_tool_result(client, monkeypatch):
    """★★ O-02 回归：技能路径的 `tool_call` 必须有**配对**的 `tool_result`。

    修复前：只发 `tool_call`（全文件唯一一处）、**零处** `tool_result` ⇒ 前端 reducer
    只有 `tool_result` 才会把卡片从「执行中」收敛（`done` 分支**不处理** tool 状态）
    ⇒ 每次使用技能都留一张**永久转圈**的"执行中"卡片（必现）。
    """
    _patch(monkeypatch)
    sid = _new_session(client, "配对")
    events = _stream(client, sid)

    calls = [d for name, d in events if name == "tool_call"]
    results = [d for name, d in events if name == "tool_result"]
    assert calls, events
    assert results, "技能路径没有 tool_result ⇒ 前端卡片永久悬停（O-02 回归）"
    # call_id 必须严格配对（前端靠它匹配；错位 ⇒ 又悬停）
    assert [d["call_id"] for d in calls] == [d["call_id"] for d in results]
    assert results[0]["ok"] is True
    # tool_result 必须先于 done（前端靠它收敛后再收尾）
    names = [name for name, _ in events]
    assert names.index("tool_result") < names.index("done")


def test_skill_path_tool_result_carries_error_on_failure(client, monkeypatch):
    """失败分支同样要发配对 `tool_result`（`ok=False` + 可读错误），否则照样悬停。"""

    class _FailSkill:
        name = "fail"

        async def run(self, ctx) -> SkillResult:  # noqa: ANN001
            return SkillResult(ok=False, content="", error="请先关联文档")

    _patch(monkeypatch, skill=_FailSkill())
    sid = _new_session(client, "失败配对")
    events = _stream(client, sid)

    results = [d for name, d in events if name == "tool_result"]
    assert results, "失败分支没有 tool_result ⇒ 卡片悬停（O-02 回归）"
    assert results[0]["ok"] is False
    assert "请先关联文档" in (results[0].get("error") or "")
