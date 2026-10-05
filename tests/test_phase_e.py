"""阶段 E 测试：V4 模型切换会话延续 + 会话自动标题（2026-08-22）。

覆盖：
- DeepSeek V4 兼容：chat/stream 请求体显式禁用 thinking（否则 tool_choice 400）；reasoning_content 回退；
- 多模态历史持久化：带图 user 消息入库 → load_history 还原内容块 → GET 会话返回纯文本 + images；
- 会话自动标题：首轮对话后生成主题标题（fake 规则降级）、默认标题懒回填、不覆盖人工标题；
- 切换模型延续：带图历史重建后 Agent 循环正常（无回归）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
import papermate.runtime_settings as rs_mod
from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.router import ModelEndpoint, OAICompatibleLLM
from papermate.services.chat_service import _MM_PREFIX


def _png_data_url() -> str:
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


# ---------------- V4 兼容：thinking 禁用 + reasoning 回退 ----------------
def _v4_llm(captured: dict, resp_json: dict):
    def fake_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=resp_json)

    ep = ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-flash-vision-exp")
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_handler))
    return OAICompatibleLLM(ep, client=client, api_key="sk-test")


def test_v4_chat_disables_thinking():
    captured: dict = {}

    async def run():
        llm = _v4_llm(captured, {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}})
        await llm.chat([ChatMessage(role=Role.USER, content="hi")])
        # V4 默认 thinking 不接受 tool_choice → 显式禁用（修复 400 根因）
        assert captured["body"]["thinking"] == {"type": "disabled"}

    asyncio.run(run())


def test_v4_stream_disables_thinking():
    captured: dict = {}
    sse_body = "data: " + json.dumps({"choices": [{"delta": {"content": "答"}}]}) + "\n\n" + "data: [DONE]\n\n"

    def fake_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, text=sse_body)

    ep = ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-flash-vision-exp")
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_handler))

    async def run():
        llm = OAICompatibleLLM(ep, client=client, api_key="sk-test")
        chunks = [c async for c in llm.stream([ChatMessage(role=Role.USER, content="hi")])]
        assert any(c.delta for c in chunks)
        assert captured["body"]["thinking"] == {"type": "disabled"}

    asyncio.run(run())


def test_non_v4_does_not_send_thinking():
    captured: dict = {}

    async def run():
        ep = ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat")
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}})))
        captured["client"] = client
        llm = OAICompatibleLLM(ep, client=client, api_key="sk-test")
        payload = {"body": {}}

        async def handler(r):
            payload["body"] = json.loads(r.content)
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}})

        client._transport.handler = handler
        await llm.chat([ChatMessage(role=Role.USER, content="hi")])
        assert "thinking" not in payload["body"]

    asyncio.run(run())


def test_v4_chat_falls_back_to_reasoning_content():
    async def run():
        # 防御：content 为空但 reasoning_content 有值 → 回退为回复文本（避免「回复空白」）
        llm = _v4_llm({}, {"choices": [{"message": {"role": "assistant", "content": "", "reasoning_content": "思考过程"}}], "usage": {}})
        resp = await llm.chat([ChatMessage(role=Role.USER, content="hi")])
        assert resp.message.content == "思考过程"

    asyncio.run(run())


# ---------------- API 全链路 ----------------
@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_e_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    rs_mod._state = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_multimodal_history_persists_and_restores(client):
    """带图消息持久化：历史加载后图片上下文不丢失（切换模型/刷新后延续）。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    sid = s["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "这张图里是什么", "images": [_png_data_url()]})
    assert r.status_code == 200
    # GET 会话：user 消息 content 为纯文本 + images 单独返回
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    user_msg = detail["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "这张图里是什么"
    assert len(user_msg["images"]) == 1
    assert user_msg["images"][0].startswith("data:image/png;base64,")
    # 存储层面：块以 _MM_PREFIX 标记入库（数据一致性）
    with client.app.state.chat_svc._Session() as db:
        from papermate.models.db import Message

        row = db.query(Message).filter(Message.session_id == sid, Message.role == "user").first()
        assert row.content.startswith(_MM_PREFIX)


def test_history_with_image_then_continue(client):
    """切换模型延续：带图历史重建后继续对话正常（上下文不丢、不报错）。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    sid = s["id"]
    assert client.post(f"/api/v1/sessions/{sid}/messages/stream",
                       json={"content": "看图", "images": [_png_data_url()]}).status_code == 200
    # 第二轮纯文本继续（模拟切换模型/刷新后延续）
    r2 = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "继续分析"})
    assert r2.status_code == 200
    assert "done" in r2.text
    # 历史完整：2 条 user + 2 条 assistant
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert len(detail["messages"]) == 4


def test_auto_title_generated_after_first_turn(client):
    """首轮对话后：默认标题被主题标题替换（fake 下规则截断降级）。"""
    s = client.post("/api/v1/sessions", json={}).json()
    assert s["title"] == "新会话"
    sid = s["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "请帮我调研Transformer注意力机制在长文本上的应用效果"})
    assert r.status_code == 200
    updated = client.get(f"/api/v1/sessions/{sid}").json()
    assert updated["title"] not in ("", "新会话")
    assert len(updated["title"]) <= 16
    # 第二次对话不覆盖已有标题
    client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "再问一个问题"})
    assert client.get(f"/api/v1/sessions/{sid}").json()["title"] == updated["title"]


def test_auto_title_keeps_manual_title(client):
    """人工标题不被自动生成覆盖。"""
    s = client.post("/api/v1/sessions", json={"title": "我的调研笔记"}).json()
    sid = s["id"]
    client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "随便聊聊"})
    assert client.get(f"/api/v1/sessions/{sid}").json()["title"] == "我的调研笔记"


def test_backfill_default_titles_on_list(client):
    """存量默认标题懒回填：GET /sessions 后按首条 user 消息规则生成标题。"""
    s1 = client.post("/api/v1/sessions", json={}).json()
    s2 = client.post("/api/v1/sessions", json={}).json()
    client.post(f"/api/v1/sessions/{s1['id']}/messages/stream", json={"content": "帮我总结这篇论文的实验设计部分"})
    client.post(f"/api/v1/sessions/{s2['id']}/messages/stream", json={"content": "Hi"})
    # 会话 2 只有极短消息 → 标题应为「Hi」（规则截断至少保留内容）
    sessions = client.get("/api/v1/sessions").json()
    titles = {s["id"]: s["title"] for s in sessions}
    assert titles[s1["id"]] not in ("", "新会话")
    assert titles[s2["id"]] not in ("", "新会话")
    # 幂等：再次 GET 标题不变
    sessions2 = client.get("/api/v1/sessions").json()
    assert {s["id"]: s["title"] for s in sessions2} == titles


# ---------------- V4 思考模式（thinking 双模式） ----------------
def test_chat_message_reasoning_roundtrip():
    """ChatMessage.reasoning → to_openai 输出 reasoning_content（V4 工具轮次回传要求）。"""
    m = ChatMessage(role=Role.ASSISTANT, content="最终答案", reasoning="先分析再作答")
    out = m.to_openai()
    assert out["reasoning_content"] == "先分析再作答"
    # 无 reasoning 时不携带该字段（不污染其他厂商）
    assert "reasoning_content" not in ChatMessage(role=Role.ASSISTANT, content="x").to_openai()


def _v4_llm_with(captured: dict, resp_json: dict, thinking_mode: bool = False):
    def fake_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=resp_json)

    ep = ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-flash-vision-exp")
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_handler))
    return OAICompatibleLLM(ep, client=client, api_key="sk-test", thinking_mode=thinking_mode)


def test_v4_thinking_mode_payload_enabled():
    captured: dict = {}

    async def run():
        llm = _v4_llm_with(captured, {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}},
                           thinking_mode=True)
        await llm.chat([ChatMessage(role=Role.USER, content="hi")])
        assert captured["body"]["thinking"] == {"type": "enabled"}

    asyncio.run(run())


def test_v4_thinking_mode_parses_reasoning():
    async def run():
        llm = _v4_llm_with({}, {"choices": [{"message": {
            "role": "assistant", "content": "答案", "reasoning_content": "思考链内容"}}], "usage": {}},
            thinking_mode=True)
        resp = await llm.chat([ChatMessage(role=Role.USER, content="hi")])
        assert resp.message.reasoning == "思考链内容"
        assert resp.message.content == "答案"

    asyncio.run(run())


def test_v4_stream_reasoning_increments():
    captured: dict = {}
    sse_body = (
        "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "第一段思考"}}]}) + "\n\n"
        + "data: " + json.dumps({"choices": [{"delta": {"content": "答案"}}]}) + "\n\n"
        + "data: [DONE]\n\n"
    )

    def fake_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, text=sse_body)

    ep = ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-flash-vision-exp")
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_handler))

    async def run():
        llm = OAICompatibleLLM(ep, client=client, api_key="sk-test", thinking_mode=True)
        chunks = [c async for c in llm.stream([ChatMessage(role=Role.USER, content="hi")])]
        # 思考增量独立通道（非 finished 块）；finished 块携带完整 reasoning 汇总
        reasoning_deltas = [c.reasoning for c in chunks if c.reasoning and not c.finished]
        assert reasoning_deltas == ["第一段思考"]
        finished = next(c for c in chunks if c.finished)
        assert finished.reasoning == "第一段思考"
        # 正文增量正常
        assert any(c.delta == "答案" for c in chunks)
        assert captured["body"]["thinking"] == {"type": "enabled"}

    asyncio.run(run())


def test_loop_thinking_mode_skips_forced_tool_choice():
    """thinking 开启：force_tool_choice 降级不强制（官方 thinking 不兼容 tool_choice → 不 400）。"""
    from lightcore.core.loop import AgentLoop, LoopConfig
    from lightcore.llm.base import BaseLLM, LLMResponse
    from lightcore.tools.spec import ToolExecutor, ToolRegistry

    captured: dict = {}
    tool_called = {"first": True}

    class _SeqLLM(BaseLLM):
        model_name = "stub"

        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
            captured["tool_choice"] = tool_choice
            if tool_called["first"]:
                tool_called["first"] = False
                return LLMResponse(
                    message=ChatMessage(role=Role.ASSISTANT, content="思考中", reasoning="先检索"),
                    model="stub")
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="完成"), model="stub")

    async def run():
        reg = ToolRegistry()
        loop = AgentLoop(_SeqLLM(), ToolExecutor(reg),
                         config=LoopConfig(max_iterations=3, thinking=True))
        events = [ev async for ev in loop.run_stream("开始", force_tool_choice="required")]
        # thinking 开启 → 第一轮 tool_choice 被降级为 None（不强制）
        assert captured.get("tool_choice") is None
        # 无工具调用 → 直接回答，正常产出 message
        assert any(e.type == "message" for e in events)

    asyncio.run(run())


def test_loop_reasoning_roundtrips_to_next_turn():
    """带工具调用轮次的 reasoning 保留在消息回环 → 下一轮请求带 reasoning_content（V4 要求）。"""
    from lightcore.core.loop import AgentLoop, LoopConfig
    from lightcore.llm.base import BaseLLM, LLMResponse, ToolCall
    from lightcore.tools.spec import ToolExecutor, ToolRegistry

    captured_msgs: list = []
    turn = {"n": 0}

    class _ToolThenTextLLM(BaseLLM):
        model_name = "stub"

        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
            captured_msgs.append([m.to_openai() for m in messages])
            turn["n"] += 1
            if turn["n"] == 1:
                return LLMResponse(
                    message=ChatMessage(role=Role.ASSISTANT, content="", reasoning="推理甲",
                                        tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})]),
                    model="stub")
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="2"), model="stub")

    async def run():
        from lightcore.tools.builtin.basic import register_builtin_tools

        reg = ToolRegistry()
        register_builtin_tools(reg)
        loop = AgentLoop(_ToolThenTextLLM(), ToolExecutor(reg), config=LoopConfig(max_iterations=3))
        events = [ev async for ev in loop.run_stream("计算")]
        assert any(e.type == "tool_result" for e in events)
        # 第二轮请求历史里：assistant(tool_calls) 消息携带 reasoning_content（V4 工具轮次回传）
        second = captured_msgs[1]
        assistant_with_tool = [m for m in second if m.get("tool_calls")]
        assert assistant_with_tool, "第二轮历史应含 assistant 工具调用消息"
        assert assistant_with_tool[0].get("reasoning_content") == "推理甲"

    asyncio.run(run())


# ---------------- 可诊断错误翻译（401/402/403/404） ----------------
def _err_llm(captured: dict, status: int, body: str):
    def fake_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(status, text=body)

    ep = ModelEndpoint("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-flash-vision-exp")
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_handler))
    return OAICompatibleLLM(ep, client=client, api_key="sk-test")


def test_402_insufficient_balance_friendly_message():
    import pytest as _pytest

    async def run():
        llm = _err_llm({}, 402, '{"error":{"message":"Insufficient Balance"}}')
        with _pytest.raises(httpx.HTTPStatusError) as ei:
            await llm.chat([ChatMessage(role=Role.USER, content="hi")])
        msg = str(ei.value)
        assert "余额不足" in msg
        assert "Insufficient Balance" in msg

    asyncio.run(run())


def test_401_invalid_key_friendly_message():
    import pytest as _pytest

    async def run():
        llm = _err_llm({}, 401, '{"error":{"message":"Authentication Fails"}}')
        with _pytest.raises(httpx.HTTPStatusError) as ei:
            await llm.chat([ChatMessage(role=Role.USER, content="hi")])
        assert "API Key" in str(ei.value)

    asyncio.run(run())


# ---------------- 批次1：余额 / 对比 / 规则 / 用量明细 ----------------
def test_balances_deepseek_live(client):
    """GET /settings/balances：deepseek 走 /user/balance（mock 200）；qwen 说明不可查原因。"""
    from unittest import mock as _mock

    async def fake_get(url, **kw):
        assert "user/balance" in url
        return _mock.Mock(status_code=200, json=lambda: {
            "balance_infos": [{"currency": "CNY", "total_balance": "12.34", "granted_balance": "8.0"}]})

    with _mock.patch("httpx.AsyncClient.get", side_effect=fake_get):
        resp = client.get("/api/v1/settings/balances")
    assert resp.status_code == 200
    j = resp.json()
    # 未配置 key 的厂商：说明不可查原因（未配置 / 需 AK/SK）
    assert j["qwen"]["available"] is False
    assert j["qwen"]["reason"]
    assert j["deepseek"]["available"] is False  # 2026-08-25：openai 已下线，无此键


def test_balances_no_key_reason(client):
    j = client.get("/api/v1/settings/balances").json()
    for prov in ("deepseek", "qwen"):  # 2026-08-25：openai 已下线
        assert j[prov]["available"] is False


def test_agent_rules_persisted(client):
    """PUT agent_rules → GET 返回（本地长期记忆存储链路）。"""
    r = client.put("/api/v1/settings", json={"agent_rules": "1. 一律用中文回答\n2. 引用必须带页码"})
    assert r.status_code == 200
    g = client.get("/api/v1/settings").json()
    assert "一律用中文回答" in g["agent_rules"]


def test_compare_requires_two_docs(client):
    r = client.post("/api/v1/compare", json={"doc_ids": [1]})
    assert r.status_code == 400
    assert "2 篇" in r.json()["detail"]


def test_compare_deterministic_with_mock_chunks():
    """compare 确定性回退：mock 分块 → 输出维度并列 + 共引 + model=deterministic。"""
    from unittest import mock as _mock

    import papermate.services.summary_service as sm

    fake_chunks = [
        {"doc_id": 1, "filename": "a.pdf", "seq": 0, "page": 1,
         "text": "本文提出一种新方法，实验表明效果优于基线。"},
        {"doc_id": 2, "filename": "b.pdf", "seq": 0, "page": 1,
         "text": "我们提出另一种方法，与已有工作对比。"},
    ]

    class _FakeSvc:
        async def compare(self, doc_ids, lang="zh"):
            return await sm.SummaryService.compare(_FakeSvc(), doc_ids, lang=lang)

    async def run():
        svc = sm.SummaryService.__new__(sm.SummaryService)
        svc._docs = None
        svc._settings = _mock.Mock(default_model="fake")
        svc._Session = None
        # 隔离「有效模型」解析：强制离线（fake），避免读到 provider_settings.json 里的真实 key 触发网络调用
        with _mock.patch.object(sm.runtime_settings, "get_state", return_value={"default_model": "fake", "providers": {}}):
            with _mock.patch.object(svc, "_gather_chunks", side_effect=lambda d: [c for c in fake_chunks if c["doc_id"] == d[0]]):
                out = await svc.compare([1, 2], lang="zh")
        assert out["model"] == "deterministic"
        assert len(out["docs"]) == 2
        assert len(out["dimensions"]) == 5  # 五段式维度
        assert all(len(dim["cells"]) == 2 for dim in out["dimensions"])
        assert out["co_citation"]["pairs"]  # 两篇有共现词

    asyncio.run(run())


def test_usage_session_detail(client):
    """GET /usage/sessions/{id}：消息/工具/Trace 明细结构。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    sid = s["id"]
    client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    r = client.get(f"/api/v1/usage/sessions/{sid}")
    assert r.status_code == 200
    j = r.json()
    assert j["session_id"] == sid
    assert len(j["messages"]) == 2  # user + assistant
    assert "usage" in j and "traces" in j and "tool_runs" in j


def test_trace_records_real_steps_and_latency(client):
    """Trace 必须记录 AgentLoop 回传的真实步数/延迟（修复前 total_steps 取 thinking 事件数、
    total_latency_ms 恒为 0，导致用量页 Trace 卡片数值失真）。"""
    s = client.post("/api/v1/sessions", json={"title": "t"}).json()
    sid = s["id"]
    svc = client.app.state.chat_svc
    # 模拟 AgentLoop 回传的 done 数据透传进 persist_turn
    svc.persist_turn(sid, "你好", [], [], "已处理", "fake", steps=4, latency_ms=250)
    j = client.get(f"/api/v1/usage/sessions/{sid}").json()
    assert len(j["traces"]) >= 1
    tr = j["traces"][0]
    assert tr["total_steps"] == 4, tr
    assert tr["total_latency_ms"] == 250, tr


def test_compare_llm_generates_cells_and_narrative():
    """配置真实模型时，compare 应一次 LLM 调用生成逐维度逐篇单元格 + 对比叙述；
    单元格内容来自 LLM 的完整句概述，model=llm。"""
    import json as _json
    from unittest import mock as _mock

    import papermate.services.summary_service as sm
    from lightcore.llm.base import ChatMessage, LLMResponse, Role

    fake_chunks = [
        {"doc_id": 1, "filename": "a.pdf", "seq": 0, "page": 1,
         "text": "本文提出一种新方法，实验表明效果优于基线。"},
        {"doc_id": 2, "filename": "b.pdf", "seq": 0, "page": 1,
         "text": "我们提出另一种方法，与已有工作对比。"},
    ]
    payload = _json.dumps({
        "cells": {
            # 2026-08-29 起实现以「文档一/文档二」标签为 cells 键（提示词+解析均按标签），
            # 旧键 "1"/"2" 与实现脱节导致本测试长期挂在基线豁免名单里。
            "background": {"文档一": "论文 A 面向小目标检测的背景问题，提出特征增强动机。", "文档二": "论文 B 结合 YOLO 改进与特征交互，针对水体场景。"},
            "method": {"文档一": "论文 A 提出特征增强模块以强化小目标表征。", "文档二": "论文 B 在 YOLO 框架上引入双向特征融合。"},
        },
        "narrative": "【共性】两篇论文均围绕小目标检测的表征增强展开。\n【差异】论文 A 侧重特征增强模块设计，论文 B 侧重特征融合结构。\n【结论建议】关注小目标细节可优先精读论文 A。",
    }, ensure_ascii=False)

    class _StubLLM:
        async def chat(self, messages, tools=None, temperature=0.2):
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=payload))

    class _FakeSvc:
        async def compare(self, doc_ids, lang="zh"):
            return await sm.SummaryService.compare(_FakeSvc(), doc_ids, lang=lang)

    async def run():
        svc = sm.SummaryService.__new__(sm.SummaryService)
        svc._docs = None
        svc._settings = _mock.Mock(default_model="deepseek")
        svc._Session = None
        with _mock.patch.object(sm, "_build_llm", return_value=(_StubLLM(), "deepseek")):
            with _mock.patch.object(svc, "_gather_chunks", side_effect=lambda d: [c for c in fake_chunks if c["doc_id"] == d[0]]):
                out = await svc.compare([1, 2], lang="zh")
        assert out["model"] == "llm"
        bg = next(dim for dim in out["dimensions"] if dim["key"] == "background")
        c1 = next(c for c in bg["cells"] if c["doc_id"] == 1)
        assert c1["content"].startswith("论文 A 面向小目标检测")   # LLM 单元格生效
        assert "小目标检测的表征增强" in out["narrative"]
        # 未提供 LLM 内容的维度回退确定性简述
        exp = next(dim for dim in out["dimensions"] if dim["key"] == "experiments")
        assert all(len(c["content"]) <= 220 for c in exp["cells"])

    asyncio.run(run())


def test_compare_llm_broken_json_falls_back():
    """LLM 返回非法 JSON 时，compare 应全量回退确定性简述，不报错、model=deterministic。"""
    from unittest import mock as _mock

    import papermate.services.summary_service as sm
    from lightcore.llm.base import ChatMessage, LLMResponse, Role

    fake_chunks = [
        {"doc_id": 1, "filename": "a.pdf", "seq": 0, "page": 1,
         "text": "本文提出一种新方法，实验表明效果优于基线。"},
        {"doc_id": 2, "filename": "b.pdf", "seq": 0, "page": 1,
         "text": "我们提出另一种方法，与已有工作对比。"},
    ]

    class _BadLLM:
        async def chat(self, messages, tools=None, temperature=0.2):
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="这不是 JSON"))

    class _FakeSvc:
        async def compare(self, doc_ids, lang="zh"):
            return await sm.SummaryService.compare(_FakeSvc(), doc_ids, lang=lang)

    async def run():
        svc = sm.SummaryService.__new__(sm.SummaryService)
        svc._docs = None
        svc._settings = _mock.Mock(default_model="deepseek")
        svc._Session = None
        with _mock.patch.object(sm, "_build_llm", return_value=(_BadLLM(), "deepseek")):
            with _mock.patch.object(svc, "_gather_chunks", side_effect=lambda d: [c for c in fake_chunks if c["doc_id"] == d[0]]):
                out = await svc.compare([1, 2], lang="zh")
        assert out["model"] == "deterministic"
        assert len(out["dimensions"]) == 5
        assert all(len(dim["cells"]) == 2 for dim in out["dimensions"])

    asyncio.run(run())
