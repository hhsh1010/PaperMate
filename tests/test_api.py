"""M3.1 集成冒烟：创建会话 → 上传文档 → SSE 流式提问 → 用量统计（docs/07 M3.1 验收）

覆盖三条验收线：
A. curl/Postman 可完成「创建会话 → 上传 PDF/MD → 流式提问」全流程；
B. 流式中途断开可重连（SSE 天然支持，本测试验证事件完整闭环：thinking→tool_call→tool_result→message→done）；
C. 并发 2 个会话互不干扰（本测试验证会话级检索隔离：各自只命中自己上传的文档）。
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    cur_name = None
    cur_data = ""
    for line in text.splitlines():
        if line.startswith("event:"):
            cur_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            cur_data = line[len("data:"):].strip()
            if cur_name is not None:
                import json

                events.append((cur_name, json.loads(cur_data)))
                cur_name, cur_data = None, ""
    return events


DOC_A = "量子纠缠是量子力学中一种非局域关联现象。\n\n两个粒子发生纠缠后，对其中一个测量会瞬间决定另一个的状态。" \
        "这在量子通信与量子计算中有重要应用。"
DOC_B = "Transformer 模型通过自注意力机制实现并行化序列建模。\n\n注意力分数由 Query 与 Key 的点积缩放后得到，" \
        "再对 Value 加权求和。该机制取代了 RNN 的递归结构。"


def _upload(client, text: str, session_id=None) -> dict:
    files = {"file": ("doc.md", text.encode("utf-8"), "text/markdown")}
    url = "/api/v1/documents"
    if session_id is not None:
        url += f"?session_id={session_id}"
    r = client.post(url, files=files)
    assert r.status_code == 200, r.text
    return r.json()


def _stream(client, session_id: int, content: str) -> tuple[str, list[str]]:
    r = client.post(f"/api/v1/sessions/{session_id}/messages/stream", json={"content": content})
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    names = [e[0] for e in events]
    message = next((d["text"] for n, d in events if n == "message"), "")
    return message, names


def test_healthz(client):
    r = client.get("/api/v1/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_flow_and_isolation(client):
    # 会话 A + 文档 A
    sa = client.post("/api/v1/sessions", json={"title": "会话A"}).json()
    da = _upload(client, DOC_A, session_id=sa["id"])
    assert da["status"] == "ready"
    assert da["num_chunks"] > 0

    # 会话 B + 文档 B
    sb = client.post("/api/v1/sessions", json={"title": "会话B"}).json()
    db = _upload(client, DOC_B, session_id=sb["id"])
    assert db["status"] == "ready"

    # 会话 A 提问（用明确指代文档的 doc_qa 问法，确保命中检索路径，验证会话隔离）
    msg_a, names_a = _stream(client, sa["id"], "文档里提到的量子纠缠是什么？")
    # 验收 B：事件闭环完整
    assert "thinking" in names_a
    assert "tool_call" in names_a
    assert "tool_result" in names_a
    assert "message_delta" in names_a
    assert "message" in names_a
    assert "done" in names_a
    # 会话隔离：A 的答案只引用文档 A（含"量子纠缠"），不应出现文档 B 的"注意力"
    assert "量子纠缠" in msg_a
    assert "注意力" not in msg_a

    # 会话 B 提问（明确指代论文）
    msg_b, names_b = _stream(client, sb["id"], "论文中注意力机制是怎么计算的？")
    assert "量子纠缠" not in msg_b
    assert "注意力" in msg_b

    # 验收 A：用量统计包含两条会话，且工具被调用
    usage = client.get("/api/v1/usage").json()
    assert len(usage) >= 2
    row_a = next(r for r in usage if r["session_id"] == sa["id"])
    assert row_a["num_messages"] >= 2
    assert row_a["num_tool_runs"] >= 1


def test_dedupe_precheck(client):
    """重复上传检测：同内容 → duplicate；同名不同内容 → same_name；全新 → 无重复。"""
    text = "重复检测专用内容：unique-marker-9f3c1a"

    # 第一次上传后，precheck 同内容应命中 duplicate
    _upload(client, text)
    files = {"file": ("doc.md", text.encode("utf-8"), "text/markdown")}
    r = client.post("/api/v1/documents/precheck", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["duplicate"] is True
    assert body["same_name"] is True
    assert body["existing"] is not None
    assert body["sha256"]  # 非空哈希

    # 同名但内容不同 → same_name=True / duplicate=False
    files2 = {"file": ("doc.md", "另一份不同内容-xyz".encode(), "text/markdown")}
    r2 = client.post("/api/v1/documents/precheck", files=files2)
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["duplicate"] is False
    assert b2["same_name"] is True
    assert b2["existing"] is not None

    # 全新文件名 + 内容 → 均 False
    files3 = {"file": ("brand-new.md", "全新内容-abc".encode(), "text/markdown")}
    r3 = client.post("/api/v1/documents/precheck", files=files3)
    assert r3.status_code == 200
    b3 = r3.json()
    assert b3["duplicate"] is False
    assert b3["same_name"] is False
    assert b3["existing"] is None


def test_settings_dedupe_warning_toggle(client):
    """重复上传提醒开关：GET 默认开 → PUT 关闭 → GET 读回 False。"""
    s = client.get("/api/v1/settings").json()
    assert s["dedupe_warning"] is True
    r = client.put("/api/v1/settings", json={"dedupe_warning": False})
    assert r.status_code == 200
    assert r.json()["dedupe_warning"] is False
    assert client.get("/api/v1/settings").json()["dedupe_warning"] is False
    # 还原，避免影响其他用例
    client.put("/api/v1/settings", json={"dedupe_warning": True})


def test_search_debug_endpoint(client):
    sa = client.post("/api/v1/sessions", json={"title": "dbg"}).json()
    da = _upload(client, DOC_A, session_id=sa["id"])
    r = client.post("/api/v1/search", json={"query": "纠缠", "top_k": 2, "doc_ids": [da["id"]]})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) >= 1
    assert any("纠缠" in h["text"] for h in hits)


def test_no_doc_reminds_missing_document(client):
    """未关联文档却指代文档内容 → 协议级拦截：不进入 Agent 循环（无 tool_call），
    直接返回「缺失文档」提醒，不编造内容。"""
    s = client.post("/api/v1/sessions", json={"title": "no-doc"}).json()
    msg, names = _stream(client, s["id"], "翻译这篇论文的摘要")
    assert "tool_call" not in names          # 协议级拦截，未构建/运行 Agent
    assert "message" in names and "done" in names
    assert "我无法基于论文内容回答" in msg     # 唯一于拦截文案，区别于 fake 的 source=none 提示
    assert "关联文档" in msg


def test_no_doc_general_question_not_blocked(client):
    """未关联文档但纯通用问题（无文档指代）→ 不拦截、不强制检索，走直接 LLM 路径给出普通 AI 回答。"""
    s = client.post("/api/v1/sessions", json={"title": "no-doc-gen"}).json()
    msg, names = _stream(client, s["id"], "什么是注意力机制？")
    assert "tool_call" not in names             # 通用问题直接走 LLM，不触发 retrieve
    assert "message_delta" in names             # 仍有流式输出
    assert "message" in names and "done" in names
    assert "基于通用知识的直接回答" in msg       # FakeLLM 直接回答占位文案（已去除流程叙述）
    assert "过程" not in msg                     # 回复模板不得含「过程」板块


def test_no_doc_command_reminds_missing_document(client):
    """未关联文档却发文档类命令（/总结）→ 协议级拦截，直接提醒先关联文档。"""
    s = client.post("/api/v1/sessions", json={"title": "no-doc-cmd"}).json()
    msg, names = _stream(client, s["id"], "/总结")
    assert "tool_call" not in names
    assert "message" in names and "done" in names
    assert "关联文档" in msg
    assert "命令类操作" in msg


def test_no_doc_research_request_not_blocked(client):
    """未关联文档但请求『搜集/推荐前沿论文』等通用研究任务 → 不应被协议级拦截，
    应放行给大模型基于通用知识（或联网）作答（修复：原 has_document_reference 把"前沿论文"
    误判为指代用户自有文档，导致无文档时弹出"请关联文档"）。"""
    s = client.post("/api/v1/sessions", json={"title": "no-doc-research"}).json()
    msg, names = _stream(client, s["id"], "搜集一些关于目标检测的前沿论文")
    assert "我无法基于论文内容回答" not in msg   # 未被误拦截
    assert "关联文档" not in msg                 # 未弹"请关联文档"提醒
    assert "message" in names and "done" in names


def test_no_doc_reminder_mentions_library_when_uploaded(client):
    """已上传文档到文献库但未关联到本会话 → 拦截文案应点明『已上传 N 篇』，
    直接修复『文档上传过却显示未上传』的困惑（上传≠关联到本会话）。"""
    _upload(client, DOC_A)  # 仅上传到文献库，不关联任何会话
    s = client.post("/api/v1/sessions", json={"title": "no-doc-lib"}).json()
    msg, names = _stream(client, s["id"], "翻译这篇论文的摘要")
    assert "tool_call" not in names
    assert "我无法基于论文内容回答" in msg
    assert "已上传" in msg            # 文献库有文档 → 提示已上传但未关联本会话
    assert "关联文档" in msg


def test_session_delete_cascade(client):
    s = client.post("/api/v1/sessions", json={"title": "to-del"}).json()
    _upload(client, DOC_A, session_id=s["id"])
    _stream(client, s["id"], "量子纠缠？")
    assert client.delete(f"/api/v1/sessions/{s['id']}").json()["ok"] is True
    assert client.get(f"/api/v1/sessions/{s['id']}").status_code == 404


def test_cross_session_memory_recall_note_wired(client):
    # M2.1：ChatService 跨会话记忆已接线——直接往全局记忆库写一条事实，
    # recall_note 应把它作为 SYSTEM 注记返回并注入后续对话（docs/07 M2.1 验收#2）。
    chat_svc = client.app.state.chat_svc
    chat_svc._memory.long.add("用户偏好用中文回答", importance=0.9)
    note = chat_svc.recall_note(1, "用户希望用什么语言交流？", k=5)
    assert note is not None
    assert note.role.value == "system"
    assert "【长期记忆】" in note.content
    assert "中文" in note.content


def test_sse_flow_writes_memory_store(client):
    # 跨会话记忆落盘点验证：先写入一条事实（触发 JSON 落盘），再完整跑一轮 SSE
    # （含 recall_note 注入 + 末尾 reflect），确认 memories.json 持久化未被破坏且含该事实。
    chat_svc = client.app.state.chat_svc
    chat_svc._memory.long.add("用户偏好用中文回答", importance=0.9)  # 触发落盘
    tmp = os.environ["PAPERMATE_DATA_DIR"]
    mem_file = Path(tmp) / "memories.json"
    assert mem_file.exists(), "写入记忆后应生成 memories.json（跨会话记忆落盘点）"

    s = client.post("/api/v1/sessions", json={"title": "mem"}).json()
    _upload(client, DOC_A, session_id=s["id"])
    _stream(client, s["id"], "量子纠缠是什么？")  # 走完整 SSE（含 recall_note 注入 + reflect）

    # 跑完仍应包含已写入的事实（持久化未被覆盖/丢失）
    data = json.loads(mem_file.read_text(encoding="utf-8"))
    assert any("中文" in it["text"] for it in data)


def test_stream_with_images_rejected_on_non_multimodal_model(client):
    """2026-08-25：带图消息仅多模态模型可发；真实非多模态模型（deepseek-v4-flash） → 400 明确提示，
    避免图片块被静默忽略或厂商 400（对应前端 ChatInput 图片按钮禁用）。
    注：fake 本地替身不真正把图发给厂商，放行（见 test_chat_with_image_streams_ok_on_fake）。"""
    s = client.post("/api/v1/sessions", json={"title": "img", "config": {"model": "deepseek-v4-flash"}}).json()
    r = client.post(
        f"/api/v1/sessions/{s['id']}/messages/stream",
        json={"content": "看图", "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="]},
    )
    assert r.status_code == 400, r.text
    assert "不支持图片" in r.json()["detail"]


def test_grounding_network_off_blocks_external_tools(client):
    """统一限制（联网关闭 + 未关联文档）：进入纯对话模式，system_prompt 标记『未关联文档 · 未开启联网』、
    要求基于通用知识直接回答；同时不注册任何独立网络工具（网络能力经大模型原生联网提供）。

    注：文档优先约束（不得引用文档外来源等）仅在『已关联文档』时注入；无文档时为纯对话模式，
    对应 _grounding_block(has_docs=False) 与需求『未上传文档时默认使用大模型对话能力，不读取任何文档』。
    """
    s = client.post("/api/v1/sessions", json={"title": "g0"}).json()
    agent = asyncio.run(client.app.state.chat_svc.build_agent(s["id"]))
    names = {sp.name for sp in agent.tools.registry.specs()}
    assert "arxiv_search" not in names
    assert "web_search" not in names
    assert "fetch_url" not in names
    assert "【对话模式（当前：未关联文档" in agent.system_prompt
    assert "未开启联网" in agent.system_prompt
    assert "请基于你的通用知识直接回答" in agent.system_prompt


def test_grounding_network_on_enables_llm_web_search(client):
    """统一限制（联网开启 + 未关联文档）：纯对话模式下 system_prompt 标记『未关联文档 · 已开启联网』，
    允许『基于大模型自身知识（含联网检索补充）』回答通用问题；不注册独立网络工具。
    （联网能力经大模型原生联网搜索提供，而非独立 web 工具，见回链 docs/统一落地方案）
    """
    s = client.post("/api/v1/sessions", json={"title": "g1"}).json()
    r = client.patch(f"/api/v1/sessions/{s['id']}", json={"config": {"enable_network": True}})
    assert r.status_code == 200
    agent = asyncio.run(client.app.state.chat_svc.build_agent(s["id"]))
    names = {sp.name for sp in agent.tools.registry.specs()}
    # 工具检索方式已移除：不再注册 arxiv_search/web_search/fetch_url 等独立网络工具
    assert "arxiv_search" not in names
    assert "web_search" not in names
    assert "fetch_url" not in names
    assert "【对话模式（当前：未关联文档" in agent.system_prompt
    assert "已开启联网" in agent.system_prompt
    assert "含联网检索补充" in agent.system_prompt


def test_inventory_exposes_skill_icon_category_and_gating(client):
    """技能 palette 数据源（GET /settings/inventory）：
    每个技能须带 icon/category 展示元数据；依赖子功能开关（dev_eval）默认 disabled。
    这是前端命令栏由『后端 SkillSpec 注册表统一驱动』的契约保障。
    """
    r = client.get("/api/v1/settings/inventory")
    assert r.status_code == 200
    skills = r.json()["skills"]
    assert len(skills) >= 5
    # 展示元数据齐备
    for sk in skills:
        assert sk["icon"] and sk["category"], f"技能 {sk['name']} 缺 icon/category"
        assert sk["commands"] and sk["commands"][0].startswith("/")
    # 命令集合与硬编码旧列表一致（回归保障）
    cmds = {c for sk in skills for c in sk["commands"]}
    assert {"/综述", "/相关工作", "/复现", "/规划", "/eval"} <= cmds
    # dev_eval 依赖 features.dev_eval（默认 False）→ enabled=False
    dev = next(s for s in skills if s["name"] == "dev_eval")
    assert dev["feature"] == "dev_eval" and dev["enabled"] is False


def test_inventory_includes_domain_tools_when_feature_on(client):
    """「可用工具」清单须与 build_agent 注册口径一致：默认 features 下应包含
    domain 5 工具（memory_search/memory_write/summarize/related_work/plan_executor），
    而非仅 basic 5 工具。回归保障 inventory() 不再漏列 feature 门控工具。
    """
    r = client.get("/api/v1/settings/inventory")
    assert r.status_code == 200
    names = {t["name"] for t in r.json()["tools"]}
    # basic 5（始终注册）
    assert {"retrieve", "pdf_extract", "citation_lookup", "calculator", "code_interpreter"} <= names
    # domain 5（默认 features.memory/summary/planning 均为 True）
    assert {"memory_search", "memory_write", "summarize", "related_work", "plan_executor"} <= names
    # 不注册独立网络工具（联网走大模型原生能力，见 test_grounding_*）
    assert "arxiv_search" not in names and "web_search" not in names and "fetch_url" not in names


def test_inventory_tools_carry_enabled_flag(client):
    """工具清单与技能同口径（2026-09-05）：**全量列出 + enabled 标记**——
    关停功能的工具灰显展示（enabled=False）而非直接隐藏；开关切换实时反映。
    此前口径：关停工具直接从清单消失，用户在设置页看不到工具全貌。"""
    r = client.get("/api/v1/settings")
    original_features = r.json()["features"]

    try:
        # 1) 默认全开 → domain 工具全部 enabled=True
        tools = {t["name"]: t for t in client.get("/api/v1/settings/inventory").json()["tools"]}
        assert all("enabled" in t for t in tools.values()), "工具条目必须携带 enabled 标记"
        for n in ("memory_search", "memory_write", "summarize", "related_work", "plan_executor"):
            assert tools[n]["enabled"] is True
        assert tools["retrieve"]["enabled"] is True  # 无门控内建工具恒 enabled

        # 2) 关停 summary → summarize/related_work enabled=False 但仍在列（全量口径）
        r = client.put("/api/v1/settings", json={"features": {"summary": False}})
        assert r.status_code == 200, r.text
        tools = {t["name"]: t for t in client.get("/api/v1/settings/inventory").json()["tools"]}
        assert tools["summarize"]["enabled"] is False
        assert tools["related_work"]["enabled"] is False
        assert tools["retrieve"]["enabled"] is True
    finally:
        # 还原 features，避免污染进程级 runtime_settings 单例（见项目教训）
        client.put("/api/v1/settings", json={"features": original_features})


def test_export_falls_back_to_parsed_text(client):
    """导出兜底（2026-09-05）：异步上传管线不写 meta_json["processing"]，导出须回退到
    解析文本（parsed_dir 落盘 → chunk 表回退），保证所有就绪文档可导出（此前一律 404）。
    可行性结论：解析文本与 /text 预览、版本对比同源，零额外解析成本。"""
    up = client.post(
        "/api/v1/documents",
        files={"file": ("导出兜底测试.md", "# 标题\n\n正文内容用于导出验证。".encode())},
    )
    assert up.status_code == 200, up.text
    doc_id = up.json()["id"]

    # md：含兜底生成的标题 + 解析正文
    r = client.get(f"/api/v1/documents/{doc_id}/export?format=md")
    assert r.status_code == 200, r.text
    assert r.headers.get("x-export-format") == "md"
    assert "导出兜底测试" in r.text and "正文内容用于导出验证。" in r.text

    # txt：去标记纯文本
    r = client.get(f"/api/v1/documents/{doc_id}/export?format=txt")
    assert r.status_code == 200 and "正文内容用于导出验证。" in r.text

    # json：结构化 {title, markdown}
    r = client.get(f"/api/v1/documents/{doc_id}/export?format=json")
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "导出兜底测试" and "正文内容" in body["markdown"]


def test_upload_rejects_duplicate_content(client):
    """服务端内容去重（2026-09-06）：此前去重只有 precheck 一个孤儿调用方（前端死代码组件），
    异步上传链路从未做服务端去重 → 同一文档可无限重复入库。现在：
    - 同内容二次上传 → 409 + existing_id/existing_filename（content_hash 主判据）；
    - 处理中（uploaded）/已就绪（ready）都拦截；failed/cancelled 放行重传；
    - allow_duplicate=true 显式逃生门。"""
    files = {"file": ("去重测试.md", "去重测试唯一内容，不应重复入库。".encode())}
    r1 = client.post("/api/v1/documents/async", files=files)
    assert r1.status_code == 200, r1.text
    first_id = r1.json()["id"]

    # 同内容、不同文件名 → 仍按 content_hash 判重拦截
    files2 = {"file": ("换个名字.md", "去重测试唯一内容，不应重复入库。".encode())}
    r2 = client.post("/api/v1/documents/async", files=files2)
    assert r2.status_code == 409, r2.text
    body = r2.json()
    assert body["existing_id"] == first_id
    assert body["existing_filename"] == "去重测试.md"

    # allow_duplicate=true 逃生门
    r3 = client.post("/api/v1/documents/async?allow_duplicate=true", files=files2)
    assert r3.status_code == 200, r3.text


def test_migration_keeps_duplicate_collection_documents(client):
    """迁移改告警式（2026-09-06）：`_unique_collection_documents` 检测到重复关联时
    **不得删除业务数据**（旧版按 MIN(id) 静默 DELETE，会丢失 seq 排序语义且无备份），
    只打 warning 并跳过唯一索引创建；无重复时正常建唯一索引。"""
    import sqlalchemy

    from papermate.models.migrations import _unique_collection_documents

    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE collection_documents (id INTEGER PRIMARY KEY, "
            "collection_id INTEGER, document_id INTEGER, seq INTEGER)"
        ))
        # 两行重复（同 collection+document），seq 不同
        conn.execute(sqlalchemy.text(
            "INSERT INTO collection_documents (collection_id, document_id, seq) VALUES (1, 2, 5), (1, 2, 3)"
        ))
        # 一行正常
        conn.execute(sqlalchemy.text(
            "INSERT INTO collection_documents (collection_id, document_id, seq) VALUES (1, 7, 1)"
        ))

    with engine.begin() as conn:
        _unique_collection_documents(conn)

    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(
            "SELECT id, collection_id, document_id, seq FROM collection_documents ORDER BY id"
        )).fetchall()
        # 关键断言：3 行全保留（未删任何数据）
        assert len(rows) == 3, f"迁移不应删除业务数据，实际剩 {len(rows)} 行"
        indexes = [i["name"] for i in sqlalchemy.inspect(engine).get_indexes("collection_documents")]
        assert "idx_coldoc" not in indexes, "有重复时不应创建唯一索引"

    # 无重复 → 建唯一索引成功
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("DELETE FROM collection_documents WHERE document_id = 2"))
    with engine.begin() as conn:
        _unique_collection_documents(conn)
    indexes = [i["name"] for i in sqlalchemy.inspect(engine).get_indexes("collection_documents")]
    assert "idx_coldoc" in indexes


def test_healthz_hides_db_path_and_probes_alive(client):
    """healthz 安全加固（2026-09-06）：不再泄露 DB 绝对路径；增加真实探活字段 db_ok。"""
    r = client.get("/api/v1/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["db_ok"] is True
    assert "db" not in body, "healthz 不应返回引擎 URL（泄露本地绝对路径）"
    assert "papermate.db" not in r.text



def test_api_edit_takes_effect_immediately(client):
    """A 档核心：面板编辑/删除必须立即对对话侧生效（此前需重启后端）。"""
    chat_svc = client.app.state.chat_svc
    ltm = chat_svc._memory.long
    item = ltm.add("用户偏好用中文回答", importance=0.9)
    note_before = chat_svc.recall_note(1, "用户偏好")
    assert note_before is not None and "中文" in note_before.content

    # 1) 编辑：新文本应立刻被召回（且 embedding 已重算）
    r = client.put(f"/api/v1/memories/{item.id}", json={"text": "用户偏好用英文回答"})
    assert r.status_code == 200, r.text
    assert r.json()["importance"] == 0.9, "未显式传 importance 时应保留原值"
    note_after = chat_svc.recall_note(1, "用户偏好用英文")
    assert note_after is not None and "英文" in note_after.content

    # 2) 删除：必须立即从召回中消失
    assert client.delete(f"/api/v1/memories/{item.id}").status_code == 200
    assert chat_svc.recall_note(1, "用户偏好") is None

    # 3) 清空
    ltm.add("临时事实", importance=0.5)
    cleared = client.post("/api/v1/memories/clear").json()
    assert cleared["removed"] >= 1
    assert len(ltm) == 0



def test_memory_embedder_switch_is_runtime_configurable(client):
    """设置页切换记忆向量引擎：落盘 + 单例重建（无需重启后端）。"""
    from papermate.memory_hub import get_memory_manager, reset_memory_manager

    # bge-m3 依赖未必安装；本用例只验证「开关被接受 + 单例被重建」，不验证具体引擎
    r = client.put("/api/v1/settings", json={"memory_embedder": "bge-m3"})
    assert r.status_code == 200, r.text
    assert r.json()["memory_embedder"] == "bge-m3"
    mgr = get_memory_manager(client.app.state.settings)
    assert mgr is not None
    # 切回默认
    assert client.put("/api/v1/settings", json={"memory_embedder": "hash"}).json()["memory_embedder"] == "hash"
    reset_memory_manager()



def test_memory_batch_delete_and_source_session(client):
    """C 档：批量删除接口 + 列表返回来源会话字段。"""
    ltm = client.app.state.chat_svc._memory.long
    a = ltm.add("记忆 A：用户偏好中文回答", importance=0.8, source_session_id=7)
    b = ltm.add("记忆 B：实验在 COCO 上评估", importance=0.5, source_session_id=7)
    ltm.add("记忆 C：PaperMate 是论文助手", importance=0.6)

    listed = client.get("/api/v1/memories").json()
    assert listed["total"] == 3
    by_id = {it["id"]: it for it in listed["items"]}
    assert by_id[a.id]["source_session_id"] == 7, "列表应返回来源会话，便于面板溯源"

    r = client.post("/api/v1/memories/batch-delete", json={"ids": [a.id, b.id, "not-exist"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["removed"] == 2
    assert body["missing"] == ["not-exist"]
    assert client.get("/api/v1/memories").json()["total"] == 1


def test_memory_scope_returned_and_changeable(client):
    """2026-08-29 分层 scope：面板列表返回 scope；PUT 可改归属（全局 ↔ 会话）。"""
    from lightcore.memory.long_term import SCOPE_SESSION

    ltm = client.app.state.chat_svc._memory.long
    g = ltm.add("用户偏好用中文回答", importance=0.9, scope="global")
    s = ltm.add("A 论文实验在 COCO 上评估 mAP 50.2", importance=0.6,
                source_session_id=7, scope=SCOPE_SESSION)

    listed = client.get("/api/v1/memories").json()
    by_id = {it["id"]: it for it in listed["items"]}
    assert by_id[g.id]["scope"] == "global"
    assert by_id[s.id]["scope"] == SCOPE_SESSION and by_id[s.id]["source_session_id"] == 7

    # 会话级记忆改为全局（提升为跨会话偏好）
    r = client.put(f"/api/v1/memories/{s.id}", json={"scope": "global"})
    assert r.status_code == 200, r.text
    assert r.json()["scope"] == "global"
    assert client.get("/api/v1/memories").json()["items"]  # 往返成功即可
    assert ltm.get(s.id).scope == "global"

    # 非法 scope 被拒
    bad = client.put(f"/api/v1/memories/{g.id}", json={"scope": "weird"})
    assert bad.status_code == 400


def test_stream_stop_cancels_backend_llm(monkeypatch, client):
    """2026-09-07 回归：模型「思考中」客户端中断 → 后端 LLM 生成立即被取消。

    旧版只在事件之间检查 is_disconnected——thinking 阶段长时间不产出事件，
    检查永不执行，LLM 调用与计费继续到自然结束。现由 watcher 轮询断连并取消
    Agent 流任务（CancelledError 打断 LLM 流式 await）。"""
    import asyncio
    import time as _time

    from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, LLMStreamChunk, Role

    class _SlowLLM(BaseLLM):
        """非 FakeLLM 的慢速流式替身：首个 delta 后长时间停留（模拟思考中），捕获取消信号。

        注意不能继承 FakeLLM——直答路径对 FakeLLM 走剧本文案分支，根本不会调用 stream。"""

        def __init__(self) -> None:
            super().__init__()
            self.cancelled = False

        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="部分内容"))

        async def stream(self, messages, tools=None, **kw):  # noqa: ANN001, ANN003
            try:
                yield LLMStreamChunk(delta="部分内容")
                await asyncio.sleep(30)  # 模拟思考中的长等待；被取消时在此抛 CancelledError
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    slow = _SlowLLM()
    real_build = client.app.state.chat_svc.build_agent

    async def _patched(session_id, cancel_event=None, restrict_doc_ids=None):
        agent = await real_build(session_id, cancel_event=cancel_event, restrict_doc_ids=restrict_doc_ids)
        agent.llm = slow
        return agent

    monkeypatch.setattr(client.app.state.chat_svc, "build_agent", _patched)

    sid = client.post("/api/v1/sessions", json={"title": "中断测试"}).json()["id"]

    # TestClient 在响应完成前 is_disconnected 恒为 False（真实 uvicorn 会及时投递断连），
    # 故用标志位模拟「客户端已断开」以隔离验证 watcher→取消 机制本身。
    from starlette.requests import Request as _Req

    flag = {"disconnected": False}
    _orig = _Req.is_disconnected

    async def _fake_is_disconnected(self):
        return flag["disconnected"] or await _orig(self)

    monkeypatch.setattr(_Req, "is_disconnected", _fake_is_disconnected)

    # 预置断连标志（= 客户端已断开）：watcher 首次轮询即命中 → 消费端中断 → 取消 LLM 流。
    # 若取消机制失效，本请求将持续 ~30s（stub 的思考等待自然结束）并被断言捕获。
    flag["disconnected"] = True
    client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    _time.sleep(0.3)
    assert slow.cancelled, "客户端断开后，后端 LLM 生成应立即被取消（而非继续跑到自然结束）"


def test_get_pipeline_cancel_event_stops_encoding(client):
    """2026-09-07 回归：停止信号传进编码线程——get_pipeline 预置取消事件 → 批次边界抛
    IndexingCancelled（旧版无进度路径传 None，取消永不到达，编码空转到结束）。"""
    import asyncio
    import threading

    from lightcore.rag.pipeline import IndexingCancelled

    sid = client.post("/api/v1/sessions", json={"title": "编码取消"}).json()["id"]
    doc = _upload(client, "编码取消测试内容，需要现场构建管线。", session_id=sid)
    svc = client.app.state.doc_svc
    svc._pipelines.pop(doc["id"], None)  # 清内存缓存
    # 清落盘缓存（模拟签名失效/重启后必须现场编码的场景；否则 load_state 命中，无需编码）
    for suffix in (".faiss", ".state"):
        (svc._settings.faiss_dir / f"doc-{doc['id']}{suffix}").unlink(missing_ok=True)

    ev = threading.Event()
    ev.set()  # 预置取消：首个批次边界（编码前）即抛

    async def run():
        with pytest.raises(IndexingCancelled):
            await svc.get_pipeline(doc["id"], cancel_event=ev)

    asyncio.run(run())


def test_stream_returns_quietly_when_encoding_cancelled(monkeypatch, client):
    """2026-09-07 回归：管线编码阶段点停止 → 以「cancelled 收尾帧」干净终止（无内容帧、无兜底落库）。

    2026-09-07 修订（F4）：原实现为裸 `return` → 端点声明 -> StreamingResponse 却返回
    JSON `null`（application/json），与 SSE 契约不符、前端流解析异常。现返回标记
    cancelled=True 的 done 帧，语义清晰且流干净收尾。
    """
    import json

    from lightcore.rag.pipeline import IndexingCancelled

    sid = client.post("/api/v1/sessions", json={"title": "编码取消流"}).json()["id"]

    async def _boom(session_id, cancel_event=None, restrict_doc_ids=None):
        raise IndexingCancelled("索引在批次边界被取消")

    monkeypatch.setattr(client.app.state.chat_svc, "build_agent", _boom)
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    assert r.status_code == 200
    assert "message" not in r.text, "编码取消不应产出任何内容帧"
    payloads = [json.loads(ln[len("data:"):].strip())
                for ln in r.text.splitlines() if ln.startswith("data:")]
    assert payloads, f"应有 done 收尾帧（旧实现返回 JSON null，无任何帧）：{r.text!r}"
    assert payloads[-1].get("cancelled") is True, f"收尾帧须标记 cancelled：{payloads}"


def test_stop_endpoint_contract(client):
    """2026-09-07：/sessions/{id}/stop 停止端点——幂等 no-op 契约（无进行中的流也 ok）。"""
    sid = client.post("/api/v1/sessions", json={"title": "停止端点"}).json()["id"]
    r = client.post(f"/api/v1/sessions/{sid}/stop")
    assert r.status_code == 200 and r.json() == {"ok": True}
    # 不存在的会话同样幂等（无副作用）
    r2 = client.post("/api/v1/sessions/999999/stop")
    assert r2.status_code == 200 and r2.json() == {"ok": True}


def test_stream_build_phase_disconnect_cancels_encoding(monkeypatch, client):
    """2026-09-07 回归：build_agent 编码阶段客户端断连 → watcher 置位取消 → 编码在
    批次边界止损（旧版此阶段无 watcher，编码空转到结束）。用耗时替身做秒表判别。"""
    import json
    import time as _time

    from lightcore.rag.pipeline import RAGPipeline as _RealRAGPipeline
    from papermate.services import document_service as ds

    class _SlowPipeline(_RealRAGPipeline):
        """编码首块前停留 5s（模拟 GPU 预热/大文档），捕捉取消。"""

        def add_chunks(self, items, doc_id="doc", **kw):  # noqa: ANN001, ANN003
            _time.sleep(5)
            return super().add_chunks(items, doc_id, **kw)

    monkeypatch.setattr(ds, "RAGPipeline", _SlowPipeline)
    # 强制现场编码：清内存缓存与磁盘 bundle
    sid = client.post("/api/v1/sessions", json={"title": "编码阶段断连"}).json()["id"]
    doc = _upload(client, "编码阶段断连测试内容。", session_id=sid)
    svc = client.app.state.doc_svc
    svc._pipelines.pop(doc["id"], None)
    for suffix in (".faiss", ".state"):
        (svc._settings.faiss_dir / f"doc-{doc['id']}{suffix}").unlink(missing_ok=True)

    # 模拟客户端已断开（TestClient 限制，见 test_stream_stop_cancels_backend_llm 说明）
    from starlette.requests import Request as _Req

    monkeypatch.setattr(_Req, "is_disconnected", lambda self: _async_true())

    async def _async_true():
        return True

    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "你好"})
    assert r.status_code == 200
    # 判别：取消生效 → IndexingCancelled → 以 cancelled 收尾帧干净终止（无 message 内容帧）；
    # 若未取消 → 编码正常完成（替身会 sleep 5s）→ 继续走直答路径，必然产出 message 帧。
    assert "message" not in r.text, "取消生效时不应产出内容帧（编码应在批次边界止损）"
    payloads = [json.loads(ln[len("data:"):].strip())
                for ln in r.text.splitlines() if ln.startswith("data:")]
    assert payloads and payloads[-1].get("cancelled") is True, f"应以 cancelled 收尾：{r.text!r}"


def test_stream_mention_restricts_agent_docs(monkeypatch, client):
    """2026-09-07 回归：结构化提及 mention_doc_ids → build_agent 只构建被提及文档的管线
    （普通对话 retrieve 不再混入其他文档内容；@文档N 序号文本仅作兜底）。"""
    d1_uploaded = _upload(client, "第一篇文档内容：关于图像分类的研究。", session_id=None)  # noqa: F841 未关联，仅入库
    sid = client.post("/api/v1/sessions", json={"title": "提及限缩"}).json()["id"]
    d2 = _upload(client, "第二篇文档内容：关于目标检测的研究。", session_id=sid)

    captured: dict = {}
    real_build = client.app.state.chat_svc.build_agent

    async def _patched(session_id, cancel_event=None, restrict_doc_ids=None):
        captured["restrict"] = restrict_doc_ids
        return await real_build(session_id, cancel_event=cancel_event, restrict_doc_ids=restrict_doc_ids)

    monkeypatch.setattr(client.app.state.chat_svc, "build_agent", _patched)

    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "总结这篇的方法", "mention_doc_ids": [d2["id"]]})
    assert r.status_code == 200
    assert captured["restrict"] == [d2["id"]], captured

    # 非法提及（已删除文档）被过滤 → restrict 回落 None（全量）
    r2 = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                     json={"content": "总结", "mention_doc_ids": [99999]})
    assert r2.status_code == 200
    assert captured["restrict"] is None, captured


def test_build_agent_injects_token_budget_from_runtime_settings(client, monkeypatch):
    """token 预算熔断必须从 runtime_settings 注入 LoopConfig（2026-09-16 可配化）。

    背景：旧硬编码 8000 对"绑定多篇文档"场景过紧（单次 prompt 已达 2659），
    熔断首次真实生效当天就误伤了正常对话 —— 可配 + 提额是用户决策。
    """
    import papermate.runtime_settings as rs

    monkeypatch.setattr(
        rs, "_state",
        {**rs.get_state(), "max_tokens_budget": 12345, "max_tokens_per_turn": 234567},
    )
    s = client.post("/api/v1/sessions", json={"title": "tb"}).json()
    agent = asyncio.run(client.app.state.chat_svc.build_agent(s["id"]))

    assert agent.config.max_tokens_budget == 12345, "单次请求上下文上限"

    # ⚠️ 2026-10-05 修正断言（原为 `== 234567`，即断言"显式 per_turn 设置生效"）：
    # 整轮上限自 **2026-10-03** 起改为**与单次同源派生**，**不再读** `max_tokens_per_turn`：
    #     resolve_turn_budget_for(model) = clamp(resolve_budget_for(model) × 5, 60k, 600k)
    # （见 `runtime_settings.py:273` 的注释：「必须与 resolve_budget_for 同源派生 ——
    #   否则单次放大、整轮不变，就会出现『单次没撞、整轮先撞』的错配」）
    # ⇒ 上面显式传的 234567 会被**忽略**，实际 = clamp(12345 × 5, 60k, 600k) = 61725。
    # 这里按**公式**断言（而非写死 61725），使常量调整时测试自动跟随、不产生假红。
    expected_turn = int(max(
        rs.TURN_BUDGET_FLOOR, min(12345 * rs.TURN_BUDGET_RATIO, rs.TURN_BUDGET_CAP)
    ))
    assert agent.config.max_tokens_per_turn == expected_turn, (
        "整轮上限 = clamp(单次 × TURN_BUDGET_RATIO, FLOOR, CAP)（派生，不再读显式设置项）"
    )
    # ⚠️ **遗留（待产品决策）**：`max_tokens_per_turn` 因此成了**死配置项** ——
    #    它仍存在于 `provider_settings.json` / `public_view` / 设置页且可修改，但**不生效**。
    #    要么删掉（需同步前端与 schema），要么改回"显式优先"。在此之前，本测试固定当前契约。
    assert agent.config.max_cost_per_turn == pytest.approx(
        rs.get_state()["cost_limit_per_turn"]
    ), "成本上限的既有接线不得被本次改动破坏"


def test_last_prompt_tokens_injected_into_loop(client):
    """方案 D′ 接线：`build_agent` 必须把 session config 里的**真实**跨轮基线注入 AgentLoop。

    AgentLoop **每轮新建**（无跨轮缓存）→ 不注入的话，本轮第 1 步（历史最长、最该压缩的那一步）
    永远拿不到真实值，方案 D′ 就退化成"首轮仍靠估算"。这是整条链路唯一容易漏掉的一环。

    ⚠️ **2026-09-28 修正契约**：`_peak_prompt_tokens` **不再**继承注入值。
    此前它继承，导致 `[本轮]` 打出的是**上一轮**的峰值（实测：本轮真实峰值 19,177 却显示 29,357），
    且因只 `max()` 向上，基线成了"只增不减的棘轮" ⇒ 压缩每轮误触发 +
    闸门每轮都去裁历史里的长工具结果。现在"本轮峰值"与"首步 prompt"都是**本轮**口径。
    """
    s = client.post("/api/v1/sessions", json={"title": "d-prime"}).json()
    # 走 service 层写入（等价于上一轮 done 时 routes_search 的回写）
    client.app.state.session_svc.update(s["id"], config={"last_prompt_tokens": 12345})

    agent = asyncio.run(client.app.state.chat_svc.build_agent(s["id"]))

    assert agent._last_real_prompt_tokens == 12345, "必须从 session config 注入（作循环内闸门基线）"
    assert agent._peak_prompt_tokens == 0, "本轮峰值须从 0 起算 —— 继承注入值会让日志显示上一轮的数字"
    assert agent._first_prompt_tokens == 0, "首步 prompt 同样从 0 起算"


def test_session_without_last_prompt_tokens_starts_at_zero(client):
    """没有历史真实值（新会话）→ 注入 0，由 loop 退回估算兜底（不得抛错）。"""
    s = client.post("/api/v1/sessions", json={"title": "fresh"}).json()

    agent = asyncio.run(client.app.state.chat_svc.build_agent(s["id"]))

    assert agent._last_real_prompt_tokens == 0


def test_stream_writes_back_real_prompt_peak(client):
    """一轮对话后必须把**首个真实 prompt**（跨轮基线）回写 session config（方案 D′ 的跨轮传递环节）。

    这是整条链路的最后一环：loop 用真实值做压缩触发 → done 事件回传 `first_prompt_tokens` →
    这里回写 → 下一轮 build_agent 注入。缺任何一环，D′ 都会静默退回"靠估算"。
    ⚠️ 2026-09-28 起回写的是**首步**而非峰值（峰值含本轮工具结果、不进下一轮历史 ⇒ 高估）。

    走 **绑文档的 agent 路径**（有真实 usage 来源）。不绑文档会走直答路径，
    而测试里的 FakeLLM 直答分支**按设计不回写**（那是估算值，见下一条用例）。
    """
    s = client.post("/api/v1/sessions", json={"title": "writeback"}).json()
    _upload(client, DOC_A, s["id"])

    _stream(client, s["id"], "这篇文档讲了什么？")

    cfg = client.app.state.session_svc.get_config(s["id"])
    assert int(cfg.get("last_prompt_tokens") or 0) > 0, (
        f"未回写真实 prompt 基线 → 下一轮无法用真实值触发压缩：{cfg}"
    )


def test_direct_path_does_not_write_back_estimated_tokens(client):
    """直答路径的 **FakeLLM 剧本分支**不得回写 —— 那里的 token 是估算值。

    若把估算写进 `last_prompt_tokens`，下一轮的压缩判据就重新依赖估算，
    方案 D′ 的收益会被悄悄抵消（这是最容易被"顺手加上"破坏的一处）。
    """
    s = client.post("/api/v1/sessions", json={"title": "no-estimate-writeback"}).json()

    _stream(client, s["id"], "你好")  # 不绑文档 → 直答路径 + FakeLLM 剧本分支

    cfg = client.app.state.session_svc.get_config(s["id"])
    assert not cfg.get("last_prompt_tokens"), f"估算值不得写入真实值字段：{cfg}"
