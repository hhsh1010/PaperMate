"""⑤（2026-10-02）：意图识别纳入**结构化 @ 提及**。

## 缺陷背景（实测，`docs/诊断-离题回答与熔断预警-2026-10-02.md` §1.4）

输入「@文档四 实验对象是什么」：
- `@文档N` 不在 `_DOC_OWNED_MARKERS` 里，「实验对象」也不在 `_DOC_QA_MARKERS` 里
  ⇒ `classify_intent` 判 **`general_qa`** ⇒ 注入的 SYSTEM 注记是
  「**与具体文档无关**：可直接回答」；
- 可同一轮 `doc_bound=True` ⇒ `force_tool_choice="required"` 又**强制它先检索**
  ⇒ **两个信号互相矛盾**（日志实证：`意图路由 … intent=general_qa doc_bound=True → agent`）。

⇒ 修法：**以结构化字段为准**（前端发 `mention_doc_ids`，可靠、已过滤无效 id），
文本里的「@文档四」只是回退渲染 ⇒ 不靠"往词表里加 `@文档N`"那种脆弱做法。

## 本文件在守什么

1. 带 `has_doc_mention` ⇒ `doc_qa`；不带 ⇒ 行为**逐字不变**（老用例全绿）；
2. **命令 / 短寒暄优先级更高**（@ 了文档但说"你好"仍是寒暄，不必检索）；
3. ★ **路由与注记必须同参** —— 否则会出现"路由按 doc_qa 走、注记却说与文档无关"的分裂；
4. 端到端：doc-bound + 结构化 mention 的那一轮，日志里的 `intent` 必须是 `doc_qa`。
"""

from __future__ import annotations

import pytest

from papermate.services.intent import classify_intent, intent_system_note


class TestClassifyWithMention:
    def test_mention_makes_it_doc_qa(self):
        """★ 核心：结构化 @ 提及 ⇒ doc_qa（不再被判"与具体文档无关"）。"""
        assert classify_intent("实验对象是什么", has_doc_mention=True).name == "doc_qa"
        # 真实场景的原文（@文档N 只是回退渲染，文本里可能已被剥离）
        assert classify_intent("@文档四 实验对象是什么", has_doc_mention=True).name == "doc_qa"

    def test_without_mention_behavior_unchanged(self):
        """不带 mention ⇒ 与改动前**逐字一致**（本文件不得改变既有分类语义）。"""
        assert classify_intent("什么是注意力机制？").name == "general_qa"
        assert classify_intent("@文档四 实验对象是什么").name == "general_qa", (
            "没有结构化 mention 时仍按文本判（@文档N 不是强指代词，这是既有语义）"
        )
        assert classify_intent("/总结这篇论文").name == "command"
        assert classify_intent("你好").name == "chitchat"
        assert classify_intent("").name == "chitchat"

    def test_command_and_chitchat_still_win(self):
        """★ 优先级：命令 > 短寒暄 > @提及。"""
        assert classify_intent("/综述", has_doc_mention=True).name == "command"
        assert classify_intent("@文档一 你好", has_doc_mention=True).name == "chitchat"


class TestIntentNoteMatchesClassification:
    def test_note_says_document_not_unrelated(self):
        """★ 反分裂：分类是 doc_qa 时，注记**不得**再说"与具体文档无关"。"""
        note = intent_system_note("实验对象是什么", has_doc_mention=True)
        assert "文档问答" in note
        assert "与具体文档无关" not in note
        assert "retrieve" in note, "doc_qa 的注记应指示先检索"

    def test_general_qa_note_unchanged(self):
        note = intent_system_note("什么是注意力机制？")
        assert "通用问答" in note and "与具体文档无关" in note


class TestRouteWiring:
    def test_route_passes_same_arg_to_classify_and_note(self):
        """源码契约：路由必须用**同一个** `_has_doc_mention` 同时喂给分类与注记。"""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py").read_text(
            encoding="utf-8")
        assert "_has_doc_mention = bool(mention_ids)" in src, (
            "必须由**结构化** mention_ids 派生（而不是解析文本里的 @文档N）"
        )
        assert "it = classify_intent(body.content, has_doc_mention=_has_doc_mention)" in src
        assert "has_doc_mention=_has_doc_mention))" in src, (
            "注记必须与分类同参，否则会出现'路由按 doc_qa 走、注记说与文档无关'的分裂"
        )
        assert src.count("classify_intent(body.content)") == 0, (
            "还有 classify_intent 调用点没接 mention（会与注记分裂）"
        )

# ============ 端到端：结构化 mention ⇒ 日志可观测到 intent=doc_qa ============

@pytest.fixture
def client():
    import os
    import tempfile

    import papermate.config as cfg_mod
    from fastapi.testclient import TestClient

    tmp = tempfile.mkdtemp(prefix="papermate_intent_mention_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_doc_bound_mention_logs_doc_qa(client, caplog):
    """★ 端到端：带 `mention_doc_ids` 的那一轮，路由日志里的 `intent` 必须是 doc_qa。

    这条日志正是本次诊断的取证入口（`意图路由 … intent=general_qa doc_bound=True → agent`
    暴露了矛盾信号）⇒ 修好之后它必须变成 `intent=doc_qa`。
    """
    import logging

    caplog.set_level(logging.INFO, logger="papermate.api.routes_search")
    s = client.post("/api/v1/sessions", json={"title": "mention-intent"}).json()
    up = client.post(
        f"/api/v1/documents?session_id={s['id']}",
        files={"file": ("d.md", "量子纠缠是一种非局域关联现象。".encode(), "text/markdown")},
    )
    assert up.status_code == 200, up.text
    doc_id = up.json()["id"]

    r = client.post(
        f"/api/v1/sessions/{s['id']}/messages/stream",
        # ★ 结构化提及（前端真实发法），正文里**没有** "@文档一"
        json={"content": "实验对象是什么", "mention_doc_ids": [doc_id]},
    )
    assert r.status_code == 200, r.text

    lines = [rec.getMessage() for rec in caplog.records if "意图路由" in rec.getMessage()]
    assert lines, "意图路由日志是本次诊断的取证入口，不得消失"
    assert "intent=doc_qa" in lines[0], f"结构化 mention 未生效：{lines[0]}"
    assert "doc_bound=True" in lines[0], f"前置条件：该会话应 doc-bound：{lines[0]}"
