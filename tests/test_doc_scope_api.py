"""场景可见性：`GET /sessions/{id}/doc-scope`（2026-09-23）。

**守的是什么**：三场景（无文档 / 单文档 / 多文档）下系统的检索行为不同，
但此前**界面完全看不出**用户处于哪个场景 —— 实测中最常见的误判是
「会话其实没绑文档 → 走了通用对话（设计如此）」，而用户以为是「文档约束失效」。

`mode` / `force_retrieval` / `doc_ids` 三者必须与 `chat_service` 的**实际判定**同源，
不能各算一套 —— 否则提示会与真实行为不符，反而制造新的误判。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_docscope_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _upload(client, text: str) -> int:
    files = {"file": ("doc.md", text.encode("utf-8"), "text/markdown")}
    r = client.post("/api/v1/documents", files=files)
    assert r.status_code == 200, r.status_code
    return r.json()["id"]


def _session(client, doc_ids: list[int] | None = None) -> int:
    sid = client.post("/api/v1/sessions", json={}).json()["id"]
    if doc_ids:
        r = client.patch(f"/api/v1/sessions/{sid}", json={"config": {"doc_ids": doc_ids}})
        assert r.status_code == 200, r.text
    return sid


def _scope(client, sid: int) -> dict:
    r = client.get(f"/api/v1/sessions/{sid}/doc-scope")
    assert r.status_code == 200, r.text
    return r.json()


def test_no_docs_is_general_mode_and_does_not_force_retrieval(client):
    """★ 场景①：无文档 → `general` + **不**强制检索（这才是「像普通 API 一样对话」的依据）。"""
    d = _scope(client, _session(client))
    assert d["mode"] == "general"
    assert d["doc_count"] == 0
    assert d["doc_ids"] == []
    assert d["force_retrieval"] is False


def test_single_doc_is_single_mode_and_forces_retrieval(client):
    """场景②：单文档 → `single` + 强制检索 + 范围恰为该篇。"""
    did = _upload(client, "# 论文\n量子纠缠是量子力学中的非局域关联现象。")
    d = _scope(client, _session(client, [did]))
    assert d["mode"] == "single"
    assert d["doc_ids"] == [did]
    assert d["force_retrieval"] is True
    assert d["documents"][0]["id"] == did


def test_multi_docs_is_multi_mode_with_all_ids(client):
    """场景③：多文档未指定 → `multi` + 范围=**全部**（不是只第一篇）。"""
    d1 = _upload(client, "# 一\n量子纠缠是非局域关联现象。")
    d2 = _upload(client, "# 二\nTransformer 用自注意力实现并行建模。")
    d = _scope(client, _session(client, [d1, d2]))
    assert d["mode"] == "multi"
    assert set(d["doc_ids"]) == {d1, d2}
    assert d["force_retrieval"] is True


def test_doc_ids_are_descending_to_match_frontend_labels(client):
    """★ id 必须**降序** —— 与前端 `allDocs`（`GET /documents` 为 id desc）过滤后的
    展示顺序一致。不一致会让「@文档N」的序号在前后端指向**不同文档**（内容指向错误的根因）。"""
    ids = [_upload(client, f"# 文档{i}\n内容{i}。") for i in range(3)]
    d = _scope(client, _session(client, ids))
    assert d["doc_ids"] == sorted(ids, reverse=True)
    assert [x["id"] for x in d["documents"]] == d["doc_ids"], "documents 顺序须与 doc_ids 一致"


def test_filenames_are_returned_for_display(client):
    """篇名用于提示文案（「基于《…》回答」），必须有值。"""
    did = _upload(client, "# 论文\n量子纠缠是非局域关联现象。")
    d = _scope(client, _session(client, [did]))
    assert d["documents"][0]["filename"] == "doc.md"


def test_unresolvable_filename_falls_back_instead_of_failing(client):
    """★ 取不到篇名（如文档已被删、id 悬空）时退回 `#id` —— **展示失败不得拖垮接口**。
    否则一个陈旧配置就能让整个提示条 500。"""
    d = _scope(client, _session(client, [999999]))
    assert d["mode"] == "single"
    assert d["documents"][0]["filename"] == "#999999"


def test_scope_matches_chat_service_verdict(client):
    """★★ 与本项核心风险对应：`doc-scope` 报告的 `force_retrieval` / `doc_ids`
    必须与 `chat_service` 的**实际判定**完全一致。

    若两处各算一套，提示就会与真实行为不符 —— 那比没有提示更糟（用户会基于错误提示下结论）。"""
    d1 = _upload(client, "# 一\n内容一。")
    d2 = _upload(client, "# 二\n内容二。")
    app = client.app
    svc = app.state.chat_svc

    for ids in (None, [d1], [d1, d2]):
        sid = _session(client, ids)
        d = _scope(client, sid)
        assert d["doc_ids"] == svc._effective_doc_ids(sid)
        assert d["force_retrieval"] == svc.should_force_tool_call(sid)


def test_unknown_session_returns_404(client):
    """不存在的会话 → 404（不能静默返回 `general`，否则会把错误伪装成正常场景）。"""
    assert client.get("/api/v1/sessions/999999/doc-scope").status_code == 404
