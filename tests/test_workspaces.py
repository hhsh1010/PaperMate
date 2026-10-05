"""A1 · 工作间后端基座测试

覆盖：
- POST /workspaces 原子创建（新建文献集 + 工作间会话）
- 加入已有文献集（collection_id 优先，忽略传入 doc_ids）
- 草稿校验：须在文献集内 / 须存在
- 无效 doc_ids / 不存在的 collection_id / 非法 mode → 400 / 404
- GET /workspaces/{session_id} 从 config_json 解析（含非工作间会话 → 404）
- CollectionService.draft_ids / ref_ids 解析语义

数据层纯本地（SQLite），不触发大模型 / 网络，可离线运行。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_ws_test_")
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


def _make_workspace(client, **kw) -> dict:
    r = client.post("/api/v1/workspaces", json=kw)
    assert r.status_code == 200, (r.status_code, r.text)
    return r.json()


def test_create_workspace_new_collection(client):
    d1 = _upload(client, "# 一\n量子纠缠的非局域关联现象。")
    d2 = _upload(client, "# 二\nTransformer 通过自注意力机制建模。")

    out = _make_workspace(client, name="课题X", description="p", doc_ids=[d1, d2], mode="research")
    assert out["name"] == "课题X"
    assert out["mode"] == "research"
    assert out["collection_id"] is not None
    assert set(out["doc_ids"]) == {d1, d2}
    assert out["draft_doc_id"] is None
    sid = out["session_id"]

    # 原子创建后：底层文献集确实存在
    assert client.get(f"/api/v1/collections/{out['collection_id']}").status_code == 200

    # 会话 config_json 正确承载工作间元信息
    s = client.get(f"/api/v1/sessions/{sid}").json()
    assert s["config"]["workspace"] is True
    assert s["config"]["collection_id"] == out["collection_id"]
    assert s["config"]["mode"] == "research"

    # GET /workspaces/{sid} 可解析
    ws = client.get(f"/api/v1/workspaces/{sid}").json()
    assert ws["session_id"] == sid
    assert ws["collection_id"] == out["collection_id"]
    assert set(ws["doc_ids"]) == {d1, d2}


def test_create_workspace_with_draft(client):
    d1 = _upload(client, "# 草稿\n我的论文草稿内容。")
    d2 = _upload(client, "# 参考\n参考文献一。")
    d3 = _upload(client, "# 参考二\n参考文献二。")

    out = _make_workspace(client, name="改稿间", doc_ids=[d1, d2, d3], draft_doc_id=d1, mode="revision")
    assert out["mode"] == "revision"
    assert out["draft_doc_id"] == d1

    # draft_ids / ref_ids 解析
    col = client.app.state.collection_svc
    assert col.draft_ids(out["collection_id"], d1) == [d1]
    assert col.ref_ids(out["collection_id"], d1) == [d2, d3]


def test_join_existing_collection_with_custom_name(client):
    """B1：关联已有文献集时可自定义工作间名（不再强制沿用文献集名），且文献集本身不被改名。"""
    d1 = _upload(client, "# 一\n内容一。")
    cid = client.post("/api/v1/collections", json={"name": "已有集", "doc_ids": [d1]}).json()["id"]

    out = _make_workspace(client, name="课题X-改稿", collection_id=cid)
    assert out["collection_id"] == cid
    assert out["name"] == "课题X-改稿"
    # 文献集原名保持不变（工作间名是会话侧概念，不污染底层数据）
    assert client.get(f"/api/v1/collections/{cid}").json()["name"] == "已有集"


def test_join_existing_collection_ignores_doc_ids(client):
    d1 = _upload(client, "# 一\n内容一。")
    d2 = _upload(client, "# 二\n内容二。")
    cid = client.post("/api/v1/collections", json={"name": "已有集", "doc_ids": [d1, d2]}).json()["id"]

    # 传入的 doc_ids 应被忽略，以文献集为准
    out = _make_workspace(client, collection_id=cid, doc_ids=[999999])
    assert out["collection_id"] == cid
    assert set(out["doc_ids"]) == {d1, d2}
    assert out["name"] == "已有集"  # 未显式命名 → 沿用文献集名


def test_invalid_doc_ids_400(client):
    r = client.post("/api/v1/workspaces", json={"name": "坏", "doc_ids": [999999]})
    assert r.status_code == 400


def test_missing_collection_404(client):
    d1 = _upload(client, "# 一\n内容。")
    r = client.post("/api/v1/workspaces", json={"name": "x", "collection_id": 999999, "draft_doc_id": d1})
    assert r.status_code == 404


def test_draft_not_in_collection_400(client):
    d1 = _upload(client, "# 草稿\n草稿内容。")
    d2 = _upload(client, "# 其他\n其他内容。")
    out = _make_workspace(client, name="集", doc_ids=[d2])  # 文献集只含 d2
    # 草稿 d1 不在文献集内 → 400
    r = client.post(
        "/api/v1/workspaces",
        json={"name": "bad", "collection_id": out["collection_id"], "draft_doc_id": d1},
    )
    assert r.status_code == 400


def test_draft_doc_not_exist_400(client):
    d1 = _upload(client, "# 一\n内容。")
    r = client.post("/api/v1/workspaces", json={"name": "x", "doc_ids": [d1], "draft_doc_id": 999999})
    assert r.status_code == 400


def test_invalid_mode_400(client):
    d1 = _upload(client, "# 一\n内容。")
    r = client.post("/api/v1/workspaces", json={"name": "x", "doc_ids": [d1], "mode": "nonsense"})
    assert r.status_code == 400


def test_get_workspace_not_a_workspace_404(client):
    # 普通会话（非工作间）→ 404
    sid = client.post("/api/v1/sessions", json={"title": "普通"}).json()["id"]
    assert client.get(f"/api/v1/workspaces/{sid}").status_code == 404
    # 不存在的会话 → 404
    assert client.get("/api/v1/workspaces/999999").status_code == 404


def test_list_sessions_carries_workspace_flag(client):
    """A2 前端左栏分组的数据来源：GET /sessions 列表必须携带 config.workspace。

    A1 只验证了详情端点，而前端分组读的是列表端点 —— 此处补回归保护，
    防止后续改动 _to_out 时漏掉 config，导致工作间静默掉出分组。
    """
    d1 = _upload(client, "# 一\n内容。")
    ws = _make_workspace(client, name="课题X", doc_ids=[d1])
    _ = client.post("/api/v1/sessions", json={"title": "普通会话"}).json()

    rows = client.get("/api/v1/sessions").json()
    by_id = {r["id"]: r for r in rows}
    assert ws["session_id"] in by_id
    assert by_id[ws["session_id"]]["config"]["workspace"] is True
    assert by_id[ws["session_id"]]["config"]["collection_id"] == ws["collection_id"]

    normals = [r for r in rows if r["config"].get("workspace") is not True]
    assert len(normals) >= 1  # 普通会话不被误判为工作间


# ---------------- B1：加入工作间 ----------------

def test_add_documents_to_workspace(client):
    d1 = _upload(client, "# 一\n内容一。")
    d2 = _upload(client, "# 二\n内容二。")
    d3 = _upload(client, "# 三\n内容三。")
    ws = _make_workspace(client, name="课题X", doc_ids=[d1])
    sid = ws["session_id"]

    r = client.post(f"/api/v1/workspaces/{sid}/documents", json={"doc_ids": [d2, d3]})
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json()["doc_ids"] == [d1, d2, d3]  # 按 seq 排序，新文档接在后面


def test_add_documents_idempotent(client):
    """幂等：重复加入已在集内的文档不产生重复行，也不改其原有顺序。"""
    d1 = _upload(client, "# 一\n内容一。")
    d2 = _upload(client, "# 二\n内容二。")
    ws = _make_workspace(client, name="课题X", doc_ids=[d1, d2])
    sid = ws["session_id"]

    out = client.post(f"/api/v1/workspaces/{sid}/documents", json={"doc_ids": [d1, d2]}).json()
    assert out["doc_ids"] == [d1, d2]  # 无重复、顺序不变

    # 混合：已存在 + 新增 → 只追加新增部分
    d3 = _upload(client, "# 三\n内容三。")
    out = client.post(f"/api/v1/workspaces/{sid}/documents", json={"doc_ids": [d1, d3]}).json()
    assert out["doc_ids"] == [d1, d2, d3]


def test_add_documents_empty_400(client):
    d1 = _upload(client, "# 一\n内容一。")
    sid = _make_workspace(client, name="x", doc_ids=[d1])["session_id"]
    assert client.post(f"/api/v1/workspaces/{sid}/documents", json={"doc_ids": []}).status_code == 400


def test_add_documents_missing_doc_400(client):
    d1 = _upload(client, "# 一\n内容一。")
    sid = _make_workspace(client, name="x", doc_ids=[d1])["session_id"]
    r = client.post(f"/api/v1/workspaces/{sid}/documents", json={"doc_ids": [999999]})
    assert r.status_code == 400
    # 失败不应污染已有文献集
    assert client.get(f"/api/v1/workspaces/{sid}").json()["doc_ids"] == [d1]


def test_add_documents_not_a_workspace_404(client):
    # 普通会话 → 404
    sid = client.post("/api/v1/sessions", json={"title": "普通"}).json()["id"]
    r = client.post(f"/api/v1/workspaces/{sid}/documents", json={"doc_ids": [1]})
    assert r.status_code == 404
    # 不存在的会话 → 404
    assert client.post("/api/v1/workspaces/999999/documents", json={"doc_ids": [1]}).status_code == 404


# ---------------- B1 反向：从工作间移除文档（DELETE /workspaces/{sid}/documents/{doc_id}） ----------------

def test_remove_document_from_workspace(client):
    d1 = _upload(client, "# 一\n内容一。")
    d2 = _upload(client, "# 二\n内容二。")
    d3 = _upload(client, "# 三\n内容三。")
    ws = _make_workspace(client, name="课题X", doc_ids=[d1, d2, d3])
    sid = ws["session_id"]

    # 移除中间文档 d2
    r = client.delete(f"/api/v1/workspaces/{sid}/documents/{d2}")
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json()["doc_ids"] == [d1, d3]

    # 源 Document 本身不应被删除（仅解绑）
    assert client.get(f"/api/v1/documents/{d2}").status_code == 200

    # 再移除 d1 → 只剩 d3
    r = client.delete(f"/api/v1/workspaces/{sid}/documents/{d1}")
    assert r.status_code == 200
    assert r.json()["doc_ids"] == [d3]

    # 幂等：再次移除已不在集内的 d1 → 404（无可移除项）
    assert client.delete(f"/api/v1/workspaces/{sid}/documents/{d1}").status_code == 404


def test_remove_document_not_a_workspace_404(client):
    # 普通会话 → 404
    sid = client.post("/api/v1/sessions", json={"title": "普通"}).json()["id"]
    assert client.delete(f"/api/v1/workspaces/{sid}/documents/1").status_code == 404
    # 不存在的会话 → 404
    assert client.delete("/api/v1/workspaces/999999/documents/1").status_code == 404


def test_remove_draft_clears_draft_flag(client):
    """移除主文献（draft_doc_id）后，draft_doc_id 应被同步清除，避免悬空引用。"""
    d1 = _upload(client, "# 草稿\n草稿内容。")
    d2 = _upload(client, "# 参考\n参考文献。")
    ws = _make_workspace(client, name="改稿间", doc_ids=[d1, d2], draft_doc_id=d1, mode="revision")
    sid = ws["session_id"]
    assert ws["draft_doc_id"] == d1

    r = client.delete(f"/api/v1/workspaces/{sid}/documents/{d1}")
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json()["draft_doc_id"] is None
    assert r.json()["doc_ids"] == [d2]

    # 会话 config_json 中 draft_doc_id 已清空
    s = client.get(f"/api/v1/sessions/{sid}").json()
    assert s["config"].get("draft_doc_id") is None


# ---------------- B2：主文献指定（PATCH /workspaces/{sid}） ----------------

def test_patch_workspace_set_and_clear_draft(client):
    d1 = _upload(client, "# 草稿\n草稿内容。")
    d2 = _upload(client, "# 参考\n参考文献。")
    sid = _make_workspace(client, name="改稿间", doc_ids=[d1, d2])["session_id"]

    # 设置主文献
    out = client.patch(f"/api/v1/workspaces/{sid}", json={"draft_doc_id": d1}).json()
    assert out["draft_doc_id"] == d1
    assert client.get(f"/api/v1/workspaces/{sid}").json()["draft_doc_id"] == d1

    # 更换主文献
    out = client.patch(f"/api/v1/workspaces/{sid}", json={"draft_doc_id": d2}).json()
    assert out["draft_doc_id"] == d2

    # 显式 null → 清除（与「未传」区分）
    out = client.patch(f"/api/v1/workspaces/{sid}", json={"draft_doc_id": None}).json()
    assert out["draft_doc_id"] is None


def test_patch_workspace_empty_patch_noop(client):
    """空 patch `{}` 不改任何字段（mode 落库值保持默认，draft 保持 None）。"""
    d1 = _upload(client, "# 一\n内容。")
    sid = _make_workspace(client, name="x", doc_ids=[d1], draft_doc_id=d1, mode="research")["session_id"]
    out = client.patch(f"/api/v1/workspaces/{sid}", json={}).json()
    assert out["draft_doc_id"] == d1
    assert out["mode"] == "research"


def test_patch_workspace_draft_not_in_collection_400(client):
    d1 = _upload(client, "# 一\n内容。")
    d2 = _upload(client, "# 二\n其他文档。")
    sid = _make_workspace(client, name="x", doc_ids=[d1])["session_id"]
    # d2 存在但不在文献集内
    r = client.patch(f"/api/v1/workspaces/{sid}", json={"draft_doc_id": d2})
    assert r.status_code == 400
    assert "文献集" in r.json()["detail"]
    # 不存在的文档
    assert client.patch(f"/api/v1/workspaces/{sid}", json={"draft_doc_id": 999999}).status_code == 400
    # 失败不污染原配置
    assert client.get(f"/api/v1/workspaces/{sid}").json()["draft_doc_id"] is None


def test_patch_workspace_invalid_mode_400_and_mode_update_ok(client):
    d1 = _upload(client, "# 一\n内容。")
    sid = _make_workspace(client, name="x", doc_ids=[d1])["session_id"]
    assert client.patch(f"/api/v1/workspaces/{sid}", json={"mode": "nonsense"}).status_code == 400
    out = client.patch(f"/api/v1/workspaces/{sid}", json={"mode": "revision"}).json()
    assert out["mode"] == "revision"


def test_patch_workspace_not_a_workspace_404(client):
    sid = client.post("/api/v1/sessions", json={"title": "普通"}).json()["id"]
    assert client.patch(f"/api/v1/workspaces/{sid}", json={"mode": "research"}).status_code == 404
    assert client.patch("/api/v1/workspaces/999999", json={}).status_code == 404


def test_draft_ref_resolution_empty(client):
    d1 = _upload(client, "# 一\n内容。")
    cid = client.post("/api/v1/collections", json={"name": "集", "doc_ids": [d1]}).json()["id"]
    col = client.app.state.collection_svc
    # 未指定草稿 → draft_ids=[], ref_ids=全部
    assert col.draft_ids(cid, None) == []
    assert col.ref_ids(cid, None) == [d1]
    # 草稿不存在于文献集 → draft_ids=[]
    assert col.draft_ids(cid, 999999) == []
