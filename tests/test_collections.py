"""F5 · 文献集（Collections）后端测试

覆盖：
- 文献集 REST CRUD（create/list/get/patch/rename/delete）+ 文档增删（add/remove）
- CollectionService.resolve_doc_ids 的按 seq 排序与并集语义（会话查询时解析为有效文档）
- 缺失文献集返回 404

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
    tmp = tempfile.mkdtemp(prefix="papermate_coll_test_")
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


def test_collections_crud_and_docs(client):
    # 先上传两篇真实文档（CollectionDocument.document_id 受 FK 约束，须指向存在的文档）
    d1 = _upload(client, "# 文档一\n量子纠缠是量子力学中的非局域关联现象。")
    d2 = _upload(client, "# 文档二\nTransformer 通过自注意力机制实现并行序列建模。")

    # 创建（含初始文档）
    r = client.post(
        "/api/v1/collections",
        json={"name": "量子计算", "description": "课题A", "doc_ids": [d1, d2]},
    )
    assert r.status_code == 200
    body = r.json()
    cid = body["id"]
    assert body["name"] == "量子计算"
    assert set(body["doc_ids"]) == {d1, d2}

    # 列表可见
    r = client.get("/api/v1/collections")
    assert r.status_code == 200
    items = r.json()["items"]
    assert any(c["id"] == cid for c in items)

    # 详情
    r = client.get(f"/api/v1/collections/{cid}")
    assert r.status_code == 200
    assert r.json()["name"] == "量子计算"

    # 改名（PATCH）
    r = client.patch(f"/api/v1/collections/{cid}", json={"name": "量子信息", "description": "课题A-改"})
    assert r.status_code == 200
    assert r.json()["name"] == "量子信息"
    assert r.json()["description"] == "课题A-改"

    # 新增文档（再上传一篇）
    d3 = _upload(client, "# 文档三\nBell 不等式实验检验了局域实在论。")
    r = client.post(f"/api/v1/collections/{cid}/documents", json={"doc_id": d3})
    assert r.status_code == 200
    r = client.get(f"/api/v1/collections/{cid}")
    assert d3 in r.json()["doc_ids"]

    # 删除文档
    r = client.delete(f"/api/v1/collections/{cid}/documents/{d3}")
    assert r.status_code == 200
    r = client.get(f"/api/v1/collections/{cid}")
    assert d3 not in r.json()["doc_ids"]

    # 缺失文献集返回 404
    assert client.get("/api/v1/collections/999999").status_code == 404
    assert client.delete("/api/v1/collections/999999").status_code == 404

    # 删除文档 → 该文档从文献集自动解除关联（FK ondelete=CASCADE）
    r = client.delete(f"/api/v1/documents/{d1}")
    assert r.status_code == 200
    r = client.get(f"/api/v1/collections/{cid}")
    assert d1 not in r.json()["doc_ids"]

    # 删除文献集
    r = client.delete(f"/api/v1/collections/{cid}")
    assert r.status_code == 200
    assert client.get(f"/api/v1/collections/{cid}").status_code == 404


def test_collection_resolve_order_dedup_and_invalid_doc(client):
    """resolve_doc_ids 保持插入顺序；doc_ids 去重；无效 doc_id 返 400 而非 500。"""
    d1 = _upload(client, "# 一\n量子纠缠的非局域关联现象。")
    d2 = _upload(client, "# 二\nTransformer 通过自注意力机制建模。")
    d3 = _upload(client, "# 三\nBell 不等式实验检验了局域实在论。")
    svc = client.app.state.collection_svc

    # 乱序插入 → resolve 按 seq 保持插入顺序
    r = client.post("/api/v1/collections", json={"name": "顺序", "doc_ids": [d3, d1, d2]})
    assert r.status_code == 200
    cid = r.json()["id"]
    assert svc.resolve_doc_ids(cid) == [d3, d1, d2]

    # 空 / 不存在返回 []
    assert svc.resolve_doc_ids(None) == []
    assert svc.resolve_doc_ids(999999) == []

    # 重复 doc_ids 应去重（P1-5）
    r = client.post("/api/v1/collections", json={"name": "去重", "doc_ids": [d1, d1, d2]})
    assert r.status_code == 200
    assert r.json()["doc_ids"] == [d1, d2]

    # 无效 doc_id：400 明确报错，而非 IntegrityError 冒泡 500（P1-1 / P1-2）
    r = client.post("/api/v1/collections", json={"name": "坏集合", "doc_ids": [999999]})
    assert r.status_code == 400
    r = client.post(f"/api/v1/collections/{cid}/documents", json={"doc_id": 999999})
    assert r.status_code == 400

    # 追加到末尾 / 移除后保持其余顺序
    d4 = _upload(client, "# 四\n量子纠错码提升逻辑比特保真度。")
    client.post(f"/api/v1/collections/{cid}/documents", json={"doc_id": d4})
    assert svc.resolve_doc_ids(cid) == [d3, d1, d2, d4]
    client.delete(f"/api/v1/collections/{cid}/documents/{d1}")
    assert svc.resolve_doc_ids(cid) == [d3, d2, d4]


def test_collection_name_normalization(client):
    """名称归一化（P1-3/4/12）：空白→默认名、超长截断、rename 空串回落默认名。"""
    r = client.post("/api/v1/collections", json={"name": "   ", "doc_ids": []})
    assert r.status_code == 200
    assert r.json()["name"] == "未命名文献集"

    r = client.post("/api/v1/collections", json={"name": "X" * 5000, "doc_ids": []})
    assert r.status_code == 200
    assert len(r.json()["name"]) == 100

    cid = client.post("/api/v1/collections", json={"name": "原名", "doc_ids": []}).json()["id"]
    r = client.patch(f"/api/v1/collections/{cid}", json={"name": ""})
    assert r.status_code == 200
    assert r.json()["name"] == "未命名文献集"


def test_session_collection_union_effective_doc_ids(client):
    """F5：会话关联文献集后，ChatService._effective_doc_ids = 文献集文档 ∪ 手动勾选文档（并集去重）。"""
    d1 = _upload(client, "# 文档一\n量子纠缠是量子力学中的非局域关联现象。")
    d2 = _upload(client, "# 文档二\nTransformer 通过自注意力机制实现并行序列建模。")
    d3 = _upload(client, "# 文档三\nBell 不等式实验检验了局域实在论。")

    # 文献集含 d1（seq 顺序无关，仅用于并集）
    r = client.post("/api/v1/collections", json={"name": "课题X", "doc_ids": [d1]})
    cid = r.json()["id"]

    # 会话手动勾选 d2，并关联文献集 cid（含 d1）
    r = client.post(
        "/api/v1/sessions",
        json={"title": "研读课题X", "config": {"collection_id": cid, "doc_ids": [d2]}},
    )
    assert r.status_code == 200
    sid = r.json()["id"]

    chat_svc = client.app.state.chat_svc
    effective = chat_svc._effective_doc_ids(sid)
    # 并集：文献集 d1 ∪ 手动 d2
    assert set(effective) == {d1, d2}

    # 再向文献集追加 d3，并集应随之扩大
    client.post(f"/api/v1/collections/{cid}/documents", json={"doc_id": d3})
    assert set(chat_svc._effective_doc_ids(sid)) == {d1, d2, d3}

    # 解除会话关联（显式 collection_id: null 删除该键）后，仅剩手动 d2
    client.patch(f"/api/v1/sessions/{sid}", json={"config": {"collection_id": None}})
    assert chat_svc._effective_doc_ids(sid) == [d2]
