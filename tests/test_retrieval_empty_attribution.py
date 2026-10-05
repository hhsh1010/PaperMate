"""C3（2026-09-28）：**检索空命中**打点 —— 让「有文档但没命中」进入归因统计。

## 被修的缺陷

`failure.RETRIEVAL_EMPTY` 在 `core/failure.py` 里**定义了、注册了、有中文标签**，
但**全库没有任何发出点**。于是 `/metrics` 的 `failure_kinds` 里永远没有这一类 ——
而它正是「**文档在库里却答不出来**」的主因，也是最能定位检索链路的信号
（其余分类都指向"系统坏了"，只有它指向"检索算法/语料/query 的问题"）。

## 为什么不能顺手发一个 `error` 事件（两条硬理由）

1. **它不是错误**：空命中后本轮照常继续（模型可换 query 再试、或如实说"没找到"），
   发 `error` 会把"正常但没命中"报成故障，污染 `unfinished_rate` 的口径。
2. ★ **会污染我刚修好的中止口径**：路由层用 `stream_error_reason`（来自 `error` 事件）
   判定"中止轮" ⇒ 发 `error` 会让**正常轮次**被当成中止轮，
   用量/notice 全部走错分支（C2 刚统一的三处口径会再次分裂）。

⇒ 正确做法：把标记打在 **tool_result 事件**上（只进 trace，**不进 prompt**），
由 `/metrics` 按轨迹计数。

## 本文件在守什么

1. **窄口径**：`source=none`（没文档）/ `retrieval_unavailable`（系统故障）/
   `pdf_extract(page=)` 页码非法 / `pdf_extract()` 没给参数 —— **四种都不算空命中**。
   漏判任何一条，这个指标就变成噪声。
2. ★ **口径统一**：`failure_kinds` 每一格都是"**有多少条轨迹**出现了该情况"。
   空命中可以在同一轮发生多次 ⇒ 必须**按轨迹计一次**，否则同一份直方图里混了两种分母
   （本项目「两把尺子」的老账）。
3. **标记不得进入提示词**：它写在事件 data 上，而注入模型的是 `output`
   —— 若塞进 `output`，就是在**改提示词内容**（未测量的行为变更）。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.rag.pipeline import RetrievalUnavailable, RetrievedChunk
from papermate.api.routes_search import is_empty_retrieval
from papermate.models.db import Trace

DOC = "量子纠缠是量子力学中一种非局域关联现象。对其中一个粒子测量会瞬间决定另一个的状态。"


# ------------------------------------------------------------ 替身


class _EmptyPipe:
    """检索**必然空命中**的管线（有 chunks，但 query 返回空）。"""

    def get_chunks(self):  # noqa: ANN201  pdf_extract 需要
        return [RetrievedChunk(id="doc-9-c0", text="正文", page=1, score=0.5)]

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        return []


class _HitPipe(_EmptyPipe):
    """检索必然命中的管线（对照组）。"""

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        return [RetrievedChunk(id="doc-9-c1", text="命中段落", page=1, score=0.9)][:top_k]


class _BoomPipe(_EmptyPipe):
    """检索**故障**的管线（区别于空命中）。"""

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        raise RetrievalUnavailable("模拟索引损坏")


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_c3_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _prepare(client, monkeypatch, pipe) -> int:
    """建会话 + 上传文档（让 doc_ids 非空）+ 把管线换成 `pipe`。返回 session_id。"""
    doc_svc = client.app.state.doc_svc
    sid = client.post("/api/v1/sessions", json={"title": "c3"}).json()["id"]
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


def _trace_events(client, sid: int) -> list[dict]:
    with client.app.state.chat_svc._Session() as s:  # noqa: SLF001  与 routes_metrics 同一取库方式
        t = s.query(Trace).filter(Trace.session_id == sid).order_by(Trace.id.desc()).first()
        assert t is not None, "本轮没有落 trace"
        return json.loads(t.events_json or "[]")


def _metrics(client, sid: int) -> dict:
    return client.get(f"/api/v1/metrics?session_id={sid}").json()


# ------------------------------------------------------------ ① 判定谓词（窄口径）


@pytest.mark.parametrize("output,expected", [
    # ---- 算：确实是"检索了但 0 命中" ----
    ({"source": "rag", "query": "q", "results": []}, True),
    ({"source": "pdf", "query": "abstract", "results": []}, True),
    # ---- 不算：四种必须排除的情形 ----
    ({"source": "none", "query": "q", "results": []}, False),          # 未关联文档，根本没检索
    ({"source": "rag", "query": "q", "results": [],
      "error": "retrieval_unavailable"}, False),                        # 系统故障 ≠ 没命中
    ({"source": "pdf", "page": 8, "text": "", "num_chunks": 0}, False),  # 页码非法（无 results 键）
    ({"source": "pdf", "query": None, "results": []}, False),            # 没给参数 ⇒ 没发生检索
    # ---- 不算：有命中 / 结构不符 ----
    ({"source": "rag", "query": "q", "results": [{"id": "c1"}]}, False),
    ({"source": "rag", "query": "q"}, False),
    (None, False),
    ("字符串", False),
])
def test_is_empty_retrieval_narrow_contract(output, expected):
    """★ 窄口径是这函数的全部价值：四种"看起来像但都不是"的情形必须排除。"""
    assert is_empty_retrieval(output) is expected


# ------------------------------------------------------------ ② 端到端：打点 + 统计


def test_empty_retrieval_is_marked_on_event_and_counted_by_metrics(client, monkeypatch):
    """★ 核心：空命中 → trace 事件带标记 → `/metrics` 出现 `retrieval_empty`。"""
    sid = _prepare(client, monkeypatch, _EmptyPipe())

    _ask(client, sid)

    events = _trace_events(client, sid)
    marked = [e for e in events
              if e.get("type") == "tool_result"
              and (e.get("data") or {}).get("retrieval_empty") is True]
    assert marked, f"空命中未打点：{[e.get('type') for e in events]}"

    m = _metrics(client, sid)
    assert m["failure_kinds"].get("retrieval_empty") == 1, m["failure_kinds"]
    # 它属于 REAL_FAILURES ⇒ 也要出现在"排除取消"的那份里
    assert m["failure_kinds_excluding_cancel"].get("retrieval_empty") == 1


def test_marker_is_not_injected_into_the_prompt(client, monkeypatch):
    """★ 标记只进 trace、**不进 prompt** —— 否则就是在改提示词（未测量的行为变更）。

    证据：标记写在**事件 data 层**；注入模型的内容来自 `output`
    （`loop._format_tool_result(deduped)`）。这里直接验证 `output` 里没有被污染。
    """
    sid = _prepare(client, monkeypatch, _EmptyPipe())

    _ask(client, sid)

    outs = [(e.get("data") or {}).get("output") for e in _trace_events(client, sid)
            if e.get("type") == "tool_result"]
    assert outs, "没有 tool_result 事件"
    for out in outs:
        assert isinstance(out, dict)
        assert "retrieval_empty" not in out, f"标记漏进了 output（会被注入 prompt）：{out}"


def test_hit_retrieval_is_not_marked(client, monkeypatch):
    """反向自检：**有命中**不得被标记，指标里也不该出现（防"永远在报"）。"""
    sid = _prepare(client, monkeypatch, _HitPipe())

    _ask(client, sid)

    assert not [e for e in _trace_events(client, sid)
                if (e.get("data") or {}).get("retrieval_empty") is True]
    assert _metrics(client, sid)["failure_kinds"].get("retrieval_empty") is None


def test_retrieval_fault_is_not_counted_as_empty(client, monkeypatch):
    """★ 反向自检：**检索故障**（`retrieval_unavailable`）不得被算成"空命中"。

    两者对用户的意义完全不同（"该重建索引" vs "换个问法"），F7 专门区分过它们 ——
    指标若混为一谈，就把"系统坏了"报成了"检索不到内容"。
    """
    sid = _prepare(client, monkeypatch, _BoomPipe())

    _ask(client, sid)

    assert not [e for e in _trace_events(client, sid)
                if (e.get("data") or {}).get("retrieval_empty") is True]
    assert _metrics(client, sid)["failure_kinds"].get("retrieval_empty") is None


# ------------------------------------------------------------ ③ 口径：按轨迹计数


def test_multiple_empty_retrievals_in_one_trace_count_once(client):
    """★★ 口径统一：一轮里空命中 3 次 → 直方图里计 **1**（不是 3）。

    `failure_kinds` 每一格都是"有多少条**轨迹**出现了该情况"（`error` 事件必然终止本轮 ⇒
    每轨迹至多 1 个）。空命中可以在同一轮发生多次，若按**次数**累加，
    同一份直方图里就混了两种分母 —— 本项目「两把尺子」的老账。
    """
    sid = client.post("/api/v1/sessions", json={"title": "c3-count"}).json()["id"]
    events = [
        {"idx": i, "type": "tool_result",
         "data": {"ok": True, "retrieval_empty": True, "output": {"source": "rag", "results": []}}}
        for i in range(3)
    ]
    with client.app.state.chat_svc._Session() as s:  # noqa: SLF001
        s.add(Trace(trace_id="t-c3", session_id=sid, events_json=json.dumps(events)))
        s.commit()

    m = _metrics(client, sid)

    assert m["failure_kinds"].get("retrieval_empty") == 1, (
        f"按次数累加了（应为按轨迹计 1）：{m['failure_kinds']}"
    )
