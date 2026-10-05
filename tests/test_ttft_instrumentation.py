"""O-12（2026-10-02）：服务端 TTFT（首字延迟）打点回归。

**为什么必须守着**：TTFT 是本项目"用户体验"的第一指标，但它的采集点只有一个 ——
`routes_search._with_stream_cleanup` 里对首个 `message_delta` 的埋点。一旦有人重写那段包装
（或把埋点挪进某个具体路径），**不会报错、只是数据永久为空**（静默失效），
而恰恰是"没有数据"会让后续的档 1/档 2 优化**无法验收**。

断言分三层：
1. **行为层**：真实跑一次流式请求（FakeLLM），`done` 载荷里必须有 `ttft_ms`，
   且它 ≤ 整轮墙钟、≥ 0（口径自洽）；
2. **口径层**：`ttft_ms` 必须发生在**首个 `message_delta` 之后**才可能被采集
   —— 即"没有 message_delta 的流不得报出 ttft"（防"用整轮耗时冒充首字"）；
3. **源码契约层**：埋点位置必须在 `_with_stream_cleanup` 内（唯一出口），
   而不是散落在各业务路径里（否则新路径会漏测）。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod

DOC = "量子纠缠是量子力学中一种非局域关联现象。\n\n两个粒子发生纠缠后，对其中一个测量会瞬间决定另一个的状态。"


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_ttft_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    cur_name, cur_data = None, ""
    for line in text.splitlines():
        if line.startswith("event:"):
            cur_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            cur_data = line[len("data:"):].strip()
            if cur_name is not None:
                events.append((cur_name, json.loads(cur_data)))
                cur_name, cur_data = None, ""
    return events


def _stream(client, session_id: int, content: str) -> list[tuple[str, dict]]:
    r = client.post(f"/api/v1/sessions/{session_id}/messages/stream", json={"content": content})
    assert r.status_code == 200, r.text
    return _parse_sse(r.text)


def _upload(client, text: str, session_id: int) -> dict:
    files = {"file": ("doc.md", text.encode("utf-8"), "text/markdown")}
    r = client.post(f"/api/v1/documents?session_id={session_id}", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def test_done_payload_carries_ttft_on_doc_bound_path(client):
    """doc-bound 路径（带检索）：done 必须有 ttft_ms，且口径自洽。"""
    s = client.post("/api/v1/sessions", json={"title": "ttft-doc"}).json()
    assert _upload(client, DOC, s["id"])["status"] == "ready"

    events = _stream(client, s["id"], "文档里提到的量子纠缠是什么？")
    names = [n for n, _ in events]
    assert "message_delta" in names, "前置条件：本轮必须真的产生了正文增量"
    done = next(d for n, d in events if n == "done")

    assert "ttft_ms" in done, f"done 载荷缺少 ttft_ms（O-12 打点失效）：{sorted(done)}"
    ttft = done["ttft_ms"]
    assert isinstance(ttft, int) and ttft >= 0, f"ttft_ms 必须是非负整数，实际 {ttft!r}"
    # 口径自洽：首字不可能晚于整轮（latency_ms 是本轮实测耗时；为 0 时跳过该断言）
    latency = done.get("latency_ms")
    if isinstance(latency, int) and latency > 0:
        assert ttft <= latency, f"首字 {ttft}ms 不得晚于整轮 {latency}ms（口径矛盾）"


def test_done_payload_carries_ttft_on_direct_path(client):
    """直答路径（无文档、无检索）：同样必须上报 —— 埋点在唯一出口，四条路径共享。"""
    s = client.post("/api/v1/sessions", json={"title": "ttft-direct"}).json()
    events = _stream(client, s["id"], "什么是注意力机制？")
    names = [n for n, _ in events]
    assert "message_delta" in names
    done = next(d for n, d in events if n == "done")
    assert isinstance(done.get("ttft_ms"), int), f"直答路径的 done 缺 ttft_ms：{sorted(done)}"


def test_ttft_not_fabricated_when_no_delta():
    """口径层：**没有正文增量**的流不得报出 ttft（防"用整轮耗时冒充首字"）。

    用最小可复现的方式验证：直接构造"只发 done、不发 message_delta"的场景不现实，
    改为断言源码契约 —— `ttft_ms` 的赋值点必须在"见过 message_delta"的条件内。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py").read_text(
        encoding="utf-8")
    # 埋点必须与 message_delta 判定**同一个条件块**内（会被执行的那段）
    assert 'if "first_delta_ms" not in _ttft and "event: message_delta" in chunk:' in src, (
        "TTFT 埋点缺少『首个 message_delta』判定 —— 可能被改成无条件赋值（会把整轮耗时当首字）"
    )


def test_ttft_probe_lives_in_single_exit_wrapper():
    """源码契约层：埋点必须在 `_with_stream_cleanup`（唯一出口）内。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py").read_text(
        encoding="utf-8")
    start = src.index("async def _with_stream_cleanup(agen):")
    end = src.index("async def gen():", start)
    body = src[start:end]
    assert "first_delta_ms" in body, "埋点不在唯一出口内 ⇒ 新增路径会漏采（静默失效）"
    # 且三条 done 出口都要上报（agent 主路径在 done_data 上、直答/技能/兜底各自显式）
    assert "ttft_ms" in src
    assert src.count("ttft_ms") >= 3, "至少 3 处 done 出口都要带上 ttft_ms"
