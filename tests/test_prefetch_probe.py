"""O-12 档 2 · P0 探针（2026-10-02）：命中率测量。

**P0 的设计取向**：不真的发起预取检索 ⇒ **零 GPU 成本、零行为变更**，只写一行日志。
因此它的守护重点是两件事：
1. **判据本身正确**（`prefetch_key` 严格相等；不得退化成"相似即命中" —— 假命中会偷换证据）；
2. **零痕迹**：开关关着时**什么都不做**（连日志都没有）；开着时**不产生任何新 SSE 事件**、
   不改任何既有载荷（否则"零行为变更"就是空话）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from papermate.services import prefetch_probe

DOC = "量子纠缠是量子力学中一种非局域关联现象。\n\n两个粒子发生纠缠后，对其中一个测量会瞬间决定另一个的状态。"


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_prefetch_probe_test_")
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


def _upload(client, text: str, session_id: int, *, allow_duplicate: bool = False) -> dict:
    files = {"file": ("doc.md", text.encode("utf-8"), "text/markdown")}
    url = f"/api/v1/documents?session_id={session_id}"
    if allow_duplicate:
        url += "&allow_duplicate=true"
    r = client.post(url, files=files)
    assert r.status_code == 200, r.text
    return r.json()


# ============ ① 判据（单一来源，P0/P1 共用） ============

class TestPrefetchKey:
    def test_whitespace_and_ascii_case_are_normalized(self):
        """去空白 + ASCII 小写 ⇒ 语义相同的写法必须判为同一 query。"""
        assert prefetch_probe.prefetch_key("注意力 机制") == prefetch_probe.prefetch_key("注意力机制")
        assert prefetch_probe.prefetch_key("  Abstract\n提取 ") == prefetch_probe.prefetch_key("abstract提取")

    def test_content_difference_is_never_a_hit(self):
        """★ 质量红线：内容不同一律不命中 —— 绝不做"包含/相似"这类模糊匹配。

        假命中的代价是**偷换证据**（把预取的证据当模型要的证据），比低命中率严重得多。
        """
        k = prefetch_probe.prefetch_key
        assert k("注意力机制") != k("自注意力 复杂度")
        assert k("注意力机制") != k("注意力机制是什么")   # 前缀包含 ≠ 相等
        assert k("摘要") != k("摘要讲了什么")

    def test_disabled_by_default(self, monkeypatch):
        """开关默认关（实验性观测，且日志会打印 query 文本）。"""
        monkeypatch.delenv(prefetch_probe.PROBE_ENV, raising=False)
        assert prefetch_probe.probe_enabled() is False
        monkeypatch.setenv(prefetch_probe.PROBE_ENV, "1")
        assert prefetch_probe.probe_enabled() is True

    def test_log_probe_marks_hit_and_miss(self, caplog):
        caplog.set_level(logging.INFO, logger="papermate.services.prefetch_probe")
        prefetch_probe.log_probe(user_query="注意力机制", model_query="注意力机制",
                                 t_first_retrieve_ms=1748.0)
        prefetch_probe.log_probe(user_query="注意力机制", model_query="自注意力复杂度",
                                 t_first_retrieve_ms=1500.0)
        prefetch_probe.log_probe(user_query="注意力机制", model_query=None,
                                 t_first_retrieve_ms=None)
        # ★★ 关键用例：**包含**成立但**相等**不成立 —— 判据若被放宽成 `in` / 相似匹配，
        #    这条会立刻变红（模型把问题扩写成更长的 query 是常见行为，绝不能算命中：
        #    那等于把它要的证据换成预取的那批）。
        prefetch_probe.log_probe(user_query="摘要", model_query="摘要讲了什么",
                                 t_first_retrieve_ms=1600.0)
        lines = [r.getMessage() for r in caplog.records if "[预取探针]" in r.getMessage()]
        assert len(lines) == 4, lines
        assert "命中=是" in lines[0]
        assert "命中=否" in lines[1] and "t_first_retrieve=1500ms" in lines[1]
        # 本轮完全没有 retrieve 的情况也要记（它计入分母：不可能命中）
        assert "命中=否" in lines[2] and "无 retrieve" in lines[2]
        assert "命中=否" in lines[3], f"前缀包含 ≠ 命中（判据被放宽？）：{lines[3]}"


# ============ ② 端到端：开着才有日志；且**零痕迹** ============

def test_probe_logs_hit_rate_on_doc_bound(client, monkeypatch, caplog):
    """doc-bound 且开关开启 ⇒ 必须有一行探针日志（含两侧 query 便于人工核对）。"""
    monkeypatch.setenv(prefetch_probe.PROBE_ENV, "1")
    caplog.set_level(logging.INFO)
    s = client.post("/api/v1/sessions", json={"title": "probe-on"}).json()
    assert _upload(client, DOC, s["id"])["status"] == "ready"

    _stream(client, s["id"], "文档里提到的量子纠缠是什么？")
    lines = [r.getMessage() for r in caplog.records if "[预取探针]" in r.getMessage()]
    assert lines, "开关开启时 doc-bound 必须打探针日志"
    assert "命中=" in lines[0] and "t_first_retrieve=" in lines[0]
    # FakeLLM 在 doc-bound 剧本下会**把用户问句原样**传给 retrieve ⇒ 这里应判「命中」，
    # 正好证明命中路径可被观测到。（"不同 query 不命中"由 TestPrefetchKey 的单元层守护。）
    assert "命中=是" in lines[0], f"同 query 应判命中：{lines[0]}"
    assert "文档里提到的量子纠缠是什么？" in lines[0], "日志必须带两侧 query 供人工核对"


def test_probe_is_silent_when_disabled(client, monkeypatch, caplog):
    """★ 开关关着 ⇒ 什么都不做（连日志都没有）。"""
    monkeypatch.delenv(prefetch_probe.PROBE_ENV, raising=False)
    caplog.set_level(logging.INFO)
    s = client.post("/api/v1/sessions", json={"title": "probe-off"}).json()
    assert _upload(client, DOC, s["id"])["status"] == "ready"

    _stream(client, s["id"], "文档里提到的量子纠缠是什么？")
    assert not [r for r in caplog.records if "[预取探针]" in r.getMessage()], "关着还打日志"


def test_probe_does_not_change_stream_payload(client, monkeypatch, caplog):
    """★★ 零行为变更：开与关的 SSE 事件序列/载荷必须**逐字一致**（除日志外无任何差异）。"""
    monkeypatch.setenv(prefetch_probe.PROBE_ENV, "1")
    s_on = client.post("/api/v1/sessions", json={"title": "on"}).json()
    assert _upload(client, DOC, s_on["id"])["status"] == "ready"
    events_on = _stream(client, s_on["id"], "文档里提到的量子纠缠是什么？")

    monkeypatch.delenv(prefetch_probe.PROBE_ENV, raising=False)
    s_off = client.post("/api/v1/sessions", json={"title": "off"}).json()
    # 同一份内容再上传会被去重拦截（409）⇒ 显式声明允许重复
    assert _upload(client, DOC, s_off["id"], allow_duplicate=True)["status"] == "ready"
    events_off = _stream(client, s_off["id"], "文档里提到的量子纠缠是什么？")

    assert [n for n, _ in events_on] == [n for n, _ in events_off], "事件序列不得因探针而变"
    # 载荷层面：探针不得新增/改写任何字段。
    # ⚠️ 排除两类**与探针无关**的易变项：① 时间/用量类字段；② 文档 id（两个会话各上传一份
    #    同内容文档 ⇒ `doc-1-c*` vs `doc-2-c*`）⇒ 比对前把 `doc-\d+` 归一化，否则会误报。
    volatile = {"session_id", "trace_id", "created_at", "latency_ms", "ttft_ms",
                "total_tokens", "cost", "steps"}

    def _norm(obj):
        return re.sub(r"doc-\d+", "doc-N", json.dumps(obj, ensure_ascii=False, sort_keys=True))

    for (n1, d1), (_n2, d2) in zip(events_on, events_off, strict=True):
        for key in set(d1) | set(d2):
            if key in volatile:
                continue
            assert _norm(d1.get(key)) == _norm(d2.get(key)), (
                f"{n1}.{key} 被探针改变了：{d1.get(key)!r} vs {d2.get(key)!r}"
            )
