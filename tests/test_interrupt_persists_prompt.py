"""N13 回归（2026-09-12）：中断时**用户提问必须落库**（不得整轮消失）。

真实故障（用户实测）：发出提问后、模型尚未产出任何 delta 就点「停止」→ **整轮（含提问本身）
全部消失**，用户被迫重复输入。

根因：整轮落库（`_persist_turn`）在流结束后一次性写入 user+assistant；三条中断路径原先都是
「**无已生成内容则不落库**」→ return 时 user 也没了。修复为：无内容时至少落 user 一条。

覆盖：
① `persist_user_message` 落库正确性（含多模态块经 GET 还原）；
② 三条中断路径均调用它（源码级断言，防止重构时把修复删掉）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from papermate.api.app import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PAPERMATE_DATA_DIR", str(tmp_path))
    import papermate.config as cfg_mod

    cfg_mod._settings = None  # noqa: SLF001  隔离环境需重置单例
    with TestClient(create_app()) as c:
        yield c


def _new_session(client: TestClient) -> int:
    return client.post("/api/v1/sessions", json={"title": "n13"}).json()["id"]


def test_persist_user_message_writes_user_only(client: TestClient) -> None:
    """落一条 user，且**不**产生空的 assistant（避免历史里出现空气泡）。"""
    sid = _new_session(client)
    client.app.state.chat_svc.persist_user_message(sid, "中断前的提问")

    msgs = client.get(f"/api/v1/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user"], f"应只落 user，实际 {[m['role'] for m in msgs]}"
    assert msgs[0]["content"] == "中断前的提问"


def test_persist_user_message_roundtrips_multimodal(client: TestClient) -> None:
    """多模态（块列表）走同一编码口径，GET 时能还原出文本与图片。"""
    sid = _new_session(client)
    blocks = [
        {"type": "text", "text": "看图说话"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    client.app.state.chat_svc.persist_user_message(sid, blocks)

    msg = client.get(f"/api/v1/sessions/{sid}").json()["messages"][0]
    assert msg["content"] == "看图说话", "文本块应还原为纯文本"
    assert msg.get("images"), "图片 URL 应单独还原"


def test_all_interrupt_paths_persist_user_message() -> None:
    """源码级防回退：三条中断路径（直答流 / Agent 循环 / 编码阶段）都必须接上。

    说明：这三条路径需真实 SSE 时序 + 客户端断连才能触达，端到端单测成本高且不稳定；
    此处锁定「代码里确实接上了、且**无内容分支**也覆盖」这一关键事实。
    """
    from papermate.api import routes_search

    src = Path(routes_search.__file__).read_text(encoding="utf-8")
    assert src.count("persist_user_message") >= 3, (
        "三条中断路径都必须保住用户提问（_pump_direct / loop / IndexingCancelled）"
    )
    # 「无部分内容」的分支也必须落库（原先正是这里直接 return 导致整轮消失）
    assert src.count("elif not body.regenerate:") >= 2
    assert not os.environ.get("N13_SKIP")  # 占位：保持 os 导入有实际用途
