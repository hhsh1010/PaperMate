"""A8/C8（2026-09-28）：**超长输入前置拦截** —— 把"熔断兜底"换成"用户可自救"。

## 被修的缺陷

`ChatRequest.content` 与设置页的 `agent_rules` **都没有长度上限**。

后果不是"报错"，而是**体验与成本双输**：`max_tokens_budget` 的判据在
**响应返回之后**才执行（`loop._run_stream_impl`）⇒ 一次超长粘贴的流程是

```
发出请求 → 模型调用并计费 → 才发现 prompt 超限 → 中止 → 用户看到「本次没有生成有效回答」
```

**钱已经花了，而兜底文案里没有一句提到"输入太长"** —— 用户完全无法自救
（不知道是文档问题、模型问题还是输入问题）。

## 本文件在守什么

1. ★ **阈值口径必须与预算体系同源**：用 `_rough_tokens`（与 `[上下文构成]`/闸门同一把尺子）
   而不是 `len()`。本文件有一条**判别样本**专门守它（某个 `len` 超限但 `tokens` 不超限的
   合法长文**不得**被拦）—— 换成 `len()` 实现会立刻变红。
2. ★ **上限由当前生效的 `max_tokens_budget` 派生**，不是写死的数字
   （同 `compact_trigger_tokens` 的"派生而非新配一项"）。
3. ★ **拦截路径不得发生任何 LLM 工作**（`build_agent` 一次都不能被调用）——
   否则"省下额度"就只是口号。
4. ★ **拦下也要留痕**：用户提问照样落库、`meta.notice` 照样写
   ⇒ 刷新后警示卡仍在（与「缺文档」拦截同一契约）。
5. **长期规则比用户输入更严**：它每轮都注入，成本是"每次请求都要付"。

## ⚠️ 用例注意：`max_tokens_budget` 是**只读**配置

`SettingsOut` 暴露它，但 `SettingsUpdate` **不接受**它（设置页只展示、不可改）。
所以本文件改预算时必须走 `runtime_settings.update({...})`，
并在用例结束后**恢复**——它是模块级全局状态，改脏会污染同进程的其它用例。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.llm.base import _rough_tokens
from papermate import runtime_settings
from papermate.runtime_settings import (
    INPUT_BUDGET_RATIO,
    RULES_BUDGET_RATIO,
    input_token_limit,
    rules_token_limit,
)

DEFAULT_BUDGET = 30000


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_a8_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def budget():
    """改 `max_tokens_budget` 并在用例结束后恢复（模块级全局状态，必须还原）。"""
    before = int(runtime_settings.get_state().get("max_tokens_budget", DEFAULT_BUDGET))

    def _set(value: int) -> None:
        runtime_settings.update({"max_tokens_budget": value})

    yield _set
    runtime_settings.update({"max_tokens_budget": before})


def _sse_named(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    name: str | None = None
    for ln in text.splitlines():
        if ln.startswith("event:"):
            name = ln.split(":", 1)[1].strip()
        elif ln.startswith("data:") and name:
            try:
                out.append((name, json.loads(ln[5:].strip())))
            except json.JSONDecodeError:
                pass
    return out


def _sid(client) -> int:
    return client.post("/api/v1/sessions", json={"title": "a8"}).json()["id"]


def _send(client, sid: int, content: str, **extra):
    return client.post(f"/api/v1/sessions/{sid}/messages/stream",
                       json={"content": content, **extra})


def _spy_build_agent(client, monkeypatch) -> dict:
    """记 `build_agent` 的调用次数（= 是否做了 LLM 准备工作）。"""
    svc = client.app.state.chat_svc
    real = svc.build_agent
    calls = {"n": 0}

    async def _spy(*args, **kwargs):
        calls["n"] += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(svc, "build_agent", _spy)
    return calls


# ---------------------------------------------- ① 上限与预算同源（纯函数）


def test_limits_are_derived_from_the_live_budget(budget):
    """★ 上限必须由**当前生效的** `max_tokens_budget` 派生，而不是写死。

    写死的实现（如常量 15000）会在调高预算时**不跟着放大**
    ⇒ 用户把预算调到 6 万后，仍被按 1.5 万拦下（静默的旧阈值）。
    """
    budget(DEFAULT_BUDGET)
    assert input_token_limit() == 15000 == int(DEFAULT_BUDGET * INPUT_BUDGET_RATIO)
    assert rules_token_limit() == 3000 == int(DEFAULT_BUDGET * RULES_BUDGET_RATIO)

    budget(60000)  # 调高 → 上限跟着放大
    assert input_token_limit() == 30000
    assert rules_token_limit() == 6000

    budget(10000)  # 调低 → 上限跟着收缩
    assert input_token_limit() == 5000
    assert rules_token_limit() == 1000


# ---------------------------------------------- ② 超长输入：拦下 + 不花额度 + 留痕


def test_oversized_input_is_intercepted_without_any_llm_work(client, monkeypatch, budget):
    """★ 核心：超长输入 → 直接文案 + notice，**不构建 Agent**（= 零 LLM 成本）。"""
    budget(DEFAULT_BUDGET)
    calls = _spy_build_agent(client, monkeypatch)
    sid = _sid(client)
    # 纯中文：`_rough_tokens` ≈ 字符数 / 1.64 ⇒ 2.5 万字符 ≈ 1.52 万 tokens > 1.5 万上限
    content = "中" * 25000
    assert _rough_tokens(content) > input_token_limit(), "样本本身必须越界，否则用例无效"

    r = _send(client, sid, content)
    assert r.status_code == 200, r.text
    pairs = _sse_named(r.text)
    names = [n for n, _ in pairs]

    assert "message" in names and "done" in names, names
    assert calls["n"] == 0, "拦截路径**不得**构建 Agent（否则仍会发生 LLM 工作）"

    done = next(d for n, d in pairs if n == "done")
    assert done["notice"]["kind"] == "input_too_long"
    assert done["steps"] == 0 and done["total_tokens"] == 0 and done["cost"] == 0.0, (
        f"拦截轮必须零用量：{done}"
    )

    # 正文说清"发生了什么 + 为什么"，三条可操作建议在 notice.suggestion（前端警示卡展示）
    text = next(d["text"] for n, d in pairs if n == "message")
    assert "太长了" in text and "tokens" in text, text
    assert "未产生任何费用" in done["notice"]["message"], done["notice"]
    suggestion = done["notice"]["suggestion"]
    assert "拆成几段" in suggestion, suggestion
    assert "文献" in suggestion, suggestion
    assert "只粘贴那一段" in suggestion, suggestion

    # 留痕：用户提问照样落库，且带结构化 notice（刷新后警示卡仍在）
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert any(m["role"] == "user" for m in detail["messages"]), "用户提问不得丢失"
    last = [m for m in detail["messages"] if m["role"] == "assistant"][-1]
    assert (last.get("meta") or {}).get("notice", {}).get("kind") == "input_too_long", last.get("meta")


def test_interception_does_not_break_short_input(client):
    """反向自检：正常输入**不得**被拦（防"永远在报警"）。"""
    sid = _sid(client)
    r = _send(client, sid, "什么是注意力机制？")

    assert "input_too_long" not in r.text
    assert r.status_code == 200


def test_token_ruler_not_char_ruler(client, budget):
    """★★ 判别样本：**超长字符数但 token 很小**的合法长文**不得**被拦。

    2 万个英文字母 ≈ 1,685 tokens（密度 11.87）—— 完全在预算内、应当正常发送；
    但它的 `len()` = 20000 > 15000。若实现误用 `len()` 当尺子，本用例立刻变红。
    （这正是本项目"两把尺子"的老账：触发线用真实值、cap 用估算值。）
    """
    budget(DEFAULT_BUDGET)
    sid = _sid(client)
    content = "a" * 20000
    assert len(content) > input_token_limit(), "样本必须在【字符】口径上越界（判别前提）"
    assert _rough_tokens(content) < input_token_limit(), "样本必须在【token】口径上合规"

    r = _send(client, sid, content)

    assert "input_too_long" not in r.text, (
        "被 `len()` 拦下了 —— 阈值口径与预算体系不同源（两把尺子）"
    )


def test_images_count_towards_the_input_budget(client, monkeypatch, budget):
    """多模态也要计入：图片按单图 384 tokens 计（`_rough_tokens` 的口径）。"""
    calls = _spy_build_agent(client, monkeypatch)
    # 预算压到 1000 ⇒ 输入上限 500 tokens；两张图 = 768 tokens 单独就越界
    budget(1000)
    sid = _sid(client)
    imgs = ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"]

    r = _send(client, sid, "看图", images=imgs)

    assert "input_too_long" in r.text, "图片未计入输入预算（只算了文本字符）"
    assert calls["n"] == 0


# ---------------------------------------------- ③ 长期规则：更严、且给可操作提示


def test_oversized_agent_rules_rejected_with_actionable_detail(client, budget):
    """★ 长期规则每轮都注入 ⇒ 上限更紧；超限用 400 + 可操作中文说明。"""
    budget(DEFAULT_BUDGET)
    huge = "中" * 6000
    assert _rough_tokens(huge) > rules_token_limit()

    r = client.put("/api/v1/settings", json={"agent_rules": huge})

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "长期规则过长" in detail and "每一次请求" in detail, detail
    assert "精简" in detail, f"必须给出可操作建议：{detail}"


def test_normal_agent_rules_accepted(client):
    """反向自检：正常规则照常保存（防"永远拒绝"）。"""
    rules = "回答用简体中文。\n引用必须标注页码。"

    r = client.put("/api/v1/settings", json={"agent_rules": rules})

    assert r.status_code == 200, r.text
    assert r.json()["agent_rules"] == rules


def test_rules_limit_is_stricter_than_input_limit(client, budget):
    """★ 量级差异是**有意为之**：规则每次请求都要付、用户输入只付一次。"""
    budget(DEFAULT_BUDGET)
    assert rules_token_limit() < input_token_limit()
