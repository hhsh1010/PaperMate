"""P3 批量小修（2026-09-29）：**正确性 + 观测**的回归测试。

本批**不含行为变更**（行为变更项 B2/B8 单独交付影响面评估，未实施）。逐项对照：

| 项 | 缺陷本体 | 本文件钉什么 |
|---|---|---|
| **B4** | `text[:80].replace(r"\\s+", " ")` 里 `r"\\s+"` 是**字面量**、不是正则 ⇒ 空白归一化**从未生效** | 仅差空白的两条**无 id** 片段必须被去重（字面量版本会漏） |
| **B3** | `len(text) > 600 → continue` 把超长片段**整条丢弃**（不是截断） | 超长片段应**截断保留**；★ 全命中都是长块时引用**不得为空**（否则误触发"零引用/疑似幻觉"） |
| **B5** | 引用达上限是**静默 break** ⇒ 用户与 trace 都不知道"少看了若干条" | 确有**新增**引用被丢 ⇒ 置位；恰好填满/全是重复 ⇒ **不**置位 |
| **C4** | 闸门裁剪只写日志、不进事件与 trace | 真的裁了 ⇒ `_gate_hits` 计数正确，并出现在 `done` |
| **C6** | 历史压缩只写日志、不进事件与 trace | 真的丢了 ⇒ `_compacted_dropped` 计数正确，并出现在 `done` |

## 为什么 C4/C6 要被单独钉住（判据盲区的老账）

`judge` 只测 `faithfulness` / `abstention`，**不测完整性** ⇒ 闸门/压缩造成的"证据被裁掉"
**在指标上完全测不出来**（指标不报警而用户会）。所以"裁剪发生过"这件事必须成为
**trace 里的可核对事实**，而不是只活在日志里 —— 与 C3「检索空命中」进归因是同一类修复。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import FakeLLM
from lightcore.rag.pipeline import RetrievedChunk
from lightcore.tools.spec import ToolExecutor, ToolRegistry
from papermate.models.db import Trace

# ------------------------------------------------------------ B4 / B3：引用清洗


def test_b4_whitespace_normalization_actually_applies():
    """★ B4 主用例：两条**仅空白不同**的无 id 片段必须被去重。

    旧实现 `text[:80].replace(r"\\s+", " ")` 里 `r"\\s+"` 是**字面量**（匹配文本 `\\s+`），
    归一化从未生效 ⇒ 若只差空白，两条会被当成不同片段**双双留下**（重复引用）。
    """
    from papermate.services.citation_clean import clean_citations

    # ⚠️ 构造要点：`clean_citations` 先按 `text[:80]` 取键、**再**归一化空白，
    # 所以两段必须**总长相等**，否则 `[:80]` 会切到不同的尾部字符、键天然不同
    # （这会把"实现没生效"与"测试构造错"混为一谈）。下面两段总长都 < 80 ⇒ 键都在窗口内。
    a = {"text": "alpha   beta   gamma   delta"}   # 多空格（27 字）
    b = {"text": "alpha\tbeta\tgamma\tdelta"}      # 制表符（23 字，归一化后与 a 同）
    out = clean_citations([a, b])
    assert len(out) == 1, (
        "仅空白不同的两条无 id 片段未被去重 ⇒ `re.sub` 未生效（B4 缺陷本体）"
    )


def test_b3_long_citation_is_truncated_not_dropped():
    """★ B3 主用例：超长片段应**截断 + 省略号**保留，而不是整条丢弃。

    ★★ 端到端约束（**2026-10-01 B2-X2 更新口径**）：清洗已收敛到后端唯一入口，
    前端 `CitationCard` **不再做任何丢弃** ⇒ 这条不再是"跨层闸门"，而是**本项目的展示口径**：
    引用卡片展示的正文上限就是 `MAX_TEXT`（截断 + 省略号，而不是消失）。
    ⚠️ 历史：B2-X2 之前前端另有一份 `cleanCitations`（`t.length > 600 → continue`），
    超长片段会被**前端静默丢弃**，而后端与 `citation_rate` 仍计入 ⇒ 已随同源化删除。
    """
    from papermate.services.citation_clean import MAX_TEXT, clean_citations

    out = clean_citations([{"id": "c1", "text": "论文正文内容。" * 400}])  # 2800 字
    assert len(out) == 1, "超长片段被整条丢弃（B3 缺陷本体）"
    t = out[0]["text"]
    assert t.endswith("…"), "截断后应带省略号标记"
    assert len(t) == MAX_TEXT, len(t)
    assert MAX_TEXT == 600, "展示上限口径（改了要同步文档与评测指纹）"


def test_b3_all_long_chunks_do_not_yield_empty_citations():
    """★★ B3 的**真正危害**：若本轮命中的块**全部**超长，引用会意外为空 ⇒
    下游 `zero_citation_notice` 据"零引用"误判"疑似幻觉"——而真相是证据很充分、只是被丢了。

    这条用例就是"截断代替丢弃"的**正当性来源**（不是偏好，是 bug）。
    """
    from papermate.services.citation_clean import clean_citations

    # ⚠️ 各段前缀不同 —— 否则 `clean_citations` 会按 `text[:80]` 把相同文本**去重成 1 条**，
    # 那测的就不是"截断"而是"文本去重"了（很容易误判）。
    res = [{"id": f"c{i}", "text": f"第{i}段。" + "这是一段足够长但超过上限的论文正文。" * 60}
           for i in range(3)]
    out = clean_citations(res)
    assert len(out) == 3, "全部长块被丢弃 ⇒ 会误触发零引用/疑似幻觉提示"


# ------------------------------------------------------------ B5：引用截断可观测


def _cit(i: int) -> dict:
    """造一条长度 ≥16、带唯一 id 的引用。

    ⚠️ 文本必须**逐条不同**：`clean_citations` 会先按 `text[:80]` 去重，
    若三条文本相同，还没走到 `dedup_citations` 就已被并成 1 条（会掩盖上限逻辑）。
    """
    return {"id": f"c{i}", "text": f"检索命中的正文内容，片段编号为第{i}条", "page": 1}


def test_b5_flag_set_only_when_a_new_citation_is_dropped():
    """★ B5 主用例：limit=2、给 3 条**新的** ⇒ 第 3 条因上限被丢 ⇒ 必须置位。"""
    from papermate.services.citation_clean import dedup_citations

    keys: set[str] = set()
    capped: list[bool] = []
    out = dedup_citations([_cit(0), _cit(1), _cit(2)], keys, 2, capped_out=capped)
    assert len(out) == 2, out
    assert capped == [True], f"确有新增引用被丢却没置位：{capped}"


def test_b5_no_flag_when_exactly_filled_without_remainder():
    """反向：恰好填满 limit 且**没有剩余** ⇒ 不得误报为"截断"。"""
    from papermate.services.citation_clean import dedup_citations

    keys: set[str] = set()
    capped: list[bool] = []
    out = dedup_citations([_cit(0), _cit(1)], keys, 2, capped_out=capped)
    assert len(out) == 2 and capped == [], f"恰好填满被误报为截断：{capped}"


def test_b5_dup_at_cap_does_not_falsely_flag():
    """反向：达上限后**下一条是重复** ⇒ 丢弃它不算"少看引用" ⇒ 不得置位。"""
    from papermate.services.citation_clean import dedup_citations

    keys: set[str] = set()
    capped: list[bool] = []
    first = dedup_citations([_cit(0), _cit(1)], keys, 2, capped_out=capped)  # 填满
    assert len(first) == 2 and capped == []
    again = dedup_citations([_cit(0)], keys, 2, capped_out=capped)  # 同一条再次出现
    assert again == [] and capped == [], "重复项被当成'被截断的新引用'"


def test_b5_signature_backward_compatible():
    """不带 `capped_out` 时行为与返回类型不变（既有调用/测试不受影响）。"""
    from papermate.services.citation_clean import dedup_citations

    keys: set[str] = set()
    out = dedup_citations([_cit(0)], keys, 10)
    assert isinstance(out, list) and len(out) == 1


# ------------------------------------------------------------ C4：闸门裁剪计数


def _gate_loop(baseline: int, mark: int) -> AgentLoop:
    """只带闸门所需字段的 loop（同 `test_token_gate.py` 的做法）。"""
    lp = object.__new__(AgentLoop)
    lp.config = LoopConfig(max_tokens_budget=1000)  # 触发线 = 700
    lp._last_real_prompt_tokens = baseline
    lp._gate_mark = mark
    lp._steps = 0
    return lp


def _tool_msg(chars: int, call_id: str = "c1") -> ChatMessage:
    return ChatMessage(role=Role.TOOL, content="x" * chars, tool_call_id=call_id)


def test_c4_gate_context_counts_elided_tool_results():
    """★ 真的裁了 ⇒ `_gate_hits` 必须 > 0（且经 `getattr` 兜底，替身缺字段也不炸）。"""
    lp = _gate_loop(baseline=700, mark=3)  # 已达触发线
    msgs = [_tool_msg(3000, "call-A")] + [
        ChatMessage(role=Role.USER, content="填充") for _ in range(9)]
    lp._gate_context(msgs)
    assert lp._gate_hits == 1, f"闸门裁了却没计数：{getattr(lp, '_gate_hits', None)}"


def test_c4_gate_no_elision_keeps_count_zero():
    """反向：未超线 ⇒ 不动手、计数保持 0。"""
    lp = _gate_loop(baseline=100, mark=2)  # 触发线 700，投影远低于
    msgs = [_tool_msg(100), ChatMessage(role=Role.USER, content="问题")]
    lp._gate_context(msgs)
    assert getattr(lp, "_gate_hits", 0) == 0


# ------------------------------------------------------------ C6：压缩计数 + done 出口


def _loop(**cfg) -> AgentLoop:
    return AgentLoop(FakeLLM(), ToolExecutor(ToolRegistry()), config=LoopConfig(**cfg))


def _hist(n: int, chars: int = 600) -> list[ChatMessage]:
    return [ChatMessage(role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
                        content=f"第{i}轮内容" * (chars // 5)) for i in range(n)]


def test_c6_compact_history_counts_dropped():
    """★ 真的丢了 ⇒ `_compacted_dropped` 与返回值的"少了多少条"一致。"""
    import asyncio

    loop = _loop(max_tokens_budget=1000)
    out = asyncio.run(loop._compact_history(_hist(4, 600)))
    assert loop._compacted_dropped == 4 - len(out) >= 1, (
        loop._compacted_dropped, len(out))


def test_c4_c6_defaults_are_zero_on_fresh_loop():
    """稳定 schema：没发生裁剪的正常轮次必须是 0（否则"有没有被裁"无法核对）。"""
    loop = _loop()
    d = loop._done_event()
    assert d.data["_gate_hits"] == 0
    assert d.data["_compacted_dropped"] == 0


def test_c4_c6_done_event_surfaces_counts():
    """★ done 是**唯一出口**：裁剪计数必须出现在 done 载荷里（随之落 trace）。"""
    loop = _loop()
    loop._gate_hits = 2
    loop._compacted_dropped = 3
    d = loop._done_event()
    assert d.data["_gate_hits"] == 2
    assert d.data["_compacted_dropped"] == 3


def test_c4_c6_counts_tolerate_missing_attrs():
    """★ 教条落点：观测代码必须能被**替身/直调**路径容下（替身缺新字段不得炸）。"""
    lp = object.__new__(AgentLoop)
    lp.config = LoopConfig()
    lp._steps = 0
    for f in ("_hidden_llm_calls", "_hidden_tokens", "_hidden_cost",
              "_acc_tokens", "_acc_prompt_tokens", "_acc_completion_tokens",
              "_acc_cost", "_acc_latency", "_peak_prompt_tokens",
              "_first_prompt_tokens", "_hidden_prompt_tokens",
              "_hidden_completion_tokens", "_hidden_latency_ms"):
        setattr(lp, f, 0)
    d = lp._done_event()  # 不得因缺 `_gate_hits` / `_compacted_dropped` 抛异常
    assert d.data["_gate_hits"] == 0 and d.data["_compacted_dropped"] == 0


# ------------------------------------------------------------ B5 端到端：done 载荷


DOC = "量子纠缠是量子力学中一种非局域关联现象。对其中一个粒子测量会瞬间决定另一个的状态。"


class _ManyPipe:
    """一次检索返回 **70 条互不相同**的命中。

    ⚠️ 文本必须**逐条不同**：`clean_citations` 会按 `text[:80]` 先去重，相同文本会被并成 1 条
    （那测的就不是"上限截断"了）。另：composite 会 `merged[:top_k]`（≤20），
    故真实触发上限要靠**调小 `_MAX_CITATIONS`**（见用例），而不是靠"造够 60 条"。
    """

    def get_chunks(self):  # noqa: ANN201  pdf_extract 需要
        return [RetrievedChunk(id="doc-9-c0", text="正文", page=1, score=0.5)]

    def query(self, _q: str, top_k: int = 3):  # noqa: ANN201
        # ⚠️ 文本必须 ≥16 字（`clean_citations` 的下限），否则会被当"太短"丢掉、
        # 引用恒为空 —— 那样测的就是"没触发"而不是"截断"了（本用例第一版就栽在这）。
        return [
            RetrievedChunk(id=f"doc-9-c{i}", text=f"第{i}号命中段落的正文内容足够长可以保留",
                           page=1, score=0.9 - i / 1000)
            for i in range(70)
        ]


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_p3_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _prepare(client, monkeypatch, pipe) -> int:
    doc_svc = client.app.state.doc_svc
    sid = client.post("/api/v1/sessions", json={"title": "p3"}).json()["id"]
    files = {"file": ("doc.md", DOC.encode(), "text/markdown")}
    assert client.post(f"/api/v1/documents?session_id={sid}", files=files).status_code == 200
    client.patch(f"/api/v1/sessions/{sid}", json={"config": {"model": "fake"}})

    async def _fake_get_pipeline(_doc_id, **_kw):
        return pipe

    monkeypatch.setattr(doc_svc, "get_pipeline", _fake_get_pipeline)
    return sid


def test_b5_citations_capped_is_visible_in_done_payload(client, monkeypatch):
    """★ B5 端到端：引用因达上限被截断 ⇒ SSE 的 done 必须带 `_citations_capped: true`，
    且该字段**随之落 trace**（`done` 在 `events` 里）—— 用户与事后分析都能知道"少看了引用"。

    说明：真实触发 60 条上限成本高（composite 每次 `[:top_k]`，实测单次仅 3 条命中），
    故把上限**调小到 1**，在**同一条真实代码路径**上验证
    "截断 → 置位 → 进 done/trace"这条链路本身（阈值取值不改变链路的正确性）。
    """
    monkeypatch.setattr("papermate.api.routes_search._MAX_CITATIONS", 1)
    sid = _prepare(client, monkeypatch, _ManyPipe())
    r = client.post(f"/api/v1/sessions/{sid}/messages/stream",
                    json={"content": "文档里的量子纠缠是什么？"})
    assert r.status_code == 200, r.text

    # 从 SSE 文本里取 `event: done` 块的 data 行并解析。
    # ⚠️ done 用通用 `sse_event`（载荷里**没有** `type` 键），不能按 `"type": "done"` 找。
    done = None
    for block in r.text.split("\n\n"):
        if block.startswith("event: done"):
            data_line = next(ln for ln in block.splitlines() if ln.startswith("data: "))
            done = json.loads(data_line[len("data: "):])
    assert done is not None, "SSE 里没有 done 帧"
    assert done.get("_citations_capped") is True, (
        f"引用被截断却没在 done 里体现：{done.get('_citations_capped')}")

    # 落 trace 侧同样可见（C4/C6/B5 的落点就是 trace）
    with client.app.state.chat_svc._Session() as s:  # noqa: SLF001
        t = s.query(Trace).filter(Trace.session_id == sid).order_by(Trace.id.desc()).first()
        assert t is not None
        evs = json.loads(t.events_json or "[]")
    done_ev = next((e for e in evs if e.get("type") == "done"), None)
    assert done_ev is not None, "trace 里没有 done 事件"
    assert done_ev["data"].get("_citations_capped") is True, "落 trace 的 done 未带该字段"


# ------------------------------------------------------------ B2-X2：引用清洗同源化
#
# 背景：清洗曾有**三份实现**（后端 / 技能侧 / 前端），且已分叉 ⇒
# 引用在 UI 里静默消失、而落库与 `citation_rate` 仍计入（「指标有数、用户看不到」）。
# 2026-10-01 收敛为**唯一入口** `papermate.services.citation_clean`，前端不再做任何过滤。
# 下面三条把"唯一入口"的口径钉住（含**有意不收** `"Citation"` 的决定）。


def test_b2x2_citation_word_is_not_a_noise_keyword():
    """★ 口径守卫：`"Citation"` **不是**噪声词（有意决定，理由见模块 docstring）。

    它是常见**正文**词（"Citation counts have grown …"）。整体当噪声会**静默丢弃合法证据** ——
    正是 B3 判定为 bug 的那类行为；且该词此前**只在**前端过滤过 ⇒ 不采纳 = **指标零漂移**。
    """
    from papermate.services.citation_clean import NOISE_KEYWORDS, clean_citations

    assert "Citation" not in NOISE_KEYWORDS, (
        "把 'Citation' 当噪声会丢掉含该词的正文片段；若确要收，必须同步改测试与文档并重跑评测指纹"
    )
    text = "Citation counts have grown rapidly in this field over the past decade."
    assert len(clean_citations([{"id": "c1", "text": text}])) == 1, "含 'Citation' 的正文被误当噪声丢弃"


def test_b2x2_noise_filter_still_works_in_single_entrypoint():
    """噪声过滤**仍然生效**（防"同源化时把过滤一起删掉"）。"""
    from papermate.services.citation_clean import clean_citations

    noisy = "收稿日期：2025-01-01　通讯作者：张三　基金项目：国家自然科学基金"
    assert clean_citations([{"id": "c1", "text": noisy}]) == []


def test_b2x2_clean_citations_is_idempotent():
    """重复清洗结果不变 —— 技能出口与 agent 路径共用同一入口，可能被调用多次，必须幂等。"""
    from papermate.services.citation_clean import clean_citations

    raw = [{"id": "c1", "text": "这是一段足够长的论文正文内容。" * 40}]
    once = clean_citations(raw)
    assert once and clean_citations(once) == once
