"""P2 零风险批次（2026-09-28）：**观测与健壮性**修复的回归测试。

本文件只验证"**该看见的能看见**"与"**不该被丢的不会丢**"，全部**零行为变更**：

| 项 | 缺陷 | 修法 |
|---|---|---|
| **A5** | 压缩只写"触发"，不写"丢了几条" ⇒ **无法区分"真压了"与"空转"** | 两条日志：真丢时打 `丢弃 N 条 · 摘要=是/否`；触发了却没丢时打"**未丢弃任何消息**" |
| **A6** | `rest = [m for m in history if m not in pinned]` 按**值**排除，语义应是"同一条消息" | 改用 `id()` 集合（**防御性**：当前 pin 判据下不可达，见 `test_pinned_note_survives_compaction` 的说明） |
| **A7** | 隐藏 LLM 调用（历史摘要 / 兜底收尾 / 失败自评）**不计入 tokens 与成本** ⇒ 熔断被低估 | **只做观测**：单列 `[隐藏调用]` 行（**不并入累计** —— 那会改熔断触发点，属行为变更，另议） |
| **A1** | 历史按**条数**静默截断（20 条），与其余 token 口径不同源且无日志 | 截断时打日志（**不改判据** —— 改判据会改每轮上下文，属行为变更） |
| **A3** | `_route` 在技能路由**之前**算 ⇒ 走技能时日志写 `→ agent`，**盖住真实路径** | 技能命中时补 `→ skill:<name>`（**不改 `_route` 取值** —— 它参与下游分支语义） |
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.fake import FakeLLM
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolSpec

LOOP = "lightcore.core.loop"
ROUTE = "papermate.api.routes_search"


def _logs(caplog, logger: str, marker: str) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == logger and marker in r.getMessage()]


# ------------------------------------------------------------ A5：压缩结果可见


def _loop(**cfg) -> AgentLoop:
    """无工具的 loop（`compact_keep_capacity` 由 `max_tokens_budget` 派生，传小值即可触发）。"""
    return AgentLoop(FakeLLM(), ToolExecutor(ToolRegistry()), config=LoopConfig(**cfg))


def _hist(n: int, chars: int = 600) -> list[ChatMessage]:
    return [ChatMessage(role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
                        content=f"第{i}轮内容" * (chars // 5)) for i in range(n)]


def test_compaction_logs_how_many_were_dropped(caplog):
    """★ A5 主用例：真丢消息时必须写"丢弃 N 条"，否则事后分不清"压了"还是"没压"。"""
    caplog.set_level(logging.INFO, logger=LOOP)
    loop = _loop(max_tokens_budget=1000)  # ⇒ keep_capacity = max(1000, 490) = 1000

    out = asyncio.run(loop._compact_history(_hist(4, 600)))

    lines = _logs(caplog, LOOP, "[历史压缩]")
    assert lines, "压缩了却没有任何日志（A5 的缺陷本体）"
    # 只断言"丢弃了正数条"而不写死条数 —— 条数依赖 `_rough_tokens` 的分档密度，
    # 写死会让用例随估算器标定一起碎（本项目在估算口径上吃过这个亏）。
    import re

    m = re.search(r"丢弃 (\d+) 条", lines[0])
    assert m and int(m.group(1)) >= 1, lines[0]
    assert len(out) == 4 - int(m.group(1)), f"日志与实际不一致：{lines[0]} / 剩余 {len(out)}"
    # FakeLLM 不做摘要 ⇒ 摘要=否（口径要写清，否则会误以为"上下文没丢"）
    assert "摘要=否" in lines[0], lines[0]


def test_triggered_but_nothing_dropped_is_also_visible(caplog):
    """★★ A5 的另一半：**触发了却一条都没丢**（空转）必须显式可见。

    成因：触发线 = 整 prompt 的**真实值**，执行线 = 仅历史的**估算值**（两把尺子）。
    没有这条日志时，空转**只能靠"日志里没有 [历史压缩] 行"来推断**，极易误判。
    """
    caplog.set_level(logging.INFO, logger=LOOP)
    loop = _loop(max_tokens_budget=1000)

    asyncio.run(loop._compact_history(_hist(2, 40), triggered=True))

    lines = _logs(caplog, LOOP, "[历史压缩]")
    assert lines and "触发了但未丢弃" in lines[0], f"空转不可见：{lines}"


def test_real_trigger_path_passes_triggered_flag(caplog):
    """★★★ 走**真实触发路径**（`_prepare_history`）时也必须看得见空转。

    ⚠️ 这条是**反向自检逼出来的**：上面那条用例是**直接调** `_compact_history(..., triggered=True)`，
    于是"调用点到底有没有把 `triggered` 传下去"**完全没被覆盖** ——
    我把调用点改成 `triggered=False` 后，上面那条**仍然通过**（静默失效）。
    ⇒ 凡是"靠调用方传参才能生效"的修复，必须有一条走**真实调用链**的用例。

    构造：`_last_real_prompt_tokens` ≥ 触发线（真实值触发），但历史估算 < 保留阈值
    ⇒ 正是 A5 描述的"超长输入/大 system prompt 撑破触发线、历史本身不大"。

    ★★ **P0（2026-10-03）后必须改构造**：本条原用 `基线=50_000`，
    在 P0 之后 `need_cut = 50000−1000 = 49000 ⇒ hist_cap=0 ⇒ **真的会丢**`
    （那是 P0 的目的，不是回归）。若要构造"空转"，基线必须**刚过触发线**：
    取 `800`（≥ 触发线 700 ✓ 会触发；`need_cut = max(0, 800−1000) = 0` ⇒ 不削）。
    ⚠️ 本条的**目的不变**（覆盖"调用链是否把 `triggered` 传下去"），只换构造。
    """
    caplog.set_level(logging.INFO, logger=LOOP)
    loop = _loop(max_tokens_budget=1000)
    # 刚好过触发线 700、但**不超**整 prompt 目标 1000 ⇒ 换算后 need_cut=0 ⇒ 无历史可削
    loop._last_real_prompt_tokens = 800

    asyncio.run(loop._prepare_history(_hist(2, 40)))

    joined = " ".join(r.getMessage() for r in caplog.records if r.name == LOOP)
    assert "触发（真实口径）" in joined, "连触发日志都没有 —— 构造不成立"
    assert "触发了但未丢弃" in joined, (
        "真实触发路径下空转不可见 ⇒ 说明调用点没把 `triggered` 传下去（A5 修复被绕过）"
    )


def test_p0_prompt_over_target_really_drops_history(caplog):
    """★★★ P0（2026-10-03）核心验收：**整 prompt 超目标 ⇒ 历史必须真的让位**。

    ## 这条守护的是"修复压缩空转"本身
    修复前：执行线拿"整 prompt 目标"（1000）去比"仅历史估算"（40）⇒ 不丢（**空转**）。
    修复后：`need_cut = 基线 − 目标 = 50000−1000 = 49000` ⇒ `hist_cap = 0`
    ⇒ 历史被削光（尽力而为），**不再空转**。

    ⚠️ 削光历史仍可能超限 ⇒ 由**既有熔断**兜住，不是本层职责
    （本层只负责"能削多少削多少"）。
    """
    caplog.set_level(logging.INFO, logger=LOOP)
    loop = _loop(max_tokens_budget=1000)
    loop._last_real_prompt_tokens = 50_000  # 远超整 prompt 目标 1000

    out = asyncio.run(loop._prepare_history(_hist(2, 40)))

    joined = " ".join(r.getMessage() for r in caplog.records if r.name == LOOP)
    assert "触发（真实口径）" in joined, "构造不成立：没走真实触发路径"
    assert "触发了但未丢弃" not in joined, (
        "★ P0 回归：整 prompt 远超目标却仍不丢历史 —— 压缩又空转了"
    )
    assert "⇒ 丢弃" in joined, f"应打出真丢日志：{joined}"
    assert len(out) < 2, f"历史应被削减，实际还剩 {len(out)} 条"


def test_silent_when_not_triggered(caplog):
    """反向自检：**未触发**且历史不大 ⇒ **不打**日志（保持信噪比，别每轮都刷）。"""
    caplog.set_level(logging.INFO, logger=LOOP)
    loop = _loop(max_tokens_budget=1000)

    asyncio.run(loop._compact_history(_hist(2, 40), triggered=False))

    assert _logs(caplog, LOOP, "[历史压缩]") == [], "未触发却打了压缩日志（信噪比）"


# ------------------------------------------------------------ A6：pinned 语义


def test_pinned_note_survives_compaction():
    """★ A6 契约：置顶注记（长期记忆/意图识别）**绝不能被当作可丢弃的历史**。

    ⚠️ **诚实说明（A6 的真实性边界）**：改用 `id()` 集合是**防御性**的 ——
    在当前 `_is_pinned_note` 判据下，"与置顶注记内容完全相同"的普通消息
    **本身也会满足 pin 条件**（同为 SYSTEM + 含关键词），因此两种写法**结果相同**，
    该缺陷**当前不可达**。保留改动是因为它把判据语义从"相等"纠正为"同一条"，
    对将来 pin 判据放宽（如允许非 SYSTEM 注记）是现成的正确行为。
    ⇒ 本用例钉的是**真正有价值的契约**（注记不被丢），不是两种写法的差异。
    """
    loop = _loop(max_tokens_budget=1000)
    note = ChatMessage(role=Role.SYSTEM, content="【长期记忆】用户偏好：回答用中文、引用标页码。")
    history = [note, *_hist(4, 600)]

    out = asyncio.run(loop._compact_history(history))

    assert note in out, "置顶注记被丢了 —— 长期记忆/意图识别会跨轮丢失"


# ------------------------------------------------------------ A7：隐藏调用可见


class _AlwaysToolLLM:
    """永远返回工具调用 ⇒ 迭代耗尽 ⇒ 走兜底收尾生成（`_finalize_without_tools`）。

    每次调用**固定** `cost=0.5` / `pt=50` / `ct=5` ⇒ "主调用成本"与"隐藏调用成本"
    都可以**精确验算**（这是 `test_hidden_usage_is_counted_into_done_but_not_into_the_fuse`
    的支点）。
    """

    model_name = "test-model"
    per_call_cost = 0.5

    def __init__(self) -> None:
        self.n = 0

    async def chat(self, messages, *args, **kw):  # noqa: ANN001, ANN002
        self.n += 1
        from lightcore.llm.base import LLMResponse, ToolCall

        if self.n <= 2:
            return LLMResponse(
                message=ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                    ToolCall(id=f"c{self.n}", name="retrieve", arguments={"query": "q"})]),
                prompt_tokens=50, completion_tokens=5, cost=self.per_call_cost,
                model=self.model_name)
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="兜底收尾回答"),
            prompt_tokens=50, completion_tokens=5, cost=self.per_call_cost,
            model=self.model_name)


def _loop_with_tool(llm) -> AgentLoop:
    # 工具名必须是 `retrieve`：`FakeLLM` 的剧本固定调 `retrieve`（改名字会 KeyError）。
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="retrieve", description="测试用",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission=Permission.SAFE,
        handler=lambda **_kw: __import__("lightcore.tools.spec", fromlist=["ToolResult"]).ToolResult(
            ok=True, output={"results": [{"id": "c1", "page": 1, "text": "正文"}]}),
    ))
    return AgentLoop(llm, ToolExecutor(reg), config=LoopConfig(max_iterations=2))


def test_hidden_llm_calls_are_logged_on_abort(caplog):
    """★ A7：中止轮的**兜底收尾生成**是隐藏调用 ⇒ 必须可见（它不计入 tokens/成本）。"""
    caplog.set_level(logging.INFO, logger=LOOP)

    async def go():
        return [ev async for ev in _loop_with_tool(_AlwaysToolLLM()).run_stream("问题")]

    asyncio.run(go())

    lines = _logs(caplog, LOOP, "[隐藏调用]")
    assert lines, "发生了隐藏调用却没有任何提示（成本/熔断被系统性低估却看不出来）"
    assert "不计入" in lines[0], lines[0]


def test_no_hidden_call_line_when_there_is_none(caplog):
    """反向自检：正常收敛轮**没有**隐藏调用 ⇒ 不该打这行（信噪比 + 证明它只在该出现时出现）。"""
    caplog.set_level(logging.INFO, logger=LOOP)

    async def go():
        return [ev async for ev in _loop_with_tool(FakeLLM()).run_stream("问题")]

    asyncio.run(go())

    assert _logs(caplog, LOOP, "[隐藏调用]") == [], "正常轮也打了隐藏调用行（说明计数被误累加）"


def test_hidden_usage_is_counted_into_done_but_not_into_the_fuse():
    """★★ A7 统计同源：隐藏用量必须进 `done`（统计完整），但**不进熔断判据**。

    这是 A7 建议里"先做统计同源、熔断不动"的**可验证表述**：
    - 对外：`total_tokens / prompt_tokens / completion_tokens / cost / latency_ms`
      = 主调用 + 隐藏调用 ⇒ 用量页与落库不再是缺一块的数字；
    - 对内：`_acc_cost`（`max_cost_per_turn` 的判据）**不含**隐藏调用
      ⇒ 熔断时机与改动前**完全一致**。
    """

    async def go(llm):
        loop = _loop_with_tool(llm)
        stream = [ev async for ev in loop.run_stream("问题")]
        return loop, stream

    llm = _AlwaysToolLLM()
    loop, events = asyncio.run(go(llm))
    done = next(e.data for e in events if e.type == "done")

    assert done["_hidden_calls"] == 1, f"应有 1 次隐藏调用（兜底收尾）：{done}"
    assert done["_hidden_tokens"] > 0, "隐藏用量未计入（A7 统计缺口的本体）"
    # ① 对外：完整
    assert done["total_tokens"] == loop._acc_tokens + loop._hidden_tokens
    assert done["prompt_tokens"] == loop._acc_prompt_tokens + loop._hidden_prompt_tokens
    assert done["completion_tokens"] == loop._acc_completion_tokens + loop._hidden_completion_tokens
    # ② 对内：熔断判据不变 —— 隐藏调用**没有**被并进 `_acc_cost`。
    #    可验算：`_acc_cost` =（总调用次数 − 隐藏次数）× 每次成本
    main_calls = llm.n - done["_hidden_calls"]
    assert loop._hidden_cost == pytest.approx(llm.per_call_cost), (
        f"隐藏成本应为 1 次：{loop._hidden_cost}"
    )
    assert loop._acc_cost == pytest.approx(main_calls * llm.per_call_cost), (
        "隐藏成本被并进了 `_acc_cost` ⇒ 熔断判据被改动（与本次决策不符）"
    )
    assert done["cost"] == pytest.approx(llm.n * llm.per_call_cost), "对外成本应含隐藏调用"
    # ③ 完整值必须**严格大于**只算主调用的值（否则等于没修）
    assert done["total_tokens"] > loop._acc_tokens


def test_done_exposes_hidden_breakdown_even_when_zero():
    """稳定 schema：正常轮次没有隐藏调用时，`_hidden_*` 也应存在且为 0。

    为什么钉这条：若只在"有隐藏调用"时才加这几个键，下游（前端/统计脚本）
    就要处理"键可能不存在" ⇒ 又一处"键缺失即静默"的老问题。
    """

    async def go():
        return [ev async for ev in _loop_with_tool(FakeLLM()).run_stream("问题")]

    done = next(e.data for e in asyncio.run(go()) if e.type == "done")

    assert done["_hidden_calls"] == 0
    assert done["_hidden_tokens"] == 0
    assert done["_hidden_cost"] == 0.0


# ------------------------------------------------------------ A1 / A3：路由级


@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="pm_obs_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_history_window_truncation_is_logged(client, caplog):
    """★ A1：历史按**条数**被砍时必须有日志 —— 用户"前面说的话被忘了"的唯一线索。"""
    caplog.set_level(logging.INFO, logger=ROUTE)
    svc = client.app.state.chat_svc
    sid = client.post("/api/v1/sessions", json={"title": "a1"}).json()["id"]
    for i in range(11):  # 22 条 > _MAX_HISTORY(20)
        svc.persist_turn(sid, f"提问{i}", [], [], f"回答{i}", "fake")

    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "再来一轮"})
    assert r.status_code == 200, r.text

    lines = _logs(caplog, ROUTE, "[历史窗口]")
    assert lines, "历史被截断却没有任何日志（A1 的缺陷本体）"
    assert "丢弃最旧 2 条" in lines[0], lines[0]
    assert "条数" in lines[0], "应点明它是按【条数】截断（与 token 口径不同源）"


def _patch_skill(monkeypatch) -> None:
    """让 app 固定命中一个**假技能**（把测试与真实技能清单解耦）。"""
    import papermate.skills as skills_pkg
    from lightcore.skills.spec import SkillResult

    class _EchoSkill:
        name = "echo"

        async def run(self, ctx):  # noqa: ANN001
            resp = await ctx.llm.chat([ChatMessage(role=Role.USER, content="user")])
            return SkillResult(ok=True, content=resp.message.content or "", citations=[])

    class _StubRouter:
        def __init__(self) -> None:
            self._s = _EchoSkill()

        def route(self, _c, _f):  # noqa: ANN001
            return self._s

        def parse_args(self, _c, _s):  # noqa: ANN001
            return []

    monkeypatch.setattr(skills_pkg, "default_router", _StubRouter)


def test_skill_path_is_logged_as_skill_not_agent(client, monkeypatch, caplog):
    """★ A3：走技能时日志必须写 `→ skill:<name>`，不能只写 `→ agent`。

    ⚠️ 命令路径有前置检查：未关联文档时**直接返回提示、根本不执行技能**
    （不关联文档 ⇒ 永远走不到技能里 ⇒ **假通过**）。故这里必须关联文档。
    """
    caplog.set_level(logging.INFO, logger=ROUTE)
    _patch_skill(monkeypatch)
    sid = client.post("/api/v1/sessions", json={"title": "a3"}).json()["id"]
    # 文档 id 不存在也无妨（假技能不做检索），但**必须关联**，否则命令路径提前返回
    client.patch(f"/api/v1/sessions/{sid}", json={"config": {"doc_ids": [1]}})

    r = client.post(f"/api/v1/sessions/{sid}/messages/stream", json={"content": "/echo"})
    assert r.status_code == 200, r.text

    lines = _logs(caplog, ROUTE, "技能覆盖")
    assert lines, "走了技能却只有 `→ agent` 那条日志（真实路径被盖住 —— A3 的缺陷本体）"
    assert "skill:echo" in lines[0], lines[0]
    assert "原判定=" in lines[0], "应同时保留原判定，便于对照两处口径为何不同"
