"""结果级去重（D2，2026-09-28）的守护测试。

## 为什么做去重（实测依据，不是"省 token 提速"）

`docs/去重换策略-影响面评估.md`：单请求的原始命中量可达**语料的 136%**；
`pdf_extract` 12 次调用占 **46%** 注入量、压缩比 **100%**；峰值 prompt 实测到
**29,357 / 30,000**，中途触发 3 次闸门裁剪（`_gate_context`）。
⇒ 去重的真实价值是**少触发裁剪**（裁剪 = 静默丢证据），不是省那 66ms。

## 本文件在守什么（四条硬约束 —— 缺一即"用质量换 token"）

1. ★★ **集合必须与上下文同步**：闸门把早期工具结果**替换**成裁剪标记后，
   去重记录若仍留着这些键，就会对模型说「已在上文提供」，而模型**看不到**
   ⇒ 静默降准（`faithfulness` 照样全绿，因为剩下的内容仍"可被支撑"）。
2. ★★ **`seen` 必须按"将要注入的字符数"比较，不能只记布尔"见过"**：
   同一条 chunk 在 `retrieve` 里被截到 **200 字**、在 `pdf_extract` 里是**全文**。
   只记布尔 ⇒ 第二次的全文被判为重复 ⇒ **模型失去它正要取回的内容**。
   （这条是**实现过程中纠正的真缺陷**，见 `test_truncated_then_full_text_is_not_deduped`。）
3. **去重不删信息**：重复命中仍返回 `id` / `page` / `score` / **首句**。
4. **只在本轮内去重**：跨轮历史可能已被 `_compact_history` 压缩 ⇒ 同样会撒谎。
"""

from __future__ import annotations

import asyncio
import logging
import re

from lightcore.core.loop import (
    AgentLoop,
    LoopConfig,
    _dedup_tool_output,
    _prepare_tool_output,
)
from lightcore.llm.base import ChatMessage, LLMResponse, Role, ToolCall
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry, ToolResult, ToolSpec

LOGGER_NAME = "lightcore.core.loop"
_TEXT_BODY = "第一章 引言。全文很长。" * 40  # 440 字，远超 retrieve 的 200 字/段压缩上限


def _results_output(ids=("doc-1-c0", "doc-1-c1"), text: str = _TEXT_BODY) -> dict:
    return {"results": [{"id": i, "page": 1, "score": 0.9, "text": text} for i in ids]}


# ============ ① 纯函数：results 型（retrieve / pdf_extract(query=)） ============

class TestDedupResults:
    def test_first_call_injects_full_text(self):
        out, keys, dropped = _dedup_tool_output("retrieve", _results_output(), {})
        assert dropped == 0, "首次调用不应去重"
        assert set(keys) == {"chunk:doc-1-c0", "chunk:doc-1-c1"}
        assert out["results"][0]["text"].startswith("第一章"), "首次必须注入正文"

    def test_second_call_keeps_ids_but_drops_body(self):
        """★ 重复命中必须**保留 id/page/score/首句**（不删信息），只降级正文。"""
        seen: dict[str, int] = {}
        _, keys, _ = _dedup_tool_output("retrieve", _results_output(), seen)
        seen.update(keys)

        out, new_keys, dropped = _dedup_tool_output("retrieve", _results_output(), seen)
        assert dropped == 2, f"两条命中都应被去重：{dropped}"
        assert new_keys == {}, "重复命中**不得**再次登记键（否则撤销语义错乱）"

        first = out["results"][0]
        assert first["id"] == "doc-1-c0", "id 必须保留（引用溯源依赖它）"
        assert first["page"] == 1 and first["score"] == 0.9, "page/score 必须保留"
        assert "重复命中" in first["text"], f"正文应降级为占位：{first['text'][:40]}"
        assert "第一章" in first["text"], "占位里必须带**首句**，否则模型无法辨认是哪一段"
        assert len(first["text"]) < 200, "占位必须显著短于原文"

    def test_truncated_then_full_text_is_not_deduped(self):
        """★★ 实现中纠正的真缺陷：**更长的正文是"新信息"，不能被当成重复。**

        `retrieve` 注入的是经 `_prepare_tool_output` **截断到 200 字**的片段，
        而 `pdf_extract(query=)` 注入的是**全文**。同一条 chunk 在两边"原始长度"相同，
        若去重跑在压缩前、或只记布尔"见过"，第二次的全文就会被换成 60 字占位
        ⇒ 模型失去它**正要取回**的内容 —— 那是"改内容（丢证据）"，不是"改分布"。
        """
        seen: dict[str, int] = {}

        # ① retrieve → 压缩后再去重：上下文里只有截断稿
        prepared = _prepare_tool_output("retrieve", _results_output(("doc-1-c0",)))
        _, keys, _ = _dedup_tool_output("retrieve", prepared, seen)
        seen.update(keys)
        assert seen["chunk:doc-1-c0"] <= 201, f"上下文里只该有截断稿：{seen}"

        # ② pdf_extract → 全文（明显更长）⇒ 必须原样注入
        out, new_keys, dropped = _dedup_tool_output(
            "pdf_extract", _results_output(("doc-1-c0",)), seen,
        )
        assert dropped == 0, "更长的正文是新信息，不能去重"
        assert out["results"][0]["text"].startswith("第一章"), "全文必须原样注入"
        assert new_keys["chunk:doc-1-c0"] > 201, "记录应抬高到全文长度"

        # ③ 反向（这才是真重复）：全文已在上下文后，截断稿再来 ⇒ 去重
        seen.update(new_keys)
        prepared2 = _prepare_tool_output("retrieve", _results_output(("doc-1-c0",)))
        _, _, dropped2 = _dedup_tool_output("retrieve", prepared2, seen)
        assert dropped2 == 1, "截断稿不比已注入的更长 ⇒ 不带来新信息，应去重"

    def test_dedup_across_tools_when_second_is_not_longer(self):
        """★ 跨工具去重**只在"第二次不比第一次更长"时成立**。

        `pdf_extract(query=)` 先注全文，之后 `retrieve` 命中同一批 chunk（截断稿）
        ⇒ 截断稿不带来新信息 ⇒ 去重。
        ⚠️ **反方向（retrieve 截断 → pdf_extract 全文）刻意不去重** —— 全文是新信息，
        见 `test_truncated_then_full_text_is_not_deduped`。
        ⇒ 这条边界决定了去重的**实际收益上限**：重叠本身（实测下限 36%）不等于可省的量。
        """
        seen: dict[str, int] = {}
        full = _prepare_tool_output("pdf_extract", _results_output())
        _, keys, _ = _dedup_tool_output("pdf_extract", full, seen)
        seen.update(keys)

        prepared = _prepare_tool_output("retrieve", _results_output())  # 截断稿
        _, _, dropped = _dedup_tool_output("retrieve", prepared, seen)
        assert dropped == 2, f"截断稿不比全文更长 ⇒ 应去重：{dropped}"

    def test_partial_overlap_only_drops_the_duplicate(self):
        seen: dict[str, int] = {}
        _, keys, _ = _dedup_tool_output("retrieve", _results_output(("doc-1-c0",)), seen)
        seen.update(keys)

        out, new_keys, dropped = _dedup_tool_output(
            "retrieve", _results_output(("doc-1-c0", "doc-1-c9")), seen,
        )
        assert dropped == 1 and set(new_keys) == {"chunk:doc-1-c9"}
        assert "重复命中" in out["results"][0]["text"]
        assert out["results"][1]["text"].startswith("第一章"), "新命中必须保留全文"

    def test_duplicate_inside_one_output_also_deduped(self):
        out, keys, dropped = _dedup_tool_output(
            "retrieve", _results_output(("doc-1-c0", "doc-1-c0")), {},
        )
        assert dropped == 1 and set(keys) == {"chunk:doc-1-c0"}
        assert "重复命中" in out["results"][1]["text"]

    def test_non_retrieval_output_untouched(self):
        payload = {"ok": True, "stdout": "print(1)"}
        out, keys, dropped = _dedup_tool_output("code_interpreter", payload, {})
        assert out is payload and keys == {} and dropped == 0


# ============ ② 纯函数：页模式（pdf_extract(page=N)） ============

class TestDedupPageMode:
    @staticmethod
    def _page(page: int = 8, text: str = "第 8 页正文。") -> dict:
        return {"page": page, "text": text * 50, "num_chunks": 3, "source": "pdf"}

    def test_same_page_second_time_is_deduped(self):
        seen: dict[str, int] = {}
        _, keys, dropped = _dedup_tool_output("pdf_extract", self._page(), seen)
        assert dropped == 0 and len(keys) == 1
        seen.update(keys)

        out, new_keys, dropped = _dedup_tool_output("pdf_extract", self._page(), seen)
        assert dropped == 1 and new_keys == {}
        assert out["page"] == 8, "页码必须保留"
        assert "重复命中" in out["text"] and "第 8 页正文" in out["text"]

    def test_same_page_number_with_different_content_not_deduped(self):
        """★ 多文档场景：两份文档都有第 8 页 —— 内容不同 ⇒ **不得**误判为重复。"""
        seen: dict[str, int] = {}
        _, keys, _ = _dedup_tool_output("pdf_extract", self._page(text="A 文档第 8 页。"), seen)
        seen.update(keys)

        _, new_keys, dropped = _dedup_tool_output(
            "pdf_extract", self._page(text="B 文档第 8 页。"), seen,
        )
        assert dropped == 0 and len(new_keys) == 1, "不同内容同页码不能去重"

    def test_empty_page_not_deduped(self):
        _, keys, dropped = _dedup_tool_output(
            "pdf_extract", {"page": 8, "text": "  ", "num_chunks": 0, "source": "pdf"}, {},
        )
        assert dropped == 0 and keys == {}


# ============ ③ 硬约束①：闸门裁剪必须撤销记录 ============

class _StubLLM:
    model_name = "stub"

    async def chat(self, *a, **kw):  # noqa: ANN002, ANN003
        raise AssertionError("本组测试不应真正调用 LLM")


def _mk_loop(**cfg) -> AgentLoop:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="retrieve", description="t",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission=Permission.SAFE, handler=lambda **_kw: ToolResult(ok=True, output={}),
    ))
    return AgentLoop(_StubLLM(), ToolExecutor(reg), LoopConfig(**cfg))


class TestGateSyncsDedup:
    """★★ 硬约束①：闸门把某条工具结果**替换**成裁剪标记后，它贡献的记录必须一并撤销。

    不撤销的后果是**静默降准**：模型上文已看不到该内容，去重却告诉它「已提供」。
    这类错误 judge 测不出来（剩余内容仍可被上下文支撑 ⇒ `faithful=True`）。
    """

    def test_elided_tool_result_revokes_its_keys(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        loop = _mk_loop(max_tokens_budget=1000)  # 触发线 = 1000 × 0.7 = 700
        tool_msg = ChatMessage(role=Role.TOOL, content="X" * 2000, tool_call_id="c1")
        messages = [
            ChatMessage(role=Role.SYSTEM, content="sys"),
            ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id="c1", name="retrieve", arguments={"query": "q"}),
            ]),
            tool_msg,
            # 尾部填充：`_GATE_KEEP_TAIL=6` 保护最后 6 条，故待裁条必须落在保护区之外
            *[ChatMessage(role=Role.USER, content="pad") for _ in range(7)],
        ]
        loop._text_keys = {"chunk:doc-1-c0": 200, "chunk:doc-1-c1": 200}
        loop._text_keys_by_msg = {id(tool_msg): ["chunk:doc-1-c0", "chunk:doc-1-c1"]}
        loop._last_real_prompt_tokens = 800  # 已 > 触发线 700
        loop._gate_mark = len(messages)

        loop._gate_context(messages)  # noqa: SLF001  直调闸门，精确覆盖该分支

        assert "已裁剪" in messages[2].content, "前置条件：该条确实被裁掉了"
        assert loop._text_keys == {}, (
            "★★ 被裁掉的内容必须从去重记录撤销 —— 否则去重会声称「已在上文提供」，"
            "而模型其实看不到它（静默降准，judge 测不出）"
        )
        assert loop._text_keys_by_msg == {}, "反向索引也要一并清理，避免悬垂"

    def test_untouched_tool_result_keeps_keys(self):
        """对照组：没有触发裁剪时，记录必须**原样保留**（否则去重形同失效）。"""
        loop = _mk_loop(max_tokens_budget=100_000)  # 触发线 70,000，远超投影
        tool_msg = ChatMessage(role=Role.TOOL, content="X" * 2000, tool_call_id="c1")
        messages = [ChatMessage(role=Role.SYSTEM, content="sys"), tool_msg]
        loop._text_keys = {"chunk:doc-1-c0": 200}
        loop._text_keys_by_msg = {id(tool_msg): ["chunk:doc-1-c0"]}
        loop._last_real_prompt_tokens = 100
        loop._gate_mark = len(messages)

        loop._gate_context(messages)  # noqa: SLF001

        assert loop._text_keys == {"chunk:doc-1-c0": 200}, "未裁剪 ⇒ 记录不应被撤销"

    def test_gate_works_on_minimal_double_without_dedup_state(self, caplog):
        """★ **闸门不得要求替身具备去重状态**（2026-09-28 实测踩到的真回归）。

        `tests/test_token_gate.py` 用 `object.__new__(AgentLoop)` 只塞闸门所需的最小字段。
        撤销代码若直接访问 `_text_keys*`，会抛 `AttributeError`，把**本来通过的用例弄红** ——
        与 2026-09-26「耗时日志反过来弄坏 `_RecordingLLM` 替身」是同一类错误。
        ⇒ 判据：**新增的横切状态一律 `getattr` 兜底**，替身没有它时撤销本就无事可做。
        """
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        loop = object.__new__(AgentLoop)  # 只塞闸门最小字段，**不跑 __init__**
        loop.config = LoopConfig(max_tokens_budget=1000)
        loop._last_real_prompt_tokens = 700
        loop._steps = 0
        tool_msg = ChatMessage(role=Role.TOOL, content="X" * 3000, tool_call_id="call-A")
        messages = [tool_msg, *[ChatMessage(role=Role.USER, content="填充") for _ in range(9)]]
        loop._gate_mark = 3

        loop._gate_context(messages)  # noqa: SLF001  不得抛异常

        assert messages[0].content.startswith("[早期工具结果已裁剪"), "裁剪本身仍须生效"
        assert messages[0].tool_call_id == "call-A", "tool_call_id 必须保留（配对不破）"


# ============ ④ 端到端：AgentLoop 内两次取回同一批内容 ============

class _ThreeStepLLM:
    """前两轮都调 retrieve（**结果完全相同**），第三轮给最终答案。"""

    def __init__(self) -> None:
        self.model_name = "test-model"
        self.n = 0

    async def chat(self, messages, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        self.n += 1
        if self.n <= 2:
            return LLMResponse(
                message=ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
                    ToolCall(id=f"c{self.n}", name="retrieve", arguments={"query": "q"}),
                ]),
                prompt_tokens=100, completion_tokens=10, latency_ms=5, model=self.model_name,
            )
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content="最终答案"),
            prompt_tokens=500, completion_tokens=20, latency_ms=5, model=self.model_name,
        )


def _registry(handler) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="retrieve", description="测试用工具",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission=Permission.SAFE, handler=handler,
    ))
    return reg


def _retrieve_ok(**_kw):
    return ToolResult(ok=True, output=_results_output())


_TOOL_LOG = re.compile(r"原始=(\d+)字 → 注入=(\d+)字（≈\d+ tokens）\| 去重=(\d+) \|")


def _tool_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[工具]" in r.getMessage()]


def _one_turn(loop: AgentLoop) -> None:
    async def _go():
        return [ev async for ev in loop.run_stream("问题", stream_tokens=False)]

    asyncio.run(_go())


class TestEndToEnd:
    def test_second_identical_retrieve_is_deduped_in_context(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        loop = AgentLoop(_ThreeStepLLM(), ToolExecutor(_registry(_retrieve_ok)), LoopConfig())

        _one_turn(loop)

        lines = _tool_lines(caplog)
        assert len(lines) == 2, f"应有两次工具调用日志：{lines}"
        m1, m2 = (_TOOL_LOG.search(x) for x in lines)
        assert m1 and m2, f"日志格式不含去重计数：{lines}"
        assert int(m1.group(3)) == 0, f"首次调用不该去重：{lines[0]}"
        assert int(m2.group(3)) == 2, f"第二次同结果应去重 2 条：{lines[1]}"
        assert int(m2.group(2)) < int(m1.group(2)), (
            f"第二次实际注入必须更小（{m1.group(2)} → {m2.group(2)} 字）"
        )

    def test_dedup_does_not_leak_across_turns(self, caplog):
        """★ 硬约束④：去重范围 = **本轮**。同一个 loop 实例跑第二轮时须重新全量注入。

        跨轮去重会撒谎 —— 跨轮历史可能已被 `_compact_history` 压缩过。
        """
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        llm = _ThreeStepLLM()
        loop = AgentLoop(llm, ToolExecutor(_registry(_retrieve_ok)), LoopConfig())

        _one_turn(loop)
        turn1 = _tool_lines(caplog)

        llm.n = 0  # 第二轮用同一剧本（LLM 策略本身不属于本轮状态）
        caplog.clear()
        _one_turn(loop)
        turn2 = _tool_lines(caplog)

        assert len(turn1) == 2 and len(turn2) == 2, (turn1, turn2)
        m1 = _TOOL_LOG.search(turn1[0])
        m2 = _TOOL_LOG.search(turn2[0])
        assert m1 and m2, (turn1, turn2)
        assert int(m2.group(3)) == 0, f"第二轮首次取数不得因上一轮而去重：{turn2[0]}"
        assert m2.group(2) == m1.group(2), "两轮首次注入量必须一致（证明记录已按轮清空）"
