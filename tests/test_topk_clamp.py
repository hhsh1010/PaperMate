"""B1（2026-09-28）：`top_k` 的**上界 + 代码 clamp** —— 防"先花钱才发现撑爆"。

## 被修的缺陷

`retrieve` / `pdf_extract(query=)` 的 `top_k` 原先**既无 schema 上界、代码也不 clamp**：
模型只要传 `pdf_extract(query='abstract', top_k=50)` 就能**单次注入数十万字符**
（`pdf_extract` 返回**未压缩全文**，单块常见 1~2k 字）。

**为什么护栏在最需要它的时刻失效**：`loop._run_stream_impl` 里"单次请求 prompt 超限"
的判据在**响应返回之后**才执行 —— 此时该次调用**已经发出并计费**。
⇒ 顺序是"**先花钱 → 再发现撑爆 → 才中止**"。

## 本文件在守什么

1. **三层里只有代码 clamp 对所有权都有效**：`ToolExecutor.execute` 是直接
   `handler(**arguments)`，**不做参数校验**；schema 的 `maximum` 只对"守规矩的模型"生效。
   ⇒ 必须有一个**纯函数**可单独测（`clamp_top_k`）。
2. ★ **clamp 命中必须告知模型**：不告知 ⇒ 模型以为"整章都拿到了" ⇒ 基于不完整内容
   给出结论（**静默降准**，judge 测不出）。
3. ★ **`page=` 模式不得被 clamp 波及**：该模式不用 `top_k`，若给它附上限流提示，
   模型会以为"按页取全文也被截断了"（**误导**）。
4. **上界必须与 schema 同源**：代码里 clamp 用 20/12，schema 里写 20/12 ——
   两处各写一遍字面量就是下一个"文档撒谎"（同 `top_k` 默认值 6 vs 3 的旧账）。
"""

from __future__ import annotations

import asyncio

import pytest

from lightcore.rag.pipeline import RetrievedChunk
from lightcore.tools.builtin import basic
from lightcore.tools.builtin.basic import (
    PDF_EXTRACT_TOP_K_MAX,
    RETRIEVE_TOP_K_MAX,
    _make_pdf_extract,
    _make_retrieve,
    clamp_top_k,
)


class _Pipe:
    """记录**实际收到的 top_k**（clamp 是否生效的唯一可信证据）。"""

    def __init__(self, n: int = 50, text_len: int = 800) -> None:
        self.received: list[int] = []
        self._text = "正文段落。" * (text_len // 5)

    def get_chunks(self):  # noqa: ANN201  pdf_extract 需先取全量 chunk
        return [RetrievedChunk(id=f"doc-1-c{i}", text=self._text, page=1, score=0.9)
                for i in range(50)]

    def query(self, _q: str, top_k: int = 6):  # noqa: ANN201
        self.received.append(top_k)
        return [RetrievedChunk(id=f"doc-1-c{i}", text=self._text, page=i + 1, score=0.9)
                for i in range(top_k)]


# ------------------------------------------------ ① 纯函数：clamp_top_k 口径表


@pytest.mark.parametrize("raw,expected_k,has_note", [
    (None, 6, False),        # 缺省 → 默认值，无需提示
    (3, 3, False),           # 正常值 → 原样
    (20, 20, False),         # **边界值本身不算超限**（<= maximum 不触发）
    (21, 20, True),          # 越界 1 即 clamp
    (500, 20, True),
    (0, 1, True),            # 下界
    (-7, 1, True),
    ("5", 5, False),         # 模型常把数字写成字符串
    ("abc", 6, True),        # 脏参数**不得抛异常**（同 memory_search 的 K7 口径）
    (2.9, 2, False),         # int() 截断
])
def test_clamp_top_k_contract(raw, expected_k, has_note):
    k, note = clamp_top_k(raw, maximum=RETRIEVE_TOP_K_MAX, default=6, tool="retrieve")

    assert k == expected_k
    assert bool(note) is has_note, f"raw={raw!r} 的提示有无不符：{note!r}"


def test_clamp_note_tells_the_model_it_was_limited():
    """★ 提示必须让模型知道「没取全」并给出补救路径，否则它会基于不完整内容下结论。"""
    _k, note = clamp_top_k(999, maximum=PDF_EXTRACT_TOP_K_MAX, default=3, tool="pdf_extract")

    assert note and "已限制为 12" in note, note
    assert "pdf_extract(page=N)" in note, f"必须给出补救路径：{note}"


def test_clamp_logs_a_warning(caplog):
    """clamp 是**异常信号**（说明模型在越界），必须能在 app.log 里看见。"""
    import logging

    caplog.set_level(logging.WARNING, logger="lightcore.tools.builtin.basic")
    clamp_top_k(99, maximum=RETRIEVE_TOP_K_MAX, default=6, tool="retrieve")

    assert any("top_k" in r.getMessage() and "clamp" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


# ------------------------------------------------ ② 真实 handler：clamp 生效


def test_retrieve_clamps_before_calling_pipeline():
    """★ 必须在 `pipeline.query` **之前** clamp —— 否则"钱已花出去"这一点没被修掉。"""
    pipe = _Pipe()
    res = asyncio.run(_make_retrieve(pipe)("查询", top_k=500))

    assert pipe.received == [RETRIEVE_TOP_K_MAX], f"管线收到的应是 clamp 后的值：{pipe.received}"
    assert len(res.output["results"]) <= RETRIEVE_TOP_K_MAX
    assert "已限制为 20" in res.output.get("note", ""), res.output.get("note")


def test_pdf_extract_query_clamps_and_notifies():
    pipe = _Pipe()
    res = asyncio.run(_make_pdf_extract(pipe)(query="abstract", top_k=500))

    assert pipe.received == [PDF_EXTRACT_TOP_K_MAX], pipe.received
    assert len(res.output["results"]) <= PDF_EXTRACT_TOP_K_MAX
    assert "已限制为 12" in res.output.get("note", ""), res.output.get("note")


def test_pdf_extract_page_mode_is_not_clamped_nor_noted():
    """★ `page=` 模式不使用 top_k ⇒ **既不该 clamp、也不该附限流提示**（否则误导模型）。"""
    pipe = _Pipe()
    # 故意同时传一个超大 top_k：page 模式必须完全忽略它（含提示）
    res = asyncio.run(_make_pdf_extract(pipe)(page=1, top_k=500))

    assert pipe.received == [], "page 模式不应调用 pipeline.query"
    assert "已限制为" not in (res.output.get("note") or ""), (
        f"page 模式被附上了与它无关的限流提示：{res.output.get('note')}"
    )


def test_normal_top_k_is_untouched():
    """反向自检：正常取值不得被 clamp、也不得出现任何提示（防"永远在报警"）。"""
    pipe = _Pipe()
    res = asyncio.run(_make_retrieve(pipe)("查询", top_k=8))

    assert pipe.received == [8]
    assert "已限制为" not in (res.output.get("note") or "")


class _NoQueryPipe:
    """只有 `get_chunks()`、**没有** `query()` 的管线 ⇒ 走 `chunks[:top_k]` 回退分支。

    这是**唯一能返回整篇文档**的路径（回退分支直接切片全量 chunk），
    所以它是"单次注入撑爆"风险最高的一处 —— 必须确认它也吃到了 clamp。
    """

    def get_chunks(self):  # noqa: ANN201
        return [RetrievedChunk(id=f"doc-1-c{i}", text="全文" * 400, page=i + 1, score=0.9)
                for i in range(40)]


def test_pdf_extract_fallback_branch_is_also_clamped():
    """★ 回退分支（无 `query` 方法）也必须在 clamp 之后切片 —— 否则可一次返回全篇。"""
    res = asyncio.run(_make_pdf_extract(_NoQueryPipe())(query="abstract", top_k=999))

    assert len(res.output["results"]) == PDF_EXTRACT_TOP_K_MAX, (
        f"回退分支未受限：取了 {len(res.output['results'])} 块（全量 40）"
    )
    assert "已限制为 12" in res.output.get("note", ""), res.output.get("note")


def test_clamp_note_does_not_clobber_existing_note():
    """提示是**追加**而非覆盖 —— 原有的"部分失败"说明不能被吃掉。"""
    out = basic._merge_note({"note": "原有的部分失败说明"}, "⚠️ 已限制为 12 条")

    assert "已限制为 12 条" in out["note"] and "原有的部分失败说明" in out["note"]


# ------------------------------------------------ ③ 上界必须与 schema 同源


def test_schema_declares_the_same_bounds_as_code():
    """★ 单一事实来源：schema 的 `maximum` 与代码 clamp 用的常量必须**同源**。

    两处各写一遍字面量，就是下一个"schema 说 6 / 实现走 3"（已被
    `test_tool_usage_guidance` 记过一笔的旧账）。
    """
    from lightcore.tools.spec import ToolRegistry

    reg = ToolRegistry()
    basic.register_builtin_tools(reg, rag_pipeline=None)
    specs = {name: reg.get(name) for name in ("retrieve", "pdf_extract")}

    expected = {"retrieve": RETRIEVE_TOP_K_MAX, "pdf_extract": PDF_EXTRACT_TOP_K_MAX}
    for name, maximum in expected.items():
        prop = specs[name].parameters["properties"]["top_k"]
        assert prop.get("maximum") == maximum, f"{name}.top_k maximum 与代码常量不一致：{prop}"
        assert prop.get("minimum") == 1, f"{name}.top_k 缺下界：{prop}"
        assert str(maximum) in prop["description"], (
            f"{name}.top_k 的描述里应写明上限 {maximum}（否则模型不知道边界）"
        )


def test_pdf_extract_bound_is_lower_than_retrieve():
    """★ 量级差异是**有意为之**：`pdf_extract` 返回**未压缩全文**，上界必须更紧。

    这条测试的价值在于：将来有人"顺手统一"两个常量时会被拦住 ——
    统一成大值 ⇒ 单次注入撑爆；统一成小值 ⇒ 整章取不全（质量损失）。
    """
    assert PDF_EXTRACT_TOP_K_MAX < RETRIEVE_TOP_K_MAX
