"""工具「用法引导」的守护测试（2026-09-28）。

## 背景：这两个工具在 schema 层几乎不可区分

实测（`docs/去重换策略-影响面评估.md`）：`pdf_extract(query=)` 12 次调用占 **46%** 注入量、
压缩比 **100%**；而 `retrieve` 压缩比 26%。原因是两者的 `query` 描述几乎同义
（`retrieve`："检索查询" / `pdf_extract`："**检索查询，返回最相关段落**"），
**模型无法从描述里分辨**，只能靠猜 —— 猜错代价 4 倍。

## 本文件守什么

1. **引导必须能区分两者**：`pdf_extract.query` 的语义被限定为「**章节主题词**」，
   且显式劝阻事实型提问（这才是"治本"：只改**引导**，不削减能力）。
2. ★ **schema 不许撒谎**：`parameters.*.default` 必须与 handler 签名的真实默认值**同源**。
   实测踩到 —— schema 写 `top_k: 6`，而 `_make_pdf_extract` 签名是 `top_k: int = 3`；
   `ToolExecutor.execute` 是直接 `handler(**arguments)`，**不会**套用 schema 的 default
   ⇒ 文档与行为不符（模型按 6 规划、实得 3）。
3. **提示词要有反向约束**：2c 只教了"章节没覆盖就 pdf_extract 取全文"，
   会造成"事实型提问也去取全文"的误用，必须补上反向指引。
"""

from __future__ import annotations

import inspect

from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolRegistry
from papermate.prompts.system import SYSTEM_PROMPT

_TOOL_NAMES = ("retrieve", "pdf_extract")


def _specs() -> dict:
    reg = ToolRegistry()
    register_builtin_tools(reg)
    return {n: reg.get(n) for n in _TOOL_NAMES}


class TestToolDescriptions:
    def test_pdf_extract_query_is_scoped_to_section_topics(self):
        spec = _specs()["pdf_extract"]
        desc = spec.parameters["properties"]["query"]["description"]
        assert "章节主题词" in desc, f"query 必须被限定为章节主题词：{desc}"
        assert "不要" in desc, "必须显式劝阻事实型提问（否则模型仍会误用）"

    def test_pdf_extract_warns_about_uncompressed_cost(self):
        spec = _specs()["pdf_extract"]
        assert "未压缩" in spec.description and "4 倍" in spec.description, (
            f"必须点明返回未压缩、注入量约为 retrieve 的 4 倍：{spec.description[:120]}"
        )

    def test_retrieve_claims_fact_lookup(self):
        """★ 光说 pdf_extract 不该做什么不够 —— 还要**正面指引**事实型提问去 retrieve。"""
        spec = _specs()["retrieve"]
        assert "事实" in spec.description, f"retrieve 必须正面承接事实型提问：{spec.description[:120]}"
        assert "压缩" in spec.description, "应说明它返回的是已压缩片段"

    def test_both_tools_mention_no_repeat(self):
        """重复取数不会二次注入 ⇒ 提示词要如实告诉模型，避免它反复试探。"""
        spec = _specs()["pdf_extract"]
        assert "重复" in spec.description, f"应提示同一页/章不要重复取：{spec.description[:120]}"


class TestSchemaDefaultsMatchImplementation:
    """★★ schema 的 `default` 必须与 handler 签名的真实默认值一致。

    `ToolExecutor.execute` 直接 `handler(**arguments)` ⇒ **schema 的 default 不生效**。
    两者不一致时，模型看到的是 6、实际拿到的是 3 —— 这是"文档撒谎"，
    会让模型对上下文预算的规划失真（而预算恰恰是紧的）。
    """

    def test_tool_k_defaults_are_honest(self):
        for name, spec in _specs().items():
            schema_default = spec.parameters["properties"]["top_k"].get("default")
            impl_default = inspect.signature(spec.handler).parameters["top_k"].default
            assert schema_default == impl_default, (
                f"{name}.top_k 的 schema 默认值 {schema_default} 与实现 {impl_default} 不一致 —— "
                "schema.default 不会被执行器套用，不一致等于文档撒谎"
            )
            assert str(schema_default) in spec.parameters["properties"]["top_k"]["description"], (
                f"{name}.top_k 的描述里也应写明真实默认值 {schema_default}"
            )


class TestPromptReverseConstraint:
    def test_system_prompt_forbids_pdf_extract_for_fact_questions(self):
        """2c 只教了「章节没覆盖就 pdf_extract 取全文」，必须补反向约束 —— 否则指令本身在鼓励误用。"""
        assert "2c 反向" in SYSTEM_PROMPT, "提示词缺少 2c 的反向约束段"
        idx = SYSTEM_PROMPT.index("2c 反向")
        clause = SYSTEM_PROMPT[idx : idx + 120]
        assert "pdf_extract(query=)" in clause, f"反向约束必须点名 pdf_extract(query=)：{clause}"
        assert "不要" in clause, f"必须是明确的禁止语气（否则模型会两边都用）：{clause}"
        assert "retrieve" in clause, f"同时要正面指向 retrieve：{clause}"

    def test_system_prompt_tells_model_not_to_repeat_fetches(self):
        """断言必须**限定在反向约束段内** —— 全局 `in` 会被提示词别处的"重复"字样命中而失效。"""
        idx = SYSTEM_PROMPT.find("2c 反向")
        assert idx >= 0, "提示词缺少 2c 的反向约束段"
        clause = SYSTEM_PROMPT[idx : idx + 140]
        assert "重复" in clause, f"2c 反向段应告知重复调用是纯浪费：{clause}"
