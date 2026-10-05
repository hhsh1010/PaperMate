"""P0-1 回复输出过滤：SYSTEM_PROMPT 硬性过滤规则 + ApiFakeLLM 摘要式回显

背景：回复中夹杂内部类名/JSON/代码细节。本测试锁定两类修复：
1. SYSTEM_PROMPT 必须包含『输出过滤』硬性规则（禁止类名/原始 JSON/代码片段）；
2. ApiFakeLLM 对检索结果的回显必须是摘要式（首句截断），不得再逐条输出全文与内部 id 结构。
"""

from __future__ import annotations

from lightcore.llm.base import Role
from papermate.services.chat_service import SYSTEM_PROMPT, ApiFakeLLM, _first_sentence


class TestSystemPromptFilter:
    def test_contains_output_filter_section(self):
        assert "输出过滤" in SYSTEM_PROMPT
        assert "严禁" in SYSTEM_PROMPT

    def test_forbids_internal_symbols(self):
        # 明确列出会泄露的内部符号
        for sym in ("ChatMessage", "AgentLoop", "RAGPipeline", "meta_json", "doc_ids"):
            assert sym in SYSTEM_PROMPT, f"过滤规则应显式点名内部符号 {sym}"

    def test_forbids_raw_json_and_code(self):
        assert "原始 JSON" in SYSTEM_PROMPT
        assert "代码" in SYSTEM_PROMPT


class TestReplyFormatNoProcess:
    """回复模板不得要求『过程 + 结果』结构，且必须显式禁止流程叙述（意图识别/文档检索/结果分析等）。"""

    def test_no_process_result_structure_required(self):
        assert "过程 + 结果" not in SYSTEM_PROMPT
        assert "采用『过程" not in SYSTEM_PROMPT

    def test_forbids_process_narration(self):
        # 格式规范必须显式禁止『过程：』类流程叙述
        assert "严禁输出『过程：』" in SYSTEM_PROMPT
        for marker in ("意图识别：", "文档检索：", "结果分析："):
            assert marker in SYSTEM_PROMPT, f"格式规范应显式点名禁止叙述 {marker}"

    def test_intent_classification_is_internal(self):
        # 意图分类须标注为内部判断，回复中不得提及分类过程/结果
        assert "内部完成" in SYSTEM_PROMPT
        assert "严禁提及分类过程或分类结果" in SYSTEM_PROMPT


class TestReasoningLanguage:
    """推理链会**直接展示在用户界面**（前端「推理过程」面板）⇒ 必须约束为简体中文（2026-09-28）。

    ★ 为什么值得单独钉一条：这条约束**没有任何自动化指标能兜住** ——
    judge 只测 `faithfulness`/`abstention`，推理链全是英文照样全绿。
    ⇒ 只能靠"提示词确实写了这条"来保证，故在此断言。
    """

    def test_system_prompt_requires_chinese_reasoning(self):
        assert "思考过程也必须用简体中文" in SYSTEM_PROMPT
        # 必须同时给出豁免口径：否则模型会为了"说中文"把检索主题词也译掉，
        # 而规则 2c 要求 retrieve 的 query 写 abstract / experiment 这类主题词。
        assert "保留原文" in SYSTEM_PROMPT

    def test_direct_answer_paths_share_the_rule(self):
        """直答 / 寒暄走的是**独立** system 常量（不是 `SYSTEM_PROMPT`）⇒ 必须同口径。

        这正是"同名两份、改一处漏一处"的高危位置：Agent 路径约束了、直答路径没约束，
        用户就会看到"有时中文思考、有时英文"，而**没有任何测试会报警**。
        """
        from papermate.api.routes_search import _CHITCHAT_SYSTEM, _GENERAL_QA_SYSTEM

        for name, prompt in (("_GENERAL_QA_SYSTEM", _GENERAL_QA_SYSTEM),
                             ("_CHITCHAT_SYSTEM", _CHITCHAT_SYSTEM)):
            assert "思考过程" in prompt and "中文" in prompt, f"{name} 缺少中文思考约束"


class TestApiFakeLLMSummary:
    def _feed_result(self, text: str):
        llm = ApiFakeLLM()
        return llm._decide(f"RESULT:{text}", [])

    def test_result_rendered_as_summary_not_fulltext(self):
        long_text = "这是第一句核心结论。这是第二句非常长的补充说明，包含了大量与问题无关的细节与内部符号 blah blah。" * 3
        payload = f'{{"source": "doc-1", "results": [{{"id": "doc-1-c0", "text": "{long_text}"}}]}}'
        msg = self._feed_result(payload)
        assert msg.role == Role.ASSISTANT
        # 摘要式：不再输出 `• [doc-1-c0] 全文`
        assert "[doc-1-c0]" not in msg.content
        # 首句保留、且只保留一句（不再出现第二句的全文）
        assert "这是第一句核心结论" in msg.content
        assert "这是第二句非常长的补充说明" not in msg.content

    def test_empty_result_keeps_note(self):
        msg = self._feed_result('{"source": "none", "results": [], "note": "当前会话未关联文档"}')
        assert "未检索到相关段落" in msg.content
        assert "当前会话未关联文档" in msg.content


class TestFirstSentence:
    def test_takes_first_sentence(self):
        assert _first_sentence("第一句。第二句") == "第一句。"

    def test_short_no_ellipsis(self):
        assert "…" not in _first_sentence("短句。")

    def test_truncates_long(self):
        out = _first_sentence("x" * 200)
        assert len(out) == 90 + 1  # 90 字符 + "…"

    def test_handles_empty(self):
        assert _first_sentence("") == ""


class TestNoDocDirectAnswerPrompt:
    """无文档+未联网态：通用回答直接路径（_GENERAL_QA_SYSTEM）与 FakeLLM 兜底文案均不得含流程叙述。"""

    def test_general_qa_system_forbids_process(self):
        from papermate.api.routes_search import _GENERAL_QA_SYSTEM

        assert "严禁输出『过程』板块" in _GENERAL_QA_SYSTEM
        for marker in ("意图识别：", "文档检索：", "结果分析："):
            assert marker in _GENERAL_QA_SYSTEM
        assert "不要模仿" in _GENERAL_QA_SYSTEM  # 防历史消息污染
        # 弱化措辞已被替换
        assert "不要分『过程』和『结果』" not in _GENERAL_QA_SYSTEM

    def test_grounding_block_no_docs_forbids_process(self):
        from papermate.services.chat_service import _grounding_block

        for net in (False, True):
            block = _grounding_block(net, has_docs=False)
            assert "不要输出『过程：』" in block
