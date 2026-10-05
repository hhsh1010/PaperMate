"""P1-1 显式意图分类：四类意图判定 + SYSTEM 注记生成。"""

from __future__ import annotations

from papermate.services.intent import classify_intent, intent_system_note


class TestClassifyIntent:
    def test_command(self):
        it = classify_intent("/总结这篇论文")
        assert it.name == "command" and it.force_retrieval is True

    def test_doc_qa_by_marker(self):
        it = classify_intent("这篇论文的摘要结论是什么")
        assert it.name == "doc_qa" and it.force_retrieval is True

    def test_doc_qa_by_question_word_and_reference(self):
        it = classify_intent("论文里提到的因果推断是什么")
        assert it.name == "doc_qa"

    def test_chitchat_short(self):
        assert classify_intent("你好").name == "chitchat"
        assert classify_intent("谢谢").name == "chitchat"

    def test_general_qa(self):
        it = classify_intent("Python 列表推导式怎么用")
        assert it.name == "general_qa"

    def test_empty(self):
        assert classify_intent("").name == "chitchat"

    def test_general_qa_not_hijacked_by_generic_research_words(self):
        """泛研究话题词（模型/方法/综述/创新/指标/介绍…）不得把通用问题误判为 doc_qa。

        根因：这些词曾是 _DOC_QA_MARKERS，导致「大语言模型是怎么工作的」「推荐几篇综述」
        等通用问题被强推进 retrieve，模型再按『文档优先』提示词回出
        「当前会话关联的文档中包含大量…」这类泄露系统状态的话术（用户报的症状）。
        """
        for q in (
            "大语言模型是怎么工作的",
            "目标检测有哪些常用方法",
            "推荐几篇关于目标检测的综述",
            "Transformer 的创新点有哪些",
            "如何评价一个模型的指标",
            "帮我介绍下扩散模型",
            "查询明天的天气",
        ):
            assert classify_intent(q).name == "general_qa", q

    def test_doc_qa_kept_for_document_structure_words(self):
        """真正的文档结构/元数据词仍应归 doc_qa（用户确实在问自己那份文档）。"""
        for q in (
            "这篇论文的摘要讲了什么",
            "第三章的公式是怎么推导的",
            "参考文献有哪些",
            "论文里对比了哪些数据集",
        ):
            assert classify_intent(q).name == "doc_qa", q


class TestIntentSystemNote:
    def test_note_contains_intent_and_hint(self):
        note = intent_system_note("这篇论文的方法是什么")
        assert "意图识别" in note and "文档问答" in note
        assert "retrieve" in note

    def test_chitchat_note_advises_no_retrieval(self):
        note = intent_system_note("你好")
        assert "不要调用检索工具" in note
