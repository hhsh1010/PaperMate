"""P1-2 检索查询改写（ER-12 简化版）：口语 → 检索式 + 缓存 + 降级回退。"""

from __future__ import annotations

import pytest

from lightcore.rag.query_rewrite import expand_sections, rewrite_query, rewrite_query_legacy


class TestRewriteChinese:
    def test_keeps_question_words_but_strips_function_words(self):
        """★ **有意变更**（2026-09-26 ④ 加固）：疑问词**保留**，只删零信息量词。

        为什么不再删疑问词：它们兼作**信号词**。`受什么影响` 删「什么」→ `受影响`
        是**语义反转**（询问影响源 → 陈述被影响）。取证见 `docs/query_rewrite-缺陷取证.md`。
        """
        out = rewrite_query("这篇文章主要讲了什么方法")
        assert "什么" in out, "疑问词是信号词，不能删（会造成语义反转）"
        assert "方法" in out

    def test_strips_filler_by_word_not_substring(self):
        # 「这个」是独立词 → 删；「了解」也是独立词 → **必须保留**
        # （旧实现按子串删「了」→ 把「了解」切成「解概念」）
        assert rewrite_query("了解这个概念") == "了解概念"

    def test_strips_leading_phrase(self):
        out = rewrite_query("请问这篇论文里面关于注意力机制是怎么描述的")
        assert "请问" not in out
        assert "注意力机制" in out

    def test_empty_fallback(self):
        # 全是语气/停用词 → 改写为空 → 回退原始 query（不阻断）
        out = rewrite_query("的吗呢啊")
        assert out == "的吗呢啊"


class TestRewriteEnglish:
    def test_strips_question_and_stop_words(self):
        out = rewrite_query("What is the main contribution of the paper?")
        low = out.lower()
        assert "what" not in low
        assert "the" not in low
        assert "contribution" in low
        assert "paper" in low

    def test_keeps_meaningful_terms(self):
        out = rewrite_query("How does self-attention work in transformer models?")
        low = out.lower()
        assert "self-attention" in low and "transformer" in low


class TestRewriteRobustness:
    def test_command_prefix_stripped(self):
        out = rewrite_query("/总结 这篇论文的实验设计")
        assert out.startswith("/") is False
        assert "实验设计" in out or "实验" in out

    def test_cache_same_result(self):
        a = rewrite_query("Transformer 的自注意力机制原理")
        b = rewrite_query("Transformer 的自注意力机制原理")
        assert a == b  # lru_cache 命中，结果一致


# ============ T-04+ ④：`rewrite_query` 加固（jieba 词边界）============
#
# 为什么单独守这一组：④ 把中文改写从「裸 `str.replace()` 子串替换」换成
# 「jieba 分词 + **整词**删除」。这类改动最容易「修好 A 又破坏 B」，
# 因此把**每一个已知破坏案例**逐条钉成参数化用例。
# 取证：docs/query_rewrite-缺陷取证.md


class TestRewriteQueryNoLongerBreaksWords:
    """④ 的核心契约：**不得把实词切成碎片**（旧实现在此全军覆没）。"""

    @pytest.mark.parametrize(("query", "must_contain"), [
        ("目的是什么", "目的"),                  # 旧：删「什么」→「目的是」→删「的」→ **`目是`**
        ("了解这个概念", "了解"),                # 旧：删「这个」→「了解概念」→删「了」→ **`解概念`**
        ("失败原因为什么必须显式记录", "原因"),    # 旧：删「什么」暴露「因为」→ **`失败原必须…`**
        ("标的是什么", "标的"),                  # 旧：→ **`标是`**
        ("定义的边界在哪", "定义"),              # 旧：删「的」→ **`义边界在哪`**
    ])
    def test_keeps_content_words(self, query, must_contain):
        out = rewrite_query(query)
        assert must_contain in out, f"实词被削碎：{query!r} → {out!r}"

    @pytest.mark.parametrize("query", [
        "重排的开销主要受什么影响",
        "取消应该被记成什么状态",
    ])
    def test_no_semantic_inversion(self, query):
        """疑问词必须保留 —— 删掉会造成**语义反转**（`受什么影响` → `受影响`）。"""
        out = rewrite_query(query)
        assert "受影响" not in out, f"出现语义反转：{query!r} → {out!r}"
        assert "什么" in out, f"疑问词被删（信号丢失）：{query!r} → {out!r}"

    def test_english_path_not_modified(self):
        """英文有天然词边界（`\\b`），④ **未改**英文路径 —— 改它只会引入风险。"""
        out = rewrite_query("What is the main contribution of the paper?")
        assert "contribution" in out and "paper" in out
        assert "what" not in out.lower() and "the" not in out.lower()


class TestRewriteQueryStillCleans:
    """④ 必须**保住清洗价值**（否则不如直接用 `expand_sections`）。"""

    def test_removes_particles(self):
        # 注意「结论」会触发章节展开（追加 conclusion），故只断言语气词被删
        out = rewrite_query("结论是什么呀")
        assert "呀" not in out
        assert "结论是什么" in out

    def test_removes_demonstratives(self):
        assert rewrite_query("了解这个机制") == "了解机制"

    def test_removes_leading_phrase(self):
        assert "请问" not in rewrite_query("请问怎么用")

    def test_removes_structural_particles(self):
        assert rewrite_query("重排的开销") == "重排开销"

    def test_section_expansion_still_works(self):
        assert "experiment" in rewrite_query("总结实验")

    def test_bare_command_still_empty(self):
        # M11 语义不变
        assert rewrite_query("/总结") == ""


class TestRewriteQueryDegradesSafely:
    """jieba 不可用时的降级：**不删任何词**（宁可少洗，不可洗坏）。"""

    def test_no_deletion_without_jieba(self, monkeypatch):
        from lightcore.rag import query_rewrite as qr

        monkeypatch.setattr(qr, "_jieba_lcut", lambda q: ())
        assert qr._rewrite_zh("了解这个机制") == "了解这个机制", "降级路径必须不删词"

    def test_jieba_lcut_returns_tuple(self):
        from lightcore.rag.query_rewrite import _jieba_lcut

        out = _jieba_lcut("目的是什么")
        assert isinstance(out, tuple) and "目的" in out


class TestRewriteQueryLegacyBaseline:
    """`rewrite_query_legacy` 必须**精确复现**加固前行为 —— 对照基线不能漂移。

    为什么守这个：评测用 `--query-rewrite legacy` 复现「改动前基线」。
    若 legacy 被"顺手修正"，基线就不再是基线，`v2 修好了什么` 将无法证明。
    """

    @pytest.mark.parametrize(("query", "expected"), [
        ("取消应该被记成什么状态", "取消应该被记成状态"),
        ("目的是什么", "目是"),
        ("了解这个概念", "解概念"),
        ("重排的开销主要受什么影响", "重排开销受影响"),
        ("失败原因为什么必须显式记录", "失败原必须显式记录"),
    ])
    def test_legacy_reproduces_old_behavior(self, query, expected):
        assert rewrite_query_legacy(query) == expected, "legacy 基线漂移了"

    def test_legacy_and_v2_differ_on_known_bug(self):
        """两者的差异必须体现在**已知缺陷案例**上（否则这个对照组毫无意义）。"""
        for q in ["目的是什么", "了解这个概念", "取消应该被记成什么状态"]:
            assert rewrite_query_legacy(q) != rewrite_query(q), f"{q!r} 上两者应不同"

    def test_legacy_keeps_capabilities(self):
        """legacy 也必须保留 /命令 剥离 + 章节展开（否则差异不纯净）。"""
        assert rewrite_query_legacy("/总结 摘要") == "摘要 abstract"
        assert rewrite_query_legacy("/总结") == ""


# ============ T-04+ ①-b：`expand_sections`（止血版查询预处理）============
#
# 为什么单独守这一组：`expand_sections` 是**上线路径**（`CompositeRAGPipeline` 默认调它），
# 而它的价值恰恰是"**不做什么**"——不做疑问词/停用词清洗。这类"负向契约"最容易在
# 后续重构中被"顺手加回去"，因此必须逐条钉死。
# 取证：docs/query_rewrite-缺陷取证.md


class TestExpandSectionsKeepsQueryIntact:
    """①-b 的核心契约：**不得破坏 query**（旧 `rewrite_query` 在此全军覆没）。"""

    def test_keeps_purpose_word(self):
        # 旧实现：'目的是什么' → '目是'（"的" 吃掉 "目的"、"什么" 吃掉后半）
        assert expand_sections("目的是什么") == "目的是什么"

    def test_keeps_understand_word(self):
        # 旧实现：'了解这个概念' → '解概念'（"了" 吃掉 "了解"）
        assert expand_sections("了解这个概念") == "了解这个概念"

    def test_keeps_reason_word(self):
        # 旧实现：'失败原因为什么必须显式记录' → '失败原必须显式记录'（"因为" 吃掉 "原因"）
        out = expand_sections("失败原因为什么必须显式记录")
        assert "原因" in out, f"「原因」被削碎：{out!r}"

    def test_no_semantic_inversion(self):
        # 旧实现：'受什么影响' → '受影响'（**语义反转**：询问影响源 → 陈述被影响）
        out = expand_sections("重排的开销主要受什么影响")
        assert "受影响" not in out, f"出现语义反转：{out!r}"
        assert out == "重排的开销主要受什么影响"

    def test_m33_case_unchanged(self):
        # T-04+ 的触发用例：旧实现削成 '取消应该被记成状态' → ap-c0 从篇内第 1 跌到第 4
        assert expand_sections("取消应该被记成什么状态") == "取消应该被记成什么状态"

    def test_english_query_unchanged(self):
        # 旧实现：'What happens when...' → 'happens budget exhausted'（丢疑问结构）
        q = "What happens when the budget is exhausted?"
        assert expand_sections(q) == q


class TestExpandSectionsKeepsCapabilities:
    """①-b 必须**保住**两项经过验证的能力（否则不如直接删 rewrite）。"""

    def test_section_expansion_preserved(self):
        # 2026-08-25 为修「翻译/总结某章节只召回尾部」加的跨语言能力
        out = expand_sections("总结实验")
        assert "experiment" in out, f"章节展开丢失：{out!r}"

    def test_section_expansion_on_mixed_query(self):
        out = expand_sections("翻译一下摘要部分")
        assert "abstract" in out and "翻译一下摘要部分" in out

    def test_command_prefix_stripped(self):
        assert expand_sections("/总结 摘要") == "摘要 abstract"

    def test_bare_command_returns_empty(self):
        # M11 语义：只发 `/总结` → 空串（由调用方决定是否回退）
        assert expand_sections("/总结") == ""

    def test_empty_input(self):
        assert expand_sections("") == ""
        assert expand_sections("   ") == ""

    def test_cache_same_result(self):
        a = expand_sections("取消应该被记成什么状态")
        b = expand_sections("取消应该被记成什么状态")
        assert a == b


class TestCompositeUsesRewriteQueryByDefault:
    """`CompositeRAGPipeline` 默认必须用**④ 加固版** `rewrite_query`（与线上同源）。"""

    def test_default_prep_is_rewrite_query(self):
        from papermate.services.chat_service import CompositeRAGPipeline

        pipe = CompositeRAGPipeline([])
        assert pipe._query_prep is rewrite_query, (
            "默认预处理必须是与线上同源的 rewrite_query（④ 加固版）"
        )

    def test_injected_prep_is_used(self):
        """注入自定义预处理必须真的生效（评测三臂对照靠它复现基线）。"""
        from papermate.services.chat_service import CompositeRAGPipeline

        seen: list[str] = []

        class _Recorder:
            def query(self, query, top_k=3):
                seen.append(query)
                return []

        def _upper(q: str) -> str:
            return q.upper()

        pipe = CompositeRAGPipeline([_Recorder()], query_prep=_upper)
        pipe.query("abc")
        assert seen == ["ABC"], f"注入的预处理未生效：{seen}"

    def test_default_does_not_damage_query(self):
        """端到端：默认路径下，子管线收到的 query **不得被削碎**（④ 的核心契约）。"""
        from papermate.services.chat_service import CompositeRAGPipeline

        seen: list[str] = []

        class _Recorder:
            def query(self, query, top_k=3):
                seen.append(query)
                return []

        CompositeRAGPipeline([_Recorder()]).query("目的是什么")
        assert seen == ["目的是什么"], f"默认路径破坏了 query：{seen}"
