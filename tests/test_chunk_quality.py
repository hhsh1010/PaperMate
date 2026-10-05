"""分块质量回归（2026-09-17）：**跨 block 累积** + **全角归一化**。

背景（真实数据实测）：会话 s18 关联的 3 篇论文，chunk **平均仅 22~39 字符**，
doc16 有 **93.6% 短于 50 字符**（最短 1 字符），且相邻 chunk 能拼回同一句 ——
检索出的"片段"零散不完整，模型无法准确判断。而设计目标是 300~500 **token**。

两条根因（本次修复）：
1. `chunker` 的 `buf` 建在 **block 循环内部** → 每个 block 立刻 flush，**永不跨 block 累积**；
   而 PDF 解析对期刊双栏排版常返回"每行一块" → 一个 chunk ≈ 一行。
   → **A：把缓冲提到循环外**（覆盖 `TestCrossBlockAccumulation`）。
2. 学术期刊 PDF 整段使用**全角 ASCII**（`ＹＯＬＯｖ３`、`１．４％`），而全角与半角在分词器里
   是**不同的 token** → BM25 与 BGE-M3 的英文/数字检索双双失效。
   → **B：解析层统一归一化**（覆盖 `TestFullwidthNormalization`）。
"""

from __future__ import annotations

import pytest

from lightcore.rag.chunker import Chunker, _iter_sentences
from lightcore.rag.parser import DocumentParser, ParsedBlock, merge_split_numbers, to_halfwidth

# ============================================================ A. 跨 block 累积


class TestCrossBlockAccumulation:
    """`buf` 必须**跨 block 累积** —— 这是碎片化的主因。"""

    def test_many_short_blocks_merge_into_one_chunk(self) -> None:
        """20 个短 block（合计 ~256 token < target 400）→ 应合并为 **1 块**。

        修复前：每个 block 一块 → 20 块（这就是实测 437 块的来源）。
        """
        blocks = [ParsedBlock(page=1, text="这是一句用于验证跨块累积的测试文本。" * 2)
                  for _ in range(20)]

        chunks = Chunker().chunk_blocks(blocks, "d")

        assert len(chunks) == 1, f"短 block 必须合并，实际 {len(chunks)} 块"
        assert len(chunks[0].text) > 500, "合并后的块应显著大于单个 block"

    def test_chunks_still_split_when_target_exceeded(self) -> None:
        """累积到 target 仍必须切块 —— 修复不能变成"永不切分"。"""
        blocks = [ParsedBlock(page=1, text="这是一句用于验证跨块累积的测试文本。" * 2)
                  for _ in range(60)]  # ~768 token

        chunks = Chunker().chunk_blocks(blocks, "d")

        assert len(chunks) >= 2, f"超过 target 必须切分，实际 {len(chunks)} 块"
        assert all(len(c.text) > 200 for c in chunks), "切出来的也应是成块内容，不是碎片"

    def test_page_boundary_forces_new_chunk(self) -> None:
        """**换页必切** —— 页号是硬语义（`pdf_extract(page=N)` / 引用溯源都依赖它）。

        见 `test_page_indexing.py::test_pipeline_from_db_chunks_preserves_pages`：
        若块跨页合并，第 2 页会取不到内容（功能损失，不只是测试问题）。
        """
        short = "这是一句很短的测试文本。"
        blocks = [ParsedBlock(page=1, text=short), ParsedBlock(page=2, text=short),
                  ParsedBlock(page=3, text=short)]

        chunks = Chunker().chunk_blocks(blocks, "d")

        assert {c.page for c in chunks} == {1, 2, 3}, "每页都必须至少有一块，页码不得丢失"

    def test_none_page_documents_do_not_trigger_page_split(self) -> None:
        """非分页文档（docx/txt，page=None）不受页边界影响，照常累积。"""
        blocks = [ParsedBlock(page=None, text="这是一句用于验证跨块累积的测试文本。" * 2)
                  for _ in range(20)]

        chunks = Chunker().chunk_blocks(blocks, "d")

        assert len(chunks) == 1, f"page=None 应正常累积（不触发换页切分），实际 {len(chunks)}"

    def test_single_block_long_text_splits_by_target(self) -> None:
        """单个超长 block 内部也要按 target 切分（原有行为不得回归）。"""
        blocks = [ParsedBlock(page=1, text="这是一句用于验证跨块累积的测试文本。" * 100)]
        # 100 句 × 17 字符 ≈ 1700 字符 ≈ 680 token → target 400 → 必然切成多块

        chunks = Chunker().chunk_blocks(blocks, "d")

        assert len(chunks) >= 2, f"长文本必须切分，实际 {len(chunks)}"
        # 切分粒度应贴近 target，而不是随手乱切（+overlap 余量）
        assert all(c.token_count <= 600 for c in chunks), [c.token_count for c in chunks]
        assert all(c.page == 1 for c in chunks)


# ============================================================ B. 全角归一化


class TestFullwidthNormalization:
    """全角 ASCII 必须转半角；**中文标点必须保留**。"""

    @pytest.mark.parametrize(("src", "want"), [
        ("ＹＯＬＯｖ３－Ｔｉｎｙ", "YOLOv3-Tiny"),      # 实测 doc16 的典型形态
        ("ｍＡＰ　ｉｓ　ｉｍｐｒｏｖｅｄ　ｂｙ　１．４％", "mAP is improved by 1.4%"),
        ("（ａ）（ｂ）", "(a)(b)"),
        ("／", "/"),
        ("＋－＝～", "+-=~"),
    ])
    def test_fullwidth_ascii_becomes_halfwidth(self, src: str, want: str) -> None:
        assert to_halfwidth(src) == want

    @pytest.mark.parametrize("text", [
        "中文，逗号；分号：冒号？问号！叹号",   # 全角区段内，但刻意保留
        "句号。顿号、《书名号》“引号”",        # 本就不在全角 ASCII 区段内
    ])
    def test_chinese_punctuation_preserved(self, text: str) -> None:
        """中文标点保持全角：对 jieba 分词无影响（都是分隔符），转了只破坏观感。"""
        assert to_halfwidth(text) == text

    def test_fullwidth_space_becomes_ascii_space(self) -> None:
        assert to_halfwidth("ａ　ｂ") == "a b"

    def test_mixed_text_only_touches_fullwidth(self) -> None:
        got = to_halfwidth("遥感图像ＹＯＬＯｖ３检测，精度１．４％。")
        assert got == "遥感图像YOLOv3检测，精度1.4%。", got

    def test_empty_and_plain_text_untouched(self) -> None:
        assert to_halfwidth("") == ""
        assert to_halfwidth("已经全是半角 ASCII and 中文。") == "已经全是半角 ASCII and 中文。"

    def test_parsed_block_normalizes_on_construction(self) -> None:
        """归一化挂在 `ParsedBlock.__post_init__` = **单一来源**，所有构造点自动生效。"""
        assert ParsedBlock(page=1, text="ＹＯＬＯ").text == "YOLO"

    def test_parse_text_path_is_normalized(self) -> None:
        """`parse_text`（md/txt 路径）同样必须归一化。"""
        blocks = DocumentParser().parse_text("ＹＯＬＯｖ３检测\n\n１．４％提升")
        assert blocks, "应有解析结果"
        assert all("Ｙ" not in b.text and "１" not in b.text for b in blocks), \
            [b.text for b in blocks]

    def test_parse_iterable_path_is_normalized(self) -> None:
        """可迭代文本块路径（`parse` 的直接构造分支）也必须归一化。"""
        blocks = DocumentParser().parse(["ＹＯＬＯｖ３", "１．４％"])
        assert [b.text for b in blocks] == ["YOLOv3", "1.4%"]


class TestSentenceSplitKeepsNumbers:
    """句子切分**不得**劈开小数点 / 版本号 / DOI / 章节号（2026-09-19）。

    旧正则把英文句点 `.` **无条件**当句子边界 → `77.4` 被切成 `77.` + `4`，
    chunk 内 `" ".join(buf)` 再拼成 `77. 4`。实测全库 631 块中 314 块（49.8%）中招
    —— 检索返回的 `mAP@0. 5`、`3. 4. 4` 无法判读（详见 `docs/检索质量诊断-分块碎片化.md` §6）。
    """

    @pytest.mark.parametrize("text", [
        "准确率 77.4 78.9 召回率 68.8 70.7 mAP@0.5",
        "3.4.4 检测结果与分析",
        "DOI：10.19651/j.cnki.emt.2005268",
        "在 DOTA v1.0 数据集上的实验结果",
        "如图1.2所示，网络结构分为三部分",
    ])
    def test_numbers_not_split(self, text: str) -> None:
        """小数点/版本号/DOI/章节号必须保持完整一个片段。"""
        assert _iter_sentences(text) == [text], _iter_sentences(text)

    @pytest.mark.parametrize("text", [
        "The model is fast. It beats the baseline.",
        "et al. Few-shot methods work.",
        "本文算法有效。实验表明其优于基线。",
        "第一组；第二组结果如下",
    ])
    def test_real_sentence_boundaries_still_split(self, text: str) -> None:
        """真正的句子边界**仍要切**（否则会退化成整段不分句，overlap 失效）。"""
        assert len(_iter_sentences(text)) == 2, _iter_sentences(text)

    def test_chunk_output_keeps_numbers_intact(self) -> None:
        """端到端：分块产物里不得出现「数字.空格数字」。"""
        blocks = [ParsedBlock(page=1, text="算法 YOLOv7 准确率 77.4 78.9 mAP@0.5 优于基线。")]
        text = Chunker().chunk_blocks(blocks, "d")[0].text

        assert "77.4" in text, text
        assert "77. 4" not in text, f"小数点被空格劈开：{text}"


class TestMergeSplitNumbers:
    """`merge_split_numbers`：PDF 排版在小数点里插的空格必须被合并（2026-09-19）。

    ⚠️ 真因不是 pymupdf —— rawdict 字符级坐标证明 **PDF 内容流里本来就有空格字符**，
    所以只能文本层后处理（换提取模式无效）。
    """

    @pytest.mark.parametrize("src,want", [
        ("准确率 77. 4% 召回率 68. 8%", "准确率 77.4% 召回率 68.8%"),
        ("mAP@0. 5：0. 95", "mAP@0.5：0.95"),
        ("3. 4. 4 检测结果与分析", "3.4.4 检测结果与分析"),
        ("arXiv:2010. 04159, 2020", "arXiv:2010.04159, 2020"),
        ("模型大小(5. 3 MB)仅为(45. 3 MB)", "模型大小(5.3 MB)仅为(45.3 MB)"),
        ("提高了2. 9个百分点", "提高了2.9个百分点"),
    ])
    def test_split_numbers_merged(self, src: str, want: str) -> None:
        assert merge_split_numbers(src) == want, merge_split_numbers(src)

    @pytest.mark.parametrize("text", [
        "et al. 3 methods",          # 句点前非数字 → 英文缩写句末，不得动
        "本文结束。下一节开始",        # 无数字句点
        "第1节 算法描述",             # 没有句点
        "version 2.0 已发布",         # 本来就完整，不得改坏
    ])
    def test_non_numeric_context_untouched(self, text: str) -> None:
        assert merge_split_numbers(text) == text, merge_split_numbers(text)

    def test_applied_on_parsed_block_construction(self) -> None:
        """与全角归一化一样挂在 `ParsedBlock.__post_init__` = 单一来源，所有解析路径生效。"""
        assert ParsedBlock(page=1, text="准确率７７. ４％").text == "准确率77.4%"

    def test_empty_text_safe(self) -> None:
        assert merge_split_numbers("") == ""
