"""P-02（2026-10-09）：`/术语表` 候选抽取与提示词收紧。

## 修的缺陷（用户实测截图）
术语表里混进了**文献题录**：`ISSN` / `CN` / `TN` / `OL` / `ACTA` / `SINICA` / `DZXB` /
`XXXX` / `XX` / `JIANG` / `CHENG` / `YANG`（刊号、刊名、文章编号占位符、作者拼音），
且释义里出现**文档之外的内容**（"YOLO = You Only Look Once" 这类世界知识）。

## 两类根因（本文件各钉一组）
1. **候选抽取过宽**：`\\b[A-Z]{2,6}\\b` 把题录区的词也当术语 ⇒ 按**行**剔除题录元数据
   （作者拼音形态上是普通大写词，孤立看无法与真缩写区分；但它们只出现在题录行里）。
2. **提示词留了"先验知识通道"**（原第 2 条允许填"常见英文表达"）⇒ 已删除并明确禁止补全。
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

# ⚠️ 不能用 `from papermate.skills import glossary` —— 包 `__init__` 里有
#   `from .glossary import SPEC as glossary`（该名字被 SkillSpec 占用）⇒ 拿到的是 **SPEC 对象**
#   而不是模块（首次写测试时踩到：7 条全红，报 `'SkillSpec' object has no attribute ...`）。
glossary = import_module("papermate.skills.glossary")

#: 实测泄漏过的那批（用户在截图里看到的"文档之外"的候选）
_NOISE = {
    "ISSN", "CN", "TN", "OL", "DOI", "DZXB", "ACTA", "SINICA",
    "XXXX", "XX", "JIANG", "CHENG", "YANG",
}

_META = """ACTA ELECTRONICA SINICA
ISSN 0372-2112
CN 11-2087/TN, OL
DOI: 10.12263/DZXB.20240001
文章编号：0372-2112(XXXX)XX-0001-08
作者：JIANG Ming, CHENG Lin, YANG Hua
第 46 卷第 7 期   Vol. 46 No. 7"""

_BODY = """本文提出一种旋转框遥感图像目标检测模型，在 DOTA 数据集上取得 78.9% 的 mAP。
DOTA 数据集包含 15 个类别；旋转框遥感图像目标检测模型对小目标更友好。"""


class TestMetaFiltering:
    def test_meta_terms_do_not_leak(self):
        terms = {t["term"] for t in glossary._extract_terms([_META, _BODY])}
        leaked = terms & _NOISE
        assert not leaked, f"题录元数据泄漏进候选（P-02 回归）：{sorted(leaked)}"

    def test_body_terms_are_kept(self):
        terms = {t["term"] for t in glossary._extract_terms([_META, _BODY])}
        assert "DOTA" in terms, "正文缩写必须保留（防'过滤过猛'）"
        assert any("旋转框" in t for t in terms), "正文中文术语必须保留（防'过滤过猛'）"

    def test_placeholder_filtered_even_in_body(self):
        """占位符 `XXXX` 即使出现在正文行也不入选（第二道保险，不依赖行过滤）。"""
        terms = {t["term"] for t in glossary._extract_terms(["正文里出现了 XXXX 与 XX 这样的占位符"])}
        assert "XXXX" not in terms and "XX" not in terms

    def test_strip_meta_lines(self):
        text = "正文第一行\nISSN 0372-2112\n正文第二行"
        assert glossary._strip_meta_lines(text) == "正文第一行\n正文第二行"

    def test_paren_def_in_meta_line_is_dropped(self):
        """括号定义（质量最高的候选）**也不得**来自题录行。"""
        terms = {t["term"] for t in glossary._extract_terms(["期刊（JOURNAL）2023 年第 3 期"])}
        assert "JOURNAL" not in terms


class TestPromptForbidsPriorKnowledge:
    """提示词静态契约：不得再留"先验知识通道"。"""

    def _src(self) -> str:
        return Path(glossary.__file__).read_text(encoding="utf-8")

    def test_prior_knowledge_channel_removed(self):
        src = self._src()
        assert "常见英文表达" not in src, (
            "提示词里仍允许'填常见英文表达' ⇒ 模型会补文档外的世界知识（P-02 的根因之一）"
        )

    def test_explicit_prohibition_present(self):
        src = self._src()
        assert "禁止用先验知识补全" in src
        assert "You Only Look Once" in src, "应把反例写进提示词（越具体越难被忽略）"
        assert "`—`" in src, "未给出全称时必须写明填 `—`"
