"""P3 批量扫尾回归（2026-09-10）：M11 / M12 / M13 / K7 等小型缺陷的锁死测试。"""

from __future__ import annotations


# ---------------- M11：纯命令剥离后不得回退含斜杠原 query ----------------
def test_rewrite_query_returns_empty_for_bare_command():
    """M11 回归：`/总结`（无正文）剥离后为空 → 返回空串。

    原先会回退到含斜杠的原 query，把命令标记当检索词送进 BM25（污染匹配）。
    """
    from lightcore.rag.query_rewrite import rewrite_query

    assert rewrite_query("/总结") == ""
    assert rewrite_query("/复现") == ""
    # 有正文时正常改写（不因该修复被吞掉）
    assert rewrite_query("/总结 注意力机制") != ""


# ---------------- M12：中文标题的 BibTeX key 必须唯一 ----------------
def test_bibtex_key_unique_for_chinese_titles():
    """M12 回归：中文标题不再被清成全空 → 多篇导出 key 不再恒为 `paper`。"""
    from papermate.processing.converter import to_bibtex

    def _key(bib: str) -> str:
        return bib.split("{", 1)[1].split(",", 1)[0]

    ka = _key(to_bibtex("基于改进YOLO的遥感图像目标检测", filename="a.pdf"))
    kb = _key(to_bibtex("融合道路邻近关系的高分遥感目标分割", filename="b.pdf"))
    assert ka != "paper" and kb != "paper", "中文标题的 key 不应退化为默认值"
    assert ka != kb, "不同中文文献的 BibTeX key 必须唯一"

    # 纯符号标题（清洗后为空）→ 哈希兜底，仍非空且不与他人相同
    ks = _key(to_bibtex("！！！", filename="c.pdf"))
    assert ks and ks != "paper"


# ---------------- M13：单发 "hi" 命中寒暄，且不误伤包含词 ----------------
def test_intent_matches_bare_hi_without_substring_false_positive():
    """M13 回归：`"hi "` 带尾空格导致单发 "hi" 不命中；改用词边界后两边都对。"""
    from papermate.services.intent import classify_intent

    assert classify_intent("hi").name == "chitchat"
    assert classify_intent("你好").name == "chitchat"
    # 包含 hi 的英文短句不应被误判为寒暄（词边界保护）
    assert classify_intent("this is fast").name != "chitchat"
    assert classify_intent("which one").name != "chitchat"


# ---------------- K7：memory_search 的 top_k 类型守卫 ----------------
def test_memory_search_tolerates_bad_top_k():
    """K7 回归：top_k 为非数字（模型生成的工具参数）时回退默认值，不抛异常。"""
    from lightcore.tools.builtin.domain import _make_memory_search  # noqa: PLC2701

    class _Mem:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def recall_long(self, _q, k=5):  # noqa: ANN001
            self.calls.append(k)
            return []

    mem = _Mem()
    search = _make_memory_search(mem)

    # N9（2026-09-11）：handler 已改 async（内部把向量编码下放线程池）→ 用 asyncio.run 驱动，
    # 与工具执行器（ToolExecutor 会 await）的真实调用方式一致。
    import asyncio as _asyncio

    assert _asyncio.run(search("q", "five")).ok is True
    assert _asyncio.run(search("q", None)).ok is True
    assert _asyncio.run(search("q", 3)).ok is True
    assert mem.calls == [5, 5, 3]  # 非法值回退 5；合法值透传
