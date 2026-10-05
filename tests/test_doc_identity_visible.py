"""N1 + N2（2026-10-02）：让 `retrieve` 结果与系统提示都带上「文档身份」。

## 事故（实测，见 docs/诊断-离题回答与熔断预警-2026-10-02.md §12）

M1 落地后，指定单篇的检索范围**确实生效**（四轮`[文档限定]` 全对、回答内容全对），
但用户仍报"模型还是无法确定指定文档"。根因：**范围生效 ≠ 模型知道**。

| 层 | 改动前|
|---|---|
| 检索结果注入 | 只有 `{id:"doc-6-c0", page, text, score}` —— **无标题、无标签** |
| SYSTEM 提示 | **完全不提**本轮限定到哪篇 |
| `retrieve` 工具 note | 只在"未关联任何文档"时给，**限定场景无note** |

⇒ 模型只能靠正文**猜**篇名。铁证（四轮内容全对、措辞暴露"在猜"）：
「**就所取到的内容看，这篇论文**…」「**本轮检索命中的**…」「**该文**提及…」。

- **N1** = 每条结果加 `doc`（「文档N」标签）+ 唯一篇时给范围 note（走既有 `_merge_note` 通道）
- **N2** = 系统提示声明"必须称它为「文档N」"（走既有 `_grounding_block` 通道）

★ 两个方案都**复用既有通道**，不新开机制（项目铁条：不另起一套）。
"""
from __future__ import annotations

import asyncio

from lightcore.tools.builtin.basic import _doc_label_of, _make_retrieve, _scope_note


# ============================================================
# 单元层：doc 标签解析（N1 的核心）
# ============================================================
class TestDocLabelOf:
    LABELS = {19: "文档一", 16: "文档二", 14: "文档三", 6: "文档四"}

    def test_parses_doc_id_from_chunk_id(self):
        """★ chunk id 形如 `doc-{id}-c{seq}`（`document_service.py:1019`），
        必须抽出中间的 doc 号再查标签。"""
        assert _doc_label_of("doc-16-c3", self.LABELS) == "文档二"
        assert _doc_label_of("doc-6-c0", self.LABELS) == "文档四"

    def test_returns_none_when_mapping_absent(self):
        """★ 映射缺失 ⇒ `None`（调用方省略 `doc` 键）。

        ⚠️ **绝不能猜**：给模型一个错标签，它会把别篇内容当成本篇——
        那是**静默降准**，比没标签危险得多。
        """
        assert _doc_label_of("doc-999-c0", self.LABELS) is None
        assert _doc_label_of("doc-16-c0", None) is None
        assert _doc_label_of("doc-16-c0", {}) is None

    def test_rejects_malformed_ids(self):
        """脏 id 一律 `None`（不得抛异常，也不得解析出错误数字）。"""
        for bad in ("doc-x-c0", "doc--c0", "16-c0", "doc-16", "", "chunk-16-c0", None):
            assert _doc_label_of(bad, self.LABELS) is None, bad

    def test_rejects_non_numeric_doc_segment(self):
        """★ doc 号段必须**纯数字** —— `doc-16x-c0` 不能被解析成 16。"""
        assert _doc_label_of("doc-16x-c0", self.LABELS) is None
        assert _doc_label_of("doc-1a6-c0", self.LABELS) is None


class TestScopeNote:
    LABELS = {19: "文档一", 16: "文档二", 14: "文档三", 6: "文档四"}

    def test_note_only_when_exactly_one_doc(self):
        """★ 只在**恰好一篇**时给 note（唯一篇才需要"明确称它"）。"""
        assert _scope_note({16: "文档二"}) is not None
        # 多篇：说"限定到 N 篇"反而让模型摇摆 ⇒ 不给
        assert _scope_note(self.LABELS) is None
        # 未限定：绝不给（否则与"可检索全部关联文档"矛盾）
        assert _scope_note(None) is None
        assert _scope_note({}) is None

    def test_note_names_the_label(self):
        """note 必须**点名**标签，并明确禁止模糊指代。"""
        note = _scope_note({16: "文档二"})
        assert "文档二" in note
        assert "这篇论文" in note and "该文" in note, "必须点名禁止哪些模糊指代"


# ============================================================
# 工具层：结果真的带上 doc + note（N1 的端到端）
# ============================================================
class _StubPipeline:
    """最小管线替身：只提供 `query`（返回带 id 的命中）。"""

    def __init__(self, ids):
        self._ids = ids

    async def query(self, q: str, top_k: int = 6):
        from lightcore.rag.pipeline import RetrievedChunk

        return [
            RetrievedChunk(id=i, text=f"{i} 的正文片段内容", page=3, score=0.9)
            for i in self._ids[:top_k]
        ]


def _run_retrieve(ids, doc_labels):
    tool = _make_retrieve(_StubPipeline(ids), doc_labels=doc_labels)
    return asyncio.run(tool(query="数据集", top_k=len(ids)))


class TestRetrieveCarriesIdentity:
    LABELS = {19: "文档一", 16: "文档二", 14: "文档三", 6: "文档四"}

    def test_each_result_has_doc_label(self):
        """★★ N1 的核心：每条结果都带 `doc` 标签 ⇒ 模型不必靠正文猜篇名。"""
        out = _run_retrieve(["doc-16-c0", "doc-16-c1"], self.LABELS).output
        assert [r["doc"] for r in out["results"]] == ["文档二", "文档二"]

    def test_note_present_when_single_doc(self):
        out = _run_retrieve(["doc-6-c0"], {6: "文档四"}).output
        assert "note" in out and "文档四" in out["note"], (
            f"唯一篇限定时必须给范围 note：{out.get('note')!r}"
        )

    def test_no_note_when_multiple_docs(self):
        out = _run_retrieve(["doc-19-c0", "doc-16-c0"], self.LABELS).output
        note = out.get("note") or ""
        assert "本轮范围" not in note, f"多篇时不得给「唯一篇」声明：{note!r}"

    def test_default_behavior_unchanged_when_no_labels(self):
        """★★ 零回归：`doc_labels=None`（默认）⇒ **不带 `doc`、不给范围 note**。

        这保证未使用 @ 提及的会话（含大量既有测试）行为**逐字不变**。
        """
        out = _run_retrieve(["doc-16-c0"], None).output
        assert all("doc" not in r for r in out["results"]), "未传标签时不得凭空造 doc 键"
        assert "note" not in out or "本轮范围" not in out["note"]

    def test_doc_field_survives_dedup_and_gate(self):
        """★ `doc` 必须在**去重/闸门之后**仍在（它们只改 text，不应丢身份）。"""
        out = _run_retrieve(["doc-16-c0", "doc-16-c1"], {16: "文档二"}).output
        for r in out["results"]:
            assert r["doc"] == "文档二"
            assert "text" in r and "id" in r and "page" in r


# ============================================================
# 提示层：系统提示必须声明（N2）
# ============================================================
class TestGroundingBlockFocus:
    def test_four_combinations_all_return_str(self):
        """★★ 四种组合（联网×是否限定）**都必须返回 str**。

        ⚠️ 这条是被真实 bug 逼出来的：`"...文字" + _focus,` 因**逗号优先级**
        返回了 **tuple**，导致 `SYSTEM_PROMPT + tuple` 抛 TypeError、
        **8 条既有 e2e 全红**。⇒ 显式断言类型，防止再次发生。
        """
        from papermate.services.chat_service import _grounding_block

        for enable_network in (False, True):
            for label in (None, "文档四"):
                r = _grounding_block(enable_network, has_docs=True, focused_label=label)
                assert isinstance(r, str), f"返回了 {type(r).__name__}：联网={enable_network} label={label}"
                assert ("本轮文档范围" in r) == (label is not None), (
                    f"声明与label 不一致：联网={enable_network} label={label}"
                )

    def test_focus_block_tells_model_to_use_label(self):
        """★★ N2 的价值就在这句：要求模型**明确称「文档N」**、禁止模糊指代。"""
        from papermate.services.chat_service import _grounding_block

        r = _grounding_block(False, has_docs=True, focused_label="文档四")
        assert "文档四" in r
        assert "这篇论文" in r and "该文" in r, "必须明确禁止模糊指代（那正是用户投诉的现象）"

    def test_no_focus_block_without_docs(self):
        """无文档会话**不得**出现本轮范围声明（那是纯大模型对话模式）。"""
        from papermate.services.chat_service import _grounding_block

        r = _grounding_block(False, has_docs=False, focused_label="文档四")
        assert "本轮文档范围" not in r, "无文档时不得注入范围声明（口径矛盾）"

    def test_default_signature_keeps_zero_regression(self):
        """★★ 老调用方（不传 focused_label）行为**逐字不变**。"""
        from papermate.services.chat_service import _grounding_block

        assert _grounding_block(False, has_docs=True) == _grounding_block(
            False, has_docs=True, focused_label=None
        )


class TestLabelSourceConsistency:
    def test_labels_built_from_FULL_set_not_restricted(self):
        """★★★ 标签映射必须基于会话**【全量】**关联集（id 降序），**不是限缩后**。

        ## 这条测试曾经断言过**相反**的东西（记录在此以免后人再犯）

        初版 N1 写的是「基于限缩后重算」，理由是"标签要与面板一致"——
        **那个理由是错的**：面板上的「文档三」本身就是按**全量**关联集编号的。

        实测事故（thinking 原文，2026-10-02 23:03）：

            本轮文档范围说用户用 @文档一 指定了唯一一篇文档，但用户实际写的是 @文档三。
            规则说『必须明确称它为「文档一」』，但用户问的是文档三。这里有冲突。

        ⇒ 模型整段推理都在纠结编号冲突，最后被迫"折中"写成
        「文档一（《…算法》）」⇒ 用户**更困惑**（比"用『这篇论文』"更糟）。

        ⇒ **正确口径**：标签永远按**全量**关联集算，与面板、与用户 @ 的文字一致。
        限定只影响"检索哪几篇"，**不影响"怎么编号"**。
        """
        from papermate.services.doc_labels import doc_label, mention_order

        full = [14, 6, 16, 19]          # 会话关联 4 篇
        restricted = [14]               # 用户 @文档三 ⇒ 限缩到只剩 id14
        # ★ 全量口径：id14 =「文档三」——与用户 @ 的文字、面板 tab 完全一致
        full_labels = {d: doc_label(i) for i, d in enumerate(mention_order(full))}
        assert full_labels[14] == "文档三"
        # 限缩后重算会得到「文档一」⇒ **与用户所说冲突**（这就是 23:03 事故的根因）
        bad_labels = {d: doc_label(i) for i, d in enumerate(mention_order(restricted))}
        assert bad_labels[14] == "文档一", "这是错误的编号（保留断言以说明为何不能这么做）"
        assert full_labels[14] != bad_labels[14], "两者必须不同，否则本测试失去意义"

    def test_source_uses_full_set_before_restriction(self):
        """★★ 源码契约：`_label_source` 必须在**限缩之前**快照。

        若有人把 doc_labels 又改回基于限缩后的 doc_ids，本用例变红
        —— 这是 2026-10-02 那个缺陷的**直接回归网**。
        """
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / "src" / "papermate" / "services" / "chat_service.py"
        ).read_text(encoding="utf-8")
        assert "_label_source = list(doc_ids)" in src, "必须在 restrict 限缩前快照全量集合"
        assert "mention_order(_label_source)" in src, "doc_labels 必须基于全量集合计算"
        assert "mention_order(list(doc_ids))" not in src, (
            "不得基于限缩后的 doc_ids 计算标签（这正是 23:03 事故的根因）"
        )

    def test_focus_note_uses_neutral_wording(self):
        """★★ 声明/.note **不得**写"用户用 @文档X 指定"（实测会让模型纠结编号冲突）。

        前端弹窗与手打对模型**长得一样**（都只是正文里的一个 @ 标签），
        若声明自称"用户用 @文档一"而正文是 @文档三，模型会整段推理去解这个"冲突"。
        ⇒ 断言两个通道的措辞都是**中性的**（只陈述"本轮只有这一篇"）。
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        basic = (root / "src" / "lightcore" / "tools" / "builtin" / "basic.py").read_text(encoding="utf-8")
        assert "用户用 @{label}" not in basic, "工具 note 不得假设用户用了某个 @ 标签"
        assert "本轮检索**只覆盖唯一一篇**文档" in basic, "应使用中性陈述"


    def test_source_contract_n1_n2_present(self):
        """源码契约：N1/N2 的接线点必须在（防有人删掉其中一半）。"""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        cs = (root / "src" / "papermate" / "services" / "chat_service.py").read_text(encoding="utf-8")
        basic = (root / "src" / "lightcore" / "tools" / "builtin" / "basic.py").read_text(encoding="utf-8")
        # N1：标签映射传进工具注册
        assert "register_builtin_tools(registry, rag_pipeline=rag, doc_labels=doc_labels)" in cs
        # N2：标签传给 grounding 块，且判据是「只有一项」
        assert "focused_label=_focused_label" in cs
        assert "if len(doc_labels) == 1 else None" in cs
        # N1：结果里注入 doc
        assert '"doc": _doc_label_of(' in basic
