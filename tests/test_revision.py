"""PaperMate · /改稿（revision 技能）双路径测试

验证统一管线：分块抽取（确定性）→ LLM 合成（配真模型）→ 确定性回退（FakeLLM）→ citations 溯源，
以及草稿解析（文件名 / 序号 / 越界 / 歧义 / 缺省）。

替身：_RealStubLLM（非 FakeLLM，模拟真实大模型返回固定文本）、
      _MockDocSvc（提供分块管线 get_pipeline + 文档元数据 get_document）。
"""

from __future__ import annotations

import asyncio

from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, Role
from lightcore.llm.fake import FakeLLM
from lightcore.rag.pipeline import RetrievedChunk
from lightcore.skills.spec import SkillContext
from papermate.skills import default_router
from papermate.skills.revision import resolve_draft


class _RealStubLLM(BaseLLM):
    """非 FakeLLM 的替身：返回固定文本（模拟真实大模型合成）。"""

    def __init__(self, text: str) -> None:
        self._text = text
        self.model_name = "stub"

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=self._text), model="stub")


class _ConstPipeline:
    def __init__(self, chunks) -> None:
        self._c = list(chunks)

    def get_chunks(self):
        return self._c


class _MockDoc:
    def __init__(self, filename: str) -> None:
        self.filename = filename


class _MockDocSvc:
    """同时支持 get_pipeline（分块）与 get_document（文件名元数据）。"""

    def __init__(self, by_id: dict, names: dict | None = None) -> None:
        self._m = by_id
        self._names = names or {}

    async def get_pipeline(self, doc_id):
        return self._m.get(doc_id)

    def get_document(self, doc_id):
        return _MockDoc(self._names.get(doc_id, f"doc{doc_id}.pdf"))


def _ctx(doc_ids, llm, chunks_by_id, names=None, args="", features=None):
    svc = _MockDocSvc({d: _ConstPipeline(chunks) for d, chunks in chunks_by_id.items()}, names)
    return SkillContext(
        query="/改稿", args=args, doc_ids=doc_ids, doc_svc=svc, llm=llm,
        features=features or {"skills": True, "revision": True},
    )


# 草稿（自己的论文，待修改）
_DRAFT_CHUNKS = [
    RetrievedChunk(id="d-c0", text="Our method achieves better results on the dataset. 我们提出一种新方法。",
                   page=1, score=0.9),
    RetrievedChunk(id="d-c1", text="实验结果表明该方法在多个基线上有效，但消融实验不足。", page=4, score=0.8),
]
# 参考文献
_REF_CHUNKS = [
    RetrievedChunk(id="r-c0", text="Related work: prior methods lack comprehensive ablation studies.",
                   page=2, score=0.9),
    RetrievedChunk(id="r-c1", text="We evaluate on three datasets with five baselines and report limitations.",
                   page=6, score=0.7),
]


# ---------------- 注册与 feature 门控 ----------------
def test_revision_registered_and_gated():
    r = default_router()
    assert r.route("/改稿", {"skills": True, "revision": True}) is not None
    assert r.route("/revision", {"skills": True, "revision": True}) is not None
    # 独立 feature 门控
    assert r.route("/改稿", {"skills": True, "revision": False}) is None
    # master 开关
    assert r.route("/改稿", {"skills": False}) is None


def test_revision_feature_key_has_default():
    from papermate.runtime_settings import DEFAULT_FEATURES

    assert "revision" in DEFAULT_FEATURES, "DEFAULT_FEATURES 必须含 revision 键"


# ---------------- 确定性回退（FakeLLM）----------------
def test_revision_fallback_grouped():
    spec = default_router().route("/改稿", {"skills": True, "revision": True})

    async def run():
        res = await spec.run(_ctx([10, 20], FakeLLM(), {10: _DRAFT_CHUNKS, 20: _REF_CHUNKS},
                                  names={10: "my_paper.pdf", 20: "ref.pdf"}))
        assert res.ok, res.error
        # 报告头部显式标注草稿，避免序号/顺序歧义时静默选错
        assert "**草稿**" in res.content
        assert "my_paper.pdf" in res.content
        # 确定性回退：草稿待修改分组 + 参考文献标准分组 + 页码溯源
        assert "一、草稿中待修改的段落" in res.content
        assert "二、参考文献可对照的标准" in res.content
        assert "（p.4）" in res.content
        # 回退提示引导配置真实模型
        assert "设置" in res.content
        assert res.citations, "改稿必须回填 citations"
        # 草稿与参考文献都要能溯源
        pages = {c.get("page") for c in res.citations}
        assert pages & {1, 4}, f"应含草稿页码，实际 {pages}"
        assert pages & {2, 6}, f"应含参考文献页码，实际 {pages}"

    asyncio.run(run())


# ---------------- 真实 LLM 路径 ----------------
def test_revision_real_llm():
    spec = default_router().route("/改稿", {"skills": True, "revision": True})
    llm = _RealStubLLM(
        "### 一、逐节修改建议\n\n#### 方法（草稿 p.1）\n- **原文摘录**：…（p.1）\n"
        "- **问题**：未说明基线\n- **修改建议**：补充对比基线\n"
        "- **参考依据**：参考文献 p.6\n\n### 二、整体问题清单\n- 必须改：消融不足\n"
    )

    async def run():
        res = await spec.run(_ctx([10, 20], llm, {10: _DRAFT_CHUNKS, 20: _REF_CHUNKS},
                                  names={10: "my_paper.pdf", 20: "ref.pdf"}))
        assert res.ok
        assert res.content.startswith("## 改稿报告")
        assert "my_paper.pdf" in res.content          # 头部草稿标注
        assert "### 一、逐节修改建议" in res.content   # LLM 产出
        assert "### 二、整体问题清单" in res.content
        assert res.citations

    asyncio.run(run())


def test_revision_llm_exception_falls_back():
    """LLM 抛异常时需回退确定性输出，不能 500 / 空回复。"""

    class _BoomLLM(BaseLLM):
        model_name = "boom"

        async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
            raise RuntimeError("boom")

    spec = default_router().route("/改稿", {"skills": True, "revision": True})

    async def run():
        res = await spec.run(_ctx([10, 20], _BoomLLM(), {10: _DRAFT_CHUNKS, 20: _REF_CHUNKS},
                                  names={10: "my_paper.pdf", 20: "ref.pdf"}))
        assert res.ok
        assert "确定性回退" in res.content
        assert res.citations

    asyncio.run(run())


# ---------------- 草稿解析 ----------------
def _svc(doc_ids, names):
    return _MockDocSvc({d: _ConstPipeline(_DRAFT_CHUNKS) for d in doc_ids}, names)


def test_resolve_draft_prefers_filename():
    svc = _svc([10, 20, 30], {10: "ref_a.pdf", 20: "my_paper.pdf", 30: "ref_b.pdf"})
    # 精确
    assert resolve_draft(svc, [10, 20, 30], "my_paper.pdf")[0] == 20
    # 唯一子串
    assert resolve_draft(svc, [10, 20, 30], "my_paper")[0] == 20
    # 大小写不敏感
    assert resolve_draft(svc, [10, 20, 30], "MY_PAPER.PDF")[0] == 20


def test_resolve_draft_strips_mention_at_prefix():
    """2026-09-06：@ 前缀剥离——用户经 @ 提及输入 `/改稿 @my_paper.pdf` 时，
    前缀属于提及语法而非文件名，不剥离会导致文件名/序号匹配全部失败而回落。"""
    svc = _svc([10, 20, 30], {10: "ref_a.pdf", 20: "my_paper.pdf", 30: "ref_b.pdf"})
    # @文件名（子串）
    assert resolve_draft(svc, [10, 20, 30], "@my_paper.pdf")[0] == 20
    # @中文序号（与前端 @ 提及 label 口径一致）
    assert resolve_draft(svc, [10, 20, 30], "@文档二")[0] == 20
    # @数字序号
    assert resolve_draft(svc, [10, 20, 30], "@3")[0] == 30


# ---------------- C1：工作间主文献（default_draft）----------------
def test_resolve_draft_ws_default_used_when_no_args():
    """无参数时：工作间主文献优先于「默认第一篇」。"""
    svc = _svc([10, 20, 30], {10: "ref_a.pdf", 20: "my_paper.pdf", 30: "ref_b.pdf"})
    draft, note = resolve_draft(svc, [10, 20, 30], "", default_draft=20)
    assert draft == 20
    assert "工作间主文献" in note


def test_resolve_draft_explicit_args_override_ws_default():
    """显式文件名/序号参数覆盖工作间主文献（主文献是默认，不是锁定）。"""
    svc = _svc([10, 20, 30], {10: "ref_a.pdf", 20: "my_paper.pdf", 30: "ref_b.pdf"})
    # 文件名精确匹配 30
    draft, _ = resolve_draft(svc, [10, 20, 30], "ref_b.pdf", default_draft=20)
    assert draft == 30
    # 序号匹配 10
    draft, _ = resolve_draft(svc, [10, 20, 30], "第1篇", default_draft=20)
    assert draft == 10


def test_resolve_draft_bad_args_fall_back_to_ws_default():
    """参数无法匹配/歧义/越界时：回落工作间主文献（而非第一篇）。"""
    svc = _svc([10, 20, 30], {10: "ref_a.pdf", 20: "my_paper.pdf", 30: "ref_b.pdf"})
    # 参数未匹配
    draft, note = resolve_draft(svc, [10, 20, 30], "不存在的文档", default_draft=20)
    assert draft == 20 and "工作间主文献" in note
    # 序号越界
    draft, note = resolve_draft(svc, [10, 20, 30], "第9篇", default_draft=20)
    assert draft == 20 and "工作间主文献" in note
    # 子串歧义
    svc2 = _svc([10, 20, 30], {10: "paper_v1.pdf", 20: "my_paper.pdf", 30: "paper_v2.pdf"})
    draft, note = resolve_draft(svc2, [10, 20, 30], "paper", default_draft=20)
    assert draft == 20 and "工作间主文献" in note


def test_resolve_draft_ws_default_out_of_scope_ignored():
    """主文献不在检索范围内（如已被移出文献集）：忽略之，走原默认第一篇。"""
    svc = _svc([10, 20], {10: "a.pdf", 20: "b.pdf"})
    draft, note = resolve_draft(svc, [10, 20], "", default_draft=999)
    assert draft == 10
    assert "工作间主文献" not in note


def test_resolve_draft_ordinal():
    svc = _svc([10, 20], {10: "a.pdf", 20: "b.pdf"})
    assert resolve_draft(svc, [10, 20], "文档二")[0] == 20
    assert resolve_draft(svc, [10, 20], "第1篇")[0] == 10
    assert resolve_draft(svc, [10, 20], "2")[0] == 20


def test_resolve_draft_out_of_range_falls_back_with_note():
    svc = _svc([10, 20], {10: "a.pdf", 20: "b.pdf"})
    draft, note = resolve_draft(svc, [10, 20], "第9篇")
    assert draft == 10, "越界应回落第一篇"
    assert "超出范围" in note and "文件名" in note, "说明里应提示改用文件名"


def test_resolve_draft_ambiguous_substring_falls_back_with_note():
    svc = _svc([10, 20, 30], {10: "paper_v1.pdf", 20: "paper_v2.pdf", 30: "ref.pdf"})
    draft, note = resolve_draft(svc, [10, 20, 30], "paper")
    assert draft == 10
    assert "匹配到多篇" in note, f"歧义应说明，实际：{note}"


def test_resolve_draft_default_first():
    svc = _svc([10, 20], {10: "a.pdf", 20: "b.pdf"})
    draft, note = resolve_draft(svc, [10, 20], "")
    assert draft == 10 and "默认取第一篇" in note
    # 未匹配的参数同样回落第一篇且说明
    draft2, note2 = resolve_draft(svc, [10, 20], "不存在的文档")
    assert draft2 == 10 and "未匹配" in note2


def test_revision_no_docs_returns_error():
    spec = default_router().route("/改稿", {"skills": True, "revision": True})

    async def run():
        res = await spec.run(_ctx([], FakeLLM(), {}))
        assert not res.ok
        assert "关联" in (res.error or ""), "应提示先关联文档"

    asyncio.run(run())


def test_revision_draft_without_text_is_graceful():
    """草稿文档存在但无分块时，不能崩，应给出可理解文案。"""
    spec = default_router().route("/改稿", {"skills": True, "revision": True})

    async def run():
        res = await spec.run(_ctx([10, 20], FakeLLM(), {10: [], 20: _REF_CHUNKS},
                                  names={10: "empty.pdf", 20: "ref.pdf"}))
        assert res.ok
        assert "empty.pdf" in res.content
        assert "未" in res.content

    asyncio.run(run())
