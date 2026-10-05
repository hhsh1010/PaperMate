"""P1/P2 技能测试（docs/统一方案 阶段C）：/找代码 /研究缺口 /实验设计 /审稿 + /复现 LLM 再加工。

验证统一管线：检索/抽取（确定性）→ LLM 合成（配真模型）→ 确定性回退（FakeLLM/异常）→ 装配 citations。
使用替身：_RealStubLLM（非 FakeLLM，模拟真实大模型返回固定文本）、_MockDocSvc（提供分块管线）。
"""

from __future__ import annotations

import asyncio

from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, Role
from lightcore.llm.fake import FakeLLM
from lightcore.rag.pipeline import RetrievedChunk
from lightcore.skills.spec import SkillContext
from papermate.skills import default_router


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


class _MockDocSvc:
    def __init__(self, by_id: dict) -> None:
        self._m = by_id

    async def get_pipeline(self, doc_id):
        return self._m.get(doc_id)


def _ctx(doc_ids, llm, chunks_by_id, features=None, enable_network=False):
    svc = _MockDocSvc({d: _ConstPipeline(chunks) for d, chunks in chunks_by_id.items()})
    return SkillContext(
        query="/x", args="x", doc_ids=doc_ids, doc_svc=svc, llm=llm,
        features=features or {"skills": True}, enable_network=enable_network,
    )


# ---------------- 注册与 feature 门控 ----------------
def test_new_skills_registered_and_gated():
    r = default_router()
    for cmd, feat in (("/找代码", "find_code"), ("/研究缺口", "research_gap"),
                      ("/实验设计", "experiment_design"), ("/审稿", "review"),
                      ("/复现", "reproduction")):
        assert r.route(cmd, {"skills": True, feat: True}) is not None, f"{cmd} 应注册"
        assert r.route(cmd, {"skills": True, feat: False}) is None, f"{cmd} 关闭后应不路由"
    # /复现 受 skills 门控 + 独立 feature=reproduction 门控（F1 修复）
    assert r.route("/复现", {"skills": True, "reproduction": True}) is not None
    assert r.route("/复现", {"skills": True, "reproduction": False}) is None
    assert r.route("/复现", {"skills": False}) is None


# ---------------- /找代码 ----------------
def test_find_code_fallback_grouped():
    chunks = [RetrievedChunk(id="d1-c0",
        text="We use PyTorch and train with batch size 32 on the ImageNet dataset. "
             "The loss function is cross-entropy. Our code is available at GitHub repo.",
        page=3, score=0.9)]
    spec = default_router().route("/找代码", {"skills": True, "find_code": True})

    async def run():
        res = await spec.run(_ctx([1], FakeLLM(), {1: chunks}))
        assert res.ok
        # 确定性回退：分组标题 + 页码溯源 + 非空引用
        assert "**框架与库**" in res.content or "**数据集与基准**" in res.content
        assert "（p.3）" in res.content
        assert res.citations, "找代码必须回填 citations"
        assert any(c.get("page") for c in res.citations)

    asyncio.run(run())


def test_find_code_real_llm():
    chunks = [RetrievedChunk(id="d1-c0",
        text="We use PyTorch and train with batch size 32 on the ImageNet dataset.",
        page=3, score=0.9)]
    spec = default_router().route("/找代码", {"skills": True, "find_code": True})
    llm = _RealStubLLM("### 核心模块映射\n- data/ : 数据加载\n\n### 三态清单\n| 配置项 | 状态 |\n|---|---|\n| 框架 | 论文描述 |")

    async def run():
        res = await spec.run(_ctx([1], llm, {1: chunks}))
        assert res.ok
        assert "核心模块映射" in res.content, "真实 LLM 路径应返回合成内容"
        assert res.citations, "真实 LLM 路径也应带回引用溯源"

    asyncio.run(run())


# ---------------- /研究缺口 ----------------
def test_research_gap_fallback_grouped():
    chunks = [RetrievedChunk(id="d1-c0",
        text="我们的方法在少样本场景仍显不足。未来工作将探索更鲁棒的训练策略，并解决该开放问题。",
        page=5, score=0.9)]
    spec = default_router().route("/研究缺口", {"skills": True, "research_gap": True})

    async def run():
        res = await spec.run(_ctx([1], FakeLLM(), {1: chunks}))
        assert res.ok
        assert "**局限与不足**" in res.content or "**未来工作**" in res.content
        assert "（p.5）" in res.content
        assert res.citations

    asyncio.run(run())


def test_research_gap_real_llm():
    chunks = [RetrievedChunk(id="d1-c0",
        text="我们的方法在少样本场景仍显不足。未来工作将探索更鲁棒的训练策略。",
        page=5, score=0.9)]
    spec = default_router().route("/研究缺口", {"skills": True, "research_gap": True})
    llm = _RealStubLLM("### 已识别的局限\n- 少样本不足（p.5）\n\n### 研究空白与机会点\n- 鲁棒训练")

    async def run():
        res = await spec.run(_ctx([1], llm, {1: chunks}))
        assert res.ok
        assert "研究空白与机会点" in res.content
        assert res.citations

    asyncio.run(run())


# ---------------- /实验设计 ----------------
def test_experiment_design_fallback_grouped():
    chunks = [RetrievedChunk(id="d1-c0",
        text="我们在 ImageNet 数据集上评测，使用 accuracy 与 mAP 作为指标，以 ResNet 为基线，"
             "训练 100 个 epoch，batch size 为 64。",
        page=7, score=0.9)]
    spec = default_router().route("/实验设计", {"skills": True, "experiment_design": True})

    async def run():
        res = await spec.run(_ctx([1], FakeLLM(), {1: chunks}))
        assert res.ok
        assert "**数据集与基准**" in res.content or "**评价指标**" in res.content
        assert "（p.7）" in res.content
        assert res.citations

    asyncio.run(run())


def test_experiment_design_real_llm():
    chunks = [RetrievedChunk(id="d1-c0",
        text="我们在 ImageNet 上评测，以 ResNet 为基线，训练 100 个 epoch。",
        page=7, score=0.9)]
    spec = default_router().route("/实验设计", {"skills": True, "experiment_design": True})
    llm = _RealStubLLM("### 推荐数据集与指标\n- ImageNet / accuracy\n\n### 消融实验建议\n- 基线对比")

    async def run():
        res = await spec.run(_ctx([1], llm, {1: chunks}))
        assert res.ok
        assert "消融实验建议" in res.content
        assert res.citations

    asyncio.run(run())


# ---------------- /审稿 ----------------
def test_review_fallback_grouped():
    chunks = [RetrievedChunk(id="d1-c0",
        text="我们首次提出一种轻量级网络，在 ImageNet 上达到 SOTA。然而该方法在跨域场景存在局限。",
        page=9, score=0.9)]
    spec = default_router().route("/审稿", {"skills": True, "review": True})

    async def run():
        res = await spec.run(_ctx([1], FakeLLM(), {1: chunks}))
        assert res.ok
        assert "**贡献与新颖性**" in res.content or "**局限**" in res.content
        assert "（p.9）" in res.content
        assert res.citations

    asyncio.run(run())


def test_review_real_llm():
    chunks = [RetrievedChunk(id="d1-c0",
        text="我们首次提出一种轻量级网络，在 ImageNet 上达到 SOTA。然而该方法存在局限。",
        page=9, score=0.9)]
    spec = default_router().route("/审稿", {"skills": True, "review": True})
    llm = _RealStubLLM("### 论文概要\n- 轻量网络\n\n### 推荐评分\n- 接收：达到 SOTA")

    async def run():
        res = await spec.run(_ctx([1], llm, {1: chunks}))
        assert res.ok
        assert "推荐评分" in res.content
        assert res.citations

    asyncio.run(run())


# ---------------- /复现 LLM 再加工 ----------------
def test_reproduction_real_llm_three_state():
    chunks = [RetrievedChunk(id="d1-c0",
        text="训练时采用 batch size 128，学习率设为 0.001，并用 Adam 优化器进行优化。",
        page=3, score=0.9)]
    spec = default_router().route("/复现", {"skills": True})
    llm = _RealStubLLM(
        "| 配置项 | 论文给出值 | 状态 | 说明 |\n|---|---|---|---|\n"
        "| batch size | 128 | 论文给出 | p.3 |\n| 学习率 | 0.001 | 论文给出 | p.3 |")

    async def run():
        res = await spec.run(_ctx([1], llm, {1: chunks}))
        assert res.ok
        assert "论文给出" in res.content, "复现 LLM 路径应返回三态清单"
        assert res.citations, "复现 LLM 路径也应带回引用溯源"

    asyncio.run(run())


def test_reproduction_fallback_keeps_full_sentences():
    """回归：/复现 确定性回退仍按句子抽取、分组 + 页码溯源（test_phase_c 的等价校验）。"""
    chunks = [RetrievedChunk(id="d1-c0", text="我们使用 batch size 128\n和学习率 0.001 训练模型。", page=1, score=0.9)]
    spec = default_router().route("/复现", {"skills": True})

    async def run():
        res = await spec.run(_ctx([1], FakeLLM(), {1: chunks}))
        assert res.ok
        assert "batch size 128" in res.content
        assert "- 我们使用 batch size 128\n" not in res.content
        assert "我们使用 batch size 128 和学习率 0.001 训练模型。" in res.content
        assert "（p.1）" in res.content
        assert res.citations and any(c.get("page") for c in res.citations)

    asyncio.run(run())


# ---------------- 联网扩展：大模型原生联网搜索（测试替身） ----------------
class _NativeStubLLM(BaseLLM):
    """模拟支持原生联网搜索的真实模型：supports_native_web_search=True，chat_web_search 直出带引用文本。"""

    def __init__(self, text: str) -> None:
        self._text = text
        self.model_name = "native-stub"

    def supports_native_web_search(self) -> bool:
        return True

    async def chat_web_search(self, messages, *, temperature=0.3):
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=self._text), model="native-stub")

    async def chat(self, messages, tools=None, *, temperature=0.3, **kw):  # pragma: no cover
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=self._text), model="native-stub")


# ---------------- 联网扩展：/复现 集成（real LLM + 原生联网） ----------------
def test_reproduction_native_enrichment_integration():
    """真实 LLM + 会话开启联网 + 模型支持原生搜索 → 主输出由 chat_web_search 直出（含引用）。"""
    chunks = [RetrievedChunk(id="d1-c0",
        text="训练时采用 batch size 128，学习率设为 0.001。", page=3, score=0.9)]
    spec = default_router().route("/复现", {"skills": True})
    llm = _NativeStubLLM(
        "| 配置项 | 论文给出值 | 状态 | 说明 |\n|---|---|---|---|\n"
        "| 学习率 | 未给出 | 待核实 | 需查官方 repo |\n| batch size | 128 | 论文给出 | p.3 |")

    async def run():
        res = await spec.run(_ctx([1], llm, {1: chunks}, enable_network=True))
        assert res.ok
        assert "待核实" in res.content
        assert res.citations

    asyncio.run(run())


def test_reproduction_no_enrichment_when_network_off():
    """会话未开联网（enable_network=False）→ 即便真实 LLM 含标记也不追加外部段，回落确定性输出。"""
    chunks = [RetrievedChunk(id="d1-c0",
        text="训练时采用 batch size 128，学习率设为 0.001。", page=3, score=0.9)]
    spec = default_router().route("/复现", {"skills": True})
    llm = _RealStubLLM(
        "| 配置项 | 状态 |\n|---|---|\n| 学习率 | 待核实 |")

    async def run():
        res = await spec.run(_ctx([1], llm, {1: chunks}, enable_network=False))
        assert res.ok
        assert "外部资源" not in res.content, "未开联网时不得追加外部资源段"

    asyncio.run(run())


# ---------------- synthesize_with_net：统一合成+联网入口（仅大模型原生联网搜索） ----------------
def test_synthesize_with_net_native_when_enabled_and_supported():
    """enable_network=True 且模型支持原生搜索 → 走 chat_web_search 直出（不再有工具检索 enrich）。"""
    from papermate.skills._common import synthesize_with_net

    ctx = _ctx([1], _NativeStubLLM("模型自带联网检索结果：https://example.com/paper"),
               {1: [RetrievedChunk(id="d1-c0", text="x", page=1, score=0.9)]},
               enable_network=True)

    async def run():
        out = await synthesize_with_net(ctx, "sys", "user", temperature=0.3)
        assert out == "模型自带联网检索结果：https://example.com/paper"

    asyncio.run(run())


def test_synthesize_with_net_unsupported_returns_none():
    """FakeLLM（不支持原生搜索）→ 即便 enable_network=True 也返回 None（调用方回退确定性输出）。"""
    from papermate.skills._common import synthesize_with_net

    ctx = _ctx([1], FakeLLM(), {1: [RetrievedChunk(id="d1-c0", text="x", page=1, score=0.9)]},
               enable_network=True)

    async def run():
        out = await synthesize_with_net(ctx, "sys", "user", temperature=0.3)
        assert out is None

    asyncio.run(run())


class _RecordingNativeLLM(_NativeStubLLM):
    """记录各路径调用的替身：区分走了联网（chat_web_search）还是普通 chat。"""

    def __init__(self, text: str, *, net_error: bool = False) -> None:
        super().__init__(text)
        self.net_calls = 0
        self.chat_calls = 0
        self._net_error = net_error

    async def chat_web_search(self, messages, *, temperature=0.3):
        self.net_calls += 1
        if self._net_error:
            raise RuntimeError("联网服务不可用")
        return await super().chat_web_search(messages, temperature=temperature)

    async def chat(self, messages, tools=None, *, temperature=0.3, **kw):
        self.chat_calls += 1
        return await super().chat(messages, tools, temperature=temperature, **kw)


def test_synthesize_with_net_disabled_falls_back_to_plain_chat():
    """M1 修复：联网关闭 + 真实模型 → 回退普通 chat 合成（不联网、不再返回 None）。

    旧实现直接 return None，导致只调本函数的 /复现、/找代码 在未开联网时
    永远只能出确定性回退——核心「三态清单」功能完全失效。旧测试
    test_..._disabled_returns_none 锁定的正是该坏契约，已按新契约更新。
    """
    from papermate.skills._common import synthesize_with_net

    llm = _RecordingNativeLLM("普通合成结果")
    ctx = _ctx([1], llm, {1: [RetrievedChunk(id="d1-c0", text="x", page=1, score=0.9)]},
               enable_network=False)

    async def run():
        out = await synthesize_with_net(ctx, "sys", "user", temperature=0.3)
        assert out == "普通合成结果"  # 来自普通 chat
        assert llm.chat_calls == 1  # 走了普通 chat
        assert llm.net_calls == 0  # 未联网

    asyncio.run(run())


def test_synthesize_with_net_net_failure_falls_back_to_plain_chat():
    """M1 修复：联网开启且模型支持，但联网调用失败 → 回退普通 chat，而非 None。"""
    from papermate.skills._common import synthesize_with_net

    llm = _RecordingNativeLLM("失败后普通合成", net_error=True)
    ctx = _ctx([1], llm, {1: [RetrievedChunk(id="d1-c0", text="x", page=1, score=0.9)]},
               enable_network=True)

    async def run():
        out = await synthesize_with_net(ctx, "sys", "user", temperature=0.3)
        assert out == "失败后普通合成"
        assert llm.net_calls == 1
        assert llm.chat_calls == 1

    asyncio.run(run())


# ---------------- U2/U3 回归：规划目标文档标签（2026-09-09 / 2026-09-10） ----------------
def test_scope_labels_map_to_real_position():
    """U2 + U3：/规划 的默认目标句「基于 文档N …」的标签口径。

    U2（2026-09-09）：原实现 `_doc_label(i) for i in range(len(scope_ids))` 传循环索引——
    无论 @ 哪篇文档都显示「文档一」。
    U3（2026-09-10）：序号口径统一为 **id 降序**（与前端文档列表、`resolve_mention` 同源），
    而非 config.doc_ids 的关联顺序。
    """
    from papermate.skills.planning_skill import _scope_labels

    doc_ids = [14, 12, 10, 13, 11, 15, 2]  # 用户按此顺序关联
    # id 降序 → [15, 14, 13, 12, 11, 10, 2]
    assert _scope_labels([15], doc_ids) == ["文档一"]   # 最大 id = 前端「文档一」
    assert _scope_labels([14], doc_ids) == ["文档二"]
    assert _scope_labels([12], doc_ids) == ["文档四"]
    assert _scope_labels([12, 10], doc_ids) == ["文档四", "文档六"]
    assert _scope_labels([999], doc_ids) == ["文档一"]  # 不在关联集：安全回退


def test_resolve_mention_uses_id_desc_order():
    """U3 回归：`@文档N` 的 N 按 id 降序解释，与前端标签同源。

    原先按 ctx.doc_ids 的**关联顺序**解析 —— 关联顺序 ≠ id 降序时，
    `@文档二` 前端高亮 A 篇、后端却检索 B 篇。
    """
    from papermate.skills._common import mention_order, resolve_mention

    doc_ids = [14, 12, 10, 13, 11, 15, 2]  # 关联顺序
    assert mention_order(doc_ids) == [15, 14, 13, 12, 11, 10, 2]

    hit, rest = resolve_mention("@文档一 帮我总结", doc_ids)
    assert hit == 15 and rest == "帮我总结"     # 文档一 = 最大 id（与前端一致）
    assert resolve_mention("@文档二 目标", doc_ids)[0] == 14
    assert resolve_mention("@文档七 目标", doc_ids)[0] == 2
    # 越界与无提及
    assert resolve_mention("@文档九 目标", doc_ids) == (None, "@文档九 目标")
    assert resolve_mention("无提及", doc_ids) == (None, "无提及")
    # 空关联集不误命中
    assert resolve_mention("@文档一 x", []) == (None, "@文档一 x")


# ---------------- M2 回归：规划检索故障不做「零证据」伪装（2026-09-10） ----------------
def test_planning_raises_when_all_pipelines_fail():
    """M2：关联文档索引**全部**加载失败时抛 RetrievalUnavailable。

    原先 `except Exception: continue` 静默吞掉故障，规划在「零证据」下照常输出子任务，
    把系统故障伪装成正常结果（与 F7/I1/K4 同口径的第五处）。
    """
    from lightcore.rag.pipeline import RetrievalUnavailable
    from papermate.skills import planning_skill

    class _BoomSvc:
        async def get_pipeline(self, _d):  # noqa: ANN001
            raise RuntimeError("索引加载失败")

    ctx = _ctx([1, 2], FakeLLM(), {1: [], 2: []})
    ctx.doc_svc = _BoomSvc()  # type: ignore[assignment]

    async def run():
        try:
            await planning_skill.run(ctx)
        except RetrievalUnavailable:
            return
        raise AssertionError("全部文档索引故障时必须抛 RetrievalUnavailable，而非输出零证据规划")

    asyncio.run(run())


# ---------------- @文档N 提及限缩（2026-09-06） ----------------
def test_find_code_mention_scopes_docs():
    """args 含 @文档N → 技能只处理被提及的文档（此前提及无效，永远跑全量关联集）。"""
    chunks = {
        11: [RetrievedChunk(id="d11-c0", text="第一篇：轻量级网络在 ImageNet 达到 SOTA，代码见 GitHub。", page=1, score=0.9)],
        22: [RetrievedChunk(id="d22-c0", text="第二篇：使用 PyTorch 训练，batch size 32，数据集 ImageNet。", page=2, score=0.9)],
    }
    spec = default_router().route("/找代码", {"skills": True, "find_code": True})

    async def run():
        res = await spec.run(SkillContext(
            query="/找代码 @文档一", args="@文档一", doc_ids=[11, 22],
            doc_svc=_MockDocSvc({k: _ConstPipeline(v) for k, v in chunks.items()}), llm=FakeLLM(),
            features={"skills": True, "find_code": True},
        ))
        assert res.ok
        assert "第二篇：使用 PyTorch" in res.content      # 命中第二篇
        assert "第一篇：轻量级网络" not in res.content    # 第一篇不参与

    asyncio.run(run())


def test_literature_review_mention_scopes_docs():
    """/综述 @文档二 → summary_svc 只收到被提及的 doc（此前永远全量）。"""

    class _RecordingSummarySvc:
        def __init__(self) -> None:
            self.rw_calls: list[list[int]] = []

        async def related_work(self, doc_ids):
            self.rw_calls.append(list(doc_ids))
            return {"draft": "相关工作草稿", "co_citation": {"pairs": []}, "citations": []}

        async def summarize(self, doc_ids):
            return {"narrative": "", "sections": []}

    spec = default_router().route("/综述", {"skills": True, "summary": True})
    svc = _RecordingSummarySvc()

    async def run():
        res = await spec.run(SkillContext(
            query="/综述 @文档一", args="@文档一", doc_ids=[11, 22],
            doc_svc=None, summary_svc=svc, llm=FakeLLM(),
            features={"skills": True, "summary": True},
        ))
        assert res.ok, res.error
        assert svc.rw_calls == [[22]], svc.rw_calls  # 限缩到被提及的文档

    asyncio.run(run())
