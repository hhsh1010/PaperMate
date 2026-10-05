"""阶段 C 测试（docs/统一方案 阶段C）：P1-5 并行 / P1-6 多查询 / code_interpreter 沙箱 /
related_work 共现 / SkillSpec+Router+3 技能。使用替身，无需网络与 API key。"""

from __future__ import annotations

import asyncio
from unittest import mock

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import BaseLLM, ChatMessage, LLMResponse, Role, ToolCall
from lightcore.llm.fake import FakeLLM
from lightcore.rag.multi_query import MultiQueryExpander, MultiQueryRAGPipeline
from lightcore.rag.pipeline import RetrievedChunk
from lightcore.skills.spec import SkillContext
from lightcore.tools.builtin.basic import DockerSandbox, MockSandbox, register_builtin_tools
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry
from papermate.services import summary_service as sm
from papermate.services.summary_service import SummaryService, _build_llm
from papermate.skills import default_router


# ---------------- P1-6 多查询融合 ----------------
class _RealStubLLM(BaseLLM):
    """非 FakeLLM 的替身：返回固定文本（模拟 LLM 改写子查询）。"""

    def __init__(self, text: str) -> None:
        self._text = text
        self.model_name = "stub"

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=self._text), model="stub")


def test_multi_query_expander_fake_single():
    async def run():
        # None 与 FakeLLM 均回退单查询（零成本、行为不变）
        assert await MultiQueryExpander(None).expand("量子纠缠") == ["量子纠缠"]
        assert await MultiQueryExpander(FakeLLM()).expand("量子纠缠") == ["量子纠缠"]

    asyncio.run(run())


def test_multi_query_expander_real_splits():
    async def run():
        llm = _RealStubLLM("子查询甲\n子查询乙\n子查询丙")
        # n=4 → want=4 → 保留 subs[:3]（3 条子查询）+ 原查询 = 4
        out = await MultiQueryExpander(llm, n=4).expand("原始问题", n=4)
        assert out[0] == "原始问题"
        assert "子查询甲" in out and "子查询乙" in out and "子查询丙" in out
        assert len(out) == 4  # 原 + 3 子查询

    asyncio.run(run())


class _StubPipeline:
    def __init__(self, by_query: dict[str, list[RetrievedChunk]]) -> None:
        self._by = by_query

    def query(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        return self._by.get(query, [])[:top_k]


def test_multi_query_pipeline_rrf_merge_dedup():
    async def run():
        # 两个子查询命中同一 chunk（id 相同）→ 应去重；不同 chunk 按 RRF 合并
        c1 = RetrievedChunk(id="d1-c0", text="A", page=1, score=0.9)
        c2 = RetrievedChunk(id="d1-c1", text="B", page=2, score=0.8)
        base = _StubPipeline({
            "子查询甲": [c1, c2],
            "子查询乙": [c1, RetrievedChunk(id="d1-c2", text="C", page=3, score=0.7)],
        })
        pipe = MultiQueryRAGPipeline(base, MultiQueryExpander(None))  # fake 模式下仍走多查询合并
        # 用真实 LLM 改写得到两个子查询
        pipe._expander = MultiQueryExpander(_RealStubLLM("子查询甲\n子查询乙"))
        out = await pipe.query("原始", top_k=3)
        ids = [c.id for c in out]
        assert ids.count("d1-c0") == 1, "同 id 必须去重"
        assert "d1-c0" in ids and "d1-c1" in ids and "d1-c2" in ids
        # 出现在两个子查询中的 c0 应排在最前（RRF 分更高）
        assert out[0].id == "d1-c0"

    asyncio.run(run())


# ---------------- code_interpreter 沙箱 ----------------
def test_code_interpreter_default_rejected():
    reg = ToolRegistry()
    register_builtin_tools(reg)  # 默认 SANDBOX=False（NFR-07）
    exe = ToolExecutor(reg)
    res = asyncio.run(exe.execute("code_interpreter", {"code": "print(1)", "lang": "python"}))
    assert not res.ok
    assert "权限" in res.error or "sandbox" in res.error


def test_code_interpreter_mock_sandbox_runs():
    reg = ToolRegistry()
    register_builtin_tools(reg, sandbox=MockSandbox("hello-world"))
    exe = ToolExecutor(reg, permission_enabled={
        Permission.SAFE: True, Permission.NETWORK: False,
        Permission.SANDBOX: True, Permission.ADMIN: False,
    })
    res = asyncio.run(exe.execute("code_interpreter", {"code": "print('hi')", "lang": "python"}))
    assert res.ok and res.output["stdout"] == "hello-world"


def test_docker_sandbox_missing_errors_clearly():
    with mock.patch("subprocess.run", side_effect=FileNotFoundError("no docker")):
        res = DockerSandbox().run("print(1)")
    assert not res["ok"]
    assert "Docker" in res["error"], "无 Docker 时应明确报错，而非静默本机执行"


# ---------------- related_work 引文共现 ----------------
def test_related_work_co_citation_two_docs():
    svc = SummaryService(None, None, None)
    chunks = [
        {"doc_id": 1, "text": "深度学习 注意力 机制 在 视觉 任务 表现 优异"},
        {"doc_id": 1, "text": "注意力 机制 的 计算 复杂 度 分析"},
        {"doc_id": 2, "text": "深度学习 注意力 卷积 网络 用于 语音 识别"},
    ]
    co = svc._co_citation(chunks, [1, 2])
    assert co["pairs"], "两篇文档应有共引对"
    p = co["pairs"][0]
    assert 0.0 <= p["jaccard"] <= 1.0
    assert "注意力" in p["shared_terms"]


def test_related_work_co_citation_single_doc_empty():
    svc = SummaryService(None, None, None)
    chunks = [{"doc_id": 1, "text": "深度学习 注意力 机制"}]
    co = svc._co_citation(chunks, [1])
    assert co["pairs"] == []
    assert co["note"]


# ---------------- SkillSpec + SkillRouter + 3 技能 ----------------
def test_skill_router_routes_commands_and_gating():
    r = default_router()
    assert r.route("/综述", {"skills": True}) is not None
    assert r.route("/相关工作 量子", {"skills": True}) is not None
    # 命令未开启 → 回落普通对话
    assert r.route("/综述", {"skills": False}) is None
    # 非命令 → 不匹配
    assert r.route("请帮我总结这篇论文", {"skills": True}) is None


def test_skill_parse_args():
    r = default_router()
    spec = r.route("/综述 图神经网络", {"skills": True})
    assert spec is not None
    assert r.parse_args("/综述 图神经网络", spec) == "图神经网络"


class _MockSummary:
    async def related_work(self, doc_ids, lang="zh"):
        return {
            "draft": "相关工作草稿",
            "co_citation": {"pairs": [{"doc_a": 1, "doc_b": 2, "shared_terms": ["注意力"], "jaccard": 0.3}],
                            "note": "近似共引"},
            "citations": [{"chunk_id": "1-c0", "page": 1, "text": "t"}],
        }

    async def summarize(self, doc_ids, lang="zh"):
        return {"sections": [{"key": "background", "title": "研究背景", "content": "背景内容"}],
                "citations": []}


def test_skill_literature_review_run():
    async def run():
        r = default_router()
        spec = r.route("/综述", {"skills": True})
        ctx = SkillContext(query="/综述", args="", doc_ids=[1, 2],
                           summary_svc=_MockSummary(), features={"skills": True})
        res = await spec.run(ctx)
        assert res.ok and "文献综述" in res.content
        assert res.citations

    asyncio.run(run())


def test_skill_related_work_no_doc_error():
    async def run():
        r = default_router()
        spec = r.route("/相关工作", {"skills": True})
        ctx = SkillContext(query="/相关工作", args="", doc_ids=[],
                           summary_svc=_MockSummary(), features={"skills": True})
        res = await spec.run(ctx)
        assert not res.ok
        assert "文档" in res.error

    asyncio.run(run())


# ---------------- P1-5 工具并行 ----------------
class _TwoToolLLM(BaseLLM):
    def __init__(self) -> None:
        self.calls = 0
        self.model_name = "stub"

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            tcs = [
                ToolCall(id="c1", name="retrieve", arguments={"query": "x"}),
                ToolCall(id="c2", name="calculator", arguments={"expression": "(1+2)*3"}),
            ]
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="", tool_calls=tcs), model="stub")
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content="done"), model="stub")


def test_agent_loop_runs_parallel_safe_tools():
    async def run():
        reg = ToolRegistry()
        register_builtin_tools(reg)  # retrieve(safe) + calculator(safe)
        loop = AgentLoop(_TwoToolLLM(), ToolExecutor(reg), config=LoopConfig(max_iterations=4))
        events = [ev async for ev in loop.run_stream("开始")]
        types = [e.type for e in events]
        assert "tool_call" in types and "tool_result" in types
        # 两个 SAFE 工具都应被执行（并行路径覆盖）
        called = [e.data["name"] for e in events if e.type == "tool_call"]
        assert "retrieve" in called and "calculator" in called
        # 最终结果产出
        final = next(e for e in reversed(events) if e.type == "message")
        assert final.data["content"] == "done"

    asyncio.run(run())


# ---------------- P0 技能：/精读 / /术语表 ----------------
class _ConstPipeline:
    """无论 query 都返回同一组 chunk 的占位管线（测试检索编排用）。"""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def query(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        return self._chunks[:top_k]

    def get_chunks(self) -> list[RetrievedChunk]:
        return self._chunks


class _MockDocSvc:
    """按 doc_id 返回管线（或 None）的占位文档服务。"""

    def __init__(self, by_id: dict[int, _ConstPipeline | None]) -> None:
        self._by = by_id

    async def get_pipeline(self, doc_id: int) -> _ConstPipeline | None:
        return self._by.get(doc_id)


class _ReportStubLLM(BaseLLM):
    """非 FakeLLM 的占位模型（走真实 LLM 生成分支）。"""

    def __init__(self, text: str = "报告") -> None:
        self.model_name = "stub"
        self._text = text

    async def chat(self, messages, tools=None, *, temperature=0.3, max_tokens=None, tool_choice=None):  # noqa: ARG002
        return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=self._text), model="stub")


def _deep_read_ctx(doc_ids, llm, focus="") -> SkillContext:
    return SkillContext(
        query=f"/精读 {focus}".strip(), args=focus, doc_ids=doc_ids,
        doc_svc=_MockDocSvc({d: _ConstPipeline([
            RetrievedChunk(id=f"d{d}-c0", text="我们提出卷积神经网络（CNN）用于图像分类，准确率达到 95%。", page=1, score=0.9),
        ]) for d in doc_ids}),
        llm=llm, features={"skills": True, "deep_read": True},
    )


def test_skill_deep_read_routing_and_gating():
    r = default_router()
    assert r.route("/精读", {"skills": True, "deep_read": True}) is not None
    assert r.route("/deep_read 方法", {"skills": True, "deep_read": True}) is not None
    # 子开关关闭 → 该命令不暴露（回落普通对话）
    assert r.route("/精读", {"skills": True, "deep_read": False}) is None
    # 总开关关闭 → 所有命令不暴露
    assert r.route("/精读", {"skills": False}) is None


def test_skill_deep_read_requires_single_doc():
    async def run():
        spec = default_router().route("/精读", {"skills": True, "deep_read": True})
        res = await spec.run(_deep_read_ctx([1, 2], FakeLLM()))
        assert not res.ok
        assert "文档" in res.error

    asyncio.run(run())


def test_skill_deep_read_mention_selects_doc():
    """2026-09-06：args 含 `@文档N` 时按提及选文——多篇关联下也可精读指定篇目。

    此前 deep_read 硬性要求「恰好关联 1 篇」，@提及只进会话关联、不缩小范围，
    `/精读 @文档三` 在 3 篇关联下被拒（用户实际场景）。"""
    async def run():
        spec = default_router().route("/精读", {"skills": True, "deep_read": True})
        ctx = SkillContext(
            query="/精读 @文档一", args="@文档一", doc_ids=[11, 22, 33],
            doc_svc=_MockDocSvc({
                11: _ConstPipeline([RetrievedChunk(id="d11-c0", text="第一篇的段落。", page=1, score=0.9)]),
                22: _ConstPipeline([RetrievedChunk(id="d22-c0", text="第二篇的段落。", page=1, score=0.9)]),
                33: _ConstPipeline([RetrievedChunk(id="d33-c0", text="第三篇独有的结论段落。", page=1, score=0.9)]),
            }),
            llm=FakeLLM(), features={"skills": True, "deep_read": True},
        )
        res = await spec.run(ctx)
        assert res.ok, res.error
        # 命中的是 doc 33 的管线（其文本独有）
        assert "第三篇独有的结论段落" in res.content
        assert "第一篇的段落" not in res.content and "第二篇的段落" not in res.content

    asyncio.run(run())


def test_skill_deep_read_mention_with_focus():
    async def run():
        spec = default_router().route("/精读", {"skills": True, "deep_read": True})
        # 提及 + focus 组合：`@文档二 方法` → 选 doc 2 且只走「方法」维度
        res = await spec.run(_deep_read_ctx([11, 22], FakeLLM(), focus="@文档二 方法"))
        assert res.ok, res.error
        assert "方法" in res.content
        assert "核心方法" not in res.content  # 未识别的 focus 不应混入全部维度

    asyncio.run(run())


def test_skill_deep_read_mention_out_of_range_falls_back():
    async def run():
        spec = default_router().route("/精读", {"skills": True, "deep_read": True})
        # 序号越界（只有 3 篇却 @文档九）→ 按未提及处理 → 多文档报错（不误读 doc 3）
        res = await spec.run(_deep_read_ctx([11, 22, 33], FakeLLM(), focus="@文档九"))
        assert not res.ok
        assert "@文档N" in res.error

    asyncio.run(run())


def test_skill_deep_read_fake_fallback():
    async def run():
        spec = default_router().route("/精读", {"skills": True, "deep_read": True})
        res = await spec.run(_deep_read_ctx([1], FakeLLM()))
        assert res.ok
        assert "确定性回退" in res.content  # fake 模型走回退，不臆造
        assert "第 1 页" in res.content

    asyncio.run(run())


def test_skill_deep_read_real_llm():
    async def run():
        spec = default_router().route("/精读", {"skills": True, "deep_read": True})
        report = "研究问题：如何提升图像分类。\n核心方法：使用 CNN。\n创新点：轻量结构。\n局限与展望：数据规模。"
        res = await spec.run(_deep_read_ctx([1], _ReportStubLLM(report)))
        assert res.ok
        assert "研究问题" in res.content and "创新点" in res.content
        assert res.citations, "真实 LLM 路径应带回引用溯源"

    asyncio.run(run())


def _glossary_ctx(doc_ids, llm) -> SkillContext:
    return SkillContext(
        query="/术语表", args="", doc_ids=doc_ids,
        doc_svc=_MockDocSvc({d: _ConstPipeline([
            RetrievedChunk(id=f"d{d}-c0", text="我们提出卷积神经网络（CNN）用于图像分类； Transformer（TRM）处理序列。", page=2, score=0.9),
        ]) for d in doc_ids}),
        llm=llm, features={"skills": True, "glossary": True},
    )


def test_skill_glossary_routing_and_gating():
    r = default_router()
    assert r.route("/术语表", {"skills": True, "glossary": True}) is not None
    assert r.route("/glossary", {"skills": True, "glossary": True}) is not None
    assert r.route("/术语表", {"skills": True, "glossary": False}) is None
    assert r.route("/术语表", {"skills": False}) is None


def test_skill_glossary_requires_doc():
    async def run():
        spec = default_router().route("/术语表", {"skills": True, "glossary": True})
        res = await spec.run(_glossary_ctx([], FakeLLM()))
        assert not res.ok
        assert "文档" in res.error

    asyncio.run(run())


def test_skill_glossary_fake_fallback():
    async def run():
        spec = default_router().route("/术语表", {"skills": True, "glossary": True})
        res = await spec.run(_glossary_ctx([1], FakeLLM()))
        assert res.ok
        assert "确定性回退" in res.content  # fake 模型走回退
        assert "CNN" in res.content and "第 2 页" in res.content

    asyncio.run(run())


def test_skill_glossary_real_llm():
    async def run():
        spec = default_router().route("/术语表", {"skills": True, "glossary": True})
        table = "| 术语 | 缩写/英文 | 中文释义 | 首次出现页 | 相关概念 |\n|---|---|---|---|---|\n| CNN | CNN | 卷积神经网络 | 第 2 页 | 深度学习 |"
        res = await spec.run(_glossary_ctx([1], _ReportStubLLM(table)))
        assert res.ok
        assert "|" in res.content
        assert res.citations, "真实 LLM 路径应带回引用溯源"

    asyncio.run(run())


def test_skill_glossary_extracts_chinese_terms():
    """术语表应能抽取中文专业术语（如 旋转框遥感图像目标检测模型），不仅限于英文缩写。"""
    from papermate.skills.glossary import _extract_terms

    texts = [
        "针对遥感图像目标检测任务，我们提出一种轻量级的旋转框遥感图像目标检测模型（YOLO-RMV4）。",
        "该模型采用环形平滑标签（CSL）将角度回归问题转化为分类问题。",
    ]
    terms = _extract_terms(texts)
    term_names = {t["term"] for t in terms}
    assert "YOLO-RMV4" in term_names
    assert "CSL" in term_names
    # 中文专业术语应被识别
    assert any("旋转框" in t or "遥感图像" in t for t in term_names), f"未识别中文术语: {term_names}"


def test_summary_service_sentence_truncate():
    """_truncate_at_sentence 应在句子边界截断，避免切断语义。"""
    from papermate.services.summary_service import _truncate_at_sentence

    text = "第一句完整内容。第二句也完整。第三句继续补充说明更多细节。"
    out = _truncate_at_sentence(text, max_len=22)
    assert out.endswith("。") or out.endswith("…")
    # 不应在句中切断
    assert "第三句" not in out or out.endswith("第三句继续补充说明更多细节。") or out.endswith("…")


def test_skill_reproduction_uses_full_sentences():
    """/复现 应按句子抽取，不应因 PDF 行断开而输出半截 bullet。"""
    r = default_router()
    spec = r.route("/复现", {"skills": True})
    assert spec is not None
    # chunk 内部含换行：旧逻辑会按行抽取并输出 "- 我们使用 batch size 128" 这种半截句
    chunks = [RetrievedChunk(id="d1-c0", text="我们使用 batch size 128\n和学习率 0.001 训练模型。", page=1, score=0.9)]
    svc = _MockDocSvc({1: _ConstPipeline(chunks)})

    async def run():
        ctx = SkillContext(
            query="/复现", args="复现", doc_ids=[1], doc_svc=svc, llm=FakeLLM(),
            features={"skills": True},
        )
        res = await spec.run(ctx)
        assert res.ok
        assert "batch size 128" in res.content
        # 不应出现只含半截子的 bullet（旧逻辑会把 "我们使用 batch size 128" 单独成行）
        assert "- 我们使用 batch size 128\n" not in res.content
        # 输出应包含完整句子
        assert "我们使用 batch size 128 和学习率 0.001 训练模型。" in res.content
        # 重构后：按类别分组（不应再是挤成一团无逻辑的纯段落），且每条带页码溯源
        assert "**优化器**" in res.content or "**学习率**" in res.content
        assert "（p.1）" in res.content or "（p.3）" in res.content, "候选应带页码溯源"
        # 引用溯源必须回填（与 /综述 等技能一致，修复此前 citations=[] 的缺陷）
        assert res.citations, "复现技能必须回填 citations"
        assert any(c.get("page") for c in res.citations), "citation 应含页码"

    asyncio.run(run())


def test_summary_honors_runtime_settings_default_model():
    """回归：summary_service 必须读 runtime_settings 里的 default_model（设置页配置的模型），
    而不是只认 config 默认的 fake。否则在设置页配了 DeepSeek 仍走离线回退、永远提示「确定性回退」。"""
    from papermate.config import Settings

    svc = SummaryService(None, Settings(default_model="fake"), None)
    fake_state = {
        "default_model": "deepseek-v4-flash-vision-exp",
        "providers": {
            "deepseek": {"api_key": "sk-test", "base_url": "", "model": "deepseek-v4-flash-vision-exp"},
        },
    }
    with mock.patch.object(sm.runtime_settings, "get_state", return_value=fake_state):
        assert svc._effective_model() == "deepseek-v4-flash-vision-exp"
        llm, provider = _build_llm(svc._effective_model(), svc._settings)
        assert provider == "deepseek"  # 而非 "fake" —— 证明真实模型被采用，不会回退离线



# ---------- 2026-08-29 对比回复逻辑重梳理：句界判断 / 流动化 / 完整句单元格 ----------

def test_is_sentence_end_skips_abbreviations():
    """英文缩写点（et al. / Fig. / e.g.）与小数点不应被误判为句末。"""
    from papermate.services.summary_service import _is_sentence_end

    t = "the method of Smith et al. works well"
    i = t.index("al.") + 2  # '.' 位置
    assert not _is_sentence_end(t, i)
    t2 = "as shown in Fig. 3 the accuracy rises"
    i2 = t2.index("Fig.") + 3
    assert not _is_sentence_end(t2, i2)
    t3 = "the value is 0.95 which is high"
    i3 = t3.index("0.95") + 2  # 小数点
    assert not _is_sentence_end(t3, i3)
    t4 = "训练完成。模型收敛"
    assert _is_sentence_end(t4, t4.index("。"))


def test_flow_text_merges_mid_sentence_linebreaks():
    """句中折行应并入所在句子；句末/段落换行保留。"""
    from papermate.services.summary_service import _flow_text

    text = "我们提出一种新的检测框架\n并在两个数据集上验证。模型显著优于基线。\n\n第二段开始。"
    out = _flow_text(text)
    assert "\n" not in out.split("。")[0]          # 第一句内无折行
    assert "检测框架并在两个数据集上验证。" in out    # 折行并入句子（CJK 直接相接）
    assert "\n\n" in out                           # 段落间隔保留


def test_first_sentences_returns_complete_sentences():
    """_first_sentences 应返回完整句子，不以半截短语结尾。"""
    from papermate.services.summary_service import _first_sentences

    text = "本文提出目标检测新方法。实验在 COCO 数据集上进行，mAP 达到 50.2%。此外还做了消融实验证明各模块有效。"
    out = _first_sentences(text, max_sentences=2, max_len=200)
    assert out.endswith("。")
    assert "消融实验" not in out                    # 只取前 2 句
    # 单句超长时句子级截断兜底
    long_text = "这是一个非常长的句子没有其他内容" * 40
    out2 = _first_sentences(long_text, max_sentences=2, max_len=50)
    assert len(out2) <= 52


def test_compare_cells_are_sentence_briefs():
    """compare 的对比表单元格应为完整句简述（≤2 句），不再是 5×500 字大段。"""
    from unittest import mock as _mock

    import papermate.services.summary_service as sm

    fake_chunks = [
        {"doc_id": 1, "filename": "a.pdf", "seq": 0, "page": 1,
         "text": "本文提出一种新方法，实验表明效果优于基线。该方法的核心是动态注意力机制。另外还引入了多尺度特征融合模块。"},
        {"doc_id": 2, "filename": "b.pdf", "seq": 0, "page": 1,
         "text": "我们提出另一种方法，与已有工作对比。实验结果验证了设计的有效性。"},
    ]

    class _FakeSvc:
        async def compare(self, doc_ids, lang="zh"):
            return await sm.SummaryService.compare(_FakeSvc(), doc_ids, lang=lang)

    async def run():
        svc = sm.SummaryService.__new__(sm.SummaryService)
        svc._docs = None
        svc._settings = _mock.Mock(default_model="fake")
        svc._Session = None
        with _mock.patch.object(sm.runtime_settings, "get_state", return_value={"default_model": "fake", "providers": {}}):
            with _mock.patch.object(svc, "_gather_chunks", side_effect=lambda d: [c for c in fake_chunks if c["doc_id"] == d[0]]):
                out = await svc.compare([1, 2], lang="zh")
        for dim in out["dimensions"]:
            for cell in dim["cells"]:
                content = cell["content"]
                if content == "（未在分块中检测到相关内容）":
                    continue
                # 单元格内容应为句子级简述：不含句中换行，长度受控
                assert "\n" not in content or content.endswith(("。", "…"))
                assert len(content) <= 220, f"单元格过长: {len(content)}"
                # ★ O-03（2026-10-01）：单元格必须带 `filename` —— 前端
                # `CompareDimensionCell.filename` 读的就是它；修复前只有 `label`
                # ⇒ `c.filename` 恒 undefined ⇒ 对比表每行渲染成「：内容」（文档名消失）。
                assert cell.get("filename") in {"a.pdf", "b.pdf"}, cell

    asyncio.run(run())


def test_skill_deep_read_structured_mention_priority():
    """2026-09-07 回归：结构化提及 mention_doc_ids 优先于 args 里的 @文档N 文本解析
    （前端展示顺序与后端解析顺序不一致时，文本解析会指向错误文档）。"""
    async def run():
        spec = default_router().route("/精读", {"skills": True, "deep_read": True})
        ctx = SkillContext(
            query="/精读 @文档一", args="@文档一", doc_ids=[11, 22],
            mention_doc_ids=[22],  # 前端明确指向 22（尽管 label 文本写的是 @文档一）
            doc_svc=_MockDocSvc({
                11: _ConstPipeline([RetrievedChunk(id="d11-c0", text="第一篇的段落。", page=1, score=0.9)]),
                22: _ConstPipeline([RetrievedChunk(id="d22-c0", text="第二篇独有的结论段落。", page=1, score=0.9)]),
            }),
            llm=FakeLLM(), features={"skills": True, "deep_read": True},
        )
        res = await spec.run(ctx)
        assert res.ok, res.error
        assert "第二篇独有的结论段落" in res.content
        assert "第一篇的段落" not in res.content

    asyncio.run(run())
