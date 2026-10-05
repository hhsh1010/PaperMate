"""学科适配器层（P1）单元测试：registry 回落 + 片段注入。"""

import asyncio

from lightcore.llm.base import ChatMessage, Role
from papermate.skills._common import boost_candidates, synthesize
from papermate.skills.disciplines import (
    DISCIPLINES,
    detect_discipline,
    discipline_boost_patterns,
    discipline_fragment,
    get_profile,
)


class _CaptureLLM:
    """捕获最后一次 chat 调用的消息列表，用于断言 system prompt 注入。"""

    def __init__(self):
        self.last_messages: list[ChatMessage] = []

    async def chat(self, messages, **kwargs):
        self.last_messages = list(messages)
        return type("R", (), {"message": type("M", (), {"content": "ok"})()})()


def test_get_profile_fallback():
    # 缺失/非法 → general（零注入，向后兼容）
    assert get_profile(None).key == "general"
    assert get_profile("").key == "general"
    assert get_profile("not-a-real-disc").key == "general"
    # 合法 → 命中
    assert get_profile("clinical").key == "clinical"
    assert get_profile("law").key == "law"
    assert get_profile("finance").key == "finance"
    assert get_profile("bioscience").key == "bioscience"


def test_registry_completeness():
    assert set(DISCIPLINES.keys()) == {
        "general", "clinical", "law", "finance", "bioscience",
        "materials", "chemistry", "social_science", "engineering", "physics",
    }


def test_new_discipline_fragments_nonempty_with_guard():
    # P3 扩充学科：框架片段非空且含防幻觉铁律
    for key in ("materials", "chemistry", "social_science", "engineering", "physics"):
        frag = discipline_fragment(key)
        assert frag, f"{key} 应有非空框架片段"
        assert "不得虚构" in frag


def test_discipline_boost_patterns():
    # general / 缺失 → 空列表（零加权）
    assert discipline_boost_patterns("general") == []
    assert discipline_boost_patterns(None) == []
    assert discipline_boost_patterns("not-a-real-disc") == []
    # 各含 boost 的学科返回非空 patterns
    for key in ("clinical", "law", "finance", "bioscience",
                "materials", "chemistry", "social_science", "engineering", "physics"):
        pats = discipline_boost_patterns(key)
        assert isinstance(pats, list) and len(pats) > 0, f"{key} 应含检索加权模式"


def test_boost_candidates_general_preserves_order():
    cands = [
        {"sent": "本文提出一种新的神经网络架构用于图像分类", "page": 1, "doc_id": 1},
        {"sent": "实验在 ImageNet 上达到 SOTA", "page": 2, "doc_id": 1},
    ]
    # general：零加权，原序返回（同一列表对象）
    assert boost_candidates(list(cands), "general") == cands


def test_boost_candidates_reorders_by_discipline():
    cands = [
        {"sent": "本文提出一种新的神经网络架构用于图像分类", "page": 1, "doc_id": 1},
        {"sent": "该材料的晶体带隙为 2.1 eV，电导率显著提升", "page": 2, "doc_id": 1},
        {"sent": "我们通过 XRD 与 SEM 表征了薄膜的相变过程", "page": 3, "doc_id": 1},
    ]
    boosted = boost_candidates(list(cands), "materials")
    # 两条材料相关候选前置，非材料句子沉底；优先项总数不变
    assert len(boosted) == 3
    assert boosted[-1]["sent"] == "本文提出一种新的神经网络架构用于图像分类"
    assert {c["sent"] for c in boosted[:2]} == {
        "该材料的晶体带隙为 2.1 eV，电导率显著提升",
        "我们通过 XRD 与 SEM 表征了薄膜的相变过程",
    }


def test_detect_discipline_p3_disciplines():
    # P3 新学科启发式识别
    assert detect_discipline("该晶体合金经掺杂退火后带隙发生变化") == "materials"
    assert detect_discipline("该反应在催化下产率显著提升，NMR 证实了机理") == "chemistry"
    assert detect_discipline("通过问卷回归分析样本效应量并进行显著性检验") == "social_science"
    assert detect_discipline("系统架构的延迟与吞吐量满足鲁棒性基准要求") == "engineering"
    # 2026-08-30 新增：物理学（此前物理类查询恒回落 general，核心场景缺覆盖）
    assert detect_discipline("量子纠缠的非局域关联如何影响量子通信") == "physics"
    assert detect_discipline("光子晶体中的能带结构与光子传输") == "physics"


def test_detect_discipline_tie_falls_back_to_general():
    """并列最高分视为无法判定 → general（与 detect_discipline 文档口径一致）。"""
    # 「量子」(physics) 与「基因」(bioscience) 各命中 1 次 → 并列
    assert detect_discipline("量子基因") == "general"
    # 单一学科明显占优时不回落
    assert detect_discipline("量子纠缠与量子比特的退相干") == "physics"


def test_fragment_general_empty():
    assert discipline_fragment("general") == ""
    assert discipline_fragment(None) == ""


def test_fragment_nonempty_and_no_hallucination_guard():
    law = discipline_fragment("law")
    assert "法条" in law and "不得虚构法条编号" in law
    clinical = discipline_fragment("clinical")
    assert "研究设计类型" in clinical and "不得虚构" in clinical
    finance = discipline_fragment("finance")
    assert "样本内" in finance and "回测" in finance
    bioscience = discipline_fragment("bioscience")
    assert "in vitro" in bioscience and "accession" in bioscience


def test_synthesize_injects_discipline_into_system():
    llm = _CaptureLLM()
    asyncio.run(synthesize(llm, "BASE SYSTEM", "hello", discipline="clinical"))
    sys_msg = next(m for m in llm.last_messages if m.role == Role.SYSTEM)
    assert "BASE SYSTEM" in sys_msg.content
    assert "医学/临床" in sys_msg.content


def test_synthesize_general_no_injection():
    llm = _CaptureLLM()
    asyncio.run(synthesize(llm, "BASE SYSTEM", "hello", discipline="general"))
    sys_msg = next(m for m in llm.last_messages if m.role == Role.SYSTEM)
    assert sys_msg.content == "BASE SYSTEM"


def test_detect_discipline_heuristic():
    # 临床：命中 RCT / 受试者 / 疗效
    assert detect_discipline("这篇 RCT 试验评估了药物的疗效与受试者安全") == "clinical"
    # 法学：命中 法条 / 判例
    assert detect_discipline("请分析该法条与判例的请求权基础") == "law"
    # 金融：命中 回测 / 因子
    assert detect_discipline("用因子模型做回测，看夏普和回撤") == "finance"
    # 生命科学：命中 基因 / 细胞
    assert detect_discipline("该基因的蛋白在细胞中表达，做了 knockout") == "bioscience"
    # 无学科关键词 → general
    assert detect_discipline("帮我写一首诗") == "general"
    assert detect_discipline("") == "general"
    assert detect_discipline(None) == "general"
