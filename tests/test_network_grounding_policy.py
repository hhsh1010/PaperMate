"""N12 回归（2026-09-12）：联网开启时的系统注记必须**场景化 + 通用 + 防幻觉**。

背景与用户需求：
用户观察「开启联网后，不特意强调模型就不会联网」。取证结论：
① `enable_search=True` 只是**授权能力**，是否真搜由模型自行判断 → 需要给出**可判断的触发场景**；
② 原注记标题为「联网限制」、措辞「仅当文档不足时**方可**启用」= 纯约束、零正向引导 → 模型趋于不用；
③ **是否具备联网能力随模型而变**（实测：qwen3-max/qwen-max/qwen3.6/3.7-flash/qwen-turbo 支持；
   qwen-plus/qwen3-vl-flash 与 DeepSeek 系列不支持）→ 注记**必须与厂商无关**，
   并明确「不具备时如实说明、不得假装检索」。

本文件锁定这三条：场景化触发、通用性（不含任何厂商/模型名）、能力前提与防幻觉。
"""

from __future__ import annotations

from papermate.services.chat_service import _grounding_block

_VENDOR_NAMES = (
    "qwen", "Qwen", "QWEN",
    "deepseek", "DeepSeek",
    "dashscope", "DashScope",
    "gpt", "GPT", "openai", "OpenAI",
    "claude", "Claude",
    "glm", "GLM", "ernie", "文心",
)


def _assert_generic(block: str) -> None:
    """通用性：注记不得绑定任何厂商 / 模型名（能力归属随模型而变）。"""
    for name in _VENDOR_NAMES:
        assert name not in block, f"联网注记不应绑定具体厂商/模型：{name}"


def test_network_on_with_docs_is_scenario_driven():
    """联网开启 + 已关联文档：必须给出可判断的触发场景，而非只写「可用」。"""
    block = _grounding_block(enable_network=True, has_docs=True)

    assert "主动联网核实" in block, "必须明确要求『主动联网核实』"
    assert "最新" in block, "缺少『最新进展/新版本』类触发场景"
    assert "实时" in block, "缺少『实时数据』类触发场景"
    assert "没有把握" in block, "缺少『自己没把握』类触发场景"
    assert "无需" in block, "缺少反向边界（避免为了联网而联网）"

    # 由「限制」改为「能力」口径：不应再出现抑制语义的标题
    assert "【联网限制" not in block, "标题不应再使用『限制』（会抑制模型使用）"
    assert "方可" not in block, "不应再用『方可』这类兜底式窄触发措辞"


def test_network_on_declares_capability_premise_and_anti_hallucination():
    """联网开启：必须声明「能力取决于模型」并禁止假装检索（防幻觉）。"""
    for has_docs in (True, False):
        block = _grounding_block(enable_network=True, has_docs=has_docs)
        assert "取决于当前所用模型" in block, "必须声明能力随模型而变"
        assert "如实说明无法联网" in block, "能力不可用时必须如实说明"
        assert "严禁假装检索" in block or "严禁假装检索过" in block, "必须禁止假装检索"
        _assert_generic(block)


def test_network_on_keeps_source_attribution_and_doc_priority():
    """保留既有硬约束：文档优先 + 来源标注 + 区分文档/联网 + 反编造。"""
    block = _grounding_block(enable_network=True, has_docs=True)
    assert "首要依据" in block
    assert "标注来源" in block
    assert "来自文档" in block and "来自联网" in block
    assert "严禁编造" in block


def test_network_off_unchanged_behavior():
    """联网关闭：保持原有严格口径（不得引入正向联网引导）。"""
    block = _grounding_block(enable_network=False, has_docs=True)
    assert "联网功能未开启" in block
    assert "严禁调用" in block
    assert "主动联网核实" not in block, "关闭时不应出现联网引导"


def test_no_docs_branch_keeps_required_phrases():
    """兼容既有测试与产品口径：无文档分支的关键短语不得丢失。"""
    for net in (False, True):
        block = _grounding_block(enable_network=net, has_docs=False)
        assert "【对话模式（当前：未关联文档" in block
        assert "不要输出『过程：』" in block
