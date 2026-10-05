"""PaperMate · 意图分类（P1-1：显式意图分类头）

把用户问题先归为四类之一，再决定是否强制检索，把「隐式推断」变为「显式决策」：

| 意图 | 判定 | 是否强制检索 |
|---|---|---|
| command   | 以 / 开头（/总结 /对比 等） | 是（对应 Service 处理） |
| doc_qa    | 提及论文/文中/摘要/方法等，或含疑问词 | 是（文档问答） |
| chitchat  | 问候/致谢/闲聊（短句） | 否（简短回应） |
| general_qa| 其余（通用知识咨询） | 否（直答或联网搜索） |

纯规则实现（零依赖、可单测）；分类结果作为 SYSTEM 注记注入对话，
与 SYSTEM_PROMPT 中的【意图分类】段共同作用，帮助 LLM 稳定理解用户意图。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: 命令前缀（/总结 /对比 等，规格 ER-24 的命令面板语义）
_COMMAND_RE = re.compile(r"^\s*/")
#: 寒暄短句标记——中文组：整句很短且包含即命中（中文无词边界概念）
_CHITCHAT_MARKERS = (
    "你好", "您好", "谢谢", "感谢", "辛苦了", "再见", "拜拜", "在吗", "在么",
    "好的", "明白了",
)
#: 寒暄英文标记——**用词边界匹配**（M13 修复 2026-09-10）：
#: 原实现把 "hi " 带尾空格塞进 `any(m in low ...)`，导致单发 "hi" 不命中；
#: 而直接去掉空格又会误伤 "this"/"which"/"ok" 之外的包含词，故改用 \b 边界。
_CHITCHAT_EN_RE = re.compile(r"\b(?:hi|hello|hey|thanks|thank you|bye|ok)\b")
#: 文档问答强标记（出现即倾向 doc_qa）。
#: **只收录"文档自身的结构 / 元数据"词**，刻意排除「模型 / 方法 / 实验 / 综述 / 创新 /
#: 指标 / 数据集 / 介绍 / 对比 / 作者」等**泛研究话题词**——后者在通用知识提问里同样高频
#: （如「大语言模型是怎么工作的」「目标检测有哪些常用方法」「推荐几篇综述」），若据此判
#: doc_qa 会把通用问题强推进 retrieve，模型再按"文档优先"提示词回出"当前会话关联的文档…"
#: 这类话术（用户报的症状）。指代用户自己文档的能力交给 has_document_reference 的强指代判定。
_DOC_QA_MARKERS = (
    "摘要", "abstract", "结论", "参考文献", "表格", "公式",
    "第几章", "第几节", "第几页", "章节", "附录",
)

#: 强文档指代标记：仅当 query 明确指代「用户已关联的文档」时命中（用于无文档时协议级拦截）。
#: 只收录**指示词 / 领属词 + 文档**这类强指代，刻意排除「论文 / 文档 / 摘要 / 方法 / 综述」等
#: 仅作为**话题**出现的通用研究型提问（如「搜集关于目标检测的前沿论文」「推荐几篇综述」
#: 「写一篇方法综述」），后者无需用户文档即可由大模型通用知识（或联网）作答，不应被拦截。
#: 若命中这些标记，说明用户确实在问「自己那份文档」——无文档时必然无法作答，应直接提醒关联文档；
#: 反之（无强指代）则放行给大模型基于通用知识/联网回答，符合"无文档→纯对话模式"的工作状态。
_DOC_OWNED_MARKERS = (
    # 指示代词 + 文档名词（"这篇/那篇论文"、"本文"、"该文档"…）
    "这篇", "那篇", "这两篇", "那两篇", "该论文", "该文档", "本论文", "本文", "本篇", "此文",
    # 领属 / 上传动作（"我上传的"、"关联的文档"、"已上传"、"我的论文"…）
    "我上传", "上传的", "我关联", "关联的文档", "已上传", "我的文档", "我的论文", "我刚传",
    # 回指 / 方位（针对某篇具体文档："文中"、"文档里"、"上述"、"前文"…）
    "文中", "文档里", "文档中", "论文里", "论文中", "上面", "上述", "前述", "前文", "前文提到",
    # 结构指代（针对某篇具体文档的章节 / 附录："第3章"、"参考文献"…）
    "第几章", "第几节", "第几页", "参考文献",
)


def has_document_reference(query: str) -> bool:
    """query 是否明确指代『用户已关联的文档』（用于无文档时的协议级拦截判定）。

    只匹配指示词 / 领属词 + 文档的强指代（如「这篇论文」「我上传的文档」「文中提到」），
    不匹配「论文 / 文档」仅作为**话题**的通用研究型提问（如「搜集关于目标检测的前沿论文」
    「推荐几篇综述」）——后者应放行给大模型基于通用知识（或联网）作答，而非弹"请关联文档"。
    """
    q = query or ""
    return any(m in q for m in _DOC_OWNED_MARKERS)


@dataclass(frozen=True)
class Intent:
    name: str  # doc_qa | general_qa | command | chitchat
    label: str  # 中文标签（注入 SYSTEM 用）
    force_retrieval: bool  # 是否建议强制检索


def classify_intent(query: str, *, has_doc_mention: bool = False) -> Intent:
    """纯规则意图分类。输入用户问题，输出 Intent。

    ### `has_doc_mention`（2026-10-02，⑤）：**结构化 @ 提及**也算"针对文档"
    起因（实测，见 `docs/诊断-离题回答与熔断预警-2026-10-02.md` §1.4）：用户输入
    「@文档四 实验对象是什么」时，`@文档N` **不在** `_DOC_OWNED_MARKERS` 里、
    「实验对象」也不在 `_DOC_QA_MARKERS` 里 ⇒ 被判 `general_qa` ⇒ 注入给模型的 SYSTEM 注记是
    「**与具体文档无关**：可直接回答」；可同时会话是 doc-bound、`force_tool_choice="required"`
    又强制它先检索 ⇒ **两个信号互相矛盾**（日志可证：`intent=general_qa doc_bound=True`）。

    ⚠️ 为什么不靠"把 `@文档N` 加进词表"：前端发的是**结构化字段** `mention_doc_ids`
    （可靠、已有序、还做过有效性过滤），文本里的「@文档四」只是回退渲染 ⇒
    **以结构化输入为准**，文本词表只作为没有结构化字段时的兜底。

    判定次序（刻意）：空 → 命令 → 短寒暄 → **@提及** → 文档词/指代 → 通用问答。
    寒暄排在 @提及之前：用户 @ 了文档但说的是"你好"，意图仍是寒暄（不必检索）。
    """
    q = (query or "").strip()
    if not q:
        return Intent("chitchat", "寒暄", False)
    if _COMMAND_RE.match(q):
        return Intent("command", "命令", True)
    low = q.lower()
    if len(q) <= 12 and (any(m in low for m in _CHITCHAT_MARKERS) or _CHITCHAT_EN_RE.search(low)):
        return Intent("chitchat", "寒暄", False)
    if has_doc_mention:
        return Intent("doc_qa", "文档问答", True)
    if any(m in q for m in _DOC_QA_MARKERS) or has_document_reference(q):
        return Intent("doc_qa", "文档问答", True)
    return Intent("general_qa", "通用问答", False)


def intent_system_note(query: str, *, has_doc_mention: bool = False) -> str:
    """生成注入对话的意图 SYSTEM 注记（供 _build_messages / routes_search 使用）。

    ★ `has_doc_mention` 必须与调用方 `classify_intent` 的实参**一致** ——
    否则会出现"路由按 doc_qa 走、注记却说与文档无关"的分裂（⑤ 的修复前提）。
    """
    it = classify_intent(query, has_doc_mention=has_doc_mention)
    hint = {
        "command": "按对应服务处理该命令。",
        "doc_qa": "问题针对文档内容：若会话已关联文档，必须先用 retrieve 检索后再作答。",
        "general_qa": "与具体文档无关：可直接回答，必要时开启『联网』调用大模型原生联网搜索。",
        "chitchat": "简短友好回应即可，不要调用检索工具。",
    }[it.name]
    return f"【意图识别】用户本轮提问属于「{it.label}」。{hint}"
