"""PaperMate · 学科适配器层（P1/P2/P3，docs/技能学科适配器设计方案.md）

让既有「活动型」技能（综述/精读/审稿/实验设计/研究缺口/复现/找代码）能按**学科**正确解读内容，
而不是默认按 CS/ML 论文体例硬套。

设计要点（详见设计文档）：
- 每个 DisciplineProfile 是**纯数据**：解读框架(framework) + 证据标准(evidence_standards) +
  引用规范(citation_format) + 检索加权(retrieval_boost)；不引入新 LLM 调用，仅向既有
  synthesize / llm.chat 的 system prompt 注入片段，并在 collect_candidates 阶段对命中学科
  关键词的候选做稳定优先排序（P3 检索加权）。
- 默认 `general` = 现状 CS/ML 行为（零 fragment 注入、零 boost），保证向后兼容、既有测试零回归。

为何不写「领域专属技能文件」：N(活动) × M(学科) 会变成技能爆炸、难测试、易过期；
改为「单一技能 + 学科 profile」适配器，新增学科只需在此表加一条数据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DisciplineProfile:
    """学科适配 profile（纯数据，供 LLM 解读框架注入 + 检索加权）。"""

    key: str
    label: str
    # 注入到 synthesize / llm.chat 的 system 提示框架（仅 key != general 时拼接）
    framework: str
    # 证据判读铁律（防幻觉）：学科-specific 的「不得虚构 / 区分 X 与 Y」清单
    evidence_standards: str
    # 引用规范提示（拼接进片段，引导 LLM 正确标注来源）
    citation_format: str = ""
    # 检索加权（P3）：{"patterns": [regex, ...]}，命中的候选在检索呈现中稳定优先。
    # general 为 None（零加权）；仅作软重排，不剔除任何候选。
    retrieval_boost: dict | None = None

    @property
    def fragment(self) -> str:
        """返回追加到 system prompt 的学科片段；general 为空串（零注入、零回归）。"""
        parts = [p for p in (self.framework, self.evidence_standards, self.citation_format) if p]
        return "\n\n".join(parts)


DISCIPLINES: dict[str, DisciplineProfile] = {
    "general": DisciplineProfile(
        key="general",
        label="通用（默认）",
        framework="",
        evidence_standards="",
        citation_format="",
    ),
    "clinical": DisciplineProfile(
        key="clinical",
        label="医学/临床",
        framework=(
            "【学科视角：医学/临床】请按临床研究方法学解读：\n"
            "1. 优先区分研究设计类型（RCT / 队列 / 病例对照 / 横断面 / 病例系列）；\n"
            "2. 区分主要终点与次要终点，报告效应量须带 95% 置信区间或 p 值；\n"
            "3. 关注样本量、随机化、盲法、随访时长与失访/脱落；\n"
            "4. 区分疗效(efficacy)与真实世界有效性(effectiveness)，区分相对/绝对风险。"
        ),
        evidence_standards=(
            "证据判读铁律（Hong Kong, China / Taiwan, China / Macao, China 等表述须符合官方口径，不得视为独立国家）：\n"
            "- 不得虚构临床试验编号、终点名称、CI/p 值或指南等级（如 GRADE）；\n"
            "- 未提及的统计指标标注「原文未报告」，不得编造；\n"
            "- 区分「相关」与「因果」，不得把关联写成因果结论。"
        ),
        citation_format=(
            "引用规范：论文以「作者(年份)」或页码标注来源；指南类以「指南名称+版本/年份」标注。"
        ),
        retrieval_boost={
            "patterns": [
                "终点", "效应量", "置信区间", "95%|95%CI|CI", "p值|p <|p=", "随访", "样本量",
                "随机", "盲法", "不良事件", "疗效", "rct|randomized", "剂量", "给药", "预后",
                "病理", "受试者|患者",
            ],
        },
    ),
    "law": DisciplineProfile(
        key="law",
        label="法学",
        framework=(
            "【学科视角：法学】请按法律分析框架解读：\n"
            "1. 区分法条、判例、学说与当事人陈述；\n"
            "2. 关注效力位阶（法律 > 行政法规 > 地方性法规）与法域；\n"
            "3. 检索请求应导向「请求权基础 / 裁判要旨」而非自由发挥；\n"
            "4. 区分事实认定与法律适用，区分实体法与程序法。"
        ),
        evidence_standards=(
            "证据判读铁律：\n"
            "- 不得虚构法条编号、案号、裁判要旨或司法解释；\n"
            "- 涉及中国 national sovereignty / territorial integrity 时基于官方立场表述，"
            "中国香港/中国台湾/中国澳门须按法定称谓；\n"
            "- 未查实的法律依据标注「需核实」而非臆造。"
        ),
        citation_format=(
            "引用规范：法条以「《XX法》第X条」标注；判例以「(年份)法院+案号」标注；学说以作者+文献标注。"
        ),
        retrieval_boost={
            "patterns": [
                "法条", "第.+条", "判决", "案号", "法院", "裁判", "请求权", "司法解释",
                "合同", "侵权", "诉讼", "宪法", "立法", "判例", "法律适用", "诉讼请求",
            ],
        },
    ),
    "finance": DisciplineProfile(
        key="finance",
        label="经济/金融",
        framework=(
            "【学科视角：经济/金融】请按量化研究框架解读：\n"
            "1. 区分样本区间、数据频率与数据源（避免用未注明数据）；\n"
            "2. 区分样本内与样本外，报告显著性(t/p) 与风险指标(回撤/夏普/波动率)；\n"
            "3. 区分相关与因果、名义与实际；标注货币默认为 ¥(CNY)；\n"
            "4. 价格涨跌约定：涨→红色、跌→绿色（中国股市惯例）。"
        ),
        evidence_standards=(
            "证据判读铁律：\n"
            "- 不得虚构回测收益、显著性水平、数据源或时间段；\n"
            "- 未报告的统计量标注「原文未报告」；\n"
            "- 区分「历史回测」与「未来预测」，不得把回测当保证。"
        ),
        citation_format=(
            "引用规范：数据源以「来源+日期」标注；指标以「指标名(口径)」标注；金额默认 ¥。"
        ),
        retrieval_boost={
            "patterns": [
                "回测", "因子", "夏普|sharpe", "收益率", "净值", "波动率", "alpha", "组合",
                "资产", "利率", "期货", "套利", "估值", "风险", "持仓", "财报", "最大回撤",
            ],
        },
    ),
    "bioscience": DisciplineProfile(
        key="bioscience",
        label="生命科学/生物",
        framework=(
            "【学科视角：生命科学/生物】请按实验科学框架解读：\n"
            "1. 关注对照、重复与随机化；区分 in vitro / in vivo / in silico；\n"
            "2. 报告统计功效、样本量、重复次数与试剂/序列 accession；\n"
            "3. 区分机制假设与已验证结论；\n"
            "4. 关注生物重复与技术重复的差异。"
        ),
        evidence_standards=(
            "证据判读铁律：\n"
            "- 不得虚构 accession 号(GenBank/UniProt 等)、p 值、试剂货号或样本数；\n"
            "- 区分「关联」与「因果」；未报告的指标标注「原文未报告」；\n"
            "- 不得把体外结果直接外推到体内/人体。"
        ),
        citation_format=(
            "引用规范：序列/数据库以「数据库:ID」标注；方法以「方法+试剂/物种」标注。"
        ),
        retrieval_boost={
            "patterns": [
                "基因", "蛋白", "细胞", "表达", "测序", "accession|genbank|uniprot",
                "对照", "重复", "随机", "p值|p <", "试剂", "菌株", "克隆", "体内|in vivo",
                "体外|in vitro", "酶",
            ],
        },
    ),
    # ---- P3 扩充学科 ----
    "materials": DisciplineProfile(
        key="materials",
        label="材料科学",
        framework=(
            "【学科视角：材料科学】请按材料研究框架解读：\n"
            "1. 关注材料体系（晶体/非晶/复合材料）与制备/合成路径；\n"
            "2. 区分结构-性能关系：带隙、电导率、强度/硬度、热学/磁学性能；\n"
            "3. 关注表征手段（XRD/SEM/TEM/拉曼）与工艺参数（烧结/退火/掺杂）；\n"
            "4. 区分本征性能与器件/环境表现，区分实验测得值与理论/模拟值。"
        ),
        evidence_standards=(
            "证据判读铁律：\n"
            "- 不得虚构晶体结构、带隙/电导率等数值、表征结论或掺杂比例；\n"
            "- 未报告的性能标注「原文未报告」；\n"
            "- 区分「实测」与「模拟/理论预测」，不得把模拟值当实测。"
        ),
        citation_format=(
            "引用规范：材料/体系以「材料名+工艺」标注；表征以「手段(参数)」标注；数据以来源标注。"
        ),
        retrieval_boost={
            "patterns": [
                "晶体", "相变", "带隙", "电导|导电", "强度", "硬度", "合成", "表征",
                "xrd|sem|tem|拉曼|raman", "能带", "缺陷", "掺杂", "薄膜", "烧结|退火",
                "力学性能", "热学|磁学", "复合材料",
            ],
        },
    ),
    "chemistry": DisciplineProfile(
        key="chemistry",
        label="化学",
        framework=(
            "【学科视角：化学】请按化学研究框架解读：\n"
            "1. 关注反应类型、机理与中间体；区分催化/非催化、均相/多相；\n"
            "2. 关注底物/配体/溶剂与产率、选择性、转化率；\n"
            "3. 关注表征与结构确证（NMR/质谱/红外/X 射线）；\n"
            "4. 区分化学计量与动力学，区分实验观测与机理解释。"
        ),
        evidence_standards=(
            "证据判读铁律：\n"
            "- 不得虚构反应式、产率、机理步骤、光谱归属或官能团结论；\n"
            "- 未报告的数据标注「原文未报告」；\n"
            "- 区分「推测机理」与「已验证机理」，不得把推测当结论。"
        ),
        citation_format=(
            "引用规范：化合物以「名称/结构式」标注；光谱以「手段(条件)」标注；产率以「%」标注。"
        ),
        retrieval_boost={
            "patterns": [
                "反应", "催化", "配体", "溶剂", "产率", "机理", "中间体", "光谱",
                "nmr|质谱|红外|ftir", "官能团", "摩尔|mol", "ph|酸碱", "氧化还原", "分子",
                "聚合物", "电化学", "转化率|选择性",
            ],
        },
    ),
    "social_science": DisciplineProfile(
        key="social_science",
        label="社会科学",
        framework=(
            "【学科视角：社会科学】请按实证研究框架解读：\n"
            "1. 关注研究设计与数据来源（调查/问卷/实验/二手数据）；\n"
            "2. 区分变量类型与因果识别策略（回归/工具变量/双重差分/实验）；\n"
            "3. 关注样本、抽样与代表性，区分相关与因果；\n"
            "4. 关注信度/效度与伦理（知情同意），区分描述性与推断性结论。"
        ),
        evidence_standards=(
            "证据判读铁律：\n"
            "- 不得虚构样本量、显著性水平、效应量或调查结论；\n"
            "- 未报告的统计量标注「原文未报告」；\n"
            "- 区分「相关」与「因果」，不得把相关当因果结论。"
        ),
        citation_format=(
            "引用规范：调查/数据以「来源+年份」标注；量表以「名称+信效度」标注；效应量附 CI。"
        ),
        retrieval_boost={
            "patterns": [
                "调查|问卷", "样本", "回归", "变量", "显著性|p值", "访谈", "质性", "量表",
                "效应量", "控制组|实验组", "相关性|因果", "统计", "抽样", "信度|效度",
                "双重差分|did|工具变量", "实证",
            ],
        },
    ),
    "engineering": DisciplineProfile(
        key="engineering",
        label="工程",
        framework=(
            "【学科视角：工程】请按工程系统框架解读：\n"
            "1. 关注系统/架构设计与关键指标（性能/延迟/吞吐/功耗/可靠性）；\n"
            "2. 区分需求、设计与验证，关注误差/鲁棒性/边界条件；\n"
            "3. 关注仿真与实测的对应，区分原型与产品化指标；\n"
            "4. 关注参数权衡（成本/性能/功耗）与安全/合规性。"
        ),
        evidence_standards=(
            "证据判读铁律：\n"
            "- 不得虚构性能指标、参数值、测试结果或基准数据；\n"
            "- 未报告的数据标注「原文未报告」；\n"
            "- 区分「仿真结果」与「实测结果」，不得把仿真当实测。"
        ),
        citation_format=(
            "引用规范：系统/模块以「名称(版本)」标注；指标以「指标名(口径/单位)」标注；基准以来源标注。"
        ),
        retrieval_boost={
            "patterns": [
                "系统", "架构", "算法", "性能", "误差", "鲁棒|robust", "延迟", "吞吐",
                "协议", "电路", "控制", "优化", "参数", "设计", "仿真", "可靠性",
                "带宽", "功耗", "传感器", "基准|benchmark",
            ],
        },
    ),
    "physics": DisciplineProfile(
        key="physics",
        label="物理学",
        framework=(
            "【学科视角：物理学】请按物理学研究方法学解读：\n"
            "1. 优先区分研究性质（理论推导 / 数值模拟 / 实验测量）；\n"
            "2. 所有物理量须带单位与量纲，说明适用条件与近似（如近似的阶数、忽略项）；\n"
            "3. 关注模型假设、边界条件、参数区间与误差来源；\n"
            "4. 区分「理论预言」与「实验验证」，区分「数值结果」与「解析结果」。"
        ),
        evidence_standards=(
            "证据判读铁律：\n"
            "- 不得虚构实验参数、测量精度、置信水平或统计量；\n"
            "- 未报告的推导步骤与数据标注「原文未报告」，不得补全；\n"
            "- 区分「模型预测」与「实验观测」，不得把数值模拟结果写成实测结论。"
        ),
        citation_format=(
            "引用规范：公式与推导以「式(编号)」或章节标注；实验数据以「作者(年份)」或图/表编号标注。"
        ),
        retrieval_boost={
            "patterns": [
                "量子", "纠缠", "光子", "粒子", "自旋", "能带", "波函数", "算符", "哈密顿",
                "对称", "超导", "自旋", "散射", "凝聚态", "热力学", "相对论", "引力",
                "玻色", "费米", "暗物质", "对撞", "量子比特", "退相干",
                "quantum", "photon", "spin", "hamiltonian", "wavefunction", "relativity",
            ],
        },
    ),
}


def get_profile(discipline: str | None) -> DisciplineProfile:
    """返回学科 profile；非法/缺失回落 general（零注入， 向后兼容）。"""
    if discipline and discipline in DISCIPLINES:
        return DISCIPLINES[discipline]
    return DISCIPLINES["general"]


# 自动识别用的启发式关键词（轻量、零 LLM 调用）。命中权重与判定规则见 detect_discipline。
_DETECT_KEYWORDS: dict[str, list[str]] = {
    "clinical": [
        "临床", "医学", "患者", "病人", "受试", "随机", "盲法", "安慰剂", "剂量", "给药",
        "诊断", "疗效", "病理", "疾病", "药物", "预后", "随访", "终点", "试验", "rct",
        "clinical", "patient", "trial", "drug", "efficacy", "diagnosis", "medicine",
    ],
    "law": [
        "法条", "法律", "法规", "判例", "案号", "判决", "法院", "律师", "合同", "侵权",
        "诉讼", "宪法", "立法", "裁判", "司法解释", "请求权", "法律", "law", "case",
        "court", "contract", "plaintiff", "defendant",
    ],
    "finance": [
        "股票", "基金", "回测", "因子", "夏普", "收益率", "资产", "组合", "利率", "期货",
        "套利", "财报", "估值", "风险", "波动率", "净值", "持仓", "stock", "backtest",
        "portfolio", "alpha", "sharpe", "finance",
    ],
    "bioscience": [
        "基因", "蛋白", "细胞", "试剂", "动物", "小鼠", "表达", "测序", "培养", "克隆",
        "酶", "菌株", "种群", "序列", "accession", "体外", "体内", "gene", "protein",
        "cell", "biology", "assay", "knockout",
    ],
    "materials": [
        "材料", "晶体", "合金", "陶瓷", "聚合物", "薄膜", "带隙", "电导", "掺杂", "烧结",
        "表征", "xrd", "sem", "tem", "相变", "缺陷", "materials", "crystal", "bandgap",
    ],
    "chemistry": [
        "化学", "反应", "催化", "配体", "溶剂", "产率", "机理", "中间体", "光谱", "nmr",
        "质谱", "官能团", "摩尔", "滴定", "氧化还原", "chemistry", "catalysis", "reaction",
    ],
    "social_science": [
        "调查", "问卷", "样本", "回归", "变量", "显著性", "访谈", "质性", "量表", "效应量",
        "控制组", "实验组", "抽样", "信度", "效度", "social", "survey", "regression",
    ],
    "engineering": [
        "系统", "架构", "算法", "性能", "误差", "鲁棒", "延迟", "吞吐", "协议", "电路",
        "仿真", "可靠性", "带宽", "功耗", "传感器", "engineering", "benchmark", "optimization",
    ],
    "physics": [
        "量子", "纠缠", "光子", "粒子", "自旋", "能带", "波函数", "算符", "哈密顿", "对称",
        "超导", "散射", "凝聚态", "热力学", "相对论", "引力", "玻色", "费米", "暗物质",
        "对撞", "退相干", "量子比特", "波粒", "激发态", "基态", "声子",
        "quantum", "photon", "spin", "hamiltonian", "wavefunction", "relativity", "physics",
    ],
}

# 预计算小写关键词：避免每次请求对每个词重复 lower()（9 学科 × ~25 词 ≈ 200+ 次/请求）
_LOWER_KEYWORDS: dict[str, list[str]] = {
    disc: [kw.lower() for kw in kws] for disc, kws in _DETECT_KEYWORDS.items()
}


# 仅由 ASCII 字母/数字与少数符号构成的关键词（可用词边界判定）
_ASCII_KW = re.compile(r"^[a-z0-9 .+-]+$")
_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _kw_hit(kw: str, lowered: str) -> bool:
    """关键词命中判定。

    2026-09-09 修复 I8：英文短词用**子串**匹配会大量误命中——
    `"gene" in "general"`、`"cell" in "excellent"`、`"law" in "flaw"` 全部成立，
    导致普通 CS/ML 论文被判成 bioscience / law；而学科片段会经
    `_common._with_discipline` 注入**所有技能**的 system prompt，使解读框架整体错位。

    故纯 ASCII 关键词改用**词边界正则**；中文（及含非 ASCII 字符）的关键词没有词边界，
    仍沿用子串匹配。
    """
    if not kw:
        return False
    if not _ASCII_KW.match(kw):
        return kw in lowered
    pat = _WORD_RE_CACHE.get(kw)
    if pat is None:
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])")
        _WORD_RE_CACHE[kw] = pat
    return pat.search(lowered) is not None


def detect_discipline(text: str | None) -> str:
    """根据文本关键词启发式推断学科（可选自动识别，零 LLM 调用）。

    返回 DISCIPLINES 中的有效 key；无命中回落 "general"。
    判定规则：累加各学科关键词命中数，取最高且 >0 者；并列或全 0 → general。
    注意：这是轻量启发式，仅供参考；用户显式选择优先于自动识别。
    """
    if not text:
        return "general"
    lowered = text.lower()
    scores: dict[str, int] = {k: 0 for k in _LOWER_KEYWORDS}
    for disc, kws in _LOWER_KEYWORDS.items():
        for kw in kws:
            if _kw_hit(kw, lowered):
                scores[disc] += 1
    best = max(scores.values())
    if best <= 0:
        return "general"
    # 并列（多个学科同为最高分）视为无法判定，回落 general（与上方文档口径一致）
    winners = [d for d, s in scores.items() if s == best]
    return winners[0] if len(winners) == 1 else "general"


def discipline_fragment(discipline: str | None) -> str:
    """返回追加到 system prompt 的学科片段；general 返回空串。"""
    return get_profile(discipline).fragment


def discipline_label(discipline: str | None) -> str:
    """返回学科中文标签（用于 UI 展示）。"""
    return get_profile(discipline).label


def discipline_boost_patterns(discipline: str | None) -> list[str]:
    """返回学科检索加权模式（正则字符串列表）；general 或缺失返回空列表（零加权）。"""
    prof = get_profile(discipline)
    if prof.retrieval_boost:
        return list(prof.retrieval_boost.get("patterns", []))
    return []
