"""生成学科适配器测试用样本论文（PDF）。

原因：PaperMate 沙箱出网隔离，arXiv/SemanticScholar 等被防火墙或限流，
无法稳定抓取覆盖 8 个学科（尤其法学/材料/化学/金融）的真实论文。
本脚本在本地用 PyMuPDF 生成 9 篇「学科打标」样本 PDF，每篇密集包含对应
学科的 detect_discipline 关键词与 retrieval_boost 模式，便于端到端验证：
  1) 会话选「自动」时按内容正确识别学科；
  2) 注入对应学科解读框架（framework + 证据铁律）；
  3) 技能抽取候选时命中学科模式的句子稳定前置（检索加权）。

生成目录：data/sample_papers/（文件名带 sample_ 前缀，便于与真实论文区分）。
用户可随时用真实 PDF 替换：通过前端上传，或放入 data/uploads/ 后经 API 入库。
"""

from __future__ import annotations

import os

import pymupdf

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"  # 黑体，覆盖简体中文
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_papers")
PAGE_W, PAGE_H = 595.0, 842.0  # A4
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 60.0, 60.0, 60.0
FS_TITLE, FS_H, FS_BODY = 16.0, 12.5, 11.0
LINE_GAP = 1.5
CHARS_PER_LINE = 42  # 中文近等宽，按字数换行


def wrap(text: str, per_line: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= per_line:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


def write_pdf(path: str, title: str, paragraphs: list[str]) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN_TOP

    def ensure_space(need: float) -> None:
        nonlocal page, y
        if y + need > PAGE_H - MARGIN_BOTTOM:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN_TOP

    # 标题
    for ln in wrap(title, 28):
        ensure_space(FS_TITLE * LINE_GAP)
        page.insert_text((MARGIN_X, y), ln, fontname="simhei", fontfile=FONT_PATH, fontsize=FS_TITLE, color=(0, 0, 0))
        y += FS_TITLE * LINE_GAP
    y += 6

    for para in paragraphs:
        # 二级标题（以 "## " 开头的段落）
        if para.startswith("## "):
            ensure_space(FS_H * LINE_GAP)
            for ln in wrap(para[3:], 36):
                ensure_space(FS_H * LINE_GAP)
                page.insert_text((MARGIN_X, y), ln, fontname="simhei", fontfile=FONT_PATH, fontsize=FS_H, color=(0.1, 0.1, 0.5))
                y += FS_H * LINE_GAP
            y += 4
            continue
        for ln in wrap(para, CHARS_PER_LINE):
            ensure_space(FS_BODY * LINE_GAP)
            page.insert_text((MARGIN_X, y), ln, fontname="simhei", fontfile=FONT_PATH, fontsize=FS_BODY, color=(0, 0, 0))
            y += FS_BODY * LINE_GAP
        y += 4

    doc.save(path, deflate=True, garbage=4)
    doc.close()


# 每篇样本：标题 + 摘要 + 若干段落（含学科关键词与"本文提出/实验结果表明"等技能通用触发句）
PAPERS: dict[str, tuple[str, list[str]]] = {
    "sample_general_cs.pdf": (
        "基于 Transformer 的文档检索与问答方法综述",
        [
            "摘要：本文提出一种基于 Transformer 的检索增强生成框架，在公开数据集上与强基线对比，"
            "通过消融实验验证了各模块的贡献。实验结果表明该方法在准确率上显著优于已有工作。",
            "## 方法与实验",
            "我们在公开视觉数据集上训练模型，使用 ResNet 作为基线，并设计消融实验评估注意力模块。"
            "训练采用 Adam 优化器，学习率设为 3e-4，批次大小为 256，训练 100 个 epoch。",
            "实验结果表明，本文方法在验证集上达到 SOTA，F1 指标较基线提升 4.2%。我们证明了检索模块对最终效果贡献最大。",
            "## 局限与展望",
            "本文的不足在于依赖大规模标注数据，未来工作将探索低资源场景下的迁移能力，并对比更多基线方法。",
        ],
    ),
    "sample_clinical.pdf": (
        "一项评估新药给药方案疗效的随机双盲对照试验",
        [
            "摘要：本研究为一项随机对照试验（RCT），采用双盲法评估新药给药方案对主要终点的影响。"
            "共纳入 320 名受试者，随机分为试验组与对照组，评估疗效与不良事件。",
            "## 研究设计",
            "本研究为前瞻性随机对照试验，受试者按 1:1 随机分配，研究者与受试者均设盲（双盲法）。"
            "主要终点为 24 周时的疗效应答率，次要终点包括不良事件发生率与生活质量评分。",
            "我们按照 GCP 规范给药，试验组给予目标剂量药物，对照组给予安慰剂，随访 24 周记录终点事件。",
            "## 统计分析与结果",
            "采用意向性治疗分析，效应量以风险比及 95% 置信区间（95%CI）报告，组间比较 p 值 < 0.05 视为显著。"
            "实验结果表明，试验组主要终点疗效显著优于对照组，置信区间未跨过无效线。",
            "随访期间试验组不良事件发生率较低，未发生严重预后不良事件；我们证明了该给药方案的安全性。本文的局限在于样本量有限，需更大队列验证。",
        ],
    ),
    "sample_law.pdf": (
        "合同效力瑕疵的请求权基础与司法裁判规则研究",
        [
            "摘要：本文分析合同条款因违反法律效力性强制性规定而无效的情形，探讨请求权基础的构造与裁判要旨的形成。",
            "## 法律规范与请求权基础",
            "根据《合同法》第 52 条规定，损害社会公共利益的合同条款无效。法院在判决中援引先前判例确立裁判要旨，"
            "支持原告的请求权基础，并区分合同效力与合同履行。",
            "本案案号（2023）某法民终 1234 号，一审法院认定事实后适用法律，二审法院维持判决。"
            "我们分析了宪法位阶下立法与司法解释的适用，明确侵权与诉讼的边界。",
            "## 裁判要旨与结论",
            "本文提出应以请求权基础为主线统一法律适用标准；我们证明了在类似案件中判例对裁判具有拘束力。"
            "研究的不足在于缺乏跨法域比较，需进一步核实不同法院的诉讼请求处理差异。",
        ],
    ),
    "sample_finance.pdf": (
        "基于多因子模型的组合回测与风险控制研究",
        [
            "摘要：本文构建多因子组合并进行样本内回测，评估夏普比率、最大回撤与波动率等风险指标。",
            "## 因子与组合构建",
            "我们选取价值、动量等因子构建资产组合，基于财报与估值模型计算持仓权重，并以利率期货进行对冲。"
            "组合再平衡频率为月度，交易成本纳入净值计算。",
            "## 回测与风险分析",
            "该因子组合经样本内回测，夏普比率达 2.1，最大回撤控制在 8%，年化收益率 15%。"
            "实验结果表明，套利策略在样本外波动率上升时风险暴露增加；我们证明了因子暴露于市场 beta 的部分可被对冲。",
            "本文的局限在于回测依赖历史数据，未来预测存在不确定性，需结合宏观利率环境重新评估持仓风险。",
        ],
    ),
    "sample_bioscience.pdf": (
        "基因过表达对细胞表型影响的体外与体内验证",
        [
            "摘要：本研究在体外培养细胞中过表达目标基因，并通过测序验证其 accession 号，评估对细胞表型的影响。",
            "## 材料与方法",
            "我们在体外培养 HEK293 细胞，利用质粒克隆使目标基因过表达，并设置空载体作为对照。"
            "实验设置生物重复与技术重复，随机分组，p 值 < 0.05 视为显著。",
            "通过 qPCR 与 Western blot 检测蛋白表达水平，测序结果已在 GenBank 登记（accession: XM_123456）。",
            "## 结果与结论",
            "实验结果表明，过表达组细胞增殖显著加快；我们在体内（in vivo）小鼠模型中验证了该趋势。"
            "我们证明了该基因调控通路的关键节点，并提供了试剂货号与菌株信息以便重复。本文局限在于样本数有限，需扩大重复次数。",
        ],
    ),
    "sample_materials.pdf": (
        "掺杂退火调控晶体薄膜带隙与电导率的材料研究",
        [
            "摘要：本研究通过掺杂与退火工艺调控晶体材料的带隙与电导率，并利用 XRD、SEM 等表征手段分析相变过程。",
            "## 材料制备与表征",
            "我们在惰性气氛下合成复合材料，经烧结与退火处理后获得致密薄膜。采用 XRD 与 SEM 表征微观结构，"
            "拉曼（Raman）光谱辅助确认晶体取向与缺陷密度。",
            "## 结构-性能关系",
            "该晶体材料经掺杂后退火，XRD 与 SEM 表征显示薄膜发生相变，带隙由 1.2 eV 降至 0.9 eV，电导率显著提升。"
            "实验结果表明，能带结构调控可优化力学性能与硬度；我们证明了退火温度对缺陷浓度有决定性影响。"
            "本文的局限在于仅报道实测值，理论模拟部分有待补充。",
        ],
    ),
    "sample_chemistry.pdf": (
        "配体促进的催化反应机理与产率优化研究",
        [
            "摘要：本文在催化条件下研究反应机理，通过 NMR 与质谱确证产物结构，产率达 85%。",
            "## 反应与催化",
            "在催化条件下，该反应以甲醇为溶剂，配体促进中间体形成，显著提高转化率与选择性。"
            "我们通过红外光谱分析官能团变化，探讨氧化还原过程。",
            "## 结构确证与机理",
            "NMR 与质谱（FTIR）确证了产物结构与分子连接关系，实验结果表明推测机理与实测一致。"
            "我们证明了配体电子效应对产率的关键作用；本文不足在于机理仍为推测，需进一步同位素标记验证。",
        ],
    ),
    "sample_social_science.pdf": (
        "基于问卷调查与双重差分的社会政策效应评估",
        [
            "摘要：本研究通过问卷调查收集样本，运用回归分析与双重差分（DID）识别社会政策的因果效应。",
            "## 数据与方法",
            "我们设计问卷收集样本数据，运用回归分析控制变量，并以工具变量处理内生性。"
            "采用量表测量关键构念，检验信度与效度；部分受访者接受质性访谈以补充数据。",
            "## 实证结果",
            "实验组与控制组的对比显示政策显著提升满意度，效应量附 95% 置信区间。"
            "双重差分（DID）估计的因果效应在 1% 水平显著（p < 0.01）；我们证明了该效应在抽样子群体中稳健。"
            "本文局限在于相关性不等于因果，需结合更多调查轮次验证。",
        ],
    ),
    "sample_engineering.pdf": (
        "面向低延迟高吞吐的分布式系统架构与鲁棒性优化",
        [
            "摘要：本文提出一种分布式系统架构，在基准测试中实现了低延迟与高吞吐，并评估其可靠性与鲁棒性。",
            "## 系统架构与协议",
            "我们设计分层架构，定义通信协议与数据流，采用控制算法优化参数以提升带宽利用率。"
            "系统在仿真环境中验证，并与既有基准（benchmark）对比。",
            "## 性能与可靠性",
            "实验结果表明，该架构在 10k 并发下延迟低于 5ms、吞吐提升 3 倍；鲁棒性测试在边界条件下保持稳定。"
            "我们证明了参数权衡（成本/性能/功耗）的可行性；本文不足在于实测仅在仿真环境，需真实传感器部署验证。",
        ],
    ),
}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, (title, paras) in PAPERS.items():
        out = os.path.abspath(os.path.join(OUT_DIR, fname))
        write_pdf(out, title, paras)
        # 校验：抽取文本确认中文可被解析（PaperMate pymupdf 管线依赖此）
        doc = pymupdf.open(out)
        txt = "".join(p.get_text() for p in doc)
        doc.close()
        assert any(ord(c) > 0x4E00 for c in txt), f"中文未嵌入: {fname}"
        print(f"OK  {fname:28s} chars={len(txt)}")
    print(f"\n生成 {len(PAPERS)} 篇样本论文 -> {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
