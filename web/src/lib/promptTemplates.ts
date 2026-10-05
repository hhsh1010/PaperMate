/**
 * 论文通用「提问 / 需求」模板（2026-09-01）
 *
 * 定位：解决「不知道该怎么形容自己的问题 / 不知道如何向 Agent 有效提问」的启动困难。
 * 与「技能命令」面板的区别：
 *   · 技能面板 = 触发 `/命令`（系统能力），面向「我知道要调用什么能力」的用户；
 *   · 本模板   = 填入一段可直接发送的自然语言问题，面向「我不知道该怎么问」的用户。
 * 因此这里放的是**问题模板**而非命令，点击后把模板原文填入输入框供用户二次编辑。
 *
 * 模板中的「这篇论文」指当前关联的文档；未关联文档时用户可自行替换措辞。
 */

export interface PromptTemplate {
  id: string;
  /** 短标题，列表里的主要可读文字 */
  label: string;
  /** 点击后填入输入框的完整问句（用户可继续编辑） */
  template: string;
}

export interface PromptTemplateGroup {
  /** 分组名，如「读懂论文」 */
  category: string;
  /** 分组标题颜色（引用 --cat-* 令牌，深浅色自动适配） */
  color: string;
  items: PromptTemplate[];
}

export const PROMPT_TEMPLATE_GROUPS: PromptTemplateGroup[] = [
  {
    category: "读懂论文",
    color: "var(--cat-blue)",
    items: [
      {
        id: "u-problem",
        label: "它在解决什么问题",
        template: "请用通俗语言说明这篇论文要解决的问题、研究动机，以及为什么这个问题重要。",
      },
      {
        id: "u-method",
        label: "把方法讲给我听",
        template: "请解释这篇论文提出的方法，尽量用比喻或具体例子说明，避免堆砌公式。",
      },
      {
        id: "u-terms",
        label: "术语与缩写",
        template: "请列出这篇论文中的专业术语和缩写，逐一解释含义，并标注它们在文中的具体所指。",
      },
      {
        id: "u-limits",
        label: "结论与局限",
        template: "这篇论文的结论是什么？作者自己承认了哪些局限，或留下了哪些未解决的问题？",
      },
    ],
  },
  {
    category: "总结梳理",
    color: "var(--cat-green)",
    items: [
      {
        id: "s-four",
        label: "四段式总结",
        template: "请用「研究问题 → 方法 → 实验 → 结论」四段式总结这篇论文，每段不超过 3 句话。",
      },
      {
        id: "s-brief",
        label: "一句话 + 摘要",
        template: "请给出这篇论文的一句话总结（30 字以内）和一段式摘要（150 字以内）。",
      },
      {
        id: "s-related",
        label: "相关工作脉络",
        template: "请梳理这篇论文与哪些工作是强相关的，并说明它们之间的关系（改进 / 对比 / 奠基 / 应用）。",
      },
      {
        id: "s-figures",
        label: "图表逐一解读",
        template: "请解释这篇论文的主要图表分别在说明什么，并指出其中最关键的那个实验结果。",
      },
    ],
  },
  {
    category: "对比评估",
    color: "var(--cat-purple)",
    items: [
      {
        id: "e-three",
        label: "三维度评价",
        template: "请从创新性、实验充分性、写作清晰度三个维度评价这篇论文，并说明你的保留意见。",
      },
      {
        id: "e-review",
        label: "模拟审稿意见",
        template: "假如你是审稿人，请给出审稿意见：主要优点、主要问题，以及你的评分建议和理由。",
      },
      {
        id: "e-claim",
        label: "结论站得住吗",
        template: "请检查这篇论文的实验是否能支撑它的结论，指出论证薄弱或过度宣称的地方。",
      },
      {
        id: "e-compare",
        label: "方法对比",
        template: "请对比这篇论文的方法与主流方法的优劣，并说明各自的适用边界。",
      },
    ],
  },
  {
    category: "方法复现",
    color: "var(--cat-amber)",
    items: [
      {
        id: "r-checklist",
        label: "复现清单",
        template: "请列出复现这篇论文所需的关键信息：数据集、模型结构、超参数、训练配置、评测指标。",
      },
      {
        id: "r-formula",
        label: "公式符号拆解",
        template: "请解释这篇论文核心公式中每个符号的含义，以及这一项在整体目标函数里的作用。",
      },
      {
        id: "r-ablation",
        label: "消融实验在看什么",
        template: "这篇论文做了哪些消融实验？每个消融分别想证明什么？",
      },
      {
        id: "r-nocode",
        label: "没有代码怎么办",
        template: "这篇论文的代码或数据是否公开？如果没有，我需要自己实现哪些部分、有哪些常见坑？",
      },
    ],
  },
  {
    category: "写作表达",
    color: "var(--cat-rose)",
    items: [
      {
        id: "w-polish",
        label: "润色为学术表达",
        template: "请把下面这段话改写成更学术、更精炼的表达：\n（在此粘贴你的文本）",
      },
      {
        id: "w-translate",
        label: "中译英（学术）",
        template: "请把下面这段中文翻译成学术英文，保持术语准确、时态规范：\n（在此粘贴你的文本）",
      },
      {
        id: "w-title",
        label: "帮我想标题",
        template: "请根据这篇论文的内容给出 5 个候选标题，并说明每个标题的侧重点。",
      },
      {
        id: "w-slides",
        label: "生成汇报提纲",
        template: "请为这篇论文生成一份 10 分钟的汇报提纲，包含每页要点和讲述时长建议。",
      },
    ],
  },
];

/** 面板顶部提示：说明模板语义与「这篇论文」指代 */
export const PROMPT_TEMPLATE_TIP =
  "不知道怎么问？点选一个模板即填入输入框，可继续编辑。模板中的「这篇论文」指当前关联的文档。";
