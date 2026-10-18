import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type {
  AppSettings,
  FigureEnrichDeps,
  FigureInstallStatus,
  InventoryOut,
  ProviderBalance,
  RagInstallStatus,
  RagModelsOut,
  SessionOut,
  StorageOut,
} from "../lib/types";
import {
  buildStorageUpdateInput,
  formatSize,
  STORAGE_SUBDIR_LABELS,
  storagePendingBanner,
} from "../lib/storageLoc";
import { useLang } from "../lib/i18n";
import { MemoryPanel } from "../components/MemoryPanel";
import { SectionCard } from "../components/SectionCard";
import { useSettings } from "../store/settings";
import { Icons } from "../lib/icons";
import { toErrorMessage } from "../lib/error";

const PROVIDERS = [
  { id: "deepseek", label: "DeepSeek", defaultBase: "https://api.deepseek.com/v1" },
  // 2026-08-25：openai 厂商整体下线（gpt 系列移除）
  { id: "qwen", label: "Qwen (DashScope)", defaultBase: "https://dashscope.aliyuncs.com/compatible-mode/v1" },
];

/** 功能开关分组（2026-08-29 收纳重排）：chat=对话与检索 / doc=文档处理 / skill=命令与技能 / exec=执行与评测 */
const FEATURE_GROUPS: { key: string; title: string; hint: string; open: boolean }[] = [
  { key: "chat", title: "对话与检索", hint: "影响聊天回复质量与检索行为", open: true },
  { key: "doc", title: "文档处理（上传增强）", hint: "上传 / 文档页相关的增强能力", open: false },
  // ★ P-06 问题 3（2026-10-10）：按**真实入口**归组 —— 改稿三件套此前被拆在两个组
  //   （revision 在"命令与技能"、para_transform 在"文档处理"、prompt_enhance 在"对话与检索"）。
  { key: "writing", title: "写作与改稿", hint: "输入框与改稿工作台的写作辅助能力", open: true },
  { key: "skill", title: "命令与技能", hint: "/ 命令触发的复合能力，关闭后回落普通对话", open: true },
  // ★ P-06 问题 5：开发者向能力明确标注，不再混在用户设置里
  { key: "exec", title: "执行与评测（开发者向）", hint: "沙箱执行与开发者评测（普通用户可忽略）", open: false },
];

// 功能命名规则（2026-09-05 统一）：用户面向名称 = 「功能名」+ 可选「（入口/形态）」。
// 入口只写用户可达的操作（/命令、按钮、所在面板）；禁止暴露内部阶段号（S*/P*，属研发沿革，见 git/实施记录）。
const FEATURES_DEF = [
  { key: "featMemory", group: "chat", label: "跨会话记忆", hint: "对话中自动记忆/召回用户偏好，并在轮末反思写入；关闭后不再注入或写入。" },
  { key: "featMultiQuery", group: "chat", label: "多查询融合检索", hint: "检索前用 LLM 改写 2–3 个子查询提升召回（fake 下自动回退单查询）。" },
  { key: "featThinking", group: "chat", label: "思考模式（展示推理链）", hint: "DeepSeek V4 系列开启后展示流式推理链；因官方 thinking 不支持 tool_choice，开启时「绑文档强制检索」自动放宽。" },
  { key: "featVerifyReply", group: "chat", label: "回复逐条核验（L2 语义）", hint: "「引用文本 ↔ 标注页码」的确定性核验（L1）自 2026-09-16 起**始终执行、零成本**；本开关只控制额外的「LLM 语义核验（L2）」：答非所问 / 引用是否支撑结论 / 页码是否明显错误，超时（8s）自动降级。只标「存疑」不自动改写回复。" },
  { key: "featReflectOnAbort", group: "chat", label: "失败轮次自评", hint: "只在「迭代耗尽 / 连续无进展」这类**失败轮次**触发：先做一次自评（限 max_tokens 400、超时 15s；超时或异常一律沿用原兜底内容），若判定「需改进」则带该建议重新收尾一次（仍不给工具，保证收敛）。**正常收敛的对话零额外调用**；自评超时或异常一律沿用原兜底内容，不影响回答。" },
  { key: "featPromptEnhance", group: "writing", label: "提示词增强", hint: "输入框「增强」按钮：把口语需求改写为更明确的任务描述（单轮元任务，按会话/全局当前模型调用）。关闭后接口返回 403（API 保留）；fake 模型下自动提示不增强。" },
  { key: "featSummary", group: "chat", label: "一键总结", hint: "对话页右栏文档面板的「结构化总结 / 相关工作 / 对比」入口；关闭后接口返回 403（API 保留）。" },
  { key: "featPaperSearch", group: "doc", label: "论文在线检索（arXiv）", hint: "文献页右上「检索论文」入口：检索 arXiv 元数据并导入解析（只拿元数据，导入才下载全文）。**默认关闭** —— 这是主动联网能力。若同时设了环境变量 PAPERMATE_ENABLE_PAPER_SEARCH，则需两者都关才真正关闭。" },
  { key: "featDiagramEnrich", group: "doc", label: "结构图识别（多模态）", hint: "上传 PDF 时用 DeepSeek-V4-Flash-Vision 把矢量结构图（架构图/流程图/框图）还原为 Mermaid 源码 + 一句话描述；需先在「模型连接」配置并测试 DeepSeek 密钥，无 key/失败自动降级为空，不影响上传。" },
  { key: "featParaTransform", group: "writing", label: "段落 AI 改写（改稿工作台）", hint: "改稿工作台选中段落后的扩写/缩写/润色/去重：单轮 LLM 改写生成候选文本，采用即写入文档（track-changes + 自动备份，可撤销）；fake 模型下返回原文不臆造。关闭后接口返回 403（API 保留）。" },
  { key: "featSkills", group: "skill", label: "技能总开关", hint: "所有命令技能的统一门控；关闭后 / 命令一律回落普通对话。" },
  { key: "featPlanning", group: "skill", label: "任务规划（/规划）", hint: "/规划 命令将目标分解为多步计划并逐子任务检索取证；fake 下规则分解。" },
  { key: "featDeepRead", group: "skill", label: "单篇精读（/精读）", hint: "对唯一关联的文档多轮检索后生成结构化精读报告（研究问题/方法/实验/结果/创新/局限）；无 key 时退化为分主题摘录+页码。关闭后 /精读 命令回落普通对话。" },
  { key: "featGlossary", group: "skill", label: "术语表（/术语表）", hint: "从关联文档抽取缩写/术语候选并生成术语表（含释义与首次出现页）；无 key 时退化为候选术语+页码表。关闭后 /术语表 命令回落普通对话。" },
  { key: "featFindCode", group: "skill", label: "找代码（/找代码）", hint: "从关联文档抽取算法/公式/伪代码/框架/仓库等代码线索并生成实现指南（论文描述/需查 repo/需自行实现 三态）；无 key 时退化为线索分组+页码。关闭后 /找代码 命令回落普通对话。" },
  { key: "featResearchGap", group: "skill", label: "研究缺口（/研究缺口）", hint: "从关联文档抽取局限/未来工作/未解决问题并生成研究缺口分析；无 key 时退化为线索分组+页码。关闭后 /研究缺口 命令回落普通对话。" },
  { key: "featExperimentDesign", group: "skill", label: "实验设计（/实验设计）", hint: "从关联文档抽取数据集/指标/基线/协议等实验设置并生成可复现实验方案；无 key 时退化为设置分组+页码。关闭后 /实验设计 命令回落普通对话。" },
  { key: "featReview", group: "skill", label: "审稿（/审稿）", hint: "从关联文档抽取贡献/方法/局限线索并生成结构化审稿意见（含推荐评分）；无 key 时退化为线索分组+页码。关闭后 /审稿 命令回落普通对话。" },
  { key: "featReproduction", group: "skill", label: "复现辅助（/复现）", hint: "从关联文档抽取超参/配置并生成三态复现清单（论文给出/待核实/需从代码补全）；无 key 时退化为候选分组+页码。关闭后 /复现 命令回落普通对话。" },
  { key: "featRevision", group: "writing", label: "改稿（/改稿）", hint: "把会话关联文档分为「你的论文（草稿）」与「参考文献」：用 /改稿 参数（文件名或文档N）指定草稿，逐节诊断并给出修改建议（原文摘录→问题→建议→参考依据）+ 整体问题清单；无 key 时退化为确定性段落对照。" },
  { key: "featCodeInterpreter", group: "exec", label: "代码解释器（沙箱）", hint: "强制 Docker 隔离执行，默认关闭；无 Docker 时明确报错、不会本机直跑。" },
  { key: "featDevEval", group: "exec", label: "开发者评测（/eval）", hint: "/eval 命令读取评测报告与近期 Trace 出指标卡；默认关闭。" },
];

/**
 * 设置页（2026-08-29 收纳重排）：
 * - 单列纵向流 + 可折叠分区（SectionCard）：全部分区默认展开（2026-09-03 用户要求），点击分组名可收起；
 * - 功能开关按域分组（对话与检索 / 文档处理 / 命令与技能 / 执行与评测），details 折叠 + 启用计数徽标；
 * - 会话覆盖上移至高频区（规则之后），信息展示类（数据/安全/余额）沉底；
 * - 底部吸附保存栏（含保存结果 toast），任何滚动位置都能一键保存。
 */
export function SettingsPage() {
  const { lang, setLang, t } = useLang();
  // ---- 全局设置（共享上下文：跨页面一致 + localStorage 缓存）----
  const { settings, patch } = useSettings();
  const [defaultModel, setDefaultModel] = useState("fake");
  const [activeProvider, setActiveProvider] = useState<string>(PROVIDERS[0].id);
  const [keys, setKeys] = useState<Record<string, string>>({ deepseek: "", qwen: "" });
  const [baseUrls, setBaseUrls] = useState<Record<string, string>>({ deepseek: "", qwen: "" });
  const [models, setModels] = useState<Record<string, string>>({ deepseek: "", qwen: "" });
  const [globalMsg, setGlobalMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [testRes, setTestRes] = useState<Record<string, { ok: boolean; msg: string }>>({});
  const [testing, setTesting] = useState<Record<string, boolean>>({});
  const [webSearchRes, setWebSearchRes] = useState<{ ok: boolean; msg: string } | null>(null);
  const [webTesting, setWebTesting] = useState(false);
  const [agentRules, setAgentRules] = useState("");
  // token 上限（2026-09-17 拆成两项，语义不同）：
  //   tokenBudget    = 单次请求上下文上限（单轮 prompt 超限即中止；也是历史压缩阈值的来源）
  //   tokenPerTurn   = 整轮累计上限（含所有迭代；防跑飞的粗护栏）
  // 背景：此前一个数当两个用，30000 只够 2-3 轮迭代（实测单轮 prompt 已达 24883）→ 正常会话误伤。
  const [tokenBudget, setTokenBudget] = useState(30000);
  const [tokenPerTurn, setTokenPerTurn] = useState(150000);
  // 记忆向量 引擎（B 档）：hash=轻量哈希（默认）/ bge-m3=真实语义向量
  const [memoryEmbedder, setMemoryEmbedder] = useState("hash");
  // RAG 检索底座（2026-09-01 设置页可切换）：embedding 模式 + 两枚开关
  const [ragEmbedder, setRagEmbedder] = useState("fake");
  const [useBgeSparse, setUseBgeSparse] = useState(false);
  const [useJieba, setUseJieba] = useState(false);
  // T-03 重排推理加速（2026-09-24）：默认值须与后端 runtime_settings 默认一致（512 / 开）
  // 512 为 2026-10-05 质量优先实测档位（原 384 截断约 35% 候选、属欠拟合档）
  const [rerankFp16, setRerankFp16] = useState(true);
  const [rerankMaxLen, setRerankMaxLen] = useState(512);
  type ReindexPhase = "idle" | "running" | "done" | "cancelled" | "error";
  const [reindexState, setReindexState] = useState<{
    phase: ReindexPhase;
    done: number;
    total: number;
    lastDoc: number | null;
    reindexed: number;
    failed: { id: number; error: string }[];
    error: string | null;
  }>({ phase: "idle", done: 0, total: 0, lastDoc: null, reindexed: 0, failed: [], error: null });
  // 取消控制器：取消时先调 api.cancelReindex()（后端在文档边界停止），
  // 再 abort 掉 SSE 连接（前端立即断开）。仅 running 期间有效。
  const reindexAbortRef = useRef<AbortController | null>(null);
  // 重建索引代际 token（2026-09-09 修复 L1）：每次启动 runReindex 递增，旧流的所有
  // 异步收尾（事件回调 / 兜底 / catch）写入状态前先比对代际，过期一律丢弃——
  // 防止「取消并重启」时旧流的 catch 覆盖新流的 running/done 状态。
  const reindexGenRef = useRef(0);
  // 2026-09-06 修复：设置页为模态渲染、关闭即卸载——卸载时中断在飞的重建索引
  // （先通知后端再 abort，与手动取消同口径）。此前 SSE 与后端任务悬空继续跑，
  // 且控制器引用随卸载丢失，用户无法再取消。
  useEffect(
    () => () => {
      reindexAbortRef.current?.abort();
      reindexAbortRef.current = null;
      void api.cancelReindex();
    },
    [],
  );
  const [balances, setBalances] = useState<Record<string, ProviderBalance> | null>(null);
  const [inventory, setInventory] = useState<InventoryOut | null>(null);
  const [invError, setInvError] = useState<string | null>(null);
  // ── 存储位置（2026-10-07）──
  const [storage, setStorage] = useState<StorageOut | null>(null);
  const [storageErr, setStorageErr] = useState<string | null>(null);
  const [storageMsg, setStorageMsg] = useState<string | null>(null);
  const [storageRoot, setStorageRoot] = useState("");
  const [storageMigrate, setStorageMigrate] = useState(true);
  const [storageSaving, setStorageSaving] = useState(false);

  // ---- 真实语义检索模型的**显式安装**（2026-10-05 桌面客户端）----
  // 语义：App 装完默认 fake（零下载）⇒ 用户在此**主动**下载 ⇒ 下载完仍需手动把
  //      「嵌入模式」切到 real 并**重建索引**才生效（这里只管下载，不偷偷改任何配置）。
  const [ragModels, setRagModels] = useState<RagModelsOut | null>(null);
  const [ragModelsErr, setRagModelsErr] = useState<string | null>(null);
  const [ragInstall, setRagInstall] = useState<RagInstallStatus | null>(null);

  /* ---- 大弹窗内的左侧导航（WorkBuddy 风格：左清单 + 右内容）----
     默认展示第 0 项「模型与厂商」；点击左侧类别时右侧仅展示对应内容（其余不展示）。 */
  const sectionsRef = useRef<HTMLDivElement>(null);
  const [activeSec, setActiveSec] = useState(0);
  const SETTING_SECTIONS: { label: string; Icon: (p: { size?: number }) => React.ReactElement }[] = [
    { label: t("settings.models"), Icon: Icons.cpu },
    { label: t("settings.features"), Icon: Icons.layers },
    { label: t("settings.rules"), Icon: Icons.terminal },
    { label: "会话覆盖", Icon: Icons.chat },
    { label: t("settings.data"), Icon: Icons.file },
    { label: t("settings.security"), Icon: Icons.lock },
    { label: t("settings.balances"), Icon: Icons.chart },
    { label: "长期记忆", Icon: Icons.file },
    { label: "检索增强", Icon: Icons.search },
  ];

  // ---- 单会话覆盖 ----
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [model, setModel] = useState("fake");
  const [dedupeWarning, setDedupeWarning] = useState(true);
  const [featMemory, setFeatMemory] = useState(true);
  const [featSummary, setFeatSummary] = useState(true);
  const [featMultiQuery, setFeatMultiQuery] = useState(false);
  const [featCodeInterpreter, setFeatCodeInterpreter] = useState(false);
  const [featSkills, setFeatSkills] = useState(true);
  const [featPlanning, setFeatPlanning] = useState(true);
  const [featDevEval, setFeatDevEval] = useState(false);
  const [featThinking, setFeatThinking] = useState(false);
  const [featPaperSearch, setFeatPaperSearch] = useState(false);
  const [featVerifyReply, setFeatVerifyReply] = useState(false);
  const [featReflectOnAbort, setFeatReflectOnAbort] = useState(false);
  const [featPromptEnhance, setFeatPromptEnhance] = useState(true);
  const [featParaTransform, setFeatParaTransform] = useState(true);
  const [featDiagramEnrich, setFeatDiagramEnrich] = useState(false);
  const [featDeepRead, setFeatDeepRead] = useState(true);
  const [featGlossary, setFeatGlossary] = useState(true);
  const [featFindCode, setFeatFindCode] = useState(true);
  const [featResearchGap, setFeatResearchGap] = useState(true);
  const [featExperimentDesign, setFeatExperimentDesign] = useState(true);
  const [featReview, setFeatReview] = useState(true);
  const [featReproduction, setFeatReproduction] = useState(true);
  const [featRevision, setFeatRevision] = useState(true);
  const [featFigureEnrich, setFeatFigureEnrich] = useState(false);
  const [enrichDeps, setEnrichDeps] = useState<FigureEnrichDeps | null>(null);
  /** 图片增强依赖安装（2026-10-09）：`[figure]` extra 的按需下载（paddleocr / 表格结构） */
  const [figInstall, setFigInstall] = useState<FigureInstallStatus | null>(null);
  /** ★ 重新检测的进行态（2026-10-10）：原先点了毫无可见反馈，被用户当成"没用" */
  const [depsChecking, setDepsChecking] = useState(false);

  /** 实时重探测三能力（不做缓存；装完后调用即可拿到真值，无需重启 App）。 */
  const refreshEnrichDeps = async () => {
    try {
      setEnrichDeps(await api.figureEnrichDeps());
    } catch {
      /* 单次失败保持旧值，不打扰 */
    }
  };

  /** ★ 重新检测（2026-10-10）：带「检测中…」进行态 —— 原先点了零可见反馈，被用户当成"没用"。
   *  反馈 = ①按钮自身变「检测中…」禁用 ②状态行尾追加「检测中…」③完成后状态行刷新为真值。 */
  const recheckDeps = async () => {
    setDepsChecking(true);
    try {
      await refreshEnrichDeps();
    } finally {
      setDepsChecking(false);
    }
  };

  // 安装进行中：每 2s 轮询（与 RAG 安装同款模式）。**结束即重探测**——
  // 探测是"现场 import"，装完立刻能看到"已装"，不必重启 App（这是 P-06 修复的一部分）。
  useEffect(() => {
    if (figInstall?.status !== "running") return;
    const timer = window.setInterval(async () => {
      try {
        const s = await api.figureDepsStatus();
        setFigInstall(s);
        if (s.status !== "running") setEnrichDeps(s.deps ?? (await api.figureEnrichDeps()));
      } catch {
        /* 轮询的单次失败不打扰用户，下一拍自动重试 */
      }
    }, 2000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [figInstall?.status]);

  const startFigInstall = async () => {
    try {
      setFigInstall(await api.installFigureDeps());
    } catch (e) {
      setFigInstall({
        status: "error",
        message: "启动安装失败",
        error: String(e),
        started_at: null,
        finished_at: null,
        elapsed_s: 0,
        deps: null,
      });
    }
  };
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  const allModels = useMemo(() => {
    if (!settings) return ["fake"];
    const list = ["fake"];
    for (const p of Object.keys(settings.providers)) {
      for (const m of settings.providers[p].available_models || []) list.push(m);
    }
    return list;
  }, [settings]);

  const normModel = (v: string, st?: AppSettings | null): string => {
    const provs = (st?.providers ?? settings?.providers ?? {}) as Record<string, { available_models?: string[] }>;
    const list = ["fake"];
    for (const p of Object.keys(provs)) {
      for (const m of provs[p].available_models || []) list.push(m);
    }
    if (list.includes(v)) return v;
    const am = provs[v]?.available_models;
    if (am && am.length) return am[0];
    return "fake";
  };

  // 后端设置变更时（首次加载 / 保存后 / 跨页面共享更新）同步表单态。
  // 关键修复：normModel 显式传入 settings，避免读取尚未更新的闭包 state 而恒返 "fake"。
  useEffect(() => {
    if (!settings) return;
    setDefaultModel(normModel(settings.default_model, settings));
    setDedupeWarning(settings.dedupe_warning !== false);
    setFeatMemory(settings.features?.memory !== false);
    setFeatSummary(settings.features?.summary !== false);
    setFeatMultiQuery(settings.features?.multi_query === true);
    setFeatCodeInterpreter(settings.features?.code_interpreter === true);
    setFeatSkills(settings.features?.skills !== false);
    setFeatPlanning(settings.features?.planning !== false);
    setFeatDevEval(settings.features?.dev_eval === true);
    setFeatThinking(settings.features?.thinking === true);
    setFeatPaperSearch(settings.features?.paper_search === true);
    setFeatVerifyReply(settings.features?.verify_reply === true);
    setFeatReflectOnAbort(settings.features?.reflect_on_abort === true);
    setFeatDiagramEnrich(settings.features?.diagram_enrich === true);
    setFeatDeepRead(settings.features?.deep_read !== false);
    setFeatGlossary(settings.features?.glossary !== false);
    setFeatFindCode(settings.features?.find_code !== false);
    setFeatResearchGap(settings.features?.research_gap !== false);
    setFeatExperimentDesign(settings.features?.experiment_design !== false);
    setFeatReview(settings.features?.review !== false);
    setFeatPromptEnhance(settings.features?.prompt_enhance !== false);
    setFeatParaTransform(settings.features?.para_transform !== false);
    setFeatReproduction(settings.features?.reproduction !== false);
    setFeatRevision(settings.features?.revision !== false);
    setFeatFigureEnrich(settings.figure_enrich?.enabled === true);
    setEnrichDeps(settings.figure_enrich_deps ?? null);
    setAgentRules(settings.agent_rules ?? "");
    setTokenBudget(settings.max_tokens_budget ?? 30000);
    setTokenPerTurn(settings.max_tokens_per_turn ?? 150000);
    setMemoryEmbedder(settings.memory_embedder ?? "hash");
    setRagEmbedder(settings.rag_embedder ?? "fake");
    setUseBgeSparse(settings.use_bge_sparse === true);
    setUseJieba(settings.use_jieba === true);
    // T-03：回填重排加速档位（缺省与后端默认对齐：fp16 开 / 512）
    // 512 为 2026-10-05 质量优先实测得出的档位（原 384 属欠拟合档，截断约 35% 候选）
    setRerankFp16(settings.rag_reranker_fp16 !== false);
    setRerankMaxLen(settings.rag_reranker_max_length ?? 512);
    setBaseUrls({
      deepseek: settings.providers.deepseek?.base_url ?? "",
      qwen: settings.providers.qwen?.base_url ?? "",
    });
    setModels({
      deepseek: settings.providers.deepseek?.model ?? "",
      qwen: settings.providers.qwen?.model ?? "",
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings]);

  const refreshSessions = async () => {
    const list = await api.listSessions();
    setSessions(list);
    if (sessionId == null && list[0]) setSessionId(list[0].id);
  };

  const refreshBalances = async () => {
    try {
      setBalances(await api.getBalances());
    } catch {
      setBalances(null);
    }
  };

  const refreshInventory = async () => {
    try {
      setInventory(await api.getInventory());
      setInvError(null);
    } catch (e: unknown) {
      setInvError(toErrorMessage(e) || "加载失败");
    }
  };

  /** ★ 存储位置：只**读**当前位置；改位置走下面的保存按钮（写"待迁移请求"）。 */
  const refreshStorage = async () => {
    try {
      const s = await api.getStorage();
      setStorage(s);
      setStorageRoot((prev) => prev || s.data_root);
      setStorageErr(null);
    } catch (e: unknown) {
      setStorageErr(toErrorMessage(e) || "加载失败");
    }
  };

  const saveStorage = async () => {
    const built = buildStorageUpdateInput(storageRoot, storageMigrate);
    if (!built.ok || !built.value) {
      setStorageErr(built.error ?? "路径无效");
      return;
    }
    setStorageSaving(true);
    setStorageMsg(null);
    try {
      const r = await api.updateStorage(built.value);
      setStorageErr(null);
      setStorageMsg(r.warnings.join("\n"));
      await refreshStorage();
    } catch (e: unknown) {
      setStorageErr(toErrorMessage(e) || "保存失败");
    } finally {
      setStorageSaving(false);
    }
  };

  const refreshRagModels = async () => {
    try {
      const [m, s] = await Promise.all([api.getRagModels(), api.ragInstallStatus()]);
      setRagModels(m);
      setRagInstall(s);
      setRagModelsErr(null);
    } catch (e: unknown) {
      setRagModelsErr(toErrorMessage(e) || (lang === "zh" ? "加载失败" : "Load failed"));
    }
  };

  // 仅在进入「检索增强」分区时拉取（避免每次打开设置页都白打两个请求）
  useEffect(() => {
    if (activeSec === 8) {
      void refreshRagModels();
      // ★ 图片增强依赖也归入本分区 ⇒ 进入时实时重探测（探测是现场 import，装完即真值）
      void refreshEnrichDeps();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSec]);

  // 下载进行中：每 2s 轮询。**不用 SSE** —— 下载是后台线程的一次性长任务，
  // 断连重连语义比轮询复杂，而这里 2s 粒度完全够（单个权重文件要下好几分钟）。
  useEffect(() => {
    if (ragInstall?.status !== "running") return;
    const timer = window.setInterval(async () => {
      try {
        const s = await api.ragInstallStatus();
        setRagInstall(s);
        if (s.status !== "running") await refreshRagModels();
      } catch {
        /* 轮询的单次失败不打扰用户，下一拍自动重试 */
      }
    }, 2000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ragInstall?.status]);

  const startRagInstall = async (keys?: string[]) => {
    try {
      setRagInstall(await api.installRagModels(keys));
    } catch (e: unknown) {
      setGlobalMsg({ kind: "err", text: toErrorMessage(e) || (lang === "zh" ? "启动下载失败" : "Failed to start") });
    }
  };

  const cancelRagInstall = async () => {
    try {
      setRagInstall(await api.cancelRagInstall());
    } catch (e: unknown) {
      setGlobalMsg({ kind: "err", text: toErrorMessage(e) || (lang === "zh" ? "取消失败" : "Cancel failed") });
    }
  };

  useEffect(() => {
    refreshSessions();
    refreshBalances();
    refreshInventory();
    refreshStorage();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // L6 修复（2026-09-10）：**去掉 settings 依赖**——原依赖链使保存全局设置后（store 更新）
  // 本 effect 重跑，把用户已改未保存的会话模型选择静默重置回配置默认值。
  // 本 effect 的语义是「切会话 / 会话列表更新时同步该会话的模型」，与全局 settings 无关；
  // normModel 需要的 settings 通过 ref 取最新值，不参与依赖。
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  useEffect(() => {
    const s = sessions.find((x) => x.id === sessionId);
    if (s) {
      setModel(normModel((s.config.model as string) || "fake", settingsRef.current));
    }
  }, [sessionId, sessions]);

  const [saving, setSaving] = useState(false);

  const saveGlobal = async () => {
    // 2026-09-06 修复：防重——此前保存按钮无 disabled/saving 态，连点会重复 PUT /settings（含密钥字段）。
    if (saving) return;
    // 2026-09-09 修复 L2：settings 未从后端加载完成时，baseUrls/models 还是空串初始值
    // （:209-216 的回填 effect 未执行）——此时保存会把 `base_url:""`、`model:""` 提交上去，
    // 覆盖后端已存配置（打开设置页后快速保存即触发）。门控：未加载不允许保存。
    if (!settings) {
      setMsg({ kind: "err", text: lang === "zh" ? "设置尚未加载完成，请稍候再试" : "Settings not loaded yet" });
      return;
    }
    setSaving(true);
    try {
      const providers: Record<string, { api_key?: string; base_url?: string; model?: string }> = {};
      for (const p of PROVIDERS) {
        const body: { api_key?: string; base_url?: string; model?: string } = {};
        if (keys[p.id]) body.api_key = keys[p.id];
        if (baseUrls[p.id] !== undefined) body.base_url = baseUrls[p.id];
        if (models[p.id] !== undefined) body.model = models[p.id];
        if (Object.keys(body).length) providers[p.id] = body;
      }
      const req: {
        default_model?: string;
        dedupe_warning?: boolean;
        agent_rules?: string;
        /** 单次请求上下文上限（≥1000 的整数才被后端接受） */
        max_tokens_budget?: number;
        /** 整轮累计 token 上限（≥1000 的整数才被后端接受） */
        max_tokens_per_turn?: number;
        features?: Record<string, boolean>;
        figure_enrich?: { enabled: boolean; methods?: string[] };
        memory_embedder?: string;
        default_discipline?: string;
        /** RAG 检索底座（2026-09-01）：embedding 模式 + 稀疏/分词开关 */
        rag_embedder?: string;
        use_bge_sparse?: boolean;
        use_jieba?: boolean;
        /** T-03：重排加速档位（fp16 / 截断长度，0=不限长） */
        rag_reranker_fp16?: boolean;
        rag_reranker_max_length?: number;
        providers?: typeof providers;
      } = {};
      if (defaultModel) req.default_model = defaultModel;
      req.dedupe_warning = dedupeWarning;
      req.agent_rules = agentRules;
      // 只在合法范围内提交，避免无效值覆盖（后端同样有 ≥1000 校验，双保险）
      if (Number.isFinite(tokenBudget) && tokenBudget >= 1000) {
        req.max_tokens_budget = Math.floor(tokenBudget);
      }
      if (Number.isFinite(tokenPerTurn) && tokenPerTurn >= 1000) {
        req.max_tokens_per_turn = Math.floor(tokenPerTurn);
      }
      req.features = {
        memory: featMemory,
        summary: featSummary,
        multi_query: featMultiQuery,
        code_interpreter: featCodeInterpreter,
        skills: featSkills,
        planning: featPlanning,
        dev_eval: featDevEval,
        thinking: featThinking,
        paper_search: featPaperSearch,
        verify_reply: featVerifyReply,
        reflect_on_abort: featReflectOnAbort,
        prompt_enhance: featPromptEnhance,
        para_transform: featParaTransform,
        diagram_enrich: featDiagramEnrich,
        deep_read: featDeepRead,
        glossary: featGlossary,
        find_code: featFindCode,
        research_gap: featResearchGap,
        experiment_design: featExperimentDesign,
        review: featReview,
        reproduction: featReproduction,
        revision: featRevision,
      };
      // 图片内容增强：预置全部方法，enabled 决定上传期是否执行（缺依赖自动降级）
      req.figure_enrich = { enabled: featFigureEnrich, methods: ["ocr", "table", "vlm"] };
      req.memory_embedder = memoryEmbedder;
      // RAG 检索底座：embedding 模式 + 稀疏/分词开关（保存后需点「重建索引」生效）
      req.rag_embedder = ragEmbedder;
      req.use_bge_sparse = useBgeSparse;
      req.use_jieba = useJieba;
      // T-03 重排加速：改完即时生效（后端会连管线缓存一起清），无需「重建索引」
      req.rag_reranker_fp16 = rerankFp16;
      req.rag_reranker_max_length = rerankMaxLen;
      if (Object.keys(providers).length) req.providers = providers;

      await patch(req);
      setKeys({ deepseek: "", qwen: "" });
      setGlobalMsg({ kind: "ok", text: lang === "zh" ? "已保存（密钥仅存于后端本地文件）" : "Saved (keys stay local)" });
      refreshBalances();
    } catch (e: unknown) {
      setGlobalMsg({ kind: "err", text: toErrorMessage(e) || "保存失败" });
    } finally {
      setSaving(false);
    }
  };

  const testProvider = async (id: string) => {
    setTesting((t) => ({ ...t, [id]: true }));
    try {
      const r = await api.testSettings(id);
      setTestRes((prev) => ({
        ...prev,
        [id]: {
          ok: r.ok,
          msg: r.ok
            ? `✅ 连接成功 · 模型 ${r.model}${r.reply ? `（${r.reply.slice(0, 40)}）` : ""}`
            : `❌ 连接失败：${r.error ?? "未知错误"}`,
        },
      }));
    } catch (e: unknown) {
      setTestRes((prev) => ({ ...prev, [id]: { ok: false, msg: `❌ 测试异常：${toErrorMessage(e) || "未知错误"}` } }));
    } finally {
      setTesting((t) => ({ ...t, [id]: false }));
    }
  };

  const testWebSearchProvider = async () => {
    setWebTesting(true);
    setWebSearchRes(null);
    try {
      const r = await api.testWebSearch(activeProvider);
      setWebSearchRes({
        ok: r.ok,
        msg: r.ok
          ? `✅ 大模型联网可用 · ${r.model ?? ""}：${(r.reply ?? "").slice(0, 48)}`
          : r.supported
            ? `❌ 实测失败：${r.error ?? "未知错误"}`
            : `⚠️ 当前厂商/模型不支持原生联网搜索：${r.error ?? ""}（请到「默认模型」切换为 Qwen 后再开启联网）`,
      });
    } catch (e: unknown) {
      setWebSearchRes({ ok: false, msg: `❌ 测试异常：${toErrorMessage(e) || "未知错误"}` });
    } finally {
      setWebTesting(false);
    }
  };

  const applySession = async () => {
    if (sessionId == null) return;
    try {
      await api.updateSession(sessionId, { model });
      setMsg({ kind: "ok", text: lang === "zh" ? "已保存" : "Saved" });
    } catch (e: unknown) {
      setMsg({ kind: "err", text: toErrorMessage(e) || "保存失败" });
    }
  };

  // 切换 embedding 模式 / 稀疏 / 分词后，必须重建索引才生效（get_pipeline 按 doc_id 缓存，不感知配置变化）
  const runReindex = async () => {
    // 2026-09-09 修复 L1：代际 token。旧实现用 `await Promise.resolve()` 想等旧流的
    // catch 先完成——但 fetch 的 reject 要经过多层微任务/宏任务，一帧根本等不到；
    // 旧 catch 会在新流 setReindexState(running) 甚至 done **之后**才执行，
    // 把新流状态覆盖为 cancelled/error（进度冻结或误报失败）。
    // 现在每次启动递增代际，本函数内所有状态写入先比对代际，过期一律丢弃。
    const gen = ++reindexGenRef.current;
    // 已有进行中的重建被点击时，视为「取消并重新开始」
    if (reindexAbortRef.current) {
      reindexAbortRef.current.abort();
      reindexAbortRef.current = null;
    }
    const ctrl = new AbortController();
    reindexAbortRef.current = ctrl;
    setReindexState({ phase: "running", done: 0, total: 0, lastDoc: null, reindexed: 0, failed: [], error: null });
    try {
      const done = await api.reindexAllStream((ev) => {
        if (gen !== reindexGenRef.current) return; // 已被更新的重建取代，丢弃过期事件（L1）
        if (ev.type === "progress") {
          setReindexState((s) => ({
            ...s,
            phase: "running",
            done: ev.done ?? s.done,
            total: ev.total ?? s.total,
            lastDoc: ev.doc_id ?? s.lastDoc,
          }));
        } else if (ev.type === "done") {
          // 后端异常路径也会产出 done 事件并携带 error（如文档列表读取失败、流中断），
          // 此时必须进 error 态，否则会显示成「完成：成功 0 / 总计 0」这种误导性成功。
          setReindexState((s) => ({
            ...s,
            phase: ev.error ? "error" : "done",
            reindexed: ev.reindexed ?? 0,
            total: ev.total ?? s.total,
            done: ev.total ?? s.done,
            failed: ev.failed ?? [],
            error: ev.error ?? null,
          }));
        } else if (ev.type === "cancelled") {
          setReindexState((s) => ({
            ...s,
            phase: "cancelled",
            reindexed: ev.reindexed ?? 0,
            total: ev.total ?? s.total,
            done: ev.done ?? s.done,
            failed: ev.failed ?? [],
          }));
        }
      }, ctrl.signal);
      // 兜底：若流未携带 done/cancelled 事件（异常截断）
      if (gen !== reindexGenRef.current) return; // 旧流的兜底，不得覆盖新流状态（L1）
      setReindexState((s) =>
        s.phase === "running"
          ? { ...s, phase: "done", reindexed: done.reindexed, total: done.total ?? s.total, failed: done.failed ?? [] }
          : s,
      );
    } catch (e: unknown) {
      if (gen !== reindexGenRef.current) return; // 旧流的收尾，不得覆盖新流状态（L1 核心修复点）
      // 用户主动取消：AbortError 不当作错误，显示「已取消」
      const isAbort = (e instanceof Error && e.name === "AbortError") || /aborted/i.test(String(toErrorMessage(e)));
      setReindexState((s) => ({
        ...s,
        phase: isAbort ? "cancelled" : "error",
        error: isAbort ? null : (toErrorMessage(e) || (lang === "zh" ? "重建失败" : "Reindex failed")),
      }));
    } finally {
      if (reindexAbortRef.current === ctrl) reindexAbortRef.current = null;
    }
  };

  // 用户点击「取消」：先通知后端停止（文档边界生效），再断开 SSE。
  // 顺序很重要——只 abort 会让后端仍在跑；只通知不 abort 会让前端卡在「进行中」等流结束。
  const cancelReindex = () => {
    const ctrl = reindexAbortRef.current;
    if (!ctrl) return;
    void api.cancelReindex().catch(() => {/* 即便后端不可达也继续 abort */});
    ctrl.abort();
  };

  const activePub = settings?.providers[activeProvider];
  const activeMeta = PROVIDERS.find((p) => p.id === activeProvider)!;
  const featMap = {
    featMemory, featSummary, featMultiQuery, featCodeInterpreter,
    featSkills, featPlanning, featDevEval, featThinking, featVerifyReply, featDiagramEnrich,
    featReflectOnAbort,
    featPromptEnhance, featParaTransform,
    featDeepRead, featGlossary, featFindCode, featResearchGap, featExperimentDesign, featReview,
    featReproduction, featRevision,
  };
  const featSetters: Record<string, (v: boolean) => void> = {
    featMemory: setFeatMemory, featSummary: setFeatSummary, featMultiQuery: setFeatMultiQuery,
    featCodeInterpreter: setFeatCodeInterpreter, featSkills: setFeatSkills,
    featPlanning: setFeatPlanning, featDevEval: setFeatDevEval, featThinking: setFeatThinking,
      featPaperSearch: setFeatPaperSearch,
    featVerifyReply: setFeatVerifyReply, featDiagramEnrich: setFeatDiagramEnrich,
    featReflectOnAbort: setFeatReflectOnAbort,
    featPromptEnhance: setFeatPromptEnhance, featParaTransform: setFeatParaTransform,
    featDeepRead: setFeatDeepRead, featGlossary: setFeatGlossary,
    featFindCode: setFeatFindCode, featResearchGap: setFeatResearchGap,
    featExperimentDesign: setFeatExperimentDesign, featReview: setFeatReview,
    featReproduction: setFeatReproduction, featRevision: setFeatRevision,
  };

  return (
    <div className="page settings">
      <header className="page-header settings-header">
        <div>
          <h2>{t("settings.title")}</h2>
          <p className="muted">{t("settings.subtitle")}</p>
        </div>
        <label className="settings-field lang-switch">
          <span className="settings-label">{t("settings.lang")}</span>
          <select
            className="settings-control"
            value={lang}
            onChange={(e) => setLang(e.target.value as "zh" | "en")}
          >
            <option value="zh">中文</option>
            <option value="en">English</option>
          </select>
        </label>
      </header>

      <nav className="wb-settings-nav" aria-label="设置分类">
        {SETTING_SECTIONS.map((s, i) => (
          <button
            key={s.label}
            type="button"
            className={activeSec === i ? "active" : ""}
            onClick={() => setActiveSec(i)}
            aria-current={activeSec === i ? "true" : undefined}
          >
            <s.Icon size={16} />
            <span>{s.label}</span>
          </button>
        ))}
      </nav>

      <div className="settings-sections" ref={sectionsRef}>
        {/* ============ 模型与厂商（默认展示） ============ */}
        {activeSec === 0 && (
        <SectionCard
          title={t("settings.models")}
          subtitle={lang === "zh" ? "新会话默认使用此处配置；Key 仅存于后端本地文件，不联网上传。" : "Default for new sessions; keys stay local."}
          badge={settings?.providers.deepseek?.configured || settings?.providers.qwen?.configured ? "已配置" : undefined}
        >
            <label className="settings-field">
              <span className="settings-label">{t("settings.defaultModel")}</span>
              <select className="settings-control" value={defaultModel} onChange={(e) => setDefaultModel(e.target.value)}>
                {allModels.map((m) => (
                  <option key={m} value={m}>{m === "fake" ? "fake（本地演示）" : m}</option>
                ))}
              </select>
            </label>

            <div className="settings-tabs" role="tablist" aria-label="厂商">
              {PROVIDERS.map((p) => {
                const pub = settings?.providers[p.id];
                return (
                  <button
                    key={p.id}
                    role="tab"
                    aria-selected={p.id === activeProvider}
                    className={`settings-tab ${p.id === activeProvider ? "active" : ""}`}
                    onClick={() => setActiveProvider(p.id)}
                  >
                    <span className="tab-name">{p.label}</span>
                    <span className={`tab-dot ${pub?.configured ? "ok" : "off"}`} title={pub?.configured ? "已配置" : "未配置"} />
                  </button>
                );
              })}
            </div>

            <div className="settings-panel" role="tabpanel">
              <div className="panel-head">
                <span className={`panel-status ${activePub?.configured ? "ok" : "off"}`}>
                  {activePub?.configured ? `✓ 已配置 · ${activePub.key_masked}` : "未配置（将使用 fake 演示模型）"}
                </span>
              </div>
              <label className="settings-field">
                <span className="settings-label">{t("settings.apiKey")}</span>
                <input
                  className="settings-control" type="password" autoComplete="off" spellCheck={false}
                  placeholder={activePub?.configured ? "已保存，留空则不修改" : "粘贴你的 API Key"}
                  value={keys[activeProvider] ?? ""}
                  onChange={(e) => setKeys((k) => ({ ...k, [activeProvider]: e.target.value }))}
                />
              </label>
              <label className="settings-field">
                <span className="settings-label">{t("settings.baseUrl")}</span>
                <input
                  className="settings-control" spellCheck={false}
                  placeholder={activeMeta.defaultBase}
                  value={baseUrls[activeProvider] ?? ""}
                  onChange={(e) => setBaseUrls((b) => ({ ...b, [activeProvider]: e.target.value }))}
                />
              </label>
              <label className="settings-field">
                <span className="settings-label">{t("settings.model")}</span>
                <select
                  className="settings-control"
                  value={models[activeProvider] ?? ""}
                  onChange={(e) => setModels((m) => ({ ...m, [activeProvider]: e.target.value }))}
                >
                  <option value="">（厂商默认）</option>
                  {(activePub?.available_models ?? []).map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </label>
              {/* 测试连接（中文诊断提示） */}
              <div className="settings-field">
                <span className="settings-label">{t("settings.test")}</span>
                <div className="test-row">
                  <button
                    className="btn-primary"
                    disabled={testing[activeProvider] || !activePub?.configured}
                    onClick={() => testProvider(activeProvider)}
                  >
                    {testing[activeProvider] ? t("settings.testing") : t("settings.test")}
                  </button>
                  {!activePub?.configured && (
                    <span className="settings-hint">{lang === "zh" ? "未配置 Key，无法测试" : "No key configured"}</span>
                  )}
                </div>
                {testRes[activeProvider] && (
                  <div className={`test-result ${testRes[activeProvider].ok ? "ok" : "err"}`}>
                    {testRes[activeProvider].msg}
                  </div>
                )}
                <div className="test-row" style={{ marginTop: 12 }}>
                  <button
                    className="btn-secondary"
                    disabled={webTesting || !activePub?.configured}
                    onClick={() => testWebSearchProvider()}
                  >
                    {webTesting ? (lang === "zh" ? "测试中…" : "Testing…") : (lang === "zh" ? "测试大模型联网" : "Test LLM web search")}
                  </button>
                  <span className="settings-hint">{lang === "zh" ? "验证当前厂商是否支持原生联网搜索（需配置对应 API Key；仅 Qwen / DeepSeek 支持）。" : "Verifies native web search for current provider (requires its API key; only Qwen / DeepSeek)."}</span>
                </div>
                {webSearchRes && (
                  <div className={`test-result ${webSearchRes.ok ? "ok" : "err"}`}>
                    {webSearchRes.msg}
                  </div>
                )}
              </div>
            </div>
        </SectionCard>
        )}

        {/* ============ 功能模块（按域分组折叠收纳） ============ */}
        {activeSec === 1 && (
        <SectionCard
          title={t("settings.features")}
          subtitle={lang === "zh" ? "控制涉及大模型 API 的模块是否启用（后端开关 + 接口保留）。点击分组名展开/收起。" : "Toggle LLM-related modules (backend switch, API kept)."}
          badge={`${
            FEATURES_DEF.filter((f) =>
              f.group === "skill" && f.key !== "featSkills"
                ? featMap.featSkills && featMap[f.key as keyof typeof featMap]
                : featMap[f.key as keyof typeof featMap],
            ).length
          }/${FEATURES_DEF.length}`}
        >
            {/* ★ P-06 问题 2（2026-10-10）：fake 模型下"开了也不生效"必须有醒目提示 ——
                实测用户在 fake 下打开开关、试用、发现"没反应" ⇒ 极易误判为 bug（静默失效同族）。 */}
            {defaultModel === "fake" && (
              <div className="settings-hint">
                ⚠ 当前默认模型 = <b>fake（本地演示）</b>：下列依赖真实大模型的能力只会得到占位/降级结果
                （技能回落、核验跳过、记忆不写入）。请先到「模型连接」配置并测试密钥，再启用这些开关。
              </div>
            )}
            {FEATURE_GROUPS.map((g) => {
              const items = FEATURES_DEF.filter((f) => f.group === g.key);
              // ★ P-06 问题 1（2026-10-10）：技能总开关关闭 ⇒ 10 个子技能实际全部停用
              //   （SkillRouter 一律回落普通对话）⇒ 计数必须反映**真实生效数**，而非拨杆位置。
              const masterOff = g.key === "skill" && !featMap.featSkills;
              const enabled = masterOff
                ? 0
                : items.filter((f) => featMap[f.key as keyof typeof featMap]).length;
              const renderSwitch = (f: (typeof items)[number]) => {
                // 子技能随总开关联动：总开关关闭 ⇒ 置灰禁用（保存的值保留，重开总开关即恢复）
                const gated = g.key === "skill" && f.key !== "featSkills" && !featMap.featSkills;
                return (
                  <label className="settings-switch" key={f.key}>
                    <input
                      type="checkbox"
                      checked={featMap[f.key as keyof typeof featMap]}
                      disabled={gated}
                      onChange={(e) => featSetters[f.key](e.target.checked)}
                    />
                    <span className="switch-track" aria-hidden="true">
                      <span className="switch-thumb" />
                    </span>
                    <span className="switch-text">
                      {f.label}：{featMap[f.key as keyof typeof featMap] ? "启用" : "关闭"}
                      {gated ? "（已随总开关停用）" : ""}
                    </span>
                    <span className="settings-hint">{f.hint}</span>
                  </label>
                );
              };
              return (
                <details key={g.key} className="feat-group" open={g.open}>
                  <summary>
                    <span className="fg-chevron" aria-hidden="true"><Icons.chevronRight size={14} /></span>
                    <span className="fg-title">{g.title}</span>
                    <span className="fg-hint">{g.hint}</span>
                    <span className="fg-count">
                      {enabled}/{items.length} 启用{masterOff ? "（总开关关闭）" : ""}
                    </span>
                  </summary>
                  <div className="fg-body">
                    {items.map(renderSwitch)}
                    {/* 通用杂项归组：重复上传提醒 → 对话与检索；图片增强 → 文档处理 */}
                    {g.key === "chat" && (
                      <label className="settings-field">
                        <span className="settings-label">记忆向量引擎（跨会话记忆的召回匹配方式）</span>
                        <select
                          className="settings-control"
                          value={memoryEmbedder}
                          onChange={(e) => setMemoryEmbedder(e.target.value)}
                        >
                          <option value="hash">hash · 轻量字符哈希（默认 · 零依赖 · 启动快）</option>
                          <option value="bge-m3">bge-m3 · 真实语义向量（需 rag 依赖 · 首次加载较慢）</option>
                        </select>
                        <span className="settings-hint">
                          hash 无语义能力（同义改写匹配不到，仅能拦住重复/近似复述）；切换为 bge-m3 后可按语义召回与去重，
                          但首次加载需载入约 2.2GB 权重。切换后会自动按新引擎重算已有记忆的向量；依赖缺失时自动降级为 hash。
                        </span>
                      </label>
                    )}
                    {g.key === "chat" && (
                      <label className="settings-field">
                        <span className="settings-label">重复上传提醒（管理员可配）</span>
                        <input
                          type="checkbox"
                          className="settings-checkbox"
                          checked={dedupeWarning}
                          onChange={(e) => setDedupeWarning(e.target.checked)}
                        />
                      </label>
                    )}
                    {g.key === "doc" && (
                      <label className="settings-switch">
                        <input
                          type="checkbox"
                          checked={featFigureEnrich}
                          onChange={(e) => setFeatFigureEnrich(e.target.checked)}
                        />
                        <span className="switch-track" aria-hidden="true">
                          <span className="switch-thumb" />
                        </span>
                        <span className="switch-text">图片增强（OCR/表格结构/VLM 图注）：{featFigureEnrich ? "启用" : "关闭"}</span>
                        <span className="settings-hint">
                          {enrichDeps
                            ? `扫描件/截图表格的「图片内文字」需 OCR 才能提取。当前依赖：OCR ${enrichDeps.ocr ? "已装" : "未装(paddleocr)"} · 表格结构 ${enrichDeps.table ? "已装" : "未装(transformers/torch)"} · VLM ${enrichDeps.vlm ? "可用" : "不可用"}。` +
                              (featFigureEnrich && !enrichDeps.ocr && !enrichDeps.table
                                ? "开启后对应能力将自动降级为空——请到「检索增强（RAG 底座）」分区安装依赖。"
                                : "开启后对后续上传的图片生效（需重新上传文档）。")
                            : "开启后对后续上传的扫描件/截图表格做 OCR 与表格结构还原（需重新上传文档）。"}
                        </span>
                      </label>
                    )}
                  </div>
                </details>
              );
            })}
        </SectionCard>
        )}

        {/* ============ 长期记忆 / Agent 规则 ============ */}
        {activeSec === 2 && (
        <SectionCard title={t("settings.rules")} subtitle={t("settings.rules.hint")}>
            <textarea
              className="settings-control rules-editor"
              rows={5}
              placeholder={lang === "zh" ? "例如：\n1. 一律用中文回答\n2. 引用必须标注页码\n3. 不确定时明确说明不确定" : "e.g.\n1. Always answer in Chinese\n2. Cite page numbers\n3. Say when unsure"}
              value={agentRules}
              onChange={(e) => setAgentRules(e.target.value)}
            />
            <span className="settings-label">
              {lang === "zh" ? "单次请求上下文上限（prompt tokens）" : "Context limit per request (prompt tokens)"}
            </span>
            <input
              className="settings-control"
              type="number"
              min={1000}
              step={1000}
              value={tokenBudget}
              onChange={(e) => setTokenBudget(Number(e.target.value))}
            />
            <p className="muted">
              {lang === "zh"
                ? "单次发给模型的 prompt 超过该值即中止本轮（防止上下文爆炸）；它同时决定历史压缩阈值（该值的一半）。绑定多篇文档时单轮 prompt 可达 2 万以上，建议不低于 30000。"
                : "A turn is aborted when a single request's prompt exceeds this (prevents runaway context); it also sets the history-compaction threshold (half of it). With several documents attached a single prompt can exceed 20k, so keep it ≥ 30000."}
            </p>
            <span className="settings-label">
              {lang === "zh" ? "整轮累计 token 上限" : "Cumulative tokens per turn"}
            </span>
            <input
              className="settings-control"
              type="number"
              min={1000}
              step={10000}
              value={tokenPerTurn}
              onChange={(e) => setTokenPerTurn(Number(e.target.value))}
            />
            <p className="muted">
              {lang === "zh"
                ? "一轮对话中所有模型调用叠加消耗的 token 上限（含多轮工具迭代）。默认 150000，正常使用不会触及；若你更在意成本，请同时收紧「单轮成本上限」。"
                : "Total tokens across every model call in one turn (including tool iterations). Default 150000 — rarely reached; for cost control prefer tightening the per-turn cost limit."}
            </p>
        </SectionCard>
        )}

        {/* ============ 单会话覆盖（WorkBuddy 卡片化） ============ */}
        {activeSec === 3 && (
        <SectionCard title="会话覆盖" subtitle="为指定会话单独设置模型（优先级高于全局默认；联网开关在对话输入区工具栏）。">
            <div className="wb-card-list">
              <div className="wb-card-row">
                <div className="wb-card-row-main">
                  <div className="wb-card-row-title">会话</div>
                  <div className="wb-card-row-desc">选择一个已存在的会话，对此会话单独生效</div>
                </div>
                <div className="wb-card-row-control">
                  <select className="settings-control" value={sessionId ?? ""} onChange={(e) => setSessionId(Number(e.target.value))}>
                    {sessions.map((s) => (
                      <option key={s.id} value={s.id}>{s.title}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="wb-card-row">
                <div className="wb-card-row-main">
                  <div className="wb-card-row-title">模型</div>
                  <div className="wb-card-row-desc">本会话使用的模型（优先级高于「模型与厂商 → 默认模型」）</div>
                </div>
                <div className="wb-card-row-control">
                  <select className="settings-control" value={model} onChange={(e) => setModel(e.target.value)}>
                    {allModels.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="settings-actions">
                <button className="btn-primary" onClick={applySession}>保存会话设置</button>
              </div>
            </div>
        </SectionCard>
        )}

        {/* ============ 数据与资产（只读清单） ============ */}
        {activeSec === 4 && (
        <SectionCard
          title={t("settings.data")}
          subtitle={lang === "zh" ? "项目可用的工具与技能清单（只读；启停通过上方功能开关）。" : "Available tools & skills (read-only; toggled above)."}
          badge={inventory ? `${inventory.tools.length} 工具 · ${inventory.skills.length} 技能` : undefined}
        >
            {invError && <div className="error-box">{invError}</div>}
            <label className="settings-field">
              <span className="settings-label">{t("settings.path")}</span>
              <code className="path-code">{settings?.data_dir ?? "…"}</code>
              <span className="settings-hint">{t("settings.path.hint")}</span>
            </label>

            {/* ── 存储位置：改位置 + 迁移（2026-10-07）──
                ★ 这里只**写待迁移请求**，真正搬迁由桌面启动器在重启时、
                  **后端启动之前**执行（那时 SQLite 未打开，不会拷到一半）。 */}
            <div className="inventory-block">
              <span className="settings-label">{lang === "zh" ? "存储位置（数据根目录）" : "Storage (data root)"}</span>
              {storageErr && <div className="error-box">{storageErr}</div>}
              {storageMsg && (
                <div className="settings-hint" style={{ whiteSpace: "pre-wrap" }}>{storageMsg}</div>
              )}
              {storage && !storage.available && (
                <div className="settings-hint">{storage.unavailable_reason}</div>
              )}
              {storage?.available && (
                <>
                  <label className="settings-field">
                    <span className="settings-label">{lang === "zh" ? "当前位置" : "Current"}</span>
                    <code className="path-code">{storage.data_root}</code>
                  </label>
                  {storage.drives.length > 0 && (
                    <div className="settings-hint">
                      {lang === "zh" ? "本机磁盘：" : "Drives: "}
                      {storage.drives
                        .map((d) => `${d.root} ${d.free_gb.toFixed(0)}GB${d.kind === "REMOTE" ? (lang === "zh" ? "（网络盘，不建议）" : " (remote)") : ""}`)
                        .join("    ")}
                    </div>
                  )}
                  {Object.entries(STORAGE_SUBDIR_LABELS).map(([key, meta]) => (
                    <label className="settings-field" key={key}>
                      <span className="settings-label">{lang === "zh" ? meta.zh : meta.en}</span>
                      <code className="path-code">{storage.dirs[key] ?? "—"}</code>
                      <span className="settings-hint">
                        {meta.hint}
                        {storage.sizes_mb[key]
                          ? `　·　${formatSize(storage.sizes_mb[key].mb, storage.sizes_mb[key].truncated, lang)}`
                          : ""}
                      </span>
                    </label>
                  ))}
                  {storagePendingBanner(storage, lang) && (
                    <div className="error-box" style={{ whiteSpace: "pre-wrap" }}>
                      {storagePendingBanner(storage, lang)}
                    </div>
                  )}
                  <label className="settings-field">
                    <span className="settings-label">{lang === "zh" ? "改到新位置" : "New location"}</span>
                    <input
                      className="settings-control"
                      value={storageRoot}
                      onChange={(e) => setStorageRoot(e.target.value)}
                      placeholder="D:\\PaperMate"
                    />
                    <span className="settings-hint">
                      {lang === "zh"
                        ? `至少需要 ${storage.min_free_gb} GB 可用空间；该目录下会新建 data / huggingface / venv / uv-cache。`
                        : `needs at least ${storage.min_free_gb} GB free; creates data / huggingface / venv / uv-cache under it.`}
                    </span>
                  </label>
                  <label className="settings-field">
                    <input
                      type="checkbox"
                      checked={storageMigrate}
                      onChange={(e) => setStorageMigrate(e.target.checked)}
                    />
                    <span className="settings-hint">
                      {lang === "zh"
                        ? "搬迁现有数据（原位置的副本会保留、不删除；venv 含绝对路径无法搬迁，会在新位置重建）"
                        : "Migrate existing data (old copy kept; venv is rebuilt because it contains absolute paths)"}
                    </span>
                  </label>
                  <button className="btn-primary" disabled={storageSaving} onClick={saveStorage}>
                    {storageSaving
                      ? (lang === "zh" ? "保存中…" : "Saving…")
                      : (lang === "zh" ? "保存（重启应用后生效）" : "Save (restart to apply)")}
                  </button>
                </>
              )}
            </div>
            <div className="inventory-block">
              <span className="settings-label">{t("settings.tools")}（{inventory?.tools.length ?? 0}）</span>
              <div className="inventory-grid">
                {(inventory?.tools ?? []).map((tl) => (
                  <span key={tl.name} className={`inv-chip ${tl.enabled === false ? "off" : ""}`} title={tl.description}>
                    {tl.name}
                    <em className={`perm perm-${tl.permission}`}>{tl.permission}</em>
                    <em className="perm">{tl.enabled === false ? "off" : "on"}</em>
                  </span>
                ))}
              </div>
            </div>
            <div className="inventory-block">
              <span className="settings-label">{t("settings.skills")}（{inventory?.skills.length ?? 0}）</span>
              <div className="inventory-grid">
                {(inventory?.skills ?? []).map((sk) => (
                  <span key={sk.name} className={`inv-chip ${sk.enabled ? "" : "off"}`} title={sk.description}>
                    <span className="inv-icon" aria-hidden="true">{sk.icon ?? <Icons.zap size={13} />}</span>
                    {sk.commands[0] ?? sk.name}
                    <em className="perm">{sk.enabled ? "on" : "off"}</em>
                  </span>
                ))}
              </div>
            </div>
        </SectionCard>
        )}

        {/* ============ 安全中心（WorkBuddy 卡片化：标题+描述+控制 三段式） ============ */}
        {activeSec === 5 && (
        <SectionCard
          title={t("settings.security")}
          subtitle={lang === "zh" ? "沙箱与权限安全配置（参考传统 Agent 安全设置项）。" : "Sandbox & permission security (agent-style)."}
        >
            <div className="wb-card-list">
              <div className="wb-card-row">
                <div className="wb-card-row-main">
                  <div className="wb-card-row-title">代码执行沙箱（Docker 隔离）</div>
                  <div className="wb-card-row-desc">
                    {lang === "zh"
                      ? "强制 Docker 隔离执行；本机无 Docker 时调用会明确报错，不会本机直跑。"
                      : "Runs only inside Docker; errors clearly if Docker is absent."}
                  </div>
                </div>
                <div className="wb-card-row-control">
                  <span className={`sec-badge ${featCodeInterpreter ? "on" : "off"}`}>
                    {featCodeInterpreter ? "已开启" : "默认关闭"}
                  </span>
                </div>
              </div>

              <div className="wb-card-row is-block">
                <div className="wb-card-row-main">
                  <div className="wb-card-row-title">工具权限分级</div>
                  <div className="wb-card-row-desc">
                    {lang === "zh"
                      ? "safe/network 可自动执行；sandbox 需显式授权（默认关）；admin 一律拒绝。"
                      : "safe/network auto; sandbox opt-in (off by default); admin denied."}
                  </div>
                </div>
                <div className="wb-card-row-control">
                  <div className="inventory-grid">
                    {(inventory?.permissions ?? []).map((p) => (
                      <span key={p} className="inv-chip"><em className={`perm perm-${p}`}>{p}</em></span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
        </SectionCard>
        )}

        {/* ============ 账户余额 ============ */}
        {activeSec === 6 && (
        <SectionCard
          title={t("settings.balances")}
          subtitle={lang === "zh" ? "各厂商账户剩余额度（DeepSeek 支持 key 直查；Qwen/OpenAI 说明原因）。" : "Remaining credits per provider."}
        >
            <div className="balance-grid">
              {balances == null && <span className="muted">{lang === "zh" ? "加载中…" : "Loading…"}</span>}
              {balances &&
                Object.entries(balances).map(([prov, b]) => (
                  <div key={prov} className={`balance-card ${b.available ? "ok" : "off"}`}>
                    <div className="balance-prov">{prov}</div>
                    {b.available ? (
                      <div className="balance-num">
                        {b.total_balance ?? 0} {b.currency ?? "CNY"}
                        <span className="muted">（可用 {b.granted_balance ?? 0}）</span>
                      </div>
                    ) : (
                      <div className="balance-reason">{b.reason ?? "不可查询"}</div>
                    )}
                  </div>
                ))}
            </div>
        </SectionCard>
        )}

        {/* ============ 长期记忆管理（2026-08-29 修复：从保存栏之后移入分区流，默认展开；此前被感知为「不可见」） ============ */}
        {activeSec === 7 && (
        <MemoryPanel />
        )}

        {/* ============ 检索增强（RAG 底座：embedding 模式 + 稀疏/分词开关 + 重建索引） ============ */}
        {activeSec === 8 && (
        <SectionCard
          title={lang === "zh" ? "检索增强（RAG 底座）" : "Retrieval Enhancement (RAG)"}
          subtitle={lang === "zh"
            ? "切换 embedding 模式与关键词检索口径；改后务必点「重建索引」使新配置对所有文档生效。"
            : "Switch embedding mode & lexical settings; reindex after changing to apply to all docs."}
        >
            {/* ── 真实语义检索模型（2026-10-05：**必须由用户显式下载**）──
                为什么单独一块：两条ธุร线不同 —— 上面是"改配置"，这里是"拉资产"，
                后者涉及约 4.5GB 下载与"不重建索引就不生效"的代价，必须让用户看得见、能取消。 */}
            <div className="settings-field">
              <span className="settings-label">{lang === "zh" ? "语义检索模型（约 4.5GB）" : "Semantic retrieval models (~4.5GB)"}</span>
              {ragModelsErr && <span className="settings-hint">{ragModelsErr}</span>}
              {/* ① 运行环境（重依赖）—— 桌面首装刻意不装，点下载时才会补装。
                  必须单独显示，否则用户点下去才发现要先下 4GB，会以为坏了。 */}
              {ragModels && (
                <span className="settings-hint">
                  {lang === "zh" ? "运行环境（PyTorch 等，约 4GB）：" : "Runtime deps (PyTorch etc., ~4GB): "}
                  {ragModels.deps_ready ? (
                    <b style={{ color: "var(--color-text-success, #3B6D11)" }}>
                      {lang === "zh" ? "已就绪" : "ready"}
                    </b>
                  ) : (
                    <b style={{ color: "var(--color-text-warning, #854F0B)" }}>
                      {lang === "zh" ? "未安装（点下方按钮会自动补装）" : "not installed (installed by the button below)"}
                    </b>
                  )}
                  {!ragModels.deps_ready && ragModels.deps_missing.length > 0 && (
                    <span style={{ opacity: 0.75 }}>
                      {" · "}
                      {ragModels.deps_missing.map((m) => m.split("：")[0]).join("、")}
                    </span>
                  )}
                  {!ragModels.deps_ready && !ragModels.deps_target?.uv_found && ragModels.deps_target?.reason && (
                    <span style={{ opacity: 0.75 }}>{" · "}{ragModels.deps_target.reason}</span>
                  )}
                </span>
              )}
              {ragModels && (
                <ul className="settings-hint" style={{ margin: 0, paddingLeft: "1.1em", listStyle: "disc" }}>
                  {ragModels.models.map((m) => (
                    <li key={m.key}>
                      <b>{m.label}</b>
                      {" · "}
                      {m.installed
                        ? (lang === "zh" ? "已安装" : "installed")
                        : (lang === "zh" ? `未安装（约 ${m.approx_gb}GB）` : `not installed (~${m.approx_gb}GB)`)}
                      {m.local_path ? <span style={{ opacity: 0.7 }}> · {m.local_path}</span> : null}
                    </li>
                  ))}
                </ul>
              )}
              {ragModels && ragModels.disk_free_gb != null && (
                <span className="settings-hint">
                  {lang === "zh" ? `缓存盘可用 ${ragModels.disk_free_gb}GB` : `Disk free: ${ragModels.disk_free_gb}GB`}
                </span>
              )}
              <span className="settings-hint">{ragModels?.hint ?? ""}</span>
            </div>

            {ragInstall && ragInstall.status === "running" ? (
              <div className="reindex-viz" role="status" aria-live="polite">
                <div className="reindex-bar-track">
                  {/* ★ P-01（2026-10-09）：进度口径升级为**真字节**。
                      `total_bytes > 0` ⇒ 画确定进度条（deps 量 uv-cache 磁盘增量、models 量模型目录体积）；
                      `total_bytes === 0` ⇒ 信号不可用（热缓存增量≈0 / 元数据缺失）⇒ **保持不确定动画**。
                      纪律不变：假装有进度比没有进度更糟（用户会以为卡住了）。 */}
                  <div
                    className={`reindex-bar${ragInstall.total_bytes > 0 ? "" : " indeterminate"}`}
                    style={{
                      width: ragInstall.total_bytes > 0 ? `${Math.round(ragInstall.percent)}%` : "100%",
                    }}
                  />
                </div>
                <div className="reindex-meta">
                  <span>
                    {ragInstall.message}
                    {/* 字节进度（主口径）；文件数降为辅助信息 */}
                    {ragInstall.total_bytes > 0
                      ? `（${formatSize(ragInstall.done_bytes / 1048576, undefined, lang)} / ${formatSize(ragInstall.total_bytes / 1048576, undefined, lang)}）`
                      : ""}
                    {ragInstall.phase === "models" && ragInstall.file_total > 0
                      ? ` · 第 ${ragInstall.file_index}/${ragInstall.file_total} 个文件`
                      : ""}
                  </span>
                  {ragInstall.elapsed_s > 0 && <span> · {Math.round(ragInstall.elapsed_s)}s</span>}
                </div>
                <div className="settings-actions">
                  <button type="button" className="settings-btn-ghost" onClick={cancelRagInstall}>
                    {lang === "zh" ? "取消下载" : "Cancel"}
                  </button>
                  <span className="settings-hint">
                    {lang === "zh"
                      ? "取消后已下载的部分会被缓存，下次继续。依赖安装阶段可能需要几分钟才生效。"
                      : "Cached progress is kept for resume. The dependency phase may take minutes to report."}
                  </span>
                </div>
              </div>
            ) : (
              <div className="settings-actions">
                <button
                  type="button"
                  className="btn-primary"
                  disabled={!ragModels || (ragModels.all_installed && ragModels.deps_ready)}
                  onClick={() => startRagInstall()}
                >
                  {!ragModels
                    ? (lang === "zh" ? "加载中…" : "Loading…")
                    : ragModels.all_installed && ragModels.deps_ready
                      ? (lang === "zh" ? "已全部就绪" : "All ready")
                      : ragModels.deps_ready
                        ? (lang === "zh" ? "下载模型（约 4.5GB）" : "Download models (~4.5GB)")
                        : (lang === "zh" ? "准备环境并下载（约 8.5GB）" : "Prepare env & download (~8.5GB)")}
                </button>
                <span className="settings-hint">
                  {lang === "zh"
                    ? "将依次完成：① 补装 PyTorch 等运行环境（约 4GB）② 下载 BGE-M3 与重排模型（约 4.5GB）。完成后请把「嵌入模式」切到 real 并点「重建索引」。"
                    : "Runs: (1) install PyTorch runtime deps (~4GB), (2) download BGE-M3 + reranker weights (~4.5GB). Then switch Embedding mode to real and reindex."}
                </span>
              </div>
            )}

            {ragInstall && ragInstall.status === "error" && (
              <span className="settings-hint">
                {lang === "zh" ? "下载失败：" : "Download failed: "}
                {ragInstall.error}
              </span>
            )}
            {ragInstall && ragInstall.status === "success" && (
              <span className="settings-hint">
                {lang === "zh" ? "下载已完成。" : "Download finished. "}
                {ragInstall.message}
              </span>
            )}

            {/* ★ 图片增强依赖（OCR / 表格结构）—— 2026-10-10 应用户要求移入本分区：
                这里是「模型与运行环境下载中心」，重依赖下载集中一处；文档处理组保留开关 + 指引。
                `[figure]` extra 与 `[rag]` 分离（pyproject），此前无安装入口 ⇒ OCR 装了也不生效。 */}
            <div className="settings-field">
              <span className="settings-label">图片增强依赖（OCR / 表格结构）</span>
              {/* ★ 2026-10-10：状态行必须在本卡片内可见（原先只在「文档处理」的开关提示里，
                  用户在本卡片点「重新检测」看不到任何变化 ⇒ 被当成"没用"）。 */}
              <div className="settings-hint">
                {enrichDeps
                  ? `当前依赖：OCR ${enrichDeps.ocr ? "已装" : "未装"} · 表格结构 ${enrichDeps.table ? "已装" : "未装"} · VLM ${enrichDeps.vlm ? "可用" : "不可用"}`
                  : "依赖状态未知"}
                {depsChecking && " · 检测中…"}
              </div>
              <div className="settings-hint">
                {figInstall?.status === "running" ? (
                  <>
                    {figInstall.message}
                    {figInstall.elapsed_s > 0 && <> · {Math.round(figInstall.elapsed_s)}s</>}{" "}
                    <button
                      type="button"
                      className="settings-btn-ghost"
                      onClick={() => {
                        void api.cancelFigureDeps().then(setFigInstall);
                      }}
                    >
                      取消
                    </button>
                  </>
                ) : (
                  <>
                    {/* ★ 与 RAG 安装按钮同款语义：缺什么显示「安装依赖」；全齐显示「已全部就绪」（禁用）——
                        保证**任何状态下都有一个可见按钮**（原先全装好时按钮整个消失，用户找不到"下载按钮"）。 */}
                    {enrichDeps && !(enrichDeps.ocr && enrichDeps.table) ? (
                      <button type="button" className="btn-secondary" onClick={() => void startFigInstall()}>
                        安装依赖（下载约 150~250MB · 落盘约 550MB）
                      </button>
                    ) : (
                      <button type="button" className="btn-primary" disabled>
                        已全部就绪
                      </button>
                    )}{" "}
                    <button type="button" className="settings-btn-ghost" disabled={depsChecking} onClick={() => void recheckDeps()}>
                      {depsChecking ? "检测中…" : "重新检测"}
                    </button>
                  </>
                )}
              </div>
              {figInstall?.status === "success" && <span className="settings-hint">✓ {figInstall.message}</span>}
              {figInstall?.status === "cancelled" && <span className="settings-hint">{figInstall.message}</span>}
              {figInstall?.status === "error" && (
                <span className="settings-hint">
                  ✗ {figInstall.error ?? figInstall.message}（详见 data/logs/app.log）
                </span>
              )}
              <span className="settings-hint">
                扫描件/截图里的文字需 OCR 才能进检索语料（安装后到「文档处理」打开「图片增强」开关，
                并<b>重新上传文档</b>）。安装走 App 自带 uv 与数据根缓存（已下载部分自动复用，可重试）；
                若尚未「准备运行环境」，会一并补齐 torch 等重依赖（约 4GB）。
              </span>
            </div>

            <label className="settings-field">
              <span className="settings-label">{lang === "zh" ? "嵌入模式（Embedding）" : "Embedding mode"}</span>
              <select
                className="settings-control"
                value={ragEmbedder}
                onChange={(e) => setRagEmbedder(e.target.value)}
              >
                <option value="fake">{lang === "zh" ? "fake · 哈希占位（默认 · 秒级 · 零依赖 · 无真实语义）" : "fake · hash (default, fast, no real semantics)"}</option>
                <option value="real">{lang === "zh" ? "real · BGE-M3 真实语义（需 rag 依赖 · ~2.2GB 权重 · 自动联动重排）" : "real · BGE-M3 (needs rag deps & ~2.2GB weights)"}</option>
              </select>
              <span className="settings-hint">
                {lang === "zh"
                  ? "fake 仅验证检索流程，不可用于真实语义召回；real 切换后自动启用 Cross-Encoder 重排，显著提升召回精度。首次加载需下载权重。"
                  : "fake only validates the pipeline (no real semantics); real auto-enables Cross-Encoder reranking for better recall (downloads weights on first load)."}
              </span>
            </label>

            <label className="settings-switch">
              <input type="checkbox" checked={useBgeSparse} onChange={(e) => setUseBgeSparse(e.target.checked)} />
              <span className="switch-track" aria-hidden="true"><span className="switch-thumb" /></span>
              <span className="switch-text">{lang === "zh" ? "BGE-M3 稀疏向量（与 BM25 并行的词法信号）" : "BGE-M3 sparse vector (lexical signal parallel to BM25)"}：{useBgeSparse ? (lang === "zh" ? "启用" : "on") : (lang === "zh" ? "关闭" : "off")}</span>
              <span className="settings-hint">
                {lang === "zh"
                  ? "在真实 BGE-M3 部署下，用同向量空间的稀疏向量作为 BM25 之外的词法信号，融合更一致；默认关闭，建议先用评测集量化收益后再开。"
                  : "Under real BGE-M3, adds a sparse vector (same space as BM25) as an extra lexical signal; off by default until eval-confirmed."}
              </span>
            </label>

            <label className="settings-switch">
              <input type="checkbox" checked={useJieba} onChange={(e) => setUseJieba(e.target.checked)} />
              <span className="switch-track" aria-hidden="true"><span className="switch-thumb" /></span>
              <span className="switch-text">{lang === "zh" ? "中英文分词（jieba 按词切分 BM25）" : "Zh/En tokenization (jieba word-segmented BM25)"}：{useJieba ? (lang === "zh" ? "启用" : "on") : (lang === "zh" ? "关闭" : "off")}</span>
              <span className="settings-hint">
                {lang === "zh"
                  ? "英文文档优化：开启后 jieba 按词切分——中文更贴合语义，英文也由字符级转为词级（修复英文 BM25 召回弱的问题）；未安装 jieba 时自动回退字符级，不影响功能。仅影响关键词检索，不影响向量检索。"
                  : "English-doc optimization: on segments via jieba — better Chinese semantics AND English word-level (fixes weak English BM25 under char-level); auto-falls back to char-level if jieba missing. Affects lexical (BM25) retrieval only."}
              </span>
            </label>

            {/* T-03 重排推理加速（2026-09-24）：两个档位改完即时生效（后端会连管线缓存一起清） */}
            <label className="settings-switch">
              <input type="checkbox" checked={rerankFp16} onChange={(e) => setRerankFp16(e.target.checked)} />
              <span className="switch-track" aria-hidden="true"><span className="switch-thumb" /></span>
              <span className="switch-text">{lang === "zh" ? "重排半精度（fp16）" : "Rerank fp16"}：{rerankFp16 ? (lang === "zh" ? "启用" : "on") : (lang === "zh" ? "关闭" : "off")}</span>
              <span className="settings-hint">
                {lang === "zh"
                  ? "实测（4 候选 / RTX 4070 Laptop）：半精度把重排 p50 从 188ms 降到 63ms（×3.0），且检索 Top-5 完全一致。⚠️ 仅在 CUDA 上生效——CPU 上 fp16 通常更慢，会自动回落 fp32。"
                  : "Measured (4 candidates / RTX 4070 Laptop): fp16 cuts rerank p50 from 188ms to 63ms (×3.0) with identical Top-5. ⚠️ CUDA only — on CPU fp16 is usually slower, so it falls back to fp32."}
              </span>
            </label>

            <label className="settings-field">
              <span className="settings-label">{lang === "zh" ? "重排候选截断长度（token）" : "Rerank max tokens"}</span>
              <select
                className="settings-control"
                value={rerankMaxLen}
                onChange={(e) => setRerankMaxLen(Number(e.target.value))}
              >
                <option value={0}>{lang === "zh" ? "0 · 不限长（等于改动前行为）" : "0 · unlimited (legacy behaviour)"}</option>
                <option value={512}>{lang === "zh" ? "512 · 推荐（默认）" : "512 · recommended (default)"}</option>
                <option value={384}>{lang === "zh" ? "384 · 截断约 35% 候选（实测已不再作默认）" : "384 · truncates ~35% candidates (no longer default)"}</option>
                <option value={256}>{lang === "zh" ? "256 · 更快，但实测 hit@5 −4.9pp" : "256 · faster, measured hit@5 −4.9pp"}</option>
                <option value={128}>128</option>
              </select>
              <span className="settings-hint">
                {lang === "zh"
                  ? "送进 Cross-Encoder 的最大 token 数。提速来自「砍长尾」：一个 batch 的成本由其中最长序列决定（要 padding 到最大值）。★ 2026-10-05 质量优先实测（221 用例）：512 全面优于 384（hit@5 +3.3pp / nDCG@5 +5.2pp，代价 +51ms），而 384 会截断约 35% 候选、256 掉 hit@5 4.9pp ⇒ 默认已上调为 512。"
                  : "Max tokens fed to the Cross-Encoder. Gains come from cutting the long tail (a batch costs as much as its longest sequence, due to padding). Measured 2026-10-05 (221 cases): 512 beats 384 across the board (hit@5 +3.3pp / nDCG@5 +5.2pp, +51ms), while 384 truncates ~35% of candidates and 256 loses 4.9pp hit@5 — default is now 512."}
              </span>
            </label>

            <div className="settings-actions">
              <button className="btn-primary" disabled={reindexState.phase === "running"} onClick={runReindex}>
                {reindexState.phase === "running" ? (
                  <span className="reindex-btn-inner">
                    <span className="reindex-spinner" aria-hidden="true" />
                    {lang === "zh" ? "重建中…" : "Reindexing…"}
                  </span>
                ) : (
                  (lang === "zh" ? "重建索引" : "Reindex")
                )}
              </button>
              {reindexState.phase === "running" && (
                <button
                  type="button"
                  className="settings-btn-ghost"
                  onClick={cancelReindex}
                  aria-label={lang === "zh" ? "取消重建" : "Cancel reindex"}
                >
                  {lang === "zh" ? "取消" : "Cancel"}
                </button>
              )}
              <span className="settings-hint">
                {lang === "zh"
                  ? "改 embedding 模式 / 稀疏 / 分词后必须点此，否则检索仍用旧配置缓存。"
                  : "Required after changing embedder / sparse / tokenization; otherwise stale cache is used."}
              </span>
            </div>

            {reindexState.phase !== "idle" && (
              <div className="reindex-viz" role="status" aria-live="polite">
                <div className="reindex-bar-track">
                  <div
                    className={`reindex-bar${reindexState.phase === "running" && reindexState.total === 0 ? " indeterminate" : ""}`}
                    style={{ width: reindexState.total > 0 ? `${Math.round((reindexState.done / reindexState.total) * 100)}%` : "100%" }}
                  />
                </div>
                <div className="reindex-meta">
                  {reindexState.phase === "running" ? (
                    <span>
                      {lang === "zh" ? "已处理 " : "Processed "}
                      <b>{reindexState.done}</b>
                      {reindexState.total > 0 ? (lang === "zh" ? ` / ${reindexState.total} 篇` : ` / ${reindexState.total}`) : ""}
                      {reindexState.lastDoc != null
                        ? (lang === "zh" ? ` · 当前 #${reindexState.lastDoc}` : ` · doc #${reindexState.lastDoc}`)
                        : ""}
                    </span>
                  ) : reindexState.phase === "done" ? (
                    <span className={`reindex-stat ${reindexState.failed.length ? "warn" : "ok"}`}>
                      {lang === "zh"
                        ? `完成：成功 ${reindexState.reindexed} / 总计 ${reindexState.total}${reindexState.failed.length ? `，失败 ${reindexState.failed.length}` : ""}`
                        : `Done: ${reindexState.reindexed}/${reindexState.total} ok${reindexState.failed.length ? `, ${reindexState.failed.length} failed` : ""}`}
                    </span>
                  ) : reindexState.phase === "cancelled" ? (
                    <span className="reindex-stat warn">
                      {lang === "zh"
                        ? `已取消：已处理 ${reindexState.done} / ${reindexState.total} 篇（${reindexState.reindexed} 篇成功）`
                        : `Cancelled: ${reindexState.done}/${reindexState.total} processed (${reindexState.reindexed} ok)`}
                    </span>
                  ) : (
                    <span className="reindex-stat err">{reindexState.error ?? (lang === "zh" ? "重建失败" : "Reindex failed")}</span>
                  )}
                </div>
                {reindexState.failed.length > 0 && reindexState.phase === "done" && (
                  <details className="reindex-failed">
                    <summary>{lang === "zh" ? `查看 ${reindexState.failed.length} 篇失败详情` : `View ${reindexState.failed.length} failures`}</summary>
                    <ul>
                      {reindexState.failed.map((f) => (
                        <li key={f.id}>
                          <b>#{f.id}</b> · {f.error}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            )}
        </SectionCard>
        )}
      </div>

      {/* ============ 吸附保存栏（始终可见） ============ */}
      <div className="settings-savebar">
        {globalMsg && (
          <div className={`settings-toast ${globalMsg.kind}`} onClick={() => setGlobalMsg(null)}>
            {globalMsg.text}
          </div>
        )}
        <button className="btn-primary btn-lg" onClick={saveGlobal} disabled={saving}>
          {saving ? "保存中…" : t("settings.save")}
        </button>
      </div>
    </div>
  );
}
