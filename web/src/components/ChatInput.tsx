import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { Icons } from "../lib/icons";
import { composerWarns } from "../lib/networkWarn";
import type { DocScopeOut, InventoryOut, NetworkCapabilityOut, PromptEnhanceResult, ProviderBalance } from "../lib/types";
import { docScopeHint } from "../lib/docScope";
import {
  isKeyVisible,
  loadDismissedKeys,
  pushDismissedKey,
  saveDismissedKeys,
} from "../lib/dismissedKeys";
import { filterSkills } from "../lib/skillFilter";
import { CtxUsageBar } from "./CtxUsagePanel";
import { PromptWizard } from "./PromptWizard";
import { toErrorMessage } from "../lib/error";

// 技能分类 → 命令栏左侧强调色（与后端 SkillSpec.category 对齐；新增分类在此补色即可）
const CATEGORY_COLOR: Record<string, string> = {
  研究: "var(--cat-green)",
  规划: "var(--cat-blue)",
  评测: "var(--cat-purple)",
  写作: "var(--cat-rose)",   // 改稿（/改稿）
};

// 学科视角选项（与后端 disciplines registry / 设置页对齐；新增学科在此追加一行）
const DISCIPLINES: { value: string; label: string }[] = [
  { value: "general", label: "通用" },
  { value: "auto", label: "自动（按内容推测）" },
  { value: "clinical", label: "医学/临床" },
  { value: "law", label: "法学" },
  { value: "finance", label: "经济/金融" },
  { value: "bioscience", label: "生命科学/生物" },
  { value: "materials", label: "材料科学" },
  { value: "chemistry", label: "化学" },
  { value: "social_science", label: "社会科学" },
  { value: "engineering", label: "工程" },
  { value: "physics", label: "物理学" },
];
// features 子键 → 中文可读名（禁用技能 tooltip 提示开启项）
const FEATURE_LABEL: Record<string, string> = {
  planning: "任务规划",
  dev_eval: "开发者评测",
  deep_read: "单篇精读",
  glossary: "术语表",
  find_code: "找代码",
  research_gap: "研究缺口",
  experiment_design: "实验设计",
  review: "审稿",
  reproduction: "复现辅助",
  revision: "改稿",
};

// 多模态（DeepSeek-V4-Flash-Vision）：图片约束（官方支持 JPEG/PNG/GIF/WebP，单张 ≤32MiB；本地单次 ≤10 张）
const MAX_IMAGES = 10;
const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
// 与后端 lightcore.llm.router.MULTIMODAL_MODELS 保持一致（2026-08-25）：仅这些模型可接收对话图片
const MULTIMODAL_MODELS = new Set(["deepseek-v4-flash-vision-exp", "qwen3-vl-flash"]);
// 输入框高度：默认 2 行；内容超过约 3 行后随内容自适应增高；最高约 10 行后内部滚动
const INPUT_MIN_H = 64;
const INPUT_MAX_H = 240;

/**
 * 对话输入区：悬浮底部 + 自动扩展 + 图片选择 + 命令收纳（details）+ 模型/余额集成。
 * - Enter 发送 / Shift+Enter 换行；流式进行中显示「停止」按钮
 * - 命令列表可收纳折叠；模型选择与余额展示固定在输入区下方
 */
export function ChatInput({
  onSend,
  onStop,
  streaming,
  disabled,
  models,
  currentModel,
  onModelChange,
  // modelSource / balances 不再解构渲染（2026-10-07 工具条瘦身，见底部注释），
  // 但 props 类型保留，调用方（ChatPage）无需改动。
  webOn,
  onWebToggle,
  webCapability,
  docScope,
  webCapabilityKey,
  discipline,
  onDisciplineChange,
  skills,
  mentionDocs,
  onMentionPick,
  mentionedDocIds,
  sessionId,
  ctxUsage,
}: {
  onSend: (text: string, images?: string[]) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
  models?: string[];
  currentModel?: string;
  onModelChange?: (m: string) => void;
  /** F4：模型生效来源——session=本会话显式指定；global=继承设置页全局默认 */
  modelSource?: "session" | "global";
  balances?: Record<string, ProviderBalance> | null;
  /** 联网开关状态（会话 config.enable_network），由 ChatPage 传入 */
  webOn?: boolean;
  /** 切换联网开关（即时持久化到会话 config） */
  onWebToggle?: () => void;
  /** 联网可用性（2026-09-12）：联网是模型原生能力，且 doc-bound 会回落模型 →
   *  「开关已开」≠「能力可用」。据此在开关旁给出预警，避免用户误以为"开了却不搜"。 */
  webCapability?: NetworkCapabilityOut | null;
  /** 会话检索范围（2026-09-23，场景可见性）：用于显示「当前处于哪个场景、本轮会不会检索论文」。
   *  会话级快照、稳定无乐观更新冲突，故不需要 webCapability 那套请求指纹。 */
  docScope?: DocScopeOut | null;
  /** `webCapability` 的**请求指纹**（发起请求时的输入，2026-09-17）：
   *  指纹与当前输入不符 = 这份数据属于"别的状态"，不得用于预警（否则会闪现错误提示）。 */
  webCapabilityKey?: string | null;
  /** 学科视角（会话 config.discipline）：影响技能解读框架（通用/自动/临床/法学/金融/生物） */
  discipline?: string;
  /** 切换学科视角（即时持久化到会话 config） */
  onDisciplineChange?: (d: string) => void;
  /** 技能清单（来自 GET /settings/inventory，驱动命令栏；由后端 SkillSpec 注册表统一维护） */
  skills?: InventoryOut["skills"];
  /** @ 提及文档列表（已关联会话的文档，带中文编号）；为空则不启用 @ 弹窗 */
  mentionDocs?: { id: number; label: string; filename: string }[];
  /** 选中某篇文档（确保已关联会话，使其进入检索上下文） */
  onMentionPick?: (id: number) => void;
  /** 本轮已 `@` 的文档 id（2026-09-23，建议③）。
   *
   *  **为什么必须传**：后端在 `@` 时会把检索范围**聚焦到被指定的篇目**
   *  （`_effective_doc_scope` 的 @ 聚焦语义），而提示条若不同步，
   *  就会在 `@` 之后仍显示「检索全部 N 篇文档」—— **提示与真实行为相反**，
   *  比没有提示更糟（用户会基于错误提示下结论）。
   *  发送后由父组件清空（`@` 只作用于当轮）。 */
  mentionedDocIds?: number[];
  /** 当前会话 id：提示词增强时可附带最近对话上下文（可选） */
  sessionId?: number;
  /** ★ P2（2026-10-03）：本轮上下文用量（由 ChatPage 从最后一条 assistant 消息取出）。
   *  缺失 ⇒ 「上下文用量」面板整体不渲染（优雅降级）。 */
  ctxUsage?: {
    peak?: number;
    budget?: number;
    warnAt?: number;
    breakdown?: Record<string, number>;
  } | null;
}) {
  // 2026-08-25：当前模型是否多模态（可接收对话图片）；与后端 MULTIMODAL_MODELS 对齐
  const canVision = currentModel != null && MULTIMODAL_MODELS.has(currentModel);
  const [text, setText] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [imgError, setImgError] = useState<string | null>(null);
  // WorkBuddy 习惯：技能默认收纳为紧凑入口，点击把手展开（避免长期占用输入区上方空间）
  const [skillsOpen, setSkillsOpen] = useState(false);
  /** ★ P0-2/P2（2026-10-03）：命令面板的**本地过滤词**。
   *
   * 解决两件事：
   * ① 12 个技能共 **24 个命令**平铺，用户要靠滚动找（诊断报告 §3）；
   * ② **英文别名在 UI 里原本完全不存在**（前端只用了 `commands[0]`）——
   *    过滤匹配 `commands` **全部**别名 ⇒ 搜 "deep" 就能找到 `/deep_read`，
   *    等于把别名变成了可发现的（见诊断报告 §2.3）。
   */
  const [skillQuery, setSkillQuery] = useState("");
  /** 过滤后的技能（★ 逻辑在 `lib/skillFilter.ts` 的纯函数里，可行为测试）。 */
  const filteredSkills = filterSkills(skills, skillQuery);
  /** F7：提问模板向导（「提问模板」按钮）开关 */
  const [wizardOpen, setWizardOpen] = useState(false);
  // @ 提及文档：弹出选择（2026-08-26）
  const [mentionOpen, setMentionOpen] = useState(false);
  // 提示词增强（2026-09-03）：调后端单轮元任务改写输入；结果预览确认后应用
  const [enhancing, setEnhancing] = useState(false);
  const [enhRes, setEnhRes] = useState<PromptEnhanceResult | null>(null);
  const [enhErr, setEnhErr] = useState<string | null>(null);
  /** 输入框手动拖拽高度（null=自适应；一旦拖拽则固定为该高度，不再随内容自动伸缩，直到下次拖拽） */
  const [userHeight, setUserHeight] = useState<number | null>(null);
  /** 模型自定义下拉（替换原生 select：支持深色主题 + 展开动画） */
  const [modelOpen, setModelOpen] = useState(false);
  const modelRef = useRef<HTMLDivElement>(null);
  /** 学科视角自定义下拉（与模型同一套弹层样式，2026-09-01 替换原生 select） */
  const [discOpen, setDiscOpen] = useState(false);
  const discRef = useRef<HTMLDivElement>(null);
  const skillBtnRef = useRef<HTMLButtonElement>(null);
  /** 技能 / 模板面板锚点：用于「点击空白处关闭」的边界判定（锚点 = 触发按钮 + 面板整体） */
  const skillAnchorRef = useRef<HTMLSpanElement>(null);
  const wizardAnchorRef = useRef<HTMLSpanElement>(null);

  /**
   * 预警横幅可消除（2026-09-17）：下面两条提示（联网能力不足 / 模型检索风险）会一直挂在输入框上方，
   * 看久了很吵。但它们是**安全提示**，不能让用户一关就永久静音 —— 所以关闭按**指纹**记忆：
   * `会话 + 提示类型 + 文案` 任一变化（换模型 / 换风险类型 / 换会话）→ 指纹变化 → 提示**自动重新出现**。
   * 存 sessionStorage（关掉标签页即重置），只保留最近 20 条避免无限累积。
   * （2026-10-07：存取/追加逻辑抽到 `lib/dismissedKeys`，与检索范围提示的关闭记忆共用同一实现。）
   */
  const [dismissedWarns, setDismissedWarns] = useState<string[]>(() =>
    loadDismissedKeys("pm.dismissedWarns"),
  );
  const dismissWarn = (key: string) => {
    setDismissedWarns((prev) => {
      const next = pushDismissedKey(prev, key);
      saveDismissedKeys("pm.dismissedWarns", next);
      return next;
    });
  };

  /**
   * 检索范围提示可关闭（2026-10-07）：「检索全部 N 篇文档：…」是**状态说明**，常驻会占一整行，
   * 用户看完即可关掉。关闭按 scopeHint 的内容 key 记忆 —— 范围一变（增删文档 / `@` 聚焦 /
   * 换会话）key 随之变化 ⇒ 提示自动重新出现，与 warns 的指纹记忆同一语义。
   * 语义差异提醒：warns 是"安全提示"（换模型就回来），这里是"状态说明"（范围变了才回来），
   * 共用的是**机制**（key 指纹 + sessionStorage + 截断 20 条），不是业务含义。
   */
  const [dismissedScopeHints, setDismissedScopeHints] = useState<string[]>(() =>
    loadDismissedKeys("pm.dismissedScopeHints"),
  );
  const dismissScopeHint = (key: string) => {
    setDismissedScopeHints((prev) => {
      const next = pushDismissedKey(prev, key);
      saveDismissedKeys("pm.dismissedScopeHints", next);
      return next;
    });
  };
  /** 输入框上方的预警横幅（联网能力不足 / 模型检索风险）：判定下沉到 `lib/networkWarn`。
   *
   * ⚠️ **不要在这里手写条件**。2026-09-17 的「提示闪烁」正是把判定散在组件里造成的：
   * 用的是**乐观更新**的 `webOn` 配**上一次请求**的 `webCapability.available`，
   * 点开关那一刻两者必然短暂矛盾 → 提示闪现约 50ms 后消失（帧级实测）。
   * 纯函数 + 请求指纹把这条时序约束钉住，并有 vitest 守护。 */
  const scopeHintRaw = docScopeHint(docScope ?? null, mentionedDocIds);
  // 可关闭（2026-10-07）：被关掉的 key 不再渲染；范围一变 key 变 ⇒ 自动重新出现
  const scopeHint =
    scopeHintRaw && isKeyVisible(dismissedScopeHints, scopeHintRaw.key) ? scopeHintRaw : null;

  const warns = composerWarns({
    sessionId: sessionId ?? null,
    webOn: Boolean(webOn),
    docsLen: mentionDocs?.length ?? 0,
    model: currentModel ?? "",
    cap: webCapability ?? null,
    capKey: webCapabilityKey ?? null,
    dismissed: dismissedWarns,
  });

  /**
   * 工作台窄面板（340px）弹窗自适应高度：面板绝对定位于输入卡上方，可用空间 =
   * 输入卡顶到面板容器（.workbench-chat，overflow:hidden 裁切者）顶的距离；
   * 固定 320px 上限在窄面板会「上端溢出容器 + 下端被输入卡遮挡」，故打开时按实测写入 maxHeight。
   * 普通宽布局（无 .workbench-chat 祖先）不受影响，走 CSS 默认值。
   */
  useLayoutEffect(() => {
    if (!skillsOpen && !wizardOpen) return;
    const anchor = skillsOpen ? skillAnchorRef.current : wizardAnchorRef.current;
    const chat = anchor?.closest(".workbench-chat");
    // 工作台合成器根节点是 .chat-composer（普通聊天为 .chat-input），二者都兼容
    const card = anchor?.closest(".chat-composer, .chat-input");
    const panel = anchor?.querySelector<HTMLElement>(".skill-panel, .wizard-panel");
    if (!anchor || !chat || !card || !panel) return;
    const avail = card.getBoundingClientRect().top - chat.getBoundingClientRect().top - 10;
    panel.style.maxHeight = `${Math.max(160, Math.min(320, Math.round(avail)))}px`;
  }, [skillsOpen, wizardOpen]);

  const [mentionQuery, setMentionQuery] = useState("");
  const [mentionIndex, setMentionIndex] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement | null>(null);
  /** L7（2026-09-10）：拖拽进行中的清理函数——组件卸载时兜底调用（防监听/body 样式残留） */
  const dragCleanupRef = useRef<(() => void) | null>(null);

  /** 提示词增强：调 POST /prompt/enhance（单轮元任务）。applied=false（fake/失败/关闭）→ 以错误形式提示 reason，不弹预览。 */
  const runEnhance = async () => {
    const raw = text.trim();
    if (!raw) {
      taRef.current?.focus();
      return;
    }
    setEnhancing(true);
    setEnhErr(null);
    setEnhRes(null);
    try {
      const r = await api.enhancePrompt(raw, sessionId);
      if (!r.applied) {
        setEnhErr(r.reason || "增强失败：请检查模型配置后重试");
        return;
      }
      setEnhRes(r);
    } catch (e: unknown) {
      setEnhErr(toErrorMessage(e) || "提示词增强失败");
    } finally {
      setEnhancing(false);
    }
  };
  const applyEnhance = () => {
    if (!enhRes) return;
    const enhanced = enhRes.enhanced;
    setText(enhanced);
    setEnhRes(null);
    // 等待 React 提交新内容后再自适应高度：否则 textarea 仍按旧高度渲染，
    // 表现为「增强后文本更长却对话框不随内容增多而变大」
    requestAnimationFrame(() => {
      const ta = taRef.current;
      if (!ta) return;
      ta.focus();
      autoResize(ta);
    });
  };
  const closeEnhance = () => {
    setEnhRes(null);
    setEnhErr(null);
  };

  /** 从光标往前找最近的 @…（中间无空白即视为正在提及） */
  const getMentionContext = (value: string, caret: number) => {
    const before = value.slice(0, caret);
    const at = before.lastIndexOf("@");
    if (at === -1) return { active: false, query: "", atPos: -1 };
    const between = before.slice(at + 1);
    if (/[\s\n]/.test(between)) return { active: false, query: "", atPos: -1 };
    return { active: true, query: between, atPos: at };
  };

  const filteredMentions = (mentionDocs ?? []).filter(
    (d) =>
      !mentionQuery ||
      d.filename.toLowerCase().includes(mentionQuery.toLowerCase()) ||
      d.label.includes(mentionQuery),
  );

  const applyMention = (doc: { id: number; label: string; filename: string }) => {
    const ta = taRef.current;
    const caret = ta?.selectionStart ?? text.length;
    const ctx = getMentionContext(text, caret);
    const atPos = ctx.atPos >= 0 ? ctx.atPos : caret;
    const before = text.slice(0, atPos);
    const after = text.slice(caret);
    const insert = `@${doc.label} `;
    const next = before + insert + after;
    setText(next);
    setMentionOpen(false);
    setMentionQuery("");
    onMentionPick?.(doc.id);
    requestAnimationFrame(() => {
      const el = taRef.current;
      if (!el) return;
      const pos = (before + insert).length;
      el.focus();
      el.setSelectionRange(pos, pos);
      autoResize(el);
    });
  };

  const submit = () => {
    const t = text.trim();
    if ((!t && images.length === 0) || streaming || disabled) return;
    if (images.length > 0 && !canVision) {
      setImgError("当前模型不支持图片，请切换 qwen3-vl-flash 或 deepseek-v4-flash-vision-exp");
      return;
    }
    onSend(t, images);
    setText("");
    setImages([]);
    setImgError(null);
    setMentionOpen(false);
    // 关键修复：setText("") 是异步的，此刻 DOM 里 textarea 仍是旧的长文本。
    // 必须等到下一帧 React 真正清空后再测量，否则会按旧内容算出高高度并写进
    // style.height，随后值被清空、内联高度却残留，导致发送后会话框卡在
    // 「空白却很高、无法缩回原尺寸」的状态。
    requestAnimationFrame(() => {
      const ta = taRef.current;
      if (!ta) return;
      // 保留用户手动拖拽的高度；否则回到 2 行自适应
      if (userHeight != null) ta.style.height = `${userHeight}px`;
      else {
        ta.style.height = "auto";
        autoResize(ta);
      }
    });
  };

  /** 输入自适应高度：默认 2 行；内容超过约 3 行后随内容增高，最高约 10 行后内部滚动。
   *  一旦用户手动拖拽过（userHeight != null），则锁定为该高度，不再自动伸缩。 */
  const autoResize = (el: HTMLTextAreaElement) => {
    if (userHeight != null) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(Math.max(el.scrollHeight, INPUT_MIN_H), INPUT_MAX_H)}px`;
  };

  /** 右下角拖拽手柄：调整输入框高度（固定后不再随内容自动伸缩，双击恢复自适应） */
  const startResize = (e: React.MouseEvent) => {
    e.preventDefault();
    const ta = taRef.current;
    if (!ta) return;
    const startY = e.clientY;
    const startH = ta.offsetHeight;
    const onMove = (ev: MouseEvent) => {
      const next = Math.max(INPUT_MIN_H, Math.min(INPUT_MAX_H, startH + (ev.clientY - startY)));
      ta.style.height = `${next}px`;
      setUserHeight(next);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.userSelect = "";
      dragCleanupRef.current = null;
    };
    // L7 修复（2026-09-10）：拖拽中组件卸载（mouseup 永不触发）会留下 window 监听与
    // body.userSelect="none"（此后整个页面无法选中文本）。用 ref 记录清理函数，
    // 由下方 unmount effect 兜底调用。
    dragCleanupRef.current = onUp;
    document.body.style.userSelect = "none";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // L7：卸载兜底——拖拽进行中卸载时清理监听与 body 样式
  useEffect(() => () => dragCleanupRef.current?.(), []);

  const pickImages = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    if (!canVision) {
      setImgError("当前模型不支持图片，请切换 qwen3-vl-flash 或 deepseek-v4-flash-vision-exp");
      return;
    }
    const pending = Array.from(files);
    // L3 修复（2026-09-10）：先按大小/格式过滤出**有效**文件，再判上限。
    // 原实现用 pending.length 计数——被 continue 跳过的非法文件也占名额，
    // 使「9 张合法 + 1 张超限」的 10 张图被整体拒绝（实际有效图不足上限）。
    const valid = pending.filter((f) => {
      if (f.size > MAX_IMAGE_BYTES) {
        setImgError(`「${f.name}」超过 32MiB 上限，已跳过`);
        return false;
      }
      if (!/^image\/(jpeg|png|gif|webp)$/.test(f.type)) {
        setImgError(`「${f.name}」格式不支持（仅 JPEG/PNG/GIF/WebP）`);
        return false;
      }
      return true;
    });
    if (images.length + valid.length > MAX_IMAGES) {
      setImgError(
        `单次最多 ${MAX_IMAGES} 张图片（本次有效 ${valid.length} 张，当前已有 ${images.length} 张）`,
      );
      return;
    }
    for (const f of valid) {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          setImages((prev) => [...prev, reader.result as string]);
        }
      };
      reader.readAsDataURL(f);
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  const modelList = models ?? ["fake"];

  // 模型 / 学科下拉：点击外部 / Esc 关闭（自定义下拉，原生 select 无法跟随深色主题）
  useEffect(() => {
    if (!modelOpen && !discOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (modelRef.current && !modelRef.current.contains(e.target as Node)) setModelOpen(false);
      if (discRef.current && !discRef.current.contains(e.target as Node)) setDiscOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setModelOpen(false);
        setDiscOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      window.removeEventListener("keydown", onKey);
    };
  }, [modelOpen, discOpen]);

  // 技能 / 提问模板面板：点击空白处关闭（与模型 / 学科下拉交互一致）
  // 边界 = .composer-tool-anchor（含触发按钮与面板），在锚点内的点击交给各自的 toggle / 选项逻辑
  useEffect(() => {
    if (!skillsOpen && !wizardOpen) return;
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (skillAnchorRef.current && !skillAnchorRef.current.contains(t)) setSkillsOpen(false);
      if (wizardAnchorRef.current && !wizardAnchorRef.current.contains(t)) setWizardOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [skillsOpen, wizardOpen]);

  /** 技能 / 向导互斥：打开一个时自动关闭另一个（避免同时占用输入区上方） */
  const toggleSkills = () => {
    const nv = !skillsOpen;
    setSkillsOpen(nv);
    if (nv) setWizardOpen(false);
  };
  const toggleWizard = () => {
    const nv = !wizardOpen;
    setWizardOpen(nv);
    if (nv) setSkillsOpen(false);
  };

  return (
    <div className="chat-composer">
      {images.length > 0 && (
        <div className="chat-input-images">
          {images.map((src, i) => (
            <span key={i} className="chat-input-image">
              <img src={src} alt={`图片 ${i + 1}`} />
              <button
                type="button"
                className="chat-input-image-remove"
                aria-label={`移除图片 ${i + 1}`}
                onClick={() => setImages((prev) => prev.filter((_, j) => j !== i))}
              >
                <Icons.x size={11} strokeWidth={2} />
              </button>
            </span>
          ))}
        </div>
      )}
      {imgError && <div className="chat-input-error">{imgError}</div>}
      {enhErr && <div className="chat-input-error">{enhErr}</div>}

      {/* 提示词增强预览：原文 vs 增强后，确认后再应用（不直接替换输入，避免误覆盖） */}
      {enhRes && (
        <div className="enhance-panel">
          <div className="enhance-panel-head">
            <span className="enhance-panel-title">
              <Icons.sparkles size={13} />
              增强预览（确认后替换输入框内容）
            </span>
            <button type="button" className="enhance-close" onClick={closeEnhance} aria-label="关闭增强预览">
              <Icons.x size={14} />
            </button>
          </div>
          <div className="enhance-cols">
            <div className="enhance-orig">
              <div className="enhance-label">原文</div>
              <p>{enhRes.original}</p>
            </div>
            <div className="enhance-new">
              <div className="enhance-label">增强后</div>
              <p>{enhRes.enhanced}</p>
            </div>
          </div>
          <div className="enhance-actions">
            <button type="button" className="enhance-btn ghost" onClick={closeEnhance}>放弃</button>
            <button type="button" className="enhance-btn primary" onClick={applyEnhance}>使用增强</button>
          </div>
        </div>
      )}

      {/* 输入区（悬浮卡片主体）：textarea 自适应高度；@ 提及弹窗相对此容器定位 */}
      <div className="chat-input">
        <textarea
          ref={taRef}
          value={text}
          aria-label="消息输入"
          placeholder={disabled ? "请先选择或创建一个会话" : "输入问题，Enter 发送，Shift+Enter 换行；输入 @ 选择文档"}
          onChange={(e) => {
            const v = e.target.value;
            setText(v);
            autoResize(e.target);
            const caret = e.target.selectionStart ?? v.length;
            const ctx = getMentionContext(v, caret);
            if (ctx.active && (mentionDocs?.length ?? 0) > 0) {
              setMentionOpen(true);
              setMentionQuery(ctx.query);
              setMentionIndex(0);
            } else {
              setMentionOpen(false);
            }
          }}
          onKeyDown={(e) => {
            if (mentionOpen && filteredMentions.length > 0) {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setMentionIndex((i) => Math.min(filteredMentions.length - 1, i + 1));
                return;
              }
              if (e.key === "ArrowUp") {
                e.preventDefault();
                setMentionIndex((i) => Math.max(0, i - 1));
                return;
              }
              if (e.key === "Enter") {
                e.preventDefault();
                applyMention(filteredMentions[mentionIndex]);
                return;
              }
              if (e.key === "Escape") {
                e.preventDefault();
                setMentionOpen(false);
                return;
              }
            }
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          disabled={disabled}
        />
        {mentionOpen && filteredMentions.length > 0 && (
          <div className="mention-popup" role="listbox" aria-label="选择文档">
            <div className="mention-popup-head">选择文档（@提及）</div>
            {filteredMentions.map((d, i) => (
              <button
                key={d.id}
                type="button"
                role="option"
                aria-selected={i === mentionIndex}
                className={`mention-item ${i === mentionIndex ? "active" : ""}`}
                onMouseEnter={() => setMentionIndex(i)}
                onMouseDown={(e) => {
                  e.preventDefault();
                  applyMention(d);
                }}
              >
                <span className="mention-label">{d.label}</span>
                <span className="mention-name">{d.filename}</span>
              </button>
            ))}
          </div>
        )}

        {/* 右下角拖拽手柄：拖动调整输入框高度（固定后不再随内容自动伸缩） */}
        <span
          className="chat-input-resize"
          onMouseDown={startResize}
          title="拖动调整输入框高度"
          aria-hidden="true"
        />

        {/* 发送 / 停止：内嵌输入框右侧，独立悬浮的圆形图标按钮（仅图标，无文字） */}
        {streaming ? (
          <button
            type="button"
            className="composer-send stop"
            onClick={onStop}
            aria-label="停止生成"
            title="停止生成"
          >
            <Icons.stop size={15} />
          </button>
        ) : (
          <button
            type="button"
            className="composer-send"
            onClick={submit}
            disabled={disabled || (!text.trim() && images.length === 0)}
            aria-label="发送"
            title="发送（Enter）"
          >
            <Icons.send size={15} />
          </button>
        )}
      </div>

      {/* 技能面板 / 向导面板改为悬浮在各自触发按钮上方（见下方 composer-tools 内的 anchor），二者互斥 */}

      {/* 联网可用性预警（2026-09-12）：开关已开但能力不可用时**必须明示**——
          否则用户会误以为「联网了却不搜」，实际是当前模型不具备联网能力。
          2026-09-17：可手动关闭（×）；关闭按「会话+文案」指纹记忆，换模型/换会话会自动重新提示。
          2026-09-17 晚：文案不再错误归因（旧实现 reason 为空时会说"当前模型不支持"，
          而真实原因可能是"开关当时是关的"），且陈旧能力数据被指纹挡在门外。 */}
      {/* 场景可见性（2026-09-23）：常驻显示当前检索范围 ——
          此前界面上看不出"会话有没有绑文档"，于是用户会把"没绑文档 → 走了通用对话"
          误判成"文档约束失效"。复用 `.composer-warn` 样式（视觉一致，零 CSS 改动），
          用 `data-tone` 区分"状态说明"与"警告"。 */}
      {scopeHint && (
        <div className="composer-warn" role="status" data-tone={scopeHint.tone} key={scopeHint.key}>
          <Icons.docs size={13} />
          <span>{scopeHint.text}</span>
          {/* 可关闭（2026-10-07）：与 warns 同一套 `.composer-warn-close`（零新 CSS）；
              关闭按内容 key 记忆，范围变化后自动重现（见 dismissScopeHint 注释）。 */}
          <button
            type="button"
            className="composer-warn-close"
            onClick={() => dismissScopeHint(scopeHint.key)}
            aria-label="关闭此提示"
            title="关闭此提示"
          >
            <Icons.x size={12} />
          </button>
        </div>
      )}

      {warns.map((w) => (
        <div className="composer-warn" role="status" key={w.key}>
          {w.kind === "web" ? <Icons.globe size={13} /> : <Icons.alert size={13} />}
          <span>{w.text}</span>
          <button
            type="button"
            className="composer-warn-close"
            onClick={() => dismissWarn(w.key)}
            aria-label="关闭此提示"
            title="关闭此提示"
          >
            <Icons.x size={12} />
          </button>
        </div>
      ))}

      {/* 底部工具条：左侧为功能入口（图片 / 技能 / 联网 / 模型 / 学科 / 余额），右侧为发送 */}
      <div className="composer-foot">
        <div className="composer-tools">
          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/gif,image/webp"
            multiple
            style={{ display: "none" }}
            onChange={(e) => pickImages(e.target.files)}
          />
          <button
            type="button"
            className="composer-icon-btn"
            disabled={!canVision || disabled || streaming}
            title={
              canVision
                ? "添加图片（多模态模型）"
                : "当前模型不支持图片，请切换 qwen3-vl-flash 或 deepseek-v4-flash-vision-exp"
            }
            onClick={() => fileRef.current?.click()}
          >
            <Icons.image size={16} />
          </button>

          {/* ★ P2（2026-10-03）：上下文用量悬浮条（参考 WorkBuddy）。
              常驻显示「38.0% · 380.4K / 1000.0K 上下文已使用」，点击展开分项面板。 */}
          <CtxUsageBar
            peak={ctxUsage?.peak}
            budget={ctxUsage?.budget}
            warnAt={ctxUsage?.warnAt}
            breakdown={ctxUsage?.breakdown}
            variant="ring"
          />

          {skills && skills.length > 0 && (
            <>
              <span className="composer-tool-anchor" ref={skillAnchorRef}>
                <button
                  type="button"
                  className={`composer-tool ${skillsOpen ? "on" : ""}`}
                  onClick={toggleSkills}
                  aria-expanded={skillsOpen}
                  ref={skillBtnRef}
                  title="展开 / 收起命令列表"
                >
                  <Icons.zap size={14} />
                  <span>命令</span>
                  {/* 2026-10-07：工具条按钮去计数徽标（用户反馈：与「提问模板」保持同等干净）；
                      数量仍可在展开面板头部查看（skill-panel-count 保留）。 */}
                  <span className="composer-tool-chevron" aria-hidden="true">
                    {skillsOpen ? <Icons.chevronDown size={12} /> : <Icons.chevronRight size={12} />}
                  </span>
                </button>
                {skillsOpen && (
                  <div className="skill-panel">
                    <div className="skill-panel-head">
                      <span className="skill-panel-title">
                        <Icons.zap size={13} />
                        命令
                        <span className="skill-panel-count">{skills.length}</span>
                      </span>
                      <button
                        type="button"
                        className="skill-panel-close"
                        onClick={() => setSkillsOpen(false)}
                        aria-label="收起命令面板"
                        title="收起命令面板"
                      >
                        <Icons.x size={13} />
                      </button>
                    </div>
                    {/* ★ P2（2026-10-03）：本地过滤。24 个命令平铺靠滚动找很难，
                        且**过滤匹配全部别名** ⇒ 英文别名因此变得**可发现**
                        （原本前端只用了 `commands[0]`，别名在 UI 里根本不存在）。 */}
                    <div className="skill-filter">
                      <input
                        type="text"
                        className="skill-filter-input"
                        value={skillQuery}
                        onChange={(e) => setSkillQuery(e.target.value)}
                        placeholder="搜索命令（中文或英文别名）"
                        aria-label="搜索命令"
                      />
                      {skillQuery && (
                        <button
                          type="button"
                          className="skill-filter-clear"
                          onClick={() => setSkillQuery("")}
                          aria-label="清除搜索"
                          title="清除搜索"
                        >
                          <Icons.x size={12} />
                        </button>
                      )}
                    </div>
                    <div className="skill-chips">
                      {filteredSkills.map((sk) => {
                        // ★ P1（2026-10-03）：**展示全部命令**（此前只取 commands[0]，
                        //   ⇒ 12 个技能的**英文别名在 UI 里完全不可见**）。
                        const cmds = sk.commands.length ? sk.commands : [sk.name];
                        const cmd = cmds[0];
                        const alias = cmds.slice(1);
                        const reason = sk.feature ? FEATURE_LABEL[sk.feature] ?? sk.feature : "高级技能";
                        const base = sk.enabled
                          ? sk.description
                          : `${sk.description}（需在设置中启用「${reason}」）`;
                        const tip = alias.length ? `${base}\n别名：${alias.join(" ")}` : base;
                        return (
                          <button
                            key={sk.name}
                            type="button"
                            className={`skill-chip ${sk.enabled ? "" : "disabled"}`}
                            disabled={disabled || streaming || !sk.enabled}
                            title={tip}
                            style={{ borderLeftColor: CATEGORY_COLOR[sk.category ?? ""] ?? "var(--muted)" }}
                            onClick={() => {
                              // ★★ P0-3（2026-10-03）：命令**必须位于文本开头**才会被后端路由
                              //   （`SkillSpec.matches` 要求以命令开头）。此前是**追加到末尾**
                              //   ⇒ 输入框已有内容时，点了命令**完全不生效且无提示**。
                              //   ⇒ 改为**插到最前面**。
                              setText((prev) => (prev.trim() ? `${cmd} ${prev.trim()}` : `${cmd} `));
                              setSkillsOpen(false);
                              setMentionOpen(false);
                              if (taRef.current) {
                                taRef.current.focus();
                                // 光标落在**命令之后**（用户接下来要写参数）
                                const pos = cmd.length + 1;
                                taRef.current.setSelectionRange(pos, pos);
                              }
                            }}
                          >
                            <span className="skill-icon" aria-hidden="true">{sk.icon ?? <Icons.zap size={13} />}</span>
                            <span className="skill-cmd">{cmd}</span>
                          </button>
                        );
                      })}
                      {filteredSkills.length === 0 && (
                        <div className="skill-empty">没有匹配的命令</div>
                      )}
                    </div>
                  </div>
                )}
              </span>

              <span className="composer-tool-anchor" ref={wizardAnchorRef}>
                <button
                  type="button"
                  className={`composer-tool ${wizardOpen ? "on" : ""}`}
                  onClick={toggleWizard}
                  aria-expanded={wizardOpen}
                  title="不知道怎么问？试试论文通用提问模板（理解 / 总结 / 评估 / 复现 / 写作）"
                >
                  <Icons.sparkles size={14} />
                  <span>提问模板</span>
                </button>
                {wizardOpen && (
                  <PromptWizard
                    onPick={(tpl) => {
                      // 模板是完整问句（非 /命令）：空输入直接填入，已有内容则另起一段追加
                      setText((prev) => (prev.trim() ? `${prev.trimEnd()}\n\n${tpl}` : tpl));
                      setWizardOpen(false);
                      setMentionOpen(false);
                      if (taRef.current) {
                        taRef.current.focus();
                        // 光标落到末尾，方便直接续写
                        const end = taRef.current.value.length;
                        taRef.current.setSelectionRange(end, end);
                      }
                    }}
                    onClose={() => setWizardOpen(false)}
                  />
                )}
              </span>
            </>
          )}

          {/* 提示词增强：把口语输入改写为更明确的任务描述（后端单轮元任务；需真实模型） */}
          <button
            type="button"
            className="composer-tool"
            onClick={runEnhance}
            disabled={disabled || enhancing}
            title="增强提示词：把当前输入改写为更清晰、具体的任务描述（需在设置页配置真实模型）"
          >
            <Icons.sparkles size={14} />
            <span>{enhancing ? "增强中…" : "增强"}</span>
          </button>

          {/* 联网开关（2026-09-12）：可用性可能因「模型不支持原生联网」或「doc-bound 回落」而不成立，
              此时按钮呈警示态，并在工具条上方给出明确原因（见 composer-warn）。 */}
          <button
            type="button"
            className={`composer-tool toggle ${webOn ? "on" : ""} ${
              webOn && webCapability && !webCapability.available ? "warn" : ""
            }`}
            onClick={onWebToggle}
            role="switch"
            aria-checked={webOn ?? false}
            disabled={disabled}
            title={
              webOn && webCapability && !webCapability.available
                ? `联网已开启但当前不可用：${webCapability.reason}`
                : "联网搜索：开启后模型可按需联网核实（能否联网取决于当前模型；不支持时会如实说明）"
            }
          >
            <Icons.globe size={14} />
            <span>联网{webOn && webCapability && !webCapability.available ? " ⚠" : ""}</span>
            <span className="web-toggle-track" aria-hidden="true"><span className="web-toggle-thumb" /></span>
          </button>

          {/* 学科视角：会话级解读框架（与模型/联网并列，替代设置页的学科下拉）。
              2026-09-01：原生 select 的展开项是系统默认样式（白底、不跟随深色主题、无圆角阴影），
              改为与「模型选择」完全同一套自定义弹层（composer-model-* 类复用），样式 / 动画 / 交互对齐。 */}
          <div className="composer-model" ref={discRef}>
            <button
              type="button"
              className={`composer-model-trigger ${discOpen ? "on" : ""}`}
              onClick={() => !disabled && setDiscOpen((v) => !v)}
              aria-haspopup="listbox"
              aria-expanded={discOpen}
              disabled={disabled}
              title="选择学科视角，影响技能的解读框架与证据标准"
            >
              <Icons.layers size={14} />
              <span className="composer-model-label">
                {DISCIPLINES.find((d) => d.value === (discipline ?? "general"))?.label ?? "通用"}
              </span>
              <span className={`composer-model-chevron ${discOpen ? "open" : ""}`} aria-hidden="true">
                <Icons.chevronDown size={12} />
              </span>
            </button>
            {discOpen && (
              <div className="composer-model-menu" role="listbox" aria-label="选择学科视角">
                {DISCIPLINES.map((d) => {
                  const active = (discipline ?? "general") === d.value;
                  return (
                    <button
                      key={d.value}
                      type="button"
                      role="option"
                      aria-selected={active}
                      className={`composer-model-item ${active ? "active" : ""}`}
                      onClick={() => {
                        onDisciplineChange?.(d.value);
                        setDiscOpen(false);
                      }}
                    >
                      <span className="composer-model-item-name">{d.label}</span>
                      {active && <Icons.check size={13} />}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* 模型：自定义下拉（原生 select 展开项无法跟随深色主题，改为可主题化的弹层 + 动画） */}
          <div className="composer-model" ref={modelRef}>
            <button
              type="button"
              className={`composer-model-trigger ${modelOpen ? "on" : ""}`}
              onClick={() => !disabled && setModelOpen((v) => !v)}
              aria-haspopup="listbox"
              aria-expanded={modelOpen}
              disabled={disabled}
              title="切换当前会话使用的模型"
            >
              <Icons.cpu size={14} />
              <span className="composer-model-label model-fixed">{currentModel ?? "fake"}</span>
              <span className={`composer-model-chevron ${modelOpen ? "open" : ""}`} aria-hidden="true">
                <Icons.chevronDown size={12} />
              </span>
            </button>
            {modelOpen && (
              <div className="composer-model-menu" role="listbox" aria-label="选择模型">
                {modelList.map((m) => (
                  <button
                    key={m}
                    type="button"
                    role="option"
                    aria-selected={m === currentModel}
                    className={`composer-model-item ${m === currentModel ? "active" : ""}`}
                    onClick={() => {
                      onModelChange?.(m);
                      setModelOpen(false);
                    }}
                  >
                    <span className="composer-model-item-name">{m === "fake" ? "fake（演示）" : m}</span>
                    {m === currentModel && <Icons.check size={13} />}
                  </button>
                ))}
              </div>
            )}
          </div>
          {/* F4 徽标与余额条已下线（2026-10-07 用户反馈：压缩工具条横向空间，保证一排放下）。
              props（modelSource / balances）保留以兼容调用方，但不再渲染。 */}
        </div>
      </div>
    </div>
  );
}
