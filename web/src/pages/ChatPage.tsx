import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { DocScopeOut, AppSettings, AssistantMessage, Citation, DocumentOut, InventoryOut, NetworkCapabilityOut, ProviderBalance, SessionOut, TraceListItem, TraceOut, WorkspaceMode, WorkspaceOut } from "../lib/types";
import { isWorkspace } from "../lib/types";
import { useStreamingChat } from "../hooks/useStreamingChat";
import { MessageList } from "../components/MessageList";
import { ChatInput } from "../components/ChatInput";
import { useSettings } from "../store/settings";
// T-08（2026-09-29）：`docLabel` 改从**真正归属处**直引。
// 原先写 `import { DocumentPreviewPanel, docLabel } from "../components/DocumentPreviewPanel"`，
// 而面板里只有一句 `export { docLabel } from "../lib/docLabel"` —— 从它引入会把**整个面板模块**
// （连带 `pdfjs-dist`）拉进 ChatPage 的静态图，使"面板下沉为 lazy"失效。
import { docLabel } from "../lib/docLabel";
import { PdfReader } from "../components/PdfReader";
import { TraceTimeline } from "../components/TraceTimeline";
import { MemoryPanel } from "../components/MemoryPanel";
import { Icons } from "../lib/icons";
import { capKeyOf } from "../lib/networkWarn";
import { toErrorMessage } from "../lib/error";

/**
 * T-08（2026-09-29）：两个**重组件**下沉为按需加载 —— 它们是 pdfjs / docx-preview
 * 在首屏模块图里的**唯一入口**，不 lazy 则分包只"拆文件"、首屏字节不变
 * （实测：拆分后 `index.html` 仍 `modulepreload` pdfjs 478 kB + docx-preview 174 kB）。
 * 两者本就**条件渲染**（右栏面板 / 改稿工作台），故 lazy 与其既有边界天然一致。
 */
const DocumentPreviewPanel = lazy(() =>
  import("../components/DocumentPreviewPanel").then((m) => ({ default: m.DocumentPreviewPanel })));
const RevisionWorkbench = lazy(() =>
  import("../components/RevisionWorkbench").then((m) => ({ default: m.RevisionWorkbench })));

/** 懒加载占位：复用既有 `.muted` 令牌类，不新增样式令牌（与 App.tsx 的 ViewLoading 同款） */
function PanelLoading() {
  return <div className="muted" style={{ margin: 12 }}>加载中…</div>;
}

/** D6（2026-09-21）：深链支持 —— 从 `#trace=<id>` 读出要定位的轨迹 id。
 *  目的：让「某一轮答错了」可以被**分享与回看**（此前弹窗固定取 list[0]，只能看最近一次）。 */
function readTraceHash(): string | null {
  try {
    const h = (window.location.hash || "").replace(/^#/, "");
    const m = /(?:^|&)trace=([^&]+)/.exec(h);
    return m ? decodeURIComponent(m[1]) : null;
  } catch {
    return null;
  }
}

/** 历史加载骨架（shimmer，见 styles.css .sk-*） */
function SkeletonThread() {
  return (
    <div className="message-list" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <div className="sk-msg" key={i}>
          <div className="sk-avatar sk-shimmer" />
          <div className="sk-body">
            <div className="sk-line sk-shimmer w80" />
            <div className="sk-line sk-shimmer w60" />
            <div className="sk-line sk-shimmer w40" />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * 问答页（画布 6:1）三栏：
 * 左栏会话列表已上移 App 侧栏；本页负责「对话主区(700) + 文档预览面板(460)」。
 * - 头部：[+ 新对话](App 侧栏) / 标题 / 模型选择 / 推理轨迹 / 联网开关
 * - 右栏：DocumentPreviewPanel（多文档 Tab + PDF 预览 + 导出）—— 依据 docs/前端文档预览功能设计方案.md v1.1
 * - 引用卡点击 → 预览面板内嵌 iframe 跳转（复用 PdfReader 弹窗能力）
 */
export function ChatPage({
  sessionId,
  sessions,
  onSessionsChange,
}: {
  sessionId: number | null;
  sessions: SessionOut[];
  onSessionsChange: React.Dispatch<React.SetStateAction<SessionOut[]>>;
}) {
  const [allDocs, setAllDocs] = useState<DocumentOut[]>([]);
  const [docIds, setDocIds] = useState<number[]>([]);
  /** B2：工作间信息（doc_ids 源自底层文献集、draft_doc_id 主文献）；普通会话恒为 null */
  const [wsInfo, setWsInfo] = useState<WorkspaceOut | null>(null);
  /** B2：主文献操作失败提示（服务端校验不通过 / 网络异常） */
  const [wsError, setWsError] = useState<string | null>(null);
  /** N6（2026-09-10）：流进行中发送被丢弃时的可见提示（原先静默丢输入） */
  const [sendNotice, setSendNotice] = useState<string | null>(null);
  /** B2 审查加固：主文献 PATCH 进行中 —— 禁用星标，防止快速连点产生并发 PATCH 竞态 */
  const [draftBusy, setDraftBusy] = useState(false);
  /** C1：工作间模式切换（研读/改稿）进行中 */
  const [modeBusy, setModeBusy] = useState(false);

  /** C1：切换工作间模式 —— 与主文献共用 PATCH /workspaces/{sid} 端点；同步 wsInfo 与左栏 config */
  const switchMode = useCallback(
    async (m: WorkspaceMode) => {
      if (sessionId == null || modeBusy || wsInfo?.mode === m) return;
      setWsError(null);
      setModeBusy(true);
      try {
        const ws = await api.updateWorkspace(sessionId, { mode: m });
        setWsInfo(ws);
        onSessionsChange((prev) =>
          prev.map((x) =>
            x.id === sessionId ? { ...x, config: { ...x.config, mode: m } } : x,
          ),
        );
      } catch (e: unknown) {
        setWsError(toErrorMessage(e) || "切换模式失败，请确认后端服务正常");
      } finally {
        setModeBusy(false);
      }
    },
    [sessionId, modeBusy, wsInfo, onSessionsChange],
  );
  const { settings } = useSettings();
  const [chatModel, setChatModel] = useState("fake");
  /** F4：当前模型生效来源——session=本会话显式指定；global=继承全局默认（设置页） */
  const [modelSource, setModelSource] = useState<"session" | "global">("global");
  const [balances, setBalances] = useState<Record<string, ProviderBalance> | null>(null);
  const [webOn, setWebOn] = useState(false);
  /** 联网可用性（2026-09-12）：联网是模型原生能力，且 doc-bound 会回落模型 →
   *  「开关已开」≠「能力可用」。用于输入区预警，避免用户误以为"开了联网却不搜"。
   *
   *  2026-09-17：数据与**请求指纹**绑定（`key` = 发起请求时的输入）。
   *  为什么必须绑：`webOn` 是乐观更新（点击即变），而这份数据来自上一次请求 ——
   *  两者短暂不同源就会让提示闪现（实测约 50ms）。判定逻辑统一走
   *  `lib/networkWarn.composerWarns`，指纹不符的数据一律不用于预警。 */
  const [webCap, setWebCap] = useState<{ data: NetworkCapabilityOut; key: string } | null>(null);
  /** 能力数据重取信号：PATCH 落库成功后自增，让 effect 再拉一次（拿到与持久状态一致的结论）。
   *  为什么需要：开关/模型是**乐观更新**，而拉取能力的请求可能抢在 PATCH 落库之前，
   *  后端会按旧状态回答 —— 判定侧用 `cap.enabled === webOn` 挡掉这类数据，
   *  这里负责把正确的那份取回来（否则预警会一直不显示）。 */
  const [webCapEpoch, setWebCapEpoch] = useState(0);
  /** 场景可见性（2026-09-23）：本会话的检索范围（0 / 1 / N 篇），供输入框上方常驻提示。
   *  它是**会话级**数据（随会话与关联文档变化），不像 webCap 受乐观更新影响 ——
   *  故不做认证指纹与 epoch，取失败就静默不显示（宁可不显示，也不显示错的状态）。 */
  const [docScope, setDocScope] = useState<DocScopeOut | null>(null);
  /** D6：本会话的轨迹列表（弹窗内可切换，不再只能看最近一次） */
  const [traceList, setTraceList] = useState<TraceListItem[]>([]);
  const [discipline, setDiscipline] = useState("general");
  const [skills, setSkills] = useState<InventoryOut["skills"] | null>(null);
  const [readerCitation, setReaderCitation] = useState<Citation | null>(null);
  const [trace, setTrace] = useState<TraceOut | null>(null);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState<string | null>(null);
  /** E1：messageId → 当前评分（1 / -1）。刷新后由 /feedback/summary 回填，按钮才不会"全灭"。 */
  const [ratedMap, setRatedMap] = useState<Record<number, number>>({});
  /** E2：导出评测样本的进行中与结果提示（失败要能看见，不能静默） */
  const [exporting, setExporting] = useState(false);
  const [exportNote, setExportNote] = useState<string | null>(null);
  /** F8：记忆管理面板开关（复用设置页 MemoryPanel 组件，零改造） */
  const [memOpen, setMemOpen] = useState(false);
  /** F6：本轮自动识别的学科（auto 模式下从 done 事件回传），用于顶栏反馈 chip */
  const [detected, setDetected] = useState<{ key: string; label: string } | null>(null);
  /** F5：本会话关联的文献集名称（顶栏徽标展示用），随 collectionId 变化按需拉取 */
  const [collectionName, setCollectionName] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  // O-08（2026-10-01）：MessageList 已 memo 化 ⇒ 传给它的回调/数组 props 必须稳定引用，
  // 否则 memo 静默失效（每次流式增量都会重渲全部消息）。
  const handleBottomRef = useCallback((el: HTMLDivElement | null) => {
    bottomRef.current = el;
  }, []);
  /** F8：记忆弹窗容器（打开时聚焦，便于键盘/读屏用户） */
  const memRef = useRef<HTMLDivElement | null>(null);
  const { items, streaming, loadingHistory, historyError, send, stop, regenerate } = useStreamingChat(sessionId);

  /** ★ P2（2026-10-03）：**最后一条** assistant 消息的上下文用量，供「上下文用量」面板。
   *  两条来源合并：实时取 done、历史（刷新后）取 ctx —— 前者优先。
   *  ⚠️ 只在"这条消息确实带数据"时才返回；全都没有则返回 null（面板不渲染）。 */
  const lastCtxUsage = useMemo(() => {
    for (let i = items.length - 1; i >= 0; i--) {
      const it = items[i];
      if (it.role !== "assistant") continue;
      const done = it.done;
      const ctx = it.ctx;
      if (!done && !ctx) continue;
      return {
        peak: done?.peak_prompt_tokens ?? ctx?.peak,
        budget: done?.max_tokens_budget ?? ctx?.budget,
        warnAt: done?.compact_trigger_tokens ?? ctx?.warnAt,
        breakdown: done?.context_breakdown ?? ctx?.breakdown,
      };
    }
    return null;
  }, [items]);

  // 发送完成后刷新会话列表：首轮对话后后端会生成主题标题，侧栏需同步显示（不再显示默认「新会话」）
  // 2026-09-07：@ 提及收集——onMentionPick 时记录精确 doc id，随消息发送（结构化提及），
  // 发送后清空。后端据此把检索/技能限缩到被提及的文档（不再依赖序号文本解析）。
  // handleMentionPick 定义在 toggleDoc 声明之后（避免 use-before-declaration）。
  const mentionPickedRef = useRef<number[]>([]);
  // 2026-09-23 修复（建议③）：上面那个 ref 需要 **state 镜像**。
  //
  // 为什么必须加：ref 变化**不触发重渲染**，于是「本轮 @ 了哪些」这个状态
  // 无法被输入框上方的「当前检索范围」提示消费 —— 结果用户 `@` 某篇后，
  // 后端**确实聚焦到该篇**（`_effective_doc_scope` 的 @ 聚焦语义），
  // 而提示条仍显示「检索全部 N 篇文档」→ **提示与真实行为相反**。
  // 那比没有提示更糟：用户会基于错误提示下结论。
  //
  // 分工：ref 仍是「发送时取值的唯一来源」（避免 state 闭包陈旧导致漏发 id）；
  // state 只用于**渲染**。两者在同一处一起更新，不会漂移。
  const [mentionPicked, setMentionPicked] = useState<number[]>([]);
  const wrapSend = useCallback(
    async (t: string, images?: string[]) => {
      const ids = [...mentionPickedRef.current];
      mentionPickedRef.current = [];
      // 发送即清空 @ 状态：下一轮不再聚焦（与后端「mention 只作用于当轮」一致）
      setMentionPicked([]);
      // N6（2026-09-10）：send 返回 false = 流进行中发送被丢弃——给出可见提示，
      // 原先静默丢弃让用户以为「发送无效」。
      const ok = await send(t, undefined, images, ids);
      setSendNotice(ok ? null : "回复生成中，请稍候再发送");
      try {
        onSessionsChange(await api.listSessions());
      } catch {
        /* 列表刷新失败不影响对话 */
      }
    },
    [send, onSessionsChange],
  );

  const refreshDocs = useCallback(async () => {
    setAllDocs(await api.listDocuments());
  }, []);

  // 全部可选模型（fake + 各厂商模型），用于模型选择器
  const allModels = useMemo(() => {
    if (!settings) return ["fake"];
    const list = ["fake"];
    for (const p of Object.keys(settings.providers)) {
      for (const m of settings.providers[p].available_models || []) list.push(m);
    }
    return list;
  }, [settings]);

  // 最后一条 assistant 消息 id：仅它显示「重新生成」按钮（受 applyEvent 作用域约束，只能重跑最后一轮）
  const lastAssistantId = useMemo(() => {
    for (let i = items.length - 1; i >= 0; i--) {
      if (items[i].role === "assistant") return items[i].id;
    }
    return undefined;
  }, [items]);

  const normModel = useCallback(
    (v: string): string => {
      if (allModels.includes(v)) return v;
      const am = settings?.providers[v]?.available_models;
      if (am && am.length) return am[0];
      return "fake";
    },
    [allModels, settings],
  );

  useEffect(() => {
    refreshDocs();
    api.getBalances().then(setBalances).catch(() => setBalances(null));
  }, [refreshDocs]);

  // 切换会话时，从会话 config 载入已关联文档、联网开关与学科视角。
  // B2：工作间会话的文献存在底层文献集（config 无 doc_ids）→ 改从 GET /workspaces 拉取。
  //     单一数据源是文献集：若把 doc_ids 冗余进 config，B1 的「加入工作间」会产生同步问题。
  // 2026-09-08 修复：① 原依赖 `sessions`（数组引用，每发一条消息就换新）→ effect 频繁重跑；
  //   ② 工作间分支开头无条件 `setWsInfo(null)` → isWorkbench(:557) 瞬时翻假，
  //      RevisionWorkbench 被卸载重建，编辑态/高亮/滚动全部丢失（改稿模式每发一条消息必现）。
  //   改为：仅在「会话 id 或该会话 config 内容」变化时重跑；拉取期间保留旧 wsInfo 不闪断。
  const sessionCfgKey = (() => {
    const cur = sessions.find((x) => x.id === sessionId);
    return cur ? JSON.stringify(cur.config ?? {}) : "";
  })();

  useEffect(() => {
    const s = sessions.find((x) => x.id === sessionId);
    if (!s) return;
    setWebOn(Boolean(s.config.enable_network));
    setDiscipline((s.config.discipline as string) || "general");
    if (isWorkspace(s)) {
      let alive = true;
      api
        .getWorkspace(s.id)
        .then((ws) => {
          if (!alive) return;
          setWsInfo(ws);
          // 主文献放数组末尾（activeDocId 取最后一个）→ 进入工作间默认激活主文献
          const others = ws.doc_ids.filter((d) => d !== ws.draft_doc_id);
          setDocIds(ws.draft_doc_id != null ? [...others, ws.draft_doc_id] : ws.doc_ids);
        })
        .catch(() => {
          if (alive) setDocIds([]); // 拉取失败：文件区置空，不打断对话
        });
      return () => {
        alive = false;
      };
    }
    setWsInfo(null);
    setDocIds((s.config.doc_ids as number[]) || []);
    // 依赖稳定的 config 内容键，而非 sessions 数组引用（见上方 2026-09-08 修复说明）
  }, [sessionId, sessionCfgKey]);

  /** B2：设置/清除工作间主文献（服务端校验「存在且在文献集内」；本地同步 wsInfo 与左栏 config） */
  const setDraftDoc = useCallback(
    async (id: number | null) => {
      if (sessionId == null || draftBusy) return;
      setWsError(null);
      setDraftBusy(true);
      try {
        const ws = await api.updateWorkspace(sessionId, { draft_doc_id: id });
        setWsInfo(ws);
        onSessionsChange((prev) =>
          prev.map((x) =>
            x.id === sessionId ? { ...x, config: { ...x.config, draft_doc_id: id } } : x,
          ),
        );
        const others = ws.doc_ids.filter((d) => d !== ws.draft_doc_id);
        setDocIds(ws.draft_doc_id != null ? [...others, ws.draft_doc_id] : ws.doc_ids);
      } catch (e: unknown) {
        setWsError(toErrorMessage(e) || "设置主文献失败，请确认后端服务正常");
      } finally {
        setDraftBusy(false);
      }
    },
    [sessionId, draftBusy, onSessionsChange],
  );

  // 同步模型选择器到当前会话；会话未显式指定模型时回落到全局默认模型（与设置页一致）。
  // F4：同时判定生效来源，供输入区标注「会话覆盖 / 全局默认」。
  useEffect(() => {
    const s = sessions.find((x) => x.id === sessionId);
    if (s) {
      const explicit = (s.config.model as string) || "";
      const base = explicit || settings?.default_model || "fake";
      setChatModel(normModel(base));
      setModelSource(explicit ? "session" : "global");
    }
  }, [sessionId, sessions, normModel, settings]);

  // 载入技能清单（GET /settings/inventory），驱动对话命令栏；失败不阻断对话
  useEffect(() => {
    let alive = true;
    api.getInventory()
      .then((inv) => { if (alive) setSkills(inv.skills ?? []); })
      .catch(() => { /* 清单加载失败则命令栏不显示，不影响对话 */ });
    return () => { alive = false; };
  }, []);

  // F6：从最新一条 assistant 消息的 done 事件读取学科识别结果，驱动顶栏「已识别为：X」反馈。
  // 仅当本轮为 auto 自动识别、且识别结果非 general 时才提示；手动指定学科不提示。
  useEffect(() => {
    // 从末尾向前查找：流式期间每个 token 都会触发本 effect，
    // 原先 [...items].reverse() 每次拷贝整个会话，长会话累积为 O(n²)。
    let last: AssistantMessage | undefined;
    for (let i = items.length - 1; i >= 0; i--) {
      const m = items[i];
      if (m.role === "assistant" && m.done) {
        last = m;
        break;
      }
    }
    if (last?.done?.discipline_auto && last.done.discipline && last.done.discipline !== "general") {
      setDetected({ key: last.done.discipline, label: last.done.discipline_label ?? last.done.discipline });
    } else {
      setDetected(null);
    }
  }, [items]);

  // F8：记忆弹窗——Esc 关闭 + 打开时锁背景滚动 + 聚焦弹窗，避免键盘用户被困与背景误滚动
  useEffect(() => {
    if (!memOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMemOpen(false);
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    memRef.current?.focus();
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [memOpen]);

  // E1：会话切换/加载后回填已有评分 —— 否则刷新后按钮全灭，用户会以为"反馈没记上"。
  useEffect(() => {
    if (sessionId == null) {
      setRatedMap({});
      return;
    }
    let alive = true;
    api
      .feedbackSummary(sessionId)
      .then((s) => {
        if (alive) setRatedMap(s.rated_message_ids ?? {});
      })
      .catch(() => {
        /* 反馈回填失败不阻断对话：最坏结果是按钮不高亮，不影响使用 */
      });
    return () => {
      alive = false;
    };
  }, [sessionId]);

  /** E1：提交评分。失败只在控制台留痕 —— 反馈是"顺手一点"的交互，
   *  弹错会打断阅读；但**按钮保持不高亮**，用户自然能看出没记上。 */
  const rateMessage = useCallback(
    async (messageId: number, rating: 1 | -1) => {
      if (sessionId == null || !Number.isFinite(messageId) || messageId <= 0) return;
      try {
        await api.submitFeedback({ session_id: sessionId, message_id: messageId, rating });
        setRatedMap((prev) => ({ ...prev, [messageId]: rating }));
      } catch (e) {
        console.warn("[feedback] 提交失败", e);
      }
    },
    [sessionId],
  );

  /** E2：把当前轨迹导出为评测样本（jsonl）。
   *  ⚠️ 导出的 `gold` 是空的 —— 检索命中只作候选，需人工确认后才能进评测集。 */
  const exportCase = useCallback(async () => {
    if (!trace?.trace_id) return;
    setExporting(true);
    setExportNote(null);
    try {
      const c = await api.exportTraceCase(trace.trace_id);
      const line = JSON.stringify(c, null, 0);
      const blob = new Blob([line + "\n"], { type: "application/x-ndjson" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `case-${trace.trace_id}.jsonl`;
      a.click();
      URL.revokeObjectURL(url);
      setExportNote(
        c.candidates.length
          ? `已导出（${c.candidates.length} 个候选块，gold 需人工补）`
          : "已导出（该轮未检索到文档）",
      );
    } catch (e) {
      setExportNote(`导出失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setExporting(false);
    }
  }, [trace?.trace_id]);

  const changeModel = useCallback(
    async (v: string) => {
      if (sessionId == null) return;
      setChatModel(v);
      // F4：一旦在本会话显式选过模型，来源即变为「会话覆盖」（乐观更新）
      setModelSource("session");
      const s = sessions.find((x) => x.id === sessionId);
      const cfg = { ...(s?.config || {}), model: v };
      try {
        await api.updateSession(sessionId, cfg);
        onSessionsChange((prev) => prev.map((x) => (x.id === sessionId ? { ...x, config: cfg } : x)));
        // 落库成功后再触发一次能力重取：切换模型会改变 effective_model（能力结论随之变）
        setWebCapEpoch((e) => e + 1);
      } catch {
        /* 模型切换失败时保留 UI 状态，不阻断对话 */
      }
    },
    [sessionId, sessions, onSessionsChange],
  );

  /** 联网开关：会话 config.enable_network 即时持久化（场景 8，无需退回设置页） */
  const toggleWeb = useCallback(async () => {
    if (sessionId == null) return;
    const next = !webOn;
    setWebOn(next);
    const s = sessions.find((x) => x.id === sessionId);
    const cfg = { ...(s?.config || {}), enable_network: next };
    try {
      await api.updateSession(sessionId, cfg);
      onSessionsChange((prev) => prev.map((x) => (x.id === sessionId ? { ...x, config: cfg } : x)));
      // 落库成功后再触发一次能力重取（2026-09-17 实测补）：
      // 开关是**乐观更新**，而这里是异步 PATCH —— 拉取能力的请求可能抢在落库之前到达后端，
      // 于是后端仍按旧开关状态回 `enabled=false`，前端就会拿它配 `supports_native=false`
      // 显示一句**归因错误**的提示（实测复现）。判定侧已用 `cap.enabled === webOn` 挡住
      // 这类数据，这里则保证"正确的那份数据"最终一定能取到。
      setWebCapEpoch((e) => e + 1);
    } catch {
      /* 保存失败保留 UI，不阻断；此时预警会被一致性问题挡住（宁可不提示，也不提示错的） */
    }
  }, [sessionId, webOn, sessions, onSessionsChange]);

  /** 学科视角：会话 config.discipline 即时持久化（与模型/联网开关同构，场景级覆盖） */
  const changeDiscipline = useCallback(
    async (v: string) => {
      if (sessionId == null) return;
      setDiscipline(v);
      const s = sessions.find((x) => x.id === sessionId);
      const cfg = { ...(s?.config || {}), discipline: v };
      try {
        await api.updateSession(sessionId, cfg);
        onSessionsChange((prev) => prev.map((x) => (x.id === sessionId ? { ...x, config: cfg } : x)));
      } catch {
        /* 保存失败保留 UI，不阻断对话 */
      }
    },
    [sessionId, sessions, onSessionsChange],
  );

  const toggleDoc = useCallback(
    async (id: number, checked: boolean) => {
      if (sessionId == null) return;
      // 已关联则不重复追加（@提及/重复勾选幂等，修复 docIds 出现重复 id 的根因）
      const next = checked ? (docIds.includes(id) ? docIds : [...docIds, id]) : docIds.filter((d) => d !== id);
      setDocIds(next);
      const s = sessions.find((x) => x.id === sessionId);
      const cfg = { ...(s?.config || {}), doc_ids: next };
      try {
        await api.updateSession(sessionId, cfg);
        onSessionsChange((prev) => prev.map((x) => (x.id === sessionId ? { ...x, config: cfg } : x)));
      } catch {
        /* 关联失败时保留 UI 状态，不阻断对话 */
      }
    },
    [sessionId, docIds, sessions, onSessionsChange],
  );

  /** B2：重拉工作间投影（添加文献后刷新 docIds 与 wsInfo），供 handleAddDoc 复用 */
  const refreshWorkspace = useCallback(async () => {
    if (sessionId == null) return;
    try {
      const ws = await api.getWorkspace(sessionId);
      setWsInfo(ws);
      const others = ws.doc_ids.filter((d) => d !== ws.draft_doc_id);
      setDocIds(ws.draft_doc_id != null ? [...others, ws.draft_doc_id] : ws.doc_ids);
    } catch {
      /* 刷新失败保留现状，不打断对话 */
    }
  }, [sessionId]);

  /** 预览面板「添加文档」：工作间走底层文献集（addWorkspaceDocs），普通会话走 config.doc_ids。
   *  关键修复：工作间文档存于 collection，绝不能写 session.config.doc_ids——
   *  toggleDoc 会触发 getWorkspace 重跑把 docIds 清零，导致"加了又消失"。
   *  工作间判定同样以「会话 config.workspace」兜底（见 closeDoc 说明），避免 wsInfo 异步间隙误判。 */
  const handleAddDoc = useCallback(
    async (id: number) => {
      if (sessionId == null) return;
      const s = sessions.find((x) => x.id === sessionId);
      const isWs = wsInfo != null || (s ? isWorkspace(s) : false);
      if (isWs) {
        try {
          await api.addWorkspaceDocs(sessionId, [id]);
          await refreshWorkspace();
        } catch (e: unknown) {
          setWsError(toErrorMessage(e) || "添加文档到工作间失败");
        }
      } else {
        await toggleDoc(id, true);
      }
    },
    [sessionId, sessions, wsInfo, refreshWorkspace, toggleDoc],
  );

  const handleMentionPick = useCallback(
    (id: number) => {
      // 2026-09-08 修复 J3：原实现直接调 toggleDoc（写 session.config.doc_ids），
      // 但**工作间文档存于底层文献集 collection**，检索时根本不读 `config.doc_ids`
      // → 工作间内 @某文献会「关联静默失败」：无任何报错，检索范围却始终不变，用户无从察觉。
      // 改为复用 handleAddDoc：工作间走 addWorkspaceDocs，普通会话才写 config.doc_ids。
      void handleAddDoc(id);
      if (!mentionPickedRef.current.includes(id)) mentionPickedRef.current.push(id);
      // 同步 state 镜像（仅供渲染「当前检索范围」提示）——ref 与 state 在此一起更新
      setMentionPicked([...mentionPickedRef.current]);
    },
    [handleAddDoc],
  );

  /** F5：取消本会话与文献集的关联（保留已手动关联的文档，仅移除 collection_id）。
   *  后端 config 为「合并」语义，显式置 null 才会删除该键；故发送 collection_id: null。 */
  const unlinkCollection = useCallback(async () => {
    if (sessionId == null) return;
    const s = sessions.find((x) => x.id === sessionId);
    if (!s?.config?.collection_id) return;
    const cfg = { ...(s.config || {}), collection_id: null } as Record<string, unknown>;
    try {
      await api.updateSession(sessionId, cfg);
      const next = { ...(s.config || {}), collection_id: undefined } as Record<string, unknown>;
      onSessionsChange((prev) => prev.map((x) => (x.id === sessionId ? { ...x, config: next } : x)));
    } catch {
      /* 取消关联失败时保留 UI，不阻断对话 */
    }
  }, [sessionId, sessions, onSessionsChange]);

  /** 关闭预览 Tab：从关联文档中移除。
   *  工作间走底层文献集（removeWorkspaceDoc），写完再 refreshWorkspace 重拉 docIds；
   *  普通会话走 toggleDoc(id,false)（写 session.config.doc_ids）。
   *  关键修复（2026-09-07）：工作间判定以「会话 config.workspace」为权威来源，
   *  不依赖闭包里的 wsInfo —— wsInfo 是异步 getWorkspace 才填充的，在创建/切换/刷新间隙
   *  它可能为 null，届时旧逻辑会误走普通会话分支、只写 config.doc_ids（工作间根本不读该字段），
   *  表现为「× 点了没反应 / 文档删不掉」。现用会话 config 兜底，任何时序下都正确走解绑路径。 */
  const closeDoc = useCallback(
    async (id: number) => {
      if (sessionId == null) return;
      const s = sessions.find((x) => x.id === sessionId);
      const isWs = wsInfo != null || (s ? isWorkspace(s) : false);
      if (isWs) {
        try {
          await api.removeWorkspaceDoc(sessionId, id);
          await refreshWorkspace();
        } catch (e: unknown) {
          setWsError(toErrorMessage(e) || "从工作间移除文档失败");
        }
        return;
      }
      toggleDoc(id, false);
    },
    [sessionId, sessions, wsInfo, refreshWorkspace, toggleDoc],
  );

  // 「纯对话模式」徽标一键唤起：关联文档弹窗（从「文献」勾选加入当前会话）
  const [assocOpen, setAssocOpen] = useState(false);
  const [assocDocs, setAssocDocs] = useState<DocumentOut[]>([]);
  const [assocChecked, setAssocChecked] = useState<Set<number>>(new Set());

  const openAssoc = useCallback(async () => {
    const list = await api.listDocuments().catch(() => [] as DocumentOut[]);
    setAssocDocs(list);
    setAssocChecked(new Set(docIds)); // 已关联的默认勾选，便于增删管理
    setAssocOpen(true);
  }, [docIds]);

  const toggleAssocCheck = useCallback((id: number) => {
    setAssocChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // 一次性写入最终关联集合（避免循环调用 toggleDoc 闭包累积 docIds 导致的覆盖问题）
  const confirmAssoc = useCallback(async () => {
    if (sessionId == null) return;
    const finalIds = [...assocChecked];
    const have = docIds;
    // 无变化直接关闭
    if (finalIds.length === have.length && finalIds.every((id) => have.includes(id))) {
      setAssocOpen(false);
      return;
    }
    const s = sessions.find((x) => x.id === sessionId);
    const cfg = { ...(s?.config || {}), doc_ids: finalIds };
    try {
      await api.updateSession(sessionId, cfg);
      setDocIds(finalIds);
      onSessionsChange((prev) => prev.map((x) => (x.id === sessionId ? { ...x, config: cfg } : x)));
    } catch {
      /* 关联失败保留 UI 状态，不阻断对话 */
    }
    setAssocOpen(false);
  }, [sessionId, assocChecked, docIds, sessions, onSessionsChange]);

  const fmtBadge = (name: string): string => {
    const ext = name.toLowerCase().split(".").pop() ?? "";
    if (ext === "pdf") return "PDF";
    if (ext === "md" || ext === "markdown" || ext === "txt") return "MD";
    return "DOCX";
  };

  // 会话切换 / 历史加载完成 / 新消息到达 → 把消息列表滚到底部（直达最新消息）。
  // 2026-09-11（用户需求「切换会话时直达最新的对话消息」）：
  //  ① **切换会话时用 `behavior: "auto"` 瞬时定位** —— 原先一律 `smooth`，长历史会从顶部
  //     一路缓动滚下来（观感是"被拖着走"，且中途还可能被新内容顶偏），不是"直达"；
  //  ② 等 `loadingHistory` 结束再滚 —— 骨架屏换成真消息后才算渲染完成；
  //  ③ 切换后补两帧 + 300ms/800ms 各再滚一次 —— 图片 / KaTeX 公式 / 代码块常在首帧之后
  //     才撑高容器，不补就会"停在半路"（用户看到的并非最新消息）。
  const prevSessionRef = useRef<number | null>(sessionId);
  const switchedAtRef = useRef(0);
  useEffect(() => {
    if (prevSessionRef.current !== sessionId) {
      prevSessionRef.current = sessionId;
      switchedAtRef.current = Date.now();
    }
  }, [sessionId]);

  useEffect(() => {
    if (loadingHistory) return; // 历史尚未渲染，滚了也会被随后撑高的内容顶走
    const justSwitched = Date.now() - switchedAtRef.current < 1500;
    const toBottom = () => {
      // 2026-09-11：用**容器的 scrollTop** 而非 `scrollIntoView` —— 后者把末尾哨兵对齐
      // 「内容盒」底边，而 `.message-list` 有 `padding-bottom: 22px`，导致**永远差一截**
      // （浏览器实测 gap=21px，用户观感就是"没滚到最新消息"）。
      const scroller = bottomRef.current?.closest<HTMLElement>(".message-list");
      if (!scroller) {
        bottomRef.current?.scrollIntoView({
          behavior: justSwitched ? "auto" : "smooth",
          block: "end",
        });
        return;
      }
      if (justSwitched) {
        scroller.scrollTop = scroller.scrollHeight; // 瞬时直达（浏览器自动 clamp 到最大滚动位）
      } else {
        scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
      }
    };
    toBottom();
    if (!justSwitched) return;
    const raf1 = requestAnimationFrame(toBottom);
    const raf2 = requestAnimationFrame(() => requestAnimationFrame(toBottom));
    const t1 = window.setTimeout(toBottom, 300);
    const t2 = window.setTimeout(toBottom, 800);
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [items, sessionId, loadingHistory]);

  // 2026-09-11（用户需求）：**研读 ⇄ 改稿模式切换时，对话区同样要直达底部最新消息**。
  // 该切换只改 `wsInfo.mode`（sessionId / items 均不变），上面的滚动 effect 不会触发；
  // 且切到改稿时 chatArea 从全宽变 340px 侧栏，消息**重新换行**、scrollHeight 大幅变化，
  // 加上 RevisionWorkbench 的 docx 异步渲染会持续改变布局，补齐窗口更长（1.5s 内多点）。
  const prevWsModeRef = useRef<string | undefined>(wsInfo?.mode);
  useEffect(() => {
    const mode = wsInfo?.mode;
    if (prevWsModeRef.current === mode) return;
    prevWsModeRef.current = mode;
    const toBottom = () => {
      const scroller = bottomRef.current?.closest<HTMLElement>(".message-list");
      if (scroller) scroller.scrollTop = scroller.scrollHeight;
    };
    // 首帧 + 50/200/500/1000/1500ms 多点补齐（docx 异步渲染会持续改变布局）
    const raf = requestAnimationFrame(toBottom);
    const timers = [50, 200, 500, 1000, 1500].map((ms) => window.setTimeout(toBottom, ms));
    return () => {
      cancelAnimationFrame(raf);
      timers.forEach((t) => window.clearTimeout(t));
    };
  }, [wsInfo?.mode]);

  // ER-19 / D6：打开「推理轨迹」。
  // D6（2026-09-21）：原实现固定取 `list[0]` —— 只能看最近一次，
  // 「上一轮为什么答错」根本无法复盘。改为：
  //   ① 拉取本会话全部轨迹并在弹窗内可切换；
  //   ② 支持 `#trace=<trace_id>` 深链（打开即定位到指定轮次，便于分享/回到某次失败）。
  const openTrace = useCallback(
    async (wantId?: string) => {
      if (sessionId == null) return;
      setTraceLoading(true);
      setTraceError(null);
      try {
        const list = await api.listTraces(sessionId);
        setTraceList(list);
        if (!list.length) {
          setTraceError("该会话暂无推理轨迹（先进行一次对话）");
          setTrace(null);
          return;
        }
        const pick =
          (wantId && list.find((x) => x.trace_id === wantId)) || list[0];
        const t = await api.getTrace(pick.trace_id);
        setTrace(t);
      } catch (e: unknown) {
        setTraceError(toErrorMessage(e) || "加载轨迹失败");
      } finally {
        setTraceLoading(false);
      }
    },
    [sessionId],
  );

  /** D6：切换弹窗里的轨迹（同时写入 hash，便于分享/回看） */
  const switchTrace = useCallback(
    async (traceId: string) => {
      setTraceLoading(true);
      setTraceError(null);
      try {
        const t = await api.getTrace(traceId);
        setTrace(t);
        try {
          window.history.replaceState(null, "", `#trace=${encodeURIComponent(traceId)}`);
        } catch {
          /* hash 写入失败不影响查看 */
        }
      } catch (e: unknown) {
        setTraceError(toErrorMessage(e) || "加载轨迹失败");
      } finally {
        setTraceLoading(false);
      }
    },
    [],
  );

  /** D6：进入页面时若 URL 带 #trace=... 则直接打开那一轮 */
  useEffect(() => {
    const id = readTraceHash();
    if (id && sessionId != null) void openTrace(id);
  }, [sessionId, openTrace]);

  const title = sessions.find((x) => x.id === sessionId)?.title;
  const activeDocId = docIds.length ? docIds[docIds.length - 1] : null;
  // O-08（2026-10-01）：activeDocs/activeDocIds memo 化 —— 引用稳定是 MessageList/ChatMessageView
  // memo 生效的前提（每渲染新建数组会使 memo 静默失效）。
  const activeDocs = useMemo(() => allDocs.filter((d) => docIds.includes(d.id)), [allDocs, docIds]);
  const activeDocIds = useMemo(() => activeDocs.map((d) => d.id), [activeDocs]);

  // 联网可用性拉取（2026-09-12）：会话 / 联网开关 / 关联文档数 / **所选模型** 任一变化，
  // 都可能改变「当前是否真的具备联网能力」，故都作为依赖。
  //
  // 2026-09-17（第一次修）：原实现**请求失败时 `setWebCap(null)`** → 偶发失败即让横幅消失、
  // 下次成功又出现 = 闪烁。改为「失败沿用旧值」（stale-while-error）。
  //
  // 2026-09-17（第二次修，根治）：仅"沿用旧值"还不够 —— 旧值与**乐观更新**的开关状态
  // 会短暂矛盾（点开关那一刻 `webOn` 已变、数据还是旧的）→ 实测仍闪约 50ms。
  // 现把「请求指纹」随数据一起存（key = **发起请求时**的输入），判定时要求指纹与当前输入
  // 一致，不符则一律不显示（见 `lib/networkWarn.composerWarns`）。
  // ⚠️ `chatModel` 必须进依赖：切换模型会改变 `effective_model`，旧模型的能力数据
  //    用来判断新模型同属错配（切完模型提示不更新，也是同一类 bug）。
  useEffect(() => {
    if (sessionId == null) {
      setWebCap(null);
      return;
    }
    let alive = true;
    const key = capKeyOf({ sessionId, webOn, docsLen: activeDocs.length, model: chatModel });
    api
      .getNetworkCapability(sessionId)
      .then((c) => {
        if (alive) setWebCap({ data: c, key });
      })
      .catch(() => {
        // 失败沿用旧值：旧指纹自然与新输入不同 → 不显示，不会误导（见上方注释）
      });
    return () => {
      alive = false;
    };
  }, [sessionId, webOn, activeDocs.length, chatModel, webCapEpoch]);


  /** 场景可见性（2026-09-23）：随**会话**与**关联文档数**变化重取检索范围。
   *
   *  依赖只放这两个：它是会话级数据，且不像 webCap 那样受乐观更新影响（无开关类的抢跑问题），
   *  所以**不需要**指纹校验与 epoch。
   *  取失败 → 静默置 null（宁可不显示，也不显示错的状态）。 */
  useEffect(() => {
    if (sessionId == null) {
      // 无会话（尚未创建）→ 不显示范围提示
      setDocScope(null);
      return;
    }
    let cancelled = false;
    void api
      .getDocScope(sessionId)
      .then((d) => {
        if (!cancelled) setDocScope(d);
      })
      .catch(() => {
        if (!cancelled) setDocScope(null);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, activeDocs.length]);
  // F5：若本会话关联了文献集（config.collection_id），顶栏展示已关联状态，可一键取消关联
  const collectionId = sessions.find((x) => x.id === sessionId)?.config?.collection_id as number | undefined;

  // F5：关联了文献集时，拉取名称用于顶栏徽标（collectionId 变化即重取）
  useEffect(() => {
    let alive = true;
    if (collectionId == null) {
      setCollectionName(null);
      return;
    }
    api
      .listCollections()
      .then((res) => {
        if (!alive) return;
        const c = res.items.find((x) => x.id === collectionId);
        setCollectionName(c?.name ?? null);
      })
      .catch(() => {
        if (alive) setCollectionName(null);
      });
    return () => {
      alive = false;
    };
  }, [collectionId]);

  // 顶栏模式徽标：随「关联文档 / 联网」状态切换（对应后端 3 种工作状态）
  const mode = useMemo(() => {
    if (docIds.length === 0) {
      return { key: "pure", label: "纯对话模式", hint: "未关联文档，仅由大模型基于自身通用知识作答（使用技能亦不读取任何论文）" };
    }
    if (webOn) {
      return { key: "rag_web", label: "检索 · 联网", hint: "已关联文档：以论文检索为首要依据，不足时由联网补充并区分来源" };
    }
    return { key: "rag", label: "检索 + 大模型", hint: "已关联文档：仅以已上传论文为唯一依据作答，不调用联网" };
  }, [docIds, webOn]);

  if (sessionId == null) {
    return (
      <div className="chat-view">
        <div className="welcome">
          <div className="welcome-mark"><Icons.fileText size={22} /></div>
          <h1>PaperMate · 论文智友</h1>
          <p>上传论文，开启带引用溯源与工具调用的智能对话。</p>
          <p className="welcome-hint">在左侧「新对话」开始，或先到「上传论文」提交资料。</p>
        </div>
      </div>
    );
  }

  /** C4：改稿工作台模式判定（工作间 + revision）；draftDocId 为空时引导先指定主文献 */
  const isWorkbench = wsInfo?.mode === "revision";
  const workbenchDraftId = wsInfo?.draft_doc_id ?? null;
  const workbenchDraftName =
    workbenchDraftId != null ? allDocs.find((d) => d.id === workbenchDraftId)?.filename : undefined;

  /** 对话区（消息流 + 输入）：研读模式为主区，改稿工作台中缩小为侧栏（共用同一 useStreamingChat 实例） */
  const chatArea = (
    <>
        {historyError && <div className="error-box" style={{ margin: "10px 22px 0" }}>加载历史失败：{historyError}</div>}
        {wsError && (
          <div className="error-box" style={{ margin: "10px 22px 0", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ flex: 1 }}>{wsError}</span>
            <button className="icon-btn" onClick={() => setWsError(null)} aria-label="关闭提示">
              <Icons.x size={14} />
            </button>
          </div>
        )}
        {/* N6（2026-09-10）：流进行中发送被丢弃的可见提示（原先静默丢输入） */}
        {sendNotice && (
          <div className="error-box" style={{ margin: "10px 22px 0", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ flex: 1 }}>{sendNotice}</span>
            <button className="icon-btn" onClick={() => setSendNotice(null)} aria-label="关闭提示">
              <Icons.x size={14} />
            </button>
          </div>
        )}
        {loadingHistory ? (
          <SkeletonThread />
        ) : (
          <MessageList
            items={items}
            onBottomRef={handleBottomRef}
            onOpenCitation={setReaderCitation}
            activeDocIds={activeDocIds}
            onRegenerate={regenerate}
            lastAssistantId={lastAssistantId}
            onRate={rateMessage}
            ratedMap={ratedMap}
          />
        )}
        <ChatInput
          onSend={wrapSend}
          onStop={stop}
          streaming={streaming}
          disabled={sessionId == null}
          models={allModels}
          currentModel={chatModel}
          onModelChange={changeModel}
          modelSource={modelSource}
          balances={balances}
          webOn={webOn}
          onWebToggle={  toggleWeb}
          docScope={docScope}
          ctxUsage={lastCtxUsage}
          webCapability={webCap?.data ?? null}
          webCapabilityKey={webCap?.key ?? null}
          discipline={discipline}
          onDisciplineChange={changeDiscipline}
          skills={skills ?? undefined}
          mentionDocs={activeDocs.map((d, i) => ({
            id: d.id,
            label: docLabel(i),
            filename: d.filename,
          }))}
          onMentionPick={handleMentionPick}
          mentionedDocIds={mentionPicked}
          sessionId={sessionId ?? undefined}
        />
    </>
  );

  return (
    <div className="chat-view">
      <div className="chat-main">
      <div className="chat-header">
        <span className="chat-title">{title ?? "对话"}</span>
        <span
          className={`mode-badge mode-${mode.key} clickable`}
          title={mode.key === "pure" ? `${mode.hint}（点击关联文档）` : `${mode.hint}（点击管理关联文档）`}
          role="button"
          tabIndex={0}
          onClick={openAssoc}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              openAssoc();
            }
          }}
        >
          <span className="mode-dot" />
          {mode.label}
        </span>
        {/* F6：auto 模式自动识别学科后，顶栏即时反馈，可一键改为手动指定 */}
        {detected && (
          <span
            className="mode-badge discipline-badge clickable"
            title="本次自动识别的学科视角，点击改为手动指定"
            role="button"
            tabIndex={0}
            onClick={() => changeDiscipline(detected.key)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                changeDiscipline(detected.key);
              }
            }}
          ><Icons.search size={14} /> 已识别为：{detected.label}
          </span>
        )}
        {/* F5：普通会话关联了文献集时，顶栏展示已关联徽标，可一键取消关联。
            C1：工作间会话不显示该徽标 —— 其 collection_id 是工作间底座，
            点「取消关联」会清掉 collection_id 直接破坏工作间（文献解析断裂）。 */}
        {collectionId != null && wsInfo == null && (
          <span
            className="mode-badge collection-badge clickable"
            title="本会话已关联文献集，查询范围 = 文献集内文档 ∪ 手动关联文档；点击取消关联"
            role="button"
            tabIndex={0}
            onClick={unlinkCollection}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                unlinkCollection();
              }
            }}
          ><Icons.book size={14} /> 已关联文献集：{collectionName ?? `#${collectionId}`}
            <span className="badge-x" aria-hidden="true"><Icons.x size={14} /></span>
          </span>
        )}
        {/* C1：工作间模式切换（研读 ⇄ 改稿）；改稿模式下 /改稿 缺省使用主文献（草稿） */}
        {wsInfo != null && (
          <span className="ws-mode-switch" role="group" aria-label="工作间模式">
            {(["research", "revision"] as const).map((m) => (
              <button
                key={m}
                className={wsInfo.mode === m ? "on" : ""}
                disabled={modeBusy}
                aria-pressed={wsInfo.mode === m}
                onClick={() => void switchMode(m)}
                title={
                  m === "revision"
                    ? "改稿模式：/改稿 默认使用主文献（草稿），其余文献为参考"
                    : "研读模式：常规文献研读与问答"
                }
              >
                {m === "revision" ? "改稿" : "研读"}
              </button>
            ))}
          </span>
        )}
        <span className="header-spacer" />

        {/* 模型选择已移至输入区下方（ChatInput 集成） */}
        <button className="panel-toggle" onClick={() => openTrace()} title="查看推理轨迹">
          轨迹 ▤
        </button>
        {/* F8：记忆管理面板入口——直接复用设置页 MemoryPanel（含 scope 切换 / 编辑 / 删除） */}
        <button className="panel-toggle" onClick={() => setMemOpen(true)} title="查看与编辑长期记忆"><Icons.brain size={14} /> 记忆
        </button>
        {/* 注：联网开关已统一集成至 ChatInput 输入区工具栏（与模型选择并列），不再于 header 重复 */}
      </div>

        {/* C4：改稿工作台分支 —— revision 模式主区为段落编辑区，对话缩小为侧栏；未指定主文献时引导 */}
        {isWorkbench && workbenchDraftId != null ? (
          <div className="workbench-layout">
            <Suspense fallback={<PanelLoading />}>
              <RevisionWorkbench docId={workbenchDraftId} docName={workbenchDraftName} />
            </Suspense>
            <div className="workbench-chat">{chatArea}</div>
          </div>
        ) : isWorkbench ? (
          <div className="rw-guide">
            <Icons.fileText size={20} />
            <p>改稿模式需要先指定主文献（草稿）。</p>
            <p className="rw-hint">在右侧文件区用「设为主文献」星标指定后，即可逐段编辑。</p>
          </div>
        ) : (
          chatArea
        )}
      </div>

      {/* 右栏 · 文档预览面板（画布 24:1）；B2：工作间会话附主文献星标操作。
          C4：改稿工作台且已指定主文献时隐藏（段落编辑区承担展示）；
          未指定主文献时保留 —— 用户需在文件区用星标完成指定（工作台有引导）。 */}
      {(!isWorkbench || workbenchDraftId == null) && (
        <Suspense fallback={<PanelLoading />}>
          <DocumentPreviewPanel
            docs={activeDocs}
            activeDocId={activeDocId}
            onSwitch={(id) => setDocIds((prev) => [...prev.filter((d) => d !== id), id])}
            onCloseDoc={closeDoc}
            onAddDoc={handleAddDoc}
            onMutated={() => { void refreshDocs(); }}
            isWorkspace={wsInfo != null}
            draftDocId={wsInfo?.draft_doc_id ?? null}
            onSetDraft={setDraftDoc}
            draftBusy={draftBusy}
          />
        </Suspense>
      )}

      {readerCitation && <PdfReader citation={readerCitation} onClose={() => setReaderCitation(null)} />}
      {/* 2026-09-09 修复 J4：原渲染条件是 `trace &&`，而 `traceError` 又在弹窗**内部**——
          会话无轨迹或加载失败时 trace 为 null → 弹窗根本不渲染 → 错误被静默吞掉，
          用户点「轨迹 ▤」后毫无反馈。改为 trace / traceError 任一存在即渲染，
          内容区容错 trace 为空；关闭时一并清掉错误，避免下次误显旧错误。 */}
      {(trace || traceError) && (
        <div
          className="modal-mask"
          onClick={() => {
            setTrace(null);
            setTraceError(null);
          }}
          aria-modal="true"
          role="dialog"
        >
          <div className="modal trace-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="reader-title">推理轨迹</div>
              {/* E2：导出为评测样本 —— 让"真实问答"能回流进评测集。
                  gold 留空（检索命中只作候选），避免用系统输出自证有效。 */}
              <button
                className="icon-btn"
                onClick={() => exportCase()}
                disabled={exporting || !trace?.trace_id}
                title="导出这一轮为评测样本（jsonl，gold 需人工补）"
                aria-label="导出为评测样本"
              >
                <Icons.download size={14} />
              </button>
              <button
                className="icon-btn"
                onClick={() => {
                  setTrace(null);
                  setTraceError(null);
                  setExportNote(null);
                }}
                aria-label="关闭"
              >
                <Icons.x size={14} />
              </button>
            </div>
            {exportNote && <div className="muted" style={{ padding: "0 12px 8px" }}>{exportNote}</div>}
            {/* D6：历史轨迹切换 —— 只能看最近一次的话，"上一轮为什么错"永远无法复盘 */}
            {traceList.length > 1 && (
              <div className="trace-picker">
                <label className="muted" htmlFor="trace-pick">
                  选择轮次（{traceList.length}）：
                </label>
                <select
                  id="trace-pick"
                  className="trace-select"
                  value={trace?.trace_id ?? ""}
                  onChange={(e) => void switchTrace(e.target.value)}
                  disabled={traceLoading}
                >
                  {traceList.map((t, i) => (
                    <option key={t.trace_id} value={t.trace_id}>
                      {i === 0 ? "最近一次" : `第 ${traceList.length - i} 次`} · {t.total_steps} 步 ·{" "}
                      {(t.total_latency_ms / 1000).toFixed(1)}s
                    </option>
                  ))}
                </select>
              </div>
            )}
            {traceError && <div className="error-box">{traceError}</div>}
            {traceLoading ? <div className="muted">加载中…</div> : trace ? <TraceTimeline trace={trace} /> : null}
          </div>
        </div>
      )}

      {/* 关联文档弹窗：由顶栏「纯对话模式」徽标一键唤起 */}
      {assocOpen && (
        <div className="modal-mask" onClick={() => setAssocOpen(false)} aria-modal="true" role="dialog">
          <div className="modal doc-picker" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div>
                <div className="reader-title">关联文档到本会话</div>
                <div className="reader-sub">
                  {assocDocs.length
                    ? `勾选要从「文献」加入对话的论文（${assocDocs.length} 篇可关联）`
                    : "文献库暂无文档，请先到「上传论文」添加。"}
                </div>
              </div>
              <button className="icon-btn" onClick={() => setAssocOpen(false)} aria-label="关闭"><Icons.x size={14} /></button>
            </div>
            <div className="doc-picker-list">
              {assocDocs.map((d) => {
                const checked = assocChecked.has(d.id);
                return (
                  <label key={d.id} className={`doc-picker-item ${checked ? "checked" : ""}`}>
                    <input type="checkbox" checked={checked} onChange={() => toggleAssocCheck(d.id)} />
                    <span className={`fmt-badge ${fmtBadge(d.filename).toLowerCase()}`}>{fmtBadge(d.filename)}</span>
                    <span className="doc-picker-name">{d.filename}</span>
                    <span className="muted">（{d.num_chunks ?? 0} 块）</span>
                  </label>
                );
              })}
            </div>
            <div className="doc-picker-actions">
              <button className="btn ghost" onClick={() => setAssocOpen(false)}>取消</button>
              <button className="btn primary" onClick={confirmAssoc}>确定关联</button>
            </div>
          </div>
        </div>
      )}

      {/* F8：记忆管理面板（复用 MemoryPanel，含 scope 切换 / 编辑 / 删除 / 清空），无需新增后端接口 */}
      {memOpen && (
        <div className="modal-mask" onClick={() => setMemOpen(false)} aria-modal="true" role="dialog" aria-label="记忆管理">
          <div
            className="modal mem-modal"
            onClick={(e) => e.stopPropagation()}
            ref={memRef}
            tabIndex={-1}
          >
            <div className="modal-head">
              <div className="reader-title">记忆管理</div>
              <button className="icon-btn" onClick={() => setMemOpen(false)} aria-label="关闭"><Icons.x size={14} /></button>
            </div>
            <MemoryPanel />
          </div>
        </div>
      )}
    </div>
  );
}
