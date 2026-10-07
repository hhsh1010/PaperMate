import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { ChatPage } from "./pages/ChatPage";
import { api } from "./api/client";
import { useBackendStatus } from "./hooks/useBackendStatus";
import { SettingsProvider } from "./store/settings";
import type { SessionOut, WorkspaceMode, WorkspaceOut } from "./lib/types";
import { isWorkspace } from "./lib/types";
import { Icons, type IconProps } from "./lib/icons";
import { toErrorMessage } from "./lib/error";
import { isTauri } from "./lib/titlebar";
import TitleBar from "./components/TitleBar";

/**
 * T-08（2026-09-29）：**路由级按需加载** —— 除默认落地页（chat）外，其余 6 个视图
 * 都不进首屏模块图（此前全部静态 import ⇒ 单 bundle 1011 kB）。
 *
 * 边界取**整页粒度**（按卡片要求，避免拆出大量小 chunk）；这些都是 **named export**，
 * 故 lazy 工厂需映射成 `{ default: m.X }`。
 * ⚠️ `ChatPage` **有意保持 eager**：它是默认落地页（`view` 初值与 hash 缺省都落到 chat），
 * 若也 lazy，首屏**必然**多一次"占位 → 内容"的闪动，与减体积的收益不成比例。
 */
const DocumentsPage = lazy(() =>
  import("./pages/DocumentsPage").then((m) => ({ default: m.DocumentsPage })));
const EvalPanel = lazy(() =>
  import("./components/EvalPanel").then((m) => ({ default: m.EvalPanel })));
const SettingsPage = lazy(() =>
  import("./pages/SettingsPage").then((m) => ({ default: m.SettingsPage })));
const LoginPage = lazy(() =>
  import("./pages/LoginPage").then((m) => ({ default: m.LoginPage })));
const HomePage = lazy(() =>
  import("./pages/HomePage").then((m) => ({ default: m.HomePage })));
const UsagePanel = lazy(() =>
  import("./components/UsagePanel").then((m) => ({ default: m.UsagePanel })));

/** 懒加载占位：复用既有 `.muted` 令牌类，不新增样式令牌（卡片要求：不改样式令牌） */
function ViewLoading() {
  return <div className="muted" style={{ margin: "22px" }}>加载中…</div>;
}

/** 视图：6 项导航（home/docs/chat/usage/eval/settings）+ login 子页；#upload 已合并进 #docs */
export type View =
  | "home"
  | "docs"
  | "chat"
  | "usage"
  | "eval"
  | "settings"
  | "login";

const VIEWS: View[] = ["home", "docs", "chat", "usage", "eval", "settings", "login"];

/** 主题模式：light / dark / system（设置页可切换，localStorage pm-theme） */
type ThemeMode = "light" | "dark" | "system";
function resolveTheme(mode: ThemeMode): "light" | "dark" {
  if (mode === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return mode;
}

function useTheme() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem("pm-theme");
    // 默认 light（画布即浅色设计）；仅显式存了 dark/system 才跟随
    return saved === "dark" || saved === "system" ? saved : "light";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = resolveTheme(themeMode);
    localStorage.setItem("pm-theme", themeMode);
  }, [themeMode]);
  return [themeMode, setThemeMode] as const;
}



/** F11：评测看板图标 / F9：研究首页图标 —— 统一使用线性图标，与侧栏观感一致 */
const EvalIcon = Icons.chart;
const HomeIcon = Icons.home;

/* 侧栏导航（6 项）；移动端底部导航保持原 4 主项（首页/评测走侧栏抽屉） */
const NAV_ITEMS: { key: View; label: string; Icon: (p: IconProps) => React.ReactElement }[] = [
  { key: "home", label: "首页", Icon: HomeIcon },
  { key: "docs", label: "文献", Icon: Icons.docs },
  { key: "chat", label: "问答", Icon: Icons.chat },
  { key: "usage", label: "用量", Icon: Icons.usage },
  { key: "eval", label: "评测", Icon: EvalIcon },
  { key: "settings", label: "设置", Icon: Icons.settings },
];

/**
 * 侧栏列表排序（2026-10-02，置顶）。
 *
 * ★★ **必须与后端 `session_service.list` 的 `order_by(pinned DESC, updated_at DESC)`
 * 逐字等价**——两处不同步就会出现"两把尺子"：后端已重排、前端按旧序显示，
 * 表现为"点���没反应 / 刷新后位置变了"。
 *
 * 抽成**导出的纯函数**（而不是内联在 `onTogglePin` 里）：
 * 内联版只能靠 `?raw` 源码文本断言，而**文本断言对"逻辑被改坏"很无力**
 * ——实测注入"去掉重排"时，源码断言仍全绿（因为排序体字面量还在），
 * 只有**真正调用一次排序**才能发现行为变了。
 */
export function sortSessionsForSidebar<T extends { pinned?: boolean; updated_at: string | null }>(
  list: T[],
): T[] {
  return [...list].sort((a, b) => {
    // 置顶优先（pinned 降序）
    if (!!b.pinned !== !!a.pinned) return b.pinned ? 1 : -1;
    // 组内按最近活动（updated_at 降序）
    return (b.updated_at ?? "").localeCompare(a.updated_at ?? "");
  });
}

/**
 * 侧栏会话条目：工作间与普通会话共用（删除二次确认逻辑一致）。
 * 工作间额外渲染语义图标、模式徽标（研读/改稿）与「已指定主文献」标记。
 *
 * ★ 重命名（2026-10-02）：**行内编辑**（不弹窗）——
 *  - 点铅笔 → 标题变输入框（`session-title-input`），自动全选；
 *  - Enter / 点对勾 提交，Esc / 点叉 取消（与删除的"二次确认"是两套独立状态，互不干扰）；
 *  - ★ 提交失败时**恢复原标题并保留编辑态**（让用户看到失败，而不是"标题悄悄没变"）。
 *  - ⚠️ 空标题不提交（`trim()` 后为空 ⇒ 视为取消）⇒ 避免出现无名会话。
 */
function SessionRow({
  session,
  active,
  confirming,
  workspace = false,
  onSelect,
  onRequestDelete,
  onRename,
  onTogglePin,
}: {
  session: SessionOut;
  active: boolean;
  confirming: boolean;
  workspace?: boolean;
  onSelect: () => void;
  onRequestDelete: () => void;
  onRename: (title: string) => Promise<void>;
  onTogglePin: () => Promise<void>;
}) {
  const mode = (session.config?.mode as WorkspaceMode) ?? "research";
  // 置顶（2026-10-02）：后端 `SessionOut.pinned` 缺省false（老数据/老接口）
  const pinned = session.pinned === true;
  // 行内编辑态（★ 与删除的 `confirming` 互不影响）
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.title);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!editing) return;
    setDraft(session.title);
    // 下一帧聚焦并全选（元素此时才刚挂上）
    const t = window.setTimeout(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    }, 0);
    return () => window.clearTimeout(t);
  }, [editing, session.title]);

  const commit = useCallback(async () => {
    const next = draft.trim();
    if (!next) {
      setEditing(false);
      return;
    }
    if (next === session.title) {
      setEditing(false);
      return;
    }
    setBusy(true);
    try {
      await onRename(next);
      setEditing(false);
    } catch {
      // ★ 失败：留在编辑态让用户看到"没改成"并可重试（静默恢复原值= 用户以为成功了）
      setDraft(session.title);
    } finally {
      setBusy(false);
    }
  }, [draft, onRename, session.title]);

  if (editing) {
    return (
      <li className="editing">
        <span className="ws-icon" aria-hidden="true">
          {workspace ? <Icons.layers size={14} /> : <Icons.chat size={14} />}
        </span>
        <input
          ref={inputRef}
          className="session-title-input"
          value={draft}
          disabled={busy}
          maxLength={120}
          aria-label="会话标题"
          onChange={(e) => setDraft(e.target.value)}
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void commit();
            } else if (e.key === "Escape") {
              e.preventDefault();
              setEditing(false);
            }
          }}
        />
        <button
          className="session-del confirm-ok"
          title="确认"
          aria-label="确认重命名"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            void commit();
          }}
        >
          <Icons.check size={14} />
        </button>
        <button
          className="session-del"
          title="取消"
          aria-label="取消重命名"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            setEditing(false);
          }}
        >
          <Icons.x size={14} />
        </button>
      </li>
    );
  }

  return (
    <li
      className={`${active ? "active " : ""}${pinned ? "pinned" : ""}`}
      onClick={onSelect}
      title={workspace ? `工作间 · ${mode === "revision" ? "改稿" : "研读"}` : session.title}
    >
      {workspace && (
        <span className="ws-icon" aria-hidden="true">
          <Icons.layers size={14} />
        </span>
      )}
      <span className="session-title">{session.title}</span>
      {/* ★ 2026-10-03（用户要求"去除工作间对话的研读标志"）：
          **只隐藏「研读」**（默认态、绝大多数工作间都是它 ⇒ 纯噪音）；
          **保留「改稿」** —— 它是少数模式，若一并隐藏，用户将**完全无法分辨**
          工作间处于改稿模式（列表里再也看不到任何模式线索）。
          ⚠️ 两者的信息仍可从列表项的 tooltip（`工作间 · 研读/改稿`）读到。 */}
      {workspace && mode === "revision" && (
        <span className="ws-mode revision">改稿</span>
      )}
      {/*★ 置顶标记**常驻**（不 hover 才现）：它是"这一条被钉住了"的状态，
          属于内容而非操作 ⇒ 隐藏起来就失去了置顶的意义。*/}
      {pinned && (
        <span className="ws-pin-mark" title="已置顶">
          <Icons.pin size={12} />
        </span>
      )}
      <button
        className={`session-del session-pin ${pinned ? "on" : ""}`}
        title={pinned ? "取消置顶" : "置顶"}
        aria-label={pinned ? "取消置顶" : "置顶"}
        aria-pressed={pinned}
        onClick={(e) => {
          e.stopPropagation();
          void onTogglePin();
        }}
      >
        <Icons.pin size={14} />
      </button>
      <button
        className="session-del session-rename"
        title="重命名"
        aria-label="重命名"
        onClick={(e) => {
          e.stopPropagation();
          setEditing(true);
        }}
      >
        <Icons.pencil size={14} />
      </button>
      <button
        className={`session-del ${confirming ? "confirming" : ""}`}
        title={confirming ? "再次点击确认删除" : "删除"}
        aria-label={confirming ? "再次点击确认删除" : "删除"}
        onClick={(e) => {
          e.stopPropagation();
          onRequestDelete();
        }}
      >
        {confirming ? <Icons.check size={14} /> : <Icons.x size={14} />}
      </button>
    </li>
  );
}

/** 应用外壳：桌面左侧栏 / 移动端顶栏 + 底部导航 + 侧栏抽屉；hash 路由 5 视图（#upload 兼容重定向 #docs）。 */
export function App() {
  const [themeMode, setThemeMode] = useTheme();
  const [view, setView] = useState<View>(() => {
    const h = typeof window !== "undefined" ? window.location.hash.replace("#", "") : "";
    // 上传页已合并进文档库：#upload 老链接 → #docs；#about 老链接已废弃 → 回落默认 chat。
    // F9：#home 现为合法视图（研究首页）；默认仍是 chat（按产品决策，首页不设为默认落地页）。
    if (h === "upload") return "docs";
    if (h === "about") return "chat";
    return (VIEWS as string[]).includes(h) ? (h as View) : "chat";
  });
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  /** ★ 2026-10-03（用户："给最左侧功能侧边区增加收起功能"）：桌面端**折叠**状态。
   *
   * ⚠️ 与 `sidebarOpen`（移动端抽屉）是**两件事**，不要混：
   * - `sidebarOpen` 只在 `@media (max-width:767px)` 下有意义（抽屉滑入/滑出）；
   * - `sidebarCollapsed` 只在**桌面**下有意义（收成窄图标条，把横向空间还给对话区）。
   * ⇒ CSS 里两者分开处理；持久化 key 独立，便于日后加"移动端也记住"而互不干扰。
   */
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem("pm-sidebar-collapsed") === "1",
  );
  const toggleSidebarCollapsed = useCallback(() => {
    setSidebarCollapsed((v) => {
      localStorage.setItem("pm-sidebar-collapsed", v ? "0" : "1");
      return !v;
    });
  }, []);
  /** 操作失败提示：避免「研读」等入口在后端不可用时点击后毫无反馈 */
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<number | null>(null);
  const confirmTimer = useRef<number | null>(null);
  const backend = useBackendStatus();

  // hash 同步：支持深链接（#docs / #chat / #usage / #settings；#login 为占位页、已移出导航；#upload → #docs；#home/#about 已废弃）
  useEffect(() => {
    const onHash = () => {
      const h = window.location.hash.replace("#", "");
      if (h === "upload") {
        setView("docs");
        history.replaceState(null, "", "#docs");
        return;
      }
      if ((VIEWS as string[]).includes(h)) setView(h as View);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    const want = "#" + view;
    if (window.location.hash !== want) history.replaceState(null, "", want);
  }, [view]);

  // 设置弹窗：ESC 关闭（与遮罩点击、关闭按钮并列的无障碍退出路径）
  useEffect(() => {
    if (view !== "settings") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setView("chat");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view]);

  const go = useCallback((v: string) => {
    if ((VIEWS as string[]).includes(v)) setView(v as View);
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const list = await api.listSessions();
      setSessions(list);
      setSessionId((cur) => cur ?? list[0]?.id ?? null);
    } catch {
      /* 后端未起：状态灯会提示，可重试 */
    }
  }, []);

  // 首次加载；后端由离线恢复在线且尚无数据时自动重载
  useEffect(() => {
    if (backend === "online") loadSessions();
    else if (backend === "offline" && sessions.length === 0) setSessions([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backend]);

  const onNew = async () => {
    const s = await api.createSession();
    setSessions((prev) => [s, ...prev]);
    setSessionId(s.id);
    setView("chat");
    setSidebarOpen(false);
  };

  const onDelete = useCallback(async (id: number) => {
    await api.deleteSession(id);
    setSessions((prev) => {
      const next = prev.filter((x) => x.id !== id);
      setSessionId((cur) => (cur === id ? (next[0]?.id ?? null) : cur));
      return next;
    });
    setConfirming(null);
  }, []);

  /**
   * ★ 重命名（2026-10-02）：**乐观更新**——先改本地列表让 UI 立即响应，
   * 失败时**回滚到原标题**并把错误抛给 `SessionRow`（它会留在编辑态让用户看到"没改成"）。
   *
   * ⚠️ 为什么不用"改完再 loadSessions()"：那会**丢掉刚刚的乐观更新**，
   * 用户会看到"标题闪一下又变回去"（本项目的 UI 时序纪律：
   * 乐观更新 × 远端数据必然有窗口，判断处要校验同源）。
   * 后端 `PATCH /sessions/{id}` 本就支持 `title`（`SessionUpdate.title`），
   * 工作间与会话是**同一张表** ⇒ 一条路径覆盖两者。
   */
  const onRename = useCallback(async (id: number, title: string) => {
    const prev = sessions.find((s) => s.id === id);
    setSessions((list) => list.map((s) => (s.id === id ? { ...s, title } : s)));
    try {
      await api.renameSession(id, title);
    } catch (e) {
      if (prev) setSessions((list) => list.map((s) => (s.id === id ? prev : s)));
      throw e;
    }
  }, [sessions]);

  /**
   * ★ 置顶/取消（2026-10-02）：**乐观更新 + 本地重排**。
   *
   * ⚠️ 为什么**必须本地重排**而不只是改 `pinned` 标志：
   * 列表顺序来自后端 `ORDER BY pinned DESC, updated_at DESC`；若只改标志不重排，
   * 置顶项会**留在原位**（用户看不到"钉住了"的效果），直到下次刷新才跳上去
   * ⇒ 表现为"点了没反应"。
   *这里复刻**同一条**排序规则（`pinned` 降序 ⇒ 活动降序），
   * 代价是"若别处也改了排序，两处要同步"——已写进测试守护。
   *
   * ⚠️ 失败时**回滚整份列表快照**（不只是标志）——因为本地重排已经动过顺序，
   * 只回滚标志会留下一个"顺序错了但标志对"的脏状态。
   */
  const onTogglePin = useCallback(
    async (id: number) => {
      const snapshot = sessions;
      const target = sessions.find((s) => s.id === id);
      if (!target) return;
      const nextPinned = !target.pinned;
      setSessions((list) => {
        const flipped = list.map((s) =>
          s.id === id ? { ...s, pinned: nextPinned } : s,
        );
        // ★ 复用导出的纯函数（与后端排序规则同源，且可被行为测试直接验证）
        return sortSessionsForSidebar(flipped);
      });
      try {
        await api.pinSession(id, nextPinned);
      } catch (e) {
        setSessions(snapshot);
        throw e;
      }
    },
    [sessions],
  );

  // 删除二次确认：首次点击进入确认态（3s 后自动取消），再点才真删
  const requestDelete = useCallback(
    (id: number) => {
      if (confirming === id) {
        if (confirmTimer.current != null) window.clearTimeout(confirmTimer.current);
        onDelete(id);
        return;
      }
      setConfirming(id);
      if (confirmTimer.current != null) window.clearTimeout(confirmTimer.current);
      confirmTimer.current = window.setTimeout(() => setConfirming(null), 3000);
    },
    [confirming, onDelete],
  );

  const activeTitle =
    view === "home"
      ? "首页"
      : view === "docs"
        ? "文献"
        : view === "chat"
          ? sessions.find((s) => s.id === sessionId)?.title || "问答"
          : view === "usage"
            ? "用量"
            : view === "eval"
              ? "忠实度看板"
              : view === "login"
                ? "登录"
                : "设置";

  const statusLabel =
    backend === "online" ? "服务在线" : backend === "offline" ? "后端离线" : "连接中…";

  /**
   * B1：创建工作间后的统一收尾 —— 先本地插入保证即时可见并选中跳转，
   * 再「尽力」刷新列表补齐时间戳。**刷新失败不阻断**：工作间在服务端已创建成功，
   * 若让 loadSessions 的失败冒泡，会把成功误报成失败。
   */
  const afterWorkspaceCreated = useCallback(
    async (ws: WorkspaceOut) => {
      setSessions((prev) => [
        {
          id: ws.session_id,
          title: ws.name,
          config: {
            workspace: true,
            collection_id: ws.collection_id,
            draft_doc_id: ws.draft_doc_id,
            mode: ws.mode,
          },
          created_at: null,
          updated_at: null,
        },
        ...prev.filter((s) => s.id !== ws.session_id),
      ]);
      setSessionId(ws.session_id);
      setView("chat");
      setActionError(null);
      await loadSessions().catch(() => {});
    },
    [loadSessions],
  );

  /**
   * B1：以所选文档创建工作间（原子建文献集 + 工作间会话），创建后直接进入该工作间。
   * 由原「新建会话研读」重定位而来（D1 收敛：不保留「普通会话关联文献」双入口）。
   * 失败时 **向上抛出** —— 让命名弹窗保留用户输入并显示错误，而非静默关闭。
   */
  const createWorkspace = useCallback(
    async (docIds: number[], name: string) => {
      try {
        const ws = await api.createWorkspace({
          name: name || undefined,
          doc_ids: docIds,
          mode: "research",
        });
        await afterWorkspaceCreated(ws);
      } catch (e: unknown) {
        throw new Error(toErrorMessage(e) || "创建工作间失败，请确认后端服务正常");
      }
    },
    [afterWorkspaceCreated],
  );

  /** B1：以已有文献集创建工作间（只关联、不重复建集）；失败同样上抛给弹窗 */
  const createWorkspaceFromCollection = useCallback(
    async (collectionId: number, name: string) => {
      try {
        const ws = await api.createWorkspace({
          name: name || undefined,
          collection_id: collectionId,
          mode: "research",
        });
        await afterWorkspaceCreated(ws);
      } catch (e: unknown) {
        throw new Error(toErrorMessage(e) || "创建工作间失败，请确认后端服务正常");
      }
    },
    [afterWorkspaceCreated],
  );

  /**
   * 首页「新建工作间」一键入口（2026-09-07）：空工作间（无文档、无文献集），
   * 后端自动建一个空文献集 + 工作间会话，创建后直接进入。名称自动编号避免重复。
   */
  const createEmptyWorkspace = useCallback(async () => {
    const idx = sessions.filter((s) => isWorkspace(s)).length + 1;
    const ws = await api.createWorkspace({
      name: `新工作间 ${idx}`,
      mode: "research",
    });
    await afterWorkspaceCreated(ws);
  }, [afterWorkspaceCreated, sessions]);

  /** 文档库「加入当前会话」：把文档并入当前【普通会话】检索范围（不新建会话，幂等去重） */
  const addDocsToSession = useCallback(
    async (docIds: number[]) => {
      if (sessionId == null || docIds.length === 0) return;
      const s = sessions.find((x) => x.id === sessionId);
      const cur = (s?.config?.doc_ids as number[]) ?? [];
      const merged = [...cur];
      for (const id of docIds) if (!merged.includes(id)) merged.push(id);
      const cfg = { ...(s?.config || {}), doc_ids: merged };
      try {
        await api.updateSession(sessionId, cfg);
        setSessions((prev) => prev.map((x) => (x.id === sessionId ? { ...x, config: cfg } : x)));
        // 跳转到问答页让变更可见（否则用户在文献页无任何反馈）
        setView("chat");
        setSidebarOpen(false);
      } catch {
        /* 关联失败不阻断 */
      }
    },
    [sessionId, sessions],
  );

  /** B1：把文档加入当前【工作间】的文献集（后端幂等去重）；失败在主区提示，不静默 */
  const addDocsToWorkspace = useCallback(
    async (docIds: number[]) => {
      if (sessionId == null || docIds.length === 0) return;
      try {
        await api.addWorkspaceDocs(sessionId, docIds);
        setView("chat");
        setSidebarOpen(false);
        setActionError(null);
      } catch (e: unknown) {
        setActionError(toErrorMessage(e) || "加入工作间失败，请确认后端服务正常");
      }
    },
    [sessionId],
  );

  /** A2：左栏分组 —— 工作间（config.workspace=true）与会话同级，工作间置顶 */
  const workspaceSessions = sessions.filter(isWorkspace);
  const chatSessions = sessions.filter((s) => !isWorkspace(s));

  /** B1：当前会话是否为工作间 —— 决定文献区「加入 *」入口的文案与目标 */
  const currentSessionIsWorkspace = isWorkspace(sessions.find((s) => s.id === sessionId));

  /** 仅登录页走 fullscreen：不显示侧栏 / 顶栏 / 底部导航（原首页也走 fullscreen，已移除） */
  const isFullscreen = view === "login";

  return (
    <SettingsProvider>
    <div className={`app ${isFullscreen ? "fullscreen" : ""} ${isTauri() ? "tauri-frameless" : ""}`}>
      <TitleBar />
      {!isFullscreen && (
        <>
      {/* 移动端顶栏 */}
      <header className="topbar">
        <button className="icon-btn" aria-label="打开菜单" onClick={() => setSidebarOpen(true)}>
          <Icons.menu size={19} />
        </button>
        <span className="topbar-title">{activeTitle}</span>
        <span className="backend-status" aria-live="polite">
          <span className={`status-dot ${backend}`} aria-hidden="true" />
          <span className="status-label">{statusLabel}</span>
        </span>
      </header>

      {/* ★ 2026-10-03：侧栏**完全收起**后的展开入口（桌面）。
          为什么放这里：侧栏收起时 `display:none` ⇒ 品牌行里的折叠按钮也一起没了
          ⇒ 必须在**侧栏之外**留一个入口，否则"收起"就变成无法恢复。
          ⚠️ 只对桌面生效（移动端本来就有 topbar 的菜单按钮）。 */}
      {/* ★ 2026-10-03（用户报"展开时先隐藏悬浮收纳按钮"）：
          **只在侧栏收起时渲染**。此前只用 `@media` 控制显示 ⇒ 展开态按钮也显示，
          而且它 `position:fixed` 在左上角 ⇒ **压在侧栏 logo 上**（见用户截图）。
          ⇒ 改为条件渲染（而非 CSS 显隐）："侧栏是否存在"与"入口是否存在"由**同一个 state** 决定，
             不会再出现两边不同步。
          ⚠️ 位置也一并调整：收起后侧栏已消失，左上角让给内容 ⇒ 按钮缩小、贴更近边缘、
             默认半透明（hover 加深），把对内容的遮挡降到最低。 */}
      {sidebarCollapsed && (
        <button
          type="button"
          className="sidebar-expand-btn"
          onClick={toggleSidebarCollapsed}
          aria-label="展开侧边栏"
          title="展开侧边栏"
        >
          <Icons.chevronRight size={15} />
        </button>
      )}

      {/* 侧栏（桌面常驻 / 移动端抽屉） */}
      <aside
        className={`sidebar ${sidebarOpen ? "sidebar-open" : ""} ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}
      >
        {/* ★ 2026-10-03：折叠按钮放在品牌行右侧 —— 收起后它仍在（图标反转为"展开"），
            所以**任何状态下都有唯一入口**，不会把用户逼到"收起后没法展开"。
            收起态的语义（按钮含义随状态反转）由图标 + aria-label 表达，不用颜色区分。 */}
        <div className="sidebar-brand">
          <span className="brand-mark">
            <Icons.bot size={18} />
          </span>
          <span className="brand-text">
            <span className="brand-name">PaperMate</span>
            <span className="brand-sub">论文智友</span>
          </span>
          <button
            type="button"
            className="sidebar-collapse-btn"
            onClick={toggleSidebarCollapsed}
            aria-expanded={!sidebarCollapsed}
            aria-label={sidebarCollapsed ? "展开侧边栏" : "收起侧边栏"}
            title={sidebarCollapsed ? "展开侧边栏" : "收起侧边栏（把空间留给对话区）"}
          >
            {sidebarCollapsed ? <Icons.chevronRight size={16} /> : <Icons.chevronLeft size={16} />}
          </button>
        </div>

        <button className="new-chat" onClick={onNew} title="新建对话" aria-label="新建对话">
          <span className="plus">
            <Icons.plusIcon size={15} />
          </span>
          <span>新对话</span>
        </button>

        {/* 2026-09-07：首页一键新建工作间（空工作间，后端自动建空文献集 + 会话） */}
        <button
          className="new-workspace"
          onClick={() => {
            createEmptyWorkspace().catch((e) => {
              // 轻量内联错误：侧栏底部短闪，避免弹窗打断首页流
              window.alert(`新建工作间失败：${toErrorMessage(e) || "未知错误"}`);
            });
          }}
          title="新建一个空工作间（可在其后从文献库加入文档）"
        >
          <span className="ws-plus">
            <Icons.layers size={14} />
          </span>
          <span>新建工作间</span>
        </button>

        {/* 主导航（画布 6 项） */}
        <nav className="sidebar-nav">
          {NAV_ITEMS.map(({ key, label, Icon }) => (
            <button
              key={key}
              className={view === key ? "active" : ""}
              /* ★ 2026-10-03：侧栏可折叠成纯图标条 ⇒ title/aria-label 成为**唯一**可读说明，
                 此前只有可见文字，收起后入口含义就完全丢失了。 */
              title={label}
              aria-label={label}
              onClick={() => {
                setView(key);
                setSidebarOpen(false);
              }}
            >
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="session-scroll">
          {/* A2：工作间分组（与会话同级，置顶） */}
          <div className="ws-group">
            <div className="session-label ws-label">
              <Icons.layers size={13} />
              <span>工作间</span>
            </div>
            <ul className="session-list ws-list">
              {workspaceSessions.map((s) => (
                <SessionRow
                  key={s.id}
                  session={s}
                  active={s.id === sessionId}
                  confirming={confirming === s.id}
                  workspace
                  onSelect={() => {
                    setSessionId(s.id);
                    setView("chat");
                    setSidebarOpen(false);
                  }}
                  onRequestDelete={() => requestDelete(s.id)}
                  onRename={(t) => onRename(s.id, t)}
                  onTogglePin={() => onTogglePin(s.id)}
                />
              ))}
              {workspaceSessions.length === 0 && (
                <li
                  className="ws-empty"
                  role="button"
                  tabIndex={0}
                  onClick={() => {
                    setView("docs");
                    setSidebarOpen(false);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setView("docs");
                      setSidebarOpen(false);
                    }
                  }}
                >
                  到「文献」勾选文档创建
                </li>
              )}
            </ul>
          </div>

          <div className="session-label">会话</div>
          <ul className="session-list">
            {chatSessions.map((s) => (
              <SessionRow
                key={s.id}
                session={s}
                active={s.id === sessionId}
                confirming={confirming === s.id}
                onSelect={() => {
                  setSessionId(s.id);
                  setView("chat");
                  setSidebarOpen(false);
                }}
                onRequestDelete={() => requestDelete(s.id)}
                onRename={(t) => onRename(s.id, t)}
                onTogglePin={() => onTogglePin(s.id)}
              />
            ))}
            {chatSessions.length === 0 && <li className="session-empty">还没有会话，点上方新建</li>}
          </ul>
        </div>

        <div className="sidebar-bottom">
          <button
            className="theme-toggle"
            onClick={() => setThemeMode((t) => (t === "dark" ? "light" : t === "light" ? "system" : "dark"))}
            title={`当前：${themeMode === "dark" ? "深色" : themeMode === "light" ? "浅色" : "跟随系统"} · 点击切换`}
          >
            {themeMode === "dark" ? <Icons.moon size={15} /> : <Icons.sun size={15} />}
            <span>{themeMode === "dark" ? "深色" : themeMode === "light" ? "浅色" : "系统"}</span>
          </button>
        </div>
      </aside>

      {/* 移动端侧栏遮罩 */}
      <div
        className={`drawer-mask ${sidebarOpen ? "drawer-open" : ""}`}
        onClick={() => setSidebarOpen(false)}
        aria-hidden="true"
      />

      {/* key={view}：视图切换时重挂载，触发 .main 的进场动效（见 styles.css 动效接入层） */}
      <main className="main" key={view}>
        {actionError && (
          <div className="error-box" style={{ margin: "12px 22px 0", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ flex: 1 }}>{actionError}</span>
            <button className="icon-btn" onClick={() => setActionError(null)} aria-label="关闭提示">
              <Icons.x size={14} />
            </button>
          </div>
        )}
        {/* T-08：懒加载视图统一挂一个 Suspense 边界（占位极轻，见 ViewLoading） */}
        <Suspense fallback={<ViewLoading />}>
        {view === "docs" && (
          <DocumentsPage
            onCreateWorkspace={createWorkspace}
            currentSessionId={sessionId}
            currentSessionIsWorkspace={currentSessionIsWorkspace}
            onAddToSession={addDocsToSession}
            onAddToWorkspace={addDocsToWorkspace}
            onCreateWorkspaceFromCollection={createWorkspaceFromCollection}
          />
        )}
        {view === "chat" && (
          <ChatPage sessionId={sessionId} sessions={sessions} onSessionsChange={setSessions} />
        )}
        {view === "usage" && (
          <UsagePanel
            onOpenChat={(id) => {
              setSessionId(id);
              setView("chat");
            }}
          />
        )}
        {view === "eval" && <EvalPanel />}
        {view === "home" && (
          <HomePage
            onOpenSession={(id) => {
              setSessionId(id);
              setView("chat");
              setSidebarOpen(false);
            }}
            onNavigate={(v) => {
              setView(v);
              setSidebarOpen(false);
            }}
            onNewChat={onNew}
            onCreateWorkspace={() =>
              createEmptyWorkspace().catch((e) => {
                setActionError(`新建工作间失败：${toErrorMessage(e) || "未知错误"}`);
              })
            }
          />
        )}
        {/* 设置：大弹窗（WorkBuddy 风格）—— 覆盖于内容区之上，遮罩/关闭按钮/ESC 均可退出 */}
        {view === "settings" && (
          <div className="wb-modal-mask" onClick={() => setView("chat")} role="presentation">
            <div
              className="wb-modal"
              role="dialog"
              aria-modal="true"
              aria-label="设置"
              onClick={(e) => e.stopPropagation()}
            >
              <button className="wb-modal-close" onClick={() => setView("chat")} aria-label="关闭设置">
                <Icons.x size={16} />
              </button>
              <SettingsPage />
            </div>
          </div>
        )}
        </Suspense>
      </main>

      {/* 移动端底部导航（4 主项） */}
      <nav className="mobile-nav" aria-label="主导航">
        {/* F11/F9：评测与首页走侧栏抽屉入口，移动端底部导航保持原 4 主项 */}
        {NAV_ITEMS.filter(({ key }) => key !== "eval" && key !== "home").map(({ key, label, Icon }) => (
          <button
            key={key}
            className={view === key ? "active" : ""}
            onClick={() => setView(key)}
          >
            <Icon size={20} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
        </>
      )}
      {view === "login" && (
        <Suspense fallback={<ViewLoading />}>
          <LoginPage onNavigate={go} onLogin={() => setView("chat")} />
        </Suspense>
      )}
    </div>
    </SettingsProvider>
  );
}
