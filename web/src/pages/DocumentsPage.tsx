import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { CollectionOut, DocumentOut, SummaryOut, UploadProgress } from "../lib/types";
import { ConfirmModal } from "../components/ConfirmModal";
import { PaperImportQueue, usePaperImportTasks } from "../components/PaperImportQueue";
import { SearchPaperDrawer } from "../components/SearchPaperDrawer";
import { Icons } from "../lib/icons";
import { toErrorMessage } from "../lib/error";
import { timeAgo } from "../lib/time";

/** 上传进度轮询上限（700ms × 300 ≈ 3.5 分钟）——J8 修复：后端卡在非终态时不再无限轮询。 */
const POLL_MAX_ATTEMPTS = 300;

/**
 * 文档库页（画布 6:145）· 批量操作栏扩展（画布 21:96）
 * - 文献列表面板（白底圆角 16：表头行 + 数据行）
 * - 勾选 ≥1 篇 → 蓝底批量栏：研读（总结/相关工作/对比）· 会话动作 · 删除 · 取消选择
 * - 删除 → 确认弹窗（画布 14:207：警告 + 待删列表 + 取消 / 确认删除红按钮）
 * - 底部统计（画布 6:219）
 *
 * 2026-08-23：分页浏览升级——页码导航（窗口化 ≤7 + 省略号）/ 上一页下一页 /
 * 跳转任意页 / 每页条数可选（10/20/50/全部）/ 列表内鼠标滚轮浏览；移除任何
 * “一屏仅若干篇”的浏览限制；后端 list_documents 同步消除 N+1 查询（性能稳定）。
 *
 * 2026-08-29（P0 体验优化）：
 * - F1 恢复研读捷径：选中 1 篇可「总结 / 相关工作」，≥2 篇可「对比 / 相关工作」，
 *   结果就地展示（不再强制跳转问答页）。复用后端 /summarize /related-work /compare，
 *   受 featSummary 门控（403 时提示到设置页开启）。
 * - F2 修复解析状态：后端就绪值为 "ready"，此前误按 "complete" 判断 → 状态点永不置绿、
 *   统计恒“部分解析中”。同时状态列中文化、上传期显示逐文件进度。
 * - F3 关联入口归一：区分「加入当前会话」（不新建）与「新建会话研读」（明确新建），
 *   消除与侧栏「新对话」的概念冲突。
 */

const PAGE_SIZE_OPTIONS = [10, 20, 50] as const;

/**
 * 文档解析状态机（与后端 document_service 一致：uploaded→parsing→chunking→embedding→ready/failed）。
 * 2026-08-29 修复：此前用 status.includes("complete") 判断就绪，但后端就绪值是 "ready"。
 */
const STATUS_LABEL: Record<string, string> = {
  uploaded: "已上传",
  parsing: "解析中",
  chunking: "分块中",
  embedding: "向量化",
  ready: "就绪",
  failed: "失败",
};
const isReady = (s: string) => String(s ?? "").toLowerCase() === "ready";
const isFailed = (s: string) => String(s ?? "").toLowerCase() === "failed";
const statusLabel = (s: string) => STATUS_LABEL[String(s ?? "").toLowerCase()] ?? String(s ?? "");

/** 窗口化页码序列：≤7 页全显；超过则首尾固定 + 当前页邻域 + 省略号 */
function pagerWindow(page: number, total: number): (number | "…")[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const want = new Set<number>([1, total, page - 1, page, page + 1]);
  const nums = [...want].filter((p) => p >= 1 && p <= total).sort((a, b) => a - b);
  const out: (number | "…")[] = [];
  let prev = 0;
  for (const p of nums) {
    if (p - prev > 1) out.push("…");
    out.push(p);
    prev = p;
  }
  return out;
}

/** 研读结果渲染（F1）：总结 / 相关工作 / 对比 共用同一 SummaryOut 结构 */
function ResearchResult({ out }: { out: SummaryOut }) {
  const cits = out.citations ?? [];
  const sections = out.sections ?? [];
  const dims = out.dimensions ?? [];
  return (
    <div className="summary-result">
      {out.narrative && <div className="summary-narrative">{out.narrative}</div>}

      {sections.map((s) => (
        <div className="summary-section" key={s.key}>
          <h4>{s.title}</h4>
          <p>{s.content}</p>
        </div>
      ))}

      {dims.map((dim) => (
        <div className="summary-section" key={dim.key}>
          <h4>{dim.title}</h4>
          {dim.cells.map((c) => (
            <p key={c.doc_id}>
              <strong>{c.filename}</strong>：{c.content}
            </p>
          ))}
        </div>
      ))}

      {out.draft && (
        <details className="summary-citations" open={sections.length === 0}>
          <summary>综述草稿</summary>
          <div className="summary-narrative">{out.draft}</div>
        </details>
      )}

      {cits.length > 0 && (
        <details className="summary-citations">
          <summary>引用来源（{cits.length}）</summary>
          <ul>
            {cits.map((c, i) => (
              <li key={`${c.id ?? i}-${i}`}>
                {c.page != null ? `第 ${c.page} 页` : "（无页码）"}：{(c.text ?? "").slice(0, 140)}
              </li>
            ))}
          </ul>
        </details>
      )}

      <div className="muted">生成模型：{out.model}</div>
    </div>
  );
}

export function DocumentsPage({
  onCreateWorkspace,
  currentSessionId,
  currentSessionIsWorkspace,
  onAddToSession,
  onAddToWorkspace,
  onCreateWorkspaceFromCollection,
}: {
  /**
   * B1：以所选文档创建工作间（原子建文献集 + 工作间会话）。
   * 由原「新建会话研读」重定位而来 —— D1 决策：会话-文献关联入口收敛进工作间，不保留双入口。
   */
  onCreateWorkspace?: (docIds: number[], name: string) => Promise<void> | void;
  /** 当前会话 id（用于「加入当前会话 / 加入工作间」；为 null 时该入口不显示） */
  currentSessionId?: number | null;
  /** 当前会话是否为工作间：决定「加入 *」入口的文案与目标 */
  currentSessionIsWorkspace?: boolean;
  /** 把文档加入当前【普通会话】的检索范围 */
  onAddToSession?: (docIds: number[]) => void;
  /** B1：把文档加入当前【工作间】的文献集（幂等去重） */
  onAddToWorkspace?: (docIds: number[]) => Promise<void> | void;
  /**
   * B1：以整个文献集创建工作间（关联已有文献集，不重复建集）。
   * 由原「研读」（新建普通会话关联文献集）重定位而来。
   */
  onCreateWorkspaceFromCollection?: (collectionId: number, name: string) => Promise<void> | void;
}) {
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [deleteTarget, setDeleteTarget] = useState<DocumentOut | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [batchDeleteOpen, setBatchDeleteOpen] = useState(false);
  const [batchDeleting, setBatchDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);
  /** 组件存活标记：研读等异步操作在页面切换（卸载）后不再 setState，避免无效渲染与内存滞留 */
  const aliveRef = useRef(true);
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  // ---- 研读捷径（F1）：就地产出结果，不再强制跳转问答页 ----
  const [researchKind, setResearchKind] = useState<"summary" | "related" | "compare" | null>(null);
  const [researchResult, setResearchResult] = useState<SummaryOut | null>(null);
  const [researchError, setResearchError] = useState<string | null>(null);
  const [researching, setResearching] = useState(false);

  // ---- 文献集（F5）：按课题/项目组织文档；会话可关联整个文献集 ----
  const [collections, setCollections] = useState<CollectionOut[]>([]);
  const [collOpen, setCollOpen] = useState(true);
  const [newCollName, setNewCollName] = useState("");
  const [manageColl, setManageColl] = useState<CollectionOut | null>(null);
  const [addToCollOpen, setAddToCollOpen] = useState(false);
  /** F5：文献集操作错误提示（此前失败静默、乐观更新谎报成功，用户无从感知） */
  const [collError, setCollError] = useState<string | null>(null);

  // B1：创建工作间命名弹窗（「勾选文档」与「文献集」两个入口共用同一套交互）
  const [wsNameOpen, setWsNameOpen] = useState<
    { kind: "docs"; docIds: number[] } | { kind: "collection"; cid: number; defaultName: string } | null
  >(null);
  const [wsName, setWsName] = useState("");
  const [wsCreating, setWsCreating] = useState(false);
  const [wsError, setWsError] = useState<string | null>(null);
  /** 删除二次确认：行内删除按钮先进入警示态，再次点击才真正触发 */
  const [confirmDelId, setConfirmDelId] = useState<number | null>(null);
  const [confirmCollId, setConfirmCollId] = useState<number | null>(null);
  useEffect(() => {
    if (confirmDelId == null) return;
    const t = setTimeout(() => setConfirmDelId(null), 3000);
    return () => clearTimeout(t);
  }, [confirmDelId]);
  useEffect(() => {
    if (confirmCollId == null) return;
    const t = setTimeout(() => setConfirmCollId(null), 3000);
    return () => clearTimeout(t);
  }, [confirmCollId]);

  const loadCollections = useCallback(async () => {
    try {
      const r = await api.listCollections();
      setCollections(r.items ?? []);
    } catch (e) {
      // 加载失败不阻断文献库使用，但需留痕便于排查
      console.warn("[collections] 加载失败：", e);
    }
  }, []);
  useEffect(() => {
    loadCollections();
  }, [loadCollections]);

  /** 新建文献集：失败返回 null 并由调用方提示；成功才由调用方清空输入（不再乐观清空） */
  const createColl = useCallback(async (name: string, docIds: number[] = []) => {
    try {
      const c = await api.createCollection(name, "", docIds);
      setCollections((prev) => [...prev, c]);
      setCollError(null);
      return c;
    } catch (e: unknown) {
      setCollError(toErrorMessage(e) || "新建文献集失败");
      return null;
    }
  }, []);

  // ---- 分页状态：页码 + 每页条数（"all" = 一次加载全部，配合列表滚轮浏览） ----
  // 默认 10 篇/页：保证文档数超过一页时分页器必然可见（此前默认 20 导致 12 篇只一页、
  // 分页器不渲染，用户感知“功能未实现”）。——2026-08-23 修正
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState<number | "all">(10);
  const [jumpTo, setJumpTo] = useState("");

  /** 刷新请求序号（J8 修复 2026-09-10）：并发 refresh 时旧响应后到会覆盖新列表
   *  （如「删除后刷新」被「轮询触发的刷新」的过期响应覆盖 → 已删文档重现）。 */
  const refreshSeqRef = useRef(0);

  const refresh = useCallback(async () => {
    const seq = ++refreshSeqRef.current;
    setLoading(true);
    setError(null);
    try {
      const list = await api.listDocuments();
      if (seq !== refreshSeqRef.current) return; // 过期响应丢弃
      setDocs(list);
    } catch (e: unknown) {
      if (seq !== refreshSeqRef.current) return;
      setError(toErrorMessage(e) || "加载文档失败");
    } finally {
      if (seq === refreshSeqRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // ---- 上传进度（2026-09-02 异步化）：异步上传后轮询每篇进度，行内展示进度条 + 取消按钮 ----
  const [progressMap, setProgressMap] = useState<Record<number, UploadProgress>>({});
  const [trackingIds, setTrackingIds] = useState<Set<number>>(new Set());

  /** 把刚异步上传的文档加入列表并开始轮询其进度（不在列表里才插入，避免分页重影） */
  const startTracking = useCallback((doc: DocumentOut) => {
    setDocs((prev) => (prev.some((x) => x.id === doc.id) ? prev : [doc, ...prev]));
    setTrackingIds((prev) => {
      const n = new Set(prev);
      n.add(doc.id);
      return n;
    });
  }, []);

  /** 停止轮询某篇的上传进度（完成/失败/取消后调用） */
  const stopTracking = useCallback((id: number) => {
    setTrackingIds((prev) => {
      const n = new Set(prev);
      n.delete(id);
      return n;
    });
    setProgressMap((prev) => {
      const n = { ...prev };
      delete n[id];
      return n;
    });
  }, []);

  /** 取消正在进行的上传：请求后端停止并清理半成品，随后刷新列表 */
  const onCancelUpload = useCallback(
    async (id: number) => {
      try {
        await api.cancelUpload(id);
      } catch {
        /* 取消请求本身失败不阻断后续刷新 */
      }
      stopTracking(id);
      await refresh();
    },
    [refresh, stopTracking],
  );

  // 轮询进行中的上传进度；任一篇进入终态（ready/failed/cancelled / 已删除）即停止其轮询并刷新列表
  // J8 修复（2026-09-10）：加轮询次数上限——后端卡在非终态时原先 700ms 无限轮询（耗电 + 无意义请求）。
  useEffect(() => {
    if (trackingIds.size === 0) return;
    let alive = true;
    let attempts = 0;
    const poll = async () => {
      attempts += 1;
      if (attempts > POLL_MAX_ATTEMPTS) {
        // 超时上限（≈3.5 分钟）：停止跟踪并提示，交由用户手动刷新查看最终状态
        setError("上传进度跟踪超时，请刷新页面查看最新状态");
        setTrackingIds(new Set());
        return;
      }
      const ids = Array.from(trackingIds);
      let terminal = false;
      await Promise.all(
        ids.map(async (id) => {
          try {
            const p = await api.getUploadProgress(id);
            if (p == null) {
              stopTracking(id);
              terminal = true;
              return;
            }
            setProgressMap((prev) => ({ ...prev, [id]: p }));
            if (p.status === "ready" || p.status === "failed" || p.status === "cancelled") {
              stopTracking(id);
              terminal = true;
            }
          } catch {
            stopTracking(id);
            terminal = true;
          }
        }),
      );
      if (terminal && alive && aliveRef.current) await refresh();
    };
    const iv = setInterval(poll, 700);
    void poll();
    return () => {
      alive = false;
      clearInterval(iv);
    };
  }, [trackingIds, stopTracking, refresh]);

  /**
   * 研读捷径（F1）：总结 / 相关工作 / 对比。
   * 复用后端保留的 /summarize /related-work /compare（受 features.summary 门控，
   * 关闭时后端返回 403，此处透传原文提示用户去设置页开启）。
   */
  const runResearch = useCallback(
    async (kind: "summary" | "related" | "compare", docIds: number[]) => {
      if (docIds.length === 0) return;
      if (kind === "compare" && docIds.length < 2) {
        setResearchError("对比至少需要选择 2 篇文献");
        setResearchKind(kind);
        setResearchResult(null);
        return;
      }
      setResearching(true);
      setResearchError(null);
      setResearchResult(null);
      setResearchKind(kind);
      try {
        const out =
          kind === "summary"
            ? await api.summarize(docIds)
            : kind === "related"
              ? await api.relatedWork(docIds)
              : await api.compare(docIds);
        if (!aliveRef.current) return;
        setResearchResult(out);
      } catch (e: unknown) {
        if (!aliveRef.current) return;
        setResearchError(toErrorMessage(e) || "研读失败");
      } finally {
        if (aliveRef.current) setResearching(false);
      }
    },
    [],
  );

  const totalPages = useMemo(
    () => (pageSize === "all" ? 1 : Math.max(1, Math.ceil(docs.length / pageSize))),
    [docs.length, pageSize],
  );

  /** 页码越界自动收敛（删除文档 / 切换每页条数 / 数据刷新后） */
  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  const pageDocs = useMemo(() => {
    if (pageSize === "all") return docs;
    const start = (page - 1) * pageSize;
    return docs.slice(start, start + pageSize);
  }, [docs, page, pageSize]);

  const goToPage = useCallback(
    (p: number) => {
      setPage(Math.min(Math.max(1, p), totalPages));
      listRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    },
    [totalPages],
  );

  /** 滚轮浏览：列表自身可滚时交给原生滚动；滚到顶/底边界时翻页。
   * 不依赖 CSS 级联，避免"DOM 有 overflow 但用户滚轮无效"的盲区。
   * 用原生非被动监听才能 preventDefault 翻页，React onWheel 默认被动会报 warning。 */
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (pageSize === "all") return;
      const scrollable = el.scrollHeight > el.clientHeight + 2;
      if (!scrollable) return; // 列表本身不可滚动则放行页面滚动
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
      const atTop = el.scrollTop <= 0;
      if (e.deltaY > 0 && atBottom && page < totalPages) {
        e.preventDefault();
        goToPage(page + 1);
      } else if (e.deltaY < 0 && atTop && page > 1) {
        e.preventDefault();
        goToPage(page - 1);
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [pageSize, page, totalPages, goToPage]);

  const submitJump = () => {
    const n = parseInt(jumpTo, 10);
    if (Number.isNaN(n)) return;
    goToPage(n);
    setJumpTo("");
  };

  /** 全选/取消全选：作用于当前页可见项（跨页选择保留，批量操作按全量计算） */
  const toggleAll = useCallback(() => {
    setSelected((prev) => {
      const ids = pageDocs.map((d) => d.id);
      const allSelected = ids.length > 0 && ids.every((id) => prev.has(id));
      const next = new Set(prev);
      if (allSelected) ids.forEach((id) => next.delete(id));
      else ids.forEach((id) => next.add(id));
      return next;
    });
  }, [pageDocs]);

  const toggle = useCallback((id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  /** 确认删除（画布 14:207 弹窗） */
  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await api.deleteDocument(deleteTarget.id);
      setDeleteTarget(null);
      setSelected((prev) => {
        const next = new Set(prev);
        next.delete(deleteTarget.id);
        return next;
      });
      await refresh();
    } catch (e: unknown) {
      setError(toErrorMessage(e) || "删除失败");
    } finally {
      setDeleting(false);
    }
  }, [deleteTarget, refresh]);

  /** 批量删除：串行执行，失败记录首条错误，最后刷新列表 */
  const confirmBatchDelete = useCallback(async () => {
    if (!selected.size) return;
    setBatchDeleting(true);
    setError(null);
    const ids = Array.from(selected);
    let failed = 0;
    let firstErr = "";
    for (const id of ids) {
      try {
        await api.deleteDocument(id);
      } catch (e: unknown) {
        failed += 1;
        if (!firstErr) firstErr = toErrorMessage(e) || "删除失败";
      }
    }
    setBatchDeleting(false);
    setBatchDeleteOpen(false);
    setSelected(new Set());
    await refresh();
    if (failed) {
      setError(`批量删除部分失败：${firstErr}（${failed}/${ids.length} 篇未删除）`);
    }
  }, [selected, refresh]);

  const totalChunks = docs.reduce((n, d) => n + (d.num_chunks || 0), 0);
  const allParsed = docs.length > 0 && docs.every((d) => isReady(d.status));
  const selectedChunks = docs.filter((d) => selected.has(d.id)).reduce((n, d) => n + (d.num_chunks || 0), 0);
  const pageSelected = pageDocs.filter((d) => selected.has(d.id)).length;
  const pager = pagerWindow(page, totalPages);

  /** B1：提交创建工作间（「勾选文档」与「文献集」两个入口共用） */
  const submitWorkspace = useCallback(async () => {
    if (!wsNameOpen) return;
    setWsCreating(true);
    setWsError(null);
    try {
      if (wsNameOpen.kind === "docs") {
        await onCreateWorkspace?.(wsNameOpen.docIds, wsName.trim());
      } else {
        await onCreateWorkspaceFromCollection?.(wsNameOpen.cid, wsName.trim());
      }
      setWsNameOpen(null);
      setWsName("");
      setSelected(new Set());
    } catch (e: unknown) {
      // 失败保留弹窗与已输入名称，便于改名重试
      setWsError(toErrorMessage(e) || "创建工作间失败，请确认后端服务正常");
    } finally {
      setWsCreating(false);
    }
  }, [wsNameOpen, wsName, onCreateWorkspace, onCreateWorkspaceFromCollection]);


    // 论文在线检索（阶段 3）：任务状态与轮询都在 hook 里，本页只负责渲染
    const [paperOpen, setPaperOpen] = useState(false);
    const paperImport = usePaperImportTasks();
  return (
    <div className="doc-library-page">
      <header className="usage-header page-head-row">
        <div>
          <h2>文献</h2>
          <p className="muted">管理你的论文库 · 共 {docs.length} 篇文献</p>
        </div>
        <div className="page-header-actions">
            <button className="ghost-btn" onClick={() => setPaperOpen(true)}>
              检索论文
            </button>
          <label className="upload-btn primary">
            <Icons.upload size={15} />
          上传文档
            <input
              ref={inputRef}
              type="file"
              accept=".pdf,.md,.markdown,.txt,.doc,.docx"
              multiple
              hidden
                onChange={async (e) => {
                  const files = e.target.files ? Array.from(e.target.files) : [];
                  if (files.length === 0) return;
                  const errs: string[] = [];
                  // 异步上传：POST 仅保存文件并立即返回（status=uploaded），重活在后台进行；
                  // 随后由进度轮询 effect 行内展示进度条，并支持随时取消。
                  for (const f of files) {
                    try {
                      const doc = await api.uploadDocumentAsync(f, currentSessionId ?? undefined);
                      startTracking(doc);
                    } catch (err: unknown) {
                      const apiErr = err as { status?: number; payload?: { existing_id?: number; existing_filename?: string } };
                      if (apiErr.status === 409) {
                        // 2026-09-06 服务端内容去重：后端已跳过入库，提示已有文档（payload 带其 id/文件名）
                        const ex = apiErr.payload ?? {};
                        errs.push(`${f.name} 与已有文档重复（#${ex.existing_id ?? "?"}「${ex.existing_filename ?? ""}」，未重复上传）`);
                      } else {
                        errs.push(`${f.name}（${toErrorMessage(err) || "未知错误"}）`);
                      }
                    }
                  }
                  e.target.value = "";
                  if (errs.length) setError(`上传失败 ${errs.length}/${files.length} 个：${errs.join("；")}`);
                }}
            />
          </label>
          <button className="ghost-btn" onClick={refresh} disabled={loading}>
            <Icons.refresh size={14} />
            {loading ? "加载中…" : "刷新"}
          </button>
        </div>
      </header>

        <PaperImportQueue
          tasks={paperImport.tasks}
          docProg={paperImport.docProg}
          onCancel={(id) => void paperImport.cancel(id)}
          onRetry={(t) => void paperImport.retry(t)}
          onOpenDoc={(docId) => {
            void onAddToSession?.([docId]);
            void refresh();
          }}
          onDismiss={paperImport.dismiss}
        />

        <SearchPaperDrawer
          open={paperOpen}
          onClose={() => setPaperOpen(false)}
          onSubmitted={(tasks) => {
            paperImport.add(tasks);
            void refresh();
          }}
        />

      {/* F5：文献集（Collections）——按课题/项目分组组织文档，可一键「研读」整个课题。
          放在左栏并自带滚动：新建条目不会把右侧文献表整体推下去（此前「新建后下方不可见」的根因）。 */}
      <div className="docs-layout">
        <aside className="docs-side">
            <section className="collections-panel">
          <div className="collections-head" onClick={() => setCollOpen((v) => !v)} role="button" tabIndex={0}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setCollOpen((v) => !v); } }}>
            <span className="collections-title"><Icons.book size={14} /> 文献集 <span className="collections-count">{collections.length}</span></span>
            <span className="collections-chevron" aria-hidden="true">{collOpen ? <Icons.chevronDown size={14} /> : <Icons.chevronRight size={14} />}</span>
          </div>
          {collOpen && (
            <div className="collections-body">
              {collError && <div className="error-box">{collError}</div>}
              <div className="collections-new">
                <input
                  className="collections-new-input"
                  placeholder="新建文献集名称"
                  value={newCollName}
                  onChange={(e) => setNewCollName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && newCollName.trim()) {
                      // 仅在创建成功后清空输入（失败时保留用户输入并提示）
                      void createColl(newCollName.trim()).then((c) => { if (c) setNewCollName(""); });
                    }
                  }}
                />
                <button
                  className="batch-btn primary"
                  disabled={!newCollName.trim()}
                  onClick={() => {
                    void createColl(newCollName.trim()).then((c) => { if (c) setNewCollName(""); });
                  }}
                ><Icons.plus size={13} />新建</button>
              </div>
              {collections.length === 0 ? (
                <div className="muted">暂无文献集。把多篇论文归为一组，可一键「研读」整个课题（多选文献后也可「归入文献集」）。</div>
              ) : (
                <ul className="collections-list">
                  {collections.map((c) => (
                    <li key={c.id} className="collection-item">
                      <span className="collection-name" title={c.name}>{c.name}</span>
                      <span className="collection-meta">{c.doc_ids.length} 篇</span>
                      <span className="collection-actions">
                        {/* B1：由「研读」（建普通会话关联文献集）重定位 —— 以文献集创建工作间 */}
                        <button
                          className="batch-btn"
                          onClick={() => {
                            setWsError(null);
                            setWsName(c.name);
                            setWsNameOpen({ kind: "collection", cid: c.id, defaultName: c.name });
                          }}
                          title="以整个文献集创建工作间（关联已有文献集，不重复建集）"
                        >
                          <Icons.layers size={12} />
                          建工作间
                        </button>
                        <button className="batch-btn ghost" onClick={() => setManageColl(c)}>管理</button>
                        <button
                          className={`batch-btn danger ${confirmCollId === c.id ? "is-confirming" : ""}`}
                          title={confirmCollId === c.id ? "再次点击确认删除" : "删除文献集（仅解除分组，不会删除论文）"}
                          onClick={() => {
                            if (confirmCollId === c.id) {
                              setConfirmCollId(null);
                              (async () => {
                                try {
                                  await api.deleteCollection(c.id);
                                  setCollections((prev) => prev.filter((x) => x.id !== c.id));
                                  setCollError(null);
                                } catch (e: unknown) {
                                  setCollError(toErrorMessage(e) || "删除文献集失败");
                                }
                              })();
                            } else {
                              setConfirmCollId(c.id);
                            }
                          }}
                        >{confirmCollId === c.id ? <Icons.alert size={12} /> : <Icons.trash size={12} />}{confirmCollId === c.id ? "确认?" : "删除"}</button>
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </section>

      </aside>



      <div className="docs-main">

        {error && <div className="error-box">{error}</div>}

        {/* 异步上传进度说明（2026-09-02）：逐篇进度条 + 取消按钮直接渲染在下方文献行内，
            此处不再用全局横幅，避免遮挡列表。 */}

        {/* 批量操作栏（画布 21:96，勾选 ≥1 后出现） */}
        {selected.size > 0 && (
          <div className="batch-bar">
            <span className="batch-info">
              已选 {selected.size} 篇（共 {docs.length} 篇）· 分块 {selectedChunks} 块
            </span>
            <button className="batch-btn ghost" onClick={() => setSelected(new Set())}>
              <Icons.x size={13} />
              取消选择
            </button>
            <span className="batch-spacer" />
            {/* 研读捷径（F1）：就地出结果，无需跳转问答页 */}
            <button
              className="batch-btn"
              disabled={researching}
              onClick={() => runResearch("summary", Array.from(selected))}
              title="对所选文献生成结构化总结（背景 / 方法 / 实验 / 结论 / 局限）"
            >
              <Icons.fileText size={13} />
              总结
            </button>
            <button
              className="batch-btn"
              disabled={researching}
              onClick={() => runResearch("related", Array.from(selected))}
              title="生成相关工作综述草稿（含引文来源）"
            >
              <Icons.link size={13} />
              相关工作
            </button>
            <button
              className="batch-btn"
              disabled={researching || selected.size < 2}
              onClick={() => runResearch("compare", Array.from(selected))}
              title={selected.size < 2 ? "对比至少需要选择 2 篇文献" : "多篇逐维度并列对比 + 共性/差异叙述"}
            >
              <Icons.arrowUpDown size={13} />
              对比
            </button>
            {/* B1：按当前会话类型分流 —— 工作间加入其文献集，普通会话加入检索范围 */}
            {currentSessionId != null &&
              (currentSessionIsWorkspace ? (
                <button
                  className="batch-btn"
                  onClick={() => onAddToWorkspace?.(Array.from(selected))}
                  title="把所选文献加入当前工作间的文献集（已在其中的自动跳过）"
                >
                  <Icons.plus size={13} />
                  加入工作间
                </button>
              ) : (
                <button
                  className="batch-btn"
                  onClick={() => onAddToSession?.(Array.from(selected))}
                  title="把所选文献加入当前会话的检索范围（不新建会话）"
                >
                  <Icons.plus size={13} />
                  加入当前会话
                </button>
              ))}
            {/* F5：把所选文献归入某个文献集（或新建后归入），支撑多论文课题组织 */}
            <button
              className="batch-btn"
              onClick={() => setAddToCollOpen(true)}
              title="把所选文献归入某个文献集（或新建后归入）"
            >
              <Icons.folder size={13} />
              归入文献集
            </button>
            <button className="batch-btn danger" onClick={() => setBatchDeleteOpen(true)}>
              <Icons.trash size={13} />
              删除
            </button>
            {/* B1：由「新建会话研读」重定位 —— 创建工作间（命名 → 原子建集 + 建会话） */}
            <button
              className="batch-btn primary"
              onClick={() => {
                setWsError(null);
                setWsName("");
                setWsNameOpen({ kind: "docs", docIds: Array.from(selected) });
              }}
              title="以所选文献创建工作间（命名后进入研读）"
            >
              <Icons.layers size={13} />
              创建工作间
            </button>
          </div>
        )}

        {/* 文献列表面板（画布 6:168：表头行 + 数据行；列表区支持滚轮浏览） */}
        <div className="doc-table-panel">
          <div className="doc-table-head">
            <label className="doc-select">
              <input
                type="checkbox"
                checked={pageDocs.length > 0 && pageSelected === pageDocs.length}
                onChange={toggleAll}
                aria-label="全选当前页"
                title="全选 / 取消全选当前页"
              />
            </label>
            <span className="doc-th file">文件名</span>
            <span className="doc-th status">状态</span>
            <span className="doc-th chunks">分块</span>
            <span className="doc-th ops">操作</span>
          </div>
          <ul className="doc-list" ref={listRef}>
            {loading && docs.length === 0 ? (
              <li className="doc-loading-row">
                <span className="spinner" aria-hidden="true" />
                加载文献中…
              </li>
            ) : (
              pageDocs.map((d) => (
                <li key={d.id} className={`doc-table-row ${selected.has(d.id) ? "selected" : ""}`}>
                  <label className="doc-select">
                    <input
                      type="checkbox"
                      checked={selected.has(d.id)}
                      onChange={() => toggle(d.id)}
                      aria-label={`选择 ${d.filename}`}
                    />
                  </label>
                  <span className="doc-table-name" title={d.filename}>
                    <span className="doc-table-name-text">{d.filename}</span>
                    {d.created_at && (
                      <span className="doc-table-time" title={new Date(d.created_at).toLocaleString()}>
                        {timeAgo(d.created_at)}
                      </span>
                    )}
                  </span>
                  {(() => {
                    const prog = progressMap[d.id];
                    const uploadingNow =
                      prog != null && ["uploaded", "parsing", "chunking", "embedding"].includes(prog.status);
                    return (
                      <span className="doc-table-status" title={String(d.status)}>
                        <span
                          className={`status-dot ${isReady(d.status) ? "ok" : isFailed(d.status) ? "err" : ""}`}
                          aria-hidden="true"
                        />
                        {statusLabel(d.status)}
                        {uploadingNow && (
                          <span className="upload-bar" title={`${prog!.done}/${prog!.total} · ${prog!.pct}%`}>
                            <span className="upload-bar-fill" style={{ width: `${prog!.pct}%` }} />
                          </span>
                        )}
                      </span>
                    );
                  })()}
                  <span className="doc-table-chunks">{d.num_chunks || 0} 块</span>
                  <span className="doc-table-ops">
                    {(() => {
                      const prog = progressMap[d.id];
                      const uploadingNow =
                        prog != null && ["uploaded", "parsing", "chunking", "embedding"].includes(prog.status);
                      return uploadingNow ? (
                        <button
                          className="doc-cancel"
                          onClick={() => onCancelUpload(d.id)}
                          title="取消上传（协作式：后台任务在阶段边界停止并清理半成品）"
                        >
                          <Icons.x size={12} />
                          取消
                        </button>
                      ) : null;
                    })()}
                    {/* B1：与批量栏同口径 —— 工作间加入文献集，普通会话加入检索范围 */}
                    {currentSessionId != null &&
                      (currentSessionIsWorkspace ? (
                        <button
                          className="doc-read"
                          onClick={() => onAddToWorkspace?.([d.id])}
                          title="把该文献加入当前工作间的文献集（已在其中的自动跳过）"
                        >
                          <Icons.plus size={12} />
                          加入工作间
                        </button>
                      ) : (
                        <button
                          className="doc-read"
                          onClick={() => onAddToSession?.([d.id])}
                          title="把该文献加入当前会话的检索范围（不新建会话）"
                        >
                          <Icons.plus size={12} />
                          加入会话
                        </button>
                      ))}
                    {/* B1：由「去问答」（建普通会话关联该文献）重定位 —— 以该文献创建工作间 */}
                    <button
                      className="doc-read"
                      onClick={() => {
                        setWsError(null);
                        setWsName(d.filename.replace(/\.[^.]+$/, ""));
                        setWsNameOpen({ kind: "docs", docIds: [d.id] });
                      }}
                      title="以该文献创建工作间（命名后进入研读）"
                    >
                      <Icons.layers size={12} />
                      建工作间
                    </button>
                    <button
                      className={`doc-del ${confirmDelId === d.id ? "is-confirming" : ""}`}
                      title={confirmDelId === d.id ? "再次点击确认删除" : "删除该文献"}
                      onClick={() => {
                        if (confirmDelId === d.id) {
                          setConfirmDelId(null);
                          setDeleteTarget(d);
                        } else {
                          setConfirmDelId(d.id);
                        }
                      }}
                    >
                      {confirmDelId === d.id ? <Icons.alert size={12} /> : <Icons.trash size={12} />}
                      {confirmDelId === d.id ? "确认?" : "删除"}
                    </button>
                  </span>
                </li>
              ))
            )}
            {!loading && docs.length === 0 && (
              <li className="doc-empty doc-table-empty">暂无文档，点击右上角「上传文档」开始</li>
            )}
          </ul>

          {/* 分页器：页码导航 / 跳转 / 每页条数。
              2026-09-08 修复：原条件 `totalPages > 1` 在 pageSize==="all" 时恒为 1
              （见 :361-364），导致整个分页器连同「每页条数」选择器一起消失 →
              选了「全部」后再也切不回分页（死路）。放宽为「或已选全部」。 */}
          {docs.length > 0 && (totalPages > 1 || pageSize === "all") && (
            <div className="doc-pager">
              <div className="doc-pager-nav">
                <button className="pager-btn" disabled={page <= 1} onClick={() => goToPage(page - 1)}>
                  <Icons.chevronLeft size={12} />
                  上一页
                </button>
                {pager.map((p, i) =>
                  p === "…" ? (
                    <span key={`dots-${i}`} className="pager-dots">…</span>
                  ) : (
                    <button
                      key={p}
                      className={`pager-btn ${p === page ? "active" : ""}`}
                      onClick={() => goToPage(p)}
                      aria-current={p === page ? "page" : undefined}
                    >
                      {p}
                    </button>
                  ),
                )}
                <button className="pager-btn" disabled={page >= totalPages} onClick={() => goToPage(page + 1)}>
                  下一页
                  <Icons.chevronRight size={12} />
                </button>
              </div>
              <div className="doc-pager-tools">
                <label className="pager-size">
                  每页
                  <select
                    value={String(pageSize)}
                    onChange={(e) => {
                      setPageSize(e.target.value === "all" ? "all" : Number(e.target.value));
                      setPage(1);
                    }}
                    aria-label="每页条数"
                  >
                    {PAGE_SIZE_OPTIONS.map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                    <option value="all">全部</option>
                  </select>
                  篇
                </label>
                <label className="pager-jump">
                  跳至
                  <input
                    type="number"
                    min={1}
                    max={totalPages}
                    value={jumpTo}
                    onChange={(e) => setJumpTo(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && submitJump()}
                    placeholder={`1-${totalPages}`}
                    aria-label="跳转页码"
                  />
                  <button className="pager-btn" onClick={submitJump}>
                    GO
                  </button>
                </label>
                <span className="pager-info">
                  第 {page} / {totalPages} 页 · 共 {docs.length} 篇{pageSize === "all" ? "（全部）" : ""}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* 研读结果面板（F1）：文献页就地产出，避免强制跳转问答页 */}
        {(researching || researchError || researchResult) && (
          <div className="research-panel">
            <div className="research-head">
              <span className="research-title">
                {researchKind === "compare"
                  ? "多篇对比"
                  : researchKind === "related"
                    ? "相关工作"
                    : "结构化总结"}
              </span>
              <span className="research-spacer" />
              <button
                className="icon-btn"
                aria-label="关闭研读结果"
                onClick={() => {
                  setResearchResult(null);
                  setResearchError(null);
                  setResearchKind(null);
                }}
              ><Icons.x size={14} /></button>
            </div>
            {researching && (
              <div className="research-loading">
                <span className="spinner" aria-hidden="true" />
                正在研读所选文献…
              </div>
            )}
            {researchError && <div className="error-box">{researchError}</div>}
            {researchResult && <ResearchResult out={researchResult} />}
          </div>
        )}

        {/* 列表底部统计（画布 6:219） */}
        {docs.length > 0 && (
          <p className="doc-list-footer">
            共 {docs.length} 篇文献 · {allParsed ? "全部解析完成" : "部分解析中"} · 总计 {totalChunks.toLocaleString()} 个检索分块
            {pageSize !== "all" && totalPages > 1 ? ` · 当前页 ${pageDocs.length} 篇（滚轮滚列表 · 滚到顶/底自动翻页）` : " · 列表内可滚轮滚动浏览"}
          </p>
        )}


      </div>
</div>


      {/* 删除确认弹窗（画布 14:207） */}
      <ConfirmModal
        open={deleteTarget !== null}
        title="删除文献"
        onClose={() => setDeleteTarget(null)}
        footer={
          <>
            <button className="modal-btn ghost" onClick={() => setDeleteTarget(null)} disabled={deleting}>
              取消
            </button>
            <button className="modal-btn danger" onClick={confirmDelete} disabled={deleting}>
              {deleting ? "删除中…" : "确认删除"}
            </button>
          </>
        }
      >
        <p className="modal-desc">
          你即将删除以下文献，删除后将同时清除其所有检索分块、引用与相关笔记。此操作不可撤销。
        </p>
        <ul className="modal-dup-list">
          {deleteTarget && (
            <li>
              <span className="dup-file">{deleteTarget.filename}</span>
              <span className="dup-meta">
                {deleteTarget.status} · {deleteTarget.num_chunks || 0} 个分块
              </span>
            </li>
          )}
        </ul>
      </ConfirmModal>

      {/* 批量删除确认弹窗 */}
      <ConfirmModal
        open={batchDeleteOpen}
        title={`批量删除 ${selected.size} 篇文献`}
        onClose={() => !batchDeleting && setBatchDeleteOpen(false)}
        footer={
          <>
            <button className="modal-btn ghost" onClick={() => setBatchDeleteOpen(false)} disabled={batchDeleting}>
              取消
            </button>
            <button className="modal-btn danger" onClick={confirmBatchDelete} disabled={batchDeleting}>
              {batchDeleting ? "删除中…" : `确认删除 ${selected.size} 篇`}
            </button>
          </>
        }
      >
        <p className="modal-desc">
          你即将删除以下 {selected.size} 篇文献，删除后将同时清除其所有检索分块、引用与相关笔记。此操作不可撤销。
        </p>
        <ul className="modal-dup-list">
          {docs
            .filter((d) => selected.has(d.id))
            .map((d) => (
              <li key={d.id}>
                <span className="dup-file">{d.filename}</span>
                <span className="dup-meta">
                  {d.status} · {d.num_chunks || 0} 个分块
                </span>
              </li>
            ))}
        </ul>
      </ConfirmModal>

      {/* F5：文献集管理弹窗（重命名 / 增删文档） */}
      {manageColl && (
        <div className="modal-mask" onClick={() => setManageColl(null)} aria-modal="true" role="dialog">
          <div className="modal coll-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="reader-title">管理文献集：{manageColl.name}</div>
              <button className="icon-btn" onClick={() => setManageColl(null)} aria-label="关闭"><Icons.x size={14} /></button>
            </div>
            <div className="coll-manage-body">
              <div className="coll-manage-row">
                <input
                  className="collections-new-input"
                  value={manageColl.name}
                  onChange={(e) => setManageColl({ ...manageColl, name: e.target.value })}
                />
                <button
                  className="batch-btn"
                  onClick={async () => {
                    // 失败时用于回滚：此刻 collections 中仍是服务端已确认的原名
                    const oldName = collections.find((c) => c.id === manageColl.id)?.name ?? manageColl.name;
                    try {
                      const updated = await api.renameCollection(manageColl.id, manageColl.name, manageColl.description);
                      // 以服务端返回为准（名称可能被归一化：空白→默认名、超长截断）
                      const name = updated?.name ?? manageColl.name;
                      setCollections((prev) => prev.map((c) => (c.id === manageColl.id ? { ...c, name } : c)));
                      setManageColl({ ...manageColl, name });
                      setCollError(null);
                    } catch (e: unknown) {
                      setCollError(toErrorMessage(e) || "重命名失败");
                      // 回滚输入框与列表，避免显示一个从未保存成功的名字
                      setManageColl({ ...manageColl, name: oldName });
                      setCollections((prev) => prev.map((c) => (c.id === manageColl.id ? { ...c, name: oldName } : c)));
                    }
                  }}
                >重命名</button>
              </div>
              <div className="muted">集内论文（{manageColl.doc_ids.length}）：</div>
              {manageColl.doc_ids.length === 0 ? (
                <div className="muted">空集合</div>
              ) : (
                <ul className="coll-doc-list">
                  {manageColl.doc_ids.map((did) => {
                    const d = docs.find((x) => x.id === did);
                    return (
                      <li key={did} className="coll-doc-item">
                        <span className="coll-doc-name">{d?.filename ?? `文档#${did}`}</span>
                        <button
                          className="icon-btn"
                          aria-label="移出文献集"
                          onClick={async () => {
                            try {
                              await api.removeCollectionDoc(manageColl.id, did);
                              setManageColl({ ...manageColl, doc_ids: manageColl.doc_ids.filter((x) => x !== did) });
                              setCollections((prev) => prev.map((c) => (c.id === manageColl.id ? { ...c, doc_ids: c.doc_ids.filter((x) => x !== did) } : c)));
                              setCollError(null);
                            } catch (e: unknown) {
                              setCollError(toErrorMessage(e) || "移出文献集失败");
                            }
                          }}
                        ><Icons.x size={14} /></button>
                      </li>
                    );
                  })}
                </ul>
              )}
              <div className="muted" style={{ marginTop: 8 }}>添加到本集合：</div>
              <div className="coll-add-list">
                {docs.filter((d) => !manageColl.doc_ids.includes(d.id)).map((d) => (
                  <button
                    key={d.id}
                    className="coll-add-chip"
                    onClick={async () => {
                      try {
                        await api.addCollectionDoc(manageColl.id, d.id);
                        setManageColl({ ...manageColl, doc_ids: [...manageColl.doc_ids, d.id] });
                        setCollections((prev) => prev.map((c) => (c.id === manageColl.id ? { ...c, doc_ids: [...c.doc_ids, d.id] } : c)));
                        setCollError(null);
                      } catch (e: unknown) {
                        setCollError(toErrorMessage(e) || "添加失败");
                      }
                    }}
                  >{d.filename}</button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* F5：批量「归入文献集」选择弹窗 */}
      {/* B1：创建工作间命名弹窗（「勾选文档」与「文献集」两个入口共用同一套交互） */}
      {wsNameOpen && (
        <div
          className="modal-mask"
          onClick={() => !wsCreating && setWsNameOpen(null)}
          aria-modal="true"
          role="dialog"
        >
          <div className="modal coll-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="reader-title">
                创建工作间
                {wsNameOpen.kind === "docs"
                  ? `（${wsNameOpen.docIds.length} 篇）`
                  : `（${collections.find((c) => c.id === wsNameOpen.cid)?.doc_ids.length ?? 0} 篇）`}
              </div>
              <button
                className="icon-btn"
                onClick={() => setWsNameOpen(null)}
                aria-label="关闭"
                disabled={wsCreating}
              >
                <Icons.x size={14} />
              </button>
            </div>
            <div className="coll-manage-body">
              <input
                className="collections-new-input"
                placeholder="给工作间起个名字（留空则用默认名）"
                value={wsName}
                onChange={(e) => setWsName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !wsCreating) void submitWorkspace();
                  if (e.key === "Escape") setWsNameOpen(null);
                }}
                autoFocus
              />
              {wsError && (
                <div className="settings-msg err" style={{ marginTop: 8 }}>
                  {wsError}
                </div>
              )}
              <div className="coll-manage-row" style={{ marginTop: 10 }}>
                <button
                  className="batch-btn ghost"
                  onClick={() => setWsNameOpen(null)}
                  disabled={wsCreating}
                >
                  取消
                </button>
                <button
                  className="batch-btn primary"
                  onClick={() => void submitWorkspace()}
                  disabled={wsCreating}
                >
                  {wsCreating ? "创建中…" : "创建工作间"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {addToCollOpen && (
        <div className="modal-mask" onClick={() => setAddToCollOpen(false)} aria-modal="true" role="dialog">
          <div className="modal coll-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="reader-title">归入文献集（{selected.size} 篇）</div>
              <button className="icon-btn" onClick={() => setAddToCollOpen(false)} aria-label="关闭"><Icons.x size={14} /></button>
            </div>
            <div className="coll-manage-body">
              <div className="coll-add-list">
                {collections.length === 0 && <div className="muted">尚无文献集，可在下方新建。</div>}
                {collections.map((c) => (
                  <button
                    key={c.id}
                    className="coll-add-chip"
                    onClick={async () => {
                      const ids = Array.from(selected);
                      // 并发提交（原先串行 await，N 篇即 N 次往返）；单篇失败不阻断其余
                      const results = await Promise.all(
                        ids.map((did) => api.addCollectionDoc(c.id, did).then(() => did, () => null)),
                      );
                      const okIds = results.filter((x): x is number => x != null);
                      const failed = ids.length - okIds.length;
                      if (okIds.length > 0) {
                        setCollections((prev) =>
                          prev.map((x) =>
                            x.id === c.id ? { ...x, doc_ids: Array.from(new Set([...x.doc_ids, ...okIds])) } : x,
                          ),
                        );
                      }
                      if (failed > 0) {
                        // 失败时不关闭弹窗，让用户看到失败数量（文档可能已被删除）
                        setCollError(`${failed} 篇归入失败（文档可能已被删除）`);
                      } else {
                        setCollError(null);
                        setAddToCollOpen(false);
                      }
                    }}
                  >{c.name}（{c.doc_ids.length}）</button>
                ))}
              </div>
              <div className="coll-manage-row" style={{ marginTop: 10 }}>
                <input
                  className="collections-new-input"
                  placeholder="或新建文献集并归入"
                  value={newCollName}
                  onChange={(e) => setNewCollName(e.target.value)}
                />
                <button
                  className="batch-btn primary"
                  disabled={!newCollName.trim()}
                  onClick={async () => {
                    const c = await createColl(newCollName.trim(), Array.from(selected));
                    // 仅成功时清空输入并关闭（失败时保留输入 + 提示）
                    if (c) {
                      setNewCollName("");
                      setAddToCollOpen(false);
                    }
                  }}
                >新建并归入</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
