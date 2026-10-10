import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { DocumentOut } from "../lib/types";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
// pdf.js worker：用 Vite ?url 引入产物，避免构建期动态 import worker 失败
import workerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";
import { Icons } from "../lib/icons";
import { saveBlob as saveFileBlob, saveOutcomeMessage } from "../lib/saveFile";
import { DocxRender } from "./DocxRender";
import { useCtrlWheelZoom } from "../lib/useCtrlWheelZoom";
import { toErrorMessage } from "../lib/error";

GlobalWorkerOptions.workerSrc = workerUrl;

/**
 * 文档预览面板（画布 24:1）· 问答页右栏
 * - 多文档 Tab 条（格式徽章 + 文件名 + ×关闭 + 末尾 + 添加）
 * - 格式工具条（当前格式 / 页码 / 缩放 / 翻页）
 * - PDF **页面渲染改走后端 MuPDF 栅格图**（GET /documents/{id}/pages/{page}/image）：
 *   规避 pdf.js 对 CID 中文字体（Identity 编码 + ToUnicode 缺失）渲染成方块的根因；
 *   pdf.js 仅保留「取总页数」用途（纯元数据，不渲染字形，CJK 安全）。
 * - 导出按钮组（PDF / Markdown + 引用）
 * 依据 docs/前端文档预览功能设计方案.md v1.1：右栏只承载「预览 + 导出」。
 */

type Fmt = "PDF" | "DOCX" | "MD";

import { docLabel } from "../lib/docLabel";

// docLabel 已抽出至 lib/docLabel.ts（2026-09-09，N1 配套）：此处再导出以兼容
// ChatPage 等既有导入路径；新代码请直接从 lib/docLabel 导入。
export { docLabel } from "../lib/docLabel";

function fmtOf(name: string): Fmt {
  const ext = name.toLowerCase().split(".").pop() ?? "";
  if (ext === "pdf") return "PDF";
  if (ext === "md" || ext === "markdown" || ext === "txt") return "MD";
  return "DOCX";
}

/** 目标渲染宽度（与画布纸张一致）与 2x 像素密度 */
const BASE_WIDTH = 356;

/**
 * PDF 单页预览（服务端栅格化，规避 CJK 方块根因）。
 * - **总页数**仍用 pdf.js `getDocument().numPages` 取（纯元数据解析，不渲染字形，对 CID 中文字体安全）。
 * - **实际页面渲染**改用后端 MuPDF 栅格图（GET /documents/{id}/pages/{page}/image）：
 *   pdf.js 用 canvas 渲染 CID 中文字体（Identity 编码 + ToUnicode 缺失）会回退成方块；
 *   服务端用内嵌字体字形直出 PNG，中文永远清晰。zoom 通过请求 dpi + 显示宽度实现。
 */
function PdfPage({
  docId,
  page,
  zoom,
  onPageCount,
}: {
  docId: number;
  page: number;
  zoom: number;
  onPageCount?: (n: number) => void;
}) {
  const taskRef = useRef<ReturnType<typeof getDocument> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [imgLoading, setImgLoading] = useState(true);

  // 仅用 pdf.js 取总页数（docId 变化才重载）；不渲染任何字形，故 CID 中文不影响
  useEffect(() => {
    let cancelled = false;
    setError(null);
    taskRef.current?.destroy().catch(() => {});
    taskRef.current = null;

    const load = async () => {
      try {
        const res = await fetch(api.documentFileUrl(docId));
        if (!res.ok) throw new Error(`加载失败（HTTP ${res.status}）`);
        const buf = await res.arrayBuffer();
        if (cancelled) return;
        const task = getDocument({ data: buf });
        taskRef.current = task;
        const pdf = await task.promise;
        if (cancelled) {
          task.destroy();
          return;
        }
        onPageCount?.(pdf.numPages);
      } catch (e: unknown) {
        if (!cancelled) setError(toErrorMessage(e) || "PDF 解析失败");
      }
    };
    load();
    return () => {
      cancelled = true;
      taskRef.current?.destroy().catch(() => {});
      taskRef.current = null;
    };
  }, [docId]); // eslint-disable-line react-hooks/exhaustive-deps

  // zoom → 请求分辨率（越高越清晰）与显示宽度（越大越放大）；上限 300 防超大图
  const dpi = Math.min(300, Math.max(20, Math.round(150 * (zoom / 100))));
  const displayWidth = BASE_WIDTH * (zoom / 100);
  const src = api.documentPageImageUrl(docId, page, { dpi });

  // 切换页码/缩放后重置加载态
  useEffect(() => {
    setImgLoading(true);
  }, [src]);

  if (error) {
    return <div className="preview-error">{error}</div>;
  }
  return (
    <div className="preview-paper" style={{ position: "relative" }}>
      <img
        src={src}
        alt={`文档 ${docId} 第 ${page} 页`}
        style={{ width: `${displayWidth}px`, maxWidth: "100%", display: "block", borderRadius: 4 }}
        onLoad={() => setImgLoading(false)}
        onError={() => {
          setImgLoading(false);
          setError("页面图像渲染失败（后端未启动或 pymupdf 缺失，请确认后端已运行）");
        }}
      />
      {imgLoading && <div className="preview-loading">加载中…</div>}
    </div>
  );
}

/** Markdown / Word / 文本渲染：fetch 解析文本（后端 /text 接口）→ 预格式展示。
 *  Word（.docx/.doc）等二进制格式直接 fetch 原始文件会乱码，统一走解析文本预览，
 *  与全文检索共用同一解析产物（预览内容 == 检索内容）。 */
function TextDocView({ docId }: { docId: number }) {
  const [text, setText] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch(api.documentTextUrl(docId))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => !cancelled && setText((j?.text as string) ?? ""))
      .catch((e: unknown) => !cancelled && setError(toErrorMessage(e) || "加载失败"));
    return () => {
      cancelled = true;
    };
  }, [docId]);
  if (error) return <div className="preview-error">{error}</div>;
  return (
    <div className="preview-paper md">
      <pre className="preview-md-text">{text || "加载中…"}</pre>
    </div>
  );
}

export function DocumentPreviewPanel({
  docs,
  activeDocId,
  onSwitch,
  onCloseDoc,
  onAddDoc,
  isWorkspace = false,
  draftDocId = null,
  onSetDraft,
  draftBusy = false,
}: {
  /** 会话已关联的文档（作为对话上下文） */
  docs: DocumentOut[];
  /** 当前预览中的文档 id */
  activeDocId: number | null;
  onSwitch: (id: number) => void;
  onCloseDoc: (id: number) => void;
  /** 从已上传文档中再次选择并添加（修复原「+」按钮无效） */
  onAddDoc: (id: number) => void;
  /** 改稿落盘后通知会话侧刷新文档元数据（2026-09-06 接线，此前 ReviewWorkbench 调用但未传） */
  onMutated?: () => void;
  /** B2：当前会话是否为工作间 —— 决定是否渲染主文献星标 */
  isWorkspace?: boolean;
  /** B2：当前主文献 id（null = 未指定） */
  draftDocId?: number | null;
  /** B2：设置/清除主文献（传 null 清除） */
  onSetDraft?: (id: number | null) => Promise<void> | void;
  /** B2 审查加固：主文献 PATCH 进行中 → 星标禁用，防并发竞态 */
  draftBusy?: boolean;
}) {
  // 缩放：工具条按钮 + Ctrl+滚轮（P1 增补；hook 用原生非 passive 监听阻止浏览器页面缩放）
  const { ref: wheelRef, zoom, setZoom } = useCtrlWheelZoom<HTMLDivElement>();
  /** 页码状态：docId → 当前页；各文档独立记忆 */
  const [pages, setPages] = useState<Record<number, number>>({});
  /** 总页数：docId → numPages（PdfPage 加载后回调） */
  const [totalPages, setTotalPages] = useState<Record<number, number>>({});
  /** 收起/展开：默认收起（只留右侧把手）；出现新的对话/文档任务时自动展开 */
  const [collapsed, setCollapsed] = useState(true);
  /** 全屏放大预览 */
  const [expanded, setExpanded] = useState(false);
  /** 收纳（2026-09-07）：折叠非活跃文档，只保留当前活跃卡 + 「N 篇未预览」下拉 */
  const [compact, setCompact] = useState(false);
  /** 收纳态下其他文档下拉的开关 */
  const [popoverOpen, setPopoverOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  // 点外部/Escape 关闭下拉
  useEffect(() => {
    if (!popoverOpen) return;
    const onDown = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) setPopoverOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPopoverOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [popoverOpen]);
  /** 添加文档选择器（从已上传文档选择） */
  const [pickerOpen, setPickerOpen] = useState(false);
  const [allDocs, setAllDocs] = useState<DocumentOut[]>([]);
  /** 上一次 docs 数量（用于检测「文档新增 → 自动展开」） */
  const prevDocsLen = useRef<number>(0);

  // 文档任务自动展开：新对话关联文档 / 新增文档时（docs 数量增加）自动展开预览；
  // 用户手动收起后仅切换预览 Tab（数量不变）不会被打断。
  useEffect(() => {
    const len = docs.length;
    if (len > 0 && len > prevDocsLen.current) {
      setCollapsed(false);
    }
    prevDocsLen.current = len;
  }, [docs.length]);

  // 打开选择器时拉取已上传文档列表（排除已在预览中的）
  // L8 修复（2026-09-10）：加请求序号——快速开关两次时旧响应可能后到并覆盖新列表
  // （与 J2/J8 同款的序号守卫模式）。
  const pickerSeqRef = useRef(0);
  useEffect(() => {
    if (!pickerOpen) return;
    const seq = ++pickerSeqRef.current;
    api
      .listDocuments()
      .then((list) => {
        if (seq === pickerSeqRef.current) setAllDocs(list);
      })
      .catch(() => {
        if (seq === pickerSeqRef.current) setAllDocs([]);
      });
  }, [pickerOpen]);

  // 全屏阅读：Esc 退出（与关闭按钮等价，避免全屏下找不到出口）
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const active = useMemo(
    () => docs.find((d) => d.id === activeDocId) ?? docs[0] ?? null,
    [docs, activeDocId],
  );

  const activePage = active ? (pages[active.id] ?? 1) : 1;
  const activeTotal = active ? (totalPages[active.id] ?? 0) : 0;
  const activeIndex = active ? docs.findIndex((d) => d.id === active.id) : -1;
  const activeLabel = activeIndex >= 0 ? docLabel(activeIndex) : "";
  const fmt = active ? fmtOf(active.filename) : "PDF";

  const setPage = useCallback(
    (delta: number) => {
      if (!active) return;
      const next = Math.min(Math.max(1, activePage + delta), Math.max(1, activeTotal));
      setPages((prev) => ({ ...prev, [active.id]: next }));
    },
    [active, activePage, activeTotal],
  );

  // ---- 导出（2026-09-01 接线：此前按钮无 onClick，纯摆设）----
  // 2026-09-01 二次调整：去掉「导出 PDF」——本地优先应用里原文件本就在用户磁盘上，
  // 导回原文件近乎零价值；只保留有真实格式转换价值的「Markdown + 引用」。
  /** 正在导出（禁用按钮，防止并发重复下载） */
  const [exporting, setExporting] = useState(false);
  /** 导出结果提示（成功 / 失败，4 秒自动消退） */
  const [exportMsg, setExportMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (!exportMsg) return;
    const t = setTimeout(() => setExportMsg(null), 4000);
    return () => clearTimeout(t);
  }, [exportMsg]);

  /** ★ P-05（2026-10-09）：统一走 lib/saveFile —— 桌面端弹原生「另存为」+ 后端写盘 + 明确路径；
   *  浏览器端内部回退 <a download>（行为不变）。取消不打扰、失败必须可见。 */
  const saveBlob = (blob: Blob, filename: string) => {
    void (async () => {
      try {
        const out = await saveFileBlob(filename, blob);
        if (out.kind === "cancelled") return;
        const msg = saveOutcomeMessage(out);
        if (msg) setExportMsg({ ok: true, text: msg });
      } catch (e: unknown) {
        setExportMsg({ ok: false, text: `导出失败：${toErrorMessage(e) || "未知错误"}` });
      }
    })();
  };

  /** 导出 Markdown + 引用 = 后端格式转换（GET /export?format=md）+ 头部引用块（来源/ID/时间） */
  const exportMarkdown = async () => {
    if (!active || exporting) return;
    setExporting(true);
    setExportMsg(null);
    try {
      const res = await fetch(api.documentExportUrl(active.id, "md"));
      if (!res.ok) {
        // 后端 404 会带中文 detail（如「先 POST /documents/{id}/process」），透传给用户
        let detail = `HTTP ${res.status}`;
        try {
          const j = await res.json();
          if (j?.detail) detail = String(j.detail);
        } catch {
          /* 非 JSON 响应保持 HTTP 码提示 */
        }
        throw new Error(detail);
      }
      const md = await res.text();
      const base = active.filename.replace(/\.[^.]+$/, "");
      const citation = [
        "---",
        `来源文档: ${active.filename}`,
        `文档 ID: ${active.id}`,
        `导出时间: ${new Date().toLocaleString()}`,
        "工具: PaperMate",
        "---",
        "",
      ].join("\n");
      saveBlob(new Blob([citation + md], { type: "text/markdown;charset=utf-8" }), `${base}.md`);
    } catch (e: unknown) {
      setExportMsg({ ok: false, text: `导出失败：${toErrorMessage(e) || "未知错误"}` });
    } finally {
      setExporting(false);
    }
  };

  // 鼠标滚轮（2026-09-07 修复）：
  //  • 缩放态（zoom !== 100）→ 不拦截，让原生滚动浏览放大内容；
  //  • 未缩放（zoom === 100）→ 切换文档（快速翻阅多篇），preventDefault 阻止浏览器叠加滚动；
  //  • Ctrl+滚轮始终留给缩放（useCtrlWheelZoom 单独监听，preventDefault:false 时原生事件亦可达）。
  //
  // L4 修复（2026-09-10）：改用**原生非 passive 监听**（见下方 effect）——React 17+ 对
  // wheel/touchstart/touchmove 一律注册 passive 监听，`onWheel` 里的 `preventDefault()`
  // 不生效（控制台告警，且未缩放时滚轮切文档与页面滚动会同时发生）。
  const onWheel = useCallback(
    (e: WheelEvent) => {
      if (e.ctrlKey) return;
      if (Math.abs(e.deltaY) < 5) return;
      // 缩放态：原生滚动处理 wheel
      if (zoom !== 100) return;
      // 未缩放：切换文档
      if (activeIndex < 0 || docs.length <= 1) return;
      const next = activeIndex + (e.deltaY > 0 ? 1 : -1);
      if (next < 0 || next >= docs.length) return;
      e.preventDefault();
      e.stopPropagation();
      onSwitch(docs[next].id);
    },
    [zoom, activeIndex, docs, onSwitch],
  );

  // L4：原生 wheel 监听（passive: false）——只有原生监听里的 preventDefault 才真正生效。
  // 注意 `wheelRef` 来自 useCtrlWheelZoom，是**回调 ref**（非 RefObject），故需自持节点引用。
  const wheelElRef = useRef<HTMLDivElement | null>(null);
  const setWheelNode = useCallback(
    (node: HTMLDivElement | null) => {
      wheelElRef.current = node;
      wheelRef(node); // 同时交给 useCtrlWheelZoom（Ctrl+滚轮缩放）
    },
    [wheelRef],
  );
  useEffect(() => {
    const el = wheelElRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const addableDocs = allDocs.filter((d) => !docs.some((x) => x.id === d.id));

  const picker = pickerOpen ? (
    <div className="modal-mask" onClick={() => setPickerOpen(false)}>
      <div className="modal doc-picker" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="reader-title">添加文档到预览</div>
            <div className="reader-sub">从已上传文档中选择（{addableDocs.length} 篇可选）</div>
          </div>
          <button className="icon-btn" onClick={() => setPickerOpen(false)} aria-label="关闭"><Icons.x size={14} /></button>
        </div>
        <div className="doc-picker-list">
          {addableDocs.length === 0 && <div className="muted" style={{ padding: 12 }}>没有可添加的文档（均已在此或暂无上传）</div>}
          {addableDocs.map((d) => (
            <button
              key={d.id}
              className="doc-picker-item"
              onClick={() => {
                onAddDoc(d.id);
                setPickerOpen(false);
              }}
            >
              <span className={`fmt-badge ${fmtOf(d.filename).toLowerCase()}`}>{fmtOf(d.filename)}</span>
              <span className="doc-picker-name">{d.filename}</span>
              <span className="muted">（{d.num_chunks ?? 0} 块）</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  ) : null;

  // 收起态：整个预览区隐藏至右侧（宽度为 0，聊天区占满），仅留一个展开把手
  if (collapsed) {
    return (
      <aside className="doc-preview-collapsed" aria-label="预览已收起">
        <button
          type="button"
          className="doc-preview-handle"
          onClick={() => setCollapsed(false)}
          title="展开文档预览"
          aria-label="展开文档预览"
        >
          <Icons.chevronLeft size={14} />
          <span className="handle-text">文档</span>
        </button>
      </aside>
    );
  }

  if (!active) {
    return (
      <aside className="doc-preview-panel">
        <div className="panel-head">
          <span className="panel-title">文档预览</span>
          <button
            className="panel-collapse"
            onClick={() => setCollapsed((c) => !c)}
            title="收起预览面板"
          >
            <Icons.panelRightClose size={12} />
            收起
          </button>
        </div>
        {!collapsed && (
          <>
            <div className="preview-empty">
              <div className="preview-empty-icon"><Icons.fileText size={22} /></div>
              <p>未关联文档</p>
              <p className="muted">在左侧「已开文档」区勾选论文，或点「+」从已上传文档添加</p>
            </div>
            <div className="preview-empty-actions">
              <button className="export-btn primary" onClick={() => setPickerOpen(true)}>＋ 添加文档</button>
            </div>
          </>
        )}
        {picker}
      </aside>
    );
  }

  return (
    <aside className={`doc-preview-panel ${expanded ? "expanded" : ""}`}>
        <div className="panel-head">
          <span className="panel-title">
            文档预览
            {expanded && <span className="panel-title-sub">全屏阅读 · Esc 退出</span>}
          </span>
          <span className="panel-head-actions">
            <button
              className="panel-collapse"
              onClick={() => { setCompact((c) => !c); setPopoverOpen(false); }}
              title={compact ? "展开为完整网格" : "折叠其他文档，只保留当前"}
            >
              {compact ? "▦ 展开" : "≡ 收纳"}
            </button>
            <button
              className="panel-collapse"
              onClick={() => setExpanded((e) => !e)}
              title={expanded ? "退出全屏阅读" : "全屏放大预览"}
            >
              {expanded ? <Icons.minimize size={13} /> : <Icons.maximize size={13} />}
              {expanded ? "还原" : "放大"}
            </button>
            <button
              className="panel-collapse"
              onClick={() => setCollapsed((c) => !c)}
              title="收起预览面板"
            >
              <Icons.panelRightClose size={13} />
              收起
            </button>
          </span>
        </div>
      {!collapsed && (
        <>
          {/* 多文档 Tab：默认 2 列网格；收纳态只保留当前活跃卡 + 「N 篇未预览」下拉 */}
          {(() => {
            const others = docs.filter((d) => d.id !== active?.id);
            const featuredIdx = active ? docs.findIndex((d) => d.id === active.id) : -1;
            if (compact) {
              return (
                <div className="preview-tabs focused">
                  {active && (
                    <button
                      key={active.id}
                      className="preview-tab featured active"
                      onClick={() => onSwitch(active.id)}
                      title={active.filename}
                    >
                      <span className={`fmt-badge ${fmtOf(active.filename).toLowerCase()}`}>{fmtOf(active.filename)}</span>
                      <span className="preview-tab-label">{docLabel(featuredIdx)}</span>
                      <span className="preview-tab-name">{active.filename}</span>
                      {isWorkspace && active.id === draftDocId && (
                        <span className="preview-tab-draft" title="主文献（草稿）">
                          <Icons.star size={11} />
                        </span>
                      )}
                      <span
                        className="preview-tab-close"
                        role="button"
                        aria-label={`关闭 ${docLabel(featuredIdx)}`}
                        title={isWorkspace ? "从工作间移除该文档" : "关闭预览"}
                        onClick={(e) => { e.stopPropagation(); onCloseDoc(active.id); }}
                      ><Icons.x size={14} /></span>
                    </button>
                  )}
                  {others.length > 0 && (
                    <div className="preview-tabs-popover-host" ref={popoverRef}>
                      <button
                        className="preview-tabs-toggle"
                        onClick={() => setPopoverOpen((o) => !o)}
                        aria-expanded={popoverOpen}
                        title={popoverOpen ? "收起其他文档" : "展开其他文档"}
                      >
                        {popoverOpen ? "▴" : "▾"} {others.length} 篇未预览
                      </button>
                      {popoverOpen && (
                        <div className="preview-tabs-popover" role="listbox">
                          {others.map((d) => {
                            const realIdx = docs.findIndex((x) => x.id === d.id);
                            return (
                              <button
                                key={d.id}
                                className="preview-tab-popover-row"
                                onClick={() => { onSwitch(d.id); setPopoverOpen(false); }}
                                title={d.filename}
                              >
                                <span className={`fmt-badge ${fmtOf(d.filename).toLowerCase()}`}>{fmtOf(d.filename)}</span>
                                <span className="preview-tab-label">{docLabel(realIdx)}</span>
                                <span className="preview-tab-name">{d.filename}</span>
                                {isWorkspace && d.id === draftDocId && (
                                  <span className="preview-tab-draft" title="主文献（草稿）">
                                    <Icons.star size={11} />
                                  </span>
                                )}
                                <span
                                  className="preview-tab-close"
                                  role="button"
                                  aria-label={`关闭 ${docLabel(realIdx)}`}
                                  title={isWorkspace ? "从工作间移除该文档" : "关闭预览"}
                                  onClick={(e) => { e.stopPropagation(); onCloseDoc(d.id); }}
                                ><Icons.x size={14} /></span>
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            }
            return (
              <div className="preview-tabs">
                {docs.map((d, i) => (
                  <button
                    key={d.id}
                    className={`preview-tab ${d.id === active.id ? "active" : ""}`}
                    onClick={() => onSwitch(d.id)}
                    title={d.filename}
                  >
                    <span className={`fmt-badge ${fmtOf(d.filename).toLowerCase()}`}>{fmtOf(d.filename)}</span>
                    <span className="preview-tab-label">{docLabel(i)}</span>
                    <span className="preview-tab-name">{d.filename}</span>
                    {isWorkspace && d.id === draftDocId && (
                      <span className="preview-tab-draft" title="主文献（草稿）">
                        <Icons.star size={11} />
                      </span>
                    )}
                    <span
                      className="preview-tab-close"
                      role="button"
                      aria-label={`关闭 ${docLabel(i)}`}
                      title={isWorkspace ? "从工作间移除该文档" : "关闭预览"}
                      onClick={(e) => {
                        e.stopPropagation();
                        onCloseDoc(d.id);
                      }}
                    ><Icons.x size={14} /></span>
                  </button>
                ))}
              </div>
            );
          })()}

          {/* 格式工具条 */}
          <div className="preview-toolbar">
            <span className={`fmt-badge ${fmt.toLowerCase()}`}>{fmt}</span>
            <span className="preview-pageno" title={active.filename}>
              <span className="preview-current-label">{activeLabel}</span>
              {/* 文件名文本包一层：ellipsis 只对单一 box 生效（长中文文件名不再逐字竖排换行） */}
              <span className="preview-pageno-text">
                {active.filename}
                {fmt === "PDF" && activeTotal > 0 ? ` · 第 ${activePage} / ${activeTotal} 页` : ""}
              </span>
            </span>
            {/* B2：工作间主文献（草稿）指定 —— 星标即切换，选中时点击=清除 */}
            {isWorkspace && active && (
              <button
                className={`ws-draft-btn ${draftDocId === active.id ? "on" : ""}`}
                disabled={draftBusy}
                title={
                  draftBusy
                    ? "正在保存…"
                    : draftDocId === active.id
                      ? "该文献是本工作间的主文献（草稿）；点击取消指定"
                      : "把该文献设为本工作间的主文献（草稿）"
                }
                onClick={() => onSetDraft?.(draftDocId === active.id ? null : active.id)}
              >
                <Icons.star size={13} />
                {draftDocId === active.id ? "主文献" : "设为主文献"}
              </button>
            )}
            <span className="toolbar-spacer" />
            <div className="zoom-ctrl">
              <button onClick={() => setZoom((z) => Math.max(50, z - 10))} aria-label="缩小">−</button>
              <span>{zoom}%</span>
              <button onClick={() => setZoom((z) => Math.min(300, z + 10))} aria-label="放大">+</button>
            </div>
            {/* 未缩放（默认 100%）→ ‹› 切换文档（所有格式）；已缩放 → ‹› 翻页（仅 PDF）；
                判据改为 zoom 状态（2026-09-07），与全屏阅读 expanded 解耦。*/}
            {zoom === 100 ? (
              <div className="page-nav" title="滚轮/‹› 切换文档">
                <button
                  aria-label="上一个文档"
                  disabled={activeIndex <= 0}
                  onClick={() => activeIndex > 0 && onSwitch(docs[activeIndex - 1].id)}
                >
                  ‹
                </button>
                <button
                  aria-label="下一个文档"
                  disabled={activeIndex < 0 || activeIndex >= docs.length - 1}
                  onClick={() => activeIndex < docs.length - 1 && onSwitch(docs[activeIndex + 1].id)}
                >
                  ›
                </button>
              </div>
            ) : (
              fmt === "PDF" && (
                <div className="page-nav" title="‹› 翻页 · 滚轮滚动浏览">
                  <button aria-label="上一页" disabled={activePage <= 1} onClick={() => setPage(-1)}>
                    ‹
                  </button>
                  <button
                    aria-label="下一页"
                    disabled={activeTotal > 0 && activePage >= activeTotal}
                    onClick={() => setPage(1)}
                  >
                    ›
                  </button>
                </div>
              )
            )}
          </div>

          {/* 内容渲染：PDF→服务端栅格图（MuPDF）/ 文本→预格式；滚轮翻页（L4：原生监听，见上方 effect） */}
          <div className="preview-paper-wrap" ref={setWheelNode}>
            {fmt === "PDF" ? (
              <PdfPage
                key={active.id}
                docId={active.id}
                page={activePage}
                zoom={zoom}
                onPageCount={(n) => setTotalPages((prev) => ({ ...prev, [active.id]: n }))}
              />
            ) : fmt === "DOCX" ? (
              /* P1 增补：docx 走 docx-preview 高保真渲染（章节/表格/图片），不再退化为纯文本流；
                 渲染完成后自适应宽度（窄面板默认缩小至恰好放下，只缩不放） */
              <DocxRender
                key={active.id}
                docId={active.id}
                zoom={zoom}
                onZoomFit={setZoom}
              />
            ) : (
              <TextDocView key={active.id} docId={active.id} />
            )}
          </div>
          {/* 提示：未缩放滚轮/‹› 切文档；已缩放滚轮滚动浏览 + ‹› 翻页；Ctrl+滚轮始终缩放 */}
          {fmt === "PDF" && (
            <div className="preview-wheel-hint">
              {zoom === 100
                ? "滚轮 / ‹ › 切换文档 · Ctrl+滚轮缩放"
                : "滚轮滚动浏览 · ‹ › 翻页 · Ctrl+滚轮缩放"}
            </div>
          )}

          {/* 底部操作：添加 / 导出（2026-09-07 改造）
              - 添加文档按钮从 Tab 栏移至此，与导出并列；
              - 移除文档只保留 Tab 栏的 ×（onCloseDoc 同源，已按会话 config.workspace 判定走工作间解绑）。 */}
          <div className="export-btn-group export-actions">
            {exportMsg && <div className={`export-msg ${exportMsg.ok ? "ok" : "err"}`}>{exportMsg.text}</div>}
            <button
              className="export-btn ghost"
              title="从已上传文档中选择并添加"
              onClick={() => setPickerOpen(true)}
            >
              ＋ 添加文档
            </button>
            <button
              className="export-btn primary"
              title="转换为 Markdown 并附引用头（来源文档 / ID / 导出时间）；需文档已解析"
              disabled={!active || exporting}
              onClick={exportMarkdown}
            >
              <Icons.download size={13} /> {exporting ? "导出中…" : "导出 Markdown"}
            </button>
          </div>
        </>
      )}
      {picker}
    </aside>
  );
}
