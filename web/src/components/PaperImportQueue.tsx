import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { PaperImportTaskDto } from "../api/client";
import {
  activeTaskCount,
  downloadPct,
  mergeTasks,
  taskStatusText,
} from "../lib/paperSearch";
import { timeAgo } from "../lib/time";

/**
 * 导入任务的轮询与操作（放在 hook 里，文献页只管渲染）。
 *
 * 为什么任务状态在**前端内存**里而不是后端任务表：任务本身是瞬时的（分钟级），
 * 落地的是文档（DB 有 status）。页面刷新后未完成的任务会丢 —— 这是已知且可接受的边界
 * （后端任务表同样不持久化，两边口径一致）。
 */
export function usePaperImportTasks() {
  const [tasks, setTasks] = useState<PaperImportTaskDto[]>([]);
  const [docProg, setDocProg] = useState<Record<string, { pct?: number; stage?: string }>>({});
  const timer = useRef<number | null>(null);

  const add = useCallback((incoming: PaperImportTaskDto[]) => {
    if (!incoming || incoming.length === 0) return;
    setTasks((prev) => mergeTasks(prev, incoming));
  }, []);

  // ★ 挂载即恢复：切到别的页面（组件卸载）再回来、或刷新页面后，
  //   从后端把**仍在跑**的任务拉回来 —— 否则用户会以为"进度丢了"（其实是后端还在跑）。
  useEffect(() => {
    void (async () => {
      try {
        const res = await api.listPaperImportTasks();
        add(res.tasks ?? []);
      } catch {
        /* 列表拉不到不影响正常使用（比如功能未开启） */
      }
    })();
  }, [add]);

  const cancel = useCallback(async (id: string) => {
    try {
      await api.cancelPaperImportTask(id);
    } catch {
      /* 取消失败不阻塞界面：下一轮轮询会给出真实状态 */
    }
  }, []);

  const retry = useCallback(async (t: PaperImportTaskDto) => {
    if (!t.pdf_url) return;
    try {
      const res = await api.importPapers([
        { pdf_url: t.pdf_url, source: t.source || "arxiv", source_id: t.source_id, version: t.version, title: t.title },
      ]);
      add(res.tasks ?? []);
    } catch {
      /* 重试失败同样交给下一轮状态 */
    }
  }, [add]);

  const dismiss = useCallback((id: string) => {
    setTasks((prev) => prev.filter((t) => t.id !== id));
  }, []);

  useEffect(() => {
    const running = tasks.filter((t) => t.status === "queued" || t.status === "downloading" || t.status === "ingesting");
    if (running.length === 0) {
      if (timer.current != null) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
      return;
    }
    if (timer.current != null) return;

    timer.current = window.setInterval(() => {
      void (async () => {
        for (const t of running) {
          try {
            const next = await api.getPaperImportTask(t.id);
            setTasks((prev) => prev.map((p) => (p.id === next.id ? next : p)));
            // ingesting 阶段补文档进度（解析阶段由文档服务提供，不在这里另造）
            if (next.status === "ingesting" && next.doc_id) {
              const prog = await api.getUploadProgress(next.doc_id);
              if (prog) {
                setDocProg((prev) => ({ ...prev, [next.id]: { pct: prog.pct, stage: prog.stage } }));
              }
            }
          } catch {
            /* 单条失败不影响其它任务 */
          }
        }
      })();
    }, 1500);

    return () => {
      if (timer.current != null) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
    };
  }, [tasks]);

  return { tasks, docProg, add, cancel, retry, dismiss };
}

/**
 * 文献页顶部的导入队列细条。
 *
 * 设计要点：
 * - **默认收起**：只显示"N 个任务进行中"，不抢文献列表的地方；
 * - ★ **失败必须带可读原因**（这是评审硬要求，不能只显示红点）；
 * - 下载中给**百分比 + 字节数**（5.8MB 实测要 228 秒，不给进度用户会以为卡死）。
 */
export function PaperImportQueue({
  tasks,
  docProg,
  onCancel,
  onRetry,
  onOpenDoc,
  onDismiss,
}: {
  tasks: PaperImportTaskDto[];
  docProg: Record<string, { pct?: number; stage?: string }>;
  onCancel: (id: string) => void;
  onRetry: (t: PaperImportTaskDto) => void;
  onOpenDoc: (docId: number) => void;
  onDismiss: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  if (tasks.length === 0) return null;

  const running = activeTaskCount(tasks);
  const failed = tasks.filter((t) => t.status === "failed").length;
  const summary = [
    running > 0 ? `${running} 个导入进行中` : "",
    failed > 0 ? `${failed} 个失败` : "",
    running === 0 && failed === 0 ? "导入完成" : "",
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="paper-queue">
      <button className="paper-queue-bar" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="paper-queue-summary">{summary}</span>
        <span className="paper-queue-toggle">{open ? "收起" : "查看"}</span>
      </button>

      {open && (
        <div className="paper-queue-list">
          {tasks.map((t) => {
            const pct = downloadPct(t);
            const text = taskStatusText(t, docProg[t.id]);
            const done = t.status === "ready" || t.status === "cancelled";
            return (
              <div key={t.id} className="paper-queue-row">
                <div className="paper-queue-main">
                  <div className="paper-queue-title">{t.title || t.source_id || t.id}</div>
                  <div className={`paper-queue-status${t.status === "failed" ? " bad" : ""}${t.status === "ready" ? " ok" : ""}`}>
                    {text}
                    {t.created_at_iso && (
                      <span className="paper-queue-at" title={new Date(t.created_at_iso).toLocaleString()}>
                        {` · ${timeAgo(t.created_at_iso)}发起`}
                      </span>
                    )}
                  </div>
                  {pct != null && (
                    <div className="paper-bar" aria-hidden="true">
                      <div className="paper-bar-in" style={{ width: `${pct}%` }} />
                    </div>
                  )}
                </div>
                <div className="paper-queue-actions">
                  {t.status === "failed" && (
                    <>
                      <button className="ghost-btn" onClick={() => onRetry(t)}>重试</button>
                      <button className="ghost-btn" onClick={() => onDismiss(t.id)}>移除</button>
                    </>
                  )}
                  {t.status === "ready" && t.doc_id != null && (
                    <button className="ghost-btn" onClick={() => onOpenDoc(t.doc_id as number)}>去问答</button>
                  )}
                  {!done && t.status !== "failed" && (
                    <button className="ghost-btn" onClick={() => onCancel(t.id)}>取消</button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
