import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api/client";
import type { PaperHitDto, PaperImportTaskDto } from "../api/client";
import {
  SCOPES,
  TOP_CATEGORIES,
  buildImportPayload,
  chineseHint,
  type Scope,
} from "../lib/paperSearch";

/**
 * 论文在线检索抽屉（阶段 3）。
 *
 * 设计约束（与产品评审一致）：
 * 1. **搜索与导入分离**：检索只拿元数据（秒级），导入是后台任务（分钟级）——
 *    所以这里**只负责搜索与提交**，进度交给 `PaperImportQueue`（关掉本抽屉也照跑）。
 * 2. **中文提示只提示不拦截**：arXiv 以英文为主，但不该替用户决定。
 * 3. **范围 + 多词** 的语义陷阱由后端 `build_query` 处理，前端只传 scope/categories 原值。
 * 4. 所有"算得出来"的东西都在 `lib/paperSearch.ts` 里（项目无 testing-library，逻辑必须可单测）。
 */
export function SearchPaperDrawer({
  open,
  onClose,
  onSubmitted,
}: {
  open: boolean;
  onClose: () => void;
  /** 提交成功后把任务交给上层（由文献页统一轮询与展示） */
  onSubmitted: (tasks: PaperImportTaskDto[]) => void;
}) {
  const [q, setQ] = useState("");
  const [scope, setScope] = useState<Scope>("all");
  const [cats, setCats] = useState<Set<string>>(new Set());
  const [hits, setHits] = useState<PaperHitDto[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [searched, setSearched] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const hint = useMemo(() => chineseHint(q), [q]);
  const payload = useMemo(() => buildImportPayload(hits, selected), [hits, selected]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    const t = window.setTimeout(() => inputRef.current?.focus(), 30);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.clearTimeout(t);
    };
  }, [open, onClose]);

  const doSearch = useCallback(async () => {
    const text = q.trim();
    if (!text) {
      setErr("请输入检索关键词");
      return;
    }
    setLoading(true);
    setErr("");
    try {
      const res = await api.searchPapers({
        q: text,
        limit: 20,
        scope,
        categories: [...cats].join(","),
      });
      setHits(res.items ?? []);
      setSelected(new Set());
      setSearched(true);
    } catch (e) {
      // 403（功能未开启）/ 400（参数不合法）/ 502（上游失败）都要**如实显示**
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
      setHits([]);
      setSearched(true);
    } finally {
      setLoading(false);
    }
  }, [q, scope, cats]);

  const toggleCat = (c: string) => {
    setCats((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c);
      else next.add(c);
      return next;
    });
  };

  const toggleHit = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const submit = async () => {
    if (payload.items.length === 0) return;
    setSubmitting(true);
    setErr("");
    try {
      const res = await api.importPapers(payload.items);
      onSubmitted(res.tasks ?? []);
      setSelected(new Set());
      onClose();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  if (!open) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-card paper-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="检索论文"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="paper-drawer-head">
          <div className="paper-drawer-title">
            检索论文 <span className="paper-badge">arXiv</span>
          </div>
          <button className="ghost-btn" onClick={onClose} aria-label="关闭">
            ×
          </button>
        </div>

        <div className="paper-search-row">
          <select
            className="paper-scope"
            value={scope}
            onChange={(e) => setScope(e.target.value as Scope)}
            aria-label="检索范围"
          >
            {SCOPES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
          <input
            ref={inputRef}
            className="paper-input"
            placeholder="关键词，例如 retrieval augmented generation"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void doSearch();
            }}
          />
          <button className="btn-primary" onClick={() => void doSearch()} disabled={loading}>
            {loading ? "检索中…" : "检索"}
          </button>
        </div>

        <div className="paper-chips">
          <span className="paper-chips-label">分类</span>
          {TOP_CATEGORIES.map((c) => (
            <button
              key={c}
              className={`paper-chip${cats.has(c) ? " on" : ""}`}
              onClick={() => toggleCat(c)}
              aria-pressed={cats.has(c)}
            >
              {c}
            </button>
          ))}
        </div>

        <div className="paper-syntax">
          支持 <code>AND</code> / <code>OR</code> / <code>&quot;精确词组&quot;</code> / <code>cat:cs.CL</code>
          ，也可直接写 <code>ti:</code> <code>au:</code> 前缀
        </div>

        {hint && <div className="paper-warn">{hint}</div>}
        {err && <div className="paper-error">{err}</div>}

        <div className="paper-results">
          {loading && <div className="paper-empty">正在检索…</div>}
          {!loading && !searched && <div className="paper-empty">输入关键词开始检索（只取元数据，不会下载 PDF）</div>}
          {!loading && searched && hits.length === 0 && !err && (
            <div className="paper-empty">没有找到结果。可以试试更短的英文关键词，或去掉分类筛选。</div>
          )}
          {!loading &&
            hits.map((h) => {
              const on = selected.has(h.source_id);
              return (
                <label key={h.source_id} className={`paper-hit${on ? " on" : ""}`}>
                  <input type="checkbox" checked={on} onChange={() => toggleHit(h.source_id)} />
                  <div className="paper-hit-main">
                    <div className="paper-hit-title">{h.title}</div>
                    <div className="paper-hit-meta">
                      {(h.authors ?? []).slice(0, 3).join(", ")}
                      {h.authors && h.authors.length > 3 ? " 等" : ""}
                      {h.published ? ` · ${h.published}` : ""}
                      {h.primary_category ? ` · ${h.primary_category}` : ""}
                    </div>
                    {h.abstract && <div className="paper-hit-abs">{h.abstract}</div>}
                  </div>
                  <a
                    className="paper-hit-link"
                    href={h.abs_url || h.pdf_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    原文 ↗
                  </a>
                </label>
              );
            })}
        </div>

        <div className="paper-drawer-foot">
          <span className="paper-count">已选 {payload.items.length} 篇</span>
          <div className="paper-foot-actions">
            <button className="ghost-btn" onClick={onClose}>
              取消
            </button>
            <button
              className="btn-primary"
              disabled={payload.items.length === 0 || submitting}
              onClick={() => void submit()}
            >
              {submitting ? "提交中…" : "导入并解析"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
