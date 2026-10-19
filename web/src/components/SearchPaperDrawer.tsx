import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Icons } from "../lib/icons";
import { api } from "../api/client";
import type { PaperHitDto, PaperImportTaskDto } from "../api/client";
import {
  SCOPES,
  TOP_CATEGORIES,
  appendHits,
  buildImportPayload,
  canLoadMore,
  capHint,
  isFullyMapped,
  chineseHint,
  parseSearchHistory,
  pushSearchHistory,
  type Scope,
  type SearchHistoryEntry,
} from "../lib/paperSearch";
import { timeAgo } from "../lib/time";

/** 检索历史的 localStorage 键（只存"关键词 + 时间"，不含任何论文数据） */
const HISTORY_KEY = "papermate.paperSearch.history";

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
  // 中文术语被替换的明细（**必须明示给用户**，不能悄悄换词）
  const [translated, setTranslated] = useState<{ zh: string; en: string }[]>([]);
  // 最近检索（带时间戳）；存在 localStorage，**只存关键词与时间**
  const [history, setHistory] = useState<SearchHistoryEntry[]>([]);
  const [used, setUsed] = useState("");
  // 分页：总数 / 是否还有更多 / 是否正在加载更多
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  // 标题中译（source_id → 中文标题）；空串表示该条没译出来 ⇒ 只显示英文原文
  const [zhTitles, setZhTitles] = useState<Record<string, string>>({});
  const [translating, setTranslating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  // 自绘下拉（原生 select 的弹出层由系统渲染，CSS 改不动 ⇒ 灰块 + 方角）
  const [scopeOpen, setScopeOpen] = useState(false);
  const scopeRef = useRef<HTMLDivElement | null>(null);

  const hint = useMemo(() => chineseHint(q), [q]);
  const payload = useMemo(() => buildImportPayload(hits, selected), [hits, selected]);

  // 点外面关下拉（比"只能再点一次按钮"自然）
  useEffect(() => {
    if (!scopeOpen) return;
    const onDown = (e: MouseEvent) => {
      if (scopeRef.current && !scopeRef.current.contains(e.target as Node)) setScopeOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setScopeOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onEsc);
    };
  }, [scopeOpen]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    try {
      setHistory(parseSearchHistory(window.localStorage.getItem(HISTORY_KEY)));
    } catch {
      /* 隐私模式等拿不到 localStorage ⇒ 静默降级为"没有历史" */
    }
    const t = window.setTimeout(() => inputRef.current?.focus(), 30);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.clearTimeout(t);
    };
  }, [open, onClose]);

  const doSearch = useCallback(async (textArg?: string) => {
    // textArg：点历史 chips 时直接用那条关键词搜（不必等 setQ 生效）
    const text = (textArg ?? q).trim();
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
      setTranslated(res.translations ?? []);
      setUsed(res.query_used || text);
      setTotal(res.total ?? 0);
      setHasMore(Boolean(res.has_more));
      setSelected(new Set());
      setSearched(true);
      // 记历史：同词只保留一条（提到最前并刷新时间戳）
      setHistory((prev) => {
        const next = pushSearchHistory(prev, text, new Date().toISOString());
        try {
          window.localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
        } catch {
          /* 存不进去不影响这次检索 */
        }
        return next;
      });
    } catch (e) {
      // 403（功能未开启）/ 400（参数不合法）/ 502（上游失败）都要**如实显示**
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
      setHits([]);
      setTranslated([]);
      setSearched(true);
    } finally {
      setLoading(false);
    }
  }, [q, scope, cats]);

  const loadMore = useCallback(async () => {
    if (loadingMore || !hasMore) return;
    setLoadingMore(true);
    try {
      // ★ 用 used（翻译后的关键词）而不是输入框当前内容 —— 否则翻页会变成另一次检索
      const res = await api.searchPapers({
        q: used || q,
        limit: 20,
        start: hits.length,
        scope,
        categories: [...cats].join(","),
      });
      setHits((prev) => appendHits(prev, res.items ?? []));
      setHasMore(Boolean(res.has_more));
      if ((res.total ?? 0) > 0) setTotal(res.total);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingMore(false);
    }
  }, [loadingMore, hasMore, used, q, hits.length, scope, cats]);

  const translateTitles = useCallback(async () => {
    if (translating || hits.length === 0) return;
    setTranslating(true);
    setErr("");
    try {
      const res = await api.translatePaperTitles(
        hits.map((h) => ({ source_id: h.source_id, title: h.title })),
      );
      const map: Record<string, string> = {};
      for (const it of res.items ?? []) {
        if (it.zh) map[it.source_id] = it.zh;
      }
      setZhTitles((prev) => ({ ...prev, ...map }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setTranslating(false);
    }
  }, [translating, hits]);

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
          <div className="paper-scope-wrap" ref={scopeRef}>
            <button
              className={`paper-scope${scopeOpen ? " on" : ""}`}
              onClick={() => setScopeOpen((v) => !v)}
              aria-haspopup="listbox"
              aria-expanded={scopeOpen}
              aria-label="检索范围"
            >
              {SCOPES.find((x) => x.value === scope)?.label ?? "全部字段"}
              <span className="paper-scope-caret" aria-hidden="true">
                ▾
              </span>
            </button>
            {scopeOpen && (
              <ul className="paper-scope-menu" role="listbox" aria-label="检索范围">
                {SCOPES.map((x) => (
                  <li key={x.value} role="option" aria-selected={scope === x.value}>
                    <button
                      className={`paper-scope-item${scope === x.value ? " on" : ""}`}
                      onClick={() => {
                        setScope(x.value as Scope);
                        setScopeOpen(false);
                      }}
                    >
                      <span>{x.label}</span>
                      {scope === x.value && <span className="paper-scope-check">✓</span>}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
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
          {/* 语法提示改成悬浮：平时只占一个小 ⓘ，省一整行 */}
          <span className="paper-syntax-tip" tabIndex={0} aria-label="检索语法说明">
            <Icons.info size={13} />
            <span className="paper-syntax-pop">
              支持 AND / OR / &quot;精确词组&quot; / cat:cs.CL，也可直接写 ti: au: 前缀
            </span>
          </span>
        </div>

        {history.length > 0 && (
          <div className="paper-history">
            <span className="paper-history-label">最近检索</span>
            {history.map((h) => (
              <button
                key={h.q}
                className="paper-history-item"
                title={`${h.q} · ${new Date(h.at).toLocaleString()}`}
                onClick={() => {
                  setQ(h.q);
                  void doSearch(h.q);
                }}
              >
                <span className="paper-history-q">{h.q}</span>
                <span className="paper-history-at">{timeAgo(h.at)}</span>
              </button>
            ))}
            <button
              className="paper-history-clear"
              onClick={() => {
                setHistory([]);
                try {
                  window.localStorage.removeItem(HISTORY_KEY);
                } catch {
                  /* ignore */
                }
              }}
            >
              清除
            </button>
          </div>
        )}

        {translated.length > 0 && (
          <div className="paper-zh-map">
            <span className="paper-zh-map-label">已换成</span>
            {translated.map((x) => (
              <span key={x.zh} className="paper-zh-item">
                <span className="paper-zh-zh">{x.zh}</span>
                <span className="paper-zh-arrow" aria-hidden="true">
                  →
                </span>
                <span className="paper-zh-en">{x.en}</span>
              </span>
            ))}
            {/* 完全覆盖查询时不重复显示同一个串 */}
            {!isFullyMapped(translated, used) && (
              <span className="paper-zh-used">实际检索：{used}</span>
            )}
          </div>
        )}
        {hint && translated.length === 0 && <div className="paper-warn">{hint}</div>}
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
                    {zhTitles[h.source_id] && (
                      <div className="paper-hit-zh">{zhTitles[h.source_id]}</div>
                    )}
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

        {searched && hits.length > 0 && (
          <div className="paper-more">
            <span className="paper-more-count">
              {total > 0
                ? `共 ${total.toLocaleString()} 条 · 已显示 ${hits.length} 条`
                : `已显示 ${hits.length} 条`}
            </span>
            <button
              className="ghost-btn paper-translate-btn"
              onClick={() => void translateTitles()}
              disabled={translating || hits.length === 0}
              title="把英文标题译成中文（一次调用，仅当前列表）"
            >
              {translating ? "翻译中…" : "翻译标题"}
            </button>
            {canLoadMore(hits.length, hasMore) ? (
              <button className="ghost-btn" onClick={() => void loadMore()} disabled={loadingMore}>
                {loadingMore ? "加载中…" : "加载更多"}
              </button>
            ) : (
              capHint(hits.length, total) && (
                <span className="paper-more-cap">{capHint(hits.length, total)}</span>
              )
            )}
          </div>
        )}

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
