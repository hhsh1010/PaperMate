import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { MemoryItemOut } from "../lib/types";
import { SectionCard } from "./SectionCard";
import { Icons } from "../lib/icons";
import { toErrorMessage } from "../lib/error";

/** 超过该长度的记忆正文默认折叠，点击「展开」查看全文（收纳） */
const CLAMP_CHARS = 90;

/** 时间格式化：YYYY-MM-DD HH:mm（本地时区） */
function fmtTime(ts: number | undefined | null): string {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

type SortKey = "created" | "importance" | "accessed";

/**
 * ER-16：长期记忆管理面板。
 * 2026-08-29 C 档增强：搜索过滤 / 三种排序 / 手动调重要性 / 批量删除 /
 * 显示创建时间与最近访问时间 / 显示来源会话。
 */
export function MemoryPanel() {
  const [items, setItems] = useState<MemoryItemOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [draftImp, setDraftImp] = useState<number | null>(null);
  // C 档：搜索 / 排序 / 批量选择
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("created");
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [batchBusy, setBatchBusy] = useState(false);
  // 收纳：长条目默认折叠，expanded 保存已展开的 id
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  // 分层 scope：分组是否收起（默认全部展开）
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());
  // Phase B：跨会话记忆功能开关（features.memory）
  const [memoryEnabled, setMemoryEnabled] = useState(true);

  useEffect(() => {
    api
      .getSettings()
      .then((s) => setMemoryEnabled(s.features?.memory !== false))
      .catch(() => setMemoryEnabled(true));
  }, []);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await api.listMemories();
      setItems(r.items);
    } catch (e: unknown) {
      setError(toErrorMessage(e) || "加载记忆失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** 搜索 + 排序后的展示列表 */
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    let out = items;
    if (q) out = out.filter((m) => m.text.toLowerCase().includes(q));
    const sorted = [...out];
    if (sort === "importance") sorted.sort((a, b) => b.importance - a.importance);
    else if (sort === "accessed") sorted.sort((a, b) => (b.last_accessed ?? 0) - (a.last_accessed ?? 0));
    else sorted.sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0));
    return sorted;
  }, [items, query, sort]);

  /** 2026-08-29 分层 scope：按作用域分组（全局 / 来源会话），组内沿用排序结果 */
  const groups = useMemo(() => {
    const map = new Map<string, { key: string; title: string; items: MemoryItemOut[] }>();
    for (const m of visible) {
      const key = m.scope === "session" ? `session:${m.source_session_id ?? "null"}` : "global";
      if (!map.has(key)) {
        map.set(key, {
          key,
          title: m.scope === "session" ? `会话 #${m.source_session_id} 记忆` : "全局记忆（跨会话偏好 / 规则）",
          items: [],
        });
      }
      map.get(key)!.items.push(m);
    }
    const arr = [...map.values()];
    arr.sort((a, b) => {
      if (a.key === "global") return -1;
      if (b.key === "global") return 1;
      const na = a.key === "session:null" ? Infinity : Number(a.key.split(":")[1]);
      const nb = b.key === "session:null" ? Infinity : Number(b.key.split(":")[1]);
      return na - nb;
    });
    return arr;
  }, [visible]);

  const startEdit = (m: MemoryItemOut) => {
    setEditing(m.id);
    setDraft(m.text);
    setDraftImp(null); // 不预填 → 未改动时保留后端原值
  };

  const saveEdit = async (id: string) => {
    try {
      await api.updateMemory(id, draft, draftImp ?? undefined);
      setEditing(null);
      setDraftImp(null);
      await refresh();
    } catch (e: unknown) {
      setError(toErrorMessage(e) || "保存失败");
    }
  };

  const remove = async (id: string) => {
    setError(null);
    const ok = window.confirm("确认删除这条记忆？此操作不可撤销。");
    if (!ok) return;
    try {
      await api.deleteMemory(id);
      setChecked((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      await refresh();
    } catch (e: unknown) {
      setError(toErrorMessage(e) || "删除失败");
    }
  };

  const toggleCheck = (id: string) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const allChecked = visible.length > 0 && visible.every((m) => checked.has(m.id));

  const toggleAll = () => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (allChecked) visible.forEach((m) => next.delete(m.id));
      else visible.forEach((m) => next.add(m.id));
      return next;
    });
  };

  /** 收纳：单条记忆正文的展开 / 收起 */
  const toggleExpand = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  /** 分组收起 / 展开 */
  const toggleGroup = (key: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  /** 调整记忆作用域：全局 ↔ 仅来源会话（session 需有来源会话，否则后端回落 global） */
  const setScope = async (m: MemoryItemOut) => {
    setError(null);
    const next = m.scope === "global" ? "session" : "global";
    if (next === "session" && m.source_session_id == null) {
      setError("该记忆没有来源会话，无法限定为会话级（保持全局）");
      return;
    }
    try {
      await api.updateMemory(m.id, m.text, undefined, next);
      await refresh();
    } catch (e: unknown) {
      setError(toErrorMessage(e) || "更新作用域失败");
    }
  };

  const batchDelete = async () => {
    if (checked.size === 0) return;
    setError(null);
    const ok = window.confirm(`确认删除选中的 ${checked.size} 条记忆？此操作不可撤销。`);
    if (!ok) return;
    setBatchBusy(true);
    try {
      const r = await api.batchDeleteMemories(Array.from(checked));
      setChecked(new Set());
      await refresh();
      setError(r.missing.length
        ? `已删除 ${r.removed} 条${r.missing.length ? `，${r.missing.length} 条未找到` : ""}`
        : `已删除 ${r.removed} 条记忆`);
    } catch (e: unknown) {
      setError(toErrorMessage(e) || "批量删除失败");
    } finally {
      setBatchBusy(false);
    }
  };

  const clearAll = async () => {
    setError(null);
    const ok = window.confirm("⚠️ 此操作非常危险，可能导致不可逆的数据丢失！\n清空所有长期记忆（包括毒化条目）。确认继续？");
    if (!ok) return;
    try {
      const r = await api.clearMemories();
      setChecked(new Set());
      await refresh();
      setError(`已清空 ${r.removed} 条记忆`);
    } catch (e: unknown) {
      setError(toErrorMessage(e) || "清空失败");
    }
  };

  return (
    <SectionCard
      title="长期记忆管理"
      subtitle="跨会话记忆（data/memories.json，全局共享）：可搜索、排序、编辑、调重要性、批量删除或一键清空。"
      badge={`${items.length} 条 · 全局共享`}
      defaultOpen
    >
      {!memoryEnabled && (
        <div className="doc-warn">
          跨会话记忆功能已关闭（设置页开关）：对话中不再自动记忆/召回/反思写入，此处仍可手动管理历史记忆。
        </div>
      )}

      {/* 工具栏：搜索 + 排序 + 批量操作 */}
      <div className="memory-toolbar">
        <input
          className="settings-control memory-search"
          placeholder="搜索记忆内容…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="搜索记忆"
        />
        <select
          className="settings-control memory-sort"
          value={sort}
          onChange={(e) => setSort(e.target.value as SortKey)}
          aria-label="排序方式"
        >
          <option value="created">按创建时间</option>
          <option value="importance">按重要性</option>
          <option value="accessed">按最近访问</option>
        </select>
        <button className="settings-btn-ghost" onClick={refresh} disabled={loading}>
          {loading ? "加载中…" : "刷新"}
        </button>
        <span className="memory-count">
          {query ? `${visible.length}/${items.length}` : items.length} 条
        </span>
      </div>

      {visible.length > 0 && (
        <div className="memory-toolbar memory-batchbar">
          <label className="memory-check-all">
            <input type="checkbox" checked={allChecked} onChange={toggleAll} aria-label="全选当前列表" />
            全选
          </label>
          <button
            className="settings-btn-danger"
            onClick={batchDelete}
            disabled={checked.size === 0 || batchBusy}
          >
            {batchBusy ? "删除中…" : `删除选中（${checked.size}）`}
          </button>
          <span className="batch-spacer" />
          <button className="settings-btn-danger" onClick={clearAll} disabled={items.length === 0}>
            清空全部
          </button>
        </div>
      )}

      {error && <div className={`settings-msg ${error.includes("已清空") || error.includes("已删除") ? "ok" : "err"}`}>{error}</div>}

      <div className="memory-list">
        {groups.map((g) => (
          <section className="memory-group" key={g.key}>
            <button
              type="button"
              className="memory-group-head"
              onClick={() => toggleGroup(g.key)}
              aria-expanded={!collapsedGroups.has(g.key)}
            >
              <span className="mg-title">{g.title}</span>
              <span className="mg-count">{g.items.length}</span>
              <span className="sc-chevron" aria-hidden="true"><Icons.chevronDown size={14} /></span>
            </button>
            {!collapsedGroups.has(g.key) && (
              <div className="memory-group-items">
                {g.items.map((m) => {
                  const isLong = m.text.length > CLAMP_CHARS;
                  const isOpen = expanded.has(m.id);
                  return (
                  <div className="memory-item" key={m.id}>
                    <div className="memory-meta">
                      <input
                        type="checkbox"
                        className="memory-check"
                        checked={checked.has(m.id)}
                        onChange={() => toggleCheck(m.id)}
                        aria-label={`选择记忆 ${m.id}`}
                      />
                      <button
                        type="button"
                        className={`memory-scope scope-${m.scope ?? "global"}`}
                        onClick={() => setScope(m)}
                        title={m.scope === "global" ? "全局：跨会话注入。点击改为仅限来源会话" : "仅来源会话注入。点击改回全局"}
                      >
                        {m.scope === "session" ? "会话" : "全局"}
                      </button>
                      <span className={`memory-imp-${m.importance >= 0.7 ? "high" : "low"}`}>
                        重要性 {m.importance.toFixed(2)}
                      </span>
                      <span className="memory-time" title={`创建 ${fmtTime(m.created_at)} · 最近访问 ${fmtTime(m.last_accessed)}`}>
                        创建 {fmtTime(m.created_at)}
                      </span>
                    </div>
                    {editing === m.id ? (
                      <div className="memory-edit">
                        <textarea
                          className="settings-control"
                          rows={3}
                          value={draft}
                          onChange={(e) => setDraft(e.target.value)}
                        />
                        <label className="memory-imp-edit">
                          重要性
                          <select
                            className="settings-control"
                            value={draftImp ?? ""}
                            onChange={(e) => setDraftImp(e.target.value === "" ? null : Number(e.target.value))}
                          >
                            <option value="">保留原值（{m.importance.toFixed(2)}）</option>
                            {[0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1].map((v) => (
                              <option key={v} value={v}>{v.toFixed(1)}</option>
                            ))}
                          </select>
                        </label>
                        <div className="settings-savebar">
                          <button className="settings-btn-primary" onClick={() => saveEdit(m.id)}>保存</button>
                          <button className="settings-btn-ghost" onClick={() => { setEditing(null); setDraftImp(null); }}>取消</button>
                        </div>
                      </div>
                    ) : (
                      <div className="memory-text-wrap">
                        <div
                          className={`memory-text ${isLong && !isOpen ? "clamped" : ""}`}
                          onClick={() => startEdit(m)}
                          title="点击编辑"
                        >
                          {m.text}
                        </div>
                        {isLong && (
                          <button
                            className="memory-more"
                            onClick={() => toggleExpand(m.id)}
                            aria-expanded={isOpen}
                          >
                            {isOpen ? "收起" : "展开全文"}
                          </button>
                        )}
                      </div>
                    )}
                    {editing !== m.id && (
                      <button className="memory-del" onClick={() => remove(m.id)} title="删除">删除</button>
                    )}
                  </div>
                  );
                })}
              </div>
            )}
          </section>
        ))}
        {items.length === 0 && <div className="muted">暂无长期记忆。</div>}
        {items.length > 0 && visible.length === 0 && <div className="muted">没有匹配「{query}」的记忆。</div>}
      </div>
    </SectionCard>
  );
}
