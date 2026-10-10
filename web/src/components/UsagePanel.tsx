import { useEffect, useMemo, useRef, useState } from "react";
import type { SessionUsageDetail, UsageRow, UsageTrendOut, UsageTrendPoint } from "../lib/types";
import { api } from "../api/client";
import { timeAgo } from "../lib/time";
import { saveBlob as saveFileBlob, saveOutcomeMessage } from "../lib/saveFile";
import { Icons } from "../lib/icons";
import { toErrorMessage } from "../lib/error";
import { MetricsPanel } from "./MetricsPanel";

/** 用量页（画布 6:220）：页头 + 4 统计卡 + 成本趋势图 + 用量明细面板（圆角 16 白底表格） */
function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

/** CSV 字段转义：含逗号/引号/换行时加引号并把内部引号翻倍 */
function escapeCsv(v: string | number): string {
  const s = String(v ?? "");
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** 纯字符串日期 +1 天（UTC 构造 + ISO 取串，**避开本地时区偏移**） */
function nextDay(d: string): string {
  const [y, m, dd] = d.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, dd + 1)).toISOString().slice(0, 10);
}

/** F10：成本趋势图（纯 SVG 柱状图，不引入图表依赖；与会话用量同源） */
function CostTrendChart({ points }: { points: UsageTrendPoint[] }) {
  if (points.length === 0) {
    return <div className="trend-empty">该时间范围内暂无用量数据</div>;
  }
  // ★ 2026-10-05 修正（视觉准确性）：接口只返回**有用量**的天，此前按 `points.length`
  // 平分槽位 ⇒ 中间没数据的天被"挤掉"、柱间距不等距 ⇒ 稀疏用量会被读成连续用量。
  // 现按**连续日历天**铺满槽位（缺失天补 0），时间轴等距；柱高仍只反映真实用量。
  const byDate = new Map(points.map((p) => [p.date, p]));
  const axis: UsageTrendPoint[] = [];
  for (let d = points[0].date; d <= points[points.length - 1].date; d = nextDay(d)) {
    axis.push(byDate.get(d) ?? { date: d, cost: 0, prompt_tokens: 0, completion_tokens: 0 });
  }
  const W = 720;
  const H = 170;
  const pad = { l: 52, r: 12, t: 12, b: 28 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const maxCost = Math.max(...axis.map((p) => p.cost));
  const slot = innerW / axis.length;
  const barW = Math.max(2, Math.min(28, slot * 0.62));
  const ticks = [maxCost, maxCost / 2, 0];

  return (
    <svg className="trend-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="每日成本趋势">
      {ticks.map((t, i) => {
        const y = pad.t + innerH - (maxCost > 0 ? (t / maxCost) * innerH : 0);
        return (
          <g key={i}>
            <line x1={pad.l} x2={W - pad.r} y1={y} y2={y} className="trend-grid" />
            <text x={pad.l - 8} y={y + 4} className="trend-tick" textAnchor="end">
              ¥{t.toFixed(t >= 1 ? 2 : 4)}
            </text>
          </g>
        );
      })}
      {axis.map((p, i) => {
        const h = maxCost > 0 ? (p.cost / maxCost) * innerH : 0;
        const x = pad.l + slot * i + (slot - barW) / 2;
        const y = pad.t + innerH - h;
        return (
          <rect
            key={p.date}
            x={x}
            y={y}
            width={barW}
            height={Math.max(h, p.cost > 0 ? 2 : 0)}
            className="trend-bar"
          >
            <title>{`${p.date}　¥${p.cost.toFixed(6)}　${p.prompt_tokens + p.completion_tokens} tokens`}</title>
          </rect>
        );
      })}
      <text x={pad.l} y={H - 8} className="trend-tick">
        {axis[0].date}
      </text>
      <text x={W - pad.r} y={H - 8} className="trend-tick" textAnchor="end">
        {axis[axis.length - 1].date}
      </text>
    </svg>
  );
}

const TREND_DAY_OPTIONS = [7, 30, 90];

export function UsagePanel({ onOpenChat }: { onOpenChat?: (sessionId: number) => void }) {
  const [rows, setRows] = useState<UsageRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionUsageDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // F10：成本趋势（近 N 天，与会话用量同源）
  const [trendDays, setTrendDays] = useState(30);
  const [trend, setTrend] = useState<UsageTrendOut | null>(null);
  const [trendError, setTrendError] = useState<string | null>(null);
  /** ★ P-05（2026-10-09）：导出 CSV 的结果提示（取消不打扰、失败可见）。 */
  const [exportMsg, setExportMsg] = useState<string | null>(null);

  useEffect(() => {
    api
      .getUsage()
      .then(setRows)
      .catch((e) => setError(toErrorMessage(e) || "加载用量失败"));
  }, []);

  // 切换天数时丢弃过期响应，避免快速切换导致旧数据覆盖新数据
  useEffect(() => {
    let alive = true;
    setTrendError(null);
    api
      .getUsageTrend(trendDays)
      .then((t) => {
        if (alive) setTrend(t);
      })
      .catch((e: unknown) => {
        if (alive) setTrendError(toErrorMessage(e) || "加载趋势失败");
      });
    return () => {
      alive = false;
    };
  }, [trendDays]);

  /** 展开查看会话详细用量（消息 / 工具调用 / Trace 轮次）；N5：序号守卫丢弃过期响应 */
  const detailSeqRef = useRef(0);
  const openDetail = async (sessionId: number) => {
    // N5 修复（2026-09-10）：快速点两个会话明细时旧响应可能后到并覆盖新响应
    // （弹窗显示错会话数据）——与 J2/J8/L8 同款的请求序号守卫。
    const seq = ++detailSeqRef.current;
    setDetailLoading(true);
    try {
      const d = await api.getSessionUsage(sessionId);
      if (seq !== detailSeqRef.current) return;
      setDetail(d);
    } catch (e: unknown) {
      if (seq !== detailSeqRef.current) return;
      setError(toErrorMessage(e) || "加载明细失败");
    } finally {
      if (seq === detailSeqRef.current) setDetailLoading(false);
    }
  };

  /** F10：导出会话用量为 CSV（BOM 前缀，保证 Excel 正确识别 UTF-8 中文） */
  const exportCsv = () => {
    const header = ["会话", "模型", "消息数", "工具调用", "提示tokens", "补全tokens", "成本(¥)", "最近活动"];
    const body = sortedRows.map((r) => [
      r.title,
      r.model,
      r.num_messages,
      r.num_tool_runs,
      r.prompt_tokens,
      r.completion_tokens,
      r.cost.toFixed(6),
      r.last_at ? new Date(r.last_at).toLocaleString() : "",
    ]);
    const csv = [header, ...body].map((cols) => cols.map(escapeCsv).join(",")).join("\r\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
    const name = `papermate-usage-${new Date().toISOString().slice(0, 10)}.csv`;
    // ★ P-05（2026-10-09）：桌面端走「原生另存为 + 后端写盘 + 明确路径」；浏览器回退 <a download>。
    void (async () => {
      try {
        const out = await saveFileBlob(name, blob);
        if (out.kind === "cancelled") return;
        setExportMsg(saveOutcomeMessage(out));
      } catch (e) {
        setExportMsg(`导出失败：${e instanceof Error ? e.message : String(e)}`);
      }
    })();
  };

  const stats = useMemo(() => {
    const sessions = rows.length;
    const tokens = rows.reduce((n, r) => n + r.prompt_tokens + r.completion_tokens, 0);
    const tools = rows.reduce((n, r) => n + r.num_tool_runs, 0);
    const cost = rows.reduce((n, r) => n + r.cost, 0);
    return { sessions, tokens, tools, cost };
  }, [rows]);

  /** 默认按「最近活动」倒序，无活动的排最后 */
  const sortedRows = useMemo(
    () =>
      [...rows].sort((a, b) => {
        const ta = a.last_at ? Date.parse(a.last_at) : 0;
        const tb = b.last_at ? Date.parse(b.last_at) : 0;
        return tb - ta;
      }),
    [rows],
  );

  // ★ 2026-10-05 口径/可读性修正：
  // ① 原标签「总对话次数」实际统计的是**会话数**（`rows.length`）—— 名实不符，改为「会话数」；
  // ② 成本改为**自适应精度**：总额很小时 `toFixed(2)` 会显示 `¥0.00`，看着像"完全没花钱"，
  //    实际可能已经产生了小额消耗 ⇒ < ¥0.01 时保留 4 位小数；
  // ③ 成本提示语明示「本地估算」—— 该数字由内置定价表折算，**不是厂商账单**。
  const costText =
    stats.cost > 0 && stats.cost < 0.01 ? `¥${stats.cost.toFixed(4)}` : `¥${stats.cost.toFixed(2)}`;
  const cards = [
    { label: "会话数", value: String(stats.sessions), trend: "全部会话", up: true },
    { label: "总 token 消耗", value: fmtTokens(stats.tokens), trend: "提示 + 补全合计", up: true },
    { label: "工具调用次数", value: String(stats.tools), trend: "全部工具调用", up: true },
    { label: "总成本", value: costText, trend: "本地估算 · 全部会话累计", up: false },
  ];

  return (
    <div className="usage-page">
      <header className="usage-header">
        <div>
          <h2>用量统计</h2>
          <p className="muted">
            按会话聚合的 token 消耗与成本概览，趋势与明细同源可对账。
            成本为<strong>本地估算</strong>（按内置定价表折算；未登记的模型按最贵档保守估算），
            <strong>非厂商账单</strong>，实际扣费以服务商为准。
          </p>
        </div>
        <button className="usage-export" onClick={exportCsv} disabled={rows.length === 0} title="导出会话用量为 CSV">
          导出 CSV
        </button>
        {exportMsg && <span className="muted">{exportMsg}</span>}
      </header>

      {error && <div className="error-box">{error}</div>}

      {/* D1 + A6（2026-09-21）：运行指标与阈值告警 —— 与成本趋势同源，故置于用量页顶部 */}
      <MetricsPanel days={7} />

      {/* 统计卡片组（画布 6:240） */}
      <div className="usage-cards">
        {cards.map((c) => (
          <div className="usage-card" key={c.label}>
            <span className="usage-card-label">{c.label}</span>
            <span className="usage-card-value">{c.value}</span>
            <span className={`usage-card-trend ${c.up ? "up" : ""}`}>{c.trend}</span>
          </div>
        ))}
      </div>

      {/* F10：成本趋势图 */}
      <div className="usage-trend-panel">
        <div className="usage-trend-head">
          <span className="usage-table-title">成本趋势</span>
          <div className="trend-days">
            {TREND_DAY_OPTIONS.map((d) => (
              <button
                key={d}
                className={`trend-day-btn ${trendDays === d ? "active" : ""}`}
                onClick={() => setTrendDays(d)}
              >
                近 {d} 天
              </button>
            ))}
          </div>
        </div>
        {trendError ? (
          <div className="error-box">{trendError}</div>
        ) : trend === null ? (
          <div className="trend-empty">趋势加载中…</div>
        ) : (
          <>
            <CostTrendChart points={trend.points} />
            <div className="trend-foot muted">
              近 {trend.days} 天累计 ¥{trend.total_cost.toFixed(4)} · 共 {trend.points.length} 天有用量记录
            </div>
          </>
        )}
      </div>

      {/* 用量明细面板（画布 6:257） */}
      <div className="usage-table-panel">
        <div className="usage-table-head">
          <span className="usage-table-title">会话用量明细</span>
        </div>
        <table className="usage-table">
          <thead>
            <tr>
              <th className="col-session">会话</th>
              <th className="col-model">模型</th>
              <th className="col-num">消息</th>
              <th className="col-num">工具</th>
              <th className="col-token">提示 tokens</th>
              <th className="col-token">补全 tokens</th>
              <th className="col-cost">成本（¥）</th>
              <th className="col-time">最近活动</th>
              <th className="col-op">操作</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((r) => (
              <tr key={r.session_id}>
                <td className="col-session">{r.title}</td>
                <td className="col-model">
                  <code>{r.model || "—"}</code>
                </td>
                <td className="col-num">{r.num_messages}</td>
                <td className="col-num">{r.num_tool_runs}</td>
                <td className="col-token">{r.prompt_tokens.toLocaleString()}</td>
                <td className="col-token">{r.completion_tokens.toLocaleString()}</td>
                <td className="col-cost">{r.cost.toFixed(4)}</td>
                <td className="col-time">{timeAgo(r.last_at)}</td>
                <td className="col-op">
                  <button className="usage-open" onClick={() => openDetail(r.session_id)} title="查看会话详细用量">
                    明细
                  </button>
                  {onOpenChat && (
                    <button className="usage-open" onClick={() => onOpenChat(r.session_id)}>
                      打开
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={9} className="usage-empty">
                  暂无用量数据（完成一次对话后此处会按会话聚合统计）
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* 会话详细用量（参考主流平台呈现：消息/工具/Trace 三层） */}
      {detail && (
        <div className="modal-mask" onClick={() => setDetail(null)}>
          <div className="modal usage-detail-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div>
                <div className="reader-title">会话用量明细：{detail.title}</div>
                <div className="reader-sub">
                  {detail.usage.num_messages} 条消息 · {detail.usage.num_tool_runs} 次工具 ·
                  {detail.usage.prompt_tokens + detail.usage.completion_tokens} tokens · ¥{detail.usage.cost.toFixed(4)}
                  {detail.usage.model ? ` · 模型 ${detail.usage.model}` : ""}
                </div>
              </div>
              <button className="icon-btn" onClick={() => setDetail(null)} aria-label="关闭"><Icons.x size={14} /></button>
            </div>

            <div className="usage-detail-body">
              <section className="usage-detail-section">
                <h4>消息（{detail.messages.length}）</h4>
                <table className="usage-table compact">
                  <thead>
                    <tr><th>角色</th><th>模型</th><th>内容</th><th>tokens</th><th>成本 ¥</th></tr>
                  </thead>
                  <tbody>
                    {detail.messages.map((m) => (
                      <tr key={m.id}>
                        <td><span className={`msg-role ${m.role}`}>{m.role}</span></td>
                        {/* 该轮**实际生效**的模型（meta_json.model）；老消息为 null → 显示 "—"，
                            不用会话当前模型倒填（那会伪造历史） */}
                        <td className="muted">{m.model ?? "—"}</td>
                        <td className="msg-content">{m.content || "—"}</td>
                        <td>{m.prompt_tokens + m.completion_tokens}</td>
                        <td>{m.cost.toFixed(6)}</td>
                      </tr>
                    ))}
                    {detail.messages.length === 0 && <tr><td colSpan={5} className="usage-empty">无消息</td></tr>}
                  </tbody>
                </table>
              </section>

              <section className="usage-detail-section">
                <h4>工具调用（{detail.tool_runs.length}）</h4>
                <table className="usage-table compact">
                  <thead>
                    <tr><th>工具</th><th>参数</th><th>结果摘要</th><th>状态</th></tr>
                  </thead>
                  <tbody>
                    {detail.tool_runs.map((t) => (
                      <tr key={t.id}>
                        <td><code>{t.tool_name}</code></td>
                        <td className="msg-content">{t.args || "—"}</td>
                        <td className="msg-content">{t.result_summary || "—"}</td>
                        <td><span className={`tool-ok ${t.ok ? "ok" : "err"}`}>{t.ok ? <Icons.check size={13} /> : <Icons.x size={13} />}</span></td>
                      </tr>
                    ))}
                    {detail.tool_runs.length === 0 && <tr><td colSpan={4} className="usage-empty">无工具调用</td></tr>}
                  </tbody>
                </table>
              </section>

              <section className="usage-detail-section">
                <h4>Agent 轮次（Trace，{detail.traces.length}）</h4>
                {detail.traces.map((tr) => (
                  <div key={tr.trace_id} className="trace-card">
                    <code>{tr.trace_id}</code>
                    <span>steps={tr.total_steps}</span>
                    <span>¥{tr.total_cost.toFixed(6)}</span>
                    <span>{tr.total_latency_ms}ms</span>
                    <div className="trace-events">{tr.event_summary.join(" → ")}</div>
                  </div>
                ))}
                {detail.traces.length === 0 && <div className="muted">无 Trace 记录</div>}
              </section>
            </div>
          </div>
        </div>
      )}
      {detailLoading && <div className="modal-mask"><div className="modal loading-mini">加载明细…</div></div>}
    </div>
  );
}
