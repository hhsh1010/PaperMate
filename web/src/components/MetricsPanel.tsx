import { useEffect, useRef, useState } from "react";
import type { MetricsOut } from "../lib/types";
import { api } from "../api/client";
import { toErrorMessage } from "../lib/error";
import { Icons } from "../lib/icons";

/** D1 + A6（2026-09-21）：运行指标 + 阈值告警。

为什么放在用量页：时延/成本/成功率本来就是"用量的另一面"，
与成本趋势图同源，没必要单开一个页面。

设计要点：
- **告警必须能解释**：每条都展示 `阈值` 与 `样本量`。只给个红灯，用户无法判断
  是"系统真出问题"还是"才 2 条样本导致的抖动"。
- **空窗不等于不健康**：没数据时显式说明"样本不足，未判定"，而不是悄悄绿着。
- `unfinished_rate` 文案刻意写"未收尾（含主动停止）"—— 叫"错误率"会把用户主动
  点停止误报成系统故障（实测 43% 里绝大多数是主动取消）。
*/
/** A4：失败分类的中文标签（与后端 `core/failure.py` 的 LABELS 对齐；仅用于展示） */
const FAILURE_LABELS: Record<string, string> = {
  llm_error: "LLM 调用失败",
  tool_error: "工具连续失败",
  no_progress: "连续无进展",
  max_iterations: "达到最大迭代",
  budget_exceeded: "预算熔断",
  retrieval_empty: "检索无结果",
  cancelled: "用户取消",
  unknown: "未标注",
};

export function MetricsPanel({ days = 7 }: { days?: number }) {
  const [m, setM] = useState<MetricsOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    setError(null);
    api
      .getMetrics(days)
      .then((r) => {
        if (aliveRef.current) setM(r);
      })
      .catch((e: unknown) => {
        if (aliveRef.current) setError(toErrorMessage(e) || "加载运行指标失败");
      });
    return () => {
      aliveRef.current = false;
    };
  }, [days]);

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`);

  // 上下文水位（2026-10-02）。★ `ctx_peak_p50` 是 `number | null`：
  // `null` = **尚未采到样本**（老轨迹没埋该字段），**必须显示"—"而不是 0%**
  // ——后者会被读成"水位极低、很安全"，属于谎报安全。
  const tok = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v)));
  const waterline = m && m.ctx_peak_p50 !== null
    ? {
        used: `${pct(m.ctx_peak_p50 / m.ctx_budget)}`,
        hint:
          `单次中位 ${tok(m.ctx_peak_p50)} / 额度 ${tok(m.ctx_budget)}` +
          ` · 触发线 ${tok(m.ctx_trigger)}（${pct(m.ctx_trigger / m.ctx_budget)}）` +
          ` · 余量 ${tok(m.ctx_headroom_median ?? 0)}` +
          (m.ctx_peak_samples ? ` · 样本 ${m.ctx_peak_samples} 条` : ""),
      }
    : null;

  const cells: { label: string; value: string; hint: string }[] = m
    ? [
        { label: "轨迹轮次", value: String(m.num_traces_total), hint: `本次统计 ${m.num_traces} 条` },
        { label: "平均步数", value: m.avg_steps.toFixed(1), hint: "每轮工具/思考步数" },
        { label: "平均时延", value: ms(m.avg_latency_ms), hint: `最长 ${ms(m.max_latency_ms)}` },
        { label: "平均成本", value: `¥${m.avg_cost.toFixed(4)}`, hint: `窗口合计 ¥${m.total_cost.toFixed(4)}` },
        ...(waterline
          ? [{
              label: "上下文水位",
              value: waterline.used,
              hint: waterline.hint,
            }]
          : [{ label: "上下文水位", value: "—", hint: "尚无样本（该指标 2026-10-02 起采集）" }]),
        { label: "未收尾率", value: pct(m.unfinished_rate), hint: "含用户主动停止" },
        { label: "工具失败率", value: pct(m.tool_error_rate), hint: `${m.tool_calls} 次调用` },
        {
          label: "好评率",
          value: m.feedback.total ? pct(m.feedback.good_rate) : "—",
          hint: m.feedback.total ? `${m.feedback.total} 条反馈` : "暂无反馈",
        },
      ]
    : [];

  return (
    <section className="panel metrics-panel">
      <div className="panel-head">
        <div className="panel-title">
          <Icons.chart size={15} /> 运行指标（近 {days} 天）
        </div>
        {m && (
          <span className={`metric-health ${m.healthy ? "ok" : "warn"}`}>
            {m.healthy ? "正常" : `${m.alerts.length} 项超阈值`}
          </span>
        )}
      </div>

      {error && <div className="error-box">{error}</div>}
      {!m && !error && <div className="muted">加载中…</div>}

      {m && (
        <>
          <div className="metric-grid">
            {cells.map((c) => (
              <div className="metric-cell" key={c.label}>
                <div className="metric-label">{c.label}</div>
                <div className="metric-value">{c.value}</div>
                <div className="metric-hint">{c.hint}</div>
              </div>
            ))}
          </div>

          {/* 上下文水位明细（2026-10-02）：把"额度水位"与"证据被动过刀"摆在明面上。
              ★ `ctx_elided_traces` 是**唯一**能看到"这轮证据被裁过"的信号 ——
              那部分完整性损失 judge 测不出来（它只测忠实度/拒答）。 */}
          {(m.ctx_over_trigger > 0 || m.ctx_over_budget > 0 || m.ctx_elided_traces > 0) && (
            <div className="failure-kinds">
              <span className="metric-label">上下文水位</span>
              <ul className="fk-list">
                {m.ctx_over_budget > 0 && (
                  <li className="fk-item">
                    越过<b>{m.ctx_over_budget}</b> 条曾超单次上限（该轮已中止）
                  </li>
                )}
                {m.ctx_over_trigger > 0 && (
                  <li className="fk-item">
                    越过<b>{m.ctx_over_trigger}</b> 条超触发线（已触发压缩/闸门）
                  </li>
                )}
                {m.ctx_elided_traces > 0 && (
                  <li className="fk-item">
                    <b>{m.ctx_elided_traces}</b> 条的上下文被动过刀（闸门裁剪/历史压缩）
                  </li>
                )}
              </ul>
              <div className="metric-hint">
                证据被裁掉的完整性损失评测测不出来（judge 只测忠实度与该不该拒答）
              </div>
            </div>
          )}

          {/* A4（2026-09-21）：失败归因分布 —— 让"失败长什么样"可见，而不是只有一个总数。
              `cancelled` 单列并弱化：它不是故障，混进故障口径会把正常行为报成缺陷。 */}
          {Object.keys(m.failure_kinds ?? {}).length > 0 && (
            <div className="failure-kinds">
              <span className="metric-label">失败归因</span>
              <ul className="fk-list">
                {Object.entries(m.failure_kinds)
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => (
                    <li key={k} className={`fk-item${k === "cancelled" ? " is-cancel" : ""}`}>
                      {FAILURE_LABELS[k] ?? k} <b>{v}</b>
                    </li>
                  ))}
              </ul>
              {m.failure_kinds.unknown > 0 && (
                <div className="metric-hint">
                  其中 {m.failure_kinds.unknown} 条未标注（多为启用归因之前的历史轨迹）
                </div>
              )}
            </div>
          )}

          {m.alerts.length === 0 ? (
            <div className="muted metric-note">
              {m.num_traces === 0
                ? "窗口内暂无轨迹，指标为空"
                : "各项均在阈值内（样本不足的项目已跳过判定）"}
            </div>
          ) : (
            <ul className="alert-list">
              {m.alerts.map((a) => (
                <li key={a.code} className={`alert-item ${a.level}`}>
                  <Icons.alert size={14} />
                  <span className="alert-msg">{a.message}</span>
                  <span className="alert-meta">
                    阈值 {a.threshold} · 样本 {a.sample}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
