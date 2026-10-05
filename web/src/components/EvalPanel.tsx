import { useEffect, useState } from "react";
import { Icons } from "../lib/icons";
import type { EvalSummary } from "../lib/types";
import { sampleWarnings } from "../lib/evalSample";
import { api } from "../api/client";
import { toErrorMessage } from "../lib/error";

/** 百分比展示：null → "—"（指标缺失时不编造数字） */
function pct(v: number | null): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}

/** 门禁徽标：达标绿 / 未达基线黄 / 无报告灰 */
function gateClass(passed: boolean | null): string {
  return passed === true ? "pass" : passed === false ? "fail" : "none";
}

/**
 * F11：端到端忠实度看板。
 * 数据来自 GET /api/v1/eval/summary（与 /eval 命令共用 collect_metrics，单一口径）；
 * 本看板为只读聚合、常驻可见：后端路由不跟随 features.dev_eval 开关、也从不返 403
 * （设计意图：避免「功能已交付却默认不可见」），与受该开关门控的 /eval 聊天命令语义不同。
 * 加载失败时由下方 error-box 展示后端返回的原始错误信息。
 */
export function EvalPanel() {
  const [data, setData] = useState<EvalSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    api
      .getEvalSummary()
      .then((d) => {
        if (alive) {
          setData(d);
          setError(null);
        }
      })
      .catch((e: unknown) => {
        if (alive) setError(toErrorMessage(e) || "加载评测数据失败");
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  /** 采样窗口描述：让人知道这些"近期指标"到底覆盖多少轮、什么时间段 */
  const windowHint = (): string => {
    if (!data) return "";
    const w = data.agent.window;
    if (w?.from && w?.to) {
      const d = (s: string) => s.slice(0, 10);
      // ★ 2026-10-05：显式带上样本量「n=」—— 这批指标只有个位数样本，
      //   不写出来读者会默认它是"总体统计"（实测 n=10 时 p95 等分位并无统计意义）。
      return `${d(w.from)}~${d(w.to)} · n=${w.sampled}`;
    }
    return `最近 ${data.agent.count} 轮`;
  };

  // 样本护栏（2026-10-05）：判据由后端算（`dev_eval._sample_quality`），此处只渲染。
  const sampleWarn = data ? sampleWarnings(data.agent.window) : [];

  const cards = data
    ? [
        {
          label: "检索 Recall@5",
          value: pct(data.retrieval["avg_recall@5"]),
          hint: `MRR@10 ${pct(data.retrieval["avg_mrr@10"])} · ${data.retrieval.num_cases ?? "—"} 样例 · ${
            data.retrieval.file ?? "无报告"
          }`,
        },
        {
          label: "溯源率",
          value: pct(data.agent.citation_rate),
          // 样本口径写在卡片上：此前"最近 10 轮"实际把空轨迹算了进去，且不知道是哪几天
          hint: `带引用的完整回答占比（${windowHint()}）`,
        },
        {
          label: "工具成功率",
          value: pct(data.agent.tool_success_rate),
          hint: `tool_result.ok / tool_call（${windowHint()}）`,
        },
        {
          label: "平均成本",
          value: `¥${data.agent.avg_cost.toFixed(4)}`,
          // 明确"有几条真的计过价"：否则 ¥0.0000 既可能是没采集，也可能是全都免费
          // ★ 2026-10-05：补「本地估算」标注 —— 该成本按内置定价表折算，**不是厂商账单**
          hint: `本地估算 · 平均步数 ${data.agent.avg_steps} · 计价 ${
            data.agent.costed_count
          }/${data.agent.count} 条 · 延迟 ${data.agent.avg_latency_ms}ms`,
        },
      ]
    : [];

  const byKind = data ? Object.entries(data.retrieval.by_kind ?? {}) : [];

  return (
    <div className="eval-page">
      <header className="eval-header">
        <div>
          <h2>忠实度看板</h2>
          <p className="muted">检索基线 + 运行期 Agent 指标，与 /eval 命令同一数据口径</p>
        </div>
        {data && (
          <span className={`eval-gate ${gateClass(data.gate.passed)}`}>
            {data.gate.passed === true ? (
              <Icons.checkCircle size={13} />
            ) : data.gate.passed === false ? (
              "!"
            ) : (
              "–"
            )}{" "}
            {data.gate.label}
            （阈值 {data.gate.threshold}）
          </span>
        )}
      </header>

      {error && <div className="error-box">{error}</div>}
      {loading && !data && <div className="trend-empty">加载中…</div>}

      {data && (
        <>
          <div className="usage-cards">
            {cards.map((c) => (
              <div className="usage-card" key={c.label}>
                <span className="usage-card-label">{c.label}</span>
                <span className="usage-card-value">{c.value}</span>
                <span className="usage-card-trend">{c.hint}</span>
              </div>
            ))}
          </div>

          {sampleWarn.length > 0 && (
            <div className="eval-sample-warn">
              <span className="eval-sample-warn-title">运行期指标的样本可信度</span>
              {sampleWarn.map((w) => (
                <span key={w.text}>{w.text}</span>
              ))}
            </div>
          )}

          <div className="eval-section">
            <div className="eval-section-head">
              <span className="usage-table-title">检索基线口径</span>
              <span className="muted">
                {data.retrieval.file ?? "无报告"}
                {data.retrieval.available && ` · 嵌入 ${data.retrieval.embedder} · profile ${data.retrieval.profile}`}
              </span>
            </div>
            {!data.retrieval.available ? (
              <div className="trend-empty">暂无检索评测报告</div>
            ) : (
              <>
                {!data.retrieval.trusted && (
                  // 口径不可信时必须显式说出来：否则用户会把"合成集/fake 嵌入"的数字当真实质量
                  <div className="error-box">
                    基线口径不可信：{data.retrieval.reason}。门禁不出结论，请用
                    <code> --embedder real --profile prod </code>重跑评测。
                  </div>
                )}
                {byKind.length > 0 && (
                  <ul className="eval-trace-list">
                    {byKind.map(([kind, m]) => (
                      <li key={kind} className="eval-trace-item">
                        <code>{kind}</code>
                        <span>n={m.n ?? "—"}</span>
                        <span>recall@5={pct(m["recall@5"])}</span>
                        <span>ndcg@5={pct(m["ndcg@5"])}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </div>

          <div className="eval-section">
            <div className="eval-section-head">
              <span className="usage-table-title">最近 Agent 轨迹</span>
              {data.agent.window && data.agent.window.excluded_empty > 0 && (
                <span className="muted">
                  共 {data.agent.window.table_total} 条，已排除 {data.agent.window.excluded_empty} 条空轨迹
                </span>
              )}
            </div>
            {data.recent_traces.length === 0 ? (
              <div className="trend-empty">暂无轨迹记录（完成一次对话后出现）</div>
            ) : (
              <ul className="eval-trace-list">
                {data.recent_traces.map((t) => (
                  <li key={t.trace_id} className="eval-trace-item">
                    <code>{t.trace_id}</code>
                    <span>steps={t.steps}</span>
                    <span>¥{t.cost.toFixed(4)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="eval-note muted">
            忠实度（faithfulness）与跨厂商成本需真实 LLM + temperature=0 评估（judge_prompts/）；
            当前看板展示检索基线与运行期可观测指标，作为阶段门禁参考。
          </div>
        </>
      )}
    </div>
  );
}
