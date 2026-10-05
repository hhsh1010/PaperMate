/**
 * 上下文用量分项（2026-10-03，P2）—— 参考 WorkBuddy 的「上下文用量」面板。
 *
 * ## 数据来源与**两条口径**（必须分清，否则会给出互相矛盾的数字）
 * | 值 | 口径 | 用途 |
 * |---|---|---|
 * | `peak_prompt_tokens / max_tokens_budget` | **真实**（厂商返回） | 大号百分比、"已使用 X/Y" |
 * | 各 `context_breakdown[k]` 之间的比例 | **估算**（`_rough_tokens`） | 分段条与分项百分比 |
 *
 * ⇒ **不能**用 `breakdown` 之和当分母去算总占用率：它是估算，与真实峰值有偏差
 *（项目实测估算器误差 p50 ≈5%，但一次性拿它替代真实值就违背了"触发线用真实值"的既有决策）。
 * 本模块刻意把两者**分开计算**，并在 UI 上标注分项为"估算占比"。
 *
 * ★ 抽成纯函数（不写在组件里）——本项目纪律：`?raw` 文本断言守不住逻辑。
 */

/** 后端 `context_breakdown` 的键 → 展示元信息。 */
export interface BucketMeta {
  key: string;
  label: string;
  /** CSS 颜色表达式（**只用既有令牌**，禁 hex —— 项目铁律） */
  color: string;
}

/**
 * 展示顺序**固定**（不按大小排）：位置稳定用户才能"用眼睛记"。
 * 顺序按 WorkBuddy 的观感：固定成本在前（提示词/工具），动态内容在后（消息/结果/注记）。
 */
export const BUCKET_META: BucketMeta[] = [
  { key: "system", label: "系统提示词", color: "var(--cat-green)" },
  { key: "tool_defs", label: "工具定义", color: "var(--cat-blue)" },
  { key: "messages", label: "对话消息", color: "var(--cat-purple)" },
  { key: "tool_results", label: "工具结果", color: "var(--cat-rose)" },
  { key: "notes", label: "记忆与注记", color: "var(--cat-amber)" },
];

/** 分项展示项。 */
export interface BucketView {
  key: string;
  label: string;
  color: string;
  /** token 数 —— ★ 已**等比缩放**到真实总量（各项之和 ≈ 面板大数字）。 */
  tokens: number;
  /** 占**分项合计**的百分比（保留 1 位）。等比缩放**不改变**它。 */
  pct: number;
}

/** 面板所需的全部数据；`null` = 数据不足，不渲染。 */
export interface CtxPanelData {
  /** 真实占用率（0–100，可能被截断到 100） */
  pct: number;
  /** 真实已用（单次峰值 prompt） */
  used: number;
  /** 单次上限 */
  total: number;
  /** 是否达到警示线 */
  warn: boolean;
  /** 分项（已按固定顺序、过滤掉 0 值） */
  buckets: BucketView[];
}

/** 千分位友好：380400 → `380.4K`；30000 → `30K`；980 → `980`。 */
export function fmtTokens(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n < 1000) return String(Math.round(n));
  const k = n / 1000;
  return Number.isInteger(k) ? `${k}K` : `${k.toFixed(1)}K`;
}

/**
 * 组装面板数据。
 *
 * @param peak      真实单次峰值 prompt（`done.peak_prompt_tokens`）
 * @param budget    单次上限（`done.max_tokens_budget`）
 * @param warnAt    警示线（`done.compact_trigger_tokens`）
 * @param breakdown 分项估算（`done.context_breakdown`）
 *
 * ⚠️ 任一必需字段缺失/非法 ⇒ 返回 `null`（**优雅降级**，老后端不得报错）。
 * ⚠️ `breakdown` 为空时**仍返回数据**（分项列表为空），因为大号百分比是真实值、
 *    依然有展示价值 —— 只有连峰值/上限都没有时才整体不渲染。
 */
export function buildCtxPanel(
  peak?: number | null,
  budget?: number | null,
  warnAt?: number | null,
  breakdown?: Record<string, number> | null,
): CtxPanelData | null {
  if (typeof peak !== "number" || !Number.isFinite(peak) || peak < 0) return null;
  if (typeof budget !== "number" || !Number.isFinite(budget) || budget <= 0) return null;

  const pct = Math.min(100, Math.round((peak / budget) * 100));
  const threshold =
    typeof warnAt === "number" && Number.isFinite(warnAt) && warnAt > 0 ? warnAt : budget * 0.7;

  const raw = breakdown ?? {};
  // 只认"正数"分项（0 值不画段；脏值剔除，见下方测试）
  const keys = BUCKET_META.map((m) => m.key).filter((k) => {
    const v = raw[k];
    return typeof v === "number" && Number.isFinite(v) && v > 0;
  });
  const sum = keys.reduce((acc, k) => acc + (raw[k] as number), 0);

  // ★★ 2026-10-03（用户："做"）：**等比缩放到真实总量**，让分项之和 = 面板大数字。
  //
  // 为什么：分项来自 `_rough_tokens` **估算**，实测比厂商真实 prompt **偏低 6%~22%**
  //（`[估算比值]` 日志 6 样本：1.22/1.20/1.06/1.09/1.10/1.15）。
  // 若原样展示，用户会看到「分项加起来 ≠ 大数字」⇒ 要么困惑、要么以为有 bug。
  //
  // ⚠️ **这不产生新的准确性**：等比缩放**不改变相对比例**（谁大谁小仍然由估算决定），
  // 只是把"估算比例"映射到"真实总量"上 —— 厂商**不返回分项明细**，这是硬约束。
  // ⇒ 面板文案必须写明这一点（见 `CtxUsagePanel` 的 footer）。
  //
  // ⚠️ 边界：`sum === 0`（无分项 / 全为脏值）⇒ `scale = 1`，不做除法。
  const scale = sum > 0 ? peak / sum : 1;

  const buckets: BucketView[] = keys.map((k) => {
    const rawTokens = raw[k] as number;
    return {
      key: k,
      label: BUCKET_META.find((m) => m.key === k)!.label,
      color: BUCKET_META.find((m) => m.key === k)!.color,
      // 缩放后的 token 数（各项之和 ≈ peak，四舍五入有 ±几的漂移）
      tokens: Math.round(rawTokens * scale),
      // 占比与缩放前**相同**（等比缩放不改比例）
      pct: Math.round((rawTokens / sum) * 1000) / 10,
    };
  });

  return { pct, used: peak, total: budget, warn: peak >= threshold, buckets };
}
