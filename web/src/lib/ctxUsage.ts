/**
 * 上下文占用率（P1-a，2026-10-03）。
 *
 * ## 为什么要有这个
 * 市面 Claude Code / Cursor 都有 percent-used 指示器；本项目**只有后端日志**
 * （`[上下文构成]`，实测 241 条），用户侧完全看不到离熔断多远 ⇒
 * 真实发生过「**突然熔断**」（session 18 曾 30630 > 30000，用户毫无预警）。
 *
 * ## ★ 为什么抽成纯函数而不是写在组件里
 * 本项目前端测试以 `?raw` **源码文本契约**为主，而文本断言**守不住逻辑正确性**
 * ——实测（2026-10-02 置顶那次）注入"把排序换掉"后，12 条文本断言**全绿假象**。
 * ⇒ 凡有分支/算术的地方，抽成可 import 的纯函数，用**行为测试**直接验证。
 */

/** 占用率的展示信息；`null` 表示**信息不足，不渲染**（优雅降级）。 */
export interface CtxUsageInfo {
  /** 0–100 的整数百分比（超出上限时截断到 100） */
  pct: number;
  /** 是否达到警示线（应由后端 `compact_trigger_tokens` 决定） */
  warn: boolean;
  /** 形如 `12.5k / 30k` —— 分子是单次峰值、分母是单次上限 */
  label: string;
}

/** 千分位友好格式：12537 → `12.5k`；30000 → `30k`；980 → `980`。 */
export function fmtK(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n < 1000) return String(Math.round(n));
  const k = n / 1000;
  // 整数 k 不显示小数（30k 而不是 30.0k）；否则保留一位（12.5k）
  return Number.isInteger(k) ? `${k}k` : `${k.toFixed(1)}k`;
}

/**
 * 计算占用率。
 *
 * @param peak   单次**峰值** prompt（后端 `peak_prompt_tokens`）
 * @param budget 单次上限（后端 `max_tokens_budget`）
 * @param warnAt 警示线（后端 `compact_trigger_tokens`，= 上限 × 0.7）
 *
 * ★ 判据是**单次峰值 ÷ 单次上限**，**不是** `total_tokens`（整轮累计）——
 * 熔断看的是单次，用累计值会严重高估（实测整轮 27k / 单次峰值仅 12.5k）。
 *
 * ★ `warnAt` 由后端下发，**前端不乘 0.7**：复制派生逻辑 = 再造一把尺子
 *（后端改比例、前端不改 ⇒ 警示时机错位）。
 * 缺 `warnAt` 时退回 70%（与当前后端一致），并在注释里标明这是**兜底**。
 *
 * ⚠️ 任一必需字段缺失/非法 ⇒ 返回 `null` ⇒ 组件**整块不渲染**（老后端不得报错）。
 */
export function ctxUsage(
  peak?: number | null,
  budget?: number | null,
  warnAt?: number | null,
): CtxUsageInfo | null {
  if (typeof peak !== "number" || !Number.isFinite(peak) || peak < 0) return null;
  if (typeof budget !== "number" || !Number.isFinite(budget) || budget <= 0) return null;

  const ratio = peak / budget;
  const pct = Math.min(100, Math.round(ratio * 100));

  const threshold =
    typeof warnAt === "number" && Number.isFinite(warnAt) && warnAt > 0
      ? warnAt
      : budget * 0.7; // 兜底：与后端 `compact_trigger_tokens` 当前比例一致

  return { pct, warn: peak >= threshold, label: `${fmtK(peak)} / ${fmtK(budget)}` };
}
