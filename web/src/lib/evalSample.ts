/**
 * 忠实度看板 ·「这份样本有多可信」的护栏文案（2026-10-05）。
 *
 * ## 为什么需要
 *
 * 看板取「最近 N 条非空 Trace」，但这份样本可能**又旧又少**，而它会被当成"现状"来读。
 * 实测踩到：窗口 = `10-03 14:28 ~ 10-04 13:13`（最新一条已 **≈19.5 小时**前），
 * 且 10 条里 **8 条早于 10-03 的一次修复** ⇒ 看板在展示"修复前的世界"。
 *
 * ## 判据在后端、措辞在前端
 *
 * `stale` / `insufficient` / `age_hours` 都由**后端**算好（`dev_eval._sample_quality`）。
 * ⚠️ 之所以不让前端算年龄：`window.to` 是**裸时间戳**（SQLite 落库的 naive UTC，无时区标记），
 * 前端 `new Date()` 会按**浏览器本地时区**解析 ⇒ GMT+8 下凭空多出 **8 小时**
 * —— 本项目已经在时间口径上吃过这个亏（离线核查时我也刚写错过一次）。
 * 这里只把后端给的小时数**措辞化**，绝不自己解析时间戳。
 */

/** 采样窗口（与后端 `agent.window` 对应；新增字段全部可选，兼容旧后端） */
export interface SampleWindow {
  limit: number;
  sampled: number;
  table_total: number;
  excluded_empty: number;
  from: string | null;
  to: string | null;
  /** 窗口内**最新一条**距现在的小时数（后端按 UTC 口径算好；无样本为 null） */
  age_hours?: number | null;
  /** 超过多少小时算"过时"（后端常量，随响应下发以便前端如实展示阈值） */
  max_age_hours?: number;
  stale?: boolean;
  /** 窗口**没填满**（表里非空样本 < limit）——比"n 小"更强的信号 */
  insufficient?: boolean;
}

export interface SampleWarning {
  level: "warn";
  text: string;
}

/** 由**后端给出的小时数**措辞（绝不自己解析时间戳） */
export function formatAgeHours(hours: number | null | undefined): string {
  if (hours == null || !Number.isFinite(hours)) return "时间未知";
  if (hours < 1) return "1 小时内";
  if (hours < 48) return `${Math.round(hours)} 小时前`;
  return `${Math.round(hours / 24)} 天前`;
}

/**
 * 生成样本可信度提示（空数组 = 无需提示）。
 *
 * 两个判据互相**独立**，可同时出现：
 *  - `stale`：最新样本太旧 ⇒ 指标可能已不代表当前版本的行为；
 *  - `insufficient`：窗口都没填满 ⇒ 连"最近 N 条"都凑不齐。
 */
export function sampleWarnings(w: SampleWindow | null | undefined): SampleWarning[] {
  if (!w) return [];
  const out: SampleWarning[] = [];

  if (w.stale) {
    const threshold = w.max_age_hours ?? 24;
    out.push({
      level: "warn",
      text:
        `样本已过时：最近一条轨迹在 ${formatAgeHours(w.age_hours)}` +
        `（阈值 ${threshold} 小时）——下方「运行期指标」可能已不代表当前版本的行为。`,
    });
  }

  if (w.insufficient) {
    out.push({
      level: "warn",
      text:
        `样本不足：窗口内仅 ${w.sampled}/${w.limit} 条` +
        `（表内非空共 ${w.table_total} 条）——小样本，只宜看趋势。`,
    });
  }

  return out;
}
