/**
 * 上下文用量**横条**的分段宽度（2026-10-03，用户要求面板"换成横条统计"）。
 *
 * ## 与圆环的差别只在"画法"，语义**完全一致**
 * - 底槽 = **100%**（总容量）；填充 = 真实占用率；填充内部按分项比例切多色。
 * - 所以横条与圆环表达的是同一组数字（可互相对照）。
 *
 * ## 为什么要有"底槽"
 * 参考图里未使用的部分是**深色底槽**（可见），而不是空白 ——
 * 这样"用了多少 / 还剩多少"一眼可见；若只画分段，未用部分会**消失**，
 * 让人误以为"总量就是 100%"。
 *
 * ★ 抽纯函数：分段宽度是**百分比**（与 SVG 弧长不同，纯算术），
 *   但"分项占比 × 占用率"这个换算写反同样**不会报错**（只是条子画歪）⇒ 必须可测。
 */

export interface BarSegment {
  key: string;
  color: string;
  /** 宽度百分比（占整条 = 总容量的百分比） */
  widthPct: number;
}

/**
 * 计算横条各分段的宽度。
 *
 * @param buckets  分项（`pct` = 该分项占**分项合计**的百分比）
 * @param usagePct 真实占用率（0–100）
 * @param gapPct   段间留白（百分比）；默认 0.4（细条上太大就断了）
 */
export function barSegments(
  buckets: { key: string; color: string; pct: number }[],
  usagePct: number,
  gapPct = 0.4,
): BarSegment[] {
  const clamped = Math.max(0, Math.min(100, Number.isFinite(usagePct) ? usagePct : 0));
  const out: BarSegment[] = [];
  buckets.forEach((b, i) => {
    const share = Math.max(0, Math.min(100, Number.isFinite(b.pct) ? b.pct : 0)) / 100;
    // 段宽 = 分项占比 × 占用率（**不**是分项占比本身 —— 那是"占分项合计"的口径）
    const raw = share * clamped;
    if (raw <= 0) {
      out.push({ key: b.key, color: b.color, widthPct: 0 });
      return;
    }
    // ⚠️ 留白只扣在**段与段之间**（n-1 个缝隙），**不是每段都扣** ——
    // 每段都扣会让"段宽之和"比占用率少 n×gap（5 段就少 2%），看着像"用量算少了"。
    const hasNext = i < buckets.length - 1;
    const w = hasNext ? Math.max(0.4, raw - gapPct) : raw;
    out.push({ key: b.key, color: b.color, widthPct: w });
  });
  return out;
}
