/**
 * 上下文用量**圆环**的几何计算（2026-10-03，用户要求"设置成圆环展示用量"）。
 *
 * ## 与横条分段条的**同一份数据、两种画法**
 * 横条是"100% 铺满、按分项比例分配"；圆环要同时表达**两件事**：
 * 1. **总量**——弧总长 = 真实占用率 `pct`（未用满时留空轨道，像参考图那样）；
 * 2. **构成**——填充部分内部再按分项比例切成多色段。
 *
 * ⇒ 所以每段长度 = `(分项占比 / 100) × (占用率 / 100) × 周长`。
 * ⚠️ 这与横条的段长**不同**（横条是 (分项占比/100) × 全宽）——不是 bug，
 *    是"是否把未用满的部分画出来"这个语义差别。
 *
 * ★ 抽成纯函数：SVG 的 `stroke-dasharray/offset` 极易写反（正负号、起点方向、周长计算），
 *   而**错了在页面上的表现是"环看起来也像回事"**（不会抛错）⇒ 必须能被行为测试钉住。
 */

/** 单个弧段的 SVG 参数。 */
export interface RingSegment {
  key: string;
  color: string;
  /** `stroke-dasharray`：`"实长 空白长"` */
  dasharray: string;
  /** `stroke-dashoffset`：负值表示沿圆周向前推进 */
  dashoffset: number;
}

/**
 * 计算圆环各分段的 SVG 参数。
 *
 * @param buckets 分项（`pct` 为该分项占**分项合计**的百分比）
 * @param usagePct 真实占用率（0–100）—— 决定**填充弧的总长**
 * @param radius 半径（viewBox 单位）
 * @param gap 段与段之间留白（viewBox 单位）；太大会在小环上"断开"，默认 1.5
 *
 * ⚠️ `circumference = 2πr`，`dasharray` 用"实长 剩余"而不是 `len, C`（逗号写法兼容但空格更通用）。
 * ⚠️ 段长至少保留 `0.01`，避免 `dasharray="0 C"` 这类退化值被某些渲染器忽略。
 */
export function ringSegments(
  buckets: { key: string; color: string; pct: number }[],
  usagePct: number,
  radius: number,
  gap = 1.5,
): { segments: RingSegment[]; circumference: number } {
  const c = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, Number.isFinite(usagePct) ? usagePct : 0));
  // 填充弧的总长（viewBox 单位）
  const filled = (clamped / 100) * c;

  const segments: RingSegment[] = [];
  let consumed = 0;
  for (const b of buckets) {
    const share = Math.max(0, Math.min(100, Number.isFinite(b.pct) ? b.pct : 0)) / 100;
    // 段长 = 分项占比 × 填充弧总长；最后一段吃掉累计误差，避免环"合不拢"
    const rawLen = share * filled;
    const len = Math.max(0.01, rawLen - gap);
    segments.push({
      key: b.key,
      color: b.color,
      dasharray: `${len} ${Math.max(0.01, c - len)}`,
      // ⚠️ 首段写成 `0` 而不是 `-0`：`-0` 在 SVG 里无害，但 JS 的 `Object.is(-0, 0)` 为 false，
      // 会让"首段 offset 必须为 0"这类断言莫名其妙地失败（也会让日志/快照里出现 `-0`）。
      dashoffset: consumed === 0 ? 0 : -consumed,
    });
    consumed += rawLen;
  }
  return { segments, circumference: c };
}
