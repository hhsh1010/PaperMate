/**
 * 上下文用量**横条**的分段宽度测试（2026-10-03）。
 *
 * ★ 核心不变量：**段宽之和 ≈ 真实占用率**（不是 100%）。
 * 与圆环语义一致：底槽 = 总容量，填充 = 用了多少，填充内部按分项比例分色。
 * ⇒ 写错就会变成"条子永远满格"（看不出用了多少）或"双重折算"（只画出一小截）。
 */
import { describe, it, expect } from "vitest";
import { barSegments } from "./ctxBar";
import panelSrc from "../components/CtxUsagePanel.tsx?raw";

const BUCKETS = [
  { key: "system", color: "var(--cat-green)", pct: 10.6 },
  { key: "tool_defs", color: "var(--cat-blue)", pct: 8.4 },
  { key: "messages", color: "var(--cat-purple)", pct: 47.9 },
  { key: "tool_results", color: "var(--cat-rose)", pct: 32.9 },
  { key: "notes", color: "var(--cat-amber)", pct: 0.2 },
];

const sum = (segs: { widthPct: number }[]) => segs.reduce((a, s) => a + s.widthPct, 0);

describe("barSegments · 几何", () => {
  it("★ 段宽之和 ≈ 真实占用率（**不是** 100%）", () => {
    // 分项占比之和 = 100% ⇒ 段宽之和 ≈ 占用率（**少掉的是段间留白**：5 段 ⇒ 4 缝 × 0.4 = 1.6）
    expect(sum(barSegments(BUCKETS, 68))).toBeCloseTo(68 - 1.6, 0);
    expect(sum(barSegments(BUCKETS, 68))).toBeLessThan(68);
  });

  it("★ 占用率 100 ⇒ 段宽之和 ≈ 100（满格）", () => {
    expect(sum(barSegments(BUCKETS, 100))).toBeCloseTo(100 - 1.6, 0);
  });

  it("★ 占用率 12 ⇒ 段宽之和 ≈ 12（低占用时只画一小截）", () => {
    // 图1 的真实场景：12.0% / 12.3K / 100K
    expect(sum(barSegments(BUCKETS, 12))).toBeCloseTo(12 - 1.6, 0);
  });

  it("★ 占比大者段更宽（顺序不能颠倒）", () => {
    const segs = barSegments(BUCKETS, 68);
    const w = (k: string) => segs.find((s) => s.key === k)!.widthPct;
    expect(w("messages")).toBeGreaterThan(w("tool_results"));
    expect(w("tool_results")).toBeGreaterThan(w("system"));
  });

  it("占用率 0 ⇒ 段宽全为 0（不画）", () => {
    for (const s of barSegments(BUCKETS, 0)) expect(s.widthPct).toBe(0);
  });

  it("无分项 ⇒ 空数组", () => {
    expect(barSegments([], 68)).toEqual([]);
  });

  it("非法输入不产生 NaN", () => {
    const segs = barSegments([{ key: "a", color: "c", pct: Number.NaN }], Number.NaN);
    for (const s of segs) expect(Number.isFinite(s.widthPct)).toBe(true);
  });

  it("极窄段保底可见（不会被 gap 吃成 0）", () => {
    // 0.2% 的分项在 12% 占用率下只有 0.024% ⇒ 会被 gap 吃掉
    const segs = barSegments(BUCKETS, 12);
    expect(segs.find((s) => s.key === "notes")!.widthPct).toBeGreaterThan(0);
  });
});

describe("面板接入（源码契约）", () => {
  it("★ 面板用横条（底槽 + 分段），不再是圆环", () => {
    expect(panelSrc).toContain("ctx-panel-bar");
    expect(panelSrc).toContain("barSegments(data.buckets, data.pct)");
    // 面板里不再有圆环
    expect(panelSrc).not.toContain("ctx-panel-ring-row");
    expect(panelSrc).not.toContain("RING_R");
  });

  it("★ 数值在上、条在下（与参考图一致）", () => {
    const sumAt = panelSrc.indexOf("ctx-panel-summary");
    const barAt = panelSrc.indexOf("ctx-panel-bar");
    expect(sumAt).toBeGreaterThan(-1);
    expect(barAt).toBeGreaterThan(sumAt);
  });

  it("★ 功能栏小圆环**保留**（两种形态并存，各司其职）", () => {
    expect(panelSrc).toContain("ctx-ring-btn");
    expect(panelSrc).toContain("ringSegments(data.buckets, data.pct, RING_BTN_R");
  });

  it("⚠️ 分段宽度已是「占整条」百分比 ⇒ 不能再套一层按占用率缩放的容器", () => {
    // 若有人再加 width: {pct}% 的外层容器，就会双重折算（条子只画出一小截）
    expect(panelSrc).not.toContain("width: `${data.pct}%`");
  });
});
