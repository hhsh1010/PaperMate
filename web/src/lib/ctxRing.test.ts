/**
 * 上下文用量**圆环**的几何测试（2026-10-03）。
 *
 * ★ 为什么这个必须做**行为测试**：
 * SVG 的 `stroke-dasharray/dashoffset` 写反（符号、起点、周长）**不会抛错** ——
 * 页面上环"看起来也像回事"（弧长差一倍、段序颠倒都只是"看着有点怪"）。
 * ⇒ 只有把几何量算出来断言，才能钉住。
 */
import { describe, it, expect } from "vitest";
import { ringSegments } from "./ctxRing";
import panelSrc from "../components/CtxUsagePanel.tsx?raw";
import chatInputSrc from "../components/ChatInput.tsx?raw";

const R = 26;
const C = 2 * Math.PI * R;

const BUCKETS = [
  { key: "system", color: "var(--cat-green)", pct: 10.6 },
  { key: "tool_defs", color: "var(--cat-blue)", pct: 8.4 },
  { key: "messages", color: "var(--cat-purple)", pct: 47.9 },
  { key: "tool_results", color: "var(--cat-rose)", pct: 32.9 },
  { key: "notes", color: "var(--cat-amber)", pct: 0.2 },
];

/** 从 `dasharray: "len rest"` 里取实长。 */
function dashLen(s: string): number {
  return Number.parseFloat(s.split(" ")[0]);
}

describe("ringSegments · 几何", () => {
  it("★ 填充总长 = 占用率 × 周长（弧长表达**总量**，未用满时留空）", () => {
    const { segments } = ringSegments(BUCKETS, 68, R);
    // 各段实长之和 ≈ 68% × C（段间 gap 会吃掉一点点，故用容差）
    const total = segments.reduce((a, s) => a + dashLen(s.dasharray), 0);
    const filled = 0.68 * C;
    expect(total).toBeLessThanOrEqual(filled + 0.01);
    expect(total).toBeGreaterThan(filled - segments.length * 2);
  });

  it("★ 占比更大的分项得到更长的弧（顺序不能颠倒）", () => {
    const { segments } = ringSegments(BUCKETS, 68, R);
    const len = (k: string) => dashLen(segments.find((s) => s.key === k)!.dasharray);
    expect(len("messages")).toBeGreaterThan(len("tool_results"));
    expect(len("tool_results")).toBeGreaterThan(len("system"));
    expect(len("system")).toBeGreaterThan(len("notes"));
  });

  it("★ dashoffset 单调递减且首段为 0（否则弧会重叠/错位）", () => {
    const { segments } = ringSegments(BUCKETS, 68, R);
    expect(segments[0].dashoffset).toBe(0);
    for (let i = 1; i < segments.length; i++) {
      expect(segments[i].dashoffset).toBeLessThan(segments[i - 1].dashoffset);
    }
  });

  it("占用率 0 ⇒ 无可见填充（各段实长退化为最小值）", () => {
    const { segments } = ringSegments(BUCKETS, 0, R);
    for (const s of segments) {
      expect(dashLen(s.dasharray)).toBeLessThan(0.02);
    }
  });

  it("占用率 100 ⇒ 各段实长之和 ≈ 周长（满环）", () => {
    const { segments } = ringSegments(BUCKETS, 100, R);
    const total = segments.reduce((a, s) => a + dashLen(s.dasharray), 0);
    expect(total).toBeGreaterThan(C - segments.length * 2);
  });

  it("⚠️ 100% 占用时每段 dasharray 的**空白部分**不得为负", () => {
    // dasharray="len rest" 里 rest 为负是非法值，某些渲染器会整段丢弃
    const { segments } = ringSegments(BUCKETS, 100, R);
    for (const s of segments) {
      expect(Number.parseFloat(s.dasharray.split(" ")[1])).toBeGreaterThan(0);
    }
  });

  it("无分项 ⇒ 空数组（只画空轨道，不抛错）", () => {
    expect(ringSegments([], 68, R).segments).toEqual([]);
  });

  it("非法输入不抛错、不产生 NaN", () => {
    const bad = [
      { key: "a", color: "c", pct: Number.NaN },
      { key: "b", color: "c", pct: -5 },
      { key: "c", color: "c", pct: 999 },
    ];
    const { segments } = ringSegments(bad, Number.NaN, R);
    for (const s of segments) {
      expect(s.dasharray).not.toMatch(/NaN/);
      expect(Number.isFinite(s.dashoffset)).toBe(true);
    }
  });

  it("gap 会把段略微缩短（视觉留白），但不会吃掉整段", () => {
    const tight = ringSegments(BUCKETS, 68, R, 0).segments;
    const loose = ringSegments(BUCKETS, 68, R, 3).segments;
    expect(dashLen(loose[0].dasharray)).toBeLessThan(dashLen(tight[0].dasharray));
    expect(dashLen(loose[0].dasharray)).toBeGreaterThan(0);
  });
});

describe("面板接入（源码契约）", () => {
  it("★ 圆环仅用于**功能栏小按钮**（面板已于 2026-10-03 改回横条）", () => {
    expect(panelSrc).toContain("<svg");
    // ⚠️ 面板的大圆环已移除；圆环只剩功能栏小按钮那一份
    expect(panelSrc).not.toContain("RING_R");
    expect(panelSrc).toContain("ringSegments(data.buckets, data.pct, RING_BTN_R");
  });

  it("★ 起点转到 12 点（rotate -90deg）—— 否则环从 3 点开始，读数方向不对", () => {
    expect(panelSrc).toContain("ctx-ring");
  });

  it("旧的横条分段条已移除（不留两套画法）", () => {
    expect(panelSrc).not.toContain("ctx-panel-stack");
    expect(panelSrc).not.toContain("ctx-panel-seg");
  });

  it("保留无障碍标签（环本身是图形，必须给出数值）", () => {
    expect(panelSrc).toContain("aria-label={`上下文已使用");
  });
});


describe("composer 的**小圆环**按钮（2026-10-03 用户：改成圆形、省空间、点击再展开）", () => {
  it("★ composer 用 variant=ring（不再占一整条文字宽度）", () => {
    expect(chatInputSrc).toContain('variant="ring"');
  });

  it("★ 面板与触发形态**解耦**：ring 与 bar 共用同一个面板", () => {
    // 若 ring 分支自己复制一份面板，两处会各自漂移
    expect(panelSrc.match(/className="ctx-panel"/g)?.length).toBe(1);
  });

  it("★ 小圆环用**自己的半径**算几何（与面板的 64 viewBox 不同）", () => {
    expect(panelSrc).toContain("RING_BTN_R");
    expect(panelSrc).toContain("ringSegments(data.buckets, data.pct, RING_BTN_R");
  });

  it("★ 圆环按钮必须给出**数值**（环本身没有文字，靠 title/aria 兜底）", () => {
    // ⚠️ 用子串断言而不是正则：正则里的 \s / { } 在多层转义下极易写坏（本次已踩）
    expect(panelSrc).toContain("title={`上下文已使用 ");
    expect(panelSrc).toContain("aria-label={`上下文已使用 ");
  });

  it("圆环按钮用的是专用 class（样式在 styles.css 里按它定位）", () => {
    // ⚠️ 不断言 styles.css：`?raw` 对 .css 不可靠（实测返回空串）⇒ 断言"用没用这个 class"即可
    expect(panelSrc).toContain("ctx-ring-btn");
  });
});


describe("★ 空态占位：无数据时**不得消失**（2026-10-03 用户报「功能栏不一样」的根因）", () => {
  it("★★ ring 模式在无数据时仍渲染占位（原实现 `return null` ⇒ 控件消失 ⇒ 布局跳动）", () => {
    // 必须存在"无数据 + ring"的专门分支
    expect(panelSrc).toContain("if (variant === \"ring\")");
    expect(panelSrc).toContain("ctx-ring-btn empty");
  });

  it("★ bar 模式（消息 footer）仍不占位 —— 那里本来就有其它 metadata 撑着", () => {
    // ⚠️ 只断言 `return null;` 存在，不写缩进/换行（多行锚点在转义下极易写坏）
    expect(panelSrc).toContain("return null;");
  });

  it("空态环给出说明文案（否则用户不知道为何是空的）", () => {
    expect(panelSrc).toContain("本轮暂无上下文数据");
  });

  it("空态按钮是 disabled（不可点开空面板）", () => {
    expect(panelSrc).toContain("className=\"ctx-ring-btn empty\"");
  });
});
