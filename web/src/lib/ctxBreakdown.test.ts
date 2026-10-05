/**
 * 上下文用量分项（P2，2026-10-03）—— **行为**测试。
 *
 * ## ★ 两条口径必须分开（本文件一半用例在守这件事）
 * | 值 | 口径 | 用途 |
 * |---|---|---|
 * | `peak / budget` | **真实** | 大号百分比、"已使用 X/Y" |
 * | 各 bucket 之间 | **估算** | 分段条、分项百分比 |
 *
 * 若用 `breakdown` 之和当分母算总占用率，就会与"触发线用真实值"这一既有决策冲突
 * ⇒ 出现"面板说 30%、而实际上已超触发线"的矛盾。
 */
import { describe, it, expect } from "vitest";
import { BUCKET_META, buildCtxPanel, fmtTokens } from "./ctxBreakdown";
import msgViewSrc from "../components/ChatMessageView.tsx?raw";
import panelSrc from "../components/CtxUsagePanel.tsx?raw";

const BUDGET = 1_000_000;
const WARN_AT = 700_000;

// 与 WorkBuddy 截图同量级：380.4K / 1000K = 38.0%
const BD = {
  system: 22_000, // 2.2% of 1000K… 这里按**分项内部**比例算
  tool_defs: 35_000,
  messages: 317_000,
  notes: 10_000,
};

describe("fmtTokens", () => {
  it("千分位友好：K 后缀，整数不带小数", () => {
    expect(fmtTokens(1_000_000)).toBe("1000K");
    expect(fmtTokens(380_400)).toBe("380.4K");
    expect(fmtTokens(980)).toBe("980");
  });

  it("非法输入返回占位符，不产生 NaN", () => {
    expect(fmtTokens(Number.NaN)).toBe("—");
    expect(fmtTokens(-1)).toBe("—");
  });
});

describe("buildCtxPanel · 总占用率用【真实】口径", () => {
  it("★ peak / budget ⇒ 百分比（不用分项之和）", () => {
    // 分项之和 = 384K，若拿它当分子会得到 38.4%；真实峰值 380_400 ⇒ 38%
    const d = buildCtxPanel(380_400, BUDGET, WARN_AT, BD);
    expect(d?.pct).toBe(38);
    expect(d?.used).toBe(380_400);
    expect(d?.total).toBe(BUDGET);
  });

  it("★ 分项之和与真实峰值**不一致**时，百分比仍取真实值", () => {
    // 故意让分项之和远大于峰值：正确实现**不受影响**
    const d = buildCtxPanel(100_000, BUDGET, WARN_AT, { messages: 9_000_000 });
    expect(d?.pct).toBe(10);
  });

  it("超上限 ⇒ 截断到 100", () => {
    expect(buildCtxPanel(1_200_000, BUDGET, WARN_AT, BD)?.pct).toBe(100);
  });

  it("达警示线 ⇒ warn", () => {
    expect(buildCtxPanel(WARN_AT, BUDGET, WARN_AT, BD)?.warn).toBe(true);
    expect(buildCtxPanel(WARN_AT - 1, BUDGET, WARN_AT, BD)?.warn).toBe(false);
  });

  it("缺 warnAt ⇒ 退回 70% 兜底", () => {
    expect(buildCtxPanel(699_999, BUDGET, undefined, BD)?.warn).toBe(false);
    expect(buildCtxPanel(700_000, BUDGET, undefined, BD)?.warn).toBe(true);
  });
});

describe("buildCtxPanel · 分项", () => {
  it("★ 分项百分比 = 各分项**内部**相对比例（估算口径）", () => {
    const d = buildCtxPanel(380_400, BUDGET, WARN_AT, BD);
    const sum = Object.values(BD).reduce((a, b) => a + b, 0);
    const messages = d?.buckets.find((b) => b.key === "messages");
    expect(messages?.pct).toBeCloseTo(Math.round((317_000 / sum) * 1000) / 10, 1);
    // 分项之和应为 100%（允许 0.1 的四舍五入漂移）
    const total = (d?.buckets ?? []).reduce((a, b) => a + b.pct, 0);
    expect(total).toBeGreaterThan(99.5);
    expect(total).toBeLessThan(100.6);
  });

  it("★ 顺序**固定**（按 BUCKET_META），不按大小排 —— 位置稳定用户才能记住", () => {
    const d = buildCtxPanel(380_400, BUDGET, WARN_AT, BD);
    expect(d?.buckets.map((b) => b.key)).toEqual(
      BUCKET_META.map((m) => m.key).filter((k) => (BD as Record<string, number>)[k] > 0),
    );
  });

  it("过滤 0 值与未知键（不画无意义的段）", () => {
    const d = buildCtxPanel(1000, BUDGET, WARN_AT, { messages: 100, system: 0, 未知: 999 });
    expect(d?.buckets.map((b) => b.key)).toEqual(["messages"]);
  });

  it("每项都带 label 与颜色（颜色取自既有令牌，非 hex）", () => {
    const d = buildCtxPanel(1000, BUDGET, WARN_AT, BD);
    for (const b of d?.buckets ?? []) {
      expect(b.label).toBeTruthy();
      expect(b.color).toMatch(/^var\(--/);
      expect(b.color).not.toMatch(/#[0-9a-fA-F]{3,8}/);
    }
  });

  it("★ 分项缺失 ⇒ 仍返回数据（大号百分比是真实值，有展示价值）", () => {
    const d = buildCtxPanel(380_400, BUDGET, WARN_AT, {});
    expect(d).not.toBeNull();
    expect(d?.pct).toBe(38);
    expect(d?.buckets).toEqual([]);
  });
});

describe("buildCtxPanel · 优雅降级", () => {
  it("★ 缺 peak / budget ⇒ null（组件整体不渲染）", () => {
    expect(buildCtxPanel(undefined, BUDGET, WARN_AT, BD)).toBeNull();
    expect(buildCtxPanel(380_400, undefined, WARN_AT, BD)).toBeNull();
    expect(buildCtxPanel(null, null, null, null)).toBeNull();
  });

  it("非法数值 ⇒ null", () => {
    expect(buildCtxPanel(Number.NaN, BUDGET, WARN_AT, BD)).toBeNull();
    expect(buildCtxPanel(100, 0, WARN_AT, BD)).toBeNull();
    expect(buildCtxPanel(-5, BUDGET, WARN_AT, BD)).toBeNull();
  });

  it("分项含脏值（字符串/NaN/负数）⇒ 被忽略，不抛错", () => {
    const dirty = { messages: "x", system: Number.NaN, notes: -5, tool_defs: 100 } as unknown as Record<string, number>;
    expect(() => buildCtxPanel(1000, BUDGET, WARN_AT, dirty)).not.toThrow();
    expect(buildCtxPanel(1000, BUDGET, WARN_AT, dirty)?.buckets.map((b) => b.key)).toEqual(["tool_defs"]);
  });
});


describe("每条消息都带用量面板（2026-10-03，用户要求「为所有对话都加上该功能」）", () => {
  it("★ ChatMessageView 使用 CtxUsageBar（可点开看分项），并已移除旧的纯文本 CtxUsage", () => {
    expect(msgViewSrc).toContain("<CtxUsageBar");
    // 反向守护：旧的纯文本组件必须已删除（否则两套并存、口径可能漂移）
    expect(msgViewSrc).not.toContain("function CtxUsage(");
    expect(msgViewSrc).not.toContain("import { ctxUsage }");
  });

  it("★ 每条消息都透传 breakdown（否则点开只有空面板）", () => {
    expect(msgViewSrc).toContain("breakdown={m.done?.context_breakdown ?? m.ctx?.breakdown}");
  });

  it("★ 面板方向为 down —— 消息列表在中间，向上弹会出视野", () => {
    expect(msgViewSrc).toContain('placement="down"');
  });

  it("★ compact —— footer 行已有 tokens/成本/步数，不能挤出去", () => {
    expect(msgViewSrc).toContain("compact");
    // ⚠️ 断言 TSX 里的**模板拼接**，不是 CSS 选择器名 —— `compact` 是运行时才加的 class
    expect(panelSrc).toContain('${compact ? "compact" : ""}');
    // compact 时不再渲染「分子/分母/文案」那三段
    expect(panelSrc).toContain("{!compact && (");
  });

  it("两条来源合并（实时 done 优先、历史 ctx 兜底）", () => {
    for (const f of ["peak_prompt_tokens ?? m.ctx?.peak", "max_tokens_budget ?? m.ctx?.budget"]) {
      expect(msgViewSrc).toContain(f);
    }
  });

  it("★ 渲染条件覆盖历史消息（m.done || m.ctx），否则刷新后消失", () => {
    expect(msgViewSrc).toContain("m.status === \"complete\" && (m.done || m.ctx)");
  });
});


describe("★ 等比缩放：分项之和 = 面板大数字（2026-10-03，用户「做」）", () => {
  // 真实场景：估算偏低约 14%（实测 [估算比值] 均值）
  const EST = { system: 8_000, tool_defs: 1_000, messages: 25_000, notes: 500 };
  const EST_SUM = 34_500; // 估算总和
  const PEAK = 39_330; // ≈ EST_SUM × 1.14

  it("★★ 分项 tokens 之和 ≈ 真实峰值（不再是估算总和）", () => {
    const d = buildCtxPanel(PEAK, BUDGET, WARN_AT, EST)!;
    const sum = d.buckets.reduce((a, b) => a + b.tokens, 0);
    // 四舍五入有 ±(项数) 的漂移
    expect(Math.abs(sum - PEAK)).toBeLessThanOrEqual(d.buckets.length);
    expect(sum).not.toBe(EST_SUM);
  });

  it("★ 缩放**不改变**相对比例（谁大谁小仍由估算决定）", () => {
    const d = buildCtxPanel(PEAK, BUDGET, WARN_AT, EST)!;
    const messages = d.buckets.find((b) => b.key === "messages")!;
    expect(messages.pct).toBeCloseTo(Math.round((25_000 / EST_SUM) * 1000) / 10, 1);
  });

  it("缩放后每项 token **变大**（因为估算系统性偏低）", () => {
    const d = buildCtxPanel(PEAK, BUDGET, WARN_AT, EST)!;
    for (const b of d.buckets) {
      expect(b.tokens).toBeGreaterThan((EST as Record<string, number>)[b.key]);
    }
  });

  it("★ sum=0（无分项）⇒ 不除零、不产生 NaN", () => {
    const d = buildCtxPanel(PEAK, BUDGET, WARN_AT, {})!;
    expect(d.buckets).toEqual([]);
    expect(Number.isFinite(d.pct)).toBe(true);
  });

  it("★ 大数字仍取真实峰值 ÷ 预算（缩放**不得**反过来影响它）", () => {
    const d = buildCtxPanel(PEAK, BUDGET, WARN_AT, EST)!;
    expect(d.pct).toBe(Math.round((PEAK / BUDGET) * 100));
    expect(d.used).toBe(PEAK);
  });

  it("面板分项**显示 token 数**（用户才能核对「加起来等于总数」）", () => {
    expect(panelSrc).toContain("fmtTokens(b.tokens)");
  });

  it("★ footer 如实说明口径（不得写成「分项为真实值」）", () => {
    expect(panelSrc).toContain("厂商不返回分项明细");
    expect(panelSrc).not.toContain("分项占比为估算值，总占用率取自");
  });
});
