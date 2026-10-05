/**
 * 上下文占用率 **行为**测试（P1-a，2026-10-03）。
 *
 * ★ 为什么是行为测试：本项目前端测试以 `?raw` 文本契约为传统，但**文本断言守不住逻辑**
 * ——实测（2026-10-02）注入"把逻辑换掉"后文本断言全绿假象。
 * 占用率含**算术与分支** ⇒ 必须直接调用纯函数验证。
 *
 * ★ 用**实测数字**做基准（trace #374）：峰值 12,537 / 上限 30,000 ⇒ 41.8% → 显示 42%。
 */
import { describe, it, expect } from "vitest";
import { ctxUsage, fmtK } from "./ctxUsage";

// 后端真实配置（`LoopConfig` 默认档）
const BUDGET = 30_000;
const WARN_AT = 21_000; // = 触发线 = 上限 × 0.7

describe("fmtK", () => {
  it("千分位友好：整数 k 不带小数，非整数保留一位", () => {
    expect(fmtK(30_000)).toBe("30k");
    expect(fmtK(12_537)).toBe("12.5k");
    expect(fmtK(980)).toBe("980");
    expect(fmtK(0)).toBe("0");
  });

  it("非法输入返回占位符而不是抛错/NaN", () => {
    expect(fmtK(Number.NaN)).toBe("—");
    expect(fmtK(-1)).toBe("—");
    expect(fmtK(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

describe("ctxUsage · 正常计算", () => {
  it("★ 用实测数字：12,537 / 30,000 ⇒ 42%", () => {
    const r = ctxUsage(12_537, BUDGET, WARN_AT);
    expect(r).not.toBeNull();
    expect(r?.pct).toBe(42);
    expect(r?.label).toBe("12.5k / 30k");
  });

  it("未达触发线 ⇒ 不警示", () => {
    expect(ctxUsage(12_537, BUDGET, WARN_AT)?.warn).toBe(false);
    expect(ctxUsage(WARN_AT - 1, BUDGET, WARN_AT)?.warn).toBe(false);
  });

  it("★ 达到触发线 ⇒ 警示（边界含等号）", () => {
    expect(ctxUsage(WARN_AT, BUDGET, WARN_AT)?.warn).toBe(true);
    expect(ctxUsage(WARN_AT + 1, BUDGET, WARN_AT)?.warn).toBe(true);
  });

  it("★ 熔断场景（超上限）⇒ pct 截断 100 且警示", () => {
    // 真实事故：session 18 曾 30630 > 30000
    const r = ctxUsage(30_630, BUDGET, WARN_AT);
    expect(r?.pct).toBe(100); // 不得 >100（进度条会溢出）
    expect(r?.warn).toBe(true);
  });
});

describe("ctxUsage · 优雅降级（老后端不得报错）", () => {
  it("★ 缺任一必需字段 ⇒ null（组件据此整块不渲染）", () => {
    expect(ctxUsage(undefined, BUDGET, WARN_AT)).toBeNull();
    expect(ctxUsage(12_537, undefined, WARN_AT)).toBeNull();
    expect(ctxUsage(null, null, null)).toBeNull();
  });

  it("非法数值 ⇒ null（不产生 NaN 百分比）", () => {
    expect(ctxUsage(Number.NaN, BUDGET, WARN_AT)).toBeNull();
    expect(ctxUsage(-1, BUDGET, WARN_AT)).toBeNull();
    expect(ctxUsage(100, 0, WARN_AT)).toBeNull(); // 分母 0 ⇒ 不可计算
    expect(ctxUsage(100, -5, WARN_AT)).toBeNull();
  });

  it("★ 缺 warnAt（老后端）⇒ 退回 70% 兜底，但仍能渲染", () => {
    // 兜底阈值 = budget × 0.7 = 21000
    expect(ctxUsage(20_999, BUDGET, undefined)?.warn).toBe(false);
    expect(ctxUsage(21_000, BUDGET, undefined)?.warn).toBe(true);
    expect(ctxUsage(12_537, BUDGET, null)?.warn).toBe(false);
  });

  it("warnAt 非法（0/NaN）⇒ 同样退回兜底，不误判为'永远警示'", () => {
    // ⚠️ 若把非法 warnAt 当 0 用，`peak >= 0` 恒真 ⇒ 永远警示（假报警）
    expect(ctxUsage(1000, BUDGET, 0)?.warn).toBe(false);
    expect(ctxUsage(1000, BUDGET, Number.NaN)?.warn).toBe(false);
  });
});

describe("ctxUsage · 口径正确性", () => {
  it("★ 分母是**单次上限**，不是整轮累计上限", () => {
    // 后端还有 max_tokens_per_turn=150000（整轮）。若拿它当分母会严重低估占用率。
    const r = ctxUsage(27_664, BUDGET, WARN_AT); // 27,664 = 实测整轮累计；这里当作峰值
    expect(r?.pct).toBe(92); // 27.664/30 ≈ 92%（若误用 150k 分母则只有 18%）
    expect(r?.warn).toBe(true);
  });

  it("峰值取自**单次**而非累计 ⇒ 二者不可混用（防止有人传错字段）", () => {
    // 实测同轮：峰值 12,537、累计 27,664 ⇒ 占用率必须用前者（42% 而非 92%）
    expect(ctxUsage(12_537, BUDGET, WARN_AT)?.pct).toBe(42);
  });
});
