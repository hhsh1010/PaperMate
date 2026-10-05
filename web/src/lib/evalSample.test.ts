/**
 * 忠实度看板 · 样本护栏文案测试（2026-10-05）。
 *
 * ★ 核心不变量：**判据来自后端**，前端只措辞。
 * 这里钉死三件事（都是"看板会不会骗人"的关键）：
 *   1. 健康样本**不得**产生噪声提示（`[]`）；
 *   2. 过时样本必须**把年龄和阈值都写出来**（否则用户无法判断"多旧算旧"）；
 *   3. 缺 `max_age_hours` 时回落到 24，且 `age_hours` 为 null 时不得崩/不得编造数字。
 */
import { describe, it, expect } from "vitest";
import { formatAgeHours, sampleWarnings, type SampleWindow } from "./evalSample";

function win(over: Partial<SampleWindow> = {}): SampleWindow {
  return {
    limit: 10,
    sampled: 10,
    table_total: 284,
    excluded_empty: 55,
    from: "2026-10-03 14:28:14.728674",
    to: "2026-10-04 13:13:09.013662",
    ...over,
  };
}

describe("formatAgeHours", () => {
  it("缺失/非法值不得编造数字", () => {
    expect(formatAgeHours(null)).toBe("时间未知");
    expect(formatAgeHours(undefined)).toBe("时间未知");
    expect(formatAgeHours(Number.NaN)).toBe("时间未知");
  });

  it("分档措辞：<1h / 小时 / 天", () => {
    expect(formatAgeHours(0)).toBe("1 小时内");
    expect(formatAgeHours(0.5)).toBe("1 小时内");
    expect(formatAgeHours(19.5)).toBe("20 小时前");
    expect(formatAgeHours(23.9)).toBe("24 小时前");
    expect(formatAgeHours(50)).toBe("2 天前");
  });
});

describe("sampleWarnings", () => {
  it("窗口缺失 ⇒ 不提示（旧后端无此字段时不得报错）", () => {
    expect(sampleWarnings(null)).toEqual([]);
    expect(sampleWarnings(undefined)).toEqual([]);
  });

  it("健康样本 ⇒ 不产生噪声", () => {
    expect(sampleWarnings(win({ age_hours: 2, stale: false, insufficient: false }))).toEqual([]);
  });

  it("★ 过时 ⇒ 必须给出年龄与阈值（否则无法判断「多旧算旧」）", () => {
    const w = win({ age_hours: 19.5, max_age_hours: 24, stale: true });
    const out = sampleWarnings(w);
    expect(out).toHaveLength(1);
    expect(out[0].level).toBe("warn");
    expect(out[0].text).toContain("样本已过时");
    expect(out[0].text).toContain("20 小时前");
    expect(out[0].text).toContain("阈值 24 小时");
  });

  it("缺 max_age_hours ⇒ 回落 24", () => {
    const out = sampleWarnings(win({ age_hours: 30, stale: true }));
    expect(out[0].text).toContain("阈值 24 小时");
  });

  it("stale=true 但 age 未知 ⇒ 用「时间未知」，不编造数字", () => {
    const out = sampleWarnings(win({ age_hours: null, stale: true }));
    expect(out[0].text).toContain("时间未知");
  });

  it("窗口未填满 ⇒ 样本不足提示含实际条数与上限", () => {
    const out = sampleWarnings(win({ sampled: 6, insufficient: true }));
    expect(out).toHaveLength(1);
    expect(out[0].text).toContain("样本不足");
    expect(out[0].text).toContain("6/10");
    expect(out[0].text).toContain("表内非空共 284 条");
  });

  it("两个判据独立 ⇒ 可同时出现（顺序稳定：先过时、后不足）", () => {
    const out = sampleWarnings(win({ age_hours: 30, stale: true, sampled: 3, insufficient: true }));
    expect(out).toHaveLength(2);
    expect(out[0].text).toContain("样本已过时");
    expect(out[1].text).toContain("样本不足");
  });
});
