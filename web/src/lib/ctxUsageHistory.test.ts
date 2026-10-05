/**
 * 上下文占用率的**历史回填**测试（P1-a2，2026-10-03）。
 *
 * ## 修的缺陷
 * P1-a 的占用率只存在于**当轮** SSE 的 `done` 事件里；刷新页面 / 切换会话后，
 * 历史消息走 `MessageOut`（**没有 done 字段**）⇒ 表现为
 * 「**当轮看得到、刷新即消失**」——与 `history.ts` 里 B2-X1 记录过的引用卡片**同一类缺陷**。
 *
 * ## 修法
 * 后端落库时把 `ctx_peak / ctx_budget / ctx_warn_at` 写进既有 `meta_json`（零 DDL），
 * 本文件测的就是「前端能否把它正确回填成 `ctx`」。
 *
 * ★ 行为测试（`historyToThread` 是导出的纯函数，可直接调用）——
 * 不用 `?raw` 文本断言（那守不住逻辑，2026-10-02 已实测踩过）。
 */
import { describe, it, expect } from "vitest";
import { historyToThread } from "./history";
import type { MessageOut } from "./types";

const base = {
  id: 7,
  role: "assistant",
  content: "回答",
  created_at: "2026-10-03T10:00:00Z",
} as unknown as MessageOut;

describe("historyToThread · 占用率回填", () => {
  it("★ 有 ctx_peak + ctx_budget ⇒ 回填 ctx（刷新后仍能显示）", () => {
    const m = {
      ...base,
      meta: { ctx_peak: 12_537, ctx_budget: 30_000, ctx_warn_at: 21_000 },
    } as unknown as MessageOut;
    const item = historyToThread(m);
    expect(item).toMatchObject({
      ctx: { peak: 12_537, budget: 30_000, warnAt: 21_000 },
    });
  });

  it("★ 缺 ctx_budget ⇒ **不填** ctx（无分母算不出百分比，宁可不显示也不编造）", () => {
    const m = { ...base, meta: { ctx_peak: 12_537 } } as unknown as MessageOut;
    expect(historyToThread(m)).not.toHaveProperty("ctx");
  });

  it("缺 ctx_peak 同理不填", () => {
    const m = { ...base, meta: { ctx_budget: 30_000 } } as unknown as MessageOut;
    expect(historyToThread(m)).not.toHaveProperty("ctx");
  });

  it("★ 无 warnAt 时仍回填（warnAt 可选，前端退 70% 兜底）", () => {
    const m = { ...base, meta: { ctx_peak: 1000, ctx_budget: 30_000 } } as unknown as MessageOut;
    expect(historyToThread(m)).toMatchObject({ ctx: { peak: 1000, budget: 30_000 } });
    expect(historyToThread(m)).not.toHaveProperty("ctx.warnAt");
  });

  it("老消息（无 meta）⇒ 不填，且不得报错", () => {
    expect(historyToThread(base)).not.toHaveProperty("ctx");
  });

  it("脏数据（类型不对）⇒ 降级不填，不得抛错", () => {
    const dirty = [
      { ctx_peak: "12537", ctx_budget: 30_000 },
      { ctx_peak: null, ctx_budget: null },
      { ctx_peak: Number.NaN, ctx_budget: 30_000 },
    ];
    for (const meta of dirty) {
      const m = { ...base, meta } as unknown as MessageOut;
      expect(() => historyToThread(m)).not.toThrow();
    }
  });

  it("★ 回填 ctx **不得**顺带把消息标成 invalid（污染「连续无效」计数）", () => {
    const m = {
      ...base,
      meta: { ctx_peak: 100, ctx_budget: 30_000 },
    } as unknown as MessageOut;
    const item = historyToThread(m);
    expect(item).not.toHaveProperty("invalid");
    expect(item).not.toHaveProperty("notice");
  });

  it("★ 与其他 meta 共存（notice/invalid/model_risk 各归各位）", () => {
    const m = {
      ...base,
      meta: {
        ctx_peak: 100,
        ctx_budget: 30_000,
        model_risk: true,
        invalid: true,
        notice: { kind: "k", title: "t", message: "m", suggestion: "s" },
      },
    } as unknown as MessageOut;
    const item = historyToThread(m);
    expect(item).toMatchObject({ ctx: { peak: 100, budget: 30_000 } });
    expect(item).toMatchObject({ modelRisk: true, invalid: true });
    expect(item).toHaveProperty("notice");
  });
});
