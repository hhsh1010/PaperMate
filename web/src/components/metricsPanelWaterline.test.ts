/** 上下文水位在`MetricsPanel` 的展示契约（2026-10-02）。
 *
 * 项目无 testing-library ⇒ 用**源码契约**断言会被执行的那段。
 *
 * ★ 守卫的重点是**"缺样本不得显示成0%"**：
 * `ctx_peak_p50` 的类型是 `number | null`，`null` = 尚未采到样本（老轨迹没埋该字段）。
 * 若渲染时把 `null` 当 0，面板会显示"上下文水位 0.0%" ⇒ 用户以为"很安全"，
 * 那是**谎报安全**（与后端"缺样本报 null 不报 0"是同一条原则的两端）。
 */
import { describe, it, expect } from "vitest";
import panelSrc from "./MetricsPanel.tsx?raw";

describe("MetricsPanel · 上下文水位", () => {
  it("有样本时显示 单次中位 / 额度 / 触发线 / 余量", () => {
    expect(panelSrc).toContain("ctx_peak_p50");
    expect(panelSrc).toContain("ctx_budget");
    expect(panelSrc).toContain("ctx_trigger");
    expect(panelSrc).toContain("ctx_headroom_median");
  });

  it("★ 无样本（null）时显示「—」而不是 0%", () => {
    // 判据：必须显式判 `!== null`，而不是让 null 落进数字格式化
    expect(panelSrc).toContain("m.ctx_peak_p50 !== null");
    expect(panelSrc).not.toContain("pct(m.ctx_peak_p50 / m.ctx_budget) ||");
  });

  it("★ 越线与「被动过刀」必须可见（唯一能看到证据被裁的信号）", () => {
    expect(panelSrc).toContain("ctx_over_budget");
    expect(panelSrc).toContain("ctx_over_trigger");
    expect(panelSrc).toContain("ctx_elided_traces");
    // 三者全为 0 时不渲染这一块（避免"0 项越线"这种噪音行）
    expect(panelSrc).toContain("m.ctx_over_trigger > 0 || m.ctx_over_budget > 0 || m.ctx_elided_traces > 0");
  });

  it("文案点明「judge 测不出完整性损失」——避免用户误以为评测能兜住", () => {
    expect(panelSrc).toContain("judge");
  });
});
