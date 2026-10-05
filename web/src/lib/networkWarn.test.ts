import { describe, it, expect } from "vitest";
import { capKeyOf, composerWarns } from "./networkWarn";
import type { NetworkCapabilityOut } from "./types";

/** 输入框上方预警横幅的判定（2026-09-17）。
 *
 * 核心回归对象是**闪烁**：`webOn` 乐观更新 vs `webCapability` 陈旧响应，
 * 点开关那一刻两者必然短暂矛盾 → 提示闪现约 50ms 后消失（浏览器帧级实测）。
 * 纯函数 + 请求指纹就是为此加的闸门：**要闪必须先过这里**。
 */

const cap = (over: Partial<NetworkCapabilityOut> = {}): NetworkCapabilityOut => ({
  session_id: 18,
  enabled: true,
  available: true,
  supports_native: true,
  effective_model: "qwen3-max",
  reason: "",
  ...over,
});

/** 组装一次判定所需的全部输入（默认：指纹匹配、开关开、模型支持联网、无风险） */
const decide = (over: Partial<Parameters<typeof composerWarns>[0]> = {}) => {
  const { capKey: overKey, ...rest } = over;
  const merged = {
    sessionId: 18,
    webOn: true,
    docsLen: 3,
    model: "qwen3-vl-flash",
    cap: cap({ supports_native: false, reason: "当前模型 qwen3-vl-flash 不支持原生联网 → 联网开关不生效" }),
    dismissed: [] as string[],
    ...rest,
  };
  const capKey =
    overKey !== undefined
      ? overKey
      : capKeyOf({ sessionId: merged.sessionId, webOn: merged.webOn, docsLen: merged.docsLen, model: merged.model });
  return composerWarns({ ...merged, capKey });
};

describe("capKeyOf", () => {
  it("同输入同指纹；任一维度变化都要变（否则陈旧数据会被误用）", () => {
    const base = { sessionId: 18, webOn: true, docsLen: 3, model: "qwen3-max" };
    expect(capKeyOf(base)).toBe(capKeyOf({ ...base }));
    expect(capKeyOf(base)).not.toBe(capKeyOf({ ...base, webOn: false }));
    expect(capKeyOf(base)).not.toBe(capKeyOf({ ...base, sessionId: 19 }));
    expect(capKeyOf(base)).not.toBe(capKeyOf({ ...base, docsLen: 4 }));
    expect(capKeyOf(base)).not.toBe(capKeyOf({ ...base, model: "qwen3.7-flash" }));
  });
});

describe("闪烁回归：陈旧能力数据不得用于预警", () => {
  it("点开联网开关的瞬间（数据还是「关闭状态」下请求的）→ 不显示任何提示", () => {
    // 复刻实测场景：用户点了「开启」→ webOn=true（乐观更新），
    // 但手上的数据是上一次「关闭」时请求的（后端 available = enabled and supports = false）。
    // 旧实现 `webOn && !available` 在此成立 → 闪现错误文案；新实现被指纹挡下。
    const stale = capKeyOf({ sessionId: 18, webOn: false, docsLen: 3, model: "qwen3-vl-flash" });

    const warns = decide({ webOn: true, cap: cap({ enabled: false, available: false, supports_native: true }), capKey: stale });

    expect(warns).toEqual([]);
  });

  it("切模型后旧模型的能力数据不显示（model 属于指纹的一部分）", () => {
    const oldKey = capKeyOf({ sessionId: 18, webOn: true, docsLen: 3, model: "qwen3.7-flash" });

    const warns = decide({
      model: "qwen3-vl-flash",
      cap: cap({ available: true, supports_native: true }),
      capKey: oldKey,
    });

    expect(warns).toEqual([]);
  });

  it("切换会话时旧会话的数据不显示", () => {
    const otherKey = capKeyOf({ sessionId: 17, webOn: true, docsLen: 3, model: "qwen3-vl-flash" });
    expect(decide({ sessionId: 18, capKey: otherKey })).toEqual([]);
  });

  it("还没有数据（cap=null）→ 不显示，且不得抛错", () => {
    expect(decide({ cap: null, capKey: null })).toEqual([]);
  });

  it("拉取抢在 PATCH 落库之前（后端回显 enabled 与前端不一致）→ 不显示", () => {
    // 第二道闸门（2026-09-17 实测补）：点开关是乐观更新，而持久化是异步 PATCH ——
    // 能力请求完全可能抢在落库之前到达后端，后端便按**旧**开关状态回答
    // （enabled=false、reason=""），前端拿它配 supports_native=false 就会显示
    // 一句归因错误的兜底文案。实测在浏览器里复现过，故用测试钉住。
    const racesAhead = cap({ enabled: false, available: false, supports_native: false, reason: "" });

    expect(decide({ webOn: true, cap: racesAhead })).toEqual([]);
    // 对照：开关确实关着时同样不报警（此处 enabled 与 webOn 一致，由 `webOn &&` 挡住）
    expect(decide({ webOn: false, cap: racesAhead })).toEqual([]);
  });

  it("后端回显一致时正常显示（证明上面那道闸门没把功能修坏）", () => {
    const warns = decide({
      webOn: true,
      cap: cap({ enabled: true, supports_native: false, reason: "当前模型 x 不支持原生联网 → 联网开关不生效" }),
    });

    expect(warns).toHaveLength(1);
    expect(warns[0].text).toContain("不支持原生联网");
  });

  it("enabled 字段缺失（更旧的后端）→ 保守不显示", () => {
    const legacy = { ...cap({ supports_native: false }), enabled: undefined } as unknown as NetworkCapabilityOut;
    expect(decide({ cap: legacy })).toEqual([]);
  });
});

describe("联网能力预警的正常显示", () => {
  it("开关已开 + 模型不支持 → 显示，且用后端的 reason 原文", () => {
    const warns = decide();

    expect(warns).toHaveLength(1);
    expect(warns[0].kind).toBe("web");
    expect(warns[0].text).toContain("qwen3-vl-flash");
  });

  it("reason 为空时兜底文案**不得错误归因**为「模型不支持」", () => {
    const warns = decide({ cap: cap({ supports_native: false, reason: "" }) });

    expect(warns).toHaveLength(1);
    expect(warns[0].text).toBe("当前模型不具备原生联网能力，联网开关不会生效");
  });

  it("开关关着（available=false 但能力具备）→ 不显示", () => {
    // `available` 是「当前是否可用」，开关关着时它也是 false；
    // 若用 `!available` 当判据就会把"开关关着"误读成"模型不支持"。
    const warns = decide({ webOn: false, cap: cap({ enabled: false, available: false, supports_native: true }) });

    expect(warns).toEqual([]);
  });

  it("开关已开 + 模型支持 → 不显示", () => {
    expect(decide({ cap: cap({ available: true, supports_native: true }) })).toEqual([]);
  });

  it("supports_native 缺失（旧后端）→ 保守不显示，不误报", () => {
    const legacy = { ...cap(), supports_native: undefined } as unknown as NetworkCapabilityOut;
    expect(decide({ cap: legacy })).toEqual([]);
  });
});

describe("模型检索风险提示", () => {
  const risky = cap({ available: true, supports_native: true, model_risk: true, model_risk_reason: "可能跳过检索，请核对引用" });

  it("显示并可关闭；关闭后不再显示（按会话+文案指纹记忆）", () => {
    const warns = decide({ cap: risky });
    expect(warns.map((w) => w.kind)).toEqual(["risk"]);

    const key = warns[0].key;
    expect(decide({ cap: risky, dismissed: [key] })).toEqual([]);
  });

  it("换会话 → 指纹变化 → 提示重新出现（关闭不该被跨会话继承）", () => {
    const first = decide({ cap: risky })[0];
    expect(decide({ sessionId: 19, cap: risky, dismissed: [first.key] })).toHaveLength(1);
  });

  it("两条同时成立时都显示（互不遮蔽）", () => {
    const warns = decide({ cap: cap({ supports_native: false, reason: "不支持", model_risk: true, model_risk_reason: "有风险" }) });
    expect(warns.map((w) => w.kind)).toEqual(["web", "risk"]);
  });
});
