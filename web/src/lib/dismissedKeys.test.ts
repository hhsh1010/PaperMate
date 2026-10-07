/**
 * `lib/dismissedKeys` 行为测试（2026-10-07）。
 *
 * 该模块被两类可关闭横幅共用（warns 安全提示 / scopeHint 状态说明），
 * 逻辑虽小但有两处**历史上真踩过的坑**必须有守护：
 * 1. 截断方向：只保留**最近** N 条 —— 写反成"保留最早 N 条"时，
 *    长会话里旧提示的关闭记忆先失效，提示"复活"且难以复现定位；
 * 2. node / 隐私模式下 `sessionStorage` 不存在 ⇒ 必须**静默降级**而不是抛错
 *    （vitest SSR 渲染 ChatInput 时会真实走到 load 路径）。
 */

import { describe, expect, it } from "vitest";

import {
  isKeyVisible,
  loadDismissedKeys,
  pushDismissedKey,
  saveDismissedKeys,
} from "./dismissedKeys";

describe("pushDismissedKey（纯函数）", () => {
  it("追加新 key 到末尾", () => {
    expect(pushDismissedKey(["a"], "b")).toEqual(["a", "b"]);
  });

  it("重复关闭同一条 → 去重且移到末尾（保序刷新）", () => {
    expect(pushDismissedKey(["a", "b", "c"], "a")).toEqual(["b", "c", "a"]);
  });

  it("★ 超出上限时淘汰的是**最早**的 key（保留最近 N 条）", () => {
    const keys = ["k1", "k2", "k3"];
    const next = pushDismissedKey(keys, "k4", 3);
    expect(next).toEqual(["k2", "k3", "k4"]); // k1 被淘汰，不是 k4
  });

  it("空列表也能追加", () => {
    expect(pushDismissedKey([], "x")).toEqual(["x"]);
  });

  it("不修改入参（纯函数）", () => {
    const keys = ["a", "b"];
    pushDismissedKey(keys, "c");
    expect(keys).toEqual(["a", "b"]);
  });
});

describe("isKeyVisible（纯函数）", () => {
  it("未关闭 → 可见", () => {
    expect(isKeyVisible(["a"], "b")).toBe(true);
  });
  it("已关闭 → 不可见", () => {
    expect(isKeyVisible(["a", "b"], "a")).toBe(false);
  });
});

describe("sessionStorage 存取（node 下静默降级）", () => {
  it("load：storage 不可用 → 返回空数组且不抛错", () => {
    expect(() => loadDismissedKeys("pm.test")).not.toThrow();
    expect(loadDismissedKeys("pm.test")).toEqual([]);
  });

  it("save：storage 不可用 → 不抛错（仅内存语义由调用方兜底）", () => {
    expect(() => saveDismissedKeys("pm.test", ["a"])).not.toThrow();
  });

  it("load：脏数据（非 JSON / 非字符串数组）→ 返回空数组", () => {
    // 直接测容错分支：node 无 sessionStorage 走 catch，这里验证的是
    // 「即便未来在浏览器里残留了坏数据，也不能让输入框挂掉」的防线是否还在。
    // 通过模拟一个最小 storage 桩注入脏数据。
    const orig = globalThis.sessionStorage;
    (globalThis as { sessionStorage?: Storage }).sessionStorage = {
      getItem: (k: string) => (k === "pm.dirty" ? "{not json" : null),
      setItem: () => undefined,
      removeItem: () => undefined,
      clear: () => undefined,
      key: () => null,
      get length() {
        return 0;
      },
    } as Storage;
    try {
      expect(loadDismissedKeys("pm.dirty")).toEqual([]);
      (globalThis as { sessionStorage?: Storage }).sessionStorage = {
        ...(undefined as unknown as Storage),
        getItem: () => '[1, "a", null]' as string, // 含非字符串元素 → 过滤后只剩 "a"
        setItem: () => undefined,
        removeItem: () => undefined,
        clear: () => undefined,
        key: () => null,
        get length() {
          return 0;
        },
      } as Storage;
      expect(loadDismissedKeys("pm.dirty")).toEqual(["a"]);
    } finally {
      (globalThis as { sessionStorage?: Storage }).sessionStorage = orig;
    }
  });
});
