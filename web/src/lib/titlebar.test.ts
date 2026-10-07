import { afterEach, describe, expect, it } from "vitest";
import {
  closeWindow,
  getTauriInternals,
  isMaximized,
  isTauri,
  minimize,
  toggleMaximize,
} from "./titlebar";

// 在 node 环境下 `window` 未定义，这里手动安装一个最小 Tauri 核心 IPC 用于单测
function installTauri(calls: string[], opts: { isMaximized?: boolean } = {}) {
  (globalThis as unknown as { window: unknown }).window = {
    __TAURI_INTERNALS__: {
      invoke: (cmd: string) => {
        calls.push(cmd.replace("plugin:window|", ""));
        if (cmd === "plugin:window|is_maximized") {
          return Promise.resolve(opts.isMaximized ?? false);
        }
        return Promise.resolve();
      },
    },
  };
}

afterEach(() => {
  delete (globalThis as unknown as { window?: unknown }).window;
});

describe("titlebar / 环境探测", () => {
  it("非 Tauri 环境：isTauri 为 false，且取 internals 返回 null", () => {
    expect(isTauri()).toBe(false);
    expect(getTauriInternals()).toBeNull();
  });

  it("Tauri 环境：isTauri 为 true，能取到 internals", () => {
    const calls: string[] = [];
    installTauri(calls);
    expect(isTauri()).toBe(true);
    expect(getTauriInternals()).not.toBeNull();
  });
});

describe("titlebar / 窗口控制（直接 invoke plugin:window|*）", () => {
  it("minimize 调起 plugin:window|minimize", async () => {
    const calls: string[] = [];
    installTauri(calls);
    await minimize();
    expect(calls).toContain("minimize");
  });

  it("toggleMaximize 调起 plugin:window|toggle_maximize", async () => {
    const calls: string[] = [];
    installTauri(calls);
    await toggleMaximize();
    expect(calls).toContain("toggle_maximize");
  });

  it("closeWindow 调起 plugin:window|close", async () => {
    const calls: string[] = [];
    installTauri(calls);
    await closeWindow();
    expect(calls).toContain("close");
  });

  it("isMaximized 透传状态", async () => {
    const calls: string[] = [];
    installTauri(calls, { isMaximized: true });
    expect(await isMaximized()).toBe(true);
    installTauri(calls, { isMaximized: false });
    expect(await isMaximized()).toBe(false);
  });

  // ★ 反向自检（break 验证 red）：底层 reject 必须向上传播，不得被静默吞掉
  it("底层拒绝时错误向上传播（不静默吞没）", async () => {
    (globalThis as unknown as { window: unknown }).window = {
      __TAURI_INTERNALS__: {
        invoke: (cmd: string) => {
          if (cmd === "plugin:window|minimize") {
            return Promise.reject(new Error("blocked"));
          }
          return Promise.resolve();
        },
      },
    };
    await expect(minimize()).rejects.toThrow("blocked");
  });
});
