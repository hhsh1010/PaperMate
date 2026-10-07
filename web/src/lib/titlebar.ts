/**
 * 桌面端（Tauri）窗口控制 —— 纯逻辑层。
 *
 * 设计要点：
 * - 不依赖 React / DOM，便于单测（见 titlebar.test.ts）。
 * - 直接走 Tauri 最底层的 `window.__TAURI_INTERNALS__.invoke('plugin:window|…')`，
 *   **不依赖** `withGlobalTauri` 注入的 `@tauri-apps/api` 命名空间（`window.__TAURI__.window`）。
 *   在「真实 UI 走远程源（http://127.0.0.1:8000）、首屏 loading 走本地源」的架构下，
 *   `window.__TAURI__` 命名空间的就绪时序更脆；`__TAURI_INTERNALS__` 是核心 IPC 对象，
 *   任何 webview 页（本地/远程）都会被注入，最稳。
 * - 所有函数对非 Tauri 环境（浏览器 / dev 模式）降级为 no-op；错误向上传播（不静默吞）。
 */

export type WindowAction =
  | "minimize"
  | "toggle_maximize"
  | "close"
  | "is_maximized"
  | "start_dragging";

interface TauriInternals {
  invoke<T = unknown>(cmd: string, args?: Record<string, unknown>): Promise<T>;
}

/** 取 Tauri 核心 IPC 对象；非 Tauri 环境返回 null。 */
export function getTauriInternals(): TauriInternals | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { __TAURI_INTERNALS__?: TauriInternals };
  return w.__TAURI_INTERNALS__ ?? null;
}

/** 当前是否运行在 Tauri 桌面壳内。 */
export function isTauri(): boolean {
  return getTauriInternals() != null;
}

/** 调底层窗口命令；非 Tauri 环境 no-op。 */
function call(cmd: WindowAction): Promise<unknown> {
  const internals = getTauriInternals();
  if (!internals) return Promise.resolve();
  return internals.invoke(`plugin:window|${cmd}`);
}

export function minimize(): Promise<void> {
  return call("minimize").then(() => undefined);
}

export function toggleMaximize(): Promise<void> {
  return call("toggle_maximize").then(() => undefined);
}

export function closeWindow(): Promise<void> {
  return call("close").then(() => undefined);
}

/** 取最大化态；非 Tauri 环境回落为 false。 */
export async function isMaximized(): Promise<boolean> {
  const internals = getTauriInternals();
  if (!internals) return false;
  try {
    return (await internals.invoke<boolean>("plugin:window|is_maximized")) ?? false;
  } catch {
    return false;
  }
}
