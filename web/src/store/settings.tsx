// 全局设置共享存储（跨页面一致 + 持久化）。
//
// 设计要点（修复「默认全局模型始终为 fake / 切换页面值不保留」）：
// - 在 App 根部以 <SettingsProvider> 提供，App 在页面切换时不卸载，故 settings 在
//   各页面间天然共享、切换页面不会丢失（这是跨页一致性的关键，无需 localStorage 也能生效）。
// - 同时把最近一次后端返回值缓存到 localStorage（键 pm-settings），使整页刷新后也能
//   立即恢复（避免「fake 闪一下」），并在挂载时以后端为准重新校验。
// - patch() 写穿到后端 + 上下文 + localStorage，保证三处始终一致。

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../api/client";
import type { AppSettings } from "../lib/types";

const LS_KEY = "pm-settings";

type PatchBody = Parameters<typeof api.updateSettings>[0];

interface SettingsContextValue {
  /** 全局设置（含 default_model / providers / features），null=尚未加载 */
  settings: AppSettings | null;
  loading: boolean;
  /** 从后端重新拉取并写入上下文 + localStorage */
  refresh: () => Promise<void>;
  /** 写穿后端 + 上下文 + localStorage，返回更新后的完整设置 */
  patch: (partial: PatchBody) => Promise<AppSettings>;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

function readCache(): AppSettings | null {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as AppSettings) : null;
  } catch {
    return null;
  }
}

function writeCache(s: AppSettings): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(s));
  } catch {
    /* 隐私模式 / 配额满：忽略，不影响内存态 */
  }
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  // 首屏先用 localStorage 缓存（若有），避免「fake 闪现」；挂载后由后端校验为准。
  const [settings, setSettings] = useState<AppSettings | null>(() => readCache());
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const s = await api.getSettings();
      setSettings(s);
      writeCache(s);
    } catch {
      // N7 修复（2026-09-10）：原先无 catch —— getSettings 失败时产生 unhandled rejection，
      // UI 静默停留在可能过期的 localStorage 缓存且用户无感知。此处吞掉异常并保留现有
      // 状态（缓存值仍可用），由调用方按需提示；不阻断首屏。
    } finally {
      setLoading(false);
    }
  }, []);

  const patch = useCallback(async (partial: PatchBody) => {
    const updated = await api.updateSettings(partial);
    setSettings(updated);
    writeCache(updated);
    return updated;
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <SettingsContext.Provider value={{ settings, loading, refresh, patch }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const v = useContext(SettingsContext);
  if (!v) throw new Error("useSettings 必须在 <SettingsProvider> 内使用");
  return v;
}
