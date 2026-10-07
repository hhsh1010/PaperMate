/**
 * 可关闭横幅的「已关闭 key 列表」存取（2026-10-07 抽取共用）。
 *
 * ## 背景
 *
 * `ChatInput` 有两类可关闭横幅，关闭记忆逻辑完全同构：
 * - `warns`（联网能力不足 / 模型检索风险）：按指纹 key 记忆（2026-09-17）；
 * - `scopeHint`（当前检索范围提示）：按内容 key 记忆（2026-10-07 新增可关闭）。
 *
 * 原先 warns 的实现内联在组件里；scopeHint 要再加一份 ⇒ 抽成本模块，
 * 两处共用同一实现，避免「同一逻辑两份拷贝各自漂移」。
 *
 * ## 语义约定（与 warns 的指纹记忆一致）
 *
 * - key 是**内容指纹**：横幅内容变化（换会话/换范围/换文档集）⇒ key 变化
 *   ⇒ 关闭记忆自然失效 ⇒ 提示**自动重新出现**（不会一关就永久静音）；
 * - 存 sessionStorage：关掉标签页即重置，不跨会话纠缠；
 * - 只保留最近 20 条，避免无限累积；
 * - ⚠️ 所有存取都 必须 try/catch：node 环境（vitest SSR 渲染）与隐私模式下
 *   `sessionStorage` 不存在或写入失败 ⇒ **静默降级为仅内存生效**，绝不抛错。
 */

const MAX_KEYS = 20;

/** 读取已关闭 key 列表；不可用 / 格式异常 → 空数组（不抛错）。 */
export function loadDismissedKeys(storageKey: string): string[] {
  try {
    const raw = sessionStorage.getItem(storageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

/** 写入已关闭 key 列表；写入失败（隐私模式等）→ 仅内存生效，不抛错。 */
export function saveDismissedKeys(storageKey: string, keys: string[]): void {
  try {
    sessionStorage.setItem(storageKey, JSON.stringify(keys));
  } catch {
    // 静默降级：不影响功能
  }
}

/** 纯函数：把 key 追加进已关闭列表（去重 + 保序 + 截断到最近 MAX_KEYS 条）。 */
export function pushDismissedKey(keys: string[], key: string, max: number = MAX_KEYS): string[] {
  return [...keys.filter((k) => k !== key), key].slice(-max);
}

/** 纯函数：横幅当前是否可见（未被关闭）。 */
export function isKeyVisible(dismissed: string[], key: string): boolean {
  return !dismissed.includes(key);
}
