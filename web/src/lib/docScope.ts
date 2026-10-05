/**
 * 会话检索范围的**展示文案**（2026-09-23，场景可见性）。
 *
 * ## 为什么需要
 *
 * 系统对三种场景有不同行为（无文档 → 通用对话；单文档 → 必须基于它；多文档 → 默认检索全部，
 * `@` 指定则聚焦该篇），但**界面上完全看不出自己处于哪个场景**。
 *
 * 实测中用户报「文档约束没生效」时，相当一部分困惑来自这种不可见性 ——
 * 最典型的是：会话其实**根本没绑文档**，于是走了通用对话，而用户以为自己在文档会话里提问。
 * 一句话提示就能消除这类误判。
 *
 * ## 为什么独立于 `networkWarn.ts`
 *
 * 两者语义不同：`networkWarn` 产出的是**警告**（需要 dismiss、需要请求指纹校验）；
 * 这里产出的是**状态说明**（当前处于哪个场景）。分开也便于各自单测。
 *
 * ⚠️ 本函数**纯函数、无副作用**：输入的 `mentionedIds` 是本轮 `@` 的文档（每轮变化），
 * 而 `scope` 是会话级快照 —— 聚焦与否只有前端同时掌握这两者才能算准。
 */

import type { DocScopeOut } from "./types";

/** 场景类型：general = 通用对话；single = 单文档；multi = 多文档 */
export type ScopeTone = "general" | "single" | "multi";

export interface ScopeHint {
  /** React key（内容变化时自动重挂载，避免复用旧节点） */
  key: string;
  text: string;
  tone: ScopeTone;
}

/** 多文档时最多列出几个篇名，超出用「等 N 篇」 */
const MAX_LISTED = 3;

/** 单篇篇名在提示里的最大长度（过长截断，避免撑破一行） */
const MAX_NAME = 18;

function shortName(name: string): string {
  const base = (name || "").replace(/\.[a-z0-9]+$/i, ""); // 去扩展名（.pdf/.docx）
  return base.length > MAX_NAME ? `${base.slice(0, MAX_NAME)}…` : base;
}

/**
 * 产出输入框上方的「当前检索范围」提示。
 *
 * - `scope` 为 `null`（尚未加载 / 加载失败）→ 返回 `null`（**宁可不显示，也不显示错的状态**）
 * - 0 篇 → 明确说「通用对话」并点出「不会检索论文内容」——
 *   这是消除「约束失效」误判的关键一句
 * - 1 篇 → 直接说基于哪一篇
 * - ≥2 篇 → 默认说「检索全部 N 篇」并列出篇名；若本轮 `@` 了某几篇，则改说「仅检索已指定的 M 篇」
 *   （与后端 `_effective_doc_scope` 的新语义一致：`@` = 聚焦）
 */
export function docScopeHint(
  scope: DocScopeOut | null,
  mentionedIds?: number[] | null,
): ScopeHint | null {
  if (!scope) return null;

  const count = scope.doc_count ?? 0;

  if (count === 0) {
    return {
      key: "scope|general|0",
      tone: "general",
      text: "通用对话 · 当前会话未关联文档，不会检索论文内容",
    };
  }

  const names = (scope.documents ?? []).map((d) => shortName(d.filename));

  // 单文档：未被 @ 时也应说明「基于它」——`@` 与否不改变范围
  if (count === 1) {
    return {
      key: `scope|single|${scope.doc_ids.join(",")}`,
      tone: "single",
      text: names[0]
        ? `基于《${names[0]}》回答`
        : `基于已关联的 1 篇文档回答`,
    };
  }

  // 多文档：本轮 @ 了 → 聚焦提示（与后端 mention 限缩同语义）
  const mentioned = (mentionedIds ?? []).filter((id) => scope.doc_ids.includes(id));
  if (mentioned.length > 0) {
    const picked = scope.documents
      .filter((d) => mentioned.includes(d.id))
      .map((d) => shortName(d.filename));
    const listed = picked.slice(0, MAX_LISTED).join("、");
    return {
      key: `scope|focus|${mentioned.join(",")}`,
      tone: "multi",
      text: picked.length > MAX_LISTED
        ? `仅检索已指定的 ${mentioned.length} 篇：${listed} 等`
        : `仅检索已指定的 ${mentioned.length} 篇：${listed}`,
    };
  }

  const listed = names.slice(0, MAX_LISTED).join("、");
  return {
    key: `scope|multi|${scope.doc_ids.join(",")}`,
    tone: "multi",
    text: names.length > MAX_LISTED
      ? `检索全部 ${count} 篇文档：${listed} 等`
      : `检索全部 ${count} 篇文档：${listed}`,
  };
}
