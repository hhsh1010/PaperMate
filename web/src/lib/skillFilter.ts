/**
 * 命令面板的**本地过滤**（P2，2026-10-03）。
 *
 * ## 为什么需要
 * 12 个技能共 **24 个命令**平铺，用户只能滚动找；且前端原本只渲染 `commands[0]`,
 * ⇒ **英文别名在 UI 里完全不存在**（诊断报告 §2.3）。
 * 过滤**匹配全部别名** ⇒ 搜 "deep" 就能发现 `/deep_read`，等于把别名变得可发现。
 *
 * ★ 抽成**纯函数**（而不是内联在组件里）——本项目纪律：
 * 有分支的逻辑必须能被**行为测试**直接调用（`?raw` 文本断言守不住逻辑，
 * 2026-10-02 已实测踩过：注入后文本断言全绿假象）。
 */

/** 过滤所需的最小形状（避免与 `SkillSpec` 强耦合）。 */
export interface FilterableSkill {
  name: string;
  description?: string;
  commands?: string[];
}

/**
 * 按关键词过滤技能。
 *
 * - 空/纯空白关键词 ⇒ **原样返回**（不做任何过滤，也不复制数组内容）
 * - 匹配范围：**全部命令**（含别名）> 技能名 > 描述
 * - `toLowerCase()` 归一：用户可能搜 `Deep_Read` / `DEEP`
 *
 * ⚠️ 与后端 `SkillSpec.matches()` 的**大小写口径保持一致**（后端已用 `casefold()`
 * 做大小写不敏感）—— 前端搜索若区分大小写，会出现"搜得到但打不出"的割裂。
 */
export function filterSkills<T extends FilterableSkill>(
  skills: T[] | null | undefined,
  query: string,
): T[] {
  const list = skills ?? [];
  const q = (query ?? "").trim().toLowerCase();
  if (!q) return list;
  return list.filter((sk) => {
    const cmds = sk.commands ?? [];
    if (cmds.some((c) => c.toLowerCase().includes(q))) return true;
    if ((sk.name ?? "").toLowerCase().includes(q)) return true;
    return (sk.description ?? "").toLowerCase().includes(q);
  });
}
