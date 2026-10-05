/**
 * 命令面板修复的测试（2026-10-03）。
 *
 * 覆盖诊断报告（`docs/诊断-命令功能代码审查-2026-10-03.md`）中的前端两项：
 * - **P0-3**：点命令 chip 时命令被**追加到末尾** ⇒ 不在开头 ⇒ 后端不路由（点了没用）
 * - **P1/P2**：`commands[0]` 导致**英文别名 UI 不可见** + 24 命令无搜索
 */
import { describe, it, expect } from "vitest";
import { filterSkills } from "./skillFilter";
import chatInputSrc from "../components/ChatInput.tsx?raw";

// 与后端实测清单一致（`/settings/inventory` 的真实形状：每技能两个命令）
const SKILLS = [
  { name: "deep_read", description: "单篇精读", commands: ["/精读", "/deep_read"] },
  { name: "literature_review", description: "文献综述", commands: ["/综述", "/literature"] },
  { name: "glossary", description: "术语表", commands: ["/术语表", "/glossary"] },
  { name: "dev_eval", description: "开发者评测", commands: ["/eval", "/评测"] },
];

describe("filterSkills · 行为", () => {
  it("空查询 / 纯空白 ⇒ 原样返回（不过滤）", () => {
    expect(filterSkills(SKILLS, "")).toHaveLength(4);
    expect(filterSkills(SKILLS, "   ")).toHaveLength(4);
  });

  it("★ 搜**英文别名**能找到技能（这正是「别名不可见」的修法）", () => {
    // 原先前端只渲染 commands[0]，`/deep_read` 在 UI 里根本不存在 ⇒ 现在可搜到
    expect(filterSkills(SKILLS, "deep").map((s) => s.name)).toEqual(["deep_read"]);
    expect(filterSkills(SKILLS, "glossary").map((s) => s.name)).toEqual(["glossary"]);
  });

  it("搜中文主命令同样命中", () => {
    expect(filterSkills(SKILLS, "术语").map((s) => s.name)).toEqual(["glossary"]);
  });

  it("★ 大小写不敏感（与后端 casefold 口径一致，避免「搜得到但打不出」）", () => {
    expect(filterSkills(SKILLS, "DEEP")).toHaveLength(1);
    expect(filterSkills(SKILLS, "Deep_Read")).toHaveLength(1);
  });

  it("也能按技能名与描述搜", () => {
    expect(filterSkills(SKILLS, "literature_review")).toHaveLength(1);
    expect(filterSkills(SKILLS, "综述")).toHaveLength(1); // description 命中
  });

  it("无匹配 ⇒ 空数组（UI 据此渲染「没有匹配的命令」）", () => {
    expect(filterSkills(SKILLS, "zzzz")).toEqual([]);
  });

  it("null / undefined / 缺 commands 不得抛错", () => {
    expect(filterSkills(null, "x")).toEqual([]);
    expect(filterSkills(undefined, "x")).toEqual([]);
    expect(() => filterSkills([{ name: "n" }], "n")).not.toThrow();
    expect(filterSkills([{ name: "n" }], "n")).toHaveLength(1);
  });
});

describe("命令 chip 的点击行为（源码契约）", () => {
  it("★★ P0-3：命令必须**插到最前面**，不得追加到末尾", () => {
    // 后端 `SkillSpec.matches` 要求文本**以命令开头** ⇒ 追加到末尾 = 点了没用且无提示
    expect(chatInputSrc).toContain("setText((prev) => (prev.trim() ? `${cmd} ${prev.trim()}` : `${cmd} `))");
    // 反向守护：旧的"追加到末尾"写法必须已消失
    expect(chatInputSrc).not.toContain("`${prev.trimEnd()} ${cmd} `");
  });

  it("插入后光标落在命令之后（用户接着写参数）", () => {
    expect(chatInputSrc).toContain("const pos = cmd.length + 1;");
  });

  it("★ P1：展示**全部**命令，不再只取 commands[0] 当唯一命令", () => {
    expect(chatInputSrc).toContain("const cmds = sk.commands.length ? sk.commands : [sk.name];");
    expect(chatInputSrc).toContain("const alias = cmds.slice(1);");
    // 别名须出现在 tooltip 里（否则用户仍不知道有别名）
    expect(chatInputSrc).toContain("别名：");
  });

  it("过滤用的是**纯函数**（逻辑可行为测试，不内联在组件里）", () => {
    expect(chatInputSrc).toContain("filterSkills(skills, skillQuery)");
  });
});
