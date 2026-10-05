/** 会话/工作间**重命名**的源码契约（2026-10-02）。
 *
 * 项目无 testing-library ⇒ 用**源码契约**断言会被执行的那段。
 *
 * ## 功能要点（改动前完全缺失）
 * 侧栏「工作间」与「对话」两段都**没有**重命名入口（只有删除）。
 * 本次补上：点铅笔 → 行内编辑（不弹窗）→ Enter/✓ 提交、Esc/✗ 取消。
 *
 * ★ 两条关键正确性（与后端 `session_service.update` 的实现对应）：
 * 1. `renameSession` **只传 `title`**：后端 `update` 对 title/config 是
 *    `if ... is not None` 各自独立判断（:96-98），所以只传 title 安全；
 *    而传 config 会用**前端可能过期的副本** merge，覆盖服务端的工作间模式/主文献。
 * 2. 失败时**回滚 + 留在编辑态**（静默恢复原值 ⇒ 用户以为成功了）。
 */
import { describe, it, expect } from "vitest";
import appSrc from "../App.tsx?raw";
import apiSrc from "../api/client.ts?raw";

// ★★ **本文件测不到 CSS**（沿用 `citationCard.test.ts:12-15` 已记录的边界）：
//项目既没有 `@types/node`（`node:fs` 不可用），`import "*.css?raw"` 在 vitest 下
// 又返回**空串**（CSS 走 vite 样式管线，不是 `?raw` 文本管线）——
// 我为此踩过一次：`toContain(".session-rename:hover")` 对空串必然失败，
// 会让人误以为样式没写。⇒ 配色/样式类改为**在 `styles.css` 的注释里落约束**，
// 测试只守护**组件与 API 的行为契约**。
import iconsSrc from "./icons.tsx?raw";

describe("会话/工作间 · 重命名", () => {
  it("★ 复用 SessionRow（两段共用一个组件 ⇒ 改一处即两处生效）", () => {
    // 侧栏「工作间」与「对话」都渲染同一个 SessionRow
    const uses = appSrc.match(/<SessionRow/g) || [];
    expect(uses.length).toBe(2);
    // 且两处都传了 onRename（否则有一段没接上）
    expect(appSrc).toContain("onRename={(t) => onRename(s.id, t)}");
    expect(appSrc.split("onRename={(t) => onRename(s.id, t)}").length - 1).toBe(2);
  });

  it("★ 提交时只传 title（绝不传 config，避免覆盖工作间设置）", () => {
    expect(apiSrc).toContain("async renameSession(");
    expect(apiSrc).toContain("body: JSON.stringify({ title })");
    // renameSession 里**不得**出现 config
    const m = apiSrc.match(/async renameSession\([\s\S]*?\n  \},/);
    expect(m).not.toBeNull();
    expect(m?.[0]).not.toContain("config");
  });

  it("★ 走同一条 PATCH /sessions/{id}，不新开 /rename 端点", () => {
    // ⚠️ 断言范围**只限代码**（去掉注释行再查）——否则会命中本文件自己写的
    // 「刻意不新开 /rename 端点」那句 docstring，变成自指的假失败。
    const code = apiSrc.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
    expect(code).not.toContain("/rename");
    const m = apiSrc.match(/async renameSession\([\s\S]*?\n  \},/);
    expect(m?.[0]).toContain("method: \"PATCH\"");
    expect(m?.[0]).toContain("/sessions/${id}");
  });

  it("行内编辑：Enter 提交 / Esc 取消 / 空标题视为取消", () => {
    expect(appSrc).toContain("const [editing, setEditing] = useState(false)");
    expect(appSrc).toContain('e.key === "Enter"');
    expect(appSrc).toContain('e.key === "Escape"');
    // 空标题 ⇒ 取消（避免出现无名会话）
    expect(appSrc).toContain("if (!next) {");
  });

  it("★ 失败时留在编辑态（不静默恢复原值）", () => {
    expect(appSrc).toContain("await onRename(next)");
    expect(appSrc).toContain("} catch {");
    // 乐观更新的回滚
    expect(appSrc).toContain("if (prev) setSessions(");
  });

  it("编辑态与删除确认态相互独立（两个按钮同时可见不打架）", () => {
    // editing 是独立 state，不是复用 confirming
    expect(appSrc).toContain("const [editing, setEditing] = useState(false)");
    expect(appSrc).not.toContain("setConfirming(s.id); \n setEditing") // 不同步互触
  });

  it("★ 铅笔按钮与删除按钮用**不同的类**（配色才能分开）", () => {
    // 实际配色在 styles.css（本文件测不到，见顶部说明）。这里守的是**前提**：
    // 两者 class 不同 ⇒ 才有独立的 hover 规则可写；
    // 若合成同一个类，铅笔会继承 `.session-del:hover` 的 danger 红色。
    expect(appSrc).toContain("session-del session-rename");
    // 删除按钮不带 rename 类（否则又变成同一个类）
    expect(appSrc).toContain('className={`session-del ${confirming ? "confirming" : ""}`}');
  });

  it("移动端折叠态：新按钮沿用 .session-del 契约（自动隐藏/显示）", () => {
    // ★ 关键：新按钮 className 里含 `session-del` ⇒ 折叠态的两条既有规则
    //   （`display:none` / `sidebar-open` 下 `display:block`）**自动覆盖**它。
    //   若去掉 session-del 类，移动端抽屉展开后按钮将**永远不显示**（回归网）。
    expect(appSrc).toContain("session-del session-rename");
    // 编辑态下输入框也必须在折叠态被隐藏，否则抽屉未展开时会漏出一个输入框
    expect(appSrc).toContain("session-title-input");
  });

  it("★ 组件内不写内联颜色（配色集中 styles.css —— 项目铁律）", () => {
    // 内联 style 里若出现 color/background 字面量，配色就散到组件里了。
    // ★ 用"编辑态那一整段"作锚点（`if (editing) {` 到其后的 `return (`）。
    const start = appSrc.indexOf("if (editing) {");
    expect(start).toBeGreaterThan(-1);
    const seg = appSrc.slice(start, start + 1600);
    expect(seg).toContain("session-title-input");
    expect(seg).not.toMatch(/style=\{\{/);
  });

  it("图标已注册（Icons.pencil）", () => {
    expect(iconsSrc).toContain("pencil: icon(");
    expect(appSrc).toContain("<Icons.pencil size={14} />");
  });

  it("点击铅笔/输入框不得触发行选中（stopPropagation）", () => {
    // 行onClick 是 onSelect；编辑时不换会话才对
    expect(appSrc).toContain("onClick={(e) => e.stopPropagation()}");
  });
});
