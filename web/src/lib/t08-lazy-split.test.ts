import { describe, expect, it } from "vitest";

import appSrc from "../App.tsx?raw";
import chatPageSrc from "../pages/ChatPage.tsx?raw";
import viteConfigSrc from "../../vite.config.ts?raw";

/**
 * T-08（2026-09-29）：**首屏分包**的守护测试。
 *
 * ## 为什么必须有它
 * 这个优化的成果**只体现在构建产物**里（`dist`），而 `dist` 不进版本库、平时也不看。
 * 一旦有人把 `lazy(...)` "顺手整理"回静态 `import`，**功能完全正常、测试全绿**，
 * 唯一的变化是首屏体积从 244 kB 悄悄涨回去（曾经 1011 kB）。
 * ⇒ 只有在**源码层**钉住边界，反弹才会报警。项目无 testing-library ⇒ 用**源码契约**。
 *
 * ⚠️ 断言必须匹配"**会被执行的语句**"：用带 `^\\s*import` 锚定的正则，
 * 避免匹配到注释里引用的旧写法（源码契约的老教训：断言"标识符出现过"挡不住被注释/短路）。
 */
describe("T-08 首屏分包（源码契约）", () => {
  it("App：默认落地页 ChatPage 保持静态导入（有意 eager，改了要同步本用例的理由）", () => {
    expect(appSrc).toMatch(/^\s*import\s+\{\s*ChatPage\s*\}\s+from\s+"\.\/pages\/ChatPage"/m);
  });

  it("App：6 个非默认视图全部走 lazy 动态导入", () => {
    for (const spec of [
      "./pages/DocumentsPage",
      "./components/EvalPanel",
      "./pages/SettingsPage",
      "./pages/LoginPage",
      "./pages/HomePage",
      "./components/UsagePanel",
    ]) {
      expect(appSrc).toContain(`import("${spec}")`);
    }
  });

  it("App：6 个非默认视图**不得**再有静态 import（否则首屏图里又全进来了）", () => {
    for (const spec of [
      "./pages/DocumentsPage",
      "./components/EvalPanel",
      "./pages/SettingsPage",
      "./pages/LoginPage",
      "./pages/HomePage",
      "./components/UsagePanel",
    ]) {
      const staticImport = new RegExp(
        `^\\s*import\\s+\\{[^}]*\\}\\s+from\\s+"${spec.replace(/[./]/g, "\\$&")}"`, "m");
      expect(appSrc).not.toMatch(staticImport);
    }
  });

  it("App：懒加载有 Suspense 边界（否则 React 会直接抛错）", () => {
    expect(appSrc).toContain("<Suspense fallback={<ViewLoading />}>");
  });

  it("ChatPage：`docLabel` 必须从 lib 直引 —— 从面板 re-export 引入会连带拉进 pdfjs", () => {
    expect(chatPageSrc).toMatch(/^\s*import\s+\{\s*docLabel\s*\}\s+from\s+"\.\.\/lib\/docLabel"/m);
    // ★ 回归点：这一行（静态 import 面板）会让"面板下沉为 lazy"彻底失效
    expect(chatPageSrc).not.toMatch(
      /^\s*import\s+\{[^}]*\}\s+from\s+"\.\.\/components\/DocumentPreviewPanel"/m);
  });

  it("ChatPage：pdfjs / docx-preview 的两个入口组件都是 lazy", () => {
    expect(chatPageSrc).toContain('import("../components/DocumentPreviewPanel")');
    expect(chatPageSrc).toContain('import("../components/RevisionWorkbench")');
    // 两个 lazy 组件都必须在 Suspense 里渲染
    expect(chatPageSrc).toContain("<Suspense fallback={<PanelLoading />}>");
  });

  it("vite：pdfjs 手工成 chunk；⚠️ **不得**手工拆 docx-preview（会诱发入口静态引用）", () => {
    expect(viteConfigSrc).toContain("manualChunks");
    expect(viteConfigSrc).toContain('return "pdfjs"');
    // ★ 实测教训（2026-09-29）：手工拆 docx-preview 会让 Rollup 为保求值顺序在**入口**
    //   注入 `import{g as Ef}`（该导出从未被使用）⇒ index.html 照旧 preload 174 kB。
    expect(viteConfigSrc).not.toContain('return "docx-preview"');
  });
});
