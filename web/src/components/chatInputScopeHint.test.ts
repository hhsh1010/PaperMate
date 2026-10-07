/**
 * 输入框「当前检索范围」提示的**组件级渲染守护**（2026-09-23，建议③）。
 *
 * ## 为什么纯函数测过 9 例了还要在组件层再测
 *
 * 本次修的不是逻辑，是**接线**：`docScopeHint(scope, mentionedIds)` 早就支持 `@` 聚焦，
 * 是组件**没传**第二个参数 —— 于是用户 `@` 某篇后后端确实聚焦了，
 * 提示条却仍说「检索全部 N 篇文档」→ **提示与真实行为相反**（比没有提示更糟：
 * 用户会基于错误提示下结论）。
 *
 * 这类「接线漏了」的缺陷，**纯函数测试与 `tsc` 都抓不到**：
 * 不传可选参数是**合法调用**，类型检查一路绿灯。
 * 只有真的把组件渲染出来、看它产出的文本，才能钉住。
 *
 * ## 为什么用 `react-dom/server` 而不是 testing-library
 *
 * 本仓 web 侧 vitest 跑 `environment: "node"`、include 只收 `.test.ts`，
 * 且未引入 testing-library。而 `react-dom` 是**运行时依赖**，
 * `renderToStaticMarkup` 在 node 下即可用 —— **零新增依赖**就能做真实渲染断言。
 * （因此本文件是 `.ts` 而非 `.tsx`：用 `createElement` 而非 JSX，以匹配 include 规则。）
 *
 * ⚠️ 本测试覆盖的是「组件拿到 `mentionedDocIds` 后是否正确展示」。
 * 「`ChatPage` 是否把 `@` 状态透传下来」属跨组件接线，node 环境下无法渲染验证，
 * 由 `ChatPage` 侧的三行（ref/state 同处更新 + 传参）保证。
 */

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ChatInput } from "./ChatInput";
import type { DocScopeOut } from "../lib/types";

/** 多文档会话（真实库形态：两篇遥感/检测论文） */
const SCOPE_MULTI: DocScopeOut = {
  session_id: 1,
  doc_count: 2,
  doc_ids: [15, 10],
  documents: [
    { id: 15, filename: "融合道路邻近关系的高分遥感目标分割方法.pdf" },
    { id: 10, filename: "DETRs Beat YOLOs.pdf" },
  ],
  force_retrieval: true,
  mode: "multi",
};

/** 未关联文档的会话 */
const SCOPE_EMPTY: DocScopeOut = {
  session_id: 2,
  doc_count: 0,
  doc_ids: [],
  documents: [],
  force_retrieval: false,
  mode: "general",
};

function renderHint(props: {
  scope?: DocScopeOut | null;
  mentioned?: number[];
}): string {
  return renderToStaticMarkup(
    createElement(ChatInput, {
      onSend: () => {},
      onStop: () => {},
      streaming: false,
      docScope: props.scope === undefined ? SCOPE_MULTI : props.scope,
      mentionedDocIds: props.mentioned,
    }),
  );
}

/** SSR 下 `useLayoutEffect` 会打一条无害告警，静音以保持门禁输出干净。
 *  ⚠️ 只静音这一条 —— 其余 `console.error` **原样转发**：
 *  全量静音会把真正的 React 报错也吞掉，那比噪音更糟。 */
let errSpy: { mockRestore: () => void };
let origError: typeof console.error;
beforeEach(() => {
  origError = console.error;
  errSpy = vi.spyOn(console, "error").mockImplementation((...args: unknown[]) => {
    if (String(args[0]).includes("useLayoutEffect")) return;
    (origError as (...a: unknown[]) => void)(...args);
  });
});
afterEach(() => errSpy.mockRestore());

describe("输入框 · 当前检索范围提示", () => {
  it("★ 本轮 @ 了某篇 → 提示必须改为「仅检索已指定的 1 篇」", () => {
    // 这条是本项的核心守护：只有 mentionedDocIds 被真正接线并消费，它才会通过。
    const html = renderHint({ mentioned: [15] });
    expect(html).toContain("仅检索已指定的 1 篇");
    // 反向：不得同时残留「检索全部」的表述（否则两条信息自相矛盾）
    expect(html).not.toContain("检索全部 2 篇");
  });

  it("未 @ → 提示为「检索全部 N 篇」", () => {
    const html = renderHint({ mentioned: [] });
    expect(html).toContain("检索全部 2 篇文档");
    expect(html).not.toContain("仅检索已指定的");
  });

  it("@ 了不在会话范围内的 id → 退回「检索全部」，不得显示「已指定的 0 篇」", () => {
    // 与后端 `_effective_doc_scope` 同口径：无效提及被忽略、退回默认范围，
    // 而不是静默变成空范围（那会让用户以为「什么都没检索」）。
    const html = renderHint({ mentioned: [999] });
    expect(html).toContain("检索全部 2 篇文档");
    expect(html).not.toContain("已指定的 0 篇");
  });

  it("未关联文档 → 明确说明「不会检索论文内容」", () => {
    // 这是消除「文档约束失效」误判的关键一句（用户最常见的困惑：
    // 以为自己在文档会话里提问，其实会话根本没绑文档）。
    const html = renderHint({ scope: SCOPE_EMPTY });
    expect(html).toContain("通用对话");
    expect(html).toContain("不会检索论文内容");
  });

  it("★ 范围提示带关闭按钮（2026-10-07 新增），且按钮在该横幅内部", () => {
    // 接线守护：关闭按钮复用 `.composer-warn-close`，必须渲染在 scopeHint 横幅**内部**。
    // ⚠️ 断言必须圈定在「本横幅段」内：warns 横幅也有同名 close 按钮且渲染在其后，
    // 若只用 indexOf(bannerIdx) 向后找，会**误匹配 warns 的按钮** ——
    // 这是本次反向自检（删按钮 → 测试仍绿）实测抓出的断言漏洞，故以下用段内匹配。
    const html = renderHint({ mentioned: [] });
    const bannerIdx = html.indexOf('data-tone="multi"');
    expect(bannerIdx).toBeGreaterThan(-1);
    // scopeHint div 的闭合：下一个 role="status"（warns 横幅开头）之前；没有 warns 则到末尾
    const nextBanner = html.indexOf('role="status"', bannerIdx);
    const segment = html.slice(bannerIdx, nextBanner === -1 ? undefined : nextBanner);
    expect(segment).toContain("composer-warn-close");
    expect(segment).toContain('aria-label="关闭此提示"');
  });

  it("通用对话（general tone）的提示同样带关闭按钮", () => {
    const html = renderHint({ scope: SCOPE_EMPTY });
    const bannerIdx = html.indexOf('data-tone="general"');
    expect(bannerIdx).toBeGreaterThan(-1);
    const nextBanner = html.indexOf('role="status"', bannerIdx);
    const segment = html.slice(bannerIdx, nextBanner === -1 ? undefined : nextBanner);
    expect(segment).toContain("composer-warn-close");
  });

  it("scope 未加载（null）→ 不渲染任何范围提示", () => {
    // 「宁可不显示，也不显示错的状态」：取数失败时不得瞎猜一个场景贴上去。
    const html = renderHint({ scope: null });
    expect(html).not.toContain("检索全部");
    expect(html).not.toContain("通用对话");
    expect(html).not.toContain("仅检索已指定");
  });
});
