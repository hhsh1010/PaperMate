import { describe, expect, it } from "vitest";

import componentSrc from "./ChatMessageView.tsx?raw";
import { visibleToolCalls } from "./ChatMessageView";
import type { ToolCallView } from "../lib/types";

function mk(name: string, id = name): ToolCallView {
  return { id, name, args: {}, state: "ok" };
}

/** 2026-09-28 用户要求：`retrieve` 属内部检索手段，不在「推理过程」里展示。 */
describe("visibleToolCalls（隐藏检索卡片）", () => {
  it("过滤掉 retrieve，保留其它工具", () => {
    const out = visibleToolCalls([mk("retrieve", "r1"), mk("pdf_extract"), mk("retrieve", "r2")]);
    expect(out.map((t) => t.name)).toEqual(["pdf_extract"]);
  });

  it("全是 retrieve → 返回空数组（面板标题随之走『推理过程（N）』分支）", () => {
    expect(visibleToolCalls([mk("retrieve", "r1"), mk("retrieve", "r2")])).toHaveLength(0);
  });

  it("不做模糊匹配：名字里含 retrieve 的其它工具（如 retrieve_v2）应保留", () => {
    // 若哪天有人把过滤写成 includes()，这条会红 —— 避免误伤同名前缀的工具
    expect(visibleToolCalls([mk("retrieve_v2")]).map((t) => t.name)).toEqual(["retrieve_v2"]);
  });

  it("纯函数：不修改入参", () => {
    const input = [mk("retrieve", "r1"), mk("pdf_extract")];
    visibleToolCalls(input);
    expect(input).toHaveLength(2);
  });
});

/** 「推理过程」默认展开 —— 此前折叠着，必须点开才看得见，等于"有展示但看不见"。
 *
 * 项目未装 testing-library（运行依赖刻意压到 4 个）⇒ 无法渲染组件做交互断言，
 * 这里退一步用**源码契约**断言：它足够拦住"有人把默认值改回折叠"这类回归。
 */
describe("推理过程默认展开（源码契约）", () => {
  it("折叠状态初始值为 true", () => {
    expect(componentSrc).toContain("useState(true)");
  });

  it("details 受控于该状态（用户仍可手动收起）", () => {
    expect(componentSrc).toContain("open={thinkOpen}");
    expect(componentSrc).toContain("onToggle=");
  });

  it("面板计数与展示同源（都用 visibleTools，不用 m.toolCalls.length）", () => {
    // 计数若仍取原始长度，会出现「3 个工具」却零张卡片
    expect(componentSrc).not.toContain("${m.toolCalls.length} 个工具");
  });
});

/** O-12 档 1：进度提示「正在检索关联文档…」的渲染契约。
 *
 * 项目无 testing-library ⇒ 用源码契约断言**会被执行的那段**：
 * 显示条件必须**同时**要求「仍在流式」与「尚无正文」——
 * 少了 `!m.text` 就会出现"正在检索"与已出现的正文并列的自相矛盾。
 */
describe("O-12 档 1 · 进度提示渲染（源码契约）", () => {
  it("渲染条件同时包含 streaming 与 !m.text", () => {
    expect(componentSrc).toContain('m.status === "streaming" && !m.text && m.progress?.text');
  });

  it("使用独立类名 msg-progress（不复用警示卡样式）", () => {
    expect(componentSrc).toContain('className="msg-progress"');
    expect(componentSrc).toContain('role="status"');
  });
});
