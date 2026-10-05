import { describe, expect, it } from "vitest";

import chatMessageSrc from "./ChatMessageView.tsx?raw";
import messageListSrc from "./MessageList.tsx?raw";
import chatPageSrc from "../pages/ChatPage.tsx?raw";

/**
 * O-08（2026-10-01）：流式渲染 memo 化的源码契约守护。
 *
 * 背景：流式输出每个增量都会触发 MessageList 重渲染；若 memo 链路被破坏
 * （如有人重新引入内联箭头函数 props），全部消息会恢复"每 token 全量重渲染"，
 * 且**不报错**（memo 失效是静默的）⇒ 只能用源码契约拦。
 * ⚠️ 断言必须指向"会被执行的那段"（真实 JSX 传参处），不能只断言标识符出现过。
 */
describe("O-08 流式渲染 memo 链路（源码契约）", () => {
  it("ChatMessageView 被 React.memo 包裹", () => {
    expect(chatMessageSrc).toContain("export const ChatMessageView = memo(");
  });

  it("MessageList 被 React.memo 包裹", () => {
    expect(messageListSrc).toContain("export const MessageList = memo(");
  });

  it("MessageList 不得恢复逐条内联包装 onRate（新引用使 memo 静默失效）", () => {
    expect(messageListSrc).not.toContain("(r) => onRate(");
    // 真实传参处：稳定回调直传
    expect(messageListSrc).toContain("onRate={onRate}");
  });

  it("ChatPage 传给 MessageList 的回调/数组 props 必须是稳定引用", () => {
    expect(chatPageSrc).toContain("onBottomRef={handleBottomRef}");
    expect(chatPageSrc).toContain("onRegenerate={regenerate}");
    expect(chatPageSrc).toContain("activeDocIds={activeDocIds}");
    expect(chatPageSrc).toContain("onRate={rateMessage}");
    // 内联箭头/内联 map 一律禁止回到 MessageList 的传参处
    expect(chatPageSrc).not.toContain("onRegenerate={() => regenerate()}");
    expect(chatPageSrc).not.toContain("onBottomRef={(el) =>");
  });

  it("ChatMessageView 的评分按钮按 (messageId, rating) 签名调用（真实执行段）", () => {
    expect(chatMessageSrc).toContain("onRate?.(Number(m.id), 1)");
    expect(chatMessageSrc).toContain("onRate?.(Number(m.id), -1)");
  });
});
