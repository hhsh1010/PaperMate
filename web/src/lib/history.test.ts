// B2-X1（2026-10-01）：历史消息 → ThreadItem 的映射契约。
//
// 为什么单独测这个纯函数：**"刷新后引用还在不在"是用户可见契约，而它此前写死为 `[]`**
// （后端只把引用写进 trace，messages 接口不回传 ⇒ 前端根本拿不到）⇒
// "指标有数、用户看不到"。把它抽到 lib + 单测，是为了让这类"静默丢失"再也不能悄悄回来。
import { describe, it, expect } from "vitest";
import { historyToThread } from "./history";
import type { AssistantMessage, MessageOut } from "./types";

function msg(over: Partial<MessageOut> = {}): MessageOut {
  return {
    id: 1,
    role: "assistant",
    content: "回答",
    prompt_tokens: 0,
    completion_tokens: 0,
    cost: 0,
    ...over,
  };
}

/** 断言并收窄成 AssistantMessage（避免测试里满屏 `as any`）。 */
function asAssistant(m: MessageOut): AssistantMessage {
  const t = historyToThread(m);
  if (t.role !== "assistant") throw new Error(`期望 assistant，实际 ${t.role}`);
  return t;
}

describe("historyToThread", () => {
  it("★ B2-X1：历史消息的引用必须回填（修复前写死 [] ⇒ 刷新即消失）", () => {
    const cites = [
      { id: "doc1-c0", page: 2, doc_id: 1, text: "证据段落" },
      { id: "doc1-c3", page: 5, doc_id: 1, text: "第二处证据", model_truncated: true },
    ];
    const t = asAssistant(msg({ citations: cites }));
    expect(t.citations).toEqual(cites);
  });

  it("无引用 / 脏数据（非数组）→ 空数组且不抛错", () => {
    expect(asAssistant(msg()).citations).toEqual([]);
    expect(asAssistant(msg({ citations: "oops" as unknown as [] })).citations).toEqual([]);
  });

  it("user 消息不受影响（不带 citations 字段）", () => {
    const t = historyToThread(msg({ role: "user", content: "问" }));
    expect(t.role).toBe("user");
    expect("citations" in t).toBe(false);
  });

  it("notice / invalid / modelRisk 的回填不变（重构不得回归）", () => {
    const notice = {
      kind: "stream_failed" as const,
      title: "回答生成中断",
      message: "内容可能不完整",
      suggestion: "可重试",
    };
    const t = asAssistant(msg({ meta: { invalid: true, notice, model_risk: true } }));
    expect(t.invalid).toBe(true);
    expect(t.notice).toEqual(notice);
    expect(t.modelRisk).toBe(true);
    // 引用与 notice 互不影响（同一轮既可能被中断、也可能已有引用）
    expect(t.citations).toEqual([]);
  });
});
