import { describe, it, expect } from "vitest";
import { applyEvent, newAssistantMessage, markComplete, setError } from "./reducer";
import type { ThreadItem } from "./types";

function start(): ThreadItem[] {
  return [newAssistantMessage("a1")];
}

describe("applyEvent", () => {
  it("追加 thinking 步骤", () => {
    const s = applyEvent(start(), "thinking", { step: 1 });
    expect((s[0] as any).thinking).toEqual(["步骤 1"]);
  });

  it("新增 running 工具调用，再解析为 ok", () => {
    let s = applyEvent(start(), "tool_call", { call_id: "c1", name: "arxiv_search", arguments: { q: "rag" } });
    const tc = (s[0] as any).toolCalls[0];
    expect(tc.state).toBe("running");
    expect(tc.name).toBe("arxiv_search");
    s = applyEvent(s, "tool_result", { call_id: "c1", ok: true, output: { source: "rag", results: [] } });
    expect((s[0] as any).toolCalls[0].state).toBe("ok");
  });

  it("追加 message_delta 流式文本", () => {
    let s = applyEvent(start(), "message_delta", { text: "Hello" });
    s = applyEvent(s, "message_delta", { text: " world" });
    expect((s[0] as any).text).toBe("Hello world");
  });

  it("message 事件写入最终文本与引用", () => {
    const s = applyEvent(start(), "message", { text: "答", citations: [{ id: 1, page: 3, text: "x" }] });
    const m = s[0] as any;
    expect(m.text).toBe("答");
    expect(m.citations).toHaveLength(1);
  });

  it("done 事件标记完成并写入用量", () => {
    const s = applyEvent(start(), "done", { steps: 2, total_tokens: 100, cost: 0.01, latency_ms: 500 });
    const m = s[0] as any;
    expect(m.status).toBe("complete");
    expect(m.done.total_tokens).toBe(100);
  });

  it("verify 事件写入核验结果且不改写文本", () => {
    const s = applyEvent(start(), "verify", {
      level: 1,
      status: "issues",
      issues: [{ severity: "suspect", scope: "citation", page: 3, reason: "引用未能在第 3 页找到" }],
    });
    const m = s[0] as any;
    expect(m.verify.level).toBe(1);
    expect(m.verify.status).toBe("issues");
    expect(m.verify.issues[0].severity).toBe("suspect");
    expect(m.text).toBe(""); // 文本未被改动
  });

  it("error 事件标记错误并提取 reason", () => {
    const s = applyEvent(start(), "error", { reason: "boom" });
    const m = s[0] as any;
    expect(m.status).toBe("error");
    expect(m.error.message).toBe("boom");
  });

  it("保持不可变（不修改入参）", () => {
    const before = start();
    const after = applyEvent(before, "message_delta", { text: "x" });
    expect((before[0] as any).text).toBe("");
    expect((after[0] as any).text).toBe("x");
  });

  it("markComplete / setError 辅助函数", () => {
    const c = markComplete(start());
    expect((c[0] as any).status).toBe("complete");
    const e = setError(start(), "oops");
    expect((e[0] as any).status).toBe("error");
    expect((e[0] as any).error.message).toBe("oops");
  });

  it("不匹配的事件名原样返回", () => {
    const s = start();
    expect(applyEvent(s, "unknown", {})).toBe(s);
  });
});

describe("O-12 档 1 · progress（首字前进度提示）", () => {
  it("挂上 phase/text 供展示", () => {
    const s = applyEvent(start(), "progress", { phase: "retrieving", text: "正在检索关联文档…" });
    expect((s[0] as any).progress).toEqual({ phase: "retrieving", text: "正在检索关联文档…" });
  });

  it("进 progress 不改 text，也不进 thinking（不得虚增推理步数）", () => {
    const s = applyEvent(start(), "progress", { phase: "retrieving", text: "正在检索…" });
    expect((s[0] as any).text).toBe("");
    expect((s[0] as any).thinking).toEqual([]);
  });

  it("★ 首个正文增量到达即清除（否则与正文自相矛盾）", () => {
    let s = applyEvent(start(), "progress", { phase: "retrieving", text: "正在检索…" });
    s = applyEvent(s, "message_delta", { text: "答案" });
    expect((s[0] as any).progress).toBeNull();
    expect((s[0] as any).text).toBe("答案");
  });

  it("★ 轮次结束/出错也必须清除（否则异常轮长期残留）", () => {
    const withProgress = applyEvent(start(), "progress", { phase: "retrieving", text: "正在检索…" });
    expect((applyEvent(withProgress, "done", {})[0] as any).progress).toBeNull();
    expect((applyEvent(withProgress, "error", { reason: "boom" })[0] as any).progress).toBeNull();
  });
});
