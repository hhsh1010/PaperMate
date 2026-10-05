import { describe, it, expect } from "vitest";
import { isWorkspace } from "./types";
import type { SessionOut } from "./types";

/**
 * A2：左栏分组判定。工作间 = sessions[].config.workspace === true，
 * 分组完全由列表数据派生，不额外请求后端 —— 故该纯函数必须严格正确。
 */
function sess(config: Record<string, unknown>): SessionOut {
  return { id: 1, title: "t", config, created_at: null, updated_at: null };
}

describe("isWorkspace", () => {
  it("config.workspace 为 true → 工作间", () => {
    expect(isWorkspace(sess({ workspace: true }))).toBe(true);
  });

  it("config.workspace 为 false / 缺失 / 空 config → 普通会话", () => {
    expect(isWorkspace(sess({ workspace: false }))).toBe(false);
    expect(isWorkspace(sess({}))).toBe(false);
    expect(isWorkspace(sess({ doc_ids: [1, 2] }))).toBe(false);
  });

  it("仅关联了文献集（collection_id）但非工作间 → 仍为普通会话", () => {
    // D1 收敛前存量会话：有 collection_id 但无 workspace 标记，不应被误分组
    expect(isWorkspace(sess({ collection_id: 3 }))).toBe(false);
  });

  it("字符串 \"true\" 不视为工作间（严格布尔判定，防 JSON 类型漂移）", () => {
    expect(isWorkspace(sess({ workspace: "true" }))).toBe(false);
  });

  it("null / undefined 安全（不抛异常）", () => {
    expect(isWorkspace(null)).toBe(false);
    expect(isWorkspace(undefined)).toBe(false);
  });

  it("分组切分：列表被完整且不重复地分为两组", () => {
    const list: SessionOut[] = [
      sess({ workspace: true }),
      sess({}),
      sess({ workspace: true }),
      sess({ collection_id: 2 }),
    ];
    const ws = list.filter(isWorkspace);
    const chat = list.filter((s) => !isWorkspace(s));
    expect(ws).toHaveLength(2);
    expect(chat).toHaveLength(2);
    expect(ws.length + chat.length).toBe(list.length);
  });
});
