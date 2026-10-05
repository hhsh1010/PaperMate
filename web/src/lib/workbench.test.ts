import { describe, it, expect } from "vitest";
import { mergeOp, dropPid, findOp, withQuote, summarize } from "./workbench";
import type { EditOp } from "./types";

/** C4：改稿工作台变更集纯函数 */
const op = (pid: string, o: EditOp["op"], text: string): EditOp => ({ pid, op: o, text });

describe("mergeOp", () => {
  it("追加新 pid", () => {
    const out = mergeOp([op("P0", "replace", "a")], op("P1", "append", "b"));
    expect(out).toHaveLength(2);
  });

  it("同 pid 覆盖（最后生效）：旧 op 被移除、新 op 追加到末尾", () => {
    const base = [op("P0", "replace", "a"), op("P2", "append", "c")];
    const out = mergeOp(base, op("P0", "delete", "x"));
    expect(out).toHaveLength(2);
    // 每 pid 仅一条 op（顺序无跨段语义）；P0 以最后一次编辑为准
    expect(out.map((e) => e.pid).sort()).toEqual(["P0", "P2"]);
    expect(findOp(out, "P0")?.op).toBe("delete");
    expect(findOp(out, "P2")?.op).toBe("append");
  });
});

describe("dropPid / findOp", () => {
  it("dropPid 移除该段全部 op", () => {
    const base = [op("P0", "replace", "a"), op("P1", "append", "b"), op("P0", "delete", "c")];
    const out = mergeOp(base, op("P1", "append", "b")); // 先确保无重复干扰
    expect(dropPid(out, "P0")).toHaveLength(1);
  });

  it("findOp 命中 / 未命中", () => {
    const base = [op("P1", "append", "b")];
    expect(findOp(base, "P1")?.op).toBe("append");
    expect(findOp(base, "P9")).toBeNull();
  });
});

describe("withQuote", () => {
  it("截取原文前 80 字做 quote", () => {
    const long = "字".repeat(120);
    expect(withQuote(op("P0", "replace", "x"), long).quote).toHaveLength(80);
    expect(withQuote(op("P0", "replace", "x"), "短文本").quote).toBe("短文本");
  });

  it("空白原文 → quote 为 null（后端跳过校验）", () => {
    expect(withQuote(op("P0", "insert_after", "x"), "   ").quote).toBeNull();
  });
});

describe("summarize", () => {
  it("分类计数", () => {
    const s = summarize([
      op("P0", "replace", "a"),
      op("P1", "replace", "b"),
      op("P2", "insert_after", "c"),
      op("P3", "delete", "d"),
    ]);
    expect(s).toEqual({ total: 4, replaces: 2, inserts: 1, deletes: 1 });
  });
});
