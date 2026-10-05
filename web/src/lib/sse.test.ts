import { describe, it, expect } from "vitest";
import { SSEParser, parseSSE } from "./sse";

describe("SSEParser", () => {
  it("解析单帧 event + data", () => {
    const frames = parseSSE('event: message_delta\ndata: {"text":"hi"}\n\n');
    expect(frames).toHaveLength(1);
    expect(frames[0].event).toBe("message_delta");
    expect(JSON.parse(frames[0].data!)).toEqual({ text: "hi" });
  });

  it("解析多帧", () => {
    const raw = "event: a\ndata: 1\n\nevent: b\ndata: 2\n\n";
    const frames = parseSSE(raw);
    expect(frames.map((f) => f.event)).toEqual(["a", "b"]);
  });

  it("兼容 CRLF 分隔符", () => {
    const frames = parseSSE("event: x\r\ndata: y\r\n\r\n");
    expect(frames[0].event).toBe("x");
    expect(frames[0].data).toBe("y");
  });

  it("去除冒号后单个前导空格", () => {
    const frames = parseSSE('event: thinking\ndata: {"step":1}\n\n');
    expect(frames[0].data).toBe('{"step":1}');
  });

  it("跨多次 push 累积不完整的帧", () => {
    const p = new SSEParser();
    expect(p.push("event: a\ndata: 1")).toHaveLength(0); // 不完整
    const rest = p.push("\n\nevent: b\ndata: 2\n\n");
    expect(rest).toHaveLength(2);
    expect(rest[0].event).toBe("a");
    expect(rest[1].event).toBe("b");
  });

  it("忽略注释行（: 开头）", () => {
    const frames = parseSSE(": comment\ndata: {" + '"x":1}\n\n');
    expect(frames).toHaveLength(1);
    expect(frames[0].event).toBeUndefined();
    expect(JSON.parse(frames[0].data!)).toEqual({ x: 1 });
  });

  // B8（2026-09-29）：后端在首个字节前先发一条**独立的**注释帧 `: keepalive\n\n`（防代理超时）。
  // 契约：整帧只有注释 ⇒ `parseFrame` 得空对象 ⇒ 返回 null ⇒ **被完全丢弃**、不产生任何帧，
  // 且**不影响后续帧**。这条用例是"后端加保活帧安全"的**前端侧依据**。
  it("丢弃独立的保活注释帧（不影响后续帧）", () => {
    const raw = ": keepalive\n\nevent: done\ndata: {\"steps\":1}\n\n";
    const frames = parseSSE(raw);
    expect(frames).toHaveLength(1); // 保活帧不计数
    expect(frames[0].event).toBe("done");
    expect(JSON.parse(frames[0].data!)).toEqual({ steps: 1 });
  });

  it("保活帧跨 push 切分也安全（半帧不误判）", () => {
    const p = new SSEParser();
    expect(p.push(": keepalive")).toHaveLength(0); // 尚未成帧
    const rest = p.push("\n\nevent: done\ndata: {}\n\n");
    expect(rest).toHaveLength(1);
    expect(rest[0].event).toBe("done");
  });

  it("多行 data 用换行拼接", () => {
    const frames = parseSSE('event: m\ndata: line1\ndata: line2\n\n');
    expect(frames[0].data).toBe("line1\nline2");
  });
});
