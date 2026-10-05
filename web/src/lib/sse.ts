// SSE 增量解析器（docs/04 §4.6.2 事件协议）。
// 后端以 `event: <name>` + `data: <json>` 行、空行分隔逐帧推送。
// 注意：SSE 流式接口是 POST（/sessions/{id}/messages/stream），不能用 EventSource，
// 因此前端用 fetch + ReadableStream 读取，本模块负责把字节流解析成结构化帧。

import type { SSEFrame } from "./types";

/**
 * 增量解析器：可多次 push 分块数据，仅返回「已完整」的帧；
 * 不完整的尾帧留在内部缓冲区，待后续数据到达或 flush() 时再处理。
 */
export class SSEParser {
  private buf = "";

  push(chunk: string): SSEFrame[] {
    this.buf += chunk;
    const frames: SSEFrame[] = [];
    for (;;) {
      const nl = this.buf.indexOf("\n\n");
      const cr = this.buf.indexOf("\r\n\r\n");
      let end = -1;
      let delim = 2;
      if (nl === -1 && cr === -1) break;
      if (cr !== -1 && (nl === -1 || cr < nl)) {
        end = cr;
        delim = 4;
      } else {
        end = nl;
        delim = 2;
      }
      const raw = this.buf.slice(0, end);
      this.buf = this.buf.slice(end + delim);
      const f = parseFrame(raw);
      if (f) frames.push(f);
    }
    return frames;
  }

  /** 取出缓冲区中残留的最后一帧（流结束时调用） */
  flush(): SSEFrame[] {
    if (!this.buf.trim()) {
      this.buf = "";
      return [];
    }
    const f = parseFrame(this.buf);
    this.buf = "";
    return f ? [f] : [];
  }
}

function parseFrame(raw: string): SSEFrame | null {
  const lines = raw.split(/\r?\n/);
  const frame: SSEFrame = {};
  for (const line of lines) {
    if (line.startsWith(":")) continue; // 注释行
    const colon = line.indexOf(":");
    if (colon === -1) {
      assignField(frame, line.trim(), "");
      continue;
    }
    const field = line.slice(0, colon).trim();
    let value = line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    assignField(frame, field, value);
  }
  if (!frame.event && !frame.data) return null;
  return frame;
}

function assignField(frame: SSEFrame, field: string, value: string): void {
  if (field === "event") frame.event = value;
  else if (field === "data") frame.data = (frame.data ? frame.data + "\n" : "") + value;
  else if (field === "id") frame.id = value;
  else if (field === "retry") frame.retry = value;
}

/** 一次性解析整段 SSE 文本（测试/简单场景用） */
export function parseSSE(raw: string): SSEFrame[] {
  const p = new SSEParser();
  return [...p.push(raw), ...p.flush()];
}
