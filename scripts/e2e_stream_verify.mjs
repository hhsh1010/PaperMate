// e2e 验证：检索（tool_choice 强制 retrieve）+ 持久化（persist_turn）双确认
// 用法：node scripts/e2e_stream_verify.mjs [session_id] [content]
import { spawn } from "node:child_process";

const BASE = "http://127.0.0.1:8000/api/v1";
const sessionId = process.argv[2] || "7";
const content = process.argv[3] || "总结文档内容";

async function getJSON(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

async function getMessages() {
  const detail = await getJSON(`/sessions/${sessionId}`);
  return detail.messages || [];
}

const before = await getMessages();
console.log(`[BEFORE] messages=${before.length}`);

const res = await fetch(`${BASE}/sessions/${sessionId}/messages/stream`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ content }),
});
console.log(`[HTTP] ${res.status} ${res.statusText}`);
if (!res.ok) { console.log(await res.text()); process.exit(1); }

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buf = "";
let toolCalls = 0, toolResults = 0, firstHit = null, finalText = "", done = null, errors = [];

function handleEvent(type, payload) {
  const ev = payload || {};
  switch (type) {
    case "tool_call":
      toolCalls++;
      console.log(`[TOOL_CALL] ${ev.name || "?"} args=${JSON.stringify(ev.arguments || {})}`);
      break;
    case "tool_result":
      toolResults++;
      const out = ev.output || {};
      const hits = Array.isArray(out.results) ? out.results.length : 0;
      const first = Array.isArray(out.results) && out.results[0] ? out.results[0] : null;
      if (first) firstHit = first;
      console.log(`[TOOL_RESULT] ${ev.name || "?"} ok=${ev.ok} hits=${hits} src=${out.source} first=${first ? first.id : "-"}`);
      break;
    case "message_delta":
      break;
    case "message":
      finalText = ev.text || "";
      console.log(`[MESSAGE] citations=${(ev.citations || []).length} textLen=${finalText.length}`);
      console.log(`  text: ${finalText.slice(0, 220).replace(/\n/g, " | ")}`);
      break;
    case "done":
      done = ev;
      console.log(`[DONE] steps=${ev.steps} tokens=${ev.total_tokens} cost=${ev.cost}`);
      break;
    case "error":
      errors.push(ev);
      console.log(`[ERROR] ${JSON.stringify(ev).slice(0, 300)}`);
      break;
    default:
      console.log(`[${type}]`);
  }
}

while (true) {
  const { done: rd, value } = await reader.read();
  if (rd) break;
  buf += decoder.decode(value, { stream: true });
  let idx;
  while ((idx = buf.indexOf("\n\n")) >= 0) {
    const raw = buf.slice(0, idx);
    buf = buf.slice(idx + 2);
    const lines = raw.split("\n");
    const evLine = lines.find((l) => l.startsWith("event:"));
    const dataLine = lines.find((l) => l.startsWith("data:"));
    if (!dataLine) continue;
    let ev = evLine ? evLine.slice(6).trim() : undefined;
    let payload;
    try { payload = JSON.parse(dataLine.slice(5).trim()); }
    catch { continue; }
    if (!ev) ev = payload?.type;  // 兼容无 event: 行的帧
    handleEvent(ev, payload);
  }
}

const after = await getMessages();
const added = after.length - before.length;
console.log(`[AFTER] messages=${after.length} added=${added}`);
const verdict = (toolCalls >= 1 && toolResults >= 1 && firstHit && added >= 2 && errors.length === 0)
  ? "PASS ✅ (retrieve 命中真实文档 + 持久化成功)"
  : "FAIL ❌";
console.log(`[VERDICT] ${verdict}`);
process.exit(verdict.startsWith("PASS") ? 0 : 2);
