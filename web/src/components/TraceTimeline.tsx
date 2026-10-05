import type { TraceEvent, TraceOut } from "../lib/types";

/** ER-19：把 trace 事件渲染为推理轨迹时间线（思维链 / 工具调用 / 结果 / 消息）。 */
export function TraceTimeline({ trace }: { trace: TraceOut }) {
  const events = (trace.events || []).slice().sort((a, b) => (a.idx ?? 0) - (b.idx ?? 0));
  if (!events.length) return <div className="muted">该轨迹没有事件记录。</div>;

  return (
    <div className="trace-timeline">
      {events.map((ev, i) => (
        // N9 修复（2026-09-10）：排序已承认 idx 可能缺失（`?? 0`），key 却直接用 ev.idx →
        // 多事件缺 idx 时 key 全为 undefined（重复 key 警告 + 节点复用错乱）。
        // 回退到排序后的位置索引，保证唯一。
        <div className={`trace-node type-${ev.type}`} key={ev.idx ?? `pos-${i}`}>
          <div className="trace-dot" aria-hidden="true" />
          <div className="trace-body">
            <div className="trace-type">{labelOf(ev.type)}</div>
            <div className="trace-detail">{renderEvent(ev)}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function labelOf(type: string): string {
  const map: Record<string, string> = {
    thinking: "推理",
    tool_call: "工具调用",
    tool_result: "工具结果",
    message: "最终回答",
    error: "错误",
    done: "完成",
  };
  return map[type] || type;
}

function renderEvent(ev: TraceEvent) {
  const d = ev.data || {};
  switch (ev.type) {
    case "thinking":
      return <pre className="trace-pre">{String(d.content ?? d.text ?? "")}</pre>;
    case "tool_call":
      return (
        <div>
          <span className="trace-tool-name">{(d.name as string) || "tool"}</span>
          <pre className="trace-pre">{JSON.stringify(d.arguments ?? {}, null, 2)}</pre>
        </div>
      );
    case "tool_result": {
      const out = d.output as { source?: string; results?: unknown[] } | undefined;
      return (
        <div>
          <span className="trace-tool-name">{(d.name as string) || "tool"} → {out?.source || ""}</span>
          {Array.isArray(out?.results) && (
            <div className="trace-hits">命中 {out!.results!.length} 段</div>
          )}
        </div>
      );
    }
    case "message":
      return (
        <div>
          <span className="trace-tool-name">已生成（含 {Array.isArray(d.citations) ? d.citations.length : 0} 处引用）</span>
        </div>
      );
    case "error":
      return <div className="trace-error">{String(d.message ?? d.error ?? "未知错误")}</div>;
    case "done":
      return (
        <div className="trace-done">
          步数 {(d.steps as number) ?? 0} · 成本 ¥{(Number(d.cost) || 0).toFixed(4)} · tokens {(d.total_tokens as number) ?? 0}
        </div>
      );
    default:
      return <pre className="trace-pre">{JSON.stringify(d, null, 2)}</pre>;
  }
}
