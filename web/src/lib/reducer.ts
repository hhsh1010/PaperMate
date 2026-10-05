// 事件归约：把后端 SSE 事件（thinking/tool_call/tool_result/message_delta/
// message/done/error）纯函数式地映射到会话消息列表，保持不可变（便于 React 渲染）。
// 该模块不依赖 React / DOM，可独立单测（见 reducer.test.ts）。

import type {
  AssistantMessage,
  Citation,
  DoneMeta,
  ErrorInfo,
  ThreadItem,
  ToolCallView,
} from "./types";

export function newAssistantMessage(id: string): AssistantMessage {
  return {
    id,
    role: "assistant",
    thinking: [],
    toolCalls: [],
    text: "",
    citations: [],
    status: "streaming",
    created_at: new Date().toISOString(), // 流式开始时刻（历史消息由后端 created_at 覆盖）
  };
}

function mapLastAssistant(
  items: ThreadItem[],
  fn: (m: AssistantMessage) => AssistantMessage,
): ThreadItem[] {
  for (let i = items.length - 1; i >= 0; i--) {
    const it = items[i];
    if (it.role === "assistant") {
      const updated = fn(it);
      return [...items.slice(0, i), updated, ...items.slice(i + 1)];
    }
  }
  return items;
}

/**
 * 把一个已解析的 SSE 事件应用到消息列表（返回新数组，不修改入参）。
 * payload 为事件 data 解析后的对象。
 */
export function applyEvent(items: ThreadItem[], event: string, payload: Record<string, unknown>): ThreadItem[] {
  switch (event) {
    case "thinking":
      // V4 思考模式：payload.reasoning 为推理链增量 → 累积到 m.reasoning；
      // 否则为步骤标签（原有语义）
      if (payload?.reasoning != null) {
        return mapLastAssistant(items, (m) => ({
          ...m,
          reasoning: (m.reasoning ?? "") + String(payload.reasoning),
        }));
      }
      return mapLastAssistant(items, (m) => ({
        ...m,
        thinking: [...m.thinking, payload?.step != null ? `步骤 ${String(payload.step)}` : "思考中"],
      }));

    case "tool_call":
      return mapLastAssistant(items, (m) => ({
        ...m,
        toolCalls: [
          ...m.toolCalls,
          {
            id: String(payload?.call_id ?? ""),
            name: String(payload?.name ?? "tool"),
            args: payload?.arguments,
            state: "running",
          } as ToolCallView,
        ],
      }));

    case "tool_result":
      return mapLastAssistant(items, (m) => ({
        ...m,
        toolCalls: m.toolCalls.map((t) =>
          t.id === String(payload?.call_id)
            ? {
                ...t,
                state: payload?.ok ? "ok" : "error",
                result: payload?.output,
                error: payload?.error as string | undefined,
              }
            : t,
        ),
      }));

    case "progress":
      // O-12 档 1：首字之前的进度提示（doc-bound 首轮静默窗口，实测首字中位 ≈4s）。
      // 只挂在本条助手消息上展示；**不进 thinking / text** ⇒ 不影响推理步数计数与回答正文。
      return mapLastAssistant(items, (m) => ({
        ...m,
        progress: {
          phase: String(payload?.phase ?? ""),
          text: String(payload?.text ?? ""),
        },
      }));

    case "message_delta":
      return mapLastAssistant(items, (m) => ({
        ...m,
        text: m.text + String(payload?.text ?? ""),
        // O-12：正文一开始流式，进度提示即失效（否则"正在检索"与已出现的正文自相矛盾）
        progress: null,
      }));

    case "message":
      return mapLastAssistant(items, (m) => ({
        ...m,
        text: payload?.text == null ? m.text : String(payload.text),
        citations: (payload?.citations ?? []) as Citation[],
      }));

    case "verify":
      // P1-2.1 回复核验结果（只读展示，不改写 text/citations）
      return mapLastAssistant(items, (m) => ({
        ...m,
        verify: payload as unknown as import("./types").VerifyResult,
      }));

    case "done": {
      const done = payload as unknown as DoneMeta;
      return mapLastAssistant(items, (m) => ({
        ...m,
        status: "complete",
        done,
        // O-12：轮次结束（无论是否产出正文）进度提示必须清除 —— 否则异常轮会长期残留
        progress: null,
        // 打断提醒：无效回复时 done 携带 notice，上挂到 m.notice 供警示卡渲染
        ...(done?.notice ? { notice: done.notice } : {}),
      }));
    }

    case "error":
      return mapLastAssistant(items, (m) => ({
        ...m,
        status: "error",
        // O-12：出错时清除进度提示（错误信息本身已经说明问题，两行并存会互相干扰）
        progress: null,
        error: {
          message: payload?.reason ?? payload?.message ?? "未知错误",
          recoverable: payload?.recoverable,
          code: payload?.code,
        } as ErrorInfo,
      }));

    default:
      return items;
  }
}

/** 用户主动停止（AbortController）或流自然结束时，把仍在 streaming 的助手消息标记为完成 */
export function markComplete(items: ThreadItem[]): ThreadItem[] {
  return mapLastAssistant(items, (m) => (m.status === "streaming" ? { ...m, status: "complete" } : m));
}

/** 网络/请求异常时把仍在 streaming 的助手消息标记为错误 */
export function setError(items: ThreadItem[], message: string): ThreadItem[] {
  return mapLastAssistant(items, (m) =>
    m.status === "streaming" ? { ...m, status: "error", error: { message } } : m,
  );
}
