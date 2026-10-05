// useStreamingChat：封装「发送消息 → 消费 SSE 流 → 归约到消息列表」的完整生命周期。
// - POST /sessions/{id}/messages/stream（SSE 是 POST，故用 fetch + ReadableStream）
// - 用 AbortController 支持「停止」按钮中断（docs/07 §M3.2 验收：停止按钮可中断 Agent）
// - 通过纯函数 reducer(applyEvent) 更新状态，错误边界清晰。
// - 切换 sessionId 时自动清空并加载历史（GET /sessions/{id}），避免看到旧会话尾巴。

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { SSEParser } from "../lib/sse";
import { applyEvent, markComplete, newAssistantMessage, setError } from "../lib/reducer";
import type { ThreadItem, UserMessage } from "../lib/types";
import { toErrorMessage } from "../lib/error";
// B2-X1：`historyToThread` 已抽到 `lib/history.ts`（纯函数 ⇒ 可单测，不必拉起 React）
import { historyToThread } from "../lib/history";

function uid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return Math.random().toString(36).slice(2);
}

export function useStreamingChat(sessionId: number | null) {
  const [items, setItems] = useState<ThreadItem[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  /** 各会话在飞流的 AbortController（N2 2026-09-10 改为按会话存）：单槽在
   *  「A 的流未结束就切到 B」时会指向他会话，导致 stop() 或并发保护判错对象。 */
  const controllersRef = useRef<Map<number, AbortController>>(new Map());
  /** 各会话在飞流的请求级标识（F8 2026-09-07；同 N2 改按会话存）：
   *  随请求发送，使 /stop 能精确命中**该会话**的流。 */
  const streamIdsRef = useRef<Map<number, string>>(new Map());
  /** 当前视图对应的会话（N2 2026-09-10）：事件写入前比对它——切走后旧流事件不得写入新视图。 */
  const sessionIdRef = useRef<number | null>(sessionId);
  sessionIdRef.current = sessionId;
  /** **有在飞流的会话集合**（N2 2026-09-10）：替代原「全局单代际 + 切会话 abort」模型——
   *  切会话不再中断生成，只是不再写入视图；切回时据此启动补偿轮询，把落库结果补上。 */
  const activeStreamsRef = useRef<Set<number>>(new Set());

  // 切换会话：清空 + 拉历史（+ 若该会话仍有在飞流，启动补偿轮询）
  useEffect(() => {
    // N2 修复（2026-09-10）：**不再 abort 在飞流**。
    // 原实现在切会话时 `abortRef.abort()`，而代际守卫只能"丢弃事件"、拦不住 abort 本身——
    // 结果正在生成的回复被真的杀掉（用户实测：A 里发长回复 → 切到 B → 回 A，该回复消失）。
    // 现在切视图只做「停止写入 + 清空」，后台生成继续；切回时靠下面的补偿轮询与
    // 流结束刷新（见 consume 的 finally）补齐落库内容。
    setStreaming(sessionId != null && activeStreamsRef.current.has(sessionId));
    setItems([]);
    setHistoryError(null);
    if (sessionId == null) return;
    const sid = sessionId;
    let cancelled = false;
    const load = () =>
      api.getSession(sid).then((detail) => {
        // 仍停留在本会话才写入（避免快速切换时旧响应覆盖）
        if (!cancelled && sessionIdRef.current === sid) setItems(detail.messages.map(historyToThread));
      });
    setLoadingHistory(true);
    void load()
      .catch((e: unknown) => {
        if (!cancelled) setHistoryError(toErrorMessage(e) || "加载历史失败");
      })
      .finally(() => {
        if (!cancelled) setLoadingHistory(false);
      });
    // N2 补偿：该会话的流仍在跑（生成中/切走后未结束）→ 每 1.5s 拉一次历史，
    // 直到流结束。这样切回来能持续看到落库进度，而不是「什么都没有」。
    let iv: number | undefined;
    if (activeStreamsRef.current.has(sid)) {
      iv = window.setInterval(() => {
        if (!activeStreamsRef.current.has(sid)) {
          if (iv) window.clearInterval(iv);
          return;
        }
        void load();
      }, 1500);
    }
    return () => {
      cancelled = true;
      if (iv) window.clearInterval(iv);
    };
  }, [sessionId]);

  /** 消费 SSE 流：把事件归约到「最后一条 assistant 消息」（applyEvent 的语义），结束/中断时收尾。 */
  const consume = useCallback(
    async (body: object) => {
      const sid = sessionId;
      if (!sid) return;
      // N2 修复（2026-09-10）：按**会话**维护在飞流状态（activeStreamsRef），替代原
      // 「全局单代际 + 切会话 abort」模型。事件写入以 sessionIdRef（当前视图）为准：
      // 切走后事件不写入视图但**流继续跑**，切回时由补偿轮询/结束时刷新补齐。
      if (activeStreamsRef.current.has(sid)) return; // 同会话并发保护
      activeStreamsRef.current.add(sid);
      const controller = new AbortController();
      controllersRef.current.set(sid, controller);
      // F8（2026-09-07）：为本次请求生成唯一流标识并随请求发送——后端据此按
      // (session_id, stream_id) 注册取消令牌，同会话并发流互不覆盖。
      const streamId =
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? crypto.randomUUID()
          : `s${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
      streamIdsRef.current.set(sid, streamId);
      setStreaming(true);
      try {
        const res = await fetch(`/api/v1/sessions/${sessionId}/messages/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...body, stream_id: streamId }),
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          throw new Error(`请求失败（HTTP ${res.status}）`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        const parser = new SSEParser();
        /** 单帧处理：解析 JSON → 归约到最后一条 assistant（push 与 flush 共用，防尾帧丢失） */
        const handleFrame = (f: { event?: string; data?: string }) => {
          if (sessionIdRef.current !== sid) return; // N2：已切走——不写入当前视图（流仍在跑）
          if (!f.event || !f.data) return;
          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(f.data);
          } catch {
            return; // 跳过非法 JSON 帧
          }
          setItems((prev) => applyEvent(prev, f.event as string, payload));
        };
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          const frames = parser.push(decoder.decode(value, { stream: true }));
          for (const f of frames) handleFrame(f);
        }
        // 2026-09-06 修复：flush 残留缓冲——末帧无 trailing 空行时（部分代理/缓冲场景）
        // 此前会被留在 buf 里丢弃，丢的往往是 done 帧（steps/cost/latency 用量行随之丢失）。
        for (const f of parser.flush()) handleFrame(f);
        if (sessionIdRef.current !== sid) return; // N2：已切走，收尾不作用于当前视图
        setItems((prev) => markComplete(prev));
      } catch (e: unknown) {
        if (sessionIdRef.current !== sid) return; // N2：切走后旧流的 catch 不得作用于当前视图
        if (e instanceof Error && e.name === "AbortError") {
          setItems((prev) => markComplete(prev));
        } else {
          setItems((prev) => setError(prev, toErrorMessage(e) || "请求异常"));
        }
      } finally {
        // N2：先登记"该会话不再有在飞流"，再处理视图收尾——
        // 切回时的补偿轮询据此停止。
        activeStreamsRef.current.delete(sid);
        if (sessionIdRef.current === sid) {
          setStreaming(false);
          // N2 关键修复：流结束（正文/文本已落库）后**重拉历史**替换本地状态——
          // 否则「切走期间生成的回复」在视图里永远不会出现（用户实测：回复消失）。
          // 稍作延迟，避开后端落库事务与 SSE 收尾的竞争。
          window.setTimeout(() => {
            if (sessionIdRef.current !== sid) return;
            void api
              .getSession(sid)
              .then((detail) => {
                if (sessionIdRef.current === sid) setItems(detail.messages.map(historyToThread));
              })
              .catch(() => {
                /* 刷新失败不影响已有视图（下次切回/发送还会再拉） */
              });
          }, 250);
        }
        if (controllersRef.current.get(sid) === controller) controllersRef.current.delete(sid);
        if (streamIdsRef.current.get(sid) === streamId) streamIdsRef.current.delete(sid);
      }
    },
    [sessionId],
  );

  const send = useCallback(
    async (
      content: string,
      tools?: string[],
      images?: string[],
      mentionDocIds?: number[],
    ): Promise<boolean> => {
      if (!sessionId) return false;
      // 2026-09-06 修复：并发保护——用 ref（非 state，无渲染延迟）判断在飞的流；
      // 此前连点发送会起双流，abortRef 单槽被覆盖导致第一条流失控。
      // N6 修复（2026-09-10）：返回 **false** 交由调用方提示——原先静默 return，
      // 用户「点了发送但毫无反应」，输入被无声丢弃。
      // N2：同会话并发保护按**会话**判定（原 abortRef 单槽可能正指向其他会话的流）
      if (activeStreamsRef.current.has(sessionId)) return false;
      const userMsg: ThreadItem = {
        id: uid(),
        role: "user",
        content,
        created_at: new Date().toISOString(),
        ...(images && images.length ? { images } : {}),
      };
      const assistantId = uid();
      setItems((prev) => [...prev, userMsg, newAssistantMessage(assistantId)]);
      // 2026-09-07：结构化 @ 提及——精确 doc id 随消息发送，后端据此限缩检索/技能选文
      await consume({
        content,
        tools: tools ?? null,
        ...(images && images.length ? { images } : {}),
        ...(mentionDocIds && mentionDocIds.length ? { mention_doc_ids: mentionDocIds } : {}),
      });
      return true;
    },
    [sessionId, consume],
  );

  /** 重新生成：仅针对「最后一轮」——移除当前最后一条 assistant 回复，用其前的用户消息重跑，
   *  并带 regenerate:true 让后端复用 DB 中已有用户消息、裁掉旧回复，避免重复落库。
   *  受 applyEvent「作用于最后一条 assistant」语义约束，仅重跑最后一轮才能保证事件正确归约。 */
  const regenerate = useCallback(async () => {
    if (!sessionId || streaming) return;
    let lastAssistantIdx = -1;
    for (let i = items.length - 1; i >= 0; i--) {
      if (items[i].role === "assistant") {
        lastAssistantIdx = i;
        break;
      }
    }
    if (lastAssistantIdx < 0) return;
    let userMsg: UserMessage | undefined;
    for (let i = lastAssistantIdx - 1; i >= 0; i--) {
      if (items[i].role === "user") {
        userMsg = items[i] as UserMessage;
        break;
      }
    }
    if (!userMsg) return;
    const oldAssistantId = items[lastAssistantIdx].id;
    const newId = uid();
    setItems((prev) => {
      const idx = prev.findIndex((m) => m.id === oldAssistantId);
      const filtered = prev.filter((m) => m.id !== oldAssistantId);
      const placeholder = newAssistantMessage(newId);
      if (idx < 0) return [...filtered, placeholder];
      return [...filtered.slice(0, idx), placeholder, ...filtered.slice(idx)];
    });
    await consume({
      content: userMsg.content,
      ...(userMsg.images && userMsg.images.length ? { images: userMsg.images } : {}),
      regenerate: true,
    });
  }, [sessionId, streaming, items, consume]);

  const stop = useCallback(() => {
    // N2：只中止**当前会话**的流——按会话取 controller/streamId，
    // 防止「A 的流未结束就切到 B」时误杀 A 的生成或传错 stream_id。
    if (sessionId == null) return;
    controllersRef.current.get(sessionId)?.abort();
    // 2026-09-07：同时通知后端显式取消（/sessions/{id}/stop）——不依赖 ASGI 断连
    // 语义，确定性终止 LLM 生成与计费（思考中 abort 连接后后端仍续跑的根治手段）。
    void api.stopStream(sessionId, streamIdsRef.current.get(sessionId) ?? undefined);
  }, [sessionId]);

  return { items, streaming, loadingHistory, historyError, send, stop, regenerate };
}
