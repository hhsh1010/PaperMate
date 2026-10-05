// history：把后端历史消息（`MessageOut`）映射为前端 `ThreadItem` 的**纯函数**。
//
// 为什么单独放 lib/（B2-X1，2026-10-01）：
//  - 它原本内联在 `hooks/useStreamingChat.ts` 里 ⇒ 要测它就得拉起 React 运行时；
//    纯映射不该有这种耦合（项目**没有 testing-library**，组件/Hook 层难测）。
//  - 抽出来后可以直接在 vitest 里断言"刷新后引用是否还在"这个**用户可见契约**。
import type { Citation, MessageOut, ThreadItem } from "./types";

/** 将后端历史消息转换为前端 ThreadItem；历史无流中事件，仅展示文本、图片、引用、费用与时间戳。 */
export function historyToThread(m: MessageOut): ThreadItem {
  if (m.role === "user") {
    return {
      id: `m${m.id}`,
      role: "user",
      content: m.content,
      created_at: m.created_at ?? null,
      ...(m.images && m.images.length ? { images: m.images } : {}),
    };
  }
  // 打断提醒：历史消息从 meta 回填 notice/invalid，刷新后保留醒目警示卡
  const meta = m.meta && typeof m.meta === "object" ? m.meta : null;
  const notice = meta?.notice ?? null;
  const invalid = Boolean(meta?.invalid) || Boolean(notice);
  // 模型风险标记（2026-09-16）：该轮所用模型在绑定文档场景可能跳过检索 → 刷新后仍提示核对引用。
  // 与 notice 严格分开：它**不代表**本轮无效，只是"风险提示"。
  const modelRisk = Boolean(meta?.model_risk);
  // ★ P1-a2（2026-10-03）：**上下文占用**从 meta 回填。
  //   此前只存在于**当轮** SSE 的 `done` 里 ⇒ 刷新 / 切会话后就消失
  //   （与下面 B2-X1 的引用卡片是**同一类缺陷**）。后端已在落库时把三个数写进 `meta_json`。
  //   ⚠️ **缺 `budget` 就不填**（`ctxUsage` 需要分母才算得出百分比）——**降级而非编造**。
  // ★ P2（2026-10-03）：分项构成（刷新后仍能显示"哪块最大"）。
  //   ⚠️ 只接受"对象且非空"；脏数据/缺失 ⇒ 不填（面板退化为只显示总占用率）。
  const rawBd = meta?.ctx_breakdown;
  const ctxBreakdown =
    rawBd && typeof rawBd === "object" && Object.keys(rawBd).length > 0
      ? (rawBd as Record<string, number>)
      : undefined;
  const ctxPeak = typeof meta?.ctx_peak === "number" ? meta.ctx_peak : undefined;
  const ctxBudget = typeof meta?.ctx_budget === "number" ? meta.ctx_budget : undefined;
  const ctxWarnAt = typeof meta?.ctx_warn_at === "number" ? meta.ctx_warn_at : undefined;
  const ctx =
    ctxPeak !== undefined && ctxBudget !== undefined
      ? {
          peak: ctxPeak,
          budget: ctxBudget,
          ...(ctxWarnAt !== undefined ? { warnAt: ctxWarnAt } : {}),
          ...(ctxBreakdown ? { breakdown: ctxBreakdown } : {}),
        }
      : undefined;
  // ★ B2-X1（2026-10-01）：引用必须从历史里回填。
  //   此前这里**写死 `citations: []`** ⇒ 引用卡片"当轮看得到、刷新即消失"
  //   （后端也只把引用写进 trace，messages 接口不回传 ⇒ 前端拿不到）。
  //   `Array.isArray` 兜底：历史脏数据（后端字段异常）不该让整页历史渲染崩掉。
  const citations: Citation[] = Array.isArray(m.citations) ? m.citations : [];
  return {
    id: `m${m.id}`,
    role: "assistant",
    thinking: [],
    toolCalls: [],
    text: m.content,
    citations,
    status: "complete",
    created_at: m.created_at ?? null,
    // 仅当确为无效回复（有 notice）时才带 notice，避免把普通回复误标
    ...(notice ? { notice } : {}),
    ...(invalid ? { invalid } : {}),
    ...(modelRisk ? { modelRisk } : {}),
    // ★ P1-a2：占用率（缺 budget 时不填，见上面的构造说明）
    ...(ctx ? { ctx } : {}),
  };
}
