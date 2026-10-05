import { memo } from "react";
import type { ThreadItem } from "../lib/types";
import { ChatMessageView } from "./ChatMessageView";
import { Icons } from "../lib/icons";

/** 消息列表：渲染整个会话线程，末尾锚点用于自动滚动到底部
 *
 * O-08（2026-10-01）：`memo` 化 —— ChatPage 因流式增量重渲染时，若 items 引用未变
 * （如仅输入框/文档面板状态变化），整列表跳过重渲染。前提是各回调 props 稳定：
 * ChatPage 侧 onBottomRef/onRegenerate/activeDocIds 已相应稳定化。
 * ⚠️ 不得在 map 里给 ChatMessageView 传内联箭头函数（新引用会使 memo 静默失效）。
 */
export const MessageList = memo(function MessageList({
  items,
  onBottomRef,
  onOpenCitation,
  activeDocIds,
  onRegenerate,
  lastAssistantId,
  onRate,
  ratedMap,
}: {
  items: ThreadItem[];
  onBottomRef?: (el: HTMLDivElement | null) => void;
  onOpenCitation?: (c: import("../lib/types").Citation) => void;
  activeDocIds?: number[];
  onRegenerate?: () => void;
  /** 最后一条 assistant 消息 id：仅它的消息显示「重新生成」按钮 */
  lastAssistantId?: string;
  /** E1：👍/👎 回调（messageId, rating）—— 须为稳定引用（useCallback） */
  onRate?: (messageId: number, rating: 1 | -1) => void;
  /** E1：messageId → 当前评分（刷新后回填高亮用） */
  ratedMap?: Record<number, number>;
}) {
  return (
    <div className="message-list">
      {items.length === 0 && (
        <div className="empty">
          <div style={{ fontSize: 26, marginBottom: 8 }}><Icons.chat size={14} /> </div>
          <div>还没有对话</div>
          <div className="muted" style={{ marginTop: 4 }}>
            在上方输入问题开始提问；如需基于论文回答，请先在右侧勾选关联文档。
          </div>
        </div>
      )}
      {items.map((it) => (
        <ChatMessageView
          key={it.id}
          item={it}
          onOpenCitation={onOpenCitation}
          activeDocIds={activeDocIds}
          onRegenerate={onRegenerate}
          showRegenerate={it.id === lastAssistantId}
          onRate={onRate}
          myRating={ratedMap?.[Number(it.id)]}
        />
      ))}
      <div ref={onBottomRef} />
    </div>
  );
});
