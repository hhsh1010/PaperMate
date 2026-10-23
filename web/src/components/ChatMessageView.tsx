import { memo, useState } from "react";
import type { AssistantMessage, ThreadItem, ToolCallView } from "../lib/types";
import { ToolCallCard } from "./ToolCallCard";
import { CitationCard } from "./CitationCard";
import { MarkdownText } from "./MarkdownText";
import { VerifyCard } from "./VerifyCard";
import { Icons } from "../lib/icons";
import { CtxUsageBar } from "./CtxUsagePanel";

/** 「推理过程」面板里**展示哪些工具卡片**（2026-09-28）。
 *
 * `retrieve` 是**内部检索手段**，不作为"工具"展示：一次提问常触发 3~5 次 retrieve
 * （`/术语表`、表格类问题尤甚），逐张卡片会把推理区刷屏，而用户只关心最终回答与
 * 引用来源（引用由 `CitationCard` 独立呈现，来源信息不会因此丢失）。
 *
 * ⚠️ **面板标题的计数也必须用本函数的结果**——否则会出现「3 个工具」却一张卡片
 * 都看不到的自相矛盾（计数与展示必须同源）。
 *
 * 返回新数组，不修改入参（纯函数，便于单测）。
 */
export function visibleToolCalls(toolCalls: ToolCallView[]): ToolCallView[] {
  return toolCalls.filter((t) => t.name !== "retrieve");
}

/** 单条消息渲染：用户气泡 / 助手（头像 + 推理 + 工具 + 流式正文 + 引用 + 用量）
 *
 * O-08（2026-10-01）：`memo` 化 —— 流式输出每个增量都会让父级 MessageList 重渲染，
 * 此前全部消息跟着重跑 Markdown 解析与 renderBlock；memo 后仅 `item` 引用变化的那一条
 * （正在流式的最后一条助手消息）重渲染，历史消息直接跳过。
 * 前提：其余 props 必须稳定 —— `onRate` 因此改为直传 `(messageId, rating)` 签名的
 * 稳定回调（由 ChatPage 的 useCallback 提供），不再由 MessageList 逐条包内联箭头函数
 * （内联函数每次渲染都是新引用，memo 会静默失效——这正是卡片警告的"失效而不报错"）。
 */
export const ChatMessageView = memo(function ChatMessageView({
  item,
  onOpenCitation,
  activeDocIds,
  onRegenerate,
  showRegenerate,
  onRate,
  myRating,
}: {
  item: ThreadItem;
  onOpenCitation?: (c: import("../lib/types").Citation) => void;
  activeDocIds?: number[];
  /** 重新生成回调（仅最后一轮助手消息显示按钮） */
  onRegenerate?: () => void;
  /** 是否在本条显示「重新生成」按钮（由父级判定为最后一条 assistant 且已完成） */
  showRegenerate?: boolean;
  /** E1：👍/👎 回调，签名 (messageId, rating)；须为稳定引用（useCallback），否则 memo 失效。 */
  onRate?: (messageId: number, rating: 1 | -1) => void;
  /** E1：本条**当前**的评分（1 / -1 / undefined=未评），用于按钮回填高亮 */
  myRating?: number;
}) {
  const [copied, setCopied] = useState(false);
  /** ★ P-05（2026-10-09）：复制失败必须可见（原先静默）。 */
  const [copyFailed, setCopyFailed] = useState(false);

  /** 时间戳展示：HH:MM（历史为后端 created_at，实时为本地时刻）；无值/非法 → 空串不渲染 */
  const fmtTime = (iso?: string | null) => {
    if (!iso) return "";
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const copyText = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setCopyFailed(false);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopyFailed(true);
      window.setTimeout(() => setCopyFailed(false), 2000);
    }
  };

  if (item.role === "user") {
    const time = fmtTime(item.created_at);
    return (
      <div className="msg user">
        <div className="avatar user"><Icons.smile size={14} /> </div>
        <div className="bubble-wrap">
          <div className="bubble">
            {item.images && item.images.length > 0 && (
              <div className="user-images">
                {item.images.map((src, i) => (
                  <img key={i} src={src} alt={`图片 ${i + 1}`} className="user-image" />
                ))}
              </div>
            )}
            {item.content}
          </div>
          <div className="msg-meta">
            {time && <span className="msg-time">{time}</span>}
            <span className="msg-actions">
              <button
                type="button"
                className="msg-action"
                title="复制"
                aria-label="复制消息"
                onClick={() => copyText(item.content)}
              >
                {copied ? <Icons.check size={15} /> : <Icons.copy size={15} />}
              </button>
            </span>
          </div>
        </div>
      </div>
    );
  }

  const m = item as AssistantMessage;
  // 推理过程默认**展开**（2026-09-28 用户要求）：此前折叠着，必须点开才看得见，
  // 等于"有展示但看不见"。用受控 `open`：用户手动收起后不会被下次渲染强行展开。
  // ★ 2026-10-23：默认**折叠**。展开时是一长串「步骤 N」+ 工具卡片，非常吵；
  //   需要排查时用户点开即可（内容不删，只收起）。
  const [thinkOpen, setThinkOpen] = useState(false);
  const hasThinking = m.thinking.length > 0;
  const hasReasoning = !!m.reasoning;
  // `retrieve` 是**内部检索手段**，不作为"工具"展示（2026-09-28 用户要求）：
  // 一次提问常触发 3~5 次 retrieve，逐张卡片会把推理区刷屏，而用户只关心
  // 最终回答与引用来源（引用由 CitationCard 独立呈现）。
  // ⚠️ 计数必须与展示**同源**（都用 visibleTools）——否则会出现"3 个工具"
  // 却一张卡片都看不到的自相矛盾。
  const visibleTools = visibleToolCalls(m.toolCalls);
  const hasTools = visibleTools.length > 0;
  const showThinking = hasThinking || hasReasoning || hasTools;
  const canRegen = Boolean(showRegenerate && onRegenerate && m.status === "complete");
  /** E1：只有「已完成 + 有服务端 id」的回答可打分 —— 流式中的 id 是临时负值，
   *  拿它打分会挂到不存在的消息上，刷新后回填必然对不上。 */
  const canRate = Boolean(onRate) && m.status === "complete" && Number(m.id) > 0;

  return (
    <div className={`msg assistant status-${m.status}`}>
      <div className="avatar assistant"><Icons.fileText size={22} /></div>
      <div className="bubble-wrap">
        <div className="bubble">
          {showThinking && (
            <details
              className="thinking"
              open={thinkOpen}
              onToggle={(e) => setThinkOpen(e.currentTarget.open)}
            >
              <summary>
                {/* 思考中指示（2026-09-17）：只在流式进行中渲染 —— 三个渐次跳动的小点。
                    纯装饰（aria-hidden）：状态变化本身由 .caret 与文本流表达，读屏不需要它。
                    完成即消失，不残留动画（无类名切换，靠渲染条件控制）。 */}
                {m.status === "streaming" && (
                  <span className="think-dots" aria-hidden="true">
                    <i />
                    <i />
                    <i />
                  </span>
                )}
                {hasTools
                  ? `推理过程（${hasThinking ? `${m.thinking.length} 步 · ` : ""}${visibleTools.length} 个工具）`
                  : `推理过程（${m.thinking.length}）`}
              </summary>
              {hasReasoning && (
                <div className="reasoning-text">
                  {m.reasoning}
                  {/* 推理链流式输出时光标跟到推理文本末尾；正文一开始流式就交回 answer 的 caret，
                      避免同时出现两个光标 */}
                  {m.status === "streaming" && !m.text && (
                    <span className="caret" aria-hidden="true" />
                  )}
                </div>
              )}
              {hasThinking && (
                <ul>
                  {m.thinking.map((t, i) => (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
              )}
              {hasTools && visibleTools.map((t) => <ToolCallCard key={t.id} tool={t} />)}
            </details>
          )}

          <div className="answer">
            {/* O-12 档 1：首字之前的进度提示（doc-bound 首轮实测首字中位 ≈4s）。
                显示条件严格：仍在流式 **且** 尚无正文；正文一到 reducer 即清除，
                此处再判 `!m.text` 是双保险，避免"进度提示与正文并列"的自相矛盾。 */}
            {m.status === "streaming" && !m.text && m.progress?.text && (
              <div className="msg-progress" role="status">{m.progress.text}</div>
            )}
            {m.text ? <MarkdownText text={m.text} /> : null}
            {m.status === "streaming" && <span className="caret" aria-hidden="true" />}
          </div>

          {m.status === "error" && m.error && <div className="error-box"><Icons.alert size={13} /> ️ {m.error.message}</div>}

          {m.status === "complete" && <CitationCard citations={m.citations} onOpen={onOpenCitation} activeDocIds={activeDocIds} />}

          {/* P1-2.1 回复逐条核验（verify 事件，只读展示） */}
          {m.status === "complete" && m.verify && <VerifyCard verify={m.verify} />}

          {/* 模型风险标记（2026-09-16）：本轮所用模型在绑定文档场景可能跳过检索 →
              该条回答上常驻「请核对引用」。刻意与 notice 区分：**不参与**「连续无效」计数，
              因此不会把正常轮次误升级为 loop_risk 强打断。 */}
          {m.modelRisk && (
            <div className="msg-model-risk" role="status">
              <Icons.alert size={13} />
              <span>该轮模型可能未基于文档检索作答（已知会跳过检索），引用与结论请务必核对原文。</span>
            </div>
          )}

          {/* 打断提醒：无效回复（缺文档 / 兜底 / 循环风险）醒目警示卡，跳脱正常回答流 */}
          {m.notice && (
            <div className={`msg-notice kind-${m.notice.kind}`} role="alert">
              <div className="msg-notice-title"><Icons.alert size={14} /> {m.notice.title}</div>
              <div className="msg-notice-message">{m.notice.message}</div>
              {m.notice.suggestion && (
                <div className="msg-notice-suggestion">{m.notice.suggestion}</div>
              )}
            </div>
          )}

          {/* ★ P1-a2（2026-10-03）：条渲染条件从 `m.done` 放宽到 **`m.done || m.ctx`**。
              历史消息（刷新 / 切会话后）走 `MessageOut`，**没有 done**，占用率改由 `m.ctx`
              从 `meta` 回填 —— 否则会出现"当轮看得到、刷新即消失"。 */}
          {m.status === "complete" && (m.done || m.ctx) && (
            <div className="done-meta">
              {m.done && (
                <>
                  {/* 2026-09-09 修复 N3：done 帧来自网络 JSON 无运行时校验，cost 为 null/undefined
                      时 toFixed 抛 TypeError 会崩掉整个消息列表。其余字段走模板字符串渲染不会崩。 */}
                  tokens: {m.done.total_tokens} · 成本: ¥
                  {typeof m.done.cost === "number" && Number.isFinite(m.done.cost) ? m.done.cost.toFixed(4) : "—"} ·
                  步数: {m.done.steps}
                </>
              )}
              {/* ★ P1-a（2026-10-03）：上下文占用率。
                  - 判据 = **单次峰值**（peak）÷ 单次上限（budget），**不是** total_tokens（整轮累计）
                    —— 熔断看的是单次；
                  - 警示线用后端下发的 `compact_trigger_tokens`（= 上限×0.7），
                    **不在前端乘 0.7**（复制派生逻辑 = 再造一把尺子）；
                  - 两条来源**合并**：实时取 `done`、历史取 `ctx`（前者优先）；
                  - ⚠️ 任一必需字段缺失 ⇒ **整块不渲染**（优雅降级，老后端/老数据不得报错）。 */}
              <CtxUsageBar
                peak={m.done?.peak_prompt_tokens ?? m.ctx?.peak}
                budget={m.done?.max_tokens_budget ?? m.ctx?.budget}
                warnAt={m.done?.compact_trigger_tokens ?? m.ctx?.warnAt}
                breakdown={m.done?.context_breakdown ?? m.ctx?.breakdown}
                placement="down"
                compact
              />
            </div>
          )}
        </div>

        <div className="msg-meta">
          {fmtTime(m.created_at) && <span className="msg-time">{fmtTime(m.created_at)}</span>}
          <span className="msg-actions">
            <button
              type="button"
              className="msg-action"
              title={copied ? "已复制" : copyFailed ? "复制失败（剪贴板不可用）" : "复制"}
              aria-label="复制消息"
              onClick={() => copyText(m.text)}
            >
              {copied ? <Icons.check size={15} /> : <Icons.copy size={15} />}
            </button>
            {/* E1（2026-09-21）：👍/👎 —— 评测样本的真实来源。
                只在「已完成且有服务端 id」的消息上显示：流式中的消息 id 是临时负值，
                拿它去打分会落到不存在的消息上（后端虽不强制外键，但回填会对不上）。 */}
            {canRate && (
              <>
                <button
                  type="button"
                  className={`msg-action${myRating === 1 ? " is-active-good" : ""}`}
                  title={myRating === 1 ? "已标记为有帮助（点击取消记录）" : "有帮助"}
                  aria-label="这条回答有帮助"
                  aria-pressed={myRating === 1}
                  onClick={() => onRate?.(Number(m.id), 1)}
                >
                  <Icons.thumbUp size={15} />
                </button>
                <button
                  type="button"
                  className={`msg-action${myRating === -1 ? " is-active-bad" : ""}`}
                  title={myRating === -1 ? "已标记为没帮助（点击改为好评）" : "没帮助"}
                  aria-label="这条回答没帮助"
                  aria-pressed={myRating === -1}
                  onClick={() => onRate?.(Number(m.id), -1)}
                >
                  <Icons.thumbDown size={15} />
                </button>
              </>
            )}
            {canRegen && (
              <button
                type="button"
                className="msg-action"
                title="重新生成"
                aria-label="重新生成回答"
                onClick={() => onRegenerate?.()}
              >
                <Icons.refresh size={15} />
              </button>
            )}
          </span>
        </div>
      </div>
    </div>
  );
});
