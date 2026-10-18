import { useEffect, useRef, useState } from "react";
import { Icons } from "../lib/icons";
import { buildCtxPanel, fmtTokens } from "../lib/ctxBreakdown";
import { ringSegments } from "../lib/ctxRing";
import { barSegments } from "../lib/ctxBar";

/** composer 小圆环按钮的半径（viewBox 32×32，与工具条控件等高）。 */
const RING_BTN_R = 13;

/**
 * 「上下文用量」面板 + 悬浮条（2026-10-03，P2）。
 *
 * 参考 WorkBuddy 的呈现：
 * - **悬浮条**：`38.0% · 380.4K / 1000.0K 上下文已使用`（常驻、可点击）
 * - **面板**：标题 + 大号百分比 + 「已使用 X / Y」+ 多色**分段进度条** + 分项列表（色点 + 百分比）
 *
 * ★ 数据两条口径（见 `lib/ctxBreakdown.ts` 的说明）：
 *   总占用率用**真实**峰值 ÷ 上限；分项百分比是**估算**内部的相对比例。
 *   刻意不混用 —— 混用会给出互相矛盾的数字。
 *
 * ⚠️ 数据缺失 ⇒ 整个组件**不渲染**（优雅降级，老后端/老数据不得报错）。
 */
export function CtxUsageBar({
  peak,
  budget,
  warnAt,
  breakdown,
  placement = "up",
  compact = false,
  variant = "bar",
}: {
  peak?: number;
  budget?: number;
  warnAt?: number;
  breakdown?: Record<string, number>;
  /** 面板弹出方向：composer 在底部 ⇒ `up`；消息列表在中间 ⇒ `down`。
   *  ★ 必须可配置：写死方向会在另一种场景下把面板**弹出屏幕外**。 */
  placement?: "up" | "down";
  /** 紧凑模式：只显示百分比（放进消息 footer 行，避免挤压 tokens/成本/步数）。 */
  compact?: boolean;
  /** 触发形态：`bar` = 带文字的横条；`ring` = **小圆环**（composer 用，省横向空间）。 */
  variant?: "bar" | "ring";
}) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  const data = buildCtxPanel(peak, budget, warnAt, breakdown);

  // 点外部 / Esc 关闭（与项目其它浮层交互一致）
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (anchorRef.current && !anchorRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!data) {
    // ★★ 2026-10-03（用户报"不同对话下方功能栏区域不一样"的**根因之一**）：
    // 原实现在无数据时 `return null` ⇒ 控件**整个消失** ⇒ 工具条里后续控件左移
    //（新会话 / 尚无助手消息 / 技能轮 都会命中）⇒ 不同对话布局不一致。
    // ⇒ `ring` 模式**始终占位**（渲染空环），保持工具条布局稳定；
    //   `bar` 模式（消息 footer 内）仍不占位（那里本来就有其它 metadata 撑着）。
    if (variant === "ring") {
      return (
        <span className="ctx-usage-anchor up" ref={anchorRef}>
          <button
            type="button"
            className="ctx-ring-btn empty"
            disabled
            aria-label="暂无上下文用量数据"
            title="本轮暂无上下文数据（尚未产生助手回复）"
          >
            <svg className="ctx-ring" viewBox="0 0 32 32" width="20" height="20" aria-hidden="true">
              <circle className="ctx-ring-track" cx="16" cy="16" r={RING_BTN_R} />
            </svg>
          </button>
        </span>
      );
    }
    return null;
  }

  // composer 的小圆环用**另一个半径**（32 viewBox），几何必须各自算
  const { segments: btnSegs } = ringSegments(data.buckets, data.pct, RING_BTN_R, 1);
  // 面板里的**横条**（2026-10-03 用户要求换回横条）：底槽 + 多色分段
  const barSegs = barSegments(data.buckets, data.pct);

  return (
    <span className={`ctx-usage-anchor ${placement}`} ref={anchorRef}>
      {/* ★ 2026-10-03（用户："把下方功能栏内的上下文模块改成圆形，点击再展开"）：
          composer 里改用一个**小圆环按钮**（不再占一整条文字宽度），
          tooltip 给出完整数值、点击展开同一个面板。 */}
      {variant === "ring" ? (
        <button
          type="button"
          className={`ctx-ring-btn ${data.warn ? "warn" : ""} ${open ? "on" : ""}`}
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={`上下文已使用 ${data.pct.toFixed(1)}%`}
          title={`上下文已使用 ${data.pct.toFixed(1)}% · ${fmtTokens(data.used)} / ${fmtTokens(data.total)}`}
        >
          <svg className="ctx-ring" viewBox="0 0 32 32" width="20" height="20" aria-hidden="true">
            <circle className="ctx-ring-track" cx="16" cy="16" r={RING_BTN_R} />
            {btnSegs.map((s) => (
              <circle
                key={s.key}
                className="ctx-ring-seg"
                cx="16"
                cy="16"
                r={RING_BTN_R}
                stroke={s.color}
                strokeDasharray={s.dasharray}
                strokeDashoffset={s.dashoffset}
              />
            ))}
          </svg>
        </button>
      ) : (
        <button
          type="button"
          className={`ctx-usage-bar ${data.warn ? "warn" : ""} ${open ? "on" : ""} ${compact ? "compact" : ""}`}
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label="查看上下文用量"
          title="查看上下文用量"
        >
          <span className="ctx-usage-bar-value">{data.pct.toFixed(1)}%</span>
          {/* compact（消息 footer 内）：只留百分比，避免把 tokens/成本/步数挤出去 */}
          {!compact && (
            <>
              <span className="ctx-usage-bar-sep">·</span>
              <span className="ctx-usage-bar-num">
                {fmtTokens(data.used)} / {fmtTokens(data.total)}
              </span>
              <span className="ctx-usage-bar-label">上下文已使用</span>
            </>
          )}
        </button>
      )}

      <div className={`collapse${open ? " open" : ""}`}>
        <div className="collapse-inner">
        <div className="ctx-panel" role="dialog" aria-label="上下文用量">
          <div className="ctx-panel-head">
            <span className="ctx-panel-title">上下文用量</span>
            <button
              type="button"
              className="ctx-panel-close"
              onClick={() => setOpen(false)}
              aria-label="关闭上下文用量面板"
              title="关闭"
            >
              <Icons.x size={13} />
            </button>
          </div>

          {/* 顶部：百分比 + 已用量（与参考图一致：数值在上，条在下） */}
          <div className="ctx-panel-summary">
            <span className={`ctx-panel-pct ${data.warn ? "warn" : ""}`}>
              {data.pct.toFixed(1)}%
            </span>
            <span className="ctx-panel-used">
              已使用 {fmtTokens(data.used)}/ {fmtTokens(data.total)}
            </span>
          </div>

          {/* ★ 2026-10-03（用户："面板换成横条统计"）：**底槽 + 多色分段**。
              底槽 = 100%（总容量），分段宽度 = 分项占比 × 占用率（`barSegments` 已算好，
              是"占整条"的百分比 ⇒ 这里**不能再**套一层按占用率缩放的容器，否则双重折算）。
              ⚠️ 底槽必须可见（深色）——否则"还剩多少"就看不出来了。 */}
          <div className="ctx-panel-bar" aria-hidden="true">
            {barSegs.map((sg) => (
              <span
                key={sg.key}
                className="ctx-panel-bar-seg"
                style={{ width: `${sg.widthPct}%`, background: sg.color }}
              />
            ))}
          </div>

          {data.buckets.length > 0 ? (
            <ul className="ctx-panel-list">
              {data.buckets.map((b) => (
                <li key={b.key} className="ctx-panel-item">
                  <span className="ctx-panel-dot" style={{ background: b.color }} aria-hidden="true" />
                  <span className="ctx-panel-name">{b.label}</span>
                  {/* ★ 显示 token 数（不只百分比）：分项之和 = 上面的「已使用」，
                      用户一眼能核对"加起来是否等于总数"（等比缩放后自洽）。 */}
                  <span className="ctx-panel-val">
                    {fmtTokens(b.tokens)} · {b.pct.toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            /* 分项缺失（老数据 / 技能路径）⇒ 只给真实占用率，**不编造**分项 */
            <div className="ctx-panel-empty">本轮无分项统计（仅显示总占用率）</div>
          )}

          {/* ⚠️ 文案必须如实说明口径：分项**不是**厂商给的，而是"估算比例 × 真实总量"。
              厂商不返回分项明细（硬约束）⇒ 不能写"分项为真实值"。 */}
          <div className="ctx-panel-foot">
            总量取自厂商真实 token；分项按本地估算的**比例**分配总量（厂商不返回分项明细）
          </div>
        </div>
        </div>
      </div>
    </span>
  );
}
