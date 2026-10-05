import { useState, type ReactNode } from "react";
import { Icons } from "../lib/icons";

/**
 * 可折叠分区卡片（2026-08-29 收纳重排 · 2026-09-01 WorkBuddy 风格升级 · 2026-09-03 默认展开）
 *
 * 设置页各分区与「长期记忆管理」面板共用：点击头部展开 / 收起，箭头随状态旋转，
 * 头部可挂徽标（如「已配置」「4/17」「12 条」）。默认展开（用户要求：详细设置直接可见）；
 * 需要默认收起的调用方显式传 defaultOpen={false}。
 *
 * 可选 onClose 回调：传入则在头部右侧渲染关闭 X（WorkBuddy 风格），用于「按 Esc/遮罩 之外的精细关闭」。
 */
export function SectionCard({
  title, subtitle, defaultOpen = true, badge, onClose, children,
}: {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  badge?: string;
  onClose?: () => void;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`settings-card collapsible fade-in ${open ? "open" : ""}`}>
      <button
        type="button"
        className="settings-card-head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="sc-head-text">
          <h3>
            {title}
            {badge ? <span className="sc-badge">{badge}</span> : null}
          </h3>
          {subtitle ? <p className="muted">{subtitle}</p> : null}
        </span>
        <span className="sc-chevron" aria-hidden="true"><Icons.chevronDown size={14} /></span>
      </button>
      {onClose && (
        <button
          type="button"
          className="wb-section-close"
          onClick={onClose}
          aria-label="关闭此分区"
          title="关闭此分区"
          onMouseDown={(e) => e.stopPropagation()}
        >
          <Icons.x size={14} />
        </button>
      )}
      {open && <div className="settings-body">{children}</div>}
    </section>
  );
}
