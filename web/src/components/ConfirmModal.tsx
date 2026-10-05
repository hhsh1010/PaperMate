import { useEffect } from "react";
import type { ReactNode } from "react";
import { Icons } from "../lib/icons";

/**
 * 通用确认弹窗（自绘，WorkBuddy 风格）：
 * - 遮罩点击 / Esc 关闭
 * - 玻璃拟态卡片 + 底部操作区（由 footer 提供）
 * - 无障碍：role="dialog" / aria-modal / aria-label
 */
export function ConfirmModal({
  open,
  title,
  onClose,
  children,
  footer,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children?: ReactNode;
  footer?: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose} aria-label="关闭"><Icons.x size={14} /></button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-footer">{footer}</div>}
      </div>
    </div>
  );
}
