import { useEffect, useState } from "react";
import {
  closeWindow,
  isMaximized,
  isTauri,
  minimize,
  toggleMaximize,
} from "../lib/titlebar";

/**
 * 独立顶条（A1）：无边框窗口（decorations:false）下的应用内标题栏。
 *
 * - 整条背景取 `var(--bg)`，与页面同色 ⇒ 视觉上「融入」整个 app，不出现突兀的功能栏。
 * - 左侧 `.titlebar-drag` 为拖拽区（`data-tauri-drag-region`，tauri 注入的 drag.js
 *   自动处理拖拽；双击该区域 = 最大化/还原，无需额外 JS）。
 * - 右侧 最小化 / 最大化 / 关闭 三个幽灵按钮，默认透明、hover 整条才显形。
 * - 仅在 Tauri 环境下渲染；浏览器 / dev 模式返回 null（无边框概念）。
 */
export default function TitleBar() {
  const [max, setMax] = useState(false);

  useEffect(() => {
    let alive = true;
    if (isTauri()) {
      isMaximized()
        .then((m) => {
          if (alive) setMax(m);
        })
        .catch(() => {});
    }
    return () => {
      alive = false;
    };
  }, []);

  if (!isTauri()) return null;

  return (
    <div className="titlebar" role="banner" aria-label="窗口控制">
      <div className="titlebar-drag" data-tauri-drag-region>
        <span className="titlebar-brand">PaperMate</span>
      </div>
      <div className="titlebar-actions">
        <button
          type="button"
          className="tb-btn"
          aria-label="最小化"
          title="最小化"
          onClick={() => minimize().catch((e) => console.error("[titlebar] minimize failed", e))}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <line x1="2" y1="9" x2="10" y2="9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
        </button>
        <button
          type="button"
          className="tb-btn"
          aria-label={max ? "还原" : "最大化"}
          title={max ? "还原" : "最大化"}
          onClick={() => {
            void toggleMaximize().catch((e) => console.error("[titlebar] toggleMaximize failed", e));
            setMax((v) => !v);
          }}
        >
          {max ? (
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <rect x="3.2" y="2.6" width="5.6" height="5.6" fill="none" stroke="currentColor" strokeWidth="1.1" />
              <rect x="2.4" y="3.4" width="5.6" height="5.6" fill="var(--bg)" stroke="currentColor" strokeWidth="1.1" />
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <rect x="2.4" y="2.4" width="7.2" height="7.2" fill="none" stroke="currentColor" strokeWidth="1.1" />
            </svg>
          )}
        </button>
        <button
          type="button"
          className="tb-btn tb-close"
          aria-label="关闭"
          title="关闭"
          onClick={() => closeWindow().catch((e) => console.error("[titlebar] close failed", e))}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <line x1="2.6" y1="2.6" x2="9.4" y2="9.4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            <line x1="9.4" y1="2.6" x2="2.6" y2="9.4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
