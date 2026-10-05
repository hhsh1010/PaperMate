import { useCallback, useEffect, useState } from "react";

/**
 * Ctrl+鼠标滚轮缩放（预览面板 / 改稿工作台共用）。
 *
 * ⚠️ 两个关键点：
 * 1. React 的 onWheel 是 passive 监听，事件里 preventDefault 无效——
 *    必须用原生 addEventListener(..., { passive: false }) 才能阻止浏览器
 *    「Ctrl+滚轮 = 页面缩放」的默认行为；
 * 2. 目标元素可能是**条件渲染**（如预览面板默认收起）——普通 ref + effect 只在
 *    mount 跑一次，元素后挂载时监听永远不会挂上。这里用 **callback ref**：
 *    元素挂载/卸载本身触发 state 变化 → effect 依赖元素引用重跑。
 */
export function useCtrlWheelZoom<T extends HTMLElement = HTMLElement>(min = 50, max = 300, step = 10) {
  const [el, setEl] = useState<T | null>(null);
  const [zoom, setZoom] = useState(100);

  /** callback ref：直接当 ref={...} 用，元素出现/消失自动触发 effect */
  const ref = useCallback((node: T | null) => setEl(node), []);

  useEffect(() => {
    if (!el) return;
    const handler = (e: WheelEvent) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      setZoom((z) => Math.min(max, Math.max(min, z - Math.sign(e.deltaY) * step)));
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [el, min, max, step]);

  return { ref, zoom, setZoom };
}
