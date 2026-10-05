import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

/**
 * 全局滚动条自动显隐：滚动容器滚动时加 .is-scrolling，停止 900ms 后移除。
 * CSS 据此切换 thumb 颜色（默认透明占位，防内容跳动）；捕获阶段监听以覆盖
 * 不冒泡的 scroll 事件；WeakMap 存每元素空闲计时器，多容器互不干扰。
 */
const scrollIdleTimers = new WeakMap<Element, number>();
document.addEventListener(
  "scroll",
  (e) => {
    const target = e.target;
    const el = target instanceof Element ? target : document.scrollingElement;
    if (!el) return;
    el.classList.add("is-scrolling");
    const prev = scrollIdleTimers.get(el);
    if (prev !== undefined) window.clearTimeout(prev);
    scrollIdleTimers.set(
      el,
      window.setTimeout(() => el.classList.remove("is-scrolling"), 900),
    );
  },
  true,
);

const el = document.getElementById("root");
if (!el) throw new Error("root element not found");

createRoot(el).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
