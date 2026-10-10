import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { installDemoApi } from "./demo/mock";
import "./styles.css";

// ★ 演示模式（`npm run build:demo` ⇒ VITE_DEMO_MODE=1）：
//   在**渲染之前**接管 fetch —— 所有 /api/v1/* 由内置样例数据应答，
//   于是 GitHub Pages 上的那个静态站点就能"点开即用"，且**不需要任何后端与密钥**。
//   同时把开关写进 <html data-demo-mode="on|off">：构建产物自检与端到端验证据此判断，
//   避免"以为开了演示模式、其实发上去的是真实构建"这类静默错误。
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "1";
document.documentElement.dataset.demoMode = DEMO_MODE ? "on" : "off";
if (DEMO_MODE) installDemoApi();

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
