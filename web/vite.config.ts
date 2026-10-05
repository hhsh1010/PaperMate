import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// 开发期把 /api 反向代理到本地 FastAPI（:8000），避免 CORS；
// 生产期由 nginx 统一转发（见 deploy/nginx.conf）。
export default defineConfig({
  plugins: [react()],
  build: {
    // T-08（2026-09-29）：pdfjs 单独成 chunk。
    // ⚠️ **只手工拆 pdfjs，不手工拆 docx-preview** —— 实测（2026-09-29）：
    // 手工拆 docx-preview 反而会让**入口静态 import 它**（Rollup 为保持该手工 chunk 的
    // 求值顺序而注入 `import{g as Ef}from"./docx-preview-*.js"`，而该导出**从未被使用**）
    // ⇒ `index.html` 照旧 `modulepreload` 174 kB ⇒ **首屏字节一点没降**。
    // docx-preview 被两个动态 chunk（预览面板 / 改稿工作台）共享，Rollup **自动分包**即可。
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return;
          if (id.includes("pdfjs-dist")) return "pdfjs";
          return;
        },
        // 自动分包会把"两个动态 chunk 的共享依赖"并成一个 chunk，并按其中**首个模块**命名 ——
        // 实测它被命名为 `useCtrlWheelZoom-*.js`（而该 hook 只有 0.4 kB，chunk 实为 docx-preview）
        // ⇒ 名字**误导排查**。这里按**内容**改名（纯命名，不改变分包结果）。
        chunkFileNames(chunk) {
          const ids = chunk.moduleIds ?? [];
          if (ids.some((id) => id.includes("docx-preview"))) return "assets/docx-preview-[hash].js";
          if (ids.some((id) => id.includes("pdfjs-dist"))) return "assets/pdfjs-[hash].js";
          return "assets/[name]-[hash].js";
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
