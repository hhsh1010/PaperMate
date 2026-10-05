/// <reference types="vite/client" />

// Vite 静态资源 ?url 导入类型（pdf.worker.mjs?url 等）
declare module "*?url" {
  const src: string;
  export default src;
}
