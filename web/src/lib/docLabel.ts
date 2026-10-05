/** 中文序号：文档一 / 文档二 … 超过 10 用「文档 11」。
 *  2026-09-09 从 DocumentPreviewPanel 抽出为独立 lib 模块（N1 配套修复）：
 *  CitationCard 只需要这个纯函数，原先却要 import 整个 711 行重组件 →
 *  传递性引入 pdfjs-dist（其 canvas.js 顶层引用 DOMMatrix）→ node 测试环境
 *  直接 ReferenceError，组件层单测无法收集。纯函数落 lib 层后组件测试可独立运行。 */
export function docLabel(index: number): string {
  const cn = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"];
  if (index >= 0 && index < cn.length) return `文档${cn[index]}`;
  return `文档 ${index + 1}`;
}
