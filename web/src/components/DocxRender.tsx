import { useEffect, useRef, useState } from "react";
import { renderAsync } from "docx-preview";
import { api } from "../api/client";
import { toErrorMessage } from "../lib/error";

/**
 * docx 高保真渲染（docx-preview）：预览面板 / 改稿工作台共用（P1）。
 *
 * 渲染原文件的章节样式、表格、图片（useBase64URL 内嵌）；纯只读。
 * 渲染失败（损坏/超版本特性）→ 提供「以纯文本查看」兜底（复用解析文本接口）。
 *
 * **自适应宽度**：docx-preview 按 A4 实宽渲染，窄容器（预览面板）会横向溢出——
 * 渲染完成后测量内容宽/容器宽，自动回调初始缩放（只缩小不放大，≤100%）。
 */
export function DocxRender({
  docId,
  zoom = 100,
  onZoomFit,
}: {
  docId: number;
  zoom?: number;
  /** 渲染完成后的自适应缩放回调（父级同步到工具条显示） */
  onZoomFit?: (zoom: number) => void;
}) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<"loading" | "ok" | "error">("loading");
  const [errMsg, setErrMsg] = useState("");
  const [fallbackText, setFallbackText] = useState<string | null>(null);
  // 回调走 ref：父级传内联箭头函数时不能让它进 effect 依赖，否则每次父级渲染都重渲染文档
  const onZoomFitRef = useRef(onZoomFit);
  onZoomFitRef.current = onZoomFit;
  // N4 修复（2026-09-10）：跟踪当前 docId——「以纯文本查看」的 fetch 是异步的，
  // 切换文档后旧响应 resolve 会把**旧文档文本**渲染到新文档上（且无法察觉）。
  // 不能用闭包里的 docId 判定（它是点击时的旧值），必须用 ref 比对当前值。
  const docIdRef = useRef(docId);
  docIdRef.current = docId;

  useEffect(() => {
    let cancelled = false;
    const container = bodyRef.current;
    if (!container) return;
    setState("loading");
    setFallbackText(null);
    container.innerHTML = "";
    (async () => {
      try {
        const res = await fetch(`/api/v1/documents/${docId}/file`);
        if (!res.ok) throw new Error(`原文件获取失败（HTTP ${res.status}）`);
        const blob = await res.blob();
        if (cancelled) return;
        await renderAsync(blob, container, undefined, {
          className: "docx",
          inWrapper: true,
          ignoreHeight: true,
          useBase64URL: true, // 图片内嵌显示
        });
        if (cancelled) return;
        setState("ok");
        // 自适应宽度：内容宽超出容器则整体缩小至恰好放下（只缩不放），下一帧再量（等布局生效）
        requestAnimationFrame(() => {
          if (cancelled) return;
          const content = container.querySelector(".docx-wrapper") as HTMLElement | null;
          const wrap = wrapRef.current;
          if (!content || !wrap) return;
          const cw = content.scrollWidth;
          const aw = wrap.clientWidth - 24; // 扣除容器内边距
          if (cw > aw && cw > 0) {
            onZoomFitRef.current?.(Math.max(50, Math.min(100, Math.floor((aw / cw) * 100))));
          }
        });
      } catch (e: unknown) {
        if (!cancelled) {
          setErrMsg(toErrorMessage(e) || "渲染失败");
          setState("error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [docId]);

  /** 渲染失败兜底：加载解析纯文本（与 TextDocView 同源）；N4：按发起时的 docId 守卫过期响应 */
  const loadFallback = () => {
    const requestedDocId = docId;
    fetch(api.documentTextUrl(docId))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => {
        if (docIdRef.current !== requestedDocId) return; // 已切换文档：丢弃旧文本（N4）
        setFallbackText((j?.text as string) ?? "（无可显示文本）");
      })
      .catch(() => {
        if (docIdRef.current !== requestedDocId) return;
        setFallbackText("（文本加载失败）");
      });
  };

  if (state === "error" && fallbackText === null) {
    return (
      <div className="docx-render">
        <div className="preview-error">
          {errMsg}
          <button type="button" className="batch-btn ghost" onClick={loadFallback}>
            以纯文本查看
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="docx-render" ref={wrapRef}>
      {state === "loading" && <div className="rw-loading"><span className="spinner" aria-hidden="true" />渲染文档中…</div>}
      {fallbackText !== null ? (
        <div className="preview-paper md">
          <pre className="preview-md-text">{fallbackText}</pre>
        </div>
      ) : (
        <div
          ref={bodyRef}
          className="docx-render-body"
          style={{ zoom: zoom / 100 } as React.CSSProperties}
        />
      )}
    </div>
  );
}
