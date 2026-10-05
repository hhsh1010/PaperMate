import { api } from "../api/client";
import type { Citation } from "../lib/types";
import { Icons } from "../lib/icons";

/** ER-22：引用跳转阅读器。点击引用卡片打开——展示引用原文，并内嵌 PDF 跳到对应页。*/
export function PdfReader({
  citation,
  onClose,
}: {
  citation: Citation;
  onClose: () => void;
}) {
  const docId = typeof citation.id === "number" ? citation.id : undefined;
  // citation.id 实际为 chunk_id（doc-{doc_id}-c{seq}）；从 id 解出 doc_id 更稳妥
  const fromId = typeof citation.id === "string" ? citation.id : "";
  // N8 修复（2026-09-10）：解出的 doc_id 必须校验为**有限正整数**——原先 `Number(...)`
  // 对 "doc-abc-c0" 得 NaN、对 "doc--c0" 得 0，而 `!= null` 对二者都为真 →
  // 渲染 `documentFileUrl(NaN)` 破损 iframe，且「无法解析文档 id」兜底分支永不显示。
  const parsedRaw = docId ?? (fromId.startsWith("doc-") ? Number(fromId.split("-")[1]) : undefined);
  const parsedDocId =
    typeof parsedRaw === "number" && Number.isFinite(parsedRaw) && parsedRaw > 0 ? parsedRaw : undefined;
  const page = citation.page ?? 1;
  const fileUrl = parsedDocId != null ? `${api.documentFileUrl(parsedDocId)}#page=${page}` : "";

  return (
    <div className="modal-mask" onClick={onClose} aria-modal="true" role="dialog">
      <div className="modal reader-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="reader-title">引用溯源</div>
            <div className="reader-sub">
              {typeof citation.id !== "undefined" ? `[${citation.id}]` : ""}
              {parsedDocId != null ? ` · 文档 #${parsedDocId}` : ""}
              {page ? ` · 第 ${page} 页` : ""}
            </div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="关闭"><Icons.x size={14} /></button>
        </div>
        <div className="reader-text">
          <div className="reader-label">引用原文</div>
          <div className="reader-quote">{citation.text || "（无原文）"}</div>
        </div>
        {parsedDocId != null && (
          <div className="reader-frame-wrap">
            <iframe
              className="reader-frame"
              src={fileUrl}
              title={`文档 ${parsedDocId} 第 ${page} 页`}
            />
            <div className="reader-hint">若浏览器未自动跳到目标页，可手动滚动到「第 {page} 页」。</div>
          </div>
        )}
        {parsedDocId == null && (
          <div className="reader-hint">无法解析文档 id，仅展示引用文本。</div>
        )}
      </div>
    </div>
  );
}
