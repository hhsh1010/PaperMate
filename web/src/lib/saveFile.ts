/** 统一「保存文件」入口（P-04 / P-05，2026-10-09）。
 *
 * ## 为什么需要它（实测根因）
 * 桌面端（Tauri / WebView2）里浏览器的 `<a download>` 会**静默落盘到系统下载文件夹**、
 * 界面零反馈 —— 用户合理地判断"功能无效"，于是连点 12 次，产生 12 个同名副本（实测）。
 * 统一改成：
 *   · **Tauri**：弹原生「另存为」（`plugin:dialog|save`）→ 把字节**交给后端写盘** → 返回最终路径；
 *   · **浏览器**：回退 `<a download>`（开发态照旧可用，行为不变）。
 *
 * ## 两条纪律
 * 1. 用户**取消**另存为 ⇒ `{kind:"cancelled"}`，调用方**不要**报错（这是正常操作，不是失败）；
 * 2. 写盘失败 ⇒ **抛出**，调用方必须给出可见错误（不许静默 —— 与"假装有进度"同族）。
 *
 * ## 为什么走"后端写盘"而不是前端 fs 插件
 * Tauri 的 fs 插件对"对话框选中的任意路径"需要在 Rust 侧动态授权 scope（`FsExt::allow_file`），
 * 那是一次 Rust 改动 + 重编译；而后端本来就拥有这些文件、写盘是它既有能力
 * （`POST /files/write` 只写不读、路径必须来自用户自己的另存为对话框）。
 */

import { api } from "../api/client";

export type SaveOutcome =
  | { kind: "saved"; path: string }
  | { kind: "cancelled" }
  | { kind: "fallback" }; // 浏览器回退：已触发 <a download>，没有可回报的路径

/** 是否运行在 Tauri 壳里（与 `lib/titlebar.ts` 同一判据：核心 IPC 对象存在）。 */
export function isTauri(): boolean {
  if (typeof window === "undefined") return false;
  const w = window as unknown as { __TAURI_INTERNALS__?: { invoke?: unknown } };
  return typeof w.__TAURI_INTERNALS__?.invoke === "function";
}

/** 字节 → base64（**纯函数**，分块避免大文件爆栈）。 */
export function bytesToBase64(bytes: Uint8Array): string {
  const CHUNK = 0x8000;
  let bin = "";
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

/** 依据文件名推断保存对话框的**类型过滤**（2026-10-10 修复"保存下来不是 docx"）。
 *  ★ 不传 filters 时 Windows 对话框是「所有文件」：扩展名可被改掉/丢失 ⇒
 *    双击落进文本编辑器显示二进制"乱码"（用户实测）。锁定类型后由对话框强制补扩展名。 */
function filtersFor(fileName: string): { name: string; extensions: string[] }[] | undefined {
  const m = /\.([A-Za-z0-9]+)$/.exec(fileName);
  return m ? [{ name: `${m[1].toUpperCase()} 文件`, extensions: [m[1].toLowerCase()] }] : undefined;
}

/** 弹原生「另存为」；返回 `null` = 用户取消。 */
async function pickSavePath(fileName: string): Promise<string | null> {
  const w = window as unknown as {
    __TAURI_INTERNALS__: { invoke: (cmd: string, args?: unknown) => Promise<unknown> };
  };
  const picked = await w.__TAURI_INTERNALS__.invoke("plugin:dialog|save", {
    options: { title: "另存为", defaultPath: fileName, filters: filtersFor(fileName) },
  });
  return typeof picked === "string" && picked.length > 0 ? picked : null;
}

/** ★ 扩展名保险（2026-10-10）：对话框返回的路径若丢了扩展名（用户手滑/系统差异），
 *  按预期后缀补齐 —— 否则文件会以无扩展名/错误扩展名落盘，双击进文本编辑器显示乱码。 */
function ensureExt(dest: string, ext: string): string {
  return dest.toLowerCase().endsWith(`.${ext.toLowerCase()}`) ? dest : `${dest}.${ext}`;
}

/** 浏览器回退：`<a download>`（保持开发态可用）。 */
export function downloadViaBrowser(fileName: string, blob: Blob, docId?: number, mode?: string): void {
  if (docId !== undefined) {
    // 文档导出走既有同源下载链接（Content-Disposition 决定文件名）
    window.location.href = `/api/v1/documents/${docId}/download?mode=${mode ?? "tracked"}`;
    return;
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** 保存**前端生成**的字节（CSV / JSONL / Markdown…）。 */
export async function saveBlob(fileName: string, blob: Blob): Promise<SaveOutcome> {
  if (!isTauri()) {
    downloadViaBrowser(fileName, blob);
    return { kind: "fallback" };
  }
  const picked = await pickSavePath(fileName);
  if (!picked) return { kind: "cancelled" };
  const m = /\.([A-Za-z0-9]+)$/.exec(fileName);
  const dest = m ? ensureExt(picked, m[1]) : picked;
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const out = await api.writeLocalFile(dest, bytesToBase64(bytes));
  return { kind: "saved", path: out.path };
}

/** 保存**后端产出**的文档导出件（修订版 / 干净版 docx）。 */
export async function saveDocxExport(
  docId: number,
  mode: "tracked" | "clean",
  fileName: string,
): Promise<SaveOutcome> {
  if (!isTauri()) {
    downloadViaBrowser(fileName, new Blob(), docId, mode);
    return { kind: "fallback" };
  }
  const picked = await pickSavePath(fileName);
  if (!picked) return { kind: "cancelled" };
  const dest = ensureExt(picked, "docx");
  const out = await api.exportDocumentTo(docId, dest, mode);
  return { kind: "saved", path: out.path };
}

/** 把 SaveOutcome 变成一句可展示的文案（`null` = 不提示，例如浏览器回退）。 */
export function saveOutcomeMessage(out: SaveOutcome, lang: "zh" | "en" = "zh"): string | null {
  if (out.kind === "cancelled") return lang === "en" ? null : null; // 取消不打扰
  if (out.kind === "fallback") {
    return lang === "en" ? "Download started (saved to your Downloads folder)." : "已开始下载（保存到「下载」文件夹）。";
  }
  return lang === "en" ? `Saved to ${out.path}` : `已保存到 ${out.path}`;
}
