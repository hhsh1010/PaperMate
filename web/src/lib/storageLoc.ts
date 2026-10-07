/**
 * 存储位置 UI 的纯逻辑（2026-10-07）
 *
 * ★ 本仓库**没有** testing-library（见 MEMORY 的工程纪律）：组件测试只能守结构、
 *   守不住逻辑 ⇒ 逻辑一律抽到这里做**行为测试**，组件只负责渲染。
 */

import type { StorageOut, StorageUpdateIn } from "./types";

export interface StorageInputResult {
  ok: boolean;
  /** ok 为 true 时给出可直接提交的载荷 */
  value?: StorageUpdateIn;
  error?: string;
}

/** 去掉尾部分隔符，但保留盘符根（`D:\` 不能变成 `D:`）。 */
export function normalizeRootPath(raw: string): string {
  const s = raw.trim();
  if (s.length <= 3 && /^[A-Za-z]:[\\/]?$/.test(s)) return `${s.slice(0, 2)}\\`;
  return s.replace(/[\\/]+$/, "");
}

/** 判断是否是"看起来像绝对路径"的写法。 */
function looksAbsolute(s: string): boolean {
  return /^[A-Za-z]:[\\/]?/.test(s) || s.startsWith("\\\\") || s.startsWith("//");
}

/**
 * 发送前的**本地**校验。
 *
 * ★ 这里只做"明显错"的拦截以给即时反馈；**权威校验在后端**（`validate_root`）
 *   与启动器（`lib.rs::validate_data_root`，重启时）—— 三处规则有意分层，
 *   凡两边都查的项必须口径一致。
 */
export function buildStorageUpdateInput(raw: string, migrate: boolean): StorageInputResult {
  const root = normalizeRootPath(raw);
  if (!root) return { ok: false, error: "请填写目标路径（例如 D:\\PaperMate）" };
  if (!looksAbsolute(root)) {
    return { ok: false, error: "请填写完整路径：需要盘符开头，例如 D:\\PaperMate" };
  }
  return { ok: true, value: { data_root: root, migrate } };
}

/** 是否有"已写入但未生效"的请求 ⇒ 设置页要显示"重启后生效"。 */
export function storageNeedsRestart(out: StorageOut | null | undefined): boolean {
  return Boolean(out?.pending_root && out.pending_root.trim());
}

/**
 * 待生效请求的说明文案；没有请求时返回 null。
 *
 * ★ 刻意写清两件事：**原位置数据不会被删**、**venv 会重建**（不能搬）。
 *   这两点是用户最担心的，不写清就会不敢点。
 */
export function storagePendingBanner(out: StorageOut | null | undefined, lang: "zh" | "en"): string | null {
  if (!storageNeedsRestart(out) || !out) return null;
  const target = out.pending_root ?? "";
  if (lang === "en") {
    return `A change to "${target}" is pending. It takes effect after you restart the app. `
      + (out.migrate
        ? "Existing data will be copied over; the copy at the old location is kept, and venv is rebuilt."
        : "The app will start with an empty data directory at the new location; the old data is left untouched.");
  }
  return `已申请把数据位置改为「${target}」，**重启应用后生效**。`
    + (out.migrate
      ? "届时会搬迁既有数据（data/ · huggingface/ · uv-cache/），**原位置的副本会保留不删**；venv 会在新位置重建。"
      : "届时将在新位置以**空数据目录**启动，原位置的数据保留不动。");
}

/** 四个子目录的中文名（UI 标签）。顺序与后端 `SUBDIRS` 一致。 */
export const STORAGE_SUBDIR_LABELS: Record<string, { zh: string; en: string; hint: string }> = {
  data: { zh: "数据", en: "Data", hint: "数据库、上传的 PDF、索引、备份、日志" },
  huggingface: { zh: "模型权重", en: "Models", hint: "BGE-M3 与 reranker 权重" },
  venv: { zh: "运行环境", en: "Runtime", hint: "Python 依赖（含 torch）；换位置时**重建**，不搬" },
  "uv-cache": { zh: "依赖缓存", en: "uv cache", hint: "依赖分发缓存，丢了会自动重下" },
};

/** 体积的展示文案（截断时必须说明是下界，否则数字会被当成真值）。 */
export function formatSize(mb: number | undefined, truncated: boolean | undefined, lang: "zh" | "en"): string {
  if (mb === undefined || Number.isNaN(mb)) return "…";
  const v = mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
  if (!truncated) return v;
  return lang === "en" ? `${v}+` : `${v}+（部分统计）`;
}