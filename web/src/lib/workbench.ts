import type { EditOp } from "./types";

/**
 * C4：改稿工作台「变更集」纯函数。
 * 变更集 = 本地暂存的 edit ops（应用前磁盘不变），后端 apply-edits 前可自由增删改。
 */

/** quote 校验阀携带的原文片段长度（防段落漂移——应用时磁盘文本应仍含此片段） */
const QUOTE_LEN = 80;

/**
 * 把一条新 op 合并进变更集：**同 pid 的 op 相互覆盖（最后生效）**。
 * 理由：同一时段的 replace 后再 delete、或反复改写，用户意图总是「以最后一次为准」；
 * 若叠加多条，后端按序执行会出现「replace 的新文字又被 delete 删掉」的意外语义。
 */
export function mergeOp(edits: EditOp[], op: EditOp): EditOp[] {
  return [...edits.filter((e) => e.pid !== op.pid), op];
}

/** 从变更集移除某段的所有 op（用户撤销该段未应用的修改）。 */
export function dropPid(edits: EditOp[], pid: string): EditOp[] {
  return edits.filter((e) => e.pid !== pid);
}

/** 查某段当前暂存的 op（无 → null）。 */
export function findOp(edits: EditOp[], pid: string): EditOp | null {
  return edits.find((e) => e.pid === pid) ?? null;
}

/** 自动携带 quote（原文片段），后端据此校验段落未漂移。 */
export function withQuote(op: Omit<EditOp, "quote">, paraText: string): EditOp {
  const anchor = paraText.trim().slice(0, QUOTE_LEN);
  return { ...op, quote: anchor || null };
}

/** 变更集摘要（底部条展示用）。 */
export function summarize(edits: EditOp[]): { total: number; replaces: number; inserts: number; deletes: number } {
  return {
    total: edits.length,
    replaces: edits.filter((e) => e.op === "replace").length,
    inserts: edits.filter((e) => e.op === "insert_after").length,
    deletes: edits.filter((e) => e.op === "delete").length,
  };
}

/** 按 pid 在文档流条目中查找「有效文本」：先段落条目，再表格 cells。 */
export function findTextByPid(items: { kind: string; pid: string; text?: string; cells?: { pid: string; text: string }[] }[], pid: string): string | null {
  for (const it of items) {
    if (it.pid === pid) return it.text ?? null;
    for (const c of it.cells ?? []) {
      if (c.pid === pid) return c.text;
    }
  }
  return null;
}

/**
 * 表格 cells 按地址解析为 行×列 网格（结构模式渲染用）。
 * 地址形如 `B{i}.R{ri}.C{ci}.P{pi}` —— 按 R 分组、组内按 C 排序。
 */
export function groupCells<T extends { pid: string }>(cells: T[]): T[][] {
  const byRow = new Map<number, T[]>();
  for (const c of cells) {
    const m = c.pid.match(/\.R(\d+)\.C(\d+)\./);
    if (!m) continue;
    const ri = Number(m[1]);
    if (!byRow.has(ri)) byRow.set(ri, []);
    byRow.get(ri)!.push(c);
  }
  return [...byRow.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, row]) => row.sort((a, b) => {
      const ca = Number(a.pid.match(/\.C(\d+)\./)![1]);
      const cb = Number(b.pid.match(/\.C(\d+)\./)![1]);
      return ca - cb;
    }));
}
