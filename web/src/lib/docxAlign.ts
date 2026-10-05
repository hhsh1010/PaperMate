import type { StreamItem } from "./types";

/**
 * P1：把 docx-preview 渲染的 DOM 与后端文档流编址（list_stream）按序对齐。
 *
 * docx-preview 严格按 XML 文档流输出 p/table（sectPr 不渲染），因此
 * 「渲染容器中 p/table 的出现顺序 == list_stream 条目顺序」。
 * 遇 table 时不再深入（其内部 p 属于该 table 的 cells）。
 *
 * 数量不匹配（渲染层折损）→ ok=false，调用方回退结构模式 —— **只影响能否在图文视图编辑，
 * 不影响正确性**：即使错位注入，quote 校验阀也会在后端拒绝（宁拒不错）。
 */
export interface AlignResult {
  ok: boolean;
  matched: number;
}

export function alignStreamDom(root: HTMLElement, items: StreamItem[]): AlignResult {
  const nodes: HTMLElement[] = [];
  const walk = (el: Element) => {
    for (const child of Array.from(el.children)) {
      const tag = child.tagName.toLowerCase();
      if (tag === "p") {
        nodes.push(child as HTMLElement);
      } else if (tag === "table") {
        nodes.push(child as HTMLElement);
        continue; // 表格内部的 p 属于 table item，不单独配对
      } else {
        walk(child);
      }
    }
  };
  walk(root);

  const targets = items.filter((it) => it.kind === "para" || it.kind === "table");
  if (nodes.length !== targets.length) {
    return { ok: false, matched: 0 };
  }

  targets.forEach((item, i) => {
    const el = nodes[i];
    el.dataset.pid = item.pid;
    el.classList.add("rw-doc-el");
    if (item.kind === "para" && item.editable === false) {
      el.classList.add("rw-doc-locked");
    }
  });
  return { ok: true, matched: targets.length };
}

/** 从点击目标向上找最近的已编址元素（事件委托用）。 */
export function findAlignedEl(target: EventTarget | null): HTMLElement | null {
  if (!(target instanceof Element)) return null;
  const el = target.closest<HTMLElement>("[data-pid]");
  return el ?? null;
}
