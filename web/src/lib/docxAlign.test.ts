// @vitest-environment jsdom
import { describe, it, expect } from "vitest";
import { alignStreamDom, findAlignedEl } from "./docxAlign";
import type { StreamItem } from "./types";

/** P1：docx-preview 渲染 DOM 与文档流编址的对齐 */

const para = (pid: string, text: string, editable = true): StreamItem => ({
  kind: "para",
  pid,
  text,
  empty: !text.trim(),
  editable,
  reason: null,
});

const table = (pid: string, cells: string[]): StreamItem => ({
  kind: "table",
  pid,
  cells: cells.map((t, i) => ({
    pid: `${pid}.R0.C${i}.P0`,
    style: null,
    text: t,
    empty: false,
    editable: true,
    reason: null,
  })),
});

/** 模拟 docx-preview 输出：wrapper > section > (p | table)* */
function buildDom(structure: ("p" | "table")[]): HTMLElement {
  const wrapper = document.createElement("div");
  const section = document.createElement("section");
  wrapper.appendChild(section);
  for (const kind of structure) {
    if (kind === "p") {
      const p = document.createElement("p");
      p.appendChild(document.createTextNode("x"));
      section.appendChild(p);
    } else {
      const tbl = document.createElement("table");
      const tr = tbl.insertRow();
      tr.insertCell().appendChild(document.createElement("p")); // 表格内的 p 不参与顶层配对
      section.appendChild(tbl);
    }
  }
  return wrapper;
}

describe("alignStreamDom", () => {
  it("p/table 混合按序注入 pid（表格内部 p 不配对）", () => {
    const root = buildDom(["p", "p", "table", "p"]);
    const items = [para("B0", "a"), para("B1", "b"), table("B2", ["c"]), para("B3", "d")];
    const r = alignStreamDom(root, items);
    expect(r).toEqual({ ok: true, matched: 4 });

    const els = root.querySelectorAll<HTMLElement>("[data-pid]");
    expect([...els].map((e) => e.dataset.pid)).toEqual(["B0", "B1", "B2", "B3"]);
    // 表格内嵌 p 不被注入
    const innerP = root.querySelector("table p") as HTMLElement;
    expect(innerP.dataset.pid).toBeUndefined();
  });

  it("数量不匹配 → ok=false（渲染折损兜底）", () => {
    const root = buildDom(["p", "p"]);
    const r = alignStreamDom(root, [para("B0", "a"), para("B1", "b"), para("B2", "c")]);
    expect(r.ok).toBe(false);
    expect(root.querySelector("[data-pid]")).toBeNull();
  });

  it("不可编辑段落加 locked 标记", () => {
    const root = buildDom(["p", "p"]);
    const items = [para("B0", "a", false), para("B1", "b")];
    alignStreamDom(root, items);
    const els = root.querySelectorAll<HTMLElement>("[data-pid]");
    expect(els[0].classList.contains("rw-doc-locked")).toBe(true);
    expect(els[1].classList.contains("rw-doc-locked")).toBe(false);
  });
});

describe("findAlignedEl", () => {
  it("从子节点向上找最近编址元素；无 → null", () => {
    const root = buildDom(["p"]);
    alignStreamDom(root, [para("B0", "a")]);
    const p = root.querySelector("[data-pid]")!;
    const span = document.createElement("span");
    p.appendChild(span);
    expect(findAlignedEl(span)?.dataset.pid).toBe("B0");
    expect(findAlignedEl(document.body)).toBeNull();
    expect(findAlignedEl(null)).toBeNull();
  });
});
