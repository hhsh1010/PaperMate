import { describe, expect, it } from "vitest";
import { parseInline, parseMarkdown } from "./markdown";

describe("parseInline", () => {
  it("普通文本", () => {
    expect(parseInline("你好")).toEqual([{ type: "text", content: "你好" }]);
  });
  it("加粗", () => {
    expect(parseInline("结论是 **关键点** 结束")).toEqual([
      { type: "text", content: "结论是 " },
      { type: "bold", content: [{ type: "text", content: "关键点" }] },
      { type: "text", content: " 结束" },
    ]);
  });
  it("行内代码", () => {
    expect(parseInline("运行 `retrieve` 工具")).toEqual([
      { type: "text", content: "运行 " },
      { type: "code", content: "retrieve" },
      { type: "text", content: " 工具" },
    ]);
  });
  it("加粗内含行内代码", () => {
    const out = parseInline("**先调用 `retrieve`**");
    expect(out[0]).toMatchObject({ type: "bold" });
    if (out[0].type === "bold") {
      expect(out[0].content).toEqual([
        { type: "text", content: "先调用 " },
        { type: "code", content: "retrieve" },
      ]);
    }
  });
});

describe("parseMarkdown", () => {
  it("标题", () => {
    const blocks = parseMarkdown("# 标题一\n\n## 标题二\n\n### 标题三");
    expect(blocks.map((b) => b.type)).toEqual(["h1", "h2", "h3"]);
  });

  it("无序与有序列表", () => {
    const blocks = parseMarkdown("- 甲\n- 乙\n\n1. 一\n2. 二");
    expect(blocks[0]).toMatchObject({ type: "ul" });
    expect(blocks[1]).toMatchObject({ type: "ol" });
    if (blocks[0].type === "ul") {
      expect(blocks[0].items.length).toBe(2);
      expect(blocks[0].items[0][0]).toMatchObject({ type: "text", content: "甲" });
    }
  });

  it("代码块", () => {
    const blocks = parseMarkdown("```python\nprint(1)\nprint(2)\n```");
    expect(blocks[0]).toEqual({ type: "code", content: "print(1)\nprint(2)" });
  });

  it("引用", () => {
    const blocks = parseMarkdown("> 注意：这是重点");
    expect(blocks[0]).toMatchObject({ type: "quote" });
    if (blocks[0].type === "quote") {
      expect(blocks[0].inline[0]).toMatchObject({ type: "text", content: "注意：这是重点" });
    }
  });

  it("段落合并与加粗", () => {
    const blocks = parseMarkdown("第一行\n第二行 **加粗**");
    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe("paragraph");
    if (blocks[0].type === "paragraph") {
      const text = blocks[0].inline.filter((t) => t.type === "text").map((t) => t.content).join("");
      expect(text).toContain("第一行");
      expect(text).toContain("第二行");
      expect(blocks[0].inline.some((t) => t.type === "bold")).toBe(true);
    }
  });

  it("过程+结果典型结构", () => {
    const md = [
      "已检索论文并核对原文。",
      "",
      "**结论**：该文提出 MC-YOLO 方法。",
      "",
      "关键点：",
      "- 多尺度特征融合（MFFE）",
      "- 细节特征提取（DFE）",
    ].join("\n");
    const blocks = parseMarkdown(md);
    const types = blocks.map((b) => b.type);
    expect(types).toEqual(["paragraph", "paragraph", "paragraph", "ul"]);
  });

  it("GFM 表格：表头 + 分隔行 + 数据行", () => {
    const md = ["| 模型 | 准确率 |", "| --- | --- |", "| BERT | 92.1% |", "| GPT | 88.4% |"].join("\n");
    const blocks = parseMarkdown(md);
    expect(blocks).toHaveLength(1);
    const t = blocks[0];
    expect(t.type).toBe("table");
    if (t.type === "table") {
      expect(t.header).toEqual(["模型", "准确率"]);
      expect(t.rows).toEqual([["BERT", "92.1%"], ["GPT", "88.4%"]]);
      expect(t.raw).toBe("| 模型 | 准确率 |\n| --- | --- |\n| BERT | 92.1% |\n| GPT | 88.4% |");
    }
  });

  it("表格可含加粗/行内代码单元格（行内解析复用）", () => {
    const md = ["| **列A** | `code` |", "| --- | --- |", "| 甲 | 乙 |"].join("\n");
    const blocks = parseMarkdown(md);
    expect(blocks[0].type).toBe("table");
  });

  it("段落紧贴表格时正确拆分（不吞并）", () => {
    const md = ["说明如下：", "| A | B |", "| --- | --- |", "| 1 | 2 |", "结尾段落"].join("\n");
    const blocks = parseMarkdown(md);
    expect(blocks.map((b) => b.type)).toEqual(["paragraph", "table", "paragraph"]);
  });

  it("孤立竖线行不误判为表格（无分隔行跟随）", () => {
    const blocks = parseMarkdown("这是 | 竖线 | 文本");
    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe("paragraph");
  });

  it("代码块内的表格行不误判", () => {
    const md = ["```", "| A | B |", "| --- | --- |", "```"].join("\n");
    const blocks = parseMarkdown(md);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe("code");
  });

  it("分隔行不进入数据行（表头后第一个才跳过，数据区 --- 按普通行处理）", () => {
    const md = ["| A | B |", "| --- | --- |", "| --- | --- |", "| 1 | 2 |"].join("\n");
    const blocks = parseMarkdown(md);
    const t = blocks[0];
    if (t.type === "table") {
      expect(t.rows).toEqual([["---", "---"], ["1", "2"]]);
    }
  });
});
