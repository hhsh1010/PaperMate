// 轻量 Markdown 解析器（纯函数，无依赖，可独立单测）。
// 仅覆盖 AI 回复常用的子集：标题 / 加粗 / 行内代码 / 代码块 / 无序与有序列表 / 引用 /
// 表格（GFM 子集）/ 段落。输出 token 树（不渲染 HTML），由 MarkdownText 组件安全地
// 转成 React 元素，杜绝 XSS。

export type MdInline =
  | { type: "text"; content: string }
  | { type: "bold"; content: MdInline[] }
  | { type: "code"; content: string };

export type MdBlock =
  | { type: "paragraph"; inline: MdInline[] }
  | { type: "h1" | "h2" | "h3"; inline: MdInline[] }
  | { type: "ul"; items: MdInline[][] }
  | { type: "ol"; items: MdInline[][] }
  | { type: "code"; content: string }
  | { type: "quote"; inline: MdInline[] }
  | { type: "table"; header: string[]; rows: string[][]; raw: string };

/** 行内解析：**加粗** 与 `行内代码`（加粗内可再含代码，递归）。
 * 注意：正则实例必须在函数内新建——模块级共享正则的 lastIndex 会被递归调用重置，
 * 导致外层 exec 死循环（2026-08-20 实测 heap OOM）。 */
export function parseInline(text: string): MdInline[] {
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  const out: MdInline[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ type: "text", content: text.slice(last, m.index) });
    const token = m[0];
    if (token.startsWith("**")) {
      out.push({ type: "bold", content: parseInline(token.slice(2, -2)) });
    } else {
      out.push({ type: "code", content: token.slice(1, -1) });
    }
    last = m.index + token.length;
  }
  if (last < text.length) out.push({ type: "text", content: text.slice(last) });
  return out;
}

const isBlockStart = (line: string): boolean =>
  /^(#{1,3}\s|[-*]\s|\d+[.、]\s|>\s?|```)/.test(line);

/** GFM 表格行（以 | 开头且以 | 结尾）；分隔行 |---|---| 也匹配 */
const isTableRow = (line: string): boolean => /^\s*\|.*\|\s*$/.test(line);
/** 表头分隔行（仅含 | - : 空格） */
const isTableSeparator = (line: string): boolean => /^\s*\|[\s:|-]+\|\s*$/.test(line);

/** 切分一行表格单元格（去掉首尾 |，按 | 拆分并 trim） */
function splitTableCells(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((c) => c.trim());
}

/** 块级解析：逐行扫描为块序列 */
export function parseMarkdown(text: string): MdBlock[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: MdBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // 代码块（围栏）
    if (line.trimStart().startsWith("```")) {
      i++;
      const buf: string[] = [];
      while (i < lines.length && !lines[i].trimStart().startsWith("```")) {
        buf.push(lines[i]);
        i++;
      }
      i++; // 跳过闭合围栏
      blocks.push({ type: "code", content: buf.join("\n") });
      continue;
    }

    // 标题
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) {
      const level = `h${h[1].length}` as "h1" | "h2" | "h3";
      blocks.push({ type: level, inline: parseInline(h[2]) });
      i++;
      continue;
    }

    // 无序列表
    if (/^\s*[-*]\s+/.test(line)) {
      const items: MdInline[][] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(parseInline(lines[i].replace(/^\s*[-*]\s+/, "")));
        i++;
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    // 有序列表
    if (/^\s*\d+[.、]\s+/.test(line)) {
      const items: MdInline[][] = [];
      while (i < lines.length && /^\s*\d+[.、]\s+/.test(lines[i])) {
        items.push(parseInline(lines[i].replace(/^\s*\d+[.、]\s+/, "")));
        i++;
      }
      blocks.push({ type: "ol", items });
      continue;
    }

    // 引用
    if (/^\s*>\s?/.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      blocks.push({ type: "quote", inline: parseInline(buf.join(" ")) });
      continue;
    }

    // 表格（GFM 子集）：当前行是表格行且下一行是分隔行才识别为表格。
    // 隔离的 | 行（无分隔行跟随）仍按段落处理，避免误伤普通文本里的竖线。
    if (isTableRow(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const rawLines: string[] = [line.trim(), lines[i + 1].trim()];
      const header = splitTableCells(line);
      const rows: string[][] = [];
      i += 2; // 跳过表头与分隔行（表头后第一个分隔行是表结构标记，不入数据行）
      while (i < lines.length && isTableRow(lines[i])) {
        rawLines.push(lines[i].trim());
        rows.push(splitTableCells(lines[i]));
        i++;
      }
      blocks.push({ type: "table", header, rows, raw: rawLines.join("\n") });
      continue;
    }

    // 空行
    if (!line.trim()) {
      i++;
      continue;
    }

    // 段落（连续非空、非块起始行合并；表格行+分隔行成对出现时也中断，交给表格分支）
    const buf: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !isBlockStart(lines[i].trimStart()) &&
      !(isTableRow(lines[i]) && i + 1 < lines.length && isTableSeparator(lines[i + 1]))
    ) {
      buf.push(lines[i]);
      i++;
    }
    blocks.push({ type: "paragraph", inline: parseInline(buf.join("\n")) });
  }

  return blocks;
}
