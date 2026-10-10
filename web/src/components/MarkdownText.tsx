import { useMemo } from "react";
import type { ReactNode } from "react";
import { parseInline, parseMarkdown } from "../lib/markdown";
import type { MdBlock, MdInline } from "../lib/markdown";

/** 行内 token → React 元素（安全渲染，不产生 HTML） */
function Inline({ tokens }: { tokens: MdInline[] }): ReactNode {
  return (
    <>
      {tokens.map((t, i) => {
        switch (t.type) {
          case "bold":
            return (
              <strong key={i}>
                <Inline tokens={t.content} />
              </strong>
            );
          case "code":
            return <code key={i}>{t.content}</code>;
          default:
            return <span key={i}>{t.content}</span>;
        }
      })}
    </>
  );
}

function renderBlock(b: MdBlock, key: number): ReactNode {
  switch (b.type) {
    case "h1":
      return <h1 key={key}><Inline tokens={b.inline} /></h1>;
    case "h2":
      return <h2 key={key}><Inline tokens={b.inline} /></h2>;
    case "h3":
      return <h3 key={key}><Inline tokens={b.inline} /></h3>;
    case "ul":
      return (
        <ul key={key}>
          {b.items.map((it, i) => (
            <li key={i}>
              <Inline tokens={it} />
            </li>
          ))}
        </ul>
      );
    case "ol":
      return (
        <ol key={key}>
          {b.items.map((it, i) => (
            <li key={i}>
              <Inline tokens={it} />
            </li>
          ))}
        </ol>
      );
    case "code":
      return <pre key={key}><code>{b.content}</code></pre>;
    case "quote":
      return (
        <blockquote key={key}>
          <Inline tokens={b.inline} />
        </blockquote>
      );
    case "table":
      return (
        <div className="md-table-wrap" key={key}>
          <div className="md-table-scroll">
            <table className="md-table">
              <thead>
                <tr>
                  {b.header.map((c, i) => (
                    <th key={i}>
                      <Inline tokens={parseInline(c)} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {b.rows.map((r, i) => (
                  <tr key={i}>
                    {r.map((c, j) => (
                      <td key={j}>
                        <Inline tokens={parseInline(c)} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {b.raw && (
            <button
              type="button"
              className="md-copy-btn"
              title="复制原始 Markdown 表格"
              onClick={(e) => {
                const btn = e.currentTarget;
                const fail = () => {
                  btn.title = "复制失败（剪贴板不可用）";
                };
                try {
                  navigator.clipboard?.writeText(b.raw).then(
                    () => {
                      // ★ P-05：补"已复制"反馈（原先成功也毫无提示）
                      const old = btn.title;
                      btn.title = "已复制";
                      window.setTimeout(() => {
                        btn.title = old;
                      }, 1500);
                    },
                    fail,
                  );
                } catch {
                  fail();
                }
              }}
            >
              复制 Markdown
            </button>
          )}
        </div>
      );
    default:
      return (
        <p key={key}>
          <Inline tokens={b.inline} />
        </p>
      );
  }
}

/** 把 AI 回复的 Markdown 安全渲染为排版规整的富文本（过程+结果结构） */
export function MarkdownText({ text }: { text: string }) {
  const blocks = useMemo(() => parseMarkdown(text), [text]);
  return <div className="md-body">{blocks.map(renderBlock)}</div>;
}
