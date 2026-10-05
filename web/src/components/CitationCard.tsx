import { useState } from "react";
import type { Citation } from "../lib/types";
import { Icons } from "../lib/icons";
import { docLabel } from "../lib/docLabel";

const MAX_VISIBLE = 5;
const MAX_TEXT_PREVIEW = 160;

// ★ B2-X2（2026-10-01）：这里**不再有** `cleanCitations` / `NOISE_KEYWORDS`。
//   引用清洗（去空/去短/超长截断/噪声词/去重）已收敛到后端**唯一入口**
//   `src/papermate/services/citation_clean.py` —— 后端产出即最终形态。
//   之所以删掉这份：它与后端规则**不一致**（超长直接丢弃 vs 后端截断；噪声词多一个 "Citation"），
//   导致同一份引用**在 UI 里消失、却仍被落库与 `citation_rate` 计入**（指标有数、用户看不到）。
//   本文件只保留**展示层**处理：折叠条数（`visibleCitations`）+ 文本预览截断（`previewText`）。

function previewText(text: string, expanded: boolean): string {
  if (expanded || text.length <= MAX_TEXT_PREVIEW) return text;
  return text.slice(0, MAX_TEXT_PREVIEW).trimEnd() + "…";
}

/** 展开态可见引用（2026-09-09 修复 N1 后的口径）：默认截断到 MAX_VISIBLE 条，expandedAll=true 才全量。
 *  原先的三元条件 `expandedAll || folded ? 截断 : 全量` 构成**死代码**——
 *  折叠态截断但列表根本不渲染（无用）；展开态 `false || false` 恒走全量分支 →
 *  hiddenCount 恒 0 → 「查看其余 N 条」按钮永不出现 → expandedAll 永远无法变为 true →
 *  MAX_VISIBLE 从未生效，长引用列表一直全量渲染（与 :54 注释"限制默认显示条数"矛盾）。
 *  抽为纯函数导出，供单测锁定口径。 */
export function visibleCitations(citations: Citation[], expandedAll: boolean): Citation[] {
  return expandedAll ? citations : citations.slice(0, MAX_VISIBLE);
}

/** 引用溯源卡片：渲染 RAG 检索结果的出处（编号 / 页码 / 片段），点击可打开原文阅读器（ER-22）。
 * 2026-08-26：限制默认显示条数，过长文本可展开。
 * 2026-08-27：新增整体收纳展开——默认折叠为标题行，点击后展开列表。
 * ★ 2026-10-01（B2-X2）：「过滤页眉页脚等噪声片段」**已移出前端** —— 清洗收敛到后端唯一入口
 *   `services/citation_clean.py`（此前前端那份与后端规则不一致，导致"UI 看不到、指标仍计入"）。
 */
export function CitationCard({
  citations,
  onOpen,
  activeDocIds,
}: {
  citations: Citation[];
  onOpen?: (c: Citation) => void;
  activeDocIds?: number[];
}) {
  const [folded, setFolded] = useState(true);
  const [expandedAll, setExpandedAll] = useState(false);
  const [expandedText, setExpandedText] = useState<Record<number, boolean>>({});

  // ★ B2-X2（2026-10-01）：卡片**不再自己清洗** —— 清洗已收敛到后端唯一入口
  //   （`papermate/services/citation_clean.py`）。此前这里有一份独立的 `cleanCitations`，
  //   规则还与后端**不一致**（超长直接丢弃、噪声词多一个 "Citation"）⇒
  //   引用在 UI 里静默消失，而落库与 `citation_rate` 仍然算它（指标有数、用户看不到）。
  //   现在只保留**展示层**处理：条数折叠（`visibleCitations`）+ 预览截断（`previewText`）。
  if (!citations.length) return null;

  const visible = visibleCitations(citations, expandedAll);
  const hiddenCount = citations.length - visible.length;
  // B2（2026-09-29）：本列表里是否存在"注入模型时被截断"的片段 ⇒ 决定是否显示脚注解释。
  const anyModelTruncated = visible.some((c) => c.model_truncated);

  return (
    <div className={`citations ${folded ? "citations-folded" : ""}`}>
      <button
        type="button"
        className="citations-title citations-title-toggle"
        onClick={() => setFolded((v) => !v)}
        aria-expanded={!folded}
        aria-label={folded ? "展开引用来源" : "收起引用来源"}
      >
        <span className="citations-title-text">
          引用来源（{citations.length}）
        </span>
        <span className="citations-title-icon" aria-hidden="true">
          {folded ? <Icons.chevronRight size={14} /> : <Icons.chevronDown size={14} />}
        </span>
      </button>
      {!folded && (
        <>
          <ol className="citations-list">
            {visible.map((c, i) => {
              const text = c.text ?? "";
              const expanded = expandedText[i] ?? false;
              const showExpand = text.length > MAX_TEXT_PREVIEW;
              // 文档友好名：doc_id 命中当前会话关联文档顺序时显示「文档一/文档二」（与右侧预览面板一致），
              // 否则回退到 chunk id / 序号，绝不暴露真实文件名（源码名）。
              const docIdx = c.doc_id != null && activeDocIds ? activeDocIds.indexOf(c.doc_id) : -1;
              const docBadge = docIdx >= 0 ? docLabel(docIdx) : (c.id != null ? String(c.id) : String(i + 1));
              return (
                <li
                  key={`${c.id ?? i}-${i}`}
                  className={`citation ${onOpen ? "clickable" : ""}`}
                  onClick={onOpen ? () => onOpen(c) : undefined}
                  role={onOpen ? "button" : undefined}
                  tabIndex={onOpen ? 0 : undefined}
                  onKeyDown={
                    onOpen
                      ? (e) => {
                          if (e.key === "Enter" || e.key === " ") onOpen(c);
                        }
                      : undefined
                  }
                >
                  <span className="citation-id">[{docBadge}]</span>
                  {c.page ? <span className="citation-page">p.{c.page}</span> : null}
                  {c.text && (
                    <span className="citation-text">
                      {previewText(text, expanded)}
                      {/* B2：该片段在注入模型时被截断 ⇒ 如实标注，避免用户以为
                          模型是基于完整段落作答（muted 色，不做成告警，防止噪声化）。 */}
                      {c.model_truncated && (
                        <span
                          className="citation-trunc"
                          title="该片段在注入模型时被截断，模型可能只看到了片段开头；此处展示的是原文，便于你核对。"
                        >
                          节选
                        </span>
                      )}
                      {showExpand && (
                        <button
                          className="citation-expand"
                          onClick={(e) => {
                            e.stopPropagation();
                            setExpandedText((prev) => ({ ...prev, [i]: !expanded }));
                          }}
                          aria-label={expanded ? "收起" : "展开"}
                        >
                          {expanded ? " 收起" : " 展开"}
                        </button>
                      )}
                    </span>
                  )}
                </li>
              );
            })}
          </ol>
          {hiddenCount > 0 && (
            <button className="citation-more" onClick={() => setExpandedAll((v) => !v)}>
              {expandedAll ? "收起引用" : `查看其余 ${hiddenCount} 条引用`}
            </button>
          )}
          {/* B2：脚注只在**确有**被截断片段时出现（不做成每条的告警徽章，避免噪声） */}
          {anyModelTruncated && (
            <p className="citation-note">
              标「节选」的片段在<strong>注入模型时被截断</strong>（模型仅见片段开头）；
              卡片展示的是原文，便于你核对。
            </p>
          )}
        </>
      )}
    </div>
  );
}
