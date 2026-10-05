/**
 * `docScope.ts` 的守护测试（2026-09-23）。
 *
 * 锁的重点是**场景边界**：哪一档该说什么话。三档（general / single / multi）
 * 加上「本轮是否 @ 聚焦」，是这段逻辑唯一的出错面。
 */

import { describe, expect, it } from "vitest";

import { docScopeHint } from "./docScope";
import type { DocScopeOut } from "./types";

const scope = (over: Partial<DocScopeOut> = {}): DocScopeOut => ({
  session_id: 1,
  doc_count: 0,
  doc_ids: [],
  documents: [],
  force_retrieval: false,
  mode: "general",
  ...over,
});

const twoDocs = scope({
  doc_count: 2,
  doc_ids: [15, 10],
  documents: [
    { id: 15, filename: "融合道路邻近关系的高分遥感目标分割方法.pdf" },
    { id: 10, filename: "DETRs Beat YOLOs on Real-time Object Detection.pdf" },
  ],
  force_retrieval: true,
  mode: "multi",
});

describe("docScopeHint", () => {
  it("未加载（null）时不显示 —— 宁可不显示也不显示错的状态", () => {
    expect(docScopeHint(null)).toBeNull();
  });

  it("★ 0 篇 → 点明「通用对话 + 不会检索论文内容」", () => {
    // 这一句是消除「文档约束失效」误判的关键：用户以为在文档会话里问，
    // 实际没绑文档 → 必须让他在输入框旁边就看见。
    const h = docScopeHint(scope());
    expect(h?.tone).toBe("general");
    expect(h?.text).toContain("通用对话");
    expect(h?.text).toContain("不会检索论文内容");
  });

  it("1 篇 → 说明基于哪一篇（并去掉扩展名）", () => {
    const h = docScopeHint(
      scope({
        doc_count: 1,
        doc_ids: [10],
        documents: [{ id: 10, filename: "DETRs Beat YOLOs.pdf" }],
        force_retrieval: true,
        mode: "single",
      }),
    );
    expect(h?.tone).toBe("single");
    expect(h?.text).toContain("DETRs Beat YOLOs");
    expect(h?.text).not.toContain(".pdf");
  });

  it("多篇未 @ → 说明检索全部", () => {
    const h = docScopeHint(twoDocs, []);
    expect(h?.text).toContain("检索全部 2 篇");
  });

  it("★ 多篇且 @ 了某篇 → 改为「仅检索已指定的」", () => {
    // 与后端 `_effective_doc_scope` 的新语义一致：@ = 聚焦，不是追加。
    const h = docScopeHint(twoDocs, [10]);
    expect(h?.text).toContain("仅检索已指定的 1 篇");
    expect(h?.text).not.toContain("检索全部");
  });

  it("@ 了不在会话范围内的 id → 忽略该提及，退回「全部」", () => {
    // 与后端一致：无效提及不静默变空（否则回答会变成"什么都检索不到"且无法解释）。
    const h = docScopeHint(twoDocs, [999]);
    expect(h?.text).toContain("检索全部 2 篇");
  });

  it("超过 3 篇时用「等」省略，不把整行撑破", () => {
    const many = scope({
      doc_count: 5,
      doc_ids: [5, 4, 3, 2, 1],
      documents: [1, 2, 3, 4, 5].map((i) => ({ id: i, filename: `论文${i}.pdf` })),
      force_retrieval: true,
      mode: "multi",
    });
    const h = docScopeHint(many, []);
    expect(h?.text).toContain("检索全部 5 篇");
    expect(h?.text).toContain("等");
  });

  it("篇名过长会截断（避免撑破一行）", () => {
    const long = scope({
      doc_count: 1,
      doc_ids: [1],
      documents: [{ id: 1, filename: "这是一个非常非常非常非常非常长的论文标题用来测试截断行为.pdf" }],
      force_retrieval: true,
      mode: "single",
    });
    const h = docScopeHint(long, []);
    expect(h?.text).toContain("…");
  });

  it("key 随范围变化 —— 否则 React 会复用旧节点导致文案不更新", () => {
    const a = docScopeHint(twoDocs, []);
    const b = docScopeHint(twoDocs, [10]);
    expect(a?.key).not.toBe(b?.key);
  });
});
