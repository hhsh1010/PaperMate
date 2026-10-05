import { describe, expect, it } from "vitest";

import cardSrc from "./CitationCard.tsx?raw";
import { visibleCitations } from "./CitationCard";
import type { Citation } from "../lib/types";

/** B2（2026-09-29）：引用卡片需如实标注"该片段在注入模型时被截断"。
 *
 * 项目未装 testing-library（运行依赖刻意压到 4 个）⇒ 无法渲染做交互断言，
 * 退一步用**源码契约**断言 —— 足够拦住"有人把标记/脚注删掉"这类回归。
 *
 * ⚠️ **本文件测不到配色**（诚实边界）：断言 `styles.css` 需要读文件，而本项目既没有
 * `@types/node`（`node:fs` 不可用），`import "*.css?raw"` 在 vitest 下又返回**空串**
 * （CSS 走 vite 样式管线，不是 `?raw` 文本管线）。因此"用 muted 色而非告警色"这条
 * 只落在 `styles.css` 的注释里，**没有测试守护** —— 若将来有人把它改成红色告警色，
 * 测试不会报警。故此处只断言"标记走**样式类**、不内联颜色"，把配色决定集中到一处。
 */
describe("引用「模型所见已截断」标注（源码契约）", () => {
  it("按条目读取后端标记（不自己重算长度 —— 判据必须与后端同源）", () => {
    // ★ 断言**渲染守卫本身**，而不只是"字符串出现过"：
    //   ⚠️ 反向自检实测：只写 `toContain("c.model_truncated")` 时，
    //   把守卫改成 `{false && c.model_truncated && (` 依然绿 —— 标记根本不渲染却测不出。
    expect(cardSrc).toContain("{c.model_truncated && (");
    // 卡片侧不得再写一遍长度阈值（那会与 lightcore 的注入判据分叉）
    expect(cardSrc).not.toMatch(/model_truncated\s*&&\s*text\.length\s*>/);
  });

  it("被截断的条目渲染标记 + 悬浮说明（说清含义，不只放一个符号）", () => {
    expect(cardSrc).toContain('className="citation-trunc"');
    expect(cardSrc).toContain("节选");
    expect(cardSrc).toContain("注入模型时被截断"); // tooltip 里说清含义
  });

  it("脚注只在确有条目被截断时出现（不做成每条告警徽章，避免噪声化）", () => {
    // 同样断言**守卫本身**（理由见上一条）
    expect(cardSrc).toContain("{anyModelTruncated && (");
    expect(cardSrc).toContain("citation-note");
  });

  it("标记走样式类、不在组件内联颜色（配色集中到 styles.css，便于保持 muted）", () => {
    const i = cardSrc.indexOf('className="citation-trunc"');
    const around = cardSrc.slice(Math.max(0, i - 200), i + 300);
    expect(around).not.toContain("style={{"); // 不内联颜色
  });
});

describe("visibleCitations 保留标记字段", () => {
  it("截断/展开都不丢 model_truncated（否则脚注会随展示条数漂移）", () => {
    const cits: Citation[] = [
      { id: "c1", text: "a".repeat(30), model_truncated: true },
      { id: "c2", text: "b".repeat(30), model_truncated: false },
    ];
    const out = visibleCitations(cits, false);
    expect(out.map((c) => c.model_truncated)).toEqual([true, false]);
  });
});

describe("B2-X2：清洗收敛到后端，卡片只做展示", () => {
  /** 2026-10-01：引用清洗曾有**三份实现**（后端 / 技能侧 / 前端），前端那份与后端**规则不一致**
   *  （超长直接丢弃 vs 后端截断；噪声词多一个 "Citation"）⇒ 同一份引用**在 UI 里消失、
   *  却仍被落库与 `citation_rate` 计入**（指标有数、用户看不到）。现已收敛到后端唯一入口
   *  `src/papermate/services/citation_clean.py`，前端只保留**展示层**（条数折叠 + 预览截断）。
   *
   *  ⚠️ **断言写法（吃过亏）**：这里**不能**写 `expect(cardSrc).not.toContain("cleanCitations")` ——
   *  源码里的解释性注释就含这个词，会**匹配到注释**而不是代码（2026-09-30 在别处已踩过一次）。
   *  故一律断言**声明/调用形态**，即"会被执行的那段"。
   */
  it("卡片直接消费传入的 citations（而非某个前端清洗后的副本）", () => {
    // ⚠️ 必须带上逗号：`/visibleCitations\(citations/` 会**误放过** `visibleCitations(citations.filter(...), …)`
    //    —— 那正是"又把清洗加回来"的形态（反向自检时实测过）。
    expect(cardSrc).toMatch(/visibleCitations\(\s*citations\s*,/);
  });

  it("前端不再保留任何清洗/过滤实现（超长丢弃、噪声词表、去重键）", () => {
    expect(cardSrc).not.toMatch(/function\s+cleanCitations/);
    expect(cardSrc).not.toMatch(/const\s+NOISE_KEYWORDS/);
    expect(cardSrc).not.toMatch(/\.length\s*>\s*600/); // 超长不再被前端丢弃
    expect(cardSrc).not.toMatch(/\.slice\(\s*0\s*,\s*80\s*\)/); // 去重键不再在前端算
  });

  it("标题只报条数（不再出现与清洗耦合的「/ 原始 N」文案）", () => {
    expect(cardSrc).not.toMatch(/原始/);
  });
});
