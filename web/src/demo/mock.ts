/**
 * 演示模式（VITE_DEMO_MODE=1）：**零后端、零密钥**的静态网页 demo。
 *
 * ## 它做什么
 * 把 `window.fetch` 换成一个小路由：所有 `/api/v1/*` 请求都由本模块用**内置样例数据**应答，
 * 因此把 `vite build --mode demo` 的产物丢到 GitHub Pages 上，点开就是一个能点的界面 ——
 * **不连任何后端、不需要任何 API key、不产生任何费用**。
 *
 * ## 边界（如实声明）
 * - 只 mock **核心三屏**（对话 / 文献列表 / 改稿工作台的段落视图）；其余端点返回 404 +
 *   明确的 `demo_not_mocked` 提示，让界面显示"未提供样例"而不是假装成功。
 * - 聊天流只发 `message_delta` / `message` / `done` 三种事件（**不发 tool_call/tool_result**：
 *   那两类事件的字段较多，样例写不全反而会让界面渲染出错误内容 —— 宁缺勿错）。
 * - 写操作（上传/改稿/导出）返回**看起来成功**的响应，但**不会真的改动任何东西**。
 */

const BASE = "/api/v1";
const DEMO_DOC_ID = 1;
const NOW = "2026-10-10T12:00:00Z";

// --------------------------------------------------------------------------- 样例数据

const SESSIONS = [
  {
    id: 1,
    title: "轻量级目标检测方法 · 研读",
    config: { model: "demo", workspace: null },
    created_at: NOW,
    updated_at: NOW,
    pinned: true,
  },
  {
    id: 2,
    title: "改稿：论文引言润色",
    config: { model: "demo", workspace: { mode: "revision" } },
    created_at: NOW,
    updated_at: NOW,
    pinned: false,
  },
];

const DOCS = [
  {
    id: DEMO_DOC_ID,
    filename: "demo_paper.docx",
    status: "ready",
    meta: { num_blocks: 10, num_chunks: 12, num_tables: 1, num_images: 2, num_diagrams: 0 },
    num_chunks: 12,
    created_at: NOW,
  },
  {
    id: 2,
    filename: "survey_lightweight_detection.pdf",
    status: "ready",
    meta: { num_blocks: 48, num_chunks: 96, num_tables: 3, num_images: 9, num_diagrams: 1 },
    num_chunks: 96,
    created_at: NOW,
  },
  {
    id: 3,
    filename: "notes_attention.md",
    status: "ready",
    meta: { num_blocks: 6, num_chunks: 6, num_tables: 0, num_images: 0, num_diagrams: 0 },
    num_chunks: 6,
    created_at: NOW,
  },
];

/** 与 scripts/demo_smoke.py 里生成的示例论文同源（便于"本地跑 vs 网页 demo"相互印证）。 */
export const DEMO_PARAGRAPHS = [
  { pid: "P0", style: "Title", text: "面向低资源场景的轻量级目标检测方法", empty: false, editable: true, reason: null },
  { pid: "P1", style: null, text: "摘要：本文提出一种轻量检测网络，在保持精度的同时显著降低参数量。", empty: false, editable: true, reason: null },
  { pid: "P2", style: "Heading 1", text: "1 引言", empty: false, editable: true, reason: null },
  { pid: "P3", style: null, text: "目标检测是计算机视觉的基础任务，但现有方法参数量大、推理成本高。", empty: false, editable: true, reason: null },
  { pid: "P4", style: null, text: "本文的核心贡献有三点：轻量骨干、跨尺度特征融合、以及一致性蒸馏训练策略。", empty: false, editable: true, reason: null },
  { pid: "P5", style: "Heading 1", text: "2 方法", empty: false, editable: true, reason: null },
  { pid: "P6", style: null, text: "我们采用深度可分离卷积替代标准卷积，并在颈部引入跨尺度注意力。", empty: false, editable: true, reason: null },
  { pid: "P7", style: null, text: "训练阶段使用一致性蒸馏：教师网络与学生网络在同一批次上对齐预测分布。", empty: false, editable: true, reason: null },
  { pid: "P8", style: "Heading 1", text: "3 实验", empty: false, editable: true, reason: null },
  { pid: "P9", style: null, text: "在公开数据集上，本方法以 1/3 的参数量达到基线 98.2% 的精度，端侧推理延迟降低 41%。", empty: false, editable: true, reason: null },
];

const DEMO_ANSWER = [
  "根据检索内容（来源：rag），本文的核心方法可概括为三部分：",
  "",
  "1. **轻量骨干**：用深度可分离卷积替代标准卷积，显著压缩参数与计算量；",
  "2. **跨尺度特征融合**：在颈部引入跨尺度注意力，缓解小目标在多尺度下的特征稀疏；",
  "3. **一致性蒸馏训练**：教师与学生网络在同一批次上对齐预测分布，提升轻量模型的精度上限。",
  "",
  "实验上，该方法以约 1/3 的参数量达到基线 98.2% 的精度，端侧推理延迟降低 41%。",
].join("\n");

const CITATIONS = [
  {
    id: "doc-1-c0",
    doc_id: 1,
    page: null,
    model_truncated: true,
    text: "方法：我们采用深度可分离卷积替代标准卷积，并在颈部引入跨尺度注意力。训练阶段使用一致性蒸馏。",
  },
  {
    id: "doc-1-c2",
    doc_id: 1,
    page: null,
    model_truncated: false,
    text: "实验：本方法以 1/3 的参数量达到基线 98.2% 的精度，端侧推理延迟降低 41%。",
  },
];

// --------------------------------------------------------------------------- 工具

const json = (data: unknown, status = 200): Response =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const notMocked = (path: string): Response =>
  json(
    {
      detail: `演示模式：该接口（${path}）未提供样例数据。完整功能请在本地运行：见 README「快速开始」。`,
      code: "demo_not_mocked",
    },
    404,
  );

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** 把一段回答切成若干增量帧，模拟真实 SSE 流式（前端用 getReader() 消费，与线上同一套）。 */
function sseStream(text: string): Response {
  const enc = new TextEncoder();
  const frame = (event: string, data: unknown) => enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const parts = text.match(/[^\n]{1,18}|\n/g) ?? [text];
      for (const p of parts) {
        controller.enqueue(frame("message_delta", { text: p }));
        await sleep(28);
      }
      controller.enqueue(frame("message", { text, citations: CITATIONS }));
      controller.enqueue(frame("done", { usage: { prompt_tokens: 2088, completion_tokens: 186, cost: 0.0 } }));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

// --------------------------------------------------------------------------- 路由

/** 路由核心：**导出以便单测**（不依赖 window，Node 环境可测）。 */
export async function demoRoute(path: string, method: string): Promise<Response | null> {
  const p = path.split("?")[0];

  // 三屏核心数据
  if (p === `${BASE}/healthz`) {
    return json({
      status: "ok",
      db_ok: true,
      network_exposed: false,
      default_model: "demo",
      default_embedder: "fake",
      rag_embedder: "fake",
      use_bge_sparse: true,
      use_jieba: true,
    });
  }
  if (p === `${BASE}/sessions` && method === "GET") return json(SESSIONS);
  if (/^\/api\/v1\/sessions\/\d+$/.test(p) && method === "GET") {
    const id = Number(p.split("/").pop());
    const base = SESSIONS.find((s) => s.id === id) ?? SESSIONS[0];
    return json({
      ...base,
      messages: [
        { id: 1, role: "user", content: "这篇论文的核心方法是什么？", prompt_tokens: 0, completion_tokens: 0, cost: 0 },
        {
          id: 2,
          role: "assistant",
          content: DEMO_ANSWER,
          prompt_tokens: 2088,
          completion_tokens: 186,
          cost: 0,
          meta: { citations: CITATIONS },
        },
      ],
      usage: { num_messages: 2, num_tool_runs: 2, prompt_tokens: 2088, completion_tokens: 186, cost: 0 },
    });
  }
  if (/^\/api\/v1\/sessions\/\d+\/doc-scope$/.test(p)) {
    return json({ doc_ids: [DEMO_DOC_ID], note: "演示模式：已关联 1 篇样例文献" });
  }
  if (/^\/api\/v1\/sessions\/\d+\/network-capability$/.test(p)) {
    return json({ enabled: false, available: false, reason: "演示模式不联网" });
  }
  if (/^\/api\/v1\/sessions\/\d+\/messages\/stream$/.test(p) && method === "POST") {
    return sseStream(DEMO_ANSWER);
  }
  if (p === `${BASE}/documents` && method === "GET") return json(DOCS);
  if (/^\/api\/v1\/documents\/\d+\/paragraphs$/.test(p)) return json(DEMO_PARAGRAPHS);
  if (/^\/api\/v1\/documents\/\d+\/apply-edits$/.test(p) && method === "POST") {
    // 演示模式：返回"看起来成功"的结果，但不改任何东西
    return json({ applied: [{ pid: "P9", op: "append" }], rejected: [], total: 1, paragraph_count_after: DEMO_PARAGRAPHS.length, backup: null });
  }
  if (/^\/api\/v1\/documents\/\d+\/export-to$/.test(p) && method === "POST") {
    return json({ path: "（演示模式：未真正写盘）demo_paper_tracked.docx", bytes: 37322 });
  }
  if (p === `${BASE}/collections` && method === "GET") {
    return json([{ id: 1, name: "演示文献集", description: "内置样例数据", created_at: NOW, doc_ids: [1, 2, 3] }]);
  }
  if (p === `${BASE}/memories` && method === "GET") {
    return json([
      { id: 1, content: "我主要研究轻量化目标检测与端侧部署。", scope: "global", created_at: NOW },
    ]);
  }

  return notMocked(p);
}

// --------------------------------------------------------------------------- 安装

/** 在 `main.tsx` **最顶部**调用：演示模式下接管 fetch 并显示一条横幅。 */
export function installDemoApi(): void {
  if (typeof window === "undefined") return; // 单测/SSR 下不安装
  const original = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    if (!path.startsWith(BASE)) return original(input, init);
    const method = (init?.method ?? (typeof input === "object" && "method" in input ? input.method : "GET")) || "GET";
    // 让"正在请求"的动画有真实节奏感
    await sleep(120);
    const res = await demoRoute(path, method.toUpperCase());
    return res ?? original(input, init);
  };

  const banner = document.createElement("div");
  banner.setAttribute("role", "note");
  banner.textContent =
    "演示模式 · 界面为真实前端，数据为内置样例；未连接任何后端，也不需要任何 API key";
  banner.style.cssText = [
    "position:fixed", "left:50%", "transform:translateX(-50%)", "bottom:14px", "z-index:9999",
    "padding:7px 14px", "border-radius:999px", "font-size:12px", "line-height:1.5",
    "background:rgba(20,24,32,.88)", "color:#fff", "box-shadow:0 6px 20px rgba(0,0,0,.25)",
    "pointer-events:none", "max-width:92vw", "text-align:center",
  ].join(";");
  document.body.appendChild(banner);
}
