// REST 客户端封装（docs/04 §4.6.1）。
// 统一处理错误（ApiError）、JSON 解析与表单上传；相对路径经 Vite 代理到后端。

import type {
  AppSettings,
  ApplyEditsOut,
  CollectionOut,
  DocumentImage,
  DocumentOut,
  DocScopeOut,
  DocumentTable,
  EditOp,
  EvalCaseOut,
  EvalSummary,
  FeedbackOut,
  FeedbackSummaryOut,
  InventoryOut,
  MemoryItemOut,
  MetricAlert,
  MetricsOut,
  ParaAction,
  ParaTransformOut,
  ParaView,
  PrecheckResult,
  ProviderBalance,
  NetworkCapabilityOut,
  SessionDetailOut,
  SessionOut,
  SessionUsageDetail,
  StreamItem,
  SummaryOut,
  TraceListItem,
  TraceOut,
  WorkspaceMode,
  WorkspaceOut,
  UploadProgress,
  PromptEnhanceResult,
  UsageRow,
  UsageTrendOut,
} from "../lib/types";

const BASE = "/api/v1";

export class ApiError extends Error {
  /** 非空时为后端错误响应体（如 409 去重的 existing_id/existing_filename），供调用方做分支处理 */
  constructor(public status: number, message: string, public payload?: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * 从错误响应体提取可读消息。
 *
 * ⚠️ `detail` 的类型**必须防御性判断**（2026-09-15 修复）：
 * 本项目后端已统一把 `detail` 压平为字符串（见 `api/errors.py`），但
 * - FastAPI 的**默认**校验错误响应里 `detail` 是**对象数组**；
 * - 反向代理 / 网关返回的错误体格式也不受我们控制。
 *
 * 若直接 `new Error(detail)` 传入数组，JS 会强转成字符串 → 用户看到 `[object Object]`，
 * 完全不知道哪里错了（该项曾被列为后端工程化评审的 P0-1）。
 *
 * 导出仅为单元测试使用（见 `client.test.ts`）。
 */
export function extractDetail(body: unknown, fallback: string): string {
  const raw = (body as { detail?: unknown } | null)?.detail;
  if (typeof raw === "string" && raw) return raw;
  if (Array.isArray(raw)) {
    const msgs = raw
      .map((e) =>
        e && typeof e === "object" && "msg" in e ? String((e as { msg: unknown }).msg) : String(e),
      )
      .filter(Boolean);
    if (msgs.length) return `请求参数不合法：${msgs.join("; ")}`;
  }
  const code = (body as { code?: unknown } | null)?.code;
  if (typeof code === "string" && code) return `${fallback}（${code}）`;
  return fallback;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    let payload: unknown;
    try {
      const body: unknown = await res.json();
      payload = body;
      detail = extractDetail(body, res.statusText);
    } catch {
      /* 响应体非 JSON，保留 statusText */
    }
    throw new ApiError(res.status, detail, payload);
  }
  return (await res.json()) as T;
}

export const api = {
  async listSessions(): Promise<SessionOut[]> {
    return json(await fetch(`${BASE}/sessions`));
  },

  /** 取会话详情（含历史消息与用量），用于切换会话时加载历史 */
  async getSession(id: number): Promise<SessionDetailOut> {
    return json(await fetch(`${BASE}/sessions/${id}`));
  },

  /** 联网可用性（2026-09-12）：联网是模型原生能力且 doc-bound 会回落模型，
   *  「开关已开」≠「能力可用」——前端据此给出可用性预警，避免用户误以为"联网了却不搜"。 */
  async getNetworkCapability(id: number): Promise<NetworkCapabilityOut> {
    return json(await fetch(`${BASE}/sessions/${id}/network-capability`));
  },

  /** 会话检索范围（2026-09-23）：供输入框常驻提示「当前处于哪个场景」——
   *  用户此前看不出"会话有没有绑文档"，于是会把"没绑文档所以走了通用对话"
   *  误判成"文档约束失效"。 */
  async getDocScope(id: number): Promise<DocScopeOut> {
    return json(await fetch(`${BASE}/sessions/${id}/doc-scope`));
  },

  async createSession(title?: string, config?: Record<string, unknown>): Promise<SessionOut> {
    return json(
      await fetch(`${BASE}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title ?? "新会话", config: config ?? {} }),
      }),
    );
  },

  async deleteSession(id: number): Promise<void> {
    const res = await fetch(`${BASE}/sessions/${id}`, { method: "DELETE" });
    if (!res.ok) throw new ApiError(res.status, "删除会话失败");
  },

  async updateSession(id: number, config: Record<string, unknown>): Promise<SessionOut> {
    return json(
      await fetch(`${BASE}/sessions/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      }),
    );
  },

  /**
   * 重命名会话/工作间（2026-10-02）。
   *
   * ★ 走**同一条** PATCH `/sessions/{id}`（后端 `SessionUpdate.title` 已支持）——
   * 刻意**不**新开 `/rename` 端点：工作间与会话是同一张 `chat_sessions` 表，
   * 再加一个端点只会多一处需要同步的分支。
   *
   * ⚠️ **只传 `title`、绝不传 `config`**：`session_service.update` 对两者是
   * `if ... is not None` 各自独立判断（:96-98），所以只传 title 是安全的；
   * 而传 config 会用**前端这份可能已过期的副本**去 merge，
   * 顺手覆盖掉服务端的工作间模式/主文献设置。
   */
  async renameSession(id: number, title: string): Promise<SessionOut> {
    return json(
      await fetch(`${BASE}/sessions/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      }),
    );
  },

  /**
   * 置顶/取消置顶（2026-10-02）。
   *
   * ★ 走**独立端点** `PATCH /sessions/{id}/pin`，不复用 `updateSession`：
   * 置顶要保证 `updated_at` **不变**，而改标题/改配置**应该**刷新它
   * ⇒ 两条路径的 `updated_at` 语义**相反**，无法共用一个合并更新。
   */
  async pinSession(id: number, pinned: boolean): Promise<SessionOut> {
    return json(
      await fetch(`${BASE}/sessions/${id}/pin`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned }),
      }),
    );
  },

  // ---------- A2：工作间（Workspaces） ----------
  /**
   * 原子创建工作间：单事务内建文献集 + 工作间会话（任一失败整体回滚）。
   * - collection_id 为空 → 用 doc_ids 新建文献集；
   * - collection_id 非空 → 关联已有文献集（doc_ids 被忽略，以文献集为准）。
   */
  async createWorkspace(body: {
    name?: string;
    description?: string;
    doc_ids?: number[];
    collection_id?: number | null;
    draft_doc_id?: number | null;
    mode?: WorkspaceMode;
  }): Promise<WorkspaceOut> {
    return json(
      await fetch(`${BASE}/workspaces`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: body.name ?? null,
          description: body.description ?? "",
          doc_ids: body.doc_ids ?? [],
          collection_id: body.collection_id ?? null,
          draft_doc_id: body.draft_doc_id ?? null,
          mode: body.mode ?? "research",
        }),
      }),
    );
  },

  // ---------- C4：改稿工作台 ----------

  /** 段落视图（正文 P{i} + 表格 T..R..C..P{pi}）；仅 .docx */
  async listParagraphs(docId: number): Promise<ParaView[]> {
    return json(await fetch(`${BASE}/documents/${docId}/paragraphs`));
  },

  /** 文档流视图（B 地址按 body 顺序，含表格 cells）——高保真渲染对齐用（P1） */
  async listStream(docId: number): Promise<StreamItem[]> {
    return json(await fetch(`${BASE}/documents/${docId}/stream`));
  },

  /** 段落 AI 改写（扩写/缩写/润色/去重，单轮元任务）；fake/失败时 applied=false 返回原文 */
  async transformText(body: {
    text: string;
    action: ParaAction;
    context_before?: string | null;
    context_after?: string | null;
    session_id?: number | null;
  }): Promise<ParaTransformOut> {
    return json(
      await fetch(`${BASE}/workbench/para-transform`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
  },

  /** 应用编辑变更集（track-changes 写入 + 自动备份）；rejected 逐条带原因 */
  async applyEdits(docId: number, edits: EditOp[]): Promise<ApplyEditsOut> {
    return json(
      await fetch(`${BASE}/documents/${docId}/apply-edits`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ edits }),
      }),
    );
  },

  /** 一键撤销：还原最新 .bak 备份（撤销前对改后状态再备份，可反悔） */
  async restoreBackup(
    docId: number,
  ): Promise<{ restored_from: string; reindex_triggered: boolean }> {
    return json(await fetch(`${BASE}/documents/${docId}/restore-backup`, { method: "POST" }));
  },

  /**
   * B1：往工作间文献集批量加入文档（幂等，已在集内的自动跳过）。
   * 返回加入后的工作间投影，供前端直接刷新。
   */
  async addWorkspaceDocs(sessionId: number, docIds: number[]): Promise<WorkspaceOut> {
    return json(
      await fetch(`${BASE}/workspaces/${sessionId}/documents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_ids: docIds }),
      }),
    );
  },

  /** B1 反向：从工作间文献集移除单一文档关联（不删除源 Document 本身）。 */
  async removeWorkspaceDoc(sessionId: number, docId: number): Promise<WorkspaceOut> {
    return json(
      await fetch(`${BASE}/workspaces/${sessionId}/documents/${docId}`, {
        method: "DELETE",
      }),
    );
  },

  /**
   * B2：PATCH 语义更新工作间配置（主文献/模式）—— 只写入显式出现的字段。
   * `draft_doc_id: null` 表示清除主文献；不传该键表示不动。
   */
  async updateWorkspace(
    sessionId: number,
    patch: { draft_doc_id?: number | null; mode?: WorkspaceMode },
  ): Promise<WorkspaceOut> {
    const body: Record<string, unknown> = {};
    if ("draft_doc_id" in patch) body.draft_doc_id = patch.draft_doc_id ?? null;
    if (patch.mode !== undefined) body.mode = patch.mode;
    return json(
      await fetch(`${BASE}/workspaces/${sessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
  },

  /** 读取工作间配置（从会话 config_json 解析）；非工作间会话 → 404 */
  async getWorkspace(sessionId: number): Promise<WorkspaceOut> {
    return json(await fetch(`${BASE}/workspaces/${sessionId}`));
  },

  async listDocuments(): Promise<DocumentOut[]> {
    return json(await fetch(`${BASE}/documents`));
  },

  // ---------- F5：文献集（Collections） ----------
  async listCollections(): Promise<{ items: CollectionOut[] }> {
    return json(await fetch(`${BASE}/collections`));
  },
  async createCollection(name: string, description = "", docIds: number[] = []): Promise<CollectionOut> {
    return json(
      await fetch(`${BASE}/collections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description, doc_ids: docIds }),
      }),
    );
  },
  async renameCollection(id: number, name: string, description?: string): Promise<CollectionOut> {
    return json(
      await fetch(`${BASE}/collections/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description }),
      }),
    );
  },
  async deleteCollection(id: number): Promise<void> {
    const res = await fetch(`${BASE}/collections/${id}`, { method: "DELETE" });
    if (!res.ok) throw new ApiError(res.status, "删除文献集失败");
  },
  async addCollectionDoc(id: number, docId: number): Promise<void> {
    const res = await fetch(`${BASE}/collections/${id}/documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId }),
    });
    if (!res.ok) throw new ApiError(res.status, "归入文献集失败");
  },
  async removeCollectionDoc(id: number, docId: number): Promise<void> {
    const res = await fetch(`${BASE}/collections/${id}/documents/${docId}`, { method: "DELETE" });
    if (!res.ok) throw new ApiError(res.status, "移出文献集失败");
  },

  /** 文档原始文件 URL（PDF 用 pdf.js/iframe 渲染；docx/txt 请用 documentTextUrl 预览文本） */
  documentFileUrl(docId: number): string {
    return `${BASE}/documents/${docId}/file`;
  },

  /** 文档解析文本 URL（Word/Markdown/TXT 统一文本预览，避免二进制乱码） */
  documentTextUrl(docId: number): string {
    return `${BASE}/documents/${docId}/text`;
  },

  /** 文档格式转换导出 URL（GET /documents/{id}/export?format=md|json|txt；需文档已 process，否则 404） */
  documentExportUrl(docId: number, format: "md" | "json" | "txt" = "md"): string {
    return `${BASE}/documents/${docId}/export?format=${format}`;
  },

  /** 单张提取图片 URL（PDF 解析导出，落盘于 upload_dir/{docId}/images/） */
  documentImageUrl(docId: number, filename: string): string {
    return `${BASE}/documents/${docId}/images/${encodeURIComponent(filename)}`;
  },

  /** PDF 页面栅格化截图 URL（P0-2.3：原图缺失/扫描件整页预览/整页存图兜底）。
   * bbox 为页面坐标 [x0,y0,x1,y1]（PDF 点），省略=整页；dpi 默认 120（上限 300）。 */
  documentPageImageUrl(docId: number, page: number, opts?: { dpi?: number; bbox?: number[] }): string {
    const params: string[] = [];
    if (opts?.dpi != null) params.push(`dpi=${opts.dpi}`);
    if (opts?.bbox?.length === 4) params.push(`bbox=${opts.bbox.join(",")}`);
    const qs = params.length ? `?${params.join("&")}` : "";
    return `${BASE}/documents/${docId}/pages/${page}/image${qs}`;
  },

  /** 列出文档提取的图片资源清单（缩略图用） */
  async listDocumentImages(docId: number): Promise<DocumentImage[]> {
    return json(await fetch(`${BASE}/documents/${docId}/images`));
  },

  /** 列出文档提取的表格清单（表格查看器用，chunk_type=table） */
  async listDocumentTables(docId: number): Promise<DocumentTable[]> {
    return json(await fetch(`${BASE}/documents/${docId}/tables`));
  },

  async uploadDocument(file: File, sessionId?: number): Promise<DocumentOut> {
    const fd = new FormData();
    fd.append("file", file);
    const url = sessionId != null ? `${BASE}/documents?session_id=${sessionId}` : `${BASE}/documents`;
    return json(await fetch(url, { method: "POST", body: fd }));
  },

  /** 异步上传（2026-09-02）：立即返回 status=uploaded 的文档，后端后台解析/向量化。
   * 配合 getUploadProgress / cancelUpload 实现进度可视化与随时取消。 */
  async uploadDocumentAsync(file: File, sessionId?: number): Promise<DocumentOut> {
    const fd = new FormData();
    fd.append("file", file);
    const url = sessionId != null ? `${BASE}/documents/async?session_id=${sessionId}` : `${BASE}/documents/async`;
    return json(await fetch(url, { method: "POST", body: fd }));
  },

  /** 停止当前流式生成（2026-09-07）：置位后端取消事件，确定性终止 LLM 生成。
   *  与前端 abort() 双管齐下——不依赖 ASGI 断连语义。静默容错（连接可能已断）。
   *
   *  F8（2026-09-07）：传 `streamId` 时**只**停止该条流（请求级精确取消，
   *  同会话并发流互不干扰）；省略时后端停止该会话的全部进行中流（兼容 / 兜底语义）。 */
  async stopStream(sessionId: number, streamId?: string): Promise<void> {
    try {
      const qs = streamId ? `?stream_id=${encodeURIComponent(streamId)}` : "";
      await fetch(`${BASE}/sessions/${sessionId}/stop${qs}`, { method: "POST" });
    } catch {
      /* 连接可能已断开，忽略 */
    }
  },

  /** 异步上传进度轮询（GET /documents/{id}/progress）：返回快照；文档已不存在 → null（前端据此停止轮询）。 */
  async getUploadProgress(id: number): Promise<UploadProgress | null> {
    const res = await fetch(`${BASE}/documents/${id}/progress`);
    if (res.status === 404) return null;
    if (!res.ok) throw new ApiError(res.status, "获取上传进度失败");
    return (await res.json()) as UploadProgress;
  },

  /** 提示词增强（POST /prompt/enhance）：把口语输入改写为更明确的任务描述。
   * 关闭（403）/fake/调用失败 → applied=false（enhanced=原文），由调用方提示 reason。 */
  async enhancePrompt(text: string, sessionId?: number): Promise<PromptEnhanceResult> {
    const res = await fetch(`${BASE}/prompt/enhance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: sessionId ?? null }),
    });
    if (!res.ok) {
      // L5 修复（2026-09-10）：403 时后端带 reason（功能被关闭的说明，见 routes_prompt），
      // 原先直接抛笼统「提示词增强失败」→ 用户看不到为什么被拒。尽量解析 body 中的原因。
      let detail = "提示词增强失败";
      try {
        // reason 优先（后端在功能关闭时特意带的中文说明）；
        // 其余情况走 extractDetail —— 避免 detail 为数组时被强转成 [object Object]
        const body = (await res.json()) as { reason?: string };
        detail =
          (typeof body?.reason === "string" && body.reason) ||
          extractDetail(body, "提示词增强失败");
      } catch {
        /* body 非 JSON：保留默认文案 */
      }
      throw new ApiError(res.status, detail);
    }
    return (await res.json()) as PromptEnhanceResult;
  },

  /** 取消正在进行的异步上传（协作式：后台任务在阶段边界/下一批次停止并清理半成品）。幂等。 */
  async cancelUpload(id: number): Promise<{ ok: boolean; cancelled?: boolean; status?: string }> {
    return json(await fetch(`${BASE}/documents/${id}/cancel`, { method: "POST" }));
  },

  /** 上传前重复检测（内容 sha256 + 文件名；供确认弹窗用） */
  async precheckDocument(file: File): Promise<PrecheckResult> {
    const fd = new FormData();
    fd.append("file", file);
    return json(await fetch(`${BASE}/documents/precheck`, { method: "POST", body: fd }));
  },

  async deleteDocument(id: number): Promise<void> {
    const res = await fetch(`${BASE}/documents/${id}`, { method: "DELETE" });
    if (!res.ok) throw new ApiError(res.status, "删除文档失败");
  },

  async getUsage(): Promise<UsageRow[]> {
    return json(await fetch(`${BASE}/usage`));
  },

  /** F10：按天聚合的用量趋势（成本趋势图数据源；与 getUsage 同源，合计可对账） */
  async getUsageTrend(days = 30): Promise<UsageTrendOut> {
    return json(await fetch(`${BASE}/usage/trend?days=${days}`));
  },

  /** F11：忠实度看板数据（检索基线 + 运行期 Agent 指标 + 门禁判定；features.dev_eval 关闭返 403） */
  async getEvalSummary(): Promise<EvalSummary> {
    return json(await fetch(`${BASE}/eval/summary`));
  },

  /** 会话级用量明细（用量页展开查看） */
  async getSessionUsage(sessionId: number): Promise<SessionUsageDetail> {
    return json(await fetch(`${BASE}/usage/sessions/${sessionId}`));
  },

  /** 各厂商账户余额（对话页模型选择器展示） */
  async getBalances(): Promise<Record<string, ProviderBalance>> {
    return json(await fetch(`${BASE}/settings/balances`));
  },

  /** 本地数据管理：工具与技能清单 */
  async getInventory(): Promise<InventoryOut> {
    return json(await fetch(`${BASE}/settings/inventory`));
  },

  async getSettings(): Promise<AppSettings> {
    return json(await fetch(`${BASE}/settings`));
  },

  async updateSettings(body: {
    default_model?: string;
    dedupe_warning?: boolean;
    features?: {
      memory?: boolean;
      summary?: boolean;
      multi_query?: boolean;
      code_interpreter?: boolean;
      skills?: boolean;
      planning?: boolean;
      dev_eval?: boolean;
      thinking?: boolean;
      deep_read?: boolean;
      glossary?: boolean;
      find_code?: boolean;
      research_gap?: boolean;
      experiment_design?: boolean;
      review?: boolean;
      reproduction?: boolean;
    };
    figure_enrich?: { enabled: boolean;  methods?: string[] };
    agent_rules?: string;
    default_discipline?: string;
    /** RAG 检索底座（2026-09-01）：embedding 模式与两枚开关 */
    rag_embedder?: string;
    use_bge_sparse?: boolean;
    use_jieba?: boolean;
    providers?: Record<string, { api_key?: string; base_url?: string; model?: string }>;
  }): Promise<AppSettings> {
    return json(
      await fetch(`${BASE}/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
  },

  async testSettings(provider?: string): Promise<{ ok: boolean; provider?: string; model?: string; reply?: string; error?: string }> {
    return json(
      await fetch(`${BASE}/settings/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(provider ? { provider } : {}),
      }),
    );
  },

  /** 探测『大模型原生联网搜索』是否可用（连通性测试，需 LLM key；满足 LLM 模块硬约束） */
  async testWebSearch(provider?: string): Promise<{ ok: boolean; supported: boolean; provider?: string; model?: string; reply?: string; error?: string }> {
    return json(
      await fetch(`${BASE}/settings/test-web-search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(provider ? { provider } : {}),
      }),
    );
  },

  // ---- A-02：结构化总结 / 相关工作 ----
  async summarize(docIds: number[]): Promise<SummaryOut> {
    return json(
      await fetch(`${BASE}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_ids: docIds }),
      }),
    );
  },

  async relatedWork(docIds: number[]): Promise<SummaryOut> {
    return json(
      await fetch(`${BASE}/related-work`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_ids: docIds }),
      }),
    );
  },

  /** 多篇文档对比（S5）：逐维度并列 + 共性/差异叙述 + 共引分析 */
  async compare(docIds: number[]): Promise<SummaryOut> {
    return json(
      await fetch(`${BASE}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_ids: docIds }),
      }),
    );
  },

  // ---- ER-19：推理轨迹 ----
  async listTraces(sessionId?: number): Promise<TraceListItem[]> {
    const q = sessionId != null ? `?session_id=${sessionId}` : "";
    return json(await fetch(`${BASE}/traces${q}`));
  },

  async getTrace(traceId: string): Promise<TraceOut> {
    return json(await fetch(`${BASE}/traces/${traceId}`));
  },

  // ---- ER-16：长期记忆管理 ----
  async listMemories(): Promise<{ total: number; items: MemoryItemOut[] }> {
    return json(await fetch(`${BASE}/memories`));
  },

  async updateMemory(
    id: string,
    text: string,
    importance?: number,
    scope?: "global" | "session",
  ): Promise<{ ok: boolean; id: string; text: string; importance?: number; scope?: string }> {
    return json(
      await fetch(`${BASE}/memories/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        // importance / scope 缺省时后端保留原值，显式传入才更新
        body: JSON.stringify(
          importance == null && scope == null
            ? { text }
            : scope == null
              ? { text, importance }
              : importance == null
                ? { text, scope }
                : { text, importance, scope },
        ),
      }),
    );
  },

  /** 批量删除记忆（2026-08-29 C 档）：返回实际删除条数与未命中 id */
  async batchDeleteMemories(
    ids: string[],
  ): Promise<{ ok: boolean; removed: number; missing: string[] }> {
    return json(
      await fetch(`${BASE}/memories/batch-delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids }),
      }),
    );
  },

  async deleteMemory(id: string): Promise<{ ok: boolean; removed: number }> {
    const res = await fetch(`${BASE}/memories/${id}`, { method: "DELETE" });
    if (!res.ok) throw new ApiError(res.status, "删除记忆失败");
    return (await res.json()) as { ok: boolean; removed: number };
  },

  /** 全量重建索引（2026-09-01）：切换 embedding 模式 / BGE 稀疏 / 中文分词后调用，使新配置对所有文档生效 */
  async reindexAll(): Promise<{ reindexed: number; failed?: { id: number; error: string }[]; total?: number }> {
    return json(await fetch(`${BASE}/documents/reindex`, { method: "POST" }));
  },

  /** 全量重建索引（SSE 流式）：实时回调 onEvent 用于进度可视化；返回最终汇总。
   *
   * 传入 `signal` 可中途取消：abort 会让 `reader.read()` 抛出 AbortError，由调用方
   * 捕获并转入「已取消」状态。注意**必须**配合 `cancelReindex()`——只 abort 的话
   * 前端断开了、后端仍在逐篇重建，白白烧 CPU/显存。
   */
  async reindexAllStream(
    onEvent: (ev: {
      type: "progress" | "done" | "cancelled";
      doc_id?: number;
      status?: "ok" | "skip" | "error";
      error?: string | null;
      done?: number;
      total?: number;
      reindexed?: number;
      failed?: { id: number; error: string }[];
    }) => void,
    signal?: AbortSignal,
  ): Promise<{ reindexed: number; failed?: { id: number; error: string }[]; total?: number }> {
    const res = await fetch(`${BASE}/documents/reindex/stream`, { method: "POST", signal });
    if (!res.ok || !res.body) throw new ApiError(res.status, "重建索引失败");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let done: { reindexed: number; failed?: { id: number; error: string }[]; total?: number } = { reindexed: 0 };
    for (;;) {
      const { done: finished, value } = await reader.read();
      if (finished) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() ?? "";
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        try {
          const ev = JSON.parse(line.slice(5).trim());
          onEvent(ev);
          if (ev.type === "done" || ev.type === "cancelled") done = ev;
        } catch {
          /* 忽略非 JSON / 心跳行 */
        }
      }
    }
    return done;
  },

  /** 请求取消正在进行中的全量重建索引（协作式：在文档边界生效）。
   *
   * 幂等——当前没有进行中的重建时调用也无副作用。调用后前端还应 abort 掉 SSE 连接，
   * 否则后端虽已停止、前端仍会停在「进行中」。
   */
  async cancelReindex(): Promise<{ ok: boolean }> {
    return json(await fetch(`${BASE}/documents/reindex/cancel`, { method: "POST" }));
  },

  async clearMemories(): Promise<{ ok: boolean; removed: number }> {
    const res = await fetch(`${BASE}/memories/clear`, { method: "POST" });
    if (!res.ok) throw new ApiError(res.status, "清空记忆失败");
    return (await res.json()) as { ok: boolean; removed: number };
  },

  /** E1：提交一条反馈（rating 只能是 1 或 -1）。同一消息可改分，后端取最后一次为准。 */
  async submitFeedback(input: {
    session_id: number;
    message_id?: number | null;
    trace_id?: string;
    rating: number;
    comment?: string;
  }): Promise<FeedbackOut> {
    const res = await fetch(`${BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
    if (!res.ok) throw new ApiError(res.status, "提交反馈失败");
    return (await res.json()) as FeedbackOut;
  },

  /** E1：反馈汇总（含每条消息当前评分，用于刷新后回填按钮高亮）。 */
  async feedbackSummary(sessionId?: number): Promise<FeedbackSummaryOut> {
    const qs = sessionId === undefined ? "" : `?session_id=${sessionId}`;
    return json(await fetch(`${BASE}/feedback/summary${qs}`));
  },

  /** E2：导出单条轨迹为评测样本。**gold 为空** —— 需人工补后才能进评测集。 */
  async exportTraceCase(traceId: string): Promise<EvalCaseOut> {
    return json(await fetch(`${BASE}/traces/${encodeURIComponent(traceId)}/export`));
  },

  /** D1：运行指标（含 A6 阈值告警）。days=统计窗口。 */
  async getMetrics(days = 7, sessionId?: number): Promise<MetricsOut> {
    const params = new URLSearchParams({ days: String(days) });
    if (sessionId !== undefined) params.set("session_id", String(sessionId));
    return json(await fetch(`${BASE}/metrics?${params.toString()}`));
  },

  /** E2：批量导出为 jsonl 文本（可直接存成 eval case 文件）。 */
  async exportTraceCasesJsonl(sessionId?: number, limit = 20): Promise<string> {
    const params = new URLSearchParams({ fmt: "jsonl", limit: String(limit) });
    if (sessionId !== undefined) params.set("session_id", String(sessionId));
    const res = await fetch(`${BASE}/traces/export?${params.toString()}`);
    if (!res.ok) throw new ApiError(res.status, "导出评测样本失败");
    return await res.text();
  },
};
