// 前端与后端（docs/04 §4.6 契约）对齐的类型定义。
// 仅描述 UI 关心的字段，后端 Pydantic Schema 见 src/papermate/schemas.py。

export type Role = "user" | "assistant";

/** 引用溯源（来自 SSE `message` 事件的 citations，由 RAG 检索结果抽取） */
export interface Citation {
  id?: string | number | null;
  page?: number | null;
  /** 所属文档 id：用于映射为「文档一/文档二」友好名（与右侧预览面板 docLabel 一致） */
  doc_id?: number | null;
  text?: string | null;
  /** B2（2026-09-29）：该片段在**注入模型**时被截断（模型可能只看到片段开头）。
   *  引用文本取自工具**完整**输出，而注入的是 `retrieve` 的 200 字/段压缩稿 ⇒ 两者不同源。
   *  为 true 时卡片给出 muted 标记 + 脚注说明（**不**用告警色，避免把正常回答染成"可疑"）。 */
  model_truncated?: boolean;
}

/** 工具调用视图（thinking/tool_call/tool_result 三个事件组合渲染） */
export interface ToolCallView {
  id: string;
  name: string;
  args: unknown;
  state: "running" | "ok" | "error";
  result?: unknown;
  error?: string;
}

/** 「打断提醒」机制：无效回复（缺文档 / 兜底 / 检索故障 / 循环风险 / 输入超长）的结构化提示（docs/打断提醒机制） */
export interface Notice {
  /**
   * missing_doc=未关联文档；fallback=模型调用失败/空回复；
   * retrieval_error=检索服务不可用（全部管线失败，区别于「文档无相关内容」）；
   * loop_risk=尾部连续无效达阈值强打断；
   * stream_failed=生成中途异常（回答可能不完整，刷新后仍可见）；
   * input_too_long=单条消息超过输入预算，已提前拦下（未消耗额度）
   */
  kind:
    | "missing_doc"
    | "fallback"
    | "retrieval_error"
    | "loop_risk"
    | "stream_failed"
    | "input_too_long";
  /** 标题（后端已带 ⚠️ 前缀，前端直接展示） */
  title: string;
  /** 原因说明 */
  message: string;
  /** 给用户的下一步可执行建议 */
  suggestion: string;
}

/** `done` 事件携带的用量元数据 */
export interface DoneMeta {
  steps: number;
  total_tokens: number;
  /** ★ P1-a（2026-10-03）：本轮**最大单次** prompt。占用率看的是**单次峰值**，
   *  与 `total_tokens`（整轮累计）严格不同。 */
  peak_prompt_tokens?: number;
  /** ★ P1-a：单次 prompt 的**分母**（后端 `max_tokens_budget`）。
   *  ⚠️ 老后端可能不带 ⇒ 前端必须**优雅降级**（不画占用条），不得报错。 */
  max_tokens_budget?: number;
  /** ★ P1-a：**警示线**（后端 `compact_trigger_tokens`，= 上限 × 0.7）。
   *  ★ 由后端下发而非前端乘 0.7 —— 否则前端复制派生逻辑 = 再造一把尺子。 */
  compact_trigger_tokens?: number;
  /** ★ P2（2026-10-03）：上下文**分项构成**（按来源：system/tool_defs/messages/
   *  tool_results/notes），供「上下文用量」面板显示"哪一块最大"。
   *  ⚠️ 值为**估算** token；未跑过 step 时为空对象 ⇒ 面板只显示总占用率。 */
  context_breakdown?: Record<string, number>;
  // 成本字段在仅缺文档协议级拦截时为 0（后端 done 不带 cost，前端做兜底）
  cost: number;
  latency_ms: number;
  /** F6：本轮解析后的学科 key（如 clinical / general），用于顶栏「已识别为：X」反馈 */
  discipline?: string;
  /** F6：学科中文标签（如 医学/临床），frontend 直接展示 */
  discipline_label?: string;
  /** F6：本轮学科是否为 auto 自动识别（仅 auto 才显示识别提示，手动指定不提示） */
  discipline_auto?: boolean;
  /** 打断提醒：仅在无效回复（缺文档/兜底/循环风险）时由 done 事件携带 */
  notice?: Notice | null;
  // ---- A7（2026-09-28）诊断字段：隐藏 LLM 调用（历史摘要/兜底收尾/失败自评）----
  // 上面 `total_tokens` / `cost` / `latency_ms` 已**包含**这些调用（统计同源），
  // 这里给出分项，便于核对"这轮到底有没有隐藏调用、占了多少"。
  // ⚠️ 下划线前缀 = **仅供诊断，不参与 UI 展示**（保持与后端同名的可 grep 性）。
  _hidden_calls?: number;
  _hidden_tokens?: number;
  _hidden_cost?: number;
  // ---- C4/C6/B5（2026-09-29）诊断字段：上下文"被动刀" + 引用截断 ----
  // 闸门裁剪 / 历史压缩此前只写日志、不进事件与 trace ⇒ 无法知道"这轮证据被裁过"；
  // 而 judge 只测 faithfulness/abstention，**裁剪造成的完整性损失测不出来**（判据盲区）。
  /** 本轮循环内闸门**实际替换**掉的早期工具结果条数（0 = 未动过） */
  _gate_hits?: number;
  /** 本轮跨轮压缩**实际丢弃**的历史消息条数（0 = 未丢） */
  _compacted_dropped?: number;
  /** 引用是否因达上限（60 条）被截断（true = 用户少看了若干条引用） */
  _citations_capped?: boolean;
}

export interface ErrorInfo {
  message: string;
  recoverable?: boolean;
  code?: string;
}

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  /** 多模态（DeepSeek-V4-Flash-Vision）：随本消息发送的图片 data URL（仅展示用，发送后即随请求提交） */
  images?: string[];
  /** 消息时间（ISO）：气泡下方 HH:MM 展示；历史来自后端，实时发送取本地时刻 */
  created_at?: string | null;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  thinking: string[];
  /** V4 思考模式：流式累积的推理链文本（thinking 事件 reasoning 增量拼接） */
  reasoning?: string;
  toolCalls: ToolCallView[];
  text: string;
  citations: Citation[];
  /** P1-2.1 回复逐条核验结果（SSE `verify` 事件，只读展示，不改写文本） */
  verify?: VerifyResult;
  status: "streaming" | "complete" | "error";
  done?: DoneMeta;
  /** ★ P1-a2（2026-10-03）：**历史回填**的上下文占用（来自 `messages.meta`）。
   *
   * ⚠️ 刻意**不复用 `done`**：`DoneMeta` 的 `steps/total_tokens/cost/latency_ms` 是**必填**，
   * 而历史接口（`MessageOut`）根本没这些 ⇒ 硬塞进 `done` 就得**编造**字段值（伪造数据）。
   * ⇒ 占用率走独立字段，两条来源（实时 `done` / 历史 `meta`）在**渲染处合并**。
   *
   * 不填的场景（**降级，非缺陷**）：技能路径 events 收集器为空（见 B2-X1）、
   * 直答路径不走 loop（无峰值）、老数据（meta 里没有 ctx_*）。
   */
  ctx?: { peak?: number; budget?: number; warnAt?: number; breakdown?: Record<string, number> };
  error?: ErrorInfo;
  /** 打断提醒：流式 done 事件上挂 / 历史从 meta 回填。存在即渲染醒目警示卡。 */
  notice?: Notice | null;
  /** 历史回填：该条是否曾被标记为无效回复（来自 messages.meta.invalid） */
  invalid?: boolean;
  /** 模型风险标记（2026-09-16，来自 messages.meta.model_risk）：
   *  该轮所用模型在「绑定文档」场景可能跳过检索 —— **不代表本轮无效**，只是提示核对引用 */
  modelRisk?: boolean;
  /** 消息时间（ISO）：气泡下方 HH:MM 展示；历史来自后端，流式取开始时刻 */
  created_at?: string | null;
  /** O-12 档 1：首字之前的**进度提示**（SSE `progress` 事件）。
   *  只在 doc-bound 首轮的静默窗口存在（实测首字中位 ≈4s），首个正文增量到达即清除。
   *  刻意与 `thinking`/`text` 分开：不进推理步数计数、不进回答正文、不落库。 */
  progress?: { phase: string; text: string } | null;
}

export type ThreadItem = UserMessage | AssistantMessage;

/** 后端 SSE 原始帧（event: / data: 行） */
export interface SSEFrame {
  event?: string;
  data?: string;
  id?: string;
  retry?: string;
}

/** 后端 REST 响应（与 schemas.py 对齐） */
export interface SessionOut {
  id: number;
  title: string;
  config: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  /** 侧栏置顶（2026-10-02，后端迁移 5）。老接口/老数据一律 `false`。
   *  ★ 排序由**后端**保证（`ORDER BY pinned DESC, updated_at DESC`）⇒
   *    前端 `filter` 分组时**保持原序**即可，**不要在前端另排**。 */
  pinned?: boolean;
}

/** 工作间模式：research（研读）| revision（改稿） */
export type WorkspaceMode = "research" | "revision";

/** C4：段落视图（与后端 GET /documents/{id}/paragraphs 对齐） */
export interface ParaView {
  pid: string;
  style: string | null;
  text: string;
  empty: boolean;
  editable: boolean;
  reason: string | null;
}

/** C4：单条 docx 编辑操作（与后端 EditOp 对齐；purge=物理移除空壳段，仅限已标记删除的正文段落） */
export interface EditOp {
  pid: string;
  op: "replace" | "append" | "insert_after" | "delete" | "purge";
  text: string;
  quote?: string | null;
}

/** C4：apply-edits 响应 */
export interface ApplyEditsOut {
  applied: { pid: string; op: string }[];
  rejected: { pid: string; op: string; reason: string }[];
  total: number;
  paragraph_count_after: number;
  backup: string | null;
  reindex_triggered?: boolean;
}

/** P1：文档流视图 —— 单元格条目 */
export interface StreamCell {
  pid: string;
  style: string | null;
  text: string;
  empty: boolean;
  editable: boolean;
  reason: string | null;
}

/** P1：文档流视图条目（GET /documents/{id}/stream）——高保真渲染按序对齐用 */
export interface StreamItem {
  kind: "para" | "table";
  pid: string;
  style?: string | null;
  text?: string;
  empty?: boolean;
  editable?: boolean;
  reason?: string | null;
  /** 段落标记已被标记删除（pPr/rPr/w:del）——标记删除后的空壳段，可「彻底删除」 */
  deleted?: boolean;
  cells?: StreamCell[];
}

/** 段落 AI 改写动作（POST /workbench/para-transform） */
export type ParaAction = "expand" | "shorten" | "polish" | "dedupe";

export interface ParaTransformOut {
  original: string;
  transformed: string;
  applied: boolean;
  action?: ParaAction;
  action_label?: string;
  reason?: string | null;
}

/**
 * A2：工作间（与后端 WorkspaceOut 对齐）。
 * 工作间 = 一个特殊会话（config.workspace=true）+ 一个底层文献集，零额外表。
 */
export interface WorkspaceOut {
  session_id: number;
  collection_id: number | null;
  name: string;
  mode: WorkspaceMode;
  /** 主文献 / 草稿；B2 起由用户在右文件区指定 */
  draft_doc_id: number | null;
  doc_ids: number[];
}

/**
 * 会话是否为工作间：由 sessions[].config.workspace 派生。
 * 左栏分组据此分流，**无需额外的列表请求**。
 */
export function isWorkspace(s: Pick<SessionOut, "config"> | null | undefined): boolean {
  return s?.config?.workspace === true;
}

/** F5：文献集（Collections）——按课题/项目组织的文档分组 */
export interface CollectionOut {
  id: number;
  name: string;
  description: string;
  created_at: string | null;
  doc_ids: number[];
}

export interface MessageOut {
  id: number;
  role: "user" | "assistant";
  content: string;
  tool_calls?: unknown;
  /** 多模态历史消息的图片 data URL 列表（内容为纯文本，图片单独返回） */
  images?: string[];
  prompt_tokens: number;
  completion_tokens: number;
  cost: number;
  /** 打断提醒元数据（历史消息回填用）：{ invalid?: boolean; notice?: Notice } */
  /** ★ P1-a2（2026-10-03）：`ctx_*` = 上下文占用（峰值 / 分母 / 警示线），
   *  用于**刷新后回填**占用率 —— 否则"当轮看得到、刷新即消失"。 */
  meta?: {
    invalid?: boolean;
    notice?: Notice;
    model_risk?: boolean;
    ctx_peak?: number;
    ctx_budget?: number;
    ctx_warn_at?: number;
    /** ★ P2：分项构成（用于刷新后回填「上下文用量」面板） */
    ctx_breakdown?: Record<string, number>;
  } | null;
  /** B2-X1（2026-10-01）：本轮引用。后端从 `messages.meta_json.citations` 提升为独立字段，
   *  供**刷新/切换会话后**回填引用卡片（此前前端拿不到 ⇒ 引用"当轮可见、刷新即消失"）。 */
  citations?: Citation[];
  /** 消息时间（UTC ISO）：前端展示 HH:MM */
  created_at?: string | null;
}

export interface SessionDetailOut extends SessionOut {
  messages: MessageOut[];
  usage: {
    num_messages: number;
    num_tool_runs: number;
    prompt_tokens: number;
    completion_tokens: number;
    cost: number;
  };
}

/** 会话「联网可用性」（GET /sessions/{id}/network-capability，2026-09-12）
 *
 *  联网是**模型原生能力**（enable_search），且 doc-bound 会话可能**回落**到不支持联网的模型，
 *  因此「开关已开」≠「能力可用」——前端据此显示预警，避免用户误以为"联网了却不搜"。 */
/** 会话「检索范围」（GET /sessions/{id}/doc-scope，2026-09-23）
 *
 *  三场景（无文档 / 单文档 / 多文档）下系统的检索行为不同，但界面上**看不出**自己处于哪个场景 ——
 *  用户会误判「约束没生效」（最常见的是：会话其实**没绑文档**，于是走了通用对话）。
 *  本类型供前端常驻提示「我现在处于哪个场景、本轮会不会检索论文」。
 *
 *  ⚠️ 只含**会话级**范围：本轮是否因 `@` 聚焦属每轮变化，由前端自行标注。 */
export interface DocScopeItem {
  id: number;
  filename: string;
}

export interface DocScopeOut {
  session_id: number;
  /** 关联文档数：0 = 通用对话；1 = 单文档；≥2 = 多文档 */
  doc_count: number;
  /** id 降序（与「文档N」标签同口径） */
  doc_ids: number[];
  documents: DocScopeItem[];
  /** 有文档则强制检索（协议级 tool_choice） */
  force_retrieval: boolean;
  /** general（0 篇）/ single（1 篇）/ multi（≥2 篇，默认检索全部） */
  mode: "general" | "single" | "multi";
}

export interface NetworkCapabilityOut {
  session_id: number;
  /** 会话「允许联网」开关状态 */
  enabled: boolean;
  /** 开关已开**且**当前实际模型支持原生联网 */
  available: boolean;
  /** 当前实际模型是否支持原生联网 */
  supports_native: boolean;
  /** 本轮实际使用的模型（2026-09-16 起恒等于用户所选——doc-bound 静默回落已取消） */
  effective_model: string;
  /** 不可用时的原因说明（用于 UI 预警文案） */
  reason: string;
  /** true = 「绑定文档 + 所选模型」组合存在风险：可能跳过检索、直接编造文档内容。
   *  发送**不会被拦截**，但 UI 应常驻提示「请核对引用」（2026-09-16 定：警示但放行） */
  model_risk?: boolean;
  /** 风险说明（含推荐模型，可直接展示给用户） */
  model_risk_reason?: string;
}

/** A6：一条告警。带 threshold / sample —— 只有 value 的红灯没法判断严重性。 */
export interface MetricAlert {
  code: string;
  level: string;
  value: number;
  threshold: number;
  sample: number;
  message: string;
}

/** D1：运行指标（零依赖 JSON，非 Prometheus）。 */
export interface MetricsOut {
  window_days: number;
  since: string | null;
  session_id: number | null;
  num_traces: number;
  num_traces_total: number;
  num_answers: number;
  num_sessions: number;
  avg_steps: number;
  avg_latency_ms: number;
  max_latency_ms: number;
  avg_cost: number;
  total_cost: number;
  /** ⚠️ 语义是「未收尾率」（含用户主动停止），**不是**错误率 */
  unfinished_rate: number;
  tool_error_rate: number;
  tool_calls: number;
  /** A4：失败归因分布（分类 → 次数）。`unknown` 非零 = 存在打点缺口，需补 */
  failure_kinds: Record<string, number>;
  /** A4：排除「用户取消」后的真实失败分布（取消不是故障） */
  failure_kinds_excluding_cancel: Record<string, number>;

  // ---- 上下文水位（2026-10-02）：让"额度够不够"事发前可见 ----
  /** 当前**生效**的单次请求上下文上限（用户可在设置页改，不是代码默认值） */
  ctx_budget: number;
  /** 派生触发线 = ctx_budget × 0.7（压缩 + 闸门在这条线动作） */
  ctx_trigger: number;
  /** 本轮**最大单次** prompt 的中位数。⚠️ `null` = 尚未采到样本，**不是 0** */
  ctx_peak_p50: number | null;
  /** 同上的峰值 */
  ctx_peak_max: number | null;
  /** ★ 余量 = ctx_budget − ctx_peak_p50（负数 = 常在越线） */
  ctx_headroom_median: number | null;
  /** 有 peak 字段的轨迹数 / 缺失数（老轨迹未埋 ⇒ 缺口占多数，属正常） */
  ctx_peak_samples: number;
  ctx_peak_missing: number;
  /** 越过触发线 / 直接越熔断线的轨迹数（>0 即说明发生过） */
  ctx_over_trigger: number;
  ctx_over_budget: number;
  /** 「被动过刀」的轨迹数（闸门裁剪 / 历史压缩）——那部分损失 judge 测不出 */
  ctx_elided_traces: number;

  feedback: { total: number; positive: number; negative: number; good_rate: number };
  thresholds: Record<string, number>;
  min_samples: Record<string, number>;
  alerts: MetricAlert[];
  healthy: boolean;
}

export interface DocumentMeta {
  num_blocks?: number;
  num_chunks?: number;
  num_tables?: number;
  num_images?: number;
  extraction_warnings?: string[];
  [key: string]: unknown;
}

export interface DocumentOut {
  id: number;
  filename: string;
  status: string;
  meta: DocumentMeta;
  num_chunks: number;
  /** F9：创建时间（研究首页「最近文献」排序用；老构造路径可能为 null） */
  created_at: string | null;
}

/** 异步上传进度快照（GET /documents/{id}/progress） */
export interface UploadProgress {
  doc_id: number;
  status: string;
  stage: string;
  done: number;
  total: number;
  pct: number;
  error: string | null;
  updated_at: number;
}

/** 提示词增强结果（POST /prompt/enhance）：applied=false 表示未实际增强（fake/失败，enhanced 为原文） */
export interface PromptEnhanceResult {
  original: string;
  enhanced: string;
  applied: boolean;
  reason: string | null;
}

/** 文档提取的图片资源清单（GET /documents/{id}/images） */
export interface DocumentImage {
  filename: string;
  page: number | null;
  kind: string; // figure | image_table
  caption: string;
}

/** 文档提取的表格清单（GET /documents/{id}/tables，chunk_type=table） */
export interface DocumentTable {
  page: number | null;
  markdown: string;
  rows: number;
  cols: number;
  bbox: number[] | null;
  header: string[];
}

/** 图片内容增强（ER-03 Level 2）开关 + 依赖可用性（设置页展示用） */
export interface FigureEnrichConfig {
  enabled: boolean;
  methods: string[]; // ocr | table | vlm
}
export interface FigureEnrichDeps {
  ocr: boolean;
  table: boolean;
  vlm: boolean;
}

/** P1-2.1 回复核验：单条问题（只读展示，不改写回复） */
export interface VerifyIssue {
  severity: "ok" | "warn" | "suspect";
  scope: "citation" | "reply";
  page?: number | null;
  text_snippet?: string | null;
  reason: string;
}
/** P1-2.1 回复核验：SSE `verify` 事件负载 */
export interface VerifyResult {
  /** 实际执行到的核验级别：1=确定性（引用↔页码），2=含 LLM 语义核验 */
  level: number;
  /** L2 状态（level=2 时存在）：ok | issues | skipped */
  l2?: "ok" | "issues" | "skipped";
  /** 汇总：issues 为空且无跳过 → ok；有任何存疑 → issues */
  status: "ok" | "issues" | "skipped";
  issues: VerifyIssue[];
}

export interface UsageRow {
  session_id: number;
  title: string;
  num_messages: number;
  num_tool_runs: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost: number;
  /** F10：会话生效模型（显式指定 > 全局默认，与 build_agent 优先级一致） */
  model: string;
  /** F10：最近一条消息时间（ISO）；无消息时为 null */
  last_at: string | null;
}

/** F10：单日用量趋势点（与 UsageRow 同源，均来自 messages 表） */
export interface UsageTrendPoint {
  date: string;
  cost: number;
  prompt_tokens: number;
  completion_tokens: number;
}

/** F10：按天聚合的用量趋势（GET /usage/trend?days=N，成本趋势图数据源） */
export interface UsageTrendOut {
  days: number;
  points: UsageTrendPoint[];
  total_cost: number;
}

/** F11：端到端忠实度看板（GET /eval/summary；指标计算与 /eval 技能共用 collect_metrics） */
export interface EvalSummary {
  retrieval: {
    available: boolean;
    /** 报告文件名（如 report_real.json）—— 让人知道这些数字是哪份报告来的 */
    file: string | null;
    /** 口径是否可信（真实嵌入 + 线上检索 profile + 指纹新鲜）。false 时门禁不出结论 */
    trusted: boolean;
    /** 不可信的原因（缺项清单），trusted=true 时为空串 */
    reason: string;
    embedder: string | null;
    profile: string | null;
    cases_file: string | null;
    num_cases: number | null;
    "avg_recall@5": number | null;
    "avg_mrr@10": number | null;
    /** 分组指标：单 gold 下 recall 饱和，分辨力在 multi_gold / nDCG */
    by_kind: Record<string, { n: number | null; "recall@5": number | null; "ndcg@5": number | null }>;
  };
  gate: { threshold: number; passed: boolean | null; label: string };
  agent: {
    available: boolean;
    count: number;
    tool_success_rate: number | null;
    citation_rate: number | null;
    avg_cost: number;
    /** 采样样本中有计价（cost>0）的条数 —— 用于解释"平均成本 ¥0"是真免费还是没采集 */
    costed_count: number;
    avg_steps: number;
    avg_latency_ms: number;
    /** 采样窗口：分母排除空轨迹后的实际样本与时间范围（避免"近期指标"其实是几天前的） */
    window: {
      limit: number;
      sampled: number;
      table_total: number;
      excluded_empty: number;
      from: string | null;
      to: string | null;
      /** 最新一条距现在的小时数（**后端按 UTC 口径算好**；前端不得自行解析裸时间戳） */
      age_hours?: number | null;
      /** 超过多少小时算"过时"（后端常量，随响应下发以便如实展示阈值） */
      max_age_hours?: number;
      stale?: boolean;
      /** 窗口未填满（表内非空样本 < limit） */
      insufficient?: boolean;
    } | null;
    traces: { trace_id: string; steps: number; cost: number }[];
  };
  recent_traces: { trace_id: string; steps: number; cost: number }[];
}

/** 会话级用量明细（GET /usage/sessions/{id}） */
export interface SessionUsageDetail {
  session_id: number;
  title: string;
  usage: { num_messages: number; num_tool_runs: number; prompt_tokens: number; completion_tokens: number; cost: number; model: string; last_at: string | null };
  /** `model` = 该轮**实际生效**的模型（meta_json.model）；修复前落库的老消息为 null */
  messages: { id: number; role: string; content: string; prompt_tokens: number; completion_tokens: number; cost: number; model: string | null }[];
  tool_runs: { id: number; tool_name: string; args: string; result_summary: string; ok: boolean }[];
  traces: { trace_id: string; total_steps: number; total_cost: number; total_latency_ms: number; event_summary: string[] }[];
}

/** 各厂商账户余额（GET /settings/balances） */
export interface ProviderBalance {
  available: boolean;
  total_balance?: number;
  granted_balance?: number;
  currency?: string;
  reason?: string;
}

/** 本地数据管理（GET /settings/inventory） */
export interface InventoryOut {
  /** enabled=false = 对应功能关停、未注册给 Agent（灰显展示） */
  tools: { name: string; description: string; permission: string; enabled: boolean }[];
  skills: { name: string; commands: string[]; description: string; feature: string | null; enabled: boolean; icon?: string; category?: string }[];
  permissions: string[];
  sandbox_enabled: boolean;
}

/** 单个厂商的公开设置（不含真实 key，已脱敏） */
export interface ProviderPublic {
  configured: boolean;
  key_masked: string;
  base_url: string;
  /** 该厂商当前选中的具体模型 id（空→厂商默认） */
  model: string;
  /** 该厂商可选模型目录（设置 key 后可选模型用） */
  available_models: string[];
}

// ── 存储位置（2026-10-07）──────────────────────────────────────────
// ★ 字段必须与 `schemas.py::StorageOut` **逐个对应**：后端 `response_model`
//   会把未声明的字段**静默裁掉**（本项目已踩过：表现为"后端返回了、前端读不到"）。
export interface StorageDrive {
  root: string;
  free_gb: number;
  total_gb: number;
  /** FIXED / REMOVABLE / REMOTE / CDROM / RAMDISK / UNKNOWN */
  kind: string;
}

export interface StorageSize {
  mb: number;
  /** true = 因文件数超限只统计了一部分，数值是**下界** */
  truncated: boolean;
}

export interface StorageOut {
  available: boolean;
  unavailable_reason: string | null;
  config_path: string | null;
  data_root: string;
  pending_root: string | null;
  migrate: boolean;
  /** 四个子目录的绝对路径：data / huggingface / venv / uv-cache */
  dirs: Record<string, string>;
  /** 只含 data 与 huggingface（venv / uv-cache 文件太多，量起来慢） */
  sizes_mb: Record<string, StorageSize>;
  drives: StorageDrive[];
  min_free_gb: number;
  restart_required: boolean;
}

export interface StorageUpdateIn {
  data_root: string;
  migrate: boolean;
}

export interface StorageUpdateOut {
  ok: boolean;
  restart_required: boolean;
  warnings: string[];
}

/** 后端 GET /api/v1/settings 返回的前端可填设置 */
export interface AppSettings {
  default_model: string;
  dedupe_warning: boolean; // 重复上传提醒开关（管理员可配，2026-08-20）
  /** 功能开关：控制涉及调用大模型 API 的模块是否启用（docs/统一方案 阶段B/C/D）
   * - memory: S3 跨会话记忆
   * - summary: S4 一键总结
   * - multi_query: P1-6 多查询融合检索增强
   * - code_interpreter: S8 沙箱代码执行（NFR 默认关闭）
   * - skills: 命令触发的复合技能（/综述 /相关工作 /复现）
   * - planning: S9 复杂规划（plan_executor 工具 + /规划 命令）
   * - dev_eval: S10 开发评测（/eval 命令，读检索基线 + 近期 Trace）
   * - thinking: V4 思考模式（展示推理链；开启时不强制 tool_choice） */
  features?: {
    memory: boolean;
    summary: boolean;
    multi_query: boolean;
    code_interpreter: boolean;
    skills: boolean;
    planning: boolean;
    dev_eval: boolean;
    thinking: boolean;
    /**
     * P1-2.1 回复逐条核验。**L1（引用↔页码确定性比对）自 2026-09-16 起常开**，
     * 本开关只控制 L2（+1 次 LLM 语义核验）。只标存疑、不自动改写回复。
     */
    verify_reply: boolean;
    /**
     * 失败轮次自评（2026-09-16 接线 Reflector）：只在循环中止路径（迭代耗尽/连续无进展）
     * 自评一次，判 NEEDS_IMPROVEMENT 则带建议重生成一次；正常收敛零额外调用。
     */
    reflect_on_abort: boolean;
    /** P1 结构图识别：多模态模型把矢量结构图还原为 Mermaid + 一句话描述（默认关闭，开启后对后续上传生效） */
    diagram_enrich: boolean;
    /** P0 单篇精读（/精读 命令技能） */
    deep_read: boolean;
    /** P0 术语表（/术语表 命令技能） */
    glossary: boolean;
    /** P1 找代码（/找代码 命令技能） */
    find_code: boolean;
    /** P1 研究缺口（/研究缺口 命令技能） */
    research_gap: boolean;
    /** P2 实验设计（/实验设计 命令技能） */
    experiment_design: boolean;
    /** P2 审稿（/审稿 命令技能） */
    review: boolean;
    /** S8 复现检索（/复现 命令技能，抽取超参 + LLM 三态复现清单） */
    reproduction: boolean;
    /** 改稿（/改稿 命令技能，对照参考文献逐节给修改建议 + 整体报告） */
    revision: boolean;
    /** 提示词增强（输入框把口语需求改写为更明确的任务描述，单轮元任务） */
    prompt_enhance?: boolean;
    /** 段落 AI 改写（改稿工作台：扩写/缩写/润色/去重，单轮元任务） */
    para_transform?: boolean;
  };
  /** 图片内容增强（ER-03 Level 2）：OCR / 表格结构 / VLM 图注（默认关闭，开启后对后续上传生效） */
  figure_enrich?: FigureEnrichConfig;
  /** 图片增强依赖可用性探测（设置页提示：哪些能力开启后真正生效） */
  figure_enrich_deps?: FigureEnrichDeps;
  /** 本地长期记忆：用户设定的全局规则/偏好（设置页可编辑，直接影响所有会话） */
  agent_rules?: string;
  /** 单次请求上下文上限（2026-09-17 语义修正）：单轮 prompt 超过它即中止本轮（防上下文爆炸），
   * 它同时是历史压缩阈值的来源（//2）。绑多篇文档时单轮 prompt 实测可达 2.5 万，故默认 30000。 */
  max_tokens_budget?: number;
  /** 整轮累计 token 上限（含所有迭代；2026-09-17 新增独立项）：防"跑飞"的粗护栏，默认 150000。
   * 真正的钱袋子护栏是 cost_limit_per_turn（¥/轮）。 */
  max_tokens_per_turn?: number;
  /** 记忆向量引擎（2026-08-29 B 档）：hash=轻量哈希（默认） / bge-m3=真实语义向量（需 rag 依赖） */
  memory_embedder?: string;
  /** RAG 检索底座（2026-09-01 设置页可切换）：
   * - rag_embedder: fake=哈希占位（秒级·零依赖·无真实语义） / real=BGE-M3 真实语义（需 rag 依赖与 ~2.2GB 权重）
   * - use_bge_sparse: 是否启用 BGE-M3 稀疏向量（与 BM25 并行的词法信号）
   * - use_jieba: 中文 BM25 是否用 jieba 按词切分（否则字符级兜底） */
  rag_embedder?: string;
  use_bge_sparse?: boolean;
  use_jieba?: boolean;
  /** T-03 重排推理加速（2026-09-24）：
   * - rag_reranker_fp16: Cross-Encoder 半精度推理（**仅 CUDA 生效**，CPU 自动回落 fp32）
   * - rag_reranker_max_length: 送重排的最大 token 数；**0 = 不限长**（改动前行为）。
   *   实测（4 候选 / RTX 4070 Laptop）：384+fp16 = 48.1ms vs 不限长+fp32 = 208.7ms（×4.34），Top-5 完全一致。 */
  rag_reranker_fp16?: boolean;
  rag_reranker_max_length?: number;
  /** 学科视角默认（2026-08-29 P1）：general/clinical/law/finance/bioscience，控制技能 LLM 解读框架 */
  default_discipline?: string;
  /** 本地存储路径（只读展示） */
  data_dir?: string;
  providers: Record<string, ProviderPublic>;
}

/** —— 真实语义检索模型的**显式安装**（2026-10-05 桌面客户端）——
 * App 装完默认 fake（绝不静默拉 ~4.5GB 权重）；用户在设置页主动点下载，
 * 下载完**也不自动启用** —— 切成 real 会让既有索引失效，需「手动启用 + 重建索引」。 */
export interface RagModelItem {
  key: string;              // embedder | reranker
  repo_id: string;
  label: string;
  /** 预计体积（GB，仅用于提示，非精确值） */
  approx_gb: number;
  /** 本机是否已有可用快照（判定口径与线上推理同源） */
  installed: boolean;
  local_path: string | null;
}

export interface RagModelsOut {
  models: RagModelItem[];
  all_installed: boolean;
  /** 当前是否已启用真实语义（rag_embedder === "real"） */
  enabled: boolean;
  embedder: string;
  /** HF 缓存盘余量（GB）；无法探测时为 null */
  disk_free_gb: number | null;
  /** ★ 运行环境（torch / FlagEmbedding / sentence-transformers，约 4GB）是否就绪。
   *  桌面首装**故意不装**（省 4GB）⇒ 新用户为 false，点下载会先补装依赖再下模型。 */
  deps_ready: boolean;
  /** 缺失明细（"未安装" 与 "装了但导入失败" 两类，后者是真故障不应反复重试） */
  deps_missing: string[];
  /** 能否在本机自动安装依赖（uv / venv 是否定位得到）与原因 */
  deps_target: { uv_found?: boolean; venv_found?: boolean; reason?: string | null };
  /** 面向用户的下一步提示（含「必须重建索引」） */
  hint: string;
}

export interface RagInstallStatus {
  status: "idle" | "running" | "success" | "cancelled" | "error";
  /** ★ 两阶段：deps（装重依赖约 4GB）→ models（下权重约 4.5GB）。
   *  桌面首装 deps_ready=false ⇒ 第一次点「下载」先跑 deps 段。
   *  ⚠️ deps 段**没有文件级进度**（uv 不上报），percent 在该段不可信 ⇒ 展示为不确定进度条。 */
  phase: "deps" | "models" | null;
  models: string[];
  current: string | null;
  current_file: string | null;
  file_index: number;
  file_total: number;
  /** ⚠️ **文件数进度**（非字节进度）：单个 2GB 权重在下完之前这一档不会跳，
   *  前端文案请写成「正在下载第 i/N 个文件」，不要用它驱动百分比条。 */
  percent: number;
  message: string;
  error: string | null;
  elapsed_s: number;
}

/** POST /api/v1/documents/precheck 返回：上传前重复检测结果 */
export interface PrecheckResult {
  duplicate: boolean;
  same_name: boolean;
  existing: DocumentOut | null;
  filename: string;
  sha256: string;
}

/** A-02：结构化总结 / 相关工作 */
export interface SummarySection {
  key: string;
  title: string;
  content: string;
}
export interface CompareDimensionCell {
  doc_id: number;
  filename: string;
  content: string;
}
export interface CompareDimension {
  key: string;
  title: string;
  cells: CompareDimensionCell[];
}
export interface CompareDocEntry {
  doc_id: number;
  filename: string;
  sections: Record<string, string>;
  num_pages: number;
}
export interface CoCitationPair {
  doc_a: number;
  doc_b: number;
  shared_terms: string[];
  jaccard: number;
}
export interface CoCitation {
  pairs: CoCitationPair[];
  note?: string;
}
export interface SummaryOut {
  doc_ids: number[];
  sections?: SummarySection[];
  draft?: string;
  citations?: Citation[];
  narrative: string;
  model: string;
  // S5 多篇对比扩展字段（summary/related_work 不返回）
  docs?: CompareDocEntry[];
  dimensions?: CompareDimension[];
  co_citation?: CoCitation;
}

/** ER-19：推理轨迹 */
export interface TraceEvent {
  idx: number;
  type: string;
  data: Record<string, unknown>;
}
export interface TraceListItem {
  trace_id: string;
  session_id: number;
  total_steps: number;
  total_cost: number;
  total_latency_ms: number;
  created_at: string | null;
  num_events: number;
}
export interface TraceOut extends TraceListItem {
  events: TraceEvent[];
}

/** ER-16：长期记忆条目 */
export interface MemoryItemOut {
  id: string;
  text: string;
  importance: number;
  created_at: number;
  last_accessed: number;
  /** 来源会话 id（2026-08-29 C 档：面板可溯源；旧数据为 null） */
  source_session_id?: number | null;
  /** 作用域（2026-08-29 分层 scope）：global=跨会话偏好/规则；session=仅来源会话的事实 */
  scope?: "global" | "session";
}

/** E1（2026-09-21）：回答反馈 —— 👍/👎 回流，评测样本的真实来源。 */
export interface FeedbackOut {
  id: number;
  session_id: number;
  message_id: number | null;
  trace_id: string;
  /** 只可能是 +1（好评）或 -1（差评） */
  rating: number;
  comment: string;
  created_at: string | null;
}

export interface FeedbackSummaryOut {
  total: number;
  positive: number;
  negative: number;
  /** 每条消息**当前**的评分（同一消息多次评分取最后一次），供按钮回填高亮 */
  rated_message_ids: Record<number, number>;
}

/** E2：导出的评测样本。`gold` 默认为空 —— 检索命中只是 `candidates`，不可直接当标准答案。 */
export interface EvalCaseOut {
  id: string;
  query: string;
  kind: string;
  docs: Record<string, string[]>;
  gold: string[];
  candidates: string[];
  note: string;
  meta: Record<string, unknown>;
}
