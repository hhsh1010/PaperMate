"""PaperMate · Pydantic 请求/响应 Schemas（docs/04 API 一览）"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    title: str | None = None
    config: dict[str, Any] | None = None


class SessionPin(BaseModel):
    """置顶请求体（2026-10-02）。

    ★ **刻意独立于 `SessionUpdate`**：置顶是**列表排序键**，
    混进 `SessionUpdate` 会让"改标题"与"改排序"共用一条路径，
    前端为改一个布尔值就得回传整个 `config`（那份 config 很可能已过期）。
    """

    pinned: bool


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    tool_calls: list[dict] | None = None
    # 多模态：历史 user 消息携带的图片 data URL 列表（content 为纯文本，图片单独返回，前端渲染缩略图）
    images: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    # 消息级元数据（JSON 解析自 messages.meta_json）：承载无效回复的结构化 notice，
    # 供前端刷新历史后回填「打断提醒」卡片（见 routes_search._build_notice）。
    # ⚠️ B2-X1：`citations` 已从 meta 中**提升为独立字段**（见下），响应里不再出现在 meta 里。
    meta: dict[str, Any] = Field(default_factory=dict)
    # B2-X1（2026-10-01）：本轮引用（`[{"id","page","doc_id","text",...}]`）。
    # 落库在 `messages.meta_json.citations`，此处**提升为显式字段**供前端刷新/切换会话后
    # 回填引用卡片（此前前端拿不到 ⇒ 引用"当轮可见、刷新即消失"）。
    citations: list[dict] = Field(default_factory=list)
    # 消息时间戳（UTC ISO）：前端气泡下方展示 HH:MM（WorkBuddy 样式）
    created_at: str | None = None


class SessionOut(BaseModel):
    id: int
    title: str
    config: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None
    #: 侧栏置顶（2026-10-02，迁移 5）。★ 老会话与老接口都是 `False`（缺列默认 0）。
    pinned: bool = False


class SessionDetailOut(SessionOut):
    messages: list[MessageOut] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class WorkspaceCreate(BaseModel):
    """创建工作间（A1）：原子地创建（或关联）一个文献集 + 一个工作间会话。

    - collection_id 为空 → 用 doc_ids 新建文献集；
    - collection_id 非空 → 关联已有文献集（doc_ids 被忽略，以文献集为准）；
    - draft_doc_id 可选 → 指定主文献/草稿（须位于文献集内）；
    - mode：research（研读）| revision（改稿），默认 research。
    """

    name: str | None = None
    description: str = ""
    doc_ids: list[int] = Field(default_factory=list)
    collection_id: int | None = None
    draft_doc_id: int | None = None
    mode: str = "research"


class EditOp(BaseModel):
    """C2：单条 docx 编辑操作（改稿直写）。

    - `pid`：段落地址（`P{i}` 正文、`T…` 表格、`B{i}` 文档流）；
    - `op`：replace（整段替换）/ append（段尾追加）/ insert_after（段后插新段）/
      delete（整段删除，track-changes 留壳）/ purge（彻底删除=物理移除段落元素）；
    - `text`：目标文字；**delete / purge 不需要文字，可为空串**（不写入内容），
      其余三种 op 必须非空，否则被 `apply_edits` 拒绝；
    - `quote`：可选校验阀——不在目标段落原文中则该 op 被拒绝（宁拒不错）；
      purge 另有安全阀：仅允许对「有效文本已清空」的**正文**段落执行（表格内段落拒绝）。
    """

    pid: str
    op: str
    text: str
    quote: str | None = None


class ApplyEdits(BaseModel):
    """C2：批量应用 docx 编辑操作（track-changes 写入）。"""

    edits: list[EditOp] = Field(default_factory=list)


class WorkspaceAddDocs(BaseModel):
    """B1：往工作间文献集批量加入文档（幂等，已在集内的自动跳过）。"""

    doc_ids: list[int] = Field(default_factory=list)


class WorkspaceUpdate(BaseModel):
    """B2：更新工作间配置（PATCH 语义 —— 只更新显式传入的字段）。

    - `draft_doc_id`：主文献/草稿；显式传 `null` 表示清除，不传表示不动；
    - `mode`：研读/改稿模式（B2 预留，C1 模式切换复用同端点）。
    路由层以 `model_fields_set` 区分「未传」与「显式 null」。
    """

    mode: str | None = None
    draft_doc_id: int | None = None


class WorkspaceOut(BaseModel):
    """工作间只读投影（从会话 config_json + 底层文献集解析）。"""

    session_id: int
    collection_id: int | None = None
    name: str
    mode: str
    draft_doc_id: int | None = None
    doc_ids: list[int] = Field(default_factory=list)


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    meta: dict[str, Any]
    num_chunks: int = 0
    # F9：创建时间（研究首页「最近文献」排序用；历史构造点未传时为 None，向后兼容）
    created_at: str | None = None


class ChatRequest(BaseModel):
    content: str
    tools: list[str] | None = None
    # 多模态（DeepSeek-V4-Flash-Vision）：base64 data URL 列表，如
    # "data:image/jpeg;base64,...."。仅该模型接受图片；图片只进入本轮 user 消息，
    # 不写入历史（与官方「图片仅限 user 消息」规范一致）。
    images: list[str] | None = None
    # 重新生成：为 True 时复用当前会话最后一条用户消息（content/images 会被服务端
    # 从 DB 最后一条 user 消息覆盖），并裁掉上一轮 assistant 回复后重跑，避免重复落库。
    regenerate: bool = False
    # 请求级流标识（2026-09-07 F8）：前端每次发送生成唯一 id，使「停止」能精确命中
    # 这一条流，而非把取消信号广播到整个会话（同会话并发流会互相覆盖 cancel 事件）。
    # 留空则退化为会话级（兼容旧客户端 / 单流场景）。
    stream_id: str = ""
    # 结构化 @ 提及（2026-09-07）：前端在 @ 文档时记录精确 doc id 随消息发送。
    # 优先于 args 里的 "@文档N" 文本解析——后者依赖前后端展示顺序一致，可能错位。
    mention_doc_ids: list[int] = Field(default_factory=list)


class PromptEnhanceRequest(BaseModel):
    """提示词增强请求：把口语化输入改写为更明确的任务描述（单轮元任务）。

    session_id 可选：传入时附加最近对话历史作为上下文（「继续/再详细点」类输入需要）；
    历史只作参考，增强仍以 text 为主目标。
    """

    text: str
    session_id: int | None = None


class ParaTransformRequest(BaseModel):
    """段落 AI 改写请求（改稿工作台：扩写/缩写/润色/去重，单轮元任务）。

    - `action`：expand / shorten / polish / dedupe；
    - `context_before/after`：相邻段落参考（仅供模型理解上下文，不被改写、不输出）；
    - `session_id` 可选：用于 resolve_llm 取会话级 → 全局默认模型。
    """

    text: str
    action: str
    context_before: str | None = None
    context_after: str | None = None
    session_id: int | None = None


class RetrievedChunkOut(BaseModel):
    id: str
    page: int | None  # None=无物理页码（docx/md/txt）；PDF 为 1 起算
    text: str
    score: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_ids: list[int] | None = None


class SummaryRequest(BaseModel):
    doc_ids: list[int]
    lang: str = "zh"
    # ★ 2026-10-22：从会话内触发时带上会话 id ⇒ 模型跟随会话所选（此前只能读全局默认）
    session_id: int | None = None


class UsageRow(BaseModel):
    session_id: int
    title: str
    num_messages: int
    num_tool_runs: int
    prompt_tokens: int
    completion_tokens: int
    cost: float
    # F10：模型与最近活动时间。用量页需按模型/时间维度查看，
    # 并与 chat 轨迹同源可追溯（model 取会话生效模型，与 build_agent 的优先级一致）。
    model: str = ""
    last_at: str | None = None


class UsageTrendPoint(BaseModel):
    """F10：单日用量趋势点（与 UsageRow 同源——均来自 messages 表）。"""

    date: str
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class UsageTrendOut(BaseModel):
    """F10：按天聚合的用量趋势（成本趋势图数据源）。"""

    days: int
    points: list[UsageTrendPoint]
    total_cost: float


class FeedbackCreate(BaseModel):
    """E1：提交反馈的入参。

    `rating` 只接受 **+1 / -1** —— 不是多余的校验：前端若传 0 或 2，
    聚合出的均值/好评率会失去意义，且这类脏数据**事后极难清洗**（看不出是误传还是真评价）。
    """

    session_id: int
    message_id: int | None = None
    trace_id: str = ""
    rating: int
    comment: str = ""


class FeedbackOut(BaseModel):
    id: int
    session_id: int
    message_id: int | None = None
    trace_id: str = ""
    rating: int
    comment: str = ""
    created_at: str | None = None


class FeedbackSummaryOut(BaseModel):
    """E1：反馈汇总。

    `rated_message_ids` 让前端能回填「这条我评过、评的是什么」——
    否则刷新后按钮全灭，用户会以为反馈没记上（与 `meta_json` 回填提醒同一类问题）。
    """

    total: int = 0
    positive: int = 0
    negative: int = 0
    rated_message_ids: dict[int, int] = {}


class MetricAlert(BaseModel):
    """A6：一条告警。

    刻意带上 `threshold` 与 `sample` —— 只有 value 的告警没法判断
    「是系统真出问题，还是样本太小导致的抖动」。
    """

    code: str
    level: str = "warn"
    value: float = 0.0
    threshold: float = 0.0
    sample: int = 0
    message: str = ""


class MetricsOut(BaseModel):
    """D1：运行指标（零依赖 JSON）。

    `thresholds` / `min_samples` **回显**：前端要能解释"为什么报警"，
    否则告警只是一个红灯，用户无法判断严重性。
    """

    window_days: int = 7
    since: str | None = None
    session_id: int | None = None
    num_traces: int = 0
    num_traces_total: int = 0
    num_answers: int = 0
    num_sessions: int = 0
    avg_steps: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    #: T-09：延迟分位（ms）。★ 平均回推不出分位，也看不出长尾。
    #: ⚠️⚠️ 必须同时看 `latency_samples` / `latency_missing` —— 实测 80.1% 的轨迹
    #: `total_latency_ms == 0`（未采集）⇒ 分位只对**非零样本**算；缺口大时不可信。
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    #: 参与分位统计的**非零**样本数（= 真正采集到延迟的轨迹数）
    latency_samples: int = 0
    #: 因 `total_latency_ms == 0` 被排除的轨迹数（**缺口**，非"很快"）
    latency_missing: int = 0
    #: ★ O-12（2026-10-02）：**服务端首字延迟**（TTFT）分布，ms。
    #: 口径：请求进入路由处理器 → 首个正文增量（**不含网络往返**）；
    #: 与单次 LLM 调用的 `ttft`（建连+排队+prefill）**不是一回事**，报告里必须写清。
    #: ⚠️ 埋点 2026-10-02 才上线 ⇒ 老轨迹没有该值，一律计入 `ttft_missing`（**不用 0 冒充**）。
    ttft_avg_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    #: 参与分位统计的样本数（真正采到首字延迟的轨迹数）
    ttft_samples: int = 0
    #: 未采集的轨迹数（老轨迹 / 未产出即结束）—— 缺口，非"首字很快"
    ttft_missing: int = 0

    # ---- 上下文水位（2026-10-02）：让"额度够不够"在事发前可见 ----
    #: 当前**生效**的 `max_tokens_budget`（单次请求熔断线）；取自 `runtime_settings`，
    #: 与 `chat_service.build_agent` 同一入口 —— **不是**代码默认值（用户可在设置页改）。
    ctx_budget: int = 30000
    #: 派生触发线 = `ctx_budget × 0.7`（压缩 + 闸门在这条线动作）
    ctx_trigger: int = 21000
    #: 本轮**最大单次** prompt 的中位数（`None` = 尚未采到样本，**不是 0**）。
    #: ⚠️ 与 `messages.prompt_tokens`（整轮**累计**）**不是一回事**，库里某条高达 82,057。
    ctx_peak_p50: int | None = None
    #: 同上的峰值
    ctx_peak_max: int | None = None
    #: ★ 余量 = `ctx_budget − ctx_peak_p50`：**中位情况下还剩多少空间可塞**（负数 = 常在越线）
    ctx_headroom_median: int | None = None
    #: 有 `peak` 字段的轨迹数 / 缺失数（老轨迹未埋 ⇒ 缺口必须可辨识）
    ctx_peak_samples: int = 0
    ctx_peak_missing: int = 0
    #: 越过 70% 触发线 / 直接越熔断线的**轨迹数**（回答"这常不常发生"）
    ctx_over_trigger: int = 0
    ctx_over_budget: int = 0
    #: ★ 「被动过刀」的轨迹数（闸门裁剪或历史压缩发生过）——
    #: 证据被裁过的那部分损失，**judge 测不出来**（它只测忠实度/拒答，不测完整性）。
    ctx_elided_traces: int = 0
    avg_cost: float = 0.0
    total_cost: float = 0.0
    #: ⚠️ 语义是「未收尾率」而非「错误率」：包含用户主动点「停止」的轨迹。
    #: 实测 43% 里绝大多数是主动取消，叫 error_rate 会把正常行为误报成故障。
    unfinished_rate: float = 0.0
    #: ★ T-09：**判断"该不该修"用这个** —— 已排除用户主动停止 / 断连，
    #: 但**包含 `unknown`**（打点缺口不能当成功：那是假通过）。
    #: 实测 54.9%（含中断）→ 20.7%（排除中断后）。
    unfinished_rate_excluding_interrupt: float = 0.0
    #: ★ 上面那个数里**确认**有失败原因的部分（`unknown` 不算确认失败）。
    #: 实测 2.4% —— 与 20.7% 的差就是"打点缺口有多大"。
    failed_confirmed_rate: float = 0.0
    #: T-09：未收尾**四分类**（`done` / `interrupted` / `unknown` / `failed`）。
    #: ⚠️ `interrupted` 的准确语义是「**中断（原因未确认）**」—— 它混了"用户主动停止"
    #: 与"进程崩溃"，因为崩溃时来不及写任何事件。
    unfinished_breakdown: dict[str, int] = {}
    #: ★ 有**明确**用户停止标记的条数（`cancelled` 事件 / kind=cancelled）。
    #: 与 `unfinished_breakdown["interrupted"]` 的差 = "中断但无法确认原因"的量。
    #: 老轨迹为 0 ⇒ 该数从 0 变正，就是取消打点真正生效的证据。
    confirmed_user_stop: int = 0
    tool_error_rate: float = 0.0
    #: A4（2026-09-21）：失败归因分布（分类 → 次数）。`unknown` 非零说明存在打点缺口。
    failure_kinds: dict[str, int] = {}
    #: 排除「用户取消」后的真实失败分布 —— cancelled 不是故障，不能混进故障口径
    failure_kinds_excluding_cancel: dict[str, int] = {}
    tool_calls: int = 0
    feedback: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    min_samples: dict[str, int] = {}
    alerts: list[MetricAlert] = []
    healthy: bool = True


class EvalCaseOut(BaseModel):
    """E2：导出的评测样本。

    ⚠️ **`kind` 默认 `candidate`、`gold` 默认空** —— 这是有意的：
    检索命中的块放在 `candidates` 里**只作人工标注参考**；
    若直接把命中当 gold，就是用系统自己的输出证明自己有效（**自证循环**）。
    人工补完 `gold` 后把 `kind` 改成 `verbatim`，才能进评测集。

    ⚠️ 字段必须**列全**：`response_model` 会裁掉未声明的字段，
    漏一个就等于导出时静默丢信息（本项目在"静默丢失"上已多次吃亏）。
    """

    id: str
    query: str
    kind: str = "candidate"
    docs: dict[str, list[str]] = {}
    gold: list[str] = []
    candidates: list[str] = []
    #: 固定文案：提醒 gold 需人工确认（防自证循环）
    note: str = ""
    #: 溯源信息：session_id / trace_id / total_steps / total_cost / total_latency_ms / created_at
    meta: dict = {}


# ---------------------------------------------- C1（2026-09-21）：补齐响应契约
# 说明：这批模型是**照着真实响应**写的（不是照代码猜的），字段类型均经实测核对。
# 例如 `memories.importance` 实测是 **float**（0.9）而非 int；
# `created_at` / `last_accessed` 是 **float 时间戳**而非 iso 字符串 —— 猜错会直接在响应校验时 500。
#
# ⚠️ 设计口径（与 C1 的目的相关）：
# - **必返回字段不给默认值** —— 若实际响应缺该字段，FastAPI 报错（显性失败），
#   这正是 C1 要的「让前后端字段不一致显性化」；给默认值会把不一致**掩盖成假数据**。
# - **确实可空的字段**（如 `source_session_id`）用 `| None = None`，这是真实语义，不是掩盖。


class OkOut(BaseModel):
    """通用确认响应：只有 `ok`。"""

    ok: bool = True


class OkRemovedOut(BaseModel):
    """删除类确认响应：`ok` + 受影响条数。"""

    ok: bool = True
    removed: int = 0


class CollectionOut(BaseModel):
    """文献集（`CollectionService._out` 的契约）。"""

    id: int
    name: str
    description: str
    created_at: str | None = None
    doc_ids: list[int]


class CollectionListOut(BaseModel):
    items: list[CollectionOut]


class MemoryItemOut(BaseModel):
    """长期记忆条目（**注意类型**：`importance` 是 float、时间是 float 时间戳）。"""

    id: str
    text: str
    importance: float
    created_at: float | None = None
    last_accessed: float | None = None
    source_session_id: int | None = None
    scope: str


class MemoryListOut(BaseModel):
    total: int
    items: list[MemoryItemOut]


class MemoryBatchDeleteOut(OkRemovedOut):
    """批量删除：额外回报**未命中的 id**（便于前端提示"这几条已经不存在了"）。"""

    missing: list[str]


class MemoryUpdateOut(BaseModel):
    ok: bool = True
    id: str
    text: str
    importance: float
    scope: str


# ---- C1 第二批：documents 相关（类型均经真实响应核对，见实施记录） ----


class PrecheckOut(BaseModel):
    """上传前重复检测（`POST /documents/precheck`）。"""

    duplicate: bool
    same_name: bool
    existing: DocumentOut | None = None
    filename: str
    sha256: str


class UploadProgressOut(BaseModel):
    """异步上传进度快照（前端轮询可视化用）。

    ⚠️ `pct` 实测是 **int**（0-100 的整数百分比），`updated_at` 是 **float 时间戳**。
    """

    doc_id: int
    status: str
    stage: str
    done: int
    total: int
    pct: int
    error: str | None = None
    updated_at: float


class CancelUploadOut(BaseModel):
    """取消异步上传（协作式）：`cancelled` 表示本次调用**是否触发了**取消。

    已处于终态时返回 `cancelled=False`（幂等无副作用）—— 这个区分对前端有用：
    「我按了取消」与「它自己已经结束了」是两种不同的结果。
    """

    ok: bool = True
    cancelled: bool
    status: str | None = None


class DocTextOut(BaseModel):
    """文档解析后的纯文本（供非 PDF 格式在线预览）。"""

    doc_id: int
    text: str


class DocImageOut(BaseModel):
    """提取的图片资源。

    ⚠️ `page` 可为 `None`：**docx 没有页码概念**，其 chunk.page 就是 None。
    实测由 C1 第二批抓出（严格契约 + docx 测试用例一起才暴露出这一点）。
    """

    filename: str
    page: int | None = None
    kind: str
    caption: str


class DocTableOut(BaseModel):
    """抽取的表格（`header` 实测是 `list[str]`，`bbox` 是 4 元素 float 列表）。

    ⚠️ `page` 可空，原因同 `DocImageOut`。
    """

    page: int | None = None
    markdown: str
    rows: int
    cols: int
    bbox: list[float]
    header: list[str]


class DocDiagramOut(BaseModel):
    """抽取的图（`mermaid` 可能为空串；`bbox` 是 4 元素 float 列表）。

    ⚠️ `page` 可空，原因同 `DocImageOut`。
    """

    page: int | None = None
    bbox: list[float]
    labels: list[str]
    file: str
    mermaid: str
    description: str
    text: str
    note: str = ""
    meta: dict[str, object] = {}


# ---- C1 第三批：traces / settings（类型均经真实响应核对） ----


class TraceListItemOut(BaseModel):
    """轨迹列表项（列表接口刻意**不返回** `events` 事件体）。"""

    trace_id: str
    session_id: int
    total_steps: int
    total_cost: float
    total_latency_ms: int
    created_at: str
    num_events: int


class TraceOut(BaseModel):
    """单条轨迹（含 `events`）。

    `events` 用 `list[dict]` 而非精确模型：事件体随类型变化（tool_call / tool_result /
    thinking / done / error），A4 之后又新增了 `failure_kind` ——
    收紧成固定模型会让**每加一个事件字段都要改契约**，得不偿失。
    """

    trace_id: str
    session_id: int
    total_steps: int
    total_cost: float
    total_latency_ms: int
    created_at: str
    events: list[dict]


class InventoryOut(BaseModel):
    """可用工具 / 技能 / 权限清单。

    `tools` / `skills` 留作 `list[dict]`：两者字段由各自注册机制决定
    （工具含 permission、技能含 commands / feature / enabled），逐项建模需同时跟随两套注册表演进。
    """

    tools: list[dict]
    skills: list[dict]
    permissions: list[str]
    sandbox_enabled: bool


class SettingsOut(BaseModel):
    """应用设置。

    顶层字段精确；`features`（开关集合）/ `providers`（厂商 map）保持 dict ——
    前者随新功能增减，写成固定模型会让**加一个开关就要改契约**。
    """

    default_model: str
    dedupe_warning: bool
    features: dict[str, bool]
    figure_enrich: dict
    figure_enrich_deps: dict
    cost_limit_per_turn: float
    max_tokens_budget: int
    # ★ 2026-10-22：auto 标志必须一并回显 —— 否则前端读不到它，"自动/显式"状态无从判断
    max_tokens_budget_auto: bool = True
    max_tokens_per_turn: int
    agent_rules: str
    memory_embedder: str
    rag_embedder: str
    use_bge_sparse: bool
    use_jieba: bool
    # T-03 重排推理加速档位（2026-09-24）：**必须显式声明** ——
    # `response_model` 会把未声明的字段**静默裁掉**（项目已知坑：C1 批次的
    # ParaTransformOut / EvalCaseOut 就因此漏过字段），表现为"后端明明返回了、前端永远读不到"。
    rag_reranker_fp16: bool
    rag_reranker_max_length: int
    default_discipline: str
    data_dir: str
    providers: dict[str, dict]


class SettingsTestOut(BaseModel):
    """模型连通性测试结果。

    ⚠️ 失败分支只返回 `{ok, provider, error}`（**没有** `model` / `reply`）→ 故给默认空串。
    这是**真实语义**（不同分支字段集本就不同），不是用默认值掩盖问题。
    """

    ok: bool
    provider: str = ""
    model: str = ""
    reply: str = ""
    error: str = ""


class WebSearchTestOut(BaseModel):
    """原生联网搜索连通性测试结果（比 `SettingsTestOut` 多一个 `supported`）。"""

    ok: bool
    supported: bool
    provider: str = ""
    model: str = ""
    reply: str = ""
    error: str = ""


# ---- C1 第四批：sessions / eval / usage（类型均经真实响应核对） ----


class NetworkCapabilityOut(BaseModel):
    """当前会话的联网与模型可用性（供前端「可用性预警」）。

    ⚠️ **判据口径**（前端用得到，写在这里防止误读）：
    - `supports_native` = **纯能力**（该模型是否支持原生联网）；
    - `available = enabled and supports_native` —— 开关关着时也是 `false`，
      所以判断「能力不具备」必须看 `supports_native`，看 `available` 会把
      「没开开关」误判成「模型不支持」。
    """

    session_id: int
    enabled: bool
    available: bool
    supports_native: bool
    effective_model: str
    reason: str
    #: 前端类型里标记为可选 → 实测确实可能不返回，故给默认值
    model_risk: bool = False
    model_risk_reason: str = ""


class EvalSummaryOut(BaseModel):
    """端到端忠实度看板数据源。

    四个子块结构差异大且会随评测维度演进 → 子块保持宽松（`dict` / `list[dict]`），
    只锁顶层字段名：**顶层字段被裁掉**才是前端最直接的破坏。
    """

    retrieval: dict
    gate: dict
    agent: dict
    recent_traces: list[dict]


class SessionUsageDetailOut(BaseModel):
    """会话级用量明细（消息 + 工具调用 + Trace 轮次）。"""

    session_id: int
    title: str
    usage: dict
    messages: list[dict]
    tool_runs: list[dict]
    traces: list[dict]


# ---- C1 第五批：summary / prompt / transform（类型均经真实响应核对） ----


class PromptEnhanceOut(BaseModel):
    """提示词增强结果。

    ⚠️ `applied=False` **不代表出错** —— 它表示「语义上没有变化」或「走了降级路径」；
    具体原因看 `reason`（成功且确有变化时 `reason` 为 `None`）。
    前端不应把 `applied=False` 一律渲染成错误。
    """

    original: str
    enhanced: str
    applied: bool
    reason: str | None = None


class ParaTransformOut(BaseModel):
    """段落改写结果（缩写 / 扩写 / 润色 / 去重）。

    口径同 `PromptEnhanceOut`：`applied=False` 是**状态**而非错误，原因见 `reason`。

    ⚠️ **`action` / `action_label` 只有成功分支才返回** —— 降级（无 key）与失败
    （调用异常）分支只返回 4 个字段（`original`/`transformed`/`applied`/`reason`）。
    故必须给默认空串，否则那两条路径会直接 500。

    > 这两个字段是**测试抓出来的**：最初契约只写了 4 个字段，而核对时恰好只走到降级路径
    > （当时 API 余额不足）→ 没发现成功路径多两个字段，
    > 直到 `test_para_transform.py` 报 `KeyError: 'action_label'` 才暴露。
    > **教训：契约字段应取「所有 return 分支的字段并集」，不是「某次实测到的字段」。**
    """

    original: str
    transformed: str
    applied: bool
    action: str = ""
    action_label: str = ""
    reason: str | None = None


class SummarySectionOut(BaseModel):
    """一键总结的五段式小节。"""

    key: str
    title: str
    content: str


class SummaryCitationOut(BaseModel):
    """总结/综述的引用条目。

    `page` 可空：文档类型不同（docx 无页码）或页码无法归一化时即 `None` ——
    与 `DocImageOut.page` 同源问题，故同样用可空类型而非必填。
    """

    doc_id: int
    filename: str
    chunk_id: str
    page: int | None = None
    text: str


class SummaryOut(BaseModel):
    """一键总结（五段式 + 小结）。"""

    doc_ids: list[int]
    sections: list[SummarySectionOut]
    citations: list[SummaryCitationOut]
    narrative: str
    #: `llm` = 叙述由模型生成；`deterministic` = 无真实模型/调用失败时的**确定性降级**
    model: str


class RelatedWorkOut(BaseModel):
    """相关工作综述。

    `co_citation`（跨文档共引分析）保持 dict：结构随共引算法演进，逐字段建模维护成本高。
    """

    doc_ids: list[int]
    draft: str
    citations: list[SummaryCitationOut]
    co_citation: dict
    model: str


# ---- C1 第六批（收口）：documents 剩余 + compare ----


class DocParagraphOut(BaseModel):
    """段落可寻址视图的一项（`P{i}` 正文 / `T..R..C..P{pi}` 表格单元格）。

    ⚠️ `style` 与 `reason` 可为 `None`：
    - `style=None` = 该段落没有命名样式（docx 允许）；
    - `reason=None` = 该段落**可编辑**；非 None（`field`）表示含域代码，前端据此禁用编辑。
    ★ 2026-10-10：媒体段（含图片/公式对象）**不再因 media 不可编辑**——走"媒体锚定"：
      仅改文字部分，媒体 run 原位保留；`media=True` 供前端提示"仅改文字，图和公式不动"。
    """

    pid: str
    style: str | None = None
    text: str
    empty: bool
    editable: bool
    reason: str | None = None
    media: bool = False
    deleted: bool


class EditApplyOut(BaseModel):
    """改稿应用结果。

    ⚠️ **`applied` 是「已应用改动的明细列表」，不是计数** ——
    （形如 `[{"pid": "P0", "op": "replace"}]`）。别按名字猜成 int：
    我第一版就是这么猜错的，被 `tests/test_docx_edit.py` 的 5 个用例当场抓出。

    ⚠️ 字段名与 `ParaTransformOut.applied`（**bool**）完全不同语义，读代码时注意区分。

    ⚠️ `reindex_triggered` 是**端点层**在 service 返回值上追加的
    （`routes_documents.apply_doc_edits` 里 `out["reindex_triggered"] = ...`）——
    只看 service 层会漏掉它。**核对契约要看端点最终返回，不是 service 原始返回。**
    """

    #: 已应用的改动明细（每项含 pid / op）
    applied: list[dict]
    #: 本次提交的操作总数
    total: int
    #: 被拒绝的改动（每项含 reason）
    rejected: list[dict]
    #: 应用前自动备份的文件名（用于 restore-backup）
    backup: str | None = None
    paragraph_count_after: int
    #: 是否已触发后台重建索引（内容变更后使 RAG 与新文本一致）
    reindex_triggered: bool = False


class RestoreBackupOut(BaseModel):
    """从备份恢复的结果（`restored_from` = 备份文件名）。"""

    restored_from: str
    #: 端点层追加：是否已触发后台重建索引（内容变更后让 RAG 与新文本一致）
    reindex_triggered: bool = False


class ReindexOut(BaseModel):
    """重建索引结果（`failed` 内为 `{id, error}`）。"""

    reindexed: int
    failed: list[dict]
    total: int


class CompareOut(BaseModel):
    """多篇文档对比（逐维度并列 + 共性/差异 + 共引）。

    `docs` / `dimensions` 保持 `list[dict]`：逐维度结构随对比算法演进，
    且不同维度的字段集不同（多态），建模成 Union 的维护成本高于收益。
    """

    doc_ids: list[int]
    docs: list[dict]
    dimensions: list[dict]
    co_citation: dict
    narrative: str
    model: str


# ---- 场景可见性（2026-09-23）：当前会话的检索范围 ----


class DocScopeItemOut(BaseModel):
    """检索范围内的文档条目（只带展示所需的最小字段）。"""

    id: int
    filename: str


class DocScopeOut(BaseModel):
    """当前会话的**检索范围**（供前端常驻提示「我现在处于哪个场景」）。

    **为什么需要**：三场景（无文档 / 单文档 / 多文档）下系统的检索行为不同，
    但界面上**完全看不出**自己处于哪个场景 —— 实测中用户报「文档约束没生效」时，
    相当一部分困惑来自"以为自己在这个场景、实际在另一个"（最典型的是会话**根本没绑文档**，
    于是走了通用对话）。

    ⚠️ 本端点只报**会话级**范围（稳定、可缓存）：**本轮是否因 `@` 而聚焦**由前端自己知道
    （它知道本轮 @ 了哪些文档），故不掺入每轮变化的 mention —— 否则这份数据每轮都失效。
    """

    session_id: int
    #: 关联文档数（0 = 通用对话；1 = 单文档；≥2 = 多文档）
    doc_count: int
    #: id **降序**（与前端「文档N」标签同口径）
    doc_ids: list[int]
    documents: list[DocScopeItemOut]
    #: 是否强制检索（= `should_force_tool_call`，即"有文档就必须先检索"）
    force_retrieval: bool
    #: `general`（0 篇，通用对话）/ `single`（1 篇）/ `multi`（≥2 篇，默认检索全部）
    mode: str


# ─────────────────────────────────────────────────────────────────────────────
# 真实语义检索模型的显式安装（2026-10-05 桌面客户端）
# 契约：GET  /api/v1/settings/rag/models              清单 + 安装状态
#       POST /api/v1/settings/rag/models/install      触发下载（后台线程）
#       POST /api/v1/settings/rag/models/install/cancel 取消
#       GET  /api/v1/settings/rag/models/install/status  轮询进度
# ─────────────────────────────────────────────────────────────────────────────
class RagModelItem(BaseModel):
    """单个本地模型的清单项。"""

    key: str                      # embedder | reranker
    repo_id: str
    label: str
    approx_gb: float              # 预计体积（提示用，非精确值）
    #: 本机是否已有**可用**快照（判定口径与线上推理同源：`hf_local.resolve_local_hf_snapshot`）
    installed: bool
    local_path: str | None = None


class RagModelsOut(BaseModel):
    """设置页「语义检索」卡片的首屏数据。"""

    models: list[RagModelItem]
    all_installed: bool           # 两个权重都就绪 ⇒ 可以考虑启用 real
    enabled: bool                 # 当前 runtime settings 的 rag_embedder == real
    embedder: str                 # 当前生效值（fake | real）
    disk_free_gb: float | None    # HF 缓存盘余量（无法探测时为 None）
    # ---- 运行环境（重依赖）状态（2026-06-06）----
    #: torch / FlagEmbedding / sentence-transformers 是否可导入。
    #: ★ 桌面首启**故意不装**（lib.rs 用 --no-install-package 排除，约省 4GB），
    #:   所以新用户这里为 false —— 点「下载」会先补装依赖再下模型。
    deps_ready: bool
    #: 缺失明细（含"未安装"与"装了但导入失败"两种，后者是真故障不能反复重试）
    deps_missing: list[str] = Field(default_factory=list)
    #: 能否在本机自动安装依赖（uv/venv 是否定位得到）与原因
    deps_target: dict[str, Any] = Field(default_factory=dict)
    hint: str                     # 面向用户的下一步提示（含"必须重建索引"）


class RagInstallReq(BaseModel):
    """触发安装。`models` 为空 ⇒ 装全部。"""

    models: list[str] | None = None


class RagInstallStatusOut(BaseModel):
    """安装任务状态（轮询用）。

    ★ P-01（2026-10-09）：**进度口径已从"文件数"升级为"字节"** ——
      `done_bytes` / `total_bytes` 由后端按真信号测量：
      deps 段量 `uv-cache` **磁盘增量**（venv 走硬链接，共享同一份数据），
      models 段量**模型目录体积**（含 `.incomplete` 半成品与断点续传已存在的部分）。
      ⇒ 前端在 `total_bytes > 0` 时**可以**显示百分比条（两段都行）。
    ⚠️ `total_bytes == 0` 表示**信号不可用**（热缓存增量≈0 / 元数据缺失）⇒ 前端保持
      "不确定动画"，**不要**把 0 当成 0% 进度（那是"假装有进度"，本项目明确禁止）。
    ⚠️ `percent` 仍保留：它是**阶段内的合成百分比**（deps 段按字节增量估、models 段优先
      按字节、无字节分母时才退回文件数）；`file_index/file_total` 退化为辅助显示。
    """

    status: str          # idle | running | success | cancelled | error
    #: ★ 两阶段：``deps``（装重依赖约 4GB）→ ``models``（下权重约 4.5GB）。
    #:   桌面首装 deps_ready=false ⇒ 第一次点「下载」会先跑 deps 段。
    #:   前端据此显示不同文案，**不要**把 deps 段的进度当成模型下载进度。
    phase: str | None = None
    models: list[str]
    current: str | None = None
    current_file: str | None = None
    file_index: int = 0
    file_total: int = 0
    percent: float = 0.0
    #: ★ P-01：真字节进度（0 = 信号不可用 ⇒ 前端用不确定动画）
    done_bytes: int = 0
    total_bytes: int = 0
    message: str = ""
    error: str | None = None
    elapsed_s: float = 0.0


# ───────────────────── 存储位置（2026-10-07）─────────────────────
# ★ 响应模型里**必须逐个声明**字段：未声明的会被 `response_model` **静默裁掉**
#   （本项目已知坑：ParaTransformOut / EvalCaseOut 因此漏过字段，
#    表现为"后端明明返回了、前端永远读不到"）。


# ───────────────────── 「另存为」导出（P-04/P-05，2026-10-09）─────────────────────
# ★ 背景：桌面端（Tauri/WebView2）里 `<a download>` 是**静默落盘到系统下载文件夹**、
#   界面零反馈（实测：用户以为无效、连点 12 次）⇒ 改为
#   「前端弹原生另存为对话框 → 交给后端写盘 → 前端提示最终路径」。
#   路径校验统一走 `api/dest_path.require_abs_dest`（只写不读、拒绝相对路径）。
class ExportToIn(BaseModel):
    """文档导出件的落盘请求（`dest` 由用户的原生「另存为」对话框选定）。"""

    dest: str
    mode: str = "tracked"  # tracked（带修订痕迹）| clean（接受全部修订）


class ExportToOut(BaseModel):
    path: str
    bytes: int


class WriteFileIn(BaseModel):
    """**前端生成**字节的落盘请求（CSV / JSONL / Markdown 等）。"""

    dest: str
    content_b64: str


class WriteFileOut(BaseModel):
    path: str
    bytes: int


class FigureInstallStatusOut(BaseModel):
    """图片增强依赖安装状态（设置页轮询用，2026-10-09 新增）。

    ★ 与 RAG 安装（deps → models 两阶段）不同：这里**只有"装包"一段**
    （`uv sync --extra figure --frozen --inexact`）；PP-OCR 的模型权重由
    paddleocr 在**首次使用时**自行下载，不在本流程内。
    """

    status: str  # idle | running | success | cancelled | error
    message: str = ""
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float = 0.0
    #: 安装结束后的三能力快照 `{ocr, table, vlm}`（运行中为 None）
    deps: dict | None = None


class StorageDrive(BaseModel):
    """本机磁盘（`ctypes` 调 GetLogicalDrives/GetDriveTypeW，不加依赖）。"""

    root: str
    free_gb: float
    total_gb: float
    kind: str  # FIXED / REMOVABLE / REMOTE / CDROM / RAMDISK / UNKNOWN


class StorageSize(BaseModel):
    mb: float
    truncated: bool  # True = 因文件数超限只统计了部分（数值是下界）


class StorageOut(BaseModel):
    """`GET /api/v1/settings/storage`"""

    available: bool
    unavailable_reason: str | None = None
    config_path: str | None = None
    data_root: str
    pending_root: str | None = None
    migrate: bool = False
    dirs: dict[str, str] = Field(default_factory=dict)
    sizes_mb: dict[str, StorageSize] = Field(default_factory=dict)
    drives: list[StorageDrive] = Field(default_factory=list)
    min_free_gb: float = 8.0
    restart_required: bool = False


class StorageUpdateIn(BaseModel):
    """`PUT /api/v1/settings/storage`"""

    data_root: str
    migrate: bool = False


class StorageUpdateOut(BaseModel):
    ok: bool
    restart_required: bool = True
    warnings: list[str] = Field(default_factory=list)
