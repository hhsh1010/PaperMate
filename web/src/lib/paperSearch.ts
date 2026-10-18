/**
 * 论文检索面板的**纯逻辑**（不依赖 React / DOM）。
 *
 * 为什么单独抽出来：项目没有 testing-library ⇒ 组件只能靠 `?raw` 守结构、守不住逻辑。
 * 所以凡是"算得出来"的东西一律放这里，配 `paperSearch.test.ts` 做行为测试。
 *
 * 这一层只管三件事：
 * 1. **关键词体检**：中文关键词在 arXiv 上命中率极低 ⇒ 只提示、不拦截；
 * 2. **导入参数**：把勾选的检索结果收敛成后端要的 payload（并且**只带必要字段**）；
 * 3. **任务展示**：把后端任务状态 + 文档解析进度合成一行给用户看的文案。
 */

/** 检索范围（与后端 `SCOPES` 一一对应，别改键名） */
export const SCOPES = [
  { value: 'all', label: '全部字段' },
  { value: 'ti', label: '仅标题' },
  { value: 'au', label: '仅作者' },
  { value: 'abs', label: '仅摘要' },
] as const;

export type Scope = (typeof SCOPES)[number]['value'];

/** 常用 arXiv 分类（前端只做快捷筛选，真正的格式校验在后端） */
export const TOP_CATEGORIES = ['cs.CL', 'cs.IR', 'cs.LG', 'cs.CV', 'cs.AI', 'cs.CR'] as const;

export type PaperHit = {
  source: string;
  source_id: string;
  version: string;
  title: string;
  abstract?: string;
  authors: string[];
  published?: string;
  pdf_url: string;
  primary_category?: string;
};

export type ImportTaskView = {
  id: string;
  status: 'queued' | 'downloading' | 'ingesting' | 'ready' | 'failed' | 'cancelled' | string;
  title?: string;
  downloaded: number;
  total: number;
  doc_id?: number | null;
  duplicate?: boolean;
  error?: string;
};

/**
 * 是否含中文（只有汉字才算；全角标点不算）。
 *
 * 用于给"arXiv 以英文为主"的提示 —— **只提示不拦截**：用户想试中文就让他试，
 * 我们不做"猜他在想什么然后拦住他"这种事。
 */
export function hasChinese(text: string): boolean {
  return /[\u4e00-\u9fa5]/.test(text ?? '');
}

/** 给中文关键词的一句提示（空串表示不需要提示） */
export function chineseHint(text: string): string {
  if (!hasChinese(text)) return '';
  return 'arXiv 收录以英文为主，中文关键词命中率很低；建议改用英文（如「检索增强」→ retrieval augmented generation）。';
}

/**
 * 把勾选结果收敛成导入 payload。
 *
 * 三条约束：
 * - **只提交命中的条目**（selected 里可能有已取消勾选的脏 id）；
 * - **只带后端用得到的字段**（别把整个 hit 丢过去，避免多余字段误导后端）；
 * - `pdf_url` 必须存在（后端还会做 SSRF 白名单校验，这里先挡一道明显的空值）。
 */
export function buildImportPayload(
  hits: PaperHit[],
  selected: Iterable<string>,
): { items: { pdf_url: string; source: string; source_id: string; version: string; title: string }[] } {
  const want = new Set(selected);
  const items = hits
    .filter((h) => want.has(h.source_id) && !!h.pdf_url)
    .map((h) => ({
      pdf_url: h.pdf_url,
      source: h.source || 'arxiv',
      source_id: h.source_id,
      version: h.version || '',
      title: h.title,
    }));
  return { items };
}

/** 人类可读的字节数（保留 1 位小数；未知长度返回空串） */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '';
  const mb = n / 1048576;
  if (mb >= 1) return `${mb.toFixed(1)} MB`;
  return `${Math.round(n / 1024)} KB`;
}

/**
 * 任务 + 文档进度的**单行文案**（右侧状态列用它）。
 *
 * `docProgress` 可选：`ingesting` 阶段后端只给 doc_id，细粒度进度在文档进度接口里。
 * ★ 失败必须带**可读原因**（这是设计评审时定下的硬要求，不能只显示红点）。
 */
export function taskStatusText(task: ImportTaskView, docProgress?: { pct?: number; stage?: string } | null): string {
  switch (task.status) {
    case 'queued':
      return '排队中';
    case 'downloading': {
      const pct = task.total > 0 ? Math.min(100, Math.round((task.downloaded / task.total) * 100)) : null;
      const size = formatBytes(task.downloaded);
      const total = formatBytes(task.total);
      if (pct === null) return size ? `下载中 · ${size}` : '下载中';
      return `下载中 ${pct}%${size && total ? ` · ${size}/${total}` : ''}`;
    }
    case 'ingesting': {
      const pct = docProgress?.pct;
      const stage = docProgress?.stage;
      if (typeof pct === 'number' && pct > 0) return `解析中 ${Math.round(pct)}%`;
      if (stage) return `解析中 · ${stage}`;
      return '解析中';
    }
    case 'ready':
      return task.duplicate ? '已在库中' : '已就绪';
    case 'cancelled':
      return '已取消';
    case 'failed':
      return task.error ? `失败：${task.error}` : '失败';
    default:
      return task.status;
  }
}

/** 下载进度条百分比（0~100；未知长度返回 null ⇒ 前端显示不确定态） */
export function downloadPct(task: ImportTaskView): number | null {
  if (task.status !== 'downloading') return null;
  if (!(task.total > 0)) return null;
  return Math.max(0, Math.min(100, Math.round((task.downloaded / task.total) * 100)));
}

/**
 * 进行中的任务数（用于文献页顶部条要不要显示、以及"3 个任务进行中"的文案）。
 * `ready/failed/cancelled` 都算结束 —— 失败已经给了重试按钮，不该继续占"进行中"的口径。
 */
export function activeTaskCount(tasks: ImportTaskView[]): number {
  return tasks.filter((t) => t.status === 'queued' || t.status === 'downloading' || t.status === 'ingesting').length;
}

/** 是否还有需要用户关注的结果（失败/就绪，给"去问答/重试"用） */
export function hasActionableTask(tasks: ImportTaskView[]): boolean {
  return tasks.some((t) => t.status === 'failed' || (t.status === 'ready' && !!t.doc_id));
}

/**
 * 合并任务列表（去重 + 新数据覆盖旧数据）。
 *
 * 为什么不直接 setState([...incoming, ...prev])：同一任务会被后端更新（下载 % / 阶段变化），
 * 直接拼接会出现**两条同 id 的任务**，进度看起来随机跳。这里以 **incoming 为准**，
 * 旧的只保留 incoming 里没有的（顺序保持"新的在前"）。
 */
export function mergeTasks<T extends { id: string }>(prev: T[], incoming: T[]): T[] {
  // ★ 已存在的任务：**原地替换**（保持它原来在第几行）。
  //   轮询每 1.5s 一次，若把"刚拿到的"提到最前，列表会在用户眼皮底下不停重排 —— 这就是"跳动"的来源之一。
  const incomingById = new Map(incoming.map((t) => [t.id, t]));
  const kept = prev.map((t) => incomingById.get(t.id) ?? t);
  // 只有**新任务**才放最前（刚提交/刚恢复时"最新"的语义成立）
  const prevIds = new Set(prev.map((t) => t.id));
  const fresh = incoming.filter((t) => !prevIds.has(t.id));
  return [...fresh, ...kept];
}

/** 一条检索历史（带时间戳：用户要求"能加时间戳的地方都加上"）。 */
export type SearchHistoryEntry = {
  q: string;
  /** ISO 时间串（落 localStorage，展示交给 timeAgo） */
  at: string;
};

/** 历史条数上限（再多就沉底了，检索框下面放不下） */
export const HISTORY_MAX = 8;

/**
 * 把一次检索压入历史（**纯函数**，不碰 localStorage）。
 *
 * 规则：
 * - 去首尾空白；空串直接返回原列表（不记空查询）；
 * - **同一关键词只保留最新一次**（提到最前 + 更新时间）—— 否则反复搜同一个词会把历史刷满；
 * - 超出 `max` 截断。
 */
export function pushSearchHistory(
  list: SearchHistoryEntry[],
  q: string,
  at: string,
  max: number = HISTORY_MAX,
): SearchHistoryEntry[] {
  const text = (q || "").trim();
  if (!text) return list;
  const rest = list.filter((x) => x.q !== text);
  return [{ q: text, at }, ...rest].slice(0, Math.max(1, max));
}

/** 从 localStorage 读历史时的**防御性解析**（脏数据不许把界面搞崩）。 */
export function parseSearchHistory(raw: string | null): SearchHistoryEntry[] {
  if (!raw) return [];
  try {
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr
      .filter((x): x is { q: unknown; at: unknown } => !!x && typeof x === "object")
      .map((x) => ({ q: String(x.q ?? "").trim(), at: String(x.at ?? "") }))
      .filter((x) => x.q.length > 0)
      .slice(0, HISTORY_MAX);
  } catch {
    return [];
  }
}

/**
 * 追加检索结果（按 `source_id` 去重）。
 *
 * ★ 为什么要去重：分页用 `start` 跳过，但 arXiv 的排序在两次请求之间**可能微变**
 * （新论文上线 / 相关度同分），按 id 去重能避免同一篇在列表里出现两次。
 */
export function appendHits<T extends { source_id: string }>(prev: T[], incoming: T[]): T[] {
  const seen = new Set(prev.map((h) => h.source_id));
  return [...prev, ...incoming.filter((h) => !seen.has(h.source_id))];
}

/**
 * 单次检索**最多展示**多少条。
 *
 * 为什么要有上限：arXiv 的 `start` 深层分页很慢、位置太深还可能直接报错；
 * 而且真到了 100 条，用户基本也该收窄关键词了（继续翻只是"看得更多、用得更少"）。
 */
export const MAX_HITS = 100;

/** 是否还能「加载更多」：没有更多 / 已到上限 ⇒ false。 */
export function canLoadMore(
  shown: number,
  hasMore: boolean,
  max: number = MAX_HITS,
): boolean {
  return hasMore && shown < max;
}

/**
 * 到上限时的一句提示（没到上限返回空串，调用方据此决定要不要渲染）。
 * 不说"没了"，而是给**下一步动作**：收窄关键词 / 用分类筛选。
 */
export function capHint(shown: number, total: number, max: number = MAX_HITS): string {
  if (shown < max) return "";
  if (total > shown) {
    return `已显示前 ${max} 条（共 ${total.toLocaleString()} 条）— 结果太多，建议收窄关键词或用分类筛选`;
  }
  return "";
}

/**
 * 翻译是否**完全覆盖**了整个查询。
 *
 * 只在"一个术语 + 结果恰好等于实际查询"时为 true —— 此时再显示「实际检索：xxx」
 * 就是把同一个字符串**重复显示两遍**（用户截图里正是这样）。部分命中/多个术语时仍要显示。
 */
export function isFullyMapped(
  translations: { zh: string; en: string }[],
  queryUsed: string,
): boolean {
  if (translations.length !== 1) return false;
  return (queryUsed || "").trim() === translations[0].en.trim();
}
