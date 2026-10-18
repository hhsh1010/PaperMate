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
