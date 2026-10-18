/**
 * 论文检索面板的纯逻辑判据（vitest）。
 *
 * 为什么这些测试重要：项目没有 testing-library ⇒ 组件只能靠 `?raw` 守结构、守不住行为。
 * 所以"中文提示 / 导入参数 / 进度文案"这些**会出错且会误导用户**的地方，全部压到纯函数上测。
 */

import { describe, expect, it } from 'vitest';

import {
  activeTaskCount,
  appendHits,
  buildImportPayload,
  chineseHint,
  downloadPct,
  formatBytes,
  hasActionableTask,
  hasChinese,
  mergeTasks,
  parseSearchHistory,
  pushSearchHistory,
  taskStatusText,
  type ImportTaskView,
  type PaperHit,
} from './paperSearch';

const hit = (over: Partial<PaperHit> = {}): PaperHit => ({
  source: 'arxiv',
  source_id: '2401.00001',
  version: 'v1',
  title: 'T',
  authors: ['A'],
  pdf_url: 'https://arxiv.org/pdf/2401.00001v1',
  ...over,
});

const task = (over: Partial<ImportTaskView> = {}): ImportTaskView => ({
  id: 't1',
  status: 'downloading',
  downloaded: 0,
  total: 0,
  ...over,
});

describe('中文关键词提示', () => {
  it('识别中文（只认汉字，全角标点不算）', () => {
    expect(hasChinese('检索增强')).toBe(true);
    expect(hasChinese('retrieval augmented generation')).toBe(false);
    expect(hasChinese('RAG（检索增强）')).toBe(true);
    expect(hasChinese('，。！？')).toBe(false);
  });

  it('英文关键词不提示，中文关键词给可执行的建议', () => {
    expect(chineseHint('rag')).toBe('');
    const h = chineseHint('检索增强');
    expect(h).toContain('英文');
    expect(h).toContain('retrieval augmented generation'); // 给了替代写法，而不是只说"别用中文"
  });
});

describe('导入 payload', () => {
  it('只提交勾中的条目，且只带后端需要的字段', () => {
    const a = hit({ source_id: 'a' });
    const b = hit({ source_id: 'b' });
    const payload = buildImportPayload([a, b], ['b']);
    expect(payload.items).toHaveLength(1);
    expect(payload.items[0].source_id).toBe('b');
    expect(Object.keys(payload.items[0]).sort()).toEqual(
      ['pdf_url', 'source', 'source_id', 'title', 'version'].sort(),
    );
  });

  it('selected 里有脏 id（不在结果里）不会凭空造条目', () => {
    const payload = buildImportPayload([hit({ source_id: 'a' })], ['a', 'ghost']);
    expect(payload.items.map((i) => i.source_id)).toEqual(['a']);
  });

  it('缺 pdf_url 的条目被跳过（后端还会做 SSRF 校验，这里先挡空值）', () => {
    const payload = buildImportPayload([hit({ source_id: 'a', pdf_url: '' })], ['a']);
    expect(payload.items).toHaveLength(0);
  });
});

describe('字节与进度', () => {
  it('formatBytes 单位正确、未知值返回空串', () => {
    expect(formatBytes(0)).toBe('');
    expect(formatBytes(-1)).toBe('');
    expect(formatBytes(2048)).toBe('2 KB');
    expect(formatBytes(5.81 * 1048576)).toBe('5.8 MB');
  });

  it('downloadPct 只在有总长度时给百分比（否则 null ⇒ 前端显示不确定态）', () => {
    expect(downloadPct(task({ total: 0 }))).toBeNull();
    expect(downloadPct(task({ downloaded: 50, total: 100 }))).toBe(50);
    expect(downloadPct(task({ downloaded: 999, total: 100 }))).toBe(100); // 防越界
    expect(downloadPct(task({ status: 'ready', downloaded: 1, total: 1 }))).toBeNull();
  });
});

describe('任务单行文案', () => {
  it('下载中带百分比与大小（228 秒的教训：必须给字节数）', () => {
    const t = task({ downloaded: Math.round(2.1 * 1048576), total: Math.round(5.8 * 1048576) });
    const s = taskStatusText(t);
    expect(s).toContain('下载中');
    expect(s).toContain('%');
    expect(s).toContain('MB');
  });

  it('下载中但长度未知 ⇒ 只显示已下载字节（不编造百分比）', () => {
    const s = taskStatusText(task({ downloaded: 1024, total: 0 }));
    expect(s).toBe('下载中 · 1 KB');
  });

  it('解析中优先用文档进度百分比，其次阶段名', () => {
    expect(taskStatusText(task({ status: 'ingesting' }), { pct: 30 })).toBe('解析中 30%');
    expect(taskStatusText(task({ status: 'ingesting' }), { stage: 'chunking' })).toBe('解析中 · chunking');
    expect(taskStatusText(task({ status: 'ingesting' }))).toBe('解析中');
  });

  it('就绪区分"新导入"与"已在库中"', () => {
    expect(taskStatusText(task({ status: 'ready' }))).toBe('已就绪');
    expect(taskStatusText(task({ status: 'ready', duplicate: true }))).toBe('已在库中');
  });

  it('失败必须带可读原因（不能只显示一个红点）', () => {
    expect(taskStatusText(task({ status: 'failed', error: '论文过大（72 MB > 60 MB 上限）' })))
      .toBe('失败：论文过大（72 MB > 60 MB 上限）');
    expect(taskStatusText(task({ status: 'failed' }))).toBe('失败');
  });
});

describe('队列汇总', () => {
  it('activeTaskCount 只数"还在跑"的（失败/完成不算）', () => {
    expect(activeTaskCount([
      task({ status: 'queued' }),
      task({ status: 'downloading' }),
      task({ status: 'ingesting' }),
      task({ status: 'ready' }),
      task({ status: 'failed' }),
    ])).toBe(3);
  });

  it('hasActionableTask：有失败或有可跳转的成品才需要提示', () => {
    expect(hasActionableTask([task({ status: 'downloading' })])).toBe(false);
    expect(hasActionableTask([task({ status: 'failed' })])).toBe(true);
    expect(hasActionableTask([task({ status: 'ready', doc_id: 7 })])).toBe(true);
    expect(hasActionableTask([task({ status: 'ready', doc_id: null })])).toBe(false);
  });
});

describe('任务合并（切页/刷新后恢复）', () => {
  it('去重且**新数据覆盖旧数据**（同一任务不能出现两条）', () => {
    const prev = [{ id: 'a', v: 1 }, { id: 'b', v: 1 }];
    const incoming = [{ id: 'a', v: 2 }];     // 后端把 a 更新了
    const merged = mergeTasks(prev, incoming);
    expect(merged).toHaveLength(2);
    expect(merged.find((x) => x.id === 'a')?.v).toBe(2);  // 以 incoming 为准
  });

  it('顺序：incoming 在前，其余保持原序', () => {
    const merged = mergeTasks([{ id: 'old1' }, { id: 'old2' }], [{ id: 'new1' }]);
    expect(merged.map((x) => x.id)).toEqual(['new1', 'old1', 'old2']);
  });

  it('空输入不炸', () => {
    expect(mergeTasks([], [])).toEqual([]);
    expect(mergeTasks([{ id: 'x' }], [])).toHaveLength(1);
  });
});

describe('检索历史（带时间戳）', () => {
  const at = '2026-10-18T14:00:00Z';

  it('新记录在最前，且同词只保留一条（提到最前 + 更新时间）', () => {
    let h = pushSearchHistory([], 'rag', at);
    h = pushSearchHistory(h, 'diffusion', at);
    h = pushSearchHistory(h, 'rag', '2026-10-18T15:00:00Z');
    expect(h.map((x) => x.q)).toEqual(['rag', 'diffusion']);
    expect(h[0].at).toBe('2026-10-18T15:00:00Z');   // 时间戳被更新，不是留旧的
  });

  it('去首尾空白；空查询不记（防"点了检索但没输入"污染历史）', () => {
    expect(pushSearchHistory([], '  ', at)).toEqual([]);
    expect(pushSearchHistory([], ' rag ', at)[0].q).toBe('rag');
  });

  it('超出上限截断（默认 8）', () => {
    let h: ReturnType<typeof pushSearchHistory> = [];
    for (let i = 0; i < 12; i += 1) h = pushSearchHistory(h, `q${i}`, at, 5);
    expect(h).toHaveLength(5);
    expect(h[0].q).toBe('q11');
  });

  it('脏 localStorage 不炸（非法 JSON / 非数组 / 空项）', () => {
    expect(parseSearchHistory(null)).toEqual([]);
    expect(parseSearchHistory('not json')).toEqual([]);
    expect(parseSearchHistory('{"a":1}')).toEqual([]);
    expect(parseSearchHistory('[{"q":"x","at":"t"},{"q":"  "},{"nope":1}]')).toEqual([{ q: 'x', at: 't' }]);
  });
});

describe('加载更多（追加去重）', () => {
  it('追加到末尾，且同一 source_id 只保留一条', () => {
    const a = [{ source_id: 'a' }, { source_id: 'b' }];
    const b = [{ source_id: 'b' }, { source_id: 'c' }];   // b 重复（分页边界重叠）
    expect(appendHits(a, b).map((x) => x.source_id)).toEqual(['a', 'b', 'c']);
  });
  it('空输入不炸', () => {
    expect(appendHits([], [])).toEqual([]);
    expect(appendHits([{ source_id: 'a' }], [])).toHaveLength(1);
  });
});
