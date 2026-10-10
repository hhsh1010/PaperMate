import type { ReactNode } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { saveDocxExport } from "../lib/saveFile";
import { renderAsync } from "docx-preview";
import { api } from "../api/client";
import type { ApplyEditsOut, EditOp, ParaAction, ParaTransformOut, StreamItem } from "../lib/types";
import { findTextByPid, groupCells, withQuote } from "../lib/workbench";
import { alignStreamDom, findAlignedEl } from "../lib/docxAlign";
import { useCtrlWheelZoom } from "../lib/useCtrlWheelZoom";
import { Icons } from "../lib/icons";
import { toErrorMessage } from "../lib/error";

/**
 * C4：改稿工作台 —— 主文献编辑区 + 变更集暂存 + 应用/撤销。
 *
 * 双视图（P1）：
 * - 图文模式（默认）：docx-preview 按原文档渲染（章节/表格/图片），点击段落只改文字；
 *   DOM 与后端文档流编址（B 地址）按序对齐，对齐失败自动回退结构模式（quote 阀兜底正确性）。
 * - 结构模式：段落卡片列表（C4 原有交互，含表格 T 地址）。
 *
 * 编辑模型（2026-09-05 重构）：**编辑即落盘** —— 任何修改（改写 / 追加 / 下插 / 删除 /
 * AI 采用）在「保存」时立即走 POST /apply-edits 写入 docx（track-changes + 自动备份），
 * 随后重载文档流并递增 docTick，使结构模式卡片与图文模式渲染同步刷新为新内容，
 * 并绿色高亮 + 滚动定位到该段。撤销走「撤销上次应用」（还原 .bak 备份，可反悔）。
 * 早期版本先入「变更集」本地暂存、需二次点「应用修改」，导致「改了看不到」——已废弃。
 *
 * 已知分期：agent 建议卡（解析 /改稿 结构化输出自动填段）依赖输出契约，后续迭代。
 */
export function RevisionWorkbench({
  docId,
  docName,
  onMutated,
}: {
  docId: number;
  docName?: string;
  /** 应用 / 撤销导致文档内容变化后通知外部（如刷新文件区预览） */
  onMutated?: () => void;
}) {
  /** 视图模式：fidelity=原文渲染（默认） | structure=段落卡片/表格网格 */
  const [viewMode, setViewMode] = useState<"fidelity" | "structure">("fidelity");

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // ---------- 图文模式状态 ----------
  const docxRef = useRef<HTMLDivElement | null>(null);
  /** 任何编辑（AI 采用 / 改写 / 追加 / 下插 / 删除）落盘后，需高亮+滚动定位的目标段 pid（渲染完成后消费） */
  const pendingFocusRef = useRef<string | null>(null);
  /** Ctrl+滚轮缩放（原生非 passive 监听；只作用于图文模式渲染区） */
  const { ref: fidWheelRef, zoom: fidZoom } = useCtrlWheelZoom<HTMLDivElement>();
  /** 文档流条目（两种视图共用数据源：结构模式渲染卡片/表格网格；图文模式用于对齐与 quote） */
  const [streamItems, setStreamItems] = useState<StreamItem[]>([]);
  const [fidLoading, setFidLoading] = useState(false);

  /** ★ P-04（2026-10-09）：另存为导出的进行态（**防重复点击**）+ 结果提示。
   *  背景：原先两个按钮是 `<a download>`，桌面端（Tauri）静默落盘、界面零反馈
   *  ⇒ 用户以为无效、连点 12 次（实测）。改为「原生另存为 → 后端写盘 → 明确路径」。 */
  const [savingDocx, setSavingDocx] = useState<"tracked" | "clean" | null>(null);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  const saveAs = useCallback(
    async (mode: "tracked" | "clean") => {
      setSavingDocx(mode);
      setSaveMsg(null);
      try {
        const base = docName?.replace(/\.docx$/i, "") || `文档${docId}`;
        // ★ 2026-10-10（用户要求）：成功**不再显示**右侧提示（长路径挤占顶栏）；
        //   仅**失败**可见（保存失败不许静默）。取消也不提示。
        await saveDocxExport(docId, mode, `${base}${mode === "clean" ? "_干净版" : ""}.docx`);
      } catch (e) {
        // ★ 失败必须可见（不许静默——与"假装有进度"同族）
        setSaveMsg(`保存失败：${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setSavingDocx(null);
      }
    },
    [docId, docName],
  );
  const [fidError, setFidError] = useState<string | null>(null);
  const [fidAligned, setFidAligned] = useState<boolean | null>(null);
  /** 图文模式选中的段落（点击注入 pid 的元素） */
  const [fidSel, setFidSel] = useState<{ pid: string; text: string; editable: boolean; media: boolean; kind: "para" | "table" } | null>(null);
  const [fidEditing, setFidEditing] = useState<{ pid: string; text: string } | null>(null);
  const [fidCompose, setFidCompose] = useState<{ pid: string; op: "append" | "insert_after"; text: string } | null>(null);
  /** 应用/撤销后自增，触发图文模式重新渲染（内容已变化） */
  const [docTick, setDocTick] = useState(0);

  // ---------- AI 段落改写（扩写/缩写/润色/去重；采用即写入文档） ----------
  type AiState = {
    pid: string;
    anchor: string; // 原段落文本（quote 锚）
    phase: "pick" | "loading" | "done";
    action?: ParaAction;
    result?: ParaTransformOut;
    editText?: string; // 结果可微调后再采用
    error?: string;
  };
  const [ai, setAi] = useState<AiState | null>(null);

  const AI_LABEL: Record<ParaAction, string> = { expand: "扩写", shorten: "缩写", polish: "润色", dedupe: "去重" };
  /** 卡片图标与单行说明，让 4 个动作语义一眼可辨 */
  const AI_HINT: Record<ParaAction, string> = {
    expand: "补充论证 · 约 1.5× 篇幅",
    shorten: "压缩要点 · 保留核心",
    polish: "语法流畅 · 学术化",
    dedupe: "消除段内重复表述",
  };
  const AI_ICON: Record<ParaAction, ReactNode> = {
    expand: <Icons.maximize size={14} />,
    shorten: <Icons.minimize size={14} />,
    polish: <Icons.sparkles size={14} />,
    dedupe: <Icons.chevronDown size={14} />,
  };

  /** 目标段的相邻段落文本（供模型理解上下文；仅参考不输出） */
  const contextOf = useCallback(
    (pid: string) => {
      const idx = streamItems.findIndex((i) => i.pid === pid);
      const before = [...streamItems.slice(0, Math.max(0, idx))].reverse().find((i) => i.kind === "para" && i.text?.trim());
      const after = streamItems.slice(idx + 1).find((i) => i.kind === "para" && i.text?.trim());
      return { before: before?.text ?? null, after: after?.text ?? null };
    },
    [streamItems],
  );

  const openAi = useCallback(
    (pid: string, anchor: string) => {
      if (!anchor.trim()) return;
      setEditing(null);
      setComposing(null);
      setFidEditing(null);
      setFidCompose(null);
      setAi({ pid, anchor, phase: "pick" });
    },
    [],
  );

  /** 请求序号（2026-09-08 修复 J2）：快速切换主文献时，先发的 `listStream(旧 docId)` 可能后到，
   *  把新文档的段落覆盖掉——后果是「结构模式显示 A 的 pid，保存却命中 B 文档」。
   *  用单调递增序号丢弃过期响应（比 AbortController 更稳：不依赖后端支持中断）。 */
  const reloadSeqRef = useRef(0);
  const reload = useCallback(async () => {
    const seq = ++reloadSeqRef.current;
    setLoading(true);
    setLoadError(null);
    try {
      // 结构模式与图文模式共用文档流数据源（按文档实际顺序，表格以 cells 携带）
      const items = await api.listStream(docId);
      if (seq !== reloadSeqRef.current) return; // 已有更新的请求发出，丢弃本次过期结果
      setStreamItems(items);
    } catch (e: unknown) {
      if (seq !== reloadSeqRef.current) return;
      setLoadError(toErrorMessage(e) || "文档加载失败");
    } finally {
      if (seq === reloadSeqRef.current) setLoading(false);
    }
  }, [docId]);

  // N14-d（2026-09-12）：**移除上一版的滚动补偿**。
  // 面板渲染在段落列表**之后**（文档末尾），它并不会挤压任何已渲染的段落——
  // 因此上一版加的「面板变高就把 scrollTop 补 delta」属于**误判**：
  // 它把原本稳定的视图整体推走，反而**制造了新的位移**（用户第四次反馈仍在跳动）。
  // 结论：本场景不需要补偿；真正需要的是让面板**脱离文档流**（见 .rw-aipanel 的 sticky/fixed 处理）。

  const runAi = useCallback(
    async (action: ParaAction) => {
      if (!ai) return;
      setAi({ ...ai, phase: "loading", action, error: undefined });
      const ctx = contextOf(ai.pid);
      try {
        const out = await api.transformText({
          text: ai.anchor,
          action,
          context_before: ctx.before,
          context_after: ctx.after,
        });
        setAi({
          ...ai,
          phase: "done",
          action,
          result: out,
          editText: out.transformed,
          error: out.applied ? undefined : (out.reason ?? "未能生成有效改写"),
        });
      } catch (e: unknown) {
        setAi({ ...ai, phase: "done", action, error: toErrorMessage(e) || "调用失败，请重试" });
      }
    },
    [ai, contextOf],
  );

  /** 编辑即落盘（实时显示的核心）：把 ops 直接写入文档（track-changes + 自动备份），
   * 随后重载文档流并高亮定位该段 → 结构模式卡片与图文模式渲染同步刷新为新内容。
   * 所有编辑入口（改写 / 追加 / 下插 / 删除 / AI 采用）统一复用，避免「暂存后无反馈」。 */
  const writeAndRefresh = useCallback(
    async (ops: EditOp[], focusPid: string | null) => {
      if (!ops.length) return true;
      // 2026-09-06 修复：并发守卫（ref 即时判，state 有渲染延迟）——此前 applying 仅禁用了
      // purge 与 AI 采用按钮，保存/删除按钮连点可并发 apply-edits（第二条会因 quote 漂移被拒）。
      if (applyingRef.current) return false;
      applyingRef.current = true;
      setApplying(true);
      setActionError(null);
      try {
        const out = await api.applyEdits(docId, ops);
        if (out.applied.length === 0 && out.rejected.length > 0) {
          setActionError(out.rejected[0]?.reason ?? "应用失败：段落可能已被改动");
          return false;
        }
        setApplyResult(out);
        // 累积高亮：多次编辑的「已改」标记要并存，供回看/定位全部改动处；
        // 由「撤销」或关闭结果条清空（见 undo 与 rw-result 关闭按钮）。
        setAppliedMap((prev) => {
          const next = new Map(prev);
          out.applied.forEach((a) => next.set(a.pid, a.op));
          return next;
        });
        if (focusPid) pendingFocusRef.current = focusPid; // 渲染完成后消费：高亮 + 滚动定位

        // N14-b（2026-09-12）：**纯改写（replace）走局部更新，绝不整篇重渲染**。
        // 背景：上一版改为「离屏渲染 + 一次性替换」后仍是割裂的——因为整篇 docx
        // （约 2.8MB DOM）**被重建**这一事实本身就会被感知为「重来一次」。
        // 对 replace 而言段落地址（pid）不变 → 只需就地改这一段：
        //   · 结构视图：setStreamItems 只更新命中项（不整表重建）；
        //   · 图文视图：直接改 [data-pid] 元素的文本，其余内容一律不动。
        // 增删类操作（insert / append / delete / purge）会改变段落地址，仍走全量刷新。
        if (ops.every((o) => o.op === "replace")) {
          setStreamItems((prev) =>
            prev.map((it) => {
              const hit = ops.find((o) => o.pid === it.pid);
              return hit ? { ...it, text: hit.text } : it;
            }),
          );
          ops.forEach((op) => {
            const el = docxRef.current?.querySelector(`[data-pid="${CSS.escape(op.pid)}"]`);
            if (!el) return;
            el.textContent = op.text;
            el.classList.add("rw-doc-applied"); // 常驻高亮：标记改动处
          });
          if (focusPid) {
            docxRef.current
              ?.querySelector(`[data-pid="${CSS.escape(focusPid)}"]`)
              ?.scrollIntoView({ behavior: "smooth", block: "center" });
            pendingFocusRef.current = null;
          }
          onMutated?.();
          return true;
        }

        await reload();
        setDocTick((t) => t + 1); // 图文模式重渲染（内容已变化 → 实时显示新文字）
        onMutated?.();
        return true;
      } catch (e: unknown) {
        setActionError(toErrorMessage(e) || "应用修改失败，请确认后端服务正常");
        return false;
      } finally {
        applyingRef.current = false;
        setApplying(false);
      }
    },
    [docId, reload, onMutated],
  );

  /** AI 改写「采用并写入文档」：改写文本作为 replace op 直接落盘（复用 writeAndRefresh）。 */
  const adoptAndApply = useCallback(async () => {
    if (!ai || ai.phase !== "done" || !ai.editText?.trim()) return;
    const pid = ai.pid;
    // 以文档流中该段【当前】文本作为 quote 锚，避免打开面板后文档已变化导致引用漂移被后端拒绝
    const anchor = findTextByPid(streamItems, pid) ?? ai.anchor;
    const op: EditOp = withQuote({ pid, op: "replace", text: ai.editText! }, anchor);
    setAi(null);
    setFidSel(null);
    await writeAndRefresh([op], pid);
  }, [ai, streamItems, writeAndRefresh]);

  const AI_ACTIONS: ParaAction[] = ["expand", "shorten", "polish", "dedupe"];

  /** 已落盘的 pid → op：编辑即写盘，用此集合保留「改过的文字」绿色高亮，直到关闭结果条/撤销 */
  const [appliedMap, setAppliedMap] = useState<Map<string, string>>(new Map()); // pid -> op（已写入文档）
  const [applying, setApplying] = useState(false);
  /** applying 的 ref 镜像：writeAndRefresh 内即时判并发（state 有渲染延迟，闭包里读旧值） */
  const applyingRef = useRef(false);
  const [restoring, setRestoring] = useState(false);
  const [applyResult, setApplyResult] = useState<ApplyEditsOut | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  /** 编辑态：replace 正在改写的段落 */
  const [editing, setEditing] = useState<{ pid: string; text: string } | null>(null);
  /** 输入态：append / insert_after 正在输入的段落 */
  const [composing, setComposing] = useState<{ pid: string; op: "append" | "insert_after"; text: string } | null>(null);

  useEffect(() => {
    setEditing(null);
    setComposing(null);
    setApplyResult(null);
    void reload();
  }, [reload]);

  // ---------- 图文模式：渲染 + 对齐（P1） ----------
  useEffect(() => {
    if (viewMode !== "fidelity") return;
    let cancelled = false;
    const container = docxRef.current;
    if (!container) return;
    setFidLoading(true);
    setFidError(null);
    setFidAligned(null);
    setFidSel(null);
    setFidEditing(null);
    setFidCompose(null);
    // N14（2026-09-12）：**不再先清空容器**。
    // 原先 `container.innerHTML = ""` 在前、异步 renderAsync 在后 → 中间存在明显空白期，
    // 且滚动位置归零；改写成功后用户看到「页面刷新一下 / 位置跳掉」的割裂感。
    // 改为「离屏渲染 → 渲染完成后一次性替换 + 恢复滚动位置」，视觉上连续无闪烁。
    (async () => {
      try {
        // 缓存穿透：docTick 变化 = 文档已落盘变更，必须取最新字节（否则浏览器返回旧渲染）
        const [fileRes, items] = await Promise.all([
          fetch(`/api/v1/documents/${docId}/file?t=${docTick}`),
          api.listStream(docId),
        ]);
        if (!fileRes.ok) throw new Error("原文件获取失败");
        const blob = await fileRes.blob();
        if (cancelled || !docxRef.current) return;
        // 离屏渲染：先把新文档渲染到临时容器，完成后再一次性替换（消除中间空白期）
        const offscreen = document.createElement("div");
        await renderAsync(blob, offscreen, undefined, {
          className: "docx",
          inWrapper: true,
          ignoreHeight: true,
          useBase64URL: true, // 图片内嵌显示
        });
        if (cancelled || !docxRef.current) return;
        const host = docxRef.current;
        // 滚动容器可能是 host 自身或其父级（样式差异），两者都取，替换后按原值恢复
        const parent = host.parentElement;
        const prevTop = host.scrollTop || parent?.scrollTop || 0;
        const prevLeft = host.scrollLeft || parent?.scrollLeft || 0;
        host.replaceChildren(...Array.from(offscreen.childNodes));
        if (prevTop) {
          if (host.scrollHeight > host.clientHeight) host.scrollTop = prevTop;
          else if (parent && parent.scrollHeight > parent.clientHeight) parent.scrollTop = prevTop;
        }
        if (prevLeft) {
          if (host.scrollWidth > host.clientWidth) host.scrollLeft = prevLeft;
          else if (parent && parent.scrollWidth > parent.clientWidth) parent.scrollLeft = prevLeft;
        }
        setStreamItems(items);
        const aligned = alignStreamDom(docxRef.current, items);
        setFidAligned(aligned.ok);
        if (!aligned.ok) {
          setFidError("图文视图与文档编址未对齐（渲染层折损），已切换为只读；请用「结构模式」编辑。");
        } else {
          // 已应用段：常驻高亮（绿色，便于回看改了哪里）；本次定位段：强提醒 + 滚动到视图中央
          appliedMap.forEach((_op, pid) => {
            docxRef.current?.querySelector(`[data-pid="${CSS.escape(pid)}"]`)?.classList.add("rw-doc-applied");
          });
          if (pendingFocusRef.current) {
            const el = docxRef.current?.querySelector(`[data-pid="${CSS.escape(pendingFocusRef.current)}"]`);
            if (el) {
              el.classList.add("rw-doc-applied");
              el.scrollIntoView({ behavior: "smooth", block: "center" });
            }
            pendingFocusRef.current = null;
          }
        }
      } catch (e: unknown) {
        if (!cancelled) setFidError(toErrorMessage(e) || "图文视图加载失败，请用「结构模式」编辑。");
      } finally {
        if (!cancelled) setFidLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [viewMode, docId, docTick]);

  /** 图文模式：点击已编址元素 → 选中（事件委托） */
  const onDocClick = useCallback(
    (e: React.MouseEvent) => {
      const el = findAlignedEl(e.target);
      if (!el?.dataset.pid) return;
      const pid = el.dataset.pid;
      const item = streamItems.find((it) => it.pid === pid);
      if (!item) return;
      setFidEditing(null);
      setFidCompose(null);
      setFidSel({
        pid,
        text: item.kind === "para" ? (item.text ?? "") : "",
        editable: item.kind === "para" && item.editable === true,
        media: item.kind === "para" && item.media === true,
        kind: item.kind,
      });
    },
    [streamItems],
  );

  /** 选中高亮：遍历容器标记当前选中元素 */
  useEffect(() => {
    const container = docxRef.current;
    if (!container) return;
    container.querySelectorAll(".rw-doc-selected").forEach((el) => el.classList.remove("rw-doc-selected"));
    if (fidSel) {
      container.querySelector(`[data-pid="${CSS.escape(fidSel.pid)}"]`)?.classList.add("rw-doc-selected");
    }
  }, [fidSel]);

  /** 结构模式：落盘后把定位段滚动到视图中央（图文模式在渲染回调内处理） */
  useEffect(() => {
    if (viewMode !== "structure" || !pendingFocusRef.current) return;
    const root = document.querySelector<HTMLElement>(".rw-paras");
    const el = root?.querySelector<HTMLElement>(`[data-pid="${CSS.escape(pendingFocusRef.current)}"]`);
    if (el) {
      el.classList.add("rw-flash");
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    pendingFocusRef.current = null;
  }, [viewMode, streamItems]);

  /** 图文模式：保存改写（replace，quote=编址时文本）→ 即时写入文档 */
  const saveFidEdit = useCallback(async () => {
    if (!fidEditing) return;
    const pid = fidEditing.pid;
    const item = streamItems.find((x) => x.pid === pid);
    const anchor = item?.text ?? "";
    if (fidEditing.text === anchor) { // 无变化
      setFidEditing(null);
      return;
    }
    // 2026-09-08 修复 J5：写盘**成功**才清编辑态。原先先 setFidEditing(null)，
    // 若 apply-edits 失败（quote 漂移 / 后端异常），用户刚输入的内容会被静默丢弃且无法重试。
    const ok = await writeAndRefresh([withQuote({ pid, op: "replace", text: fidEditing.text }, anchor)], pid);
    if (ok) setFidEditing(null);
  }, [fidEditing, streamItems, writeAndRefresh]);

  /** 图文模式：提交追加/下插 → 即时写入文档 */
  const submitFidCompose = useCallback(async () => {
    if (!fidCompose || !fidCompose.text.trim()) {
      setFidCompose(null);
      return;
    }
    const pid = fidCompose.pid;
    const item = streamItems.find((x) => x.pid === pid);
    const op = fidCompose.op;
    const text = fidCompose.text;
    // 同 saveFidEdit（J5）：写盘成功才清输入态，失败保留用户输入
    const ok = await writeAndRefresh([withQuote({ pid, op, text }, item?.text ?? "")], pid);
    if (ok) setFidCompose(null);
  }, [fidCompose, streamItems, writeAndRefresh]);

  /** 图文模式：删除整段 → 即时写入文档 */
  const fidDelete = useCallback(async () => {
    if (!fidSel?.editable) return;
    const pid = fidSel.pid;
    const item = streamItems.find((x) => x.pid === pid);
    await writeAndRefresh([withQuote({ pid, op: "delete", text: "" }, item?.text ?? "")], pid);
  }, [fidSel, streamItems, writeAndRefresh]);

  /** 结构模式：保存改写 → 即时写入文档 */
  const saveEdit = useCallback(async () => {
    if (!editing) return;
    const pid = editing.pid;
    const anchor = findTextByPid(streamItems, pid) ?? "";
    if (editing.text === anchor) { // 无变化
      setEditing(null);
      return;
    }
    // 同 saveFidEdit（J5）：写盘成功才清编辑态，失败保留用户输入
    const ok = await writeAndRefresh([withQuote({ pid, op: "replace", text: editing.text }, anchor)], pid);
    if (ok) setEditing(null);
  }, [editing, streamItems, writeAndRefresh]);

  /** 结构模式：提交追加/下插 → 即时写入文档 */
  const submitCompose = useCallback(async () => {
    if (!composing || !composing.text.trim()) {
      setComposing(null);
      return;
    }
    const pid = composing.pid;
    const anchor = findTextByPid(streamItems, pid);
    const op = composing.op;
    const text = composing.text;
    if (anchor === null) {
      setComposing(null);
      return;
    }
    // 同 saveFidEdit（J5）：写盘成功才清输入态，失败保留用户输入
    const ok = await writeAndRefresh([withQuote({ pid, op, text }, anchor)], pid);
    if (ok) setComposing(null);
  }, [composing, streamItems, writeAndRefresh]);

  /** 结构模式：删除整段 → 即时写入文档 */
  const markDelete = useCallback(
    async (pid: string) => {
      const anchor = findTextByPid(streamItems, pid);
      if (anchor === null) return;
      await writeAndRefresh([withQuote({ pid, op: "delete", text: "" }, anchor)], pid);
    },
    [streamItems, writeAndRefresh],
  );

  /** 结构模式：彻底删除（purge）——物理移除标记删除后的空壳段落。
   * 后端安全阀：仅允许有效文本已清空的正文段落；移除后 pid 不复存在，无需定位、清其高亮。 */
  const purgePara = useCallback(
    async (pid: string) => {
      const ok = await writeAndRefresh([{ pid, op: "purge", text: "" }], null);
      if (ok) {
        setAppliedMap((prev) => {
          const next = new Map(prev);
          next.delete(pid);
          return next;
        });
      }
    },
    [writeAndRefresh],
  );

  const undo = useCallback(async () => {
    // 2026-09-08 修复 J7：原先只判 restoring——apply 写入期间点「撤销」会让
    // restoreBackup 与 apply-edits **并发**，备份被刚写入的内容污染，撤销后回不到预期状态。
    if (restoring || applyingRef.current) return;
    setRestoring(true);
    setActionError(null);
    try {
      await api.restoreBackup(docId);
      setApplyResult(null);
      setAppliedMap(new Map());
      setFidSel(null);
      await reload();
      setDocTick((t) => t + 1); // 图文模式重渲染（内容已回退）
      onMutated?.();
    } catch (e: unknown) {
      setActionError(toErrorMessage(e) || "撤销失败（可能没有可用备份）");
    } finally {
      setRestoring(false);
    }
  }, [restoring, docId, reload, onMutated]);

  const opLabel: Record<EditOp["op"], string> = {
    replace: "已改写",
    append: "已追加",
    insert_after: "已下插",
    delete: "已删除",
    purge: "已彻底删除",
  };

  /** 结果条 4s 自动消失：只收起横幅；appliedMap 绿色高亮保留，供回看改动位置（撤销时才清） */
  useEffect(() => {
    if (!applyResult) return;
    const t = setTimeout(() => setApplyResult(null), 4000);
    return () => clearTimeout(t);
  }, [applyResult]);

  return (
    <div className="rw">
      {/* 工作台头部（紧凑单行）+ 视图切换（P1：图文模式 / 结构模式） */}
      <div className="rw-head">
        <span className="rw-title" title={docName ?? `文档 #${docId}`}>
          <Icons.fileText size={14} />
          <span className="rw-title-text">{docName ?? `文档 #${docId}`}</span>
        </span>
        <span className="rw-spacer" />
        <span className="rw-viewtabs" role="group" aria-label="工作台视图" title="纯文本编辑 · 保存时保留格式与修订痕迹">
          <button
            className={viewMode === "fidelity" ? "on" : ""}
            onClick={() => setViewMode("fidelity")}
            title="按原文档渲染（章节/表格/图片），点击段落改文字"
          >
            图文
          </button>
          <button
            className={viewMode === "structure" ? "on" : ""}
            onClick={() => setViewMode("structure")}
            title="段落卡片列表（含表格单元格逐段编辑）"
          >
            结构
          </button>
        </span>
      </div>

      {viewMode === "structure" && loading && <div className="rw-loading"><span className="spinner" aria-hidden="true" />加载段落中…</div>}
      {viewMode === "structure" && loadError && (
        <div className="rw-error">
          {loadError}
          <button className="batch-btn ghost" onClick={() => void reload()}>重试</button>
        </div>
      )}

      {/* 应用结果反馈（紧凑：不含备份长文件名；4s 自动消失，绿色高亮保留供回看） */}
      {applyResult && (
        <div className="rw-result ok" role="status">
          <Icons.checkCircle size={13} />
          已应用 {applyResult.applied.length} 处修改
          {applyResult.rejected.length > 0 && ` · ${applyResult.rejected.length} 处被安全拒绝`}
          <button
            className="icon-btn"
            onClick={() => setApplyResult(null)}
            aria-label="关闭提示"
            title="已自动备份，可用「撤销上次应用」还原"
          >
            <Icons.x size={13} />
          </button>
        </div>
      )}
      {actionError && (
        <div className="rw-error">
          {actionError}
          <button className="icon-btn" onClick={() => setActionError(null)} aria-label="关闭提示">
            <Icons.x size={13} />
          </button>
        </div>
      )}

      {/* P1 图文模式：原文档渲染 + 点击段落改文字（成功态不再占一条提示栏，仅保留错误提示） */}
      {viewMode === "fidelity" && (
        <>
          {fidLoading && <div className="rw-loading"><span className="spinner" aria-hidden="true" />渲染原文档中…</div>}
          {fidError && <div className="rw-error">{fidError}</div>}
          <div className="rw-docx-wrap" ref={fidWheelRef} style={{ zoom: fidZoom / 100 } as React.CSSProperties} title="点击段落改文字 · Ctrl+滚轮缩放">
            <div ref={docxRef} className="rw-docx" onClick={onDocClick} role="presentation" />
          </div>

          {/* 选中段操作条 + 编辑面板 */}
          {fidSel && fidAligned === true && (
            <div className="rw-fidpanel">
              {fidEditing ? (
                <div className="rw-edit">
                  <textarea
                    className="rw-textarea"
                    value={fidEditing.text}
                    onChange={(e) => setFidEditing({ ...fidEditing, text: e.target.value })}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) saveFidEdit();
                      if (e.key === "Escape") setFidEditing(null);
                    }}
                    rows={Math.min(10, Math.max(3, Math.ceil(fidEditing.text.length / 50)))}
                    autoFocus
                  />
                  <div className="rw-edit-ops">
                    <span className="rw-hint">Ctrl+Enter 保存 · Esc 取消 · {fidEditing.pid}</span>
                    <button className="batch-btn ghost" onClick={() => setFidEditing(null)}>取消</button>
                    <button className="batch-btn primary" onClick={saveFidEdit} disabled={applying}>保存并写入文档</button>
                  </div>
                </div>
              ) : fidCompose ? (
                <div className="rw-edit">
                  <textarea
                    className="rw-textarea"
                    placeholder={fidCompose.op === "append" ? "追加到本段末尾的内容…" : "插入的新段落内容…"}
                    value={fidCompose.text}
                    onChange={(e) => setFidCompose({ ...fidCompose, text: e.target.value })}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submitFidCompose();
                      if (e.key === "Escape") setFidCompose(null);
                    }}
                    rows={3}
                    autoFocus
                  />
                  <div className="rw-edit-ops">
                    <span className="rw-hint">Ctrl+Enter 保存 · Esc 取消 · {fidCompose.pid}</span>
                    <button className="batch-btn ghost" onClick={() => setFidCompose(null)}>取消</button>
                    <button className="batch-btn primary" onClick={submitFidCompose} disabled={applying}>
                      {fidCompose.op === "append" ? "追加并写入文档" : "下插并写入文档"}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="rw-fidsel">
                  <span className="rw-pid">{fidSel.pid}</span>
                  {fidSel.editable ? (
                    <>
                      <span className="rw-ops" style={{ display: "inline-flex" }}>
                        <button className="rw-op" onClick={() => { setFidEditing({ pid: fidSel.pid, text: fidSel.text }); }}>
                          <Icons.copy size={12} /> 改写
                        </button>
                        <button className="rw-op" onClick={() => setFidCompose({ pid: fidSel.pid, op: "append", text: "" })}>
                          <Icons.plus size={12} /> 追加
                        </button>
                        <button className="rw-op" onClick={() => setFidCompose({ pid: fidSel.pid, op: "insert_after", text: "" })}>
                          <Icons.arrowDown size={12} /> 下插
                        </button>
                        <button className="rw-op danger" onClick={fidDelete} disabled={applying}>
                          <Icons.trash size={12} /> 删除
                        </button>
                        <button className="rw-op" title="AI 改写（扩写/缩写/润色/去重）" onClick={() => openAi(fidSel.pid, fidSel.text)}>
                          <Icons.sparkles size={12} /> AI功能
                        </button>
                        <button className="rw-op" onClick={() => setFidSel(null)}>取消选择</button>
                      </span>
                      {/* ★ 2026-10-10：媒体段解锁编辑——提示"仅改文字"，图和公式原地保留 */}
                      {fidSel.media && (
                        <span className="rw-hint">此段含图片/公式：仅修改文字部分，图和公式保持原位不动</span>
                      )}
                    </>
                  ) : (
                    <span className="rw-hint">此段不可编辑（空段或含域代码）{fidSel.kind === "table" ? "，表格请在「结构」模式编辑" : ""}</span>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* 结构模式：按文档实际顺序渲染段落卡片 + 表格网格（数据源 = 文档流） */}
      {viewMode === "structure" && (
      <div className="rw-paras">
        {streamItems.map((item) => {
          // 表格：整块网格渲染（行×列），单元格点击进入编辑
          if (item.kind === "table") {
            const rows = groupCells(item.cells ?? []);
            const anyApplied = (item.cells ?? []).some((c) => appliedMap.has(c.pid)) || appliedMap.has(item.pid);
            return (
              <div key={item.pid} data-pid={item.pid} className={`rw-para rw-table-block ${anyApplied ? "applied" : ""}`}>
                <div className="rw-para-head">
                  <span className="rw-pid">{item.pid}</span>
                  <span className="rw-style">表格 · {rows.length} 行</span>
                  {anyApplied && <span className="rw-pending applied">已修改</span>}
                </div>
                <table className="rw-grid">
                  <tbody>
                    {rows.map((row, ri) => (
                      <tr key={ri}>
                        {row.map((cell) => {
                          const isCellEditing = editing?.pid === cell.pid;
                          return (
                            <td key={cell.pid} className={`${appliedMap.has(cell.pid) ? "cell-applied" : ""}`}>
                              {isCellEditing ? (
                                <div className="rw-edit">
                                  <textarea
                                    className="rw-textarea"
                                    value={editing.text}
                                    onChange={(e) => setEditing({ ...editing, text: e.target.value })}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) saveEdit();
                                      if (e.key === "Escape") setEditing(null);
                                    }}
                                    rows={3}
                                    autoFocus
                                  />
                                  <div className="rw-edit-ops">
                                    <button className="batch-btn ghost" onClick={() => setEditing(null)}>取消</button>
                                    <button className="batch-btn primary" onClick={saveEdit} disabled={applying}>保存</button>
                                  </div>
                                </div>
                              ) : (
                                <div
                                  className={`rw-cell ${cell.editable ? "editable" : ""}`}
                                  title={cell.editable ? "点击编辑此单元格" : cell.reason ?? ""}
                                  onClick={() => {
                                    if (!cell.editable) return;
                                    setEditing({ pid: cell.pid, text: cell.text });
                                    setComposing(null);
                                  }}
                                >
                                  {cell.empty ? <span className="rw-hint">（空）</span> : cell.text}
                                  {appliedMap.has(cell.pid) && <span className="rw-pending applied">{opLabel[appliedMap.get(cell.pid) as EditOp["op"]]}</span>}
                                </div>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }

          // 段落卡片（B 地址）
          const p = {
            pid: item.pid,
            style: item.style ?? null,
            text: item.text ?? "",
            empty: item.empty ?? false,
            editable: item.editable ?? false,
            reason: item.reason ?? null,
            deleted: item.deleted ?? false,
          };
          const isEditing = editing?.pid === p.pid;
          const isComposing = composing?.pid === p.pid;
          // 标记删除后的空壳段：文档级事实（段落标记带 w:del），跨刷新稳定，不依赖前端状态
          const shellDeleted = p.empty === true && p.deleted === true;
          const appliedOp = appliedMap.get(p.pid) as EditOp["op"] | undefined;
          return (
            <div key={p.pid} data-pid={p.pid} className={`rw-para ${appliedOp || shellDeleted ? "applied" : ""} ${p.empty ? "empty" : ""}`}>
              <div className="rw-para-head">
                <span className="rw-pid">{p.pid}</span>
                {p.style && <span className="rw-style">{p.style}</span>}
                {appliedOp && <span className="rw-pending applied">{opLabel[appliedOp]}</span>}
                {!appliedOp && shellDeleted && <span className="rw-pending applied">已删除</span>}
                <span className="rw-spacer" />
                {/* 空壳段提供「彻底删除」物理移除入口（track-changes 不物理删段，壳会一直留着） */}
                {shellDeleted && !applying && (
                  <button
                    className="rw-op danger"
                    title="物理移除该空标记段落（清除痕迹；落盘前有备份，可用「撤销上次应用」还原）"
                    onClick={() => void purgePara(p.pid)}
                  >
                    <Icons.trash size={12} /> 彻底删除
                  </button>
                )}
                {p.editable && !isEditing && (
                  <span className="rw-ops">
                    <button
                      className="rw-op"
                      title="改写此段"
                      onClick={() => {
                        setEditing({ pid: p.pid, text: p.text });
                        setComposing(null);
                      }}
                    >
                      <Icons.copy size={12} /> 改写
                    </button>
                    <button
                      className="rw-op"
                      title="段尾追加一句"
                      onClick={() => {
                        setComposing({ pid: p.pid, op: "append", text: "" });
                        setEditing(null);
                      }}
                    >
                      <Icons.plus size={12} /> 追加
                    </button>
                    <button
                      className="rw-op"
                      title="在此段后插入新段"
                      onClick={() => {
                        setComposing({ pid: p.pid, op: "insert_after", text: "" });
                        setEditing(null);
                      }}
                    >
                      <Icons.arrowDown size={12} /> 下插
                    </button>
                    <button className="rw-op danger" title="删除整段（保留修订痕迹，可撤销）" onClick={() => markDelete(p.pid)} disabled={applying}>
                      <Icons.trash size={12} /> 删除
                    </button>
                    <button className="rw-op" title="AI 改写（扩写/缩写/润色/去重）" onClick={() => openAi(p.pid, p.text)}>
                      <Icons.sparkles size={12} /> AI功能
                    </button>
                  </span>
                )}
              </div>

              {/* 正文 / 编辑态 / 输入态 */}
              {isEditing ? (
                <div className="rw-edit">
                  <textarea
                    className="rw-textarea"
                    value={editing.text}
                    onChange={(e) => setEditing({ ...editing, text: e.target.value })}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) saveEdit();
                      if (e.key === "Escape") setEditing(null);
                    }}
                    rows={Math.min(10, Math.max(2, Math.ceil(editing.text.length / 40)))}
                    autoFocus
                  />
                  <div className="rw-edit-ops">
                    <span className="rw-hint">Ctrl+Enter 保存 · Esc 取消</span>
                    <button className="batch-btn ghost" onClick={() => setEditing(null)}>取消</button>
                    <button className="batch-btn primary" onClick={saveEdit}>保存并写入文档</button>
                  </div>
                </div>
              ) : isComposing ? (
                <div className="rw-edit">
                  <textarea
                    className="rw-textarea"
                    placeholder={composing.op === "append" ? "追加到本段末尾的内容…" : "插入的新段落内容…"}
                    value={composing.text}
                    onChange={(e) => setComposing({ ...composing, text: e.target.value })}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submitCompose();
                      if (e.key === "Escape") setComposing(null);
                    }}
                    rows={3}
                    autoFocus
                  />
                  <div className="rw-edit-ops">
                    <span className="rw-hint">Ctrl+Enter 保存 · Esc 取消</span>
                    <button className="batch-btn ghost" onClick={() => setComposing(null)}>取消</button>
                    <button className="batch-btn primary" onClick={submitCompose} disabled={applying}>
                      {composing.op === "append" ? "追加并写入文档" : "下插并写入文档"}
                    </button>
                  </div>
                </div>
              ) : (
                <p className={`rw-text ${p.empty ? "rw-text-empty" : ""}`}>
                  {p.empty ? "（空段落）" : p.text}
                  {!p.editable && !p.empty && (
                    <span className="rw-locked" title={p.reason === "field" ? "域代码段落（目录/题注等），跳过编辑" : "含图片/对象，跳过编辑"}>
                      <Icons.lock size={11} /> {p.reason === "field" ? "域" : "媒体"}
                    </span>
                  )}
                </p>
              )}
            </div>
          );
        })}
      </div>
      )}

      {/* AI 段落改写面板（两模式共用；无头部行，关闭为右上悬浮角标，压低占用） */}
      {ai && (
        <div className="rw-aipanel">
          <button className="rw-ai-close icon-btn" onClick={() => setAi(null)} aria-label="关闭">
            <Icons.x size={13} />
          </button>

          {/* pick：4 动作单行紧凑按钮（说明转 tooltip，压低面板占用） */}
          {ai.phase === "pick" && (
            <div className="rw-ai-cards">
              {AI_ACTIONS.map((a) => (
                <button key={a} className={`rw-ai-card kind-${a}`} title={AI_HINT[a]} onClick={() => void runAi(a)}>
                  <span className="rw-ai-card-mark" aria-hidden="true">{AI_ICON[a]}</span>
                  <span className="rw-ai-card-title">{AI_LABEL[a]}</span>
                </button>
              ))}
            </div>
          )}

          {/* loading */}
          {ai.phase === "loading" && (
            <div className="rw-ai-loading">
              <span className="spinner" aria-hidden="true" />
              <span>正在{ai.action ? AI_LABEL[ai.action] : ""}…</span>
            </div>
          )}

          {/* done：原文对照（折叠）+ 结果可微调 + 主操作 */}
          {ai.phase === "done" && (
            <>
              {/* 原文折叠：点击展开对照原文，避免改完忘了改哪段 */}
              <details className="rw-ai-orig">
                <summary>
                  <span className="rw-ai-section-tag">原文</span>
                  <span className="rw-hint">（点击展开对照）</span>
                </summary>
                <div className="rw-ai-orig-body">{ai.anchor}</div>
              </details>

              {ai.error && (
                <div className="rw-error">
                  {ai.error}
                  {ai.action && (
                    <button className="batch-btn ghost" onClick={() => void runAi(ai.action!)}>重试</button>
                  )}
                </div>
              )}

              <div className="rw-ai-section-head">
                <span className="rw-ai-section-tag kind-out">改写后</span>
                {ai.action && <span className="rw-hint">由 {AI_LABEL[ai.action]} 生成</span>}
              </div>
              <textarea
                className="rw-textarea rw-ai-edit"
                value={ai.editText ?? ""}
                onChange={(e) => setAi({ ...ai, editText: e.target.value })}
                rows={Math.min(8, Math.max(3, Math.ceil((ai.editText?.length ?? 0) / 50)))}
                placeholder="AI 改写结果将出现在此，可直接修改"
              />

              <div className="rw-ai-foot">
                <span className="rw-hint">采用后将直接写入文档（自动备份，可撤销）</span>
                <div className="rw-ai-foot-ops">
                  {ai.action && (
                    <button className="batch-btn ghost" onClick={() => void runAi(ai.action!)}>重试</button>
                  )}
                  <button
                    className="batch-btn primary rw-ai-adopt"
                    disabled={!ai.editText?.trim() || ai.editText === ai.anchor || applying}
                    onClick={() => void adoptAndApply()}
                  >
                    <Icons.check size={13} /> 采用并写入文档
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* 撤销（编辑即落盘，故只需撤销备份；改后↔改前可反复翻转）+ 保存导出（C4 增补） */}
      <div className="rw-undo-row">
        <button className="batch-btn ghost" disabled={restoring || loading || applying} onClick={() => void undo()}>
          <Icons.refresh size={12} /> {restoring ? "撤销中…" : "撤销上次应用"}
        </button>
        <span className="rw-hint">应用时自动备份原稿；撤销可反悔</span>
        <span className="rw-spacer" />
        {/* ★ P-04（2026-10-09）：桌面端改「原生另存为 → 后端写盘 → 明确反馈」。
            原先是原生 <a download>：在 Tauri/WebView2 里静默落盘到系统下载文件夹、界面零反馈
            ⇒ 实测用户连点 12 次产生 12 个副本。浏览器端由 saveDocxExport 内部回退为下载链接。 */}
        <button
          className="batch-btn ghost"
          disabled={savingDocx !== null}
          onClick={() => void saveAs("tracked")}
          title="另存为带修订痕迹的 .docx（Word 中可继续接受/拒绝修订）"
        >
          <Icons.download size={12} /> {savingDocx === "tracked" ? "保存中…" : "修订版 .docx"}
        </button>
        <button
          className="batch-btn ghost"
          disabled={savingDocx !== null}
          onClick={() => void saveAs("clean")}
          title="另存为接受全部修订后的干净 .docx（可直接投稿/提交）"
        >
          <Icons.download size={12} /> {savingDocx === "clean" ? "保存中…" : "干净版 .docx"}
        </button>
        {saveMsg && <span className="rw-hint">{saveMsg}</span>}
      </div>
    </div>
  );
}
