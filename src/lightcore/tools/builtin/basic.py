"""LightCore · 内置工具（docs/05 §12 内置工具，MVP 先落 3 个可跑占位）

1. retrieve   —— RAG 检索（safe）。必须传入 rag_pipeline（会话关联文档）；未关联时明确告知无法检索，不返回假数据。
2. calculator —— 安全表达式计算（safe）。
3. code_interpreter —— 沙箱代码执行（sandbox，NFR-07 默认关闭）。强制 Docker 隔离；无 Docker 时明确报错，
                  绝不本机直跑；测试可注入 MockSandbox 验证编排逻辑。

权限分级见 ToolSpec.permission。
"""

from __future__ import annotations

import ast
import logging
import operator
import time
from typing import Any

from ...rag.invoke import call_query
from ...rag.page_utils import normalize_page, validate_requested_page
from ..spec import Permission, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


# ---- `top_k` 硬上界与 clamp（B1，2026-09-28）----
#:
#: ### 为什么必须有
#: `top_k` 原先**既无 schema 上界、代码也不 clamp** ⇒ 模型只要传 `pdf_extract(query='abstract', top_k=50)`
#: 就能单次注入数十万字符。而"单次请求 prompt 超限"的判据在 **响应返回之后** 才执行
#: （`loop._run_stream_impl`），此时该次调用**已经发出并计费** ——
#: 也就是"**先把钱花出去，再发现撑爆**"，护栏在最需要它的时刻失效。
#:
#: ### 两个上限为什么不同量级（有意为之，不是疏漏）
#: - `retrieve`：注入前每段被压缩到 ≤200 字（`loop._RETRIEVE_MAX_TEXT`）
#:   ⇒ 20 段 ≈ 4k 字 ≈ 2.7k tokens，相对 30,000 的单次上限是安全水位。
#: - `pdf_extract(query=)`：**返回全文、不压缩**（单块常见 1~2k 字）
#:   ⇒ 12 块 ≈ 12~24k 字，是"仍够用、但不至于单次撞破上限"的水位。
#: 线上实测模型的实际取值是 6/8/10（`retrieve`）与 3~8（`pdf_extract`）⇒ 正常用法碰不到上限。
RETRIEVE_TOP_K_MAX = 20
PDF_EXTRACT_TOP_K_MAX = 12


def _merge_note(output: dict, note: str | None) -> dict:
    """把一条附加提示并进 `output["note"]`（**不覆盖**既有 note）。B1 用。"""
    if not note:
        return output
    base = (output.get("note") or "").strip()
    output["note"] = f"{note} {base}".strip()
    return output


# ---- N1（2026-10-02）：文档身份 ----
def _doc_label_of(chunk_id: str, doc_labels: dict | None) -> str | None:
    """从 chunk id（`doc-{id}-c{seq}`）解析 doc 号→ 「文档N」标签。

    ★ **解析失败/映射缺失一律返回 `None`**（调用方据此省略 `doc` 键）：
    给模型一个**错的**标签比不给标签更危险——它会据此把别篇的内容当成本篇。
    """
    if not doc_labels or not isinstance(chunk_id, str):
        return None
    parts = chunk_id.split("-")
    # 形如 ["doc", "6", "c0"]；doc 号段必须**纯数字**（防 "doc-x-c0" 之类脏数据）
    if len(parts) < 3 or parts[0] != "doc" or not parts[1].isdigit():
        return None
    label = doc_labels.get(int(parts[1]))
    return label if isinstance(label, str) and label else None


def _scope_note(doc_labels: dict | None) -> str | None:
    """本轮限定到**唯一一篇**时，给模型一句"只有这一篇"的声明（N1 的 N2 部分）。

    ⚠️ 只在**恰好一篇**时给：多篇限定时说"限定到 N 篇"反而让模型在 N 篇间摇摆
    （"我该以哪篇为主？"）；未限定时**绝不给**（否则与"可检索全部关联文档"矛盾）。
    """
    if not doc_labels or len(doc_labels) != 1:
        return None
    label = next(iter(doc_labels.values()))
    # ⚠️ **中性措辞**（2026-10-02 实测修正）：不要写"用户用 @文档一 指定"——
    #   前端弹窗与手打对模型**长得一样**，若正文写的 @ 标签与本note 不一致，
    #   实测模型会整段推理纠结"哪里冲突"（thinking 原文："这里有冲突…我决定折中…"）
    #   ⇒ 反而**更困惑**。这里只陈述"本轮只有这一篇"，标签作为**事实**给出。
    return (
        f"【本轮范围】本轮检索**只覆盖唯一一篇**文档：{label}；"
        f"下列片段全部来自它，回答时请**明确称它为「{label}」**，"
        "不要用「这篇论文/该文/所取到的内容」等模糊指代——"
        "用户无法据此确认你读的是哪一篇。"
    )


def clamp_top_k(raw: Any, *, maximum: int, default: int, tool: str) -> tuple[int, str | None]:
    """把模型给的 `top_k` 收敛到 `[1, maximum]`，返回 `(生效值, 给模型的提示或 None)`。

    ### 为什么 schema 的 `maximum` 不够，必须在代码里再 clamp 一次
    schema 只是"**对遵守 JSON schema 的模型**的第一道提示"，而它在这里**不是唯一入口**：

    1. `ToolExecutor.execute` 直接 `handler(**arguments)`，**不做任何参数校验**；
    2. 部分厂商/模型对 schema 的服从不稳定（本项目实测过 `tool_choice='required'` 被无视）；
    3. 本地替身（`FakeLLM` / 测试 / CLI）与后续新增的调用方可传任意值。

    ⇒ 三层里只有**代码 clamp 是唯一对所有入口都有效**的那一层。

    ### 约定
    - 非整数 / 缺省 → 回退 `default`（与既有 `memory_search` 的 K7 口径一致：**不抛异常**，
      否则一次脏参数会让整个工具调用失败）；
    - `< 1` → 取 1；
    - `> maximum` → 取 `maximum`，并返回一条**给模型看的**提示 —— 必须让它知道"没取全"，
      否则它会以为自己已经拿到整章内容而给出不完整结论（那是**静默降准**）。
    """
    try:
        k = default if raw is None else int(raw)
    except (TypeError, ValueError):
        return default, f"⚠️ top_k 取值「{raw}」不是整数，已按默认值 {default} 处理。"
    if k < 1:
        return 1, f"⚠️ top_k 最小为 1（收到 {k}），已按 1 处理。"
    if k > maximum:
        logger.warning(
            "%s: top_k=%d 超过单次上限 %d，已 clamp（B1：防「先花钱才发现撑爆」）",
            tool, k, maximum,
        )
        return maximum, (
            f"⚠️ top_k={k} 超过单次上限 {maximum}，**已限制为 {maximum} 条**（防止单次注入撑爆上下文）。"
            f"若确实需要更多内容，请分多次取：改用 pdf_extract(page=N) 逐页取，"
            f"或换不同 query 主题词再检索一次。"
        )
    return k, None


# ---- 1. retrieve：RAG 检索（必须传入 rag_pipeline；无管线时明确拒绝，不返回任何假数据）----
def _make_retrieve(pipeline=None, doc_labels: dict | None = None):
    # 2026-09-10 N4 修复：handler 改为 **async**。原因见下方 query 调用处注释——
    # MultiQueryRAGPipeline.query 是协程，同步 handler 无法 await 它，导致检索工具必然失败。
    #
    # ★★ N1（2026-10-02）`doc_labels`：**doc_id → 「文档N」标签**映射（由调用方按
    # `mention_order` 口径构造）。作用：让检索结果**自带身份**。
    #
    # ## 为什么必须加（实测事故）
    # 注入给模型的结果原本只有 `{id: "doc-6-c0", page, text, score}` ——
    # **没有标题、没有"文档四"标签** ⇒ 模型不知道"这段出自哪篇" ⇒ 只能靠正文**猜**。
    # 实测铁证（四轮回答内容全对、但措辞暴露"在猜"）：
    #   「**就所取到的内容看，这篇论文**…」「**本轮检索命中的**…」「**该文**提及…」。
    # 用户视角就是"模型还是无法确定指定文档"。
    # ⇒ 范围限缩（`restrict_doc_ids`）只解决了"搜哪篇"，**没解决"模型知不知道是那篇"**。
    #
    # ⚠️ 刻意**只加标签、不加文件名**：文件名可能很长（每条 × top_k 会明显推高 token）。
    #   标签是**短且稳定**的（"文档四"），且与用户在面板上看到的一致。
    async def _retrieve(query: str, top_k: int = 6) -> ToolResult:
        # B1（2026-09-28）：**先 clamp 再检索** —— 必须在 `pipeline.query` 之前，
        # 否则"先把钱花出去（超量检索+超量注入）、再发现撑爆"（见 `clamp_top_k` 说明）。
        top_k, _clamp_note = clamp_top_k(
            top_k, maximum=RETRIEVE_TOP_K_MAX, default=6, tool="retrieve")
        if pipeline is None:
            # 关联文档为空（doc_ids=[]）：返回中性空结果，不返回任何假数据，
            # 也不主动要求"关联文档"（该提示仅由协议级拦截对【显式指代自有文档】的问题给出）。
            # 无文档时由 _grounding_block(has_docs=False) 指示模型：基于通用知识作答、不引用具体论文。
            return ToolResult(
                ok=True,
                output={
                    "query": query,
                    "results": [],
                    "source": "none",
                    "note": "（当前会话未关联任何文档，retrieve 未检索；将基于大模型通用知识作答，不引用具体论文。）",
                },
            )
        try:
            # N4（2026-09-10）+ N9（2026-09-11）两条判断已**收敛为唯一实现**
            # `lightcore.rag.invoke.call_query`：异步管线 await、同步 GPU 管线下放线程池、
            # 返回值仍 awaitable 再兜底。历史教训（都真实发生过）：
            #   ① 漏判异步 ⇒ 拿到协程对象 ⇒ 迭代抛 `TypeError: 'coroutine' object is not iterable`
            #      ⇒ 工具连续失败 3 次 ⇒ agent 提前中止（真实库 09-10 单日 21 次）；
            #   ② 忘了 to_thread ⇒ 同步管线上跑在事件循环 ⇒ 阻塞 262.82s
            #      ⇒「停止」无效、API 无响应、GPU 满载（N9 实测）。
            _t0 = time.perf_counter()
            results = await call_query(pipeline, query, top_k=top_k)
            # 2026-09-11：**检索耗时日志**（定位"首次对话非常慢"——此前检索路径完全无日志，
            # 52s 空档无从判断是「管线构建/等锁」还是「检索本身」）。
            # 2026-09-28：改为记录**完整命中 id 列表**（原先只记"第一次命中"）——
            # 那是算「同一份内容被反复取回多少次」的唯一依据：判重需要**全量** id，
            # 只看第一条既算不出重合度、也无法区分"不同 query 命中同一批块"这种情况。
            # ⚠️ 只为观测，不参与任何判据（不要拿日志做业务分支）。
            logger.info(
                "retrieve: %d 命中，耗时 %.2fs，命中 id=%s",
                len(results), time.perf_counter() - _t0,
                ",".join(str(getattr(r, "id", "-")) for r in results) or "无",
            )
        except Exception as e:  # noqa: BLE001  仅识别 RetrievalUnavailable，其余照常上抛
            # F7（2026-09-07）：严格区分「检索故障」与「检索到但无命中」。
            # 延迟导入避免模块级引入 RAG 重依赖（embedder / 索引等）。
            from ...rag.pipeline import RetrievalUnavailable

            if isinstance(e, RetrievalUnavailable):
                # 仍带 source="rag" 供路由层识别；ok=False 使其计入工具失败，
                # 不被模型误当成「文档确实没有相关内容」。
                return ToolResult(
                    ok=False,
                    output={
                        "query": query,
                        "results": [],
                        "source": "rag",
                        "error": "retrieval_unavailable",  # 与路由层 notice kind 配套的标记
                        "note": (
                            "【检索服务暂时不可用】文档已关联，但检索全部失败"
                            "（可能是索引损坏或嵌入/重排服务异常）。"
                            "这不是『文档中没有相关内容』：请勿回答『知识库没有相关内容』，"
                            "也不要建议用户重新上传文档；应如实告知检索暂时不可用，"
                            "并建议稍后重试或检查服务状态。"
                        ),
                        "detail": str(e),
                    },
                )
            raise
        return ToolResult(ok=True, output=_merge_note(_with_partial_failure_note({
            "query": query,
            # 页码归一化：仅 ≥1 的有效页码透传，其余（0/None/非法）→ None，
            # 避免 LLM 看到「第 0 页」并当作真实页码引用（见 lightcore/rag/page_utils）。
            # ★★ N1（2026-10-02）：每条补 `doc`（「文档N」标签）—— 让结果**自带身份**，
            # 模型不必靠正文猜篇名。`doc` 由 chunk id 的 `doc-{id}-c{seq}` 前缀解析，
            # 再经调用方传入的 `doc_labels` 映射；**映射缺失/解析失败一律省略该键**
            #（宁可少个字段，也不要给模型一个错的标签 —— 错标签比没标签更危险）。
            "results": [
                {
                    "id": r.id,
                    "page": normalize_page(r.page),
                    "text": r.text,
                    "score": round(r.score, 4),
                    **({"doc": _doc_label_of(r.id, doc_labels)} if _doc_label_of(r.id, doc_labels) else {}),
                }
                for r in results
            ],
            "source": "rag",
        }, pipeline), _scope_note(doc_labels) or _clamp_note))

    return _retrieve


def _partial_failure_of(pipeline) -> dict | None:
    """读取管线**最近一次**检索的部分失败信息；无/不可用 → `None`。

    **为什么需要**（2026-09-23）：全失败会抛 `RetrievalUnavailable`（F7，有完整信号），
    但**部分失败只 warning 就静默继续** —— 调用方拿到「较少的结果」，
    与「文档里确实没有相关内容」在数据上无法区分。用户于是以为检索了全部文档，
    而回答实际只基于其中一部分，模型也不会声明这一点。与 F7 要修的错同一类。

    ⚠️ **穿透包装**：`features.multi_query` 开启时管线是
    `MultiQueryRAGPipeline(CompositeRAGPipeline(...))`，失败信息挂在**内层** composite 上。
    这里沿 `_base` 向下找（限深 4，避免异常结构导致死循环）。
    """
    node = pipeline
    for _ in range(4):
        if node is None:
            return None
        info = getattr(node, "last_partial_failure", None)
        if info:
            return info
        node = getattr(node, "_base", None)
    return None


def _with_partial_failure_note(output: dict, pipeline) -> dict:
    """若最近一次检索**部分失败**，在工具输出里如实标记 + 指示模型声明。

    两处刻意的设计：
    - **不改 `ok`**：部分结果仍然可用（改 ok=False 会把它计入工具失败预算，
      可能触发 `max_tool_failures` 提前中止，把「覆盖不全」升级成「整轮失败」）。
    - **必须给 note**：只加结构化字段，模型未必会读；note 是面向模型的显式指令，
      与 F7 的 `retrieval_unavailable` 同款做法。
    """
    info = _partial_failure_of(pipeline)
    if not info:
        return output
    ids = info.get("failed_doc_ids") or []
    cnt, total = int(info.get("failed_count") or 0), int(info.get("total") or 0)
    if cnt == 0 or total == 0:
        return output
    out = dict(output)
    out["partial_failure"] = {
        "failed_doc_ids": ids,
        "failed_count": cnt,
        "total": total,
        "searched_count": total - cnt,
    }
    which = f"（文档 id：{', '.join(str(i) for i in ids)}）" if ids else ""
    warn = (
        f"【本次检索不完整】{total} 篇关联文档中有 {cnt} 篇检索失败{which}，"
        f"以上结果仅来自其余 {total - cnt} 篇。"
        "若回答涉及未检索到的文档内容，请如实说明该部分未能检索、"
        "必要时建议用户重试；不要据现有结果推断其内容，也不要谎称已检索全部文档。"
    )
    # **追加**而非覆盖：调用方可能已有更具体的指引（如 pdf_extract 的「未检索到相关内容」
    # 兜底指导），覆盖会把它弄丢。两条信息都要保留。
    prev = out.get("note")
    out["note"] = f"{prev} {warn}" if prev else warn
    return out


# ---- 1b. pdf_extract：按页/按查询提取原文（safe，依赖已索引 chunk）----
def _all_chunks(pipeline):
    """从 retrieve 管线取出全部已索引 chunk（兼容 CompositeRAGPipeline 与单 RAGPipeline）。"""
    if pipeline is None:
        return None
    if hasattr(pipeline, "get_all_chunks"):
        return pipeline.get_all_chunks()
    if hasattr(pipeline, "get_chunks"):
        return pipeline.get_chunks()
    return None


def _make_pdf_extract(pipeline=None):
    # N4 修复（2026-09-10）：同 `_make_retrieve`——Handler 改 async 以支持协程形式的 query。
    async def _pdf_extract(
        page: int | None = None,
        query: str | None = None,
        top_k: int = 3,
        doc_id: int | None = None,
    ) -> ToolResult:
        # ★ 2026-10-03：`doc_id` 用于**多文档会话**里精确指定取哪一篇的原文。
        #   背景：`page` 模式原先是 `sel = [c for c in chunks if c.page == norm]`
        #   ⇒ **不区分文档** ⇒ 实测在绑定 8 篇的会话里，`page=9` 会把 **6 篇**的第 9 页
        #   拼在一起（22~38 chunks），再撞 4000 字上限被截断 ⇒ **静默降准**。
        chunks = _all_chunks(pipeline)
        if chunks is None:
            return ToolResult(ok=True, output={"source": "none",
                "note": "（当前会话未关联任何文档，pdf_extract 未执行。）"})
        available_pages = sorted({getattr(c, "page", None) for c in chunks if getattr(c, "page", None)})
        if page is not None:
            # 边界校验：页码必须 ≥1 且在文档可用页码范围内，非法/超范围给出明确提示而非静默空结果
            norm, note = validate_requested_page(page, set(available_pages) or None)
            if norm is None:
                return ToolResult(ok=True, output={"page": page, "text": "", "num_chunks": 0,
                                                  "source": "pdf", "note": note})
            if note:
                return ToolResult(ok=True, output={"page": norm, "text": "", "num_chunks": 0,
                                                  "source": "pdf", "note": note})
            # ★ 2026-10-03：先按 doc_id 过滤（多文档会话里"取哪一篇"必须能指定）
            pool = chunks
            if doc_id is not None:
                pool = [c for c in chunks if _doc_id_of(c) == doc_id]
                if not pool:
                    return ToolResult(ok=True, output={"page": page, "text": "", "num_chunks": 0,
                                                      "source": "pdf",
                                                      "note": f"当前会话未关联文档 doc_id={doc_id}。"})
            sel = [c for c in pool if getattr(c, "page", None) == norm]
            # ⚠️ 仍未指定 doc_id 且命中多篇 ⇒ **如实告知**（不给"混拼"而不说明）。
            #   判据用**实际命中的文档数**（不是会话绑定数）—— 只有真拼上了才提示。
            _hit_docs = sorted({_doc_id_of(c) for c in sel if _doc_id_of(c) is not None})
            _note = None
            if doc_id is None and len(_hit_docs) > 1:
                _note = (f"⚠️ 本次结果拼接了 {len(_hit_docs)} 篇文档（doc_id={_hit_docs}）的第 {norm} 页"
                         f"，内容是**多篇混排**。如需单篇原文，请带 doc_id 参数重新调用。")
            text = "\n".join(getattr(c, "text", "") for c in sel)
            # 2026-09-28：命中 id 日志（与 `retrieve` 同口径）——
            # `pdf_extract` 此前**完全无命中日志**，而它是不压缩注入的最大来源，
            # 不记 id 就无法回答「同一页/同一批块被反复取回多少次」。
            logger.info(
                "pdf_extract(page): page=%d 命中 %d 块，命中 id=%s",
                norm, len(sel), ",".join(str(getattr(c, "id", "-")) for c in sel) or "无",
            )
            return ToolResult(ok=True, output={"page": norm, "text": text,
                                              "num_chunks": len(sel), "source": "pdf",
                                              **({"note": _note} if _note else {})})
        results: list[dict] = []
        _clamp_note: str | None = None
        if query:
            # B1（2026-09-28）：**先 clamp 再检索**。本工具返回**未压缩全文**，
            # 是单次注入量的最大来源 ⇒ 上界比 `retrieve` 低（见 `PDF_EXTRACT_TOP_K_MAX`）。
            # ⚠️ 放在 query 分支内的理由（**不是**"否则会污染 page 模式的提示" —— 曾这么写过，
            # 经反向自检证明是错的：提示只在 query 分支的两个 return 上合并，提前 clamp 也不会
            # 影响 page 模式的输出）。真实理由有两条：
            #   ① `top_k` 在语义上**只属于 query 模式**，page 模式根本不用它；
            #   ② 避免为不使用的参数做无意义的 clamp 与告警日志。
            top_k, _clamp_note = clamp_top_k(
                top_k, maximum=PDF_EXTRACT_TOP_K_MAX, default=3, tool="pdf_extract")
            if hasattr(pipeline, "query"):
                try:
                    # 形态适配（同步/异步）+ 线程池下放：统一走 `call_query`
                    # （N4/N9 的唯一实现，见其模块 docstring）。
                    _res = await call_query(pipeline, query, top_k=top_k)
                    for r in _res:
                        results.append({"id": r.id, "page": normalize_page(r.page), "text": r.text})
                except Exception as e:  # noqa: BLE001  仅识别 RetrievalUnavailable，其余照常上抛
                    # 2026-09-09 修复 I4：与同文件 `_retrieve` 口径一致——**检索故障 ≠ 文档无内容**。
                    # 直接抛会让整个工具调用失败；若吞掉则会落到下面的「未检索到相关内容」
                    # 空结果兜底，把系统故障误导成「文档里没有」，诱导用户重复上传。
                    # 延迟导入（同 `_retrieve`）：避免模块级引入 RAG 重依赖（embedder / 索引等）。
                    from ...rag.pipeline import RetrievalUnavailable

                    if not isinstance(e, RetrievalUnavailable):
                        raise
                    return ToolResult(
                        ok=False,
                        output={
                            "query": query,
                            "results": [],
                            "source": "pdf",  # 供路由层识别：确实走的是文档检索
                            "error": "retrieval_unavailable",  # 与路由层 notice kind 配套的标记
                            "note": (
                                "【检索服务暂时不可用】文档已解析入库，但检索全部失败"
                                "（可能是索引损坏或嵌入/重排服务异常）。"
                                "这不是『文档中没有相关内容』：请勿回答『文档中未提及』，"
                                "也不要建议用户重新上传；应如实告知检索暂时不可用并建议稍后重试。"
                            ),
                            "detail": str(e),
                        },
                    )
            else:
                for c in chunks[:top_k]:
                    results.append({"id": getattr(c, "id", ""), "page": normalize_page(getattr(c, "page", None)),
                                    "text": getattr(c, "text", "")})
        # 空结果兜底指引（2026-08-25 修复）：模型误判『无内容』会叫用户重传 PDF；
        # 这里改为给出可用页码/章节提示，引导模型自行取数，而非向用户索要文件。
        if not results:
            note = (
                "未检索到相关内容。该文档已解析入库，可直接取数，请勿请用户重新上传/提供 PDF。"
                f"可用页码示例：{available_pages[:10]}{'…' if len(available_pages) > 10 else ''}。"
                "建议：① pdf_extract(query='abstract'/'experiment'/'method' 等章节主题词) 取该章节全文；"
                "② pdf_extract(page=N) 取指定页全文；③ 或 retrieve(query='摘要/实验…'主题词) 再检索。"
            )
            return ToolResult(ok=True, output=_merge_note(_with_partial_failure_note(
                {"query": query, "results": [], "source": "pdf", "note": note}, pipeline), _clamp_note))
        # 与 `_retrieve` 同口径（2026-09-23）：部分失败必须如实标记，否则用户会
        # 以为「取遍了全部文档」，而结果实际只来自其中一部分。
        # 2026-09-28：命中 id 日志 —— `query=` 模式本质是"一次检索但**不压缩**返回全文"，
        # 它常与同轮的 `retrieve` 目标重叠；记录完整 id 才能量化"同一批块被取回几次"。
        logger.info(
            "pdf_extract(query): top_k=%d 命中 %d 块，命中 id=%s",
            top_k, len(results), ",".join(str(r.get("id", "-")) for r in results) or "无",
        )
        return ToolResult(ok=True, output=_merge_note(_with_partial_failure_note(
            {"query": query, "results": results, "source": "pdf"}, pipeline), _clamp_note))

    return _pdf_extract


def _doc_id_of(chunk: object) -> int | None:
    """从 chunk 的逻辑 id（形如 `doc-16-c7`）取出数字文档号。

    ★ **容错优先**：解析不出就返回 `None`，**绝不抛异常** ——
    这是分流/观测辅助，出错不该把工具调用搞崩（与项目里"观测代码必须 getattr 兜底"同源）。
    """
    rid = str(getattr(chunk, "id", "") or "")
    parts = rid.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


# ---- 1c. citation_lookup：论文引用元数据核实（safe，防幻觉）----
def _make_citation_lookup(pipeline=None):
    def _citation_lookup() -> ToolResult:
        chunks = _all_chunks(pipeline)
        if chunks is None:
            return ToolResult(ok=True, output={"docs": [], "source": "none",
                "note": "当前会话未关联任何文档。"})
        docs: dict[str, dict] = {}
        for c in chunks:
            cid = getattr(c, "id", "")
            doc_id = cid.split("-c")[0] if "-c" in cid else cid
            d = docs.setdefault(doc_id, {"doc_id": doc_id, "count": 0, "pages": set()})
            d["count"] += 1
            pg = getattr(c, "page", None)
            if pg is not None:
                d["pages"].add(pg)
        out = [{
            "doc_id": v["doc_id"],
            "num_chunks": v["count"],
            "page_min": (sorted(v["pages"])[0] if v["pages"] else None),
            "page_max": (sorted(v["pages"])[-1] if v["pages"] else None),
        } for v in docs.values()]
        return ToolResult(ok=True, output={"docs": out, "source": "pdf"})

    return _citation_lookup


# ---- 2. calculator：安全算术 ----
_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# 幂运算规模上限（2026-09-09 修复 I7）：`9**9**9` 之类会被 operator.pow 真的去算，
# 长时间占满 CPU 且结果大到无意义。先按量级拒绝，再执行。
_POW_MAX_BASE = 1e6
_POW_MAX_EXP = 64


def _calculator(expression: str) -> ToolResult:
    # 2026-09-09 修复 I7：模型可能传非字符串（如 expression: 123）。
    # ast.parse 此时抛的是 **TypeError**，而原实现只捕 SyntaxError → 异常直冲工具调用，
    # 表现为工具整体失败而非一句可读的提示。
    if not isinstance(expression, str):
        return ToolResult(ok=False, output=None,
                          error=f"表达式必须是字符串，实际收到 {type(expression).__name__}")
    try:
        node = ast.parse(expression, mode="eval").body
    except (SyntaxError, TypeError, ValueError) as e:
        return ToolResult(ok=False, output=None, error=f"表达式语法错误: {e}")
    if not isinstance(node, ast.BinOp | ast.UnaryOp | ast.Constant | ast.Expression):
        return ToolResult(ok=False, output=None, error="仅支持基础算术表达式")
    try:
        value = _eval_node(node)
    except Exception as e:  # noqa: BLE001
        return ToolResult(ok=False, output=None, error=f"计算失败: {e}")
    return ToolResult(ok=True, output={"expression": expression, "result": value})


def _eval_node(node: Any) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return float(node.value)
        raise ValueError("仅支持数值常量")
    # 幂运算先做规模校验（见 _POW_MAX_* 说明），避免 `9**9**9` 长时间占满 CPU
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base = _eval_node(node.left)
        exp = _eval_node(node.right)
        if abs(base) > _POW_MAX_BASE or exp > _POW_MAX_EXP:
            raise ValueError(
                f"幂运算规模过大：底数绝对值需 ≤{_POW_MAX_BASE:g}、指数需 ≤{_POW_MAX_EXP:g}"
            )
        return operator.pow(base, exp)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        val = _eval_node(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    raise ValueError(f"不支持的节点: {type(node).__name__}")


# ---- 3. code_interpreter：沙箱代码执行（NFR-07 强制 Docker 隔离，默认关闭）----
import subprocess  # noqa: E402  (置于工具定义前，仅本模块使用)


class SandboxBackend:
    """沙箱后端协议：run(code, lang) -> {ok, stdout, stderr, error}。"""

    def run(self, code: str, lang: str = "python") -> dict:  # pragma: no cover - 接口
        raise NotImplementedError


class DockerSandbox(SandboxBackend):
    """强制 Docker 隔离执行（NFR-07）。

    参数：--network=none --memory=512m --cpus=1 --pids-limit=64，30s 超时，只读根、无宿主机挂载。
    无 Docker 守护进程时明确报错，绝不回退到本机直跑（避免沙箱逃逸）。
    """

    def __init__(self, image: str = "python:3.12-slim") -> None:
        self._image = image

    def run(self, code: str, lang: str = "python") -> dict:
        if lang not in ("python", "py"):
            return {"ok": False, "stdout": "", "stderr": "",
                    "error": f"暂仅支持 python，收到 lang={lang}"}
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--network=none", "--memory=512m",
                 "--cpus=1", "--pids-limit=64", "-i", self._image, "python3", "-"],
                input=code.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError:
            return {"ok": False, "stdout": "", "stderr": "",
                    "error": "本地未检测到 Docker 守护进程（NFR 要求 Docker 隔离），无法执行沙箱代码。"
                             "请在开启『代码解释器』前安装并启动 Docker，或关闭该能力。"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "", "error": "沙箱执行超时（>30s）被中止。"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "stdout": "", "stderr": "", "error": f"沙箱启动失败: {type(e).__name__}: {e}"}
        ok = proc.returncode == 0
        return {
            "ok": ok,
            "stdout": proc.stdout.decode("utf-8", "replace"),
            "stderr": proc.stderr.decode("utf-8", "replace"),
            "error": None if ok else proc.stderr.decode("utf-8", "replace") or f"退出码 {proc.returncode}",
        }


class MockSandbox(SandboxBackend):
    """测试/离线用 Mock 沙箱：返回预设 stdout，不真正执行（仅用于单测沙箱编排逻辑）。"""

    def __init__(self, stdout: str = "mock-output") -> None:
        self._stdout = stdout

    def run(self, code: str, lang: str = "python") -> dict:
        return {"ok": True, "stdout": self._stdout, "stderr": "", "error": None}


def _make_code_interpreter(sandbox: SandboxBackend | None = None):
    def _code_interpreter(code: str, lang: str = "python") -> ToolResult:
        # 防御：未注入沙箱后端则拒绝（不应发生，权限门控已先于此处拦截）
        if sandbox is None:
            return ToolResult(ok=False, output=None,
                              error="沙箱后端未配置，代码执行被拒绝（NFR-07 要求 Docker 隔离）。")
        res = sandbox.run(code, lang)
        if not res["ok"]:
            return ToolResult(ok=False, output=None, error=res["error"])
        return ToolResult(ok=True, output={
            "lang": lang,
            "stdout": res["stdout"],
            "stderr": res.get("stderr", ""),
            "sandbox": type(sandbox).__name__,
        })

    return _code_interpreter


# ---- 注册入口 ----
def register_builtin_tools(
    registry, rag_pipeline=None, sandbox=None, doc_labels: dict | None = None,
) -> None:
    """注册内置工具。

    ★ N1（2026-10-02）：doc_labels = {doc_id: 「文档N」标签}。
    仅用于让 retrieve 的结果**自带身份**（见 _make_retrieve 的说明）。
    口径由调用方按 mention_order（id 降序）构造 ⇒ 与面板标签、@文档N 解析同源。
    传 None（默认）⇒ **行为与改动前完全一致**（不带 doc 键、不给范围 note）。
    """
    sandbox = sandbox or DockerSandbox()  # 默认走 Docker 隔离；无 Docker 时执行明确报错
    registry.register(ToolSpec(
        name="retrieve",
        description="从已上传论文的向量库中检索与 query 最相关的段落，返回带 id/页码的片段用于引用溯源。"
                    "**找具体事实/数值/结论（谁做了什么、准确率多少、是否成立）一律用本工具**；"
                    "它返回的是**已压缩**片段（每段截断至 200 字），单次注入量小。"
                    "注意：论文已在系统中解析入库，直接用本工具取数即可；回复中切勿请用户『重新上传/提供 PDF』（文档早已可用）。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "检索查询：写**主题词**（如 'abstract' / '摘要' / '实验设置'），"
                                         "不要写『翻译摘要』这类任务短语（任务词会稀释相关性、召回无关片段）"},
                "top_k": {"type": "integer",
                          "description": "返回条数（默认 6，翻译/总结整段时建议调大以覆盖完整章节）；"
                                         f"单次上限 {RETRIEVE_TOP_K_MAX}，超过会被自动限制为上限并告知",
                          "default": 6, "minimum": 1, "maximum": RETRIEVE_TOP_K_MAX},
            },
            "required": ["query"],
        },
        permission=Permission.SAFE,
        handler=_make_retrieve(rag_pipeline, doc_labels=doc_labels),
    ))
    registry.register(ToolSpec(
        name="pdf_extract",
        description="精确取原文：① pdf_extract(page=N) 取第 N 页**全文**；"
                    "② pdf_extract(query='abstract') 取某个**章节**的全文（query 只能是章节主题词）。"
                    "⚠️ 本工具返回**未压缩**全文，单次注入量约为 retrieve 的 4 倍 —— "
                    "**找具体事实/数值请用 retrieve**，只有确实需要整页/整章原文时才用本工具；"
                    "同一页/同一章在本轮取过一次即可，不要重复取（重复内容不会二次注入）。"
                    "注意：论文已在系统中解析入库、原文件已留存，直接用本工具取数（含图页面也读留存原文件）即可；"
                    "切勿在回复中请用户『重新上传 PDF 原文件 / 提供 PDF』——文档早已可用，所需内容本工具可直接给出。",
        parameters={
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "指定页码，提取该页全部原文（与 query 二选一）"},
                "query": {"type": "string",
                          "description": "**章节主题词**（如 'abstract' / 'experiment' / 'method' / 'conclusion'），"
                                         "用于一次取回该章节全文；**不要**用它做事实型提问"
                                         "（检索算法与 retrieve 相同，但本工具返回全文、不压缩）。与 page 二选一"},
                # ★ 2026-10-03：多文档会话里"取哪一篇"必须能指定 ——
                #   原先 page 模式跨文档拼接（实测 8 篇会话里 page=9 拼了 6 篇，22~38 chunks）。
                #   ⚠️ schema 与实现**必须同源**：ToolExecutor 是 `handler(**arguments)`，
                #   **不会**套用 schema 的 default；这里只声明类型与语义。
                "doc_id": {"type": "integer",
                           "description": "**多文档会话必填**：指定取哪一篇文档的原文（doc_id 即『文档N』对应的数字 id）。"
                                          "不填则当会话仅关联一篇；关联多篇且未指定时，返回值会明确标注『多篇混排』。"},
                # ⚠️ 默认值必须与实现同源：`_make_pdf_extract` 签名是 `top_k: int = 3`。
                # 此前 schema 写 6 / 实参缺省走 3 —— 属**文档撒谎**：`ToolExecutor.execute`
                # 是直接 `handler(**arguments)`，**不会**套用 schema 的 default。
                # 2026-09-28 对齐到实现的真实默认值（**零运行时变更**）。
                # B1（2026-09-28）：补 `minimum`/`maximum` —— 本工具返回**未压缩全文**，
                # 上界刻意低于 retrieve（它每段压缩到 200 字）。schema 只是第一道提示，
                # 代码里另有一次 clamp（`clamp_top_k` 是唯一对所有入口有效的那层）。
                "top_k": {"type": "integer",
                          "description": f"query 模式返回条数（缺省 3；翻译/总结整段时显式调大以覆盖完整章节）；"
                                         f"单次上限 {PDF_EXTRACT_TOP_K_MAX}，超过会被自动限制为上限并告知",
                          "default": 3, "minimum": 1, "maximum": PDF_EXTRACT_TOP_K_MAX},
            },
            "required": [],
        },
        permission=Permission.SAFE,
        handler=_make_pdf_extract(rag_pipeline),
    ))
    registry.register(ToolSpec(
        name="citation_lookup",
        description=(
            "返回当前关联文档的引用元数据（文档标识、段落数、页码范围），用于核实引用、杜绝编造 DOI/页码。"
            "⚠️ 本工具**无参数**，且结果在整个会话内**恒定**（只取决于已关联的文档，不随时间变化）"
            "—— 整个会话**只需调用一次**；后续需要这些信息时，请直接使用你已获得的结果，"
            "**不要重复调用**。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission=Permission.SAFE,
        handler=_make_citation_lookup(rag_pipeline),
    ))
    registry.register(ToolSpec(
        name="calculator",
        description="安全地计算一个基础算术表达式（仅 + - * / ** % 与正负号）。",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "算术表达式，如 (1+2)*3"}},
            "required": ["expression"],
        },
        permission=Permission.SAFE,
        handler=_calculator,
    ))
    registry.register(ToolSpec(
        name="code_interpreter",
        description="在隔离沙箱中执行 Python 代码片段（数学验证、复现实验）。NFR-07：强制 Docker 隔离，"
                    "默认关闭，需显式开启『代码解释器』能力；无 Docker 时明确报错，不会在本机直跑。",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "待执行代码"},
                "lang": {"type": "string", "description": "语言（暂仅 python）", "default": "python"},
            },
            "required": ["code"],
        },
        permission=Permission.SANDBOX,
        handler=_make_code_interpreter(sandbox),
    ))
