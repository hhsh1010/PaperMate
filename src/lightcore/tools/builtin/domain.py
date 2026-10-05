"""LightCore · Phase B 领域工具（S3 跨会话记忆 + S4 一键总结，docs/统一方案 阶段B）

这些工具涉及「调用大模型 API」或长期记忆存储，受后端 feature 开关
（runtime_settings.features）控制：仅在对应功能开启时由 ChatService.build_agent
注册（见 chat_service）。这样即便关闭某个功能，工具也完全不暴露在 Agent 的函数列表中。

- memory_search / memory_write ：S3 跨会话记忆（features.memory）
- summarize / related_work    ：S4 一键总结（features.summary）

工具本身均为 Permission.SAFE（本地/内部服务调用，无外部副作用），由 ToolExecutor 默认可执行。
"""

from __future__ import annotations

import asyncio
import re

from ...llm.fake import FakeLLM
from ...planning.planner import Planner
from ...rag.invoke import call_query
from ..spec import Permission, ToolResult, ToolSpec


def _is_sentence_end(text: str, idx: int) -> bool:
    """判断 text[idx] 是否为真正句末标点（排除小数点 0.001）。"""
    ch = text[idx]
    if ch in "。！？!?；;\n":
        return True
    if ch == ".":
        prev_digit = idx > 0 and text[idx - 1].isdigit()
        next_digit = idx + 1 < len(text) and text[idx + 1].isdigit()
        return not (prev_digit and next_digit)
    return False


def _truncate_at_sentence(text: str, max_len: int = 200) -> str:
    """在句末标点处截断，避免硬字符截断产生半截证据。"""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    for i in range(max_len, -1, -1):
        if _is_sentence_end(text, i):
            return text[: i + 1].strip()
    # 回退到最近空白
    cut = text.rfind(" ", max_len - 40, max_len + 1)
    return text[: cut if cut != -1 else max_len].strip() + "…"


# ---------------- S9：复杂规划（Plan-then-Execute）----------------
def _rule_decompose(goal: str, max_steps: int) -> list[dict]:
    """fake 模型（无 key）下的规则分解：按中文连词/标点切分为 ≤max_steps 子任务。

    真实模型走 Planner（LLM 驱动），见 run_planning；此处保证无 key 时也能产出结构化计划，
    与 multi_query 在 fake 下回退单查询的「优雅降级」策略一致。
    """
    seps = r"[，,。；;、\n]+|然后|并且|以及|还有|加上|先|再|最后|接着|之后|首先|其次|第一步|第二步|第三步|第四步|第五步"
    parts = re.split(seps, goal)
    steps: list[dict] = []
    for raw in parts:
        raw = raw.strip("0123456789.、 ").strip()
        if not raw:
            continue
        steps.append({"index": len(steps) + 1, "description": raw, "depends_on": []})
        if len(steps) >= max_steps:
            break
    if not steps:
        steps = [{"index": 1, "description": goal, "depends_on": []}]
    return steps


def _resolve_order(steps: list[dict]) -> tuple[list[dict], list[str]]:
    """按 `depends_on` 做**拓扑排序**，返回 (有序步骤, 警告列表)。

    **为什么需要**（C6，2026-09-21）：`depends_on` 此前只是**展示信息** ——
    生成时被解析出来、原样塞进结果里，执行循环却**无条件按原顺序跑**。
    于是 LLM 说"步骤 3 依赖步骤 5"时，步骤 3 会在步骤 5 之前执行，
    依赖形同虚设；失败也不会阻断下游。

    **三个防御**（LLM 生成的依赖图不可信）：
    - **无效引用**（依赖不存在的 index，或依赖自己）→ 丢弃该边并记警告；
    - **成环** → **降级为原顺序**并记警告（宁可顺序执行，也不要因为环而卡死）；
    - 依赖在前的正常情况 → 严格按拓扑序。

    ⚠️ 排序只保证"依赖先于被依赖者"，**不改变原有相对顺序**（同层保持原序，输出稳定）。
    """
    warnings: list[str] = []
    by_index: dict[int, dict] = {}
    for i, s in enumerate(steps):
        idx = s.get("index")
        if not isinstance(idx, int):
            warnings.append(f"步骤 {i + 1} 的 index 非法（{idx!r}），按位置编号")
            idx = i + 1
            s["index"] = idx
        by_index[idx] = s

    # ① 清洗依赖边：去掉无效引用与自依赖
    clean: dict[int, list[int]] = {}
    for idx, s in by_index.items():
        deps: list[int] = []
        for d in s.get("depends_on") or []:
            if d == idx:
                warnings.append(f"步骤 {idx} 依赖自身，已忽略")
                continue
            if d not in by_index:
                warnings.append(f"步骤 {idx} 依赖不存在的步骤 {d}，已忽略")
                continue
            deps.append(d)
        clean[idx] = deps

    # ② Kahn 拓扑排序（同层按原顺序，保证输出稳定）
    order: list[int] = []
    remaining = list(by_index.keys())
    while remaining:
        ready = [i for i in remaining if all(d in order for d in clean[i])]
        if not ready:
            # 有环：降级为原顺序（不卡死），并如实记录
            warnings.append("计划依赖存在环，已降级为原顺序执行")
            return list(steps), warnings
        order.extend(ready)
        remaining = [i for i in remaining if i not in ready]

    ordered = [by_index[i] for i in order]
    if order != list(by_index.keys()):
        warnings.append("已按 depends_on 重排执行顺序")
    return ordered, warnings


async def run_planning(goal: str, llm, rag, max_steps: int = 5) -> dict:
    """把复杂目标分解为 ≤max_steps 子任务，并逐子任务检索取证，返回结构化执行报告。

    - fake 模型：规则分解（零成本、行为确定）。
    - 真实模型：用 Planner（LLM 驱动）分解；失败回退规则分解。
    - 逐子任务执行 = 对每个子任务描述做一次 retrieve（bounded、cheap），把命中段落作为证据；
      最终答复由主 Agent 循环基于该证据合成（满足「逐子任务执行 → 汇总」）。
    """
    goal = (goal or "").strip()
    if not goal:
        return {"ok": False, "error": "目标不能为空"}
    max_steps = max(1, min(int(max_steps), 5))
    if isinstance(llm, FakeLLM):
        steps = _rule_decompose(goal, max_steps)
    else:
        try:
            plan = await Planner(llm, max_steps=max_steps).plan(goal)
            steps = [
                {"index": s.index, "description": s.description, "depends_on": s.depends_on}
                for s in plan.steps
            ][:max_steps] or _rule_decompose(goal, max_steps)
        except Exception:  # noqa: BLE001  # 规划/LLM 失败降级为规则分解
            steps = _rule_decompose(goal, max_steps)
    # C6：先按 depends_on 拓扑排序，再执行
    steps, order_warnings = _resolve_order(steps)

    execution: list[dict] = []
    retrieval_failed = False  # K4：检索服务故障标记（区别于"单步无证据"）
    failed_steps: set[int] = set()  # C6：真失败的步骤（用于向依赖它的下游传播）

    for s in steps:
        idx = s["index"]
        deps = s.get("depends_on") or []

        # ── C6 失败传播 ──
        # 只有**依赖的前置步骤真失败**时才跳过。这里刻意用独立的 `failed_steps`
        # 而不是"看 evidence 是否为空"：**"无证据"不是失败**（那是"文档里没有"，是有效结论，
        # 沿用 K4 的既有口径），把两者混同会让"文档确实没写"被误判成"执行失败"。
        blocked = [d for d in deps if d in failed_steps]
        if blocked:
            execution.append({
                "step": idx, "description": s["description"], "depends_on": deps,
                "evidence": [], "status": "skipped",
                "reason": f"前置步骤 {blocked} 执行失败，已跳过（依赖未满足）",
            })
            continue

        evidence: list[dict] = []
        status = "ok"
        if rag is not None:
            try:
                # ★ 形态适配（2026-10-01）：`features.multi_query` 开启时 rag 是
                # `MultiQueryRAGPipeline`，其 `query` 是 **async** —— 原先此处**同步取值拿到
                # 协程对象**，`for c in chunks` 抛 `TypeError: 'coroutine' object is not iterable`，
                # 被下面的容错 except 吞掉 ⇒ **每一步的 evidence 静默为空**，而 K4 的
                # "检索服务故障"标记**不会触发**（那是 `RetrievalUnavailable`，不是 TypeError）
                # ⇒ 把**系统故障伪装成"文档里没有"**。统一走 `call_query`（N4/N9 唯一实现）。
                chunks = await call_query(rag, s["description"], top_k=3)
                evidence = [
                    {"id": c.id, "page": c.page, "text": _truncate_at_sentence(c.text or "", 200)}
                    for c in chunks
                ]
            except Exception as e:  # noqa: BLE001 单步取证失败不拖垮整体
                # 2026-09-09 修复 K4（F7/I1 口径）：检索**服务故障** ≠ 无证据——
                # 把故障伪装成"每步都无证据"会误导模型得出"文档不支持该计划"的结论。
                # 在结果中显式标记（延迟导入同 basic.py：避免模块级引入 RAG 重依赖）。
                from ...rag.pipeline import RetrievalUnavailable

                if isinstance(e, RetrievalUnavailable):
                    retrieval_failed = True
                    status = "failed"  # C6：服务故障才算失败 → 可向下游传播
                    failed_steps.add(idx)
                evidence = []
        execution.append({
            "step": idx, "description": s["description"],
            "depends_on": deps, "evidence": evidence, "status": status,
        })
    skipped = [ex for ex in execution if ex.get("status") == "skipped"]
    result = {
        "ok": True,
        "goal": goal,
        "num_steps": len(steps),
        "plan": steps,
        "execution": execution,
        "note": f"已按 ≤{max_steps} 步分解并逐子任务检索证据；请基于以上证据合成最终答复。",
    }
    # C6：把编排决策如实回传（重排/跳过都要能看见，否则"为什么这步没跑"无从解释）
    if order_warnings:
        result["plan_warnings"] = order_warnings
    if skipped:
        result["skipped_steps"] = [
            {"step": ex["step"], "reason": ex.get("reason", "")} for ex in skipped
        ]
    if retrieval_failed:
        # K4：显式告知检索服务故障，防止模型把空证据解读为「文档不支持该计划」。
        # ⚠️ 这里是**整体替换** note —— 所以 C6 的跳过提示必须追加在**它之后**，
        # 否则会被覆盖掉（实测：两个信息同时成立时只剩 K4 那条，测试抓出来的）。
        result["retrieval_unavailable"] = True
        result["note"] = (
            "【检索服务暂时不可用】索引加载失败，各步骤证据为空是系统故障所致，"
            "并非文档中没有相关内容；请勿据此断言计划不可行，应告知用户稍后重试。"
        )
    if skipped:
        # 追加（而非替换）：两条提示可能同时成立，都要传给模型
        result["note"] += (
            f" 其中 {len(skipped)} 个步骤因前置失败被跳过，"
            "请在答复中说明哪些部分未能完成及原因，不要臆测其结论。"
        )
    return result


def _make_plan_executor(llm, rag):
    async def _fn(goal: str, max_steps: int = 5) -> ToolResult:
        out = await run_planning(goal, llm, rag, max_steps)
        if not out.get("ok"):
            return ToolResult(ok=False, output=None, error=out.get("error", "规划失败"))
        return ToolResult(ok=True, output=out)

    return _fn


def register_planning_tools(registry, llm, rag, enabled: bool = True) -> None:
    """注册 S9 复杂规划工具 plan_executor。enabled=False 时不注册（功能关闭）。"""
    if not enabled:
        return
    registry.register(ToolSpec(
        name="plan_executor",
        description="当目标复杂、需要拆解为多步子任务时使用：把用户目标分解为 ≤5 个有序子任务，"
                    "并逐子任务检索相关段落作为证据，返回结构化执行报告（含计划与各步取证），"
                    "供你据此合成最终答复。仅对明确需要规划的复杂任务调用。",
        parameters={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "需要规划/拆解的复杂目标或任务"},
                "max_steps": {"type": "integer", "description": "最大子任务数（1~5）", "default": 5},
            },
            "required": ["goal"],
        },
        permission=Permission.SAFE,
        handler=_make_plan_executor(llm, rag),
    ))


# ---------------- S3：跨会话记忆 ----------------
def _make_memory_search(memory):
    # N9 同源修复（2026-09-11）：改为 **async handler** —— `recall_long` 内部含 BGE-M3
    # 向量编码（同步、计算密集），在事件循环里直接调用会阻塞循环（与检索工具同病根）。
    # `ToolExecutor` 支持 awaitable handler，故下放线程池即可，语义不变。
    async def _memory_search(query: str, top_k: int = 5) -> ToolResult:
        # K7 修复（2026-09-10）：top_k 是模型生成的工具参数（可能是字符串 "five" / None）——
        # 原先 `int(top_k)` 直接抛 ValueError/TypeError 使整个工具调用失败；
        # 与同文件 _memory_write 对 importance 的 try/except 口径统一。
        try:
            k = int(top_k)
        except (TypeError, ValueError):
            k = 5
        items = await asyncio.to_thread(memory.recall_long, query, k=max(k, 1))
        return ToolResult(ok=True, output={
            "query": query,
            "memories": [
                {"id": it.id, "text": it.text, "importance": round(it.importance, 2)}
                for it in items
            ],
            "count": len(items),
        })

    return _memory_search


def _make_memory_write(memory):
    # N9 同源修复（2026-09-11）：同 `_memory_search`，写入路径也含向量编码 → 下放线程池。
    async def _memory_write(text: str, importance: float = 0.5) -> ToolResult:
        text = (text or "").strip()
        if not text:
            return ToolResult(ok=False, output=None, error="记忆内容不能为空")
        # 复用长期记忆的毒化拦截，避免重新引入『状态观察型/拒接型』毒化条目
        # （见 long_term._is_poisoned 与 2026-08-19 三续的修复）
        if memory.long._is_poisoned(text):
            return ToolResult(
                ok=False, output=None,
                error="该内容属于临时状态观察/拒接型表述，不写入长期记忆。",
            )
        try:
            imp = max(0.0, min(1.0, float(importance)))
        except (TypeError, ValueError):
            imp = 0.5
        it = await asyncio.to_thread(memory.long.add, text, importance=imp)
        return ToolResult(ok=True, output={
            "id": it.id, "text": it.text, "importance": round(it.importance, 2),
        })

    return _memory_write


def register_memory_tools(registry, memory, enabled: bool = True) -> None:
    """注册 S3 跨会话记忆工具。enabled=False 时不注册（功能关闭）。"""
    if not enabled:
        return
    registry.register(ToolSpec(
        name="memory_search",
        description="召回与 query 相关的长期记忆（用户偏好 / 历史结论），用于『用户此前有没有提过类似需求』类问题。返回带重要性的记忆条目。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索长期记忆的查询"},
                "top_k": {"type": "integer", "description": "返回条数", "default": 5},
            },
            "required": ["query"],
        },
        permission=Permission.SAFE,
        handler=_make_memory_search(memory),
    ))
    registry.register(ToolSpec(
        name="memory_write",
        description="把一条用户偏好 / 长期结论显式写入长期记忆（跨会话共享）。用于用户明确要求『记住…』时。",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要写入的记忆文本"},
                "importance": {"type": "number", "description": "重要性 0~1", "default": 0.5},
            },
            "required": ["text"],
        },
        permission=Permission.SAFE,
        handler=_make_memory_write(memory),
    ))


# ---------------- S4：一键总结 ----------------
def _make_summary_handler(summary_svc, kind: str, default_doc_ids):
    async def _fn(doc_ids: list[int] | None = None, lang: str = "zh") -> ToolResult:
        ids = doc_ids or default_doc_ids or []
        if not ids:
            return ToolResult(
                ok=False, output=None,
                error="未指定文档（doc_ids 为空且当前会话未关联文档）。请先在右栏关联文档。",
            )
        try:
            if kind == "summary":
                out = await summary_svc.summarize(ids, lang=lang)
            else:
                out = await summary_svc.related_work(ids, lang=lang)
        except Exception as e:  # noqa: BLE001 工具失败需被 Agent 循环捕获
            return ToolResult(ok=False, output=None, error=f"{type(e).__name__}: {e}")
        return ToolResult(ok=True, output={"kind": kind, "doc_ids": ids, "result": out})

    return _fn


def register_summary_tools(registry, summary_svc, default_doc_ids=None, enabled: bool = True) -> None:
    """注册 S4 一键总结工具。enabled=False 时不注册（功能关闭）。"""
    if not enabled:
        return
    registry.register(ToolSpec(
        name="summarize",
        description="对指定文档产出『背景 / 方法 / 实验 / 结论 / 局限』五段式结构化总结（含引用溯源）。S4 一键总结。",
        parameters={
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "integer"},
                            "description": "文档 id 列表；留空则用当前会话关联文档"},
                "lang": {"type": "string", "description": "输出语言", "default": "zh"},
            },
            "required": [],
        },
        permission=Permission.SAFE,
        handler=_make_summary_handler(summary_svc, "summary", default_doc_ids),
    ))
    registry.register(ToolSpec(
        name="related_work",
        description="对指定文档产出『相关工作』综述草稿 + 引文来源。S4 一键总结。",
        parameters={
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "integer"},
                            "description": "文档 id 列表；留空则用当前会话关联文档"},
                "lang": {"type": "string", "description": "输出语言", "default": "zh"},
            },
            "required": [],
        },
        permission=Permission.SAFE,
        handler=_make_summary_handler(summary_svc, "related_work", default_doc_ids),
    ))
