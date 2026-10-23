"""PaperMate · 对话服务（docs/04 时序 + docs/07 M3.1）

- 为每个会话构建 AgentLoop：LLM（fake 或真实端点）+ 会话关联文档的组合检索 CompositeRAGPipeline
  + 内置工具（retrieve/calculator/code_interpreter 沙箱）。
- 流式运行 AgentLoop.run_stream，收集事件交给路由层做 SSE；运行结束持久化消息/工具记录/trace。
- 多会话隔离：会话历史按 session_id 从 DB 重建；文档管线按 doc_id 缓存，天然互不干扰。

说明：成本/延迟为基于 token 粗算的估算（fake 模型恒为 0）；真实精确成本来自 LightCore LLM 层
（M1.1 已实现），后续可由 AgentLoop 的 done 事件回传做精确归集（M4.2 可观测性）。
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, Role, ToolCall, _rough_tokens
from lightcore.llm.fake import FakeLLM
from lightcore.llm.router import (
    DOC_BOUND_UNRELIABLE_MODELS,
    MODEL_TO_PROVIDER,
    PRESET_ENDPOINTS,
    PROVIDER_MODELS,
    ModelEndpoint,
    OAICompatibleLLM,
    estimate_cost,
)
from lightcore.memory.poison import POISON_PATTERNS, is_poisoned
from lightcore.rag.pipeline import RAGPipeline, RetrievalUnavailable, RetrievedChunk
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry

from .. import runtime_settings
from ..config import Settings
from ..memory_hub import get_memory_manager
from ..models.db import Message, ToolRun, Trace
from ..prompts import SYSTEM_PROMPT
from .db_retry import write_with_retry
from .doc_labels import doc_label, mention_order
from .document_service import DocumentService
from .session_service import SessionService

# 多模态历史消息存储标记：Message.content 以此前缀开头时，其后的 JSON 为 OpenAI 兼容内容块列表
# （text/image_url）。普通文本消息不可能以控制字符开头，可安全区分，无需改表结构。
_MM_PREFIX = "\x00mm\x00"

logger = logging.getLogger(__name__)


def _encode_user_content(content: str | list[dict]) -> str:
    """用户消息入库存串：块列表 → 前缀标记 JSON；纯文本原样。"""
    if isinstance(content, list):
        return _MM_PREFIX + json.dumps(content, ensure_ascii=False)
    return content


def _decode_message_content(content: str) -> str | list[dict]:
    """消息出库还原：前缀标记 JSON → 块列表；否则原样字符串。"""
    if content.startswith(_MM_PREFIX):
        try:
            return json.loads(content[len(_MM_PREFIX):])
        except json.JSONDecodeError:
            return content
    return content


def _grounding_block(
    enable_network: bool, has_docs: bool = True, focused_label: str | None = None,
) -> str:
    """随会话『联网』开关与『是否关联文档』动态注入的统一限制约束（确保所有模型行为一致）。

    - 未关联任何文档（has_docs=False）：进入「纯大模型对话」模式，允许基于通用知识作答
      （不引用任何论文，因为无文档可引）；仅当用户问题指向某篇具体论文时，提示先去关联文档。
      此模式解决「无文档时仍被网络关闭约束禁言」的矛盾，使未上传文档的会话可正常聊天。
    - 已关联文档 + 联网关闭（默认）：严禁访问/引用文档外来源，严禁臆测；内容不足须明说并提示开启联网。
    - 已关联文档 + 联网开启：仍以文档为首要依据，仅不足时可用联网工具补充，且须标注来源、区分文档/联网。

    ★★ N2（2026-10-02）`focused_label`：本轮用户用 `@文档N` **限定了唯一一篇**时的声明。
    ## 为什么需要（实测事故）
    N1 已让检索结果**自带 `doc` 标签**，但模型仍可能用「这篇论文/该文」模糊指代
    （实测四轮回答内容全对，措辞却是「就所取到的内容看，这篇论文…」⇒ 用户无法确认它读的是哪篇）。
    ⇒ 在**系统提示层**明确"本轮只读这一篇、且必须称它为「文档N」"，让首轮就定调。
    ⚠️ 只在**恰好一篇**时注入；多篇/未限定都不注入（否则与实际范围矛盾）。
    """
    if not has_docs:
        if enable_network:
            return (
                "\n\n【对话模式（当前：未关联文档 · 已开启联网）】\n"
                "- 本会话未关联任何文档，以下为通用知识对话，不引用任何论文内容。\n"
                "- 联网检索能力**是否可用取决于当前所用模型**：具备时，遇『最新进展 / 实时数据 / "
                "需公开核实的事实 / 自己没把握』一类问题应主动联网核实并标注来源；"
                "不具备时须**如实说明无法联网**，严禁假装检索或编造来源。\n"
                "- 你可基于大模型自身知识（含联网检索补充）回答通用问题；若用户希望研读某篇具体论文，"
                "请明确说明：需先在右侧「关联文档」勾选该论文，才能基于其内容作答。\n"
                "- 直接给答案，不要输出『过程：』『意图识别』『文档检索』等流程叙述。\n"
            )
        return (
            "\n\n【对话模式（当前：未关联文档 · 未开启联网）】\n"
            "- 本会话未关联任何文档，以下为通用知识对话，请基于你的通用知识直接回答，不引用任何论文。\n"
            "- 若用户问题指向某篇具体论文的内容，请明确说明：需要先在右侧「关联文档」勾选该论文，"
            "才能基于其内容检索作答；不要编造论文内容。\n"
            "- 直接给答案，不要输出『过程：』『意图识别』『文档检索』等流程叙述。\n"
        )
    # ★★ N2（2026-10-02）：本轮限定到**唯一一篇**时的声明（口径见 docstring）。
    # ⚠️ 只在 focused_label 非空（即恰好一篇）时注入；多篇/未限定都不给。
    #
    # ★★ 文案要点（2026-10-02 实测修正）：**不得假设用户是用弹窗选的**。
    #   前端弹窗会发结构化 id（用户看到的是tab 上的「文档三」），
    #   手打则只是正文里的「@文档三」文字 ⇒ 两种来源对模型**长得一样**。
    #   若声明写"用户用 @文档一 指定"，而用户正文写的是 @文档三，
    #   模型会整段推理去纠结"哪里冲突"（实测 thinking 原文证实）⇒ **必须用中性表述**。
    _focus = (
        "\n\n【本轮文档范围（用户已指定）】\n"
        f"- 用户本轮**指定了唯一一篇**文档：{focused_label}；检索只会返回该篇内容。\n"
        f"- 回答时**必须明确称它为「{focused_label}」**（如「{focused_label}提到…」），"
        "禁止用「这篇论文」「该文」「所取到的内容」等模糊指代——"
        "用户无法据此确认你读的是哪一篇。\n"
        f"- 检索结果每条都带 doc 字段标明来源篇目；请据此写明是{focused_label}的内容。\n"
        f"- ★ 你的任务是回答用户的问题，**不要纠结编号**；"
        f"若正文里的 @ 标签与本声明不一致，**以检索结果 doc 字段为准**"
        f"（它标记的才是你真正读到的那篇），并直接使用{focused_label}这个标签。\n"
        if focused_label
        else ""
    )
    if enable_network:
        return (
            "\n\n【联网检索（当前：已开启）】\n"
            "- 前提：联网检索能力**是否可用取决于当前所用模型**。具备该能力时按下列规则使用；"
            "若不具备，必须**如实说明无法联网**，严禁假装检索过、严禁编造来源与链接。\n"
            "- 以下情形应主动联网核实后再作答，不要仅凭既有记忆作答：\n"
            "  ① 涉及近期进展、新版本或新发布（『最新』『今年』『刚发布』一类问题）；\n"
            "  ② 需要实时或具体的数据（榜单排名、版本号、价格、发布日期、仓库地址）；\n"
            "  ③ 文档未覆盖、但可通过公开渠道核实的事实；\n"
            "  ④ 你对答案的时效性或准确性没有把握。\n"
            "- 文档或常识即可回答的问题**无需**联网，不要为了联网而联网。\n"
            "- 答复仍以用户已上传文档为首要依据；联网补充的信息必须标注来源（链接或出处），"
            "并明确区分『来自文档』与『来自联网』，不得将外部内容冒充为文档内容。\n"
            "- 仍严禁编造；联网检索结果亦须如实引用，不确定的内容标注『待人工核实』。"
            + _focus
        )
    return (
        "\n\n【联网限制（当前：关闭）】\n"
        "- 联网功能未开启：大模型原生联网搜索不可用，严禁调用，也不得访问或引用"
        "文档之外的任何数据来源（含你自身记忆或先验知识中的外部事实）。\n"
        "- 所有回复必须以已上传文档的实际内容为唯一依据，严禁编造或引入与文档无关的事实、数据、信息。\n"
        "- 当文档内容不足以回答时，必须明确说明『现有文档不足以回答该问题』，"
        "并建议用户在设置页开启『联网』功能，而非自行推断、猜测或补充文档外知识。"
        + _focus
    )


class ApiFakeLLM(FakeLLM):
    """API 用的 FakeLLM：用用户真实问题作为检索 query，并把工具结果整理成可读摘要。

    仅用于无 key 的本地演示（与 cli.CliFakeLLM 行为一致，但独立定义避免 api→cli 反向依赖）。
    """

    def __init__(self, top_k: int = 3, model_name: str = "fake-api") -> None:
        super().__init__(model_name)
        self.top_k = top_k

    def _decide(self, last_content: str, messages: list[ChatMessage]) -> ChatMessage:
        if last_content.startswith("RESULT:"):
            payload = _parse_result(last_content)
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not results:
                # 检索为空（含 source=none：会话未关联文档）。把 tool 返回的 note 一并回显，
                # 让用户明白"为什么没有检索内容"，而非误以为是检索失败。
                note = payload.get("note") if isinstance(payload, dict) else None
                msg = "（未检索到相关段落）"
                if note:
                    msg += f"\n\n{note}"
                return ChatMessage(role=Role.ASSISTANT, content=msg)
            items = "\n".join(f"• {_first_sentence(r.get('text', ''))}" for r in results)
            src = payload.get("source", "")
            return ChatMessage(
                role=Role.ASSISTANT,
                content=(
                    f"根据检索内容（来源：{src}），为您整理出 {len(results)} 处相关要点：\n"
                    f"{items}\n\n以上为原文要点摘录，可结合下方「引用来源」定位具体出处。"
                ),
            )
        user_q = next((m.content for m in reversed(messages) if m.role == Role.USER), last_content)
        return ChatMessage(
            role=Role.ASSISTANT,
            content="我先检索一下相关段落。",
            tool_calls=[ToolCall(id="call_1", name="retrieve",
                                 arguments={"query": user_q, "top_k": self.top_k})],
        )


class CompositeRAGPipeline:
    """把多个文档的 RAGPipeline 组合成一个 retrieve 可用的管线（跨文档合并 Top-k）。

    查询预处理（P1-2 / ER-12；2026-09-26 T-04+ 收敛）：
    默认走 **`query_rewrite.rewrite_query`（④ 加固版）** —— 口语 → 检索式，
    用 **jieba 分词提供词边界 + 按整词删除零信息量词**，**保留疑问词**（它们是信号词）。

    历史：加固前 `rewrite_query` 用裸 `str.replace()` 子串替换，会破坏 query
    （`目的是什么`→`目是`、`受什么影响`→`受影响`），一度临时改用最保守的
    `expand_sections`（一个词都不删）止血；④ 加固后两者**实测等价**
    （221 例 hit/recall 全同，v2 的 ndcg 略优），故切回 `rewrite_query`。
    评测三臂对照：`--query-rewrite {sections,v2,legacy}`；取证见
    `docs/query_rewrite-缺陷取证.md`。
    """

    def __init__(
        self,
        pipelines: list[RAGPipeline],
        doc_ids: list[int] | None = None,
        *,
        query_prep: Callable[[str], str] | None = None,
    ) -> None:
        self._pipelines = [p for p in pipelines if p is not None]
        #: 查询预处理函数。`None` = 默认 `rewrite_query`（④ 加固版，与线上同源）。
        #: 评测注入其它实现做三臂对照（报告写 `query_rewrite_mode` 自证）。
        if query_prep is None:
            from lightcore.rag.query_rewrite import rewrite_query

            query_prep = rewrite_query
        self._query_prep = query_prep
        # 与 `self._pipelines` **逐位对应**的 doc id（用于把\"哪一篇失败了\"报成可读信息）。
        # 可选：不传时只报数量，行为不变（调用方按需补齐）。
        ids = list(doc_ids or [])
        self._doc_ids: list[int | None] = (
            ids[: len(self._pipelines)] if len(ids) == len(self._pipelines)
            else [None] * len(self._pipelines)
        )
        #: ★ 最近一次 query 的**部分失败**信息（2026-09-23）。
        #:
        #: `None` = 本次无失败（或尚未查询过）；非 None 时形如
        #: `{"failed_doc_ids": [...], "failed_count": n, "total": m}`。
        #:
        #: **为什么需要它**：全失败会抛 `RetrievalUnavailable`（F7，有完整信号），
        #: 但**部分失败只 warning 就静默继续** —— 调用方拿到的是\"较少的结果\"，
        #: 与\"文档里确实没有相关内容\"在数据上无法区分。于是用户以为检索了全部文档，
        #: 而回答实际只基于其中一部分，且模型不会声明这一点。这与 F7 要修的错同一类。
        #:
        #: ⚠️ 这是**实例可变状态**，在此处可接受，理由：实例由 `_build_agent` **按轮创建**，
        #: 轮内工具调用串行；multi_query 模式下多个子查询并发调用同一实例，
        #: 但它们面对的是**同一批文档**、同一失败成因 → 失败篇目集合相同，
        #: 后写覆盖前写不改变结论。**不要**把它升级为跨轮的全局状态。
        self.last_partial_failure: dict | None = None

    def query(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        # T-04+（2026-09-26）：默认 `expand_sections`（只做命令剥离 + 章节展开）。
        # 旧 `rewrite_query` 的疑问词/停用词清洗会破坏 query（`目的是什么`→`目是`），
        # 且改写结果非空 ⇒ 下面的 `or query` 回退保护不触发 ⇒ 静默劣化检索。
        rewritten = self._query_prep(query) or query
        merged: list[RetrievedChunk] = []
        failed_ids: list[int] = []
        failed = 0
        for pos, p in enumerate(self._pipelines):
            try:
                merged.extend(p.query(rewritten, top_k=top_k))
            except Exception:  # noqa: BLE001  单篇管线故障不拖垮跨文档检索
                failed += 1
                doc_id = self._doc_ids[pos] if pos < len(self._doc_ids) else None
                if doc_id is not None:
                    failed_ids.append(doc_id)
                logger.warning("检索管线失败（已跳过该文档）", exc_info=True)
                continue
        # ★ 2026-09-23：记录**部分失败**，供检索工具如实告知（此前只有 warning，用户不可见）。
        # 全失败分支不写这里 —— 那条路径抛异常，上层已有专门的 retrieval_unavailable 信号。
        self.last_partial_failure = (
            {"failed_doc_ids": failed_ids, "failed_count": failed, "total": len(self._pipelines)}
            if failed and self._pipelines and failed < len(self._pipelines)
            else None
        )
        # F7（2026-09-07）：全部管线失败 ≠ 文档无内容。必须显式抛出以示区分——
        # 否则「检索故障」会被上层当成「文档里没有相关内容」告知用户（系统提示词第 2.2 条
        # 要求空结果时回答『知识库中没有相关内容』），诱使用户重复上传而非排查系统。
        if self._pipelines and failed == len(self._pipelines):
            raise RetrievalUnavailable(
                f"全部 {failed}/{len(self._pipelines)} 个检索管线均失败",
                failed=failed,
                total=len(self._pipelines),
            )
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    def get_all_chunks(self) -> list:
        """聚合所有子管线的原文 chunk，供 pdf_extract / citation_lookup 等工具复用（跨文档）。"""
        chunks: list = []
        for p in self._pipelines:
            chunks.extend(p.get_chunks())
        return chunks


def _parse_result(last_content: str):
    try:
        return ast.literal_eval(last_content[len("RESULT:"):].strip())
    except Exception:  # noqa: BLE001  # ast 解析失败降级返回空结果
        return {}


def _first_sentence(text: str, max_chars: int = 90) -> str:
    """取首句并截断（摘要式回显用，避免把检索块全文/内部符号原样输出）。"""
    if not text:
        return ""
    for sep in ("。", "！", "？", "；", ". ", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < max_chars:
            text = text[: idx + len(sep.strip())]
            break
    text = text.strip().replace("\n", " ")
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


class ChatService:
    def __init__(
        self,
        session_svc: SessionService,
        doc_svc: DocumentService,
        settings: Settings,
        SessionLocal,
        summary_svc=None,
        collection_svc=None,
    ) -> None:
        self._sessions = session_svc
        self._docs = doc_svc
        self._settings = settings
        self._Session = SessionLocal
        self._summary = summary_svc
        # F5：文献集服务——会话可关联整个文献集（config.collection_id），解析为有效文档
        self._collections = collection_svc
        # 跨会话记忆（全局单例，持久化到 data/memories.json）——满足 M2.1 验收#2
        # 2026-08-29 A 档：改为共享单例（papermate.memory_hub），使设置页「长期记忆管理」
        # 面板的编辑 / 删除即时对对话生效（此前面板只写文件、对话侧读进程内存，需重启才同步）。
        # B 档：改为按需解析（见 _memory 属性），切换记忆向量引擎后无需重启。

    @property
    def _memory(self):
        """共享的长期记忆管理器（每次取用都解析单例，保证切换引擎后立即生效）。"""
        return get_memory_manager(self._settings)

    # ---------- 构建会话级 Agent ----------
    def _llm_from_raw(self, raw: str) -> Any:
        """按 raw（provider 名或具体 model id）解析 LLM 实例；未知/缺失一律退化 ApiFakeLLM。

        供 build_agent（主对话）与 resolve_llm（提示词增强等轻量元任务）共用，保持同源。
        """
        # 解析 provider：raw 可能是 provider 名（deepseek），也可能是具体 model id（deepseek-chat）
        if raw in PRESET_ENDPOINTS:
            provider = raw
        elif raw in MODEL_TO_PROVIDER:
            provider = MODEL_TO_PROVIDER[raw]
        else:
            provider = "fake"
        if provider == "fake":
            return ApiFakeLLM()
        ep = PRESET_ENDPOINTS.get(provider)
        if ep is None:
            return ApiFakeLLM()  # 未知 provider → 退化 fake，避免崩溃
        pset = runtime_settings.get_state()["providers"].get(provider, {}) or {}
        base_url = pset.get("base_url") or ep.base_url
        # 具体模型：显式指定(且属于该 provider) > 该 provider 已选 > 厂商默认
        if raw in PROVIDER_MODELS.get(provider, []):
            concrete = raw
        else:
            concrete = pset.get("model") or ep.default_model
        # key 显式传入（优先于全局 os.environ，避免并发请求相互污染）
        return OAICompatibleLLM(
            ModelEndpoint(ep.name, base_url, ep.api_key_env, concrete),
            api_key=pset.get("api_key") or None,
        )

    def _resolve_raw_model(self, session_id: int | None = None) -> str:
        """模型优先级：会话 config.model → 运行时默认模型（前端可设）→ 后端 env 默认。"""
        raw: str | None = None
        if session_id is not None:
            try:
                raw = self._sessions.get_config(session_id).get("model")
            except Exception:  # noqa: BLE001  会话异常回退全局默认
                raw = None
        return raw or runtime_settings.get_state()["default_model"] or self._settings.default_model

    def resolve_llm(self, session_id: int | None = None):
        """按「会话 config.model → 运行时默认模型 → env 默认」解析一个 LLM 实例。

        与 build_agent 的模型优先级完全同源（_resolve_raw_model），供提示词增强等
        轻量元任务在不开 AgentLoop 的情况下复用同一套厂商/密钥/模型选择。
        """
        return self._llm_from_raw(self._resolve_raw_model(session_id))

    def effective_model(self, session_id: int) -> str:
        """返回本轮实际用于对话的模型 id —— **恒等于用户所选**，绝不再静默换模型。

        2026-09-16 策略变更（用户决策）：此前 doc-bound 会话对「绑文档不可靠」的模型
        （DOC_BOUND_UNRELIABLE_MODELS，如 qwen3-vl-flash）会**静默回落**到
        deepseek-v4-flash-vision-exp。**实测危害**：用户选了 Qwen，请求却发给 DeepSeek，
        按 DeepSeek 计费、报 DeepSeek 的错 —— 用户完全无法自救（他以为在用 Qwen）
        （当日计费反推：会话 18 三条历史的成本分毫不差等于 DeepSeek 定价，铁证）。
        现改为：**不回落**，改用 `doc_bound_model_risk` 给出**警示但不拦截**，由用户自己判断。
        """
        return self._resolve_raw_model(session_id)

    def doc_bound_model_risk(self, session_id: int) -> str:
        """doc-bound（绑定了文档）会话若选择了「绑文档不可靠」的模型，返回**给用户看的警示**；无风险返回 ""。

        「不可靠」= 已知在该场景会不调用检索、直接编造文档内容（DOC_BOUND_UNRELIABLE_MODELS，
        详见 lightcore/llm/router.py）。

        ⚠️ 2026-09-16 晚：策略从「入口硬阻断」改为**「警示但放行」**（用户选 B）：
        阻断虽然零幻觉，但也彻底剥夺了用户使用自费模型的权利。现在照常调用用户所选模型，
        只在 UI 上常驻警示（输入框上方横幅 + 该轮回答上的风险标记），
        把「是否接受幻觉风险」的决定权交还用户 —— 但**必须让用户看得见风险**。
        """
        raw = self._resolve_raw_model(session_id)
        if self.should_force_tool_call(session_id) and raw in DOC_BOUND_UNRELIABLE_MODELS:
            return (
                f"当前模型 {raw} 在「绑定文档」的会话中可能跳过检索、直接编造文档内容（已知问题）。"
                f"回答请务必核对引用来源。如需可靠检索，建议更换为 qwen3-max / qwen-max / qwen-turbo。"
            )
        return ""

    async def build_agent(
        self,
        session_id: int,
        cancel_event: threading.Event | None = None,
        restrict_doc_ids: list[int] | None = None,
    ) -> AgentLoop:
        # 2026-09-07：restrict_doc_ids——@ 提及指定文档时，检索管线只构建被提及的篇目
        # （普通对话的 retrieve 不再混入其他文档内容）。None/空 = 全部关联文档。
        cfg = self._sessions.get_config(session_id)
        # 模型解析：effective_model 现在恒等于用户所选（2026-09-16 起不再回落）；
        # doc-bound + 不可靠模型的风险由 routes_search 用 doc_bound_model_risk 警示（不拦截）。
        raw = self.effective_model(session_id)
        llm = self._llm_from_raw(raw)

        doc_ids = self._effective_doc_ids(session_id)
        # ★★ N1（2026-10-02）：**标签口径 = 会话【全量】关联集**（id 降序），
        # 必须在 `restrict_doc_ids` 限缩**之前**算好。
        #
        # ⚠️⚠️ **不要基于限缩后的 doc_ids 重算**（2026-10-02 踩过，这是真实缺陷）：
        #   用户 @文档三（id 14，会话共 4 篇）⇒ 限缩后只剩 [14] ⇒ 重算得「文档一」
        #   ⇒ N2 声明变成"用户用 @文档一 指定了…"，而**用户明明写的是 @文档三**。
        #   实测模型整段推理都在纠结这个冲突（thinking 原文）：
        #     "本轮文档范围说用户用 @文档一 指定了唯一一篇文档，但用户实际写的是 @文档三。
        #      规则说『必须明确称它为「文档一」』，但用户问的是文档三。这里有冲突。"
        #   ⇒ 模型被迫"折中"（写成「文档一（《…算法》）」），反而**更困惑**。
        #   **面板上的「文档三」是按全量关联集编号的** ⇒ 标签必须与之一致。
        _label_source = list(doc_ids)
        if restrict_doc_ids:
            allowed = set(restrict_doc_ids)
            doc_ids = [d for d in doc_ids if d in allowed] or doc_ids  # 全部失效时防御性回落全量
        # 让 `retrieve` 结果**自带身份**（每条带「文档N」标签，见 `basic._doc_label_of`）。
        doc_labels = {
            d: doc_label(idx) for idx, d in enumerate(mention_order(_label_source))
        }
        # 会话级『联网』开关（前端「会话覆盖 → 允许联网」持久化到 session.config.enable_network）。
        # 默认 False（安全默认：未主动开启时模型既不能调联网工具、也不得引用任何文档外来源）。
        enable_network = bool(cfg.get("enable_network", False))
        # 2026-09-11：多文档管线**并发加载**。
        # 原先 `for d in doc_ids: await get_pipeline(d)` 是**串行**的——每篇首次访问都要
        # load_state（读 faiss + 反序列化）并等待 embedder 单例就绪，单篇 20~30s；
        # 4 篇串起来 80s+，用户感知为「又在重建索引」（实测 doc3=33.4s、doc8=0.1s、doc9=22s）。
        # 并发后总耗时≈最慢一篇。`_get_embedder` / `_get_tokenizer` / `_get_reranker`
        # 各有 asyncio.Lock 保护，不会重复加载 2.2GB 权重。
        # 异常语义保持不变：任一篇抛错（含 IndexingCancelled）即上抛，与原串行首个异常一致。
        loaded = await asyncio.gather(
            *(self._docs.get_pipeline(d, cancel_event=cancel_event) for d in doc_ids),
            return_exceptions=True,
        )
        pipelines: list[RAGPipeline] = []
        # 与 pipelines **逐位对应**的 doc id：让「部分失败」能报出具体是哪几篇
        # （而非只有数量）。注意 `loaded` 与 `doc_ids` 同为 `asyncio.gather` 的输入顺序。
        loaded_doc_ids: list[int] = []
        for pos, item in enumerate(loaded):
            if isinstance(item, BaseException):
                raise item
            if item is not None:
                pipelines.append(item)
                if pos < len(doc_ids):
                    loaded_doc_ids.append(doc_ids[pos])
        rag = (
            CompositeRAGPipeline(pipelines, doc_ids=loaded_doc_ids) if pipelines else None
        )

        # 功能开关（features）：控制涉及大模型 API 的模块是否启用（docs/统一方案 阶段B/C）
        features = runtime_settings.get_state().get("features", dict(runtime_settings.DEFAULT_FEATURES))
        # P1-6 多查询融合：真实 LLM 改写子查询提升召回；fake 下 MultiQueryExpander 自动回退单查询。受开关控制。
        if features.get("multi_query", False) and rag is not None:
            from lightcore.rag.multi_query import MultiQueryExpander, MultiQueryRAGPipeline
            _expander = MultiQueryExpander(llm if not isinstance(llm, ApiFakeLLM) else None)
            rag = MultiQueryRAGPipeline(rag, _expander)

        registry = ToolRegistry()
        register_builtin_tools(registry, rag_pipeline=rag, doc_labels=doc_labels)
        # ★ 2026-10-23：retrieve 被压缩后（每段 ≤200 字）模型可按 id 取全文
        from .doc_tools import register_doc_tools

        register_doc_tools(registry, self._Session)
        # 注：原生联网能力当前仅 Qwen（DashScope）已验证可用，受会话『联网』开关与系统提示词门控，
        # 不再注册独立的网络工具（arxiv_search/web_search/fetch_url）。DeepSeek 等厂商 chat/completions
        # 不支持 web_search 工具，故不注入。

        # 按 feature 开关注册领域工具：关闭则不注册，工具完全不暴露给 Agent（满足"后端启用开关"约束）。
        if features.get("memory", True):
            from lightcore.tools.builtin.domain import register_memory_tools

            register_memory_tools(registry, self._memory)
        if features.get("summary", True) and self._summary is not None:
            from lightcore.tools.builtin.domain import register_summary_tools

            register_summary_tools(registry, self._summary, default_doc_ids=doc_ids)

        # S9 复杂规划（plan_executor 工具，受 features.planning 门控；关闭则不暴露给 LLM）
        if features.get("planning", True):
            from lightcore.tools.builtin.domain import register_planning_tools

            register_planning_tools(registry, llm, rag)

        # NFR-07：代码执行默认关闭，仅当 features.code_interpreter 显式开启时才允许 SANDBOX 权限。
        executor = ToolExecutor(registry, permission_enabled={
            Permission.SAFE: True,
            Permission.NETWORK: enable_network,
            Permission.SANDBOX: bool(features.get("code_interpreter", False)),
            Permission.ADMIN: False,
        })
        # 构造时先用默认 AgentLoop；routes_search 会进一步按『会话语义绑文档』决定是否
        # 在第一轮 THINK 强制 tool_choice='required'，避免 LLM 因历史上下文偏见拒调 retrieve
        # （见 2026-08-19 三续）。
        # 成本熔断上限（默认 ¥1/轮，可配）：来自运行时设置，注入 AgentLoop
        cost_limit = runtime_settings.get_state().get("cost_limit_per_turn", 1.0)
        # token 上限（2026-09-17 拆成两项，口径明确）：
        #   max_tokens_budget   = **单次请求上下文上限**（单轮 prompt 超限即中止；防上下文爆炸）
        #   max_tokens_per_turn = **整轮累计上限**（含所有迭代；防跑飞，与 max_cost_per_turn 互补）
        # 拆分前是一个数当两个用（30000 既当累计熔断又当压缩阈值来源），实测只够 2-3 轮迭代
        # —— 而单轮 prompt 实测已达 24883，正常多轮会话必然误伤。
        # ★ 2026-10-03：改为**按当前会话模型自动推断**（单一来源 `resolve_budget_for`）。
        # ⚠️ 必须传 `raw`（本会话实际用的模型）—— 否则退回显式值、与实际生效的预算不一致。
        # ★ 因为 build_agent **每轮**都执行，「会话中途换模型」会自动生效（无需写回会话配置）。
        token_budget = runtime_settings.resolve_budget_for(raw)
        # ★ 2026-10-03：改为**与单次预算同源派生**（resolve_turn_budget_for）。
        #   起因：单次 budget 由 30k 提到 100k 后，固定的 150000 整轮上限**先于**单次被撞
        #   ⇒ 报「本轮累计 token 超限（156085 > 150000）」。两个量必须同步放大。
        turn_budget = runtime_settings.resolve_turn_budget_for(raw)
        # V4 思考模式（features.thinking）：开启 → LLM payload 显式启用 thinking 并展示推理链；
        # 关闭 → 显式禁用（默认，保证 tool_choice 强制与标准 content 流式兼容）。
        thinking_on = bool(features.get("thinking", False))
        if isinstance(llm, OAICompatibleLLM):
            llm.thinking_mode = thinking_on
        # 本地长期记忆：用户设定的全局规则/偏好 → SYSTEM 注记注入（直接影响所有会话）
        rules = (runtime_settings.get_state().get("agent_rules") or "").strip()
        # ★★ N2（2026-10-02）：本轮限定到**唯一一篇**时，把标签传给 grounding 块
        # （由它注入"必须称它为「文档N」"的声明）。
        # ⚠️ 判据是**限缩后的 doc_labels 只有一项** —— 与 N1 的范围 note 同一口径，
        # 两处必须同步（否则一个说"只有一篇"、另一个给多篇标签）。
        _focused_label = next(iter(doc_labels.values())) if len(doc_labels) == 1 else None
        system_prompt = SYSTEM_PROMPT + _grounding_block(
            enable_network, has_docs=bool(doc_ids), focused_label=_focused_label,
        )
        if rules:
            system_prompt = f"{system_prompt}\n\n【用户设定的长期规则（必须遵守）】\n{rules}"
        agent = AgentLoop(llm=llm, tools=executor,
                          config=LoopConfig(temperature=0.3, max_cost_per_turn=float(cost_limit),
                                            max_tokens_budget=token_budget,
                                            max_tokens_per_turn=turn_budget,
                                            thinking=thinking_on, web_search=enable_network,
                                            # features.reflect_on_abort：失败轮次自评（默认关）；
                                            # 只在循环中止路径生效，正常收敛零额外调用。
                                            reflect_on_abort=bool(features.get("reflect_on_abort", False))),
                          system_prompt=system_prompt,
                          # 方案 D′（2026-09-17）：上一轮的真实 prompt 峰值（存 session config，
                          # 零 DDL）→ 本轮压缩的触发判据。AgentLoop 每轮新建，所以真实值必须
                          # 跨轮传递，否则第 1 步（历史最长、最该压缩的那一步）永远没有真实值可用。
                          last_prompt_tokens=int(cfg.get("last_prompt_tokens") or 0))
        return agent

    def _effective_doc_ids(self, session_id: int) -> list[int]:
        """F5：会话有效文档 = 关联文献集文档 ∪ 会话手动勾选文档（并集，去重）。

        2026-09-07：id **降序**——必须与前端 allDocs（GET /documents 为 id desc）过滤后的
        展示顺序一致，否则 @文档N 序号在前后端指向不同文档（内容指向错误 bug 的根因）。"""
        cfg = self._sessions.get_config(session_id)
        coll_id = cfg.get("collection_id")
        coll_docs = self._collections.resolve_doc_ids(coll_id) if (self._collections and coll_id) else []
        return sorted(set(coll_docs) | set(cfg.get("doc_ids") or []), reverse=True)

    def should_force_tool_call(self, session_id: int) -> bool:
        """当会话已关联文档（doc_ids / 文献集 非空）时，第一轮 THINK 强制 LLM 调用工具。

        即便『会话配置指明要研究论文』，LLM 仍可能因历史上下文中大量『无文档』拒接记录
        选择直接拒接回答。把『必须先用 retrieve』变成协议级硬约束（OpenAI 兼容 tool_choice）
        才能从架构上消除这类幻觉拒接。
        """
        return bool(self._effective_doc_ids(session_id))

    # ---------- 跨会话记忆接入（M2.1）----------
    # 过滤「状态观察型」表述：这些是某一次拒接/守门结论，不是用户偏好，不应作为 SYSTEM 注记再次注入。
    # 历史教训：早期 reflect() 把『系统中无任何上传文档』『对重复指令静默跳过』等毒化结论写入了长期记忆，
    # 后续 recall 时被反复注入，Qwen 据此连续拒调 retrieve 工具（见 2026-08-19（三续））。
    # 毒化条目过滤：唯一来源为 lightcore.memory.poison（2026-08-29 C 档合并，
    # 此前本类自带一份 _POISON_PATTERNS，与 long_term._WRITE_BLOCKLIST 并存且已漂移）。
    _POISON_PATTERNS = POISON_PATTERNS

    @classmethod
    def _is_poisoned(cls, text: str) -> bool:
        return is_poisoned(text)

    #: 召回相对下限：仅保留得分 ≥ 最高分 × 该比例的记忆（0 = 不过滤）
    _MIN_SCORE_RATIO = 0.6

    def recall_note(self, session_id: int, query: str, k: int = 5) -> ChatMessage | None:
        """返回应注入本轮对话的长期记忆 SYSTEM 注记；无则 None。

        设计点：
        - 过滤掉"状态观察型/拒接型"的毒化条目（见 _POISON_PATTERNS）。这些条目曾经让 LLM
          把『无文档』『静默跳过』当成硬规则反复复读，跳过 retrieve 工具。
        - 与 system_prompt（在 _build_messages 末尾）的硬性规则共同作用，确保 LLM 把
          工具真实返回视为事实，而不是把记忆注记当作事实。
        - 2026-08-29 A 档：加相对下限 min_score_ratio——丢弃得分低于「最高分 × 比例」
          的条目，挡掉完全不沾边却被 top-k 捞进来、污染 SYSTEM 注记的记忆。
          用相对而非绝对阈值，避免与具体 embedder 的相似度分布耦合。
        - 2026-08-29 分层 scope：传入 session_id 启用作用域过滤——全局偏好（global）
          跨会话注入，会话事实（session）仅在其来源会话召回，避免不同论文会话间事实串台。
        """
        recalled = self._memory.recall_long(
            query, k=max(k * 2, 10), min_score_ratio=self._MIN_SCORE_RATIO,
            session_id=session_id,
        )  # 多召回以抵消过滤损耗
        recalled = [it for it in recalled if not self._is_poisoned(it.text)][:k]
        if not recalled:
            return None
        content = "【长期记忆】\n" + "\n".join(f"- {it.text}" for it in recalled)
        return ChatMessage(role=Role.SYSTEM, content=content)

    async def reflect(self, session_id: int, llm, conversation: list[ChatMessage]) -> None:
        """一轮结束后反思写入长期记忆（跨会话持久化）。

        2026-08-29 C 档：
        - 传入 session_key 与 source_session_id，使记忆可溯源到产生它的会话，
          并让降本策略按会话计轮（而非全局混计）；
        - 是否真正发起 LLM 调用由 MemoryManager.reflect_policy 决定（auto/always/off），
          未触发时零开销返回。
        """
        # 2026-09-07 修复：不再改写全局单例属性——并发会话会相互覆盖 source_session_id，
        # 记忆被错误归属到另一个会话。改为显式入参传入（MemoryManager.reflect 同步支持）。
        await self._memory.reflect(llm, conversation=conversation, session_key=str(session_id),
                                   source_session_id=session_id)

    # ---------- 会话自动标题（按主题生成，避免默认「新会话」）----------
    async def auto_title(self, session_id: int, llm, first_user_text: str) -> None:
        """会话标题仍为默认值时，按首条用户消息生成简洁、有辨识度的主题标题。

        - 真实 LLM：一次轻量调用生成 ≤12 字标题（去引号/标点）。
        - fake / 失败：规则降级——取首行/首句截断 ≤16 字（零成本，行为确定）。
        - 已有人工标题（非默认）时不覆盖；一次生成后不再重复（标题持久化在 sessions.title）。
        """
        s = self._sessions.get(session_id)
        if s is None:
            return
        if (s.title or "").strip() not in ("", "新会话"):
            return  # 已有标题（含历史回填或人工改名）→ 不覆盖
        text = (first_user_text or "").strip()
        if not text:
            return
        title: str | None = None
        if not isinstance(llm, FakeLLM):  # ApiFakeLLM 继承自 FakeLLM，天然覆盖
            try:
                prompt = (
                    "为以下用户首次提问生成一个不超过 12 个字的会话标题，"
                    "要求简洁、能反映讨论主题、不要标点与引号、直接输出标题：\n" + text[:200]
                )
                resp = await llm.chat([ChatMessage(role=Role.USER, content=prompt)], tools=None, temperature=0.3)
                cand = (resp.message.content or "").strip().strip('"「」“”').strip()
                if 1 <= len(cand) <= 20:
                    title = cand
            except Exception:  # noqa: BLE001 标题生成失败不阻断主流程，走规则降级
                title = None
        if title is None:
            for sep in ("\n", "。", "？", "！", ";", ". ", "? ", "! "):
                idx = text.find(sep)
                if 0 < idx <= 16:
                    text = text[:idx]
                    break
            title = (text[:16]).strip()
        if title:
            self._sessions.update(session_id, title=title)

    # ---------- 持久化一轮对话 ----------
    @staticmethod
    def _decode_user_turn(content: str) -> tuple[str, list[str]]:
        """把存储的用户消息还原为 (文本, 图片dataURL列表)，供「重新生成」复用最后一轮用户输入。"""
        decoded = _decode_message_content(content)
        if isinstance(decoded, list):
            text_parts = [b.get("text", "") for b in decoded if isinstance(b, dict) and b.get("type") == "text"]
            images = [
                b["image_url"]["url"]
                for b in decoded
                if isinstance(b, dict) and b.get("type") == "image_url" and b.get("image_url", {}).get("url")
            ]
            return ("\n".join(p for p in text_parts if p) or "", images)
        return (decoded or "", [])

    def get_last_user_turn(self, session_id: int) -> tuple[str, list[str]] | None:
        """取当前会话最后一条用户消息的 (文本, 图片列表)；无则返回 None（不可重新生成）。"""
        with self._Session() as db:
            row = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            if row is None:
                return None
            return self._decode_user_turn(row.content)

    def trim_last_assistant(self, session_id: int) -> tuple[str, list[str]] | None:
        """「重新生成」前置：裁掉当前会话最后一轮的 assistant 回复及其 tool_runs/trace，
        返回该轮用户消息的 (文本, 图片列表) 供重跑复用；无用户消息则返回 None。

        2026-09-07 修复（跨表 id 比较缺陷）：Message / ToolRun / Trace 是三张
        **各自独立自增**的表，彼此无 turn_id 外键，因此不能用「Xxx.id > last_user.id」
        做跨表比较——它既可能一条都不命中（漏删 → 被重生成那轮的工具记录残留、
        用量页 num_tool_runs 虚高、Trace 页出现孤儿记录），也可能命中更早轮次（误删）。

        现改为按「本轮起点时间」清理：
        - Message 仍在同表内按 id 比较（同表同序，语义正确）；
        - ToolRun / Trace 按 created_at >= 最后一条 user 消息的 created_at 判定归属本轮
          （同一会话内写入时序单调递增，语义成立）。
        """
        with self._Session() as db:
            last_user = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.id.desc())
                .first()
            )
            if last_user is None:
                return None
            db.query(Message).filter(Message.session_id == session_id, Message.id > last_user.id).delete(
                synchronize_session=False
            )
            # 本轮起点 = 最后一条 user 消息的创建时间。
            # created_at 为 None 仅见于历史脏数据 → 退化为「不清理」：宁可残留也不误删其它轮次。
            _cut = getattr(last_user, "created_at", None)
            if _cut is not None:
                db.query(ToolRun).filter(
                    ToolRun.session_id == session_id, ToolRun.created_at >= _cut
                ).delete(synchronize_session=False)
                db.query(Trace).filter(
                    Trace.session_id == session_id, Trace.created_at >= _cut
                ).delete(synchronize_session=False)
            db.commit()
            return self._decode_user_turn(last_user.content)

    def persist_turn(
        self,
        session_id: int,
        user_content: str | list[dict],
        history: list[ChatMessage],
        events: list[Any],
        final_text: str,
        model: str,
        cost_override: float | None = None,
        total_tokens_override: int | None = None,
        # 精确分项用量（2026-09-16，问题 2 修复）：优先于 total_tokens_override。
        # 起因：`total_tokens_override` 只能承载一个总数，落库时被写成 `pt=总数, ct=0`
        # → 用量页丢失输入/输出拆分；且旧代码用 `is not None` 判断，
        # **`0` 也会命中**，于是「精确值 0」反而覆盖掉了本地粗算兜底
        # （实测：正常走完 Agent 循环的轮次写 0，没发 done 的才落粗算值）。
        prompt_tokens_override: int | None = None,
        completion_tokens_override: int | None = None,
        # 本轮引用（`[{"id","page","doc_id","text"}]`，由 routes_search 从 retrieve 结果算出）。
        # 落进最终回答事件的 data.citations，供「溯源率」指标与历史回放使用（2026-09-16）。
        citations: list[dict] | None = None,
        # AgentLoop 回传的真实迭代步数 / 累计延迟（来自 done 事件）；
        # 非循环路径（技能/兜底）无此数据时留 None，回落 thinking 事件数 / 0。
        steps: int | None = None,
        latency_ms: int | None = None,
        # 「重新生成」：True 时该用户消息已存在于 DB，仅追加新的 assistant 回复，
        # 不再重复新增 user 消息，避免历史里出现重复提问。
        regenerate: bool = False,
        # 消息级元数据（JSON）：承载无效回复结构化 notice / invalid 标记，
        # 供前端从历史回填「打断提醒」卡片（见 routes_search._build_notice）。仅作用于 assistant 消息。
        meta: dict | None = None,
    ) -> None:
        # token 粗算（用于用量面板；fake 模型成本恒 0）。若 AgentLoop 回传精确成本则优先采用，
        # 保证"用量页成本与实际调用一致"（见 阶段D 成本熔断 + trace 精确化）。
        # 多模态：user_content 为内容块列表时按 文本+图片(384/张) 计。
        pt = _rough_tokens(user_content) + sum(_rough_tokens(m.content) for m in history)
        ct = _rough_tokens(final_text)

        # ① 先定 token 分项（2026-09-16 问题 2 修复）：只接受**正数**——
        #    `0` 表示"上游没有这个数"，不是"这轮真的是 0 token"。
        exact_pt = prompt_tokens_override if (prompt_tokens_override or 0) > 0 else None
        exact_ct = completion_tokens_override if (completion_tokens_override or 0) > 0 else None
        if exact_pt is not None or exact_ct is not None:
            if exact_pt is not None:
                pt = exact_pt
            if exact_ct is not None:
                ct = exact_ct
            # 分项不全（只有总数）时，用总数补齐缺失项，避免总量对不上
            total = total_tokens_override if (total_tokens_override or 0) > 0 else None
            if total is not None and len([x for x in (exact_pt, exact_ct) if x is not None]) == 1:
                if exact_pt is not None:
                    ct = max(total - pt, 0)
                else:
                    pt = max(total - ct, 0)
        elif (total_tokens_override or 0) > 0:
            # 遗留路径：上游只给总数（无分项）→ 全部记在 prompt 侧，并在日志里留痕，
            # 便于发现"还有调用方没升级到分项口径"。
            pt, ct = total_tokens_override, 0
            logger.debug("persist_turn：仅收到 total_tokens_override=%d，分项缺失", total_tokens_override)

        # ② 再算成本 —— **必须在分项定下来之后**。旧顺序（先算 cost 再覆盖 token）
        #    会用"覆盖前的粗算 token"去计价，实测差 1000 倍（1000 tokens → 3e-06 而非 0.003）。
        #    取值规则：override 为假值（None / 0）时回退本地估算 —— 上游因"拿不到用量"回传的 0
        #    不是真值；回退是安全的（未定价 / 假模型两边都算 0）。
        cost = cost_override if cost_override else estimate_cost(model, pt, ct)
        if cost_override is not None and not cost_override and cost:
            # 0 成本但本地能算出值 → 记一条，便于发现"上游没传用量"的回归
            logger.warning(
                "本轮成本上游回传 0，已按本地估算记账（model=%s, pt=%d, ct=%d, cost=%.6f）",
                model, pt, ct, cost,
            )

        # 配对 tool_call / tool_result 形成工具记录
        tool_calls: dict[str, dict] = {}
        tool_runs: list[ToolRun] = []
        for ev in events:
            if ev.type == "tool_call":
                tool_calls[ev.data["call_id"]] = ev.data
            elif ev.type == "tool_result":
                call = tool_calls.get(ev.data["call_id"], {})
                out = ev.data.get("output", {})
                summary = ""
                if isinstance(out, dict) and "results" in out:
                    summary = f"命中 {len(out['results'])} 段"
                tool_runs.append(ToolRun(
                    session_id=session_id,
                    tool_name=call.get("name", "tool"),
                    args_json=json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    result_summary=summary,
                    ok=bool(ev.data.get("ok", True)),
                    latency_ms=0,
                ))

        events_payload = [{"idx": i, "type": e.type, "data": e.data} for i, e in enumerate(events)]
        # 引用溯源落库（2026-09-16，问题 3 修复）：把本轮引用写进**最终回答事件**的 data。
        # 为什么必须写：`citation_rate`（评测看板的「溯源率」）判据是
        # `message.data.citations`，而此前**没有任何生产者写过这个字段**（引用只走 SSE 给前端）
        # → 指标恒为 0，用户看到"溯源率 0%"会以为引用功能坏了。
        # 注意：`e.data` 与调用方传入的事件对象共享同一 dict 引用，这里**新建 dict** 再附加，
        # 避免悄悄改动调用方手里的事件（隐藏副作用）。
        if citations:
            target = None
            for item in events_payload:
                if item["type"] == "message" and (item["data"] or {}).get("content"):
                    target = item  # 取最后一个"完整回答"事件（delta 片段不算回答）
            if target is not None:
                target["data"] = {**(target["data"] or {}), "citations": citations}
        trace_id = f"s{session_id}-{int(__import__('time').time())}"
        # 优先用 AgentLoop 回传的真实迭代步数；非循环路径（技能/兜底）无 steps 时回退 thinking 事件数
        total_steps = steps if steps is not None else sum(1 for e in events if e.type == "thinking")

        # B2-X1（2026-10-01）：引用**也要落进消息级 meta**，供"刷新 / 切换会话后再回来"回填引用卡片。
        # 此前 citations 只进 Trace 的事件（服务 `citation_rate` 指标），messages 侧**毫无痕迹**
        # ⇒ 前端只能把历史消息的 `citations` 写死为 `[]` ⇒ **当轮看得到、刷新即消失**
        #   （典型的"指标有数、用户看不到"）。
        # 放 meta_json 而非新列：与既有 `notice`/`model` 元数据同处、**零 DDL**（见下 meta_payload 注释）。
        if citations:
            meta = {**(meta or {}), "citations": citations}

        write_with_retry(lambda: self._persist_turn_once(
            session_id, user_content, final_text, tool_runs,
            events_payload, pt, ct, cost, trace_id, total_steps, latency_ms or 0,
            regenerate=regenerate, meta=meta, model=model,
        ))

    def _persist_turn_once(
        self,
        session_id: int,
        user_content: str | list[dict],
        final_text: str,
        tool_runs: list[ToolRun],
        events_payload: list[dict],
        pt: int,
        ct: int,
        cost: float,
        trace_id: str,
        total_steps: int,
        latency_ms: int = 0,
        regenerate: bool = False,
        meta: dict | None = None,
        model: str = "",
    ) -> None:
        with self._Session() as db:
            # 多模态：块列表经 _MM_PREFIX 标记 JSON 入库，load_history 还原为内容块，
            # 保证切换模型/刷新后图片上下文不丢失（历史消息数据一致性）。
            # 「重新生成」：该用户消息已存在，跳过新增，仅追加新的 assistant 回复，避免重复落库。
            if not regenerate:
                db.add(Message(session_id=session_id, role="user", content=_encode_user_content(user_content)))
            # 记录本轮**实际生效的模型**（2026-09-16，体检问题 7）：
            # 此前 messages 只有 token/cost 而**没有模型**，成本异常时无法回答
            # "这笔钱花在哪个模型上"（用量页显示的 model 来自会话 config，是"该用什么"，
            # 不是"实际用了什么"——回退/降级路径下两者会不一致）。
            # 落进 meta_json：零 DDL、与既有 invalid/notice 元数据同处，SQL 侧可用
            # json_extract(meta_json,'$.model') 按模型对账。
            meta_payload = {**(meta or {})}
            if model:
                meta_payload["model"] = model
            db.add(Message(session_id=session_id, role="assistant", content=final_text,
                           prompt_tokens=pt, completion_tokens=ct, cost=cost,
                           meta_json=json.dumps(meta_payload, ensure_ascii=False) if meta_payload else "{}"))
            for tr in tool_runs:
                db.add(tr)
            db.add(Trace(
                session_id=session_id,
                trace_id=trace_id,
                events_json=json.dumps(events_payload, ensure_ascii=False),
                total_steps=total_steps,
                total_cost=cost,
                # 真实延迟（来自 AgentLoop 累计的 LLM 请求耗时）；非循环路径回落 0
                total_latency_ms=latency_ms,
            ))
            db.commit()

    def persist_user_message(self, session_id: int, user_content: str | list[dict]) -> None:
        """**单独落库用户提问**（N13 修复，2026-09-12）。

        背景：整轮落库（`_persist_turn`）在流结束后一次性写入 user+assistant。中断路径原先
        「**无已生成内容则不落库**」→ 用户发出提问后模型尚未产出任何 delta 就点停止时，
        **整轮（含提问本身）全部消失**，用户被迫重复输入（用户实测反馈）。

        本方法保证「**提问永不丢失**」：中断且无内容时只落 user 一条；有部分内容时仍走
        `persist_turn`（user + 部分 assistant + meta.interrupted）。
        """
        with self._Session() as db:
            db.add(Message(session_id=session_id, role="user", content=_encode_user_content(user_content)))
            db.commit()

    def count_usage(self, session_id: int) -> dict:
        """会话用量聚合（单条会话）。

        审查 P2-2：改用 SQL 聚合（COUNT/SUM/MAX）一次取回，避免把该会话全部消息/工具记录
        拉到 Python 内存再聚合（会话消息多时产生内存放大 + 每会话 2 次全量查询）。
        跨会话的批量聚合由路由层负责（未来可加 count_usage_all 一次 GROUP BY 多会话）。
        """
        from sqlalchemy import func

        with self._Session() as db:
            msg = db.query(
                func.count(Message.id),
                func.coalesce(func.sum(Message.prompt_tokens), 0),
                func.coalesce(func.sum(Message.completion_tokens), 0),
                func.coalesce(func.sum(Message.cost), 0.0),
                func.max(Message.created_at),
            ).filter(Message.session_id == session_id).first()
            run = db.query(func.count(ToolRun.id)).filter(ToolRun.session_id == session_id).first()
            num_msg, pt, ct, cost, last = msg
            num_runs = run[0] if run else 0
            return {
                "num_messages": int(num_msg or 0),
                "num_tool_runs": int(num_runs or 0),
                "prompt_tokens": int(pt or 0),
                "completion_tokens": int(ct or 0),
                "cost": round(float(cost or 0.0), 6),
                "last_at": last,
            }

    def count_usage_all(self) -> dict[int, dict]:
        """跨会话批量用量聚合（O-09，2026-10-01）。

        此前 `/usage` 路由对每个会话各调一次 `count_usage` ⇒ 每会话 2 条聚合 SQL
        （N+1，随会话数线性放大）。现改为**两条 GROUP BY** 一次取回全部会话，
        字段口径与 `count_usage` 逐字段一致（含空会话的零值兜底，由路由层补默认）。
        """
        from sqlalchemy import func

        out: dict[int, dict] = {}
        with self._Session() as db:
            msg_rows = db.query(
                Message.session_id,
                func.count(Message.id),
                func.coalesce(func.sum(Message.prompt_tokens), 0),
                func.coalesce(func.sum(Message.completion_tokens), 0),
                func.coalesce(func.sum(Message.cost), 0.0),
                func.max(Message.created_at),
            ).group_by(Message.session_id).all()
            run_rows = db.query(
                ToolRun.session_id, func.count(ToolRun.id)
            ).group_by(ToolRun.session_id).all()
        runs = {sid: n for sid, n in run_rows}
        # 会话可能「只有工具调用、无消息」——只按 Message GROUP BY 会漏掉它们，
        # 而 count_usage(sid) 会返回 num_tool_runs>0 ⇒ 口径不一致。以两边会话并集为准。
        out = {sid: {"num_messages": 0, "num_tool_runs": int(n), "prompt_tokens": 0,
                     "completion_tokens": 0, "cost": 0.0, "last_at": None}
               for sid, n in runs.items()}
        for sid, num_msg, pt, ct, cost, last in msg_rows:
            cur = out.setdefault(sid, {"num_messages": 0, "num_tool_runs": 0, "prompt_tokens": 0,
                                       "completion_tokens": 0, "cost": 0.0, "last_at": None})
            cur.update({
                "num_messages": int(num_msg or 0),
                "num_tool_runs": int(runs.get(sid, 0)),
                "prompt_tokens": int(pt or 0),
                "completion_tokens": int(ct or 0),
                "cost": round(float(cost or 0.0), 6),
                "last_at": last,
            })
        return out
