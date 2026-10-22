"""PaperMate · 运行时设置路由（前端填 API key / 默认模型，docs/04 §4.6 + 本次需求）

- GET  /api/v1/settings  → 返回脱敏后的当前设置（不含真实 key，含各厂商可选模型）
- PUT  /api/v1/settings  → 保存默认模型与各家 API key/base_url/model，落盘并注入 os.environ
- POST /api/v1/settings/test → 用运行时 key 对指定厂商做一次非流式调用，返回 ok/error（前端「测试连接」用）
- GET  /api/v1/settings/balances → 各厂商账户余额（DeepSeek 支持 key 直查；Qwen/OpenAI 说明不可查原因）
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from lightcore.llm.base import ChatMessage, Role, _rough_tokens
from lightcore.llm.router import (
    MODEL_TO_PROVIDER,
    PRESET_ENDPOINTS,
    ModelEndpoint,
    OAICompatibleLLM,
)

from ..memory_hub import reset_memory_manager
from ..runtime_settings import get_state, public_view, rules_token_limit, update
from ..schemas import (
    FigureInstallStatusOut,
    InventoryOut,
    RagInstallReq,
    RagInstallStatusOut,
    RagModelsOut,
    SettingsOut,
    SettingsTestOut,
    WebSearchTestOut,
)
from ..services import figure_deps
from ..services.model_installer import (
    AlreadyRunning,
    ModelInstallError,
    get_model_installer,
)

router = APIRouter()


def _mask_secrets(text: str) -> str:
    """把已配置的 API 密钥替换为掩码（2026-09-06 安全加固）。

    第三方 SDK/httpx 的异常文本可能包含完整请求（含 Authorization 头或带 key 的 URL），
    连接测试把 `str(e)[:300]` 直通前端会泄露密钥片段。此处用运行时已配置的所有密钥做子串替换。
    """
    try:
        providers = get_state().get("providers") or {}
    except Exception:  # noqa: BLE001  脱敏失败不能放大异常
        return text
    for p in providers.values():
        key = (p or {}).get("api_key")
        if key and len(key) >= 8 and key in text:
            text = text.replace(key, "***masked***")
    return text


class ProviderIn(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None  # 具体模型 id（如 deepseek-reasoner）；空→厂商默认


class SettingsUpdate(BaseModel):
    default_model: str | None = None
    dedupe_warning: bool | None = None  # 重复上传提醒开关（2026-08-20）
    # Phase B 功能开关（docs/统一方案 阶段B）：memory=跨会话记忆，summary=一键总结
    features: dict[str, bool] | None = None
    # 图片内容增强（ER-03 Level 2）：{"enabled": bool, "methods": ["ocr","table","vlm"]}
    figure_enrich: dict | None = None
    providers: dict[str, ProviderIn] | None = None
    agent_rules: str | None = None  # 本地长期记忆：用户设定的全局规则/偏好（直接影响所有会话）
    # 记忆向量引擎（2026-08-29 B 档）："hash" 轻量哈希（默认） / "bge-m3" 真实语义向量
    memory_embedder: str | None = None
    # RAG 检索底座（2026-09-01 设置页可切换）：rag_embedder=fake|real；use_bge_sparse / use_jieba 为 bool
    rag_embedder: str | None = None
    use_bge_sparse: bool | None = None
    use_jieba: bool | None = None
    # 重排推理加速（T-03，2026-09-24）：fp16 半精度 + 候选截断长度（0 = 不限长 = 改动前行为）
    rag_reranker_fp16: bool | None = None
    rag_reranker_max_length: int | None = None
    # 单次请求上下文预算（★ 2026-10-22：前端一直在发这个字段，但此前**后端模型里没有它**
    #   ⇒ Pydantic 静默丢弃 ⇒ 设置页调预算"改了没反应"。属典型的死配置项。）
    max_tokens_budget: int | None = None
    # True（默认）= 按模型窗口自动推断 clamp(窗口×0.25, 12k, 100k)；False = 用上面的显式值
    max_tokens_budget_auto: bool | None = None


class SettingsTestReq(BaseModel):
    provider: str | None = None  # 不传则测当前默认厂商


@router.get("/settings", response_model=SettingsOut)
def get_settings() -> dict:
    return public_view(get_state())


@router.put("/settings", response_model=SettingsOut)
def put_settings(body: SettingsUpdate, request: Request) -> dict:
    payload: dict = {}
    if body.default_model is not None:
        payload["default_model"] = body.default_model
    if body.dedupe_warning is not None:
        payload["dedupe_warning"] = body.dedupe_warning
    if body.features is not None:
        payload["features"] = body.features
    if body.figure_enrich is not None:
        payload["figure_enrich"] = body.figure_enrich
    # ---- A8/C8（2026-09-28）：长期规则的**长度上限** ----
    # 为什么单独限它、且比用户输入更严：`agent_rules` 会被**每一轮、每个会话**拼进
    # system prompt（`chat_service.build_agent`）⇒ 它的成本是「每次请求都要付」，
    # 而用户输入只付一次。所以比例取 `RULES_BUDGET_RATIO`（10%）。
    # 口径同源：`_rough_tokens`（与预算体系同一把尺子），阈值由**当前生效的**
    # `max_tokens_budget` 派生，用户调高预算时上限自动跟着放大。
    # 这里用 400 而非聊天流的"软拦截"：它是**设置保存**，用户在界面上能直接看到错误并改小，
    # 不需要（也不该）借道聊天流的 notice 机制。
    if body.agent_rules is not None:
        _rules_tokens = _rough_tokens(body.agent_rules)
        _rules_limit = rules_token_limit()
        if _rules_tokens > _rules_limit:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"长期规则过长：约 {_rules_tokens:,} tokens（{len(body.agent_rules):,} 字符），"
                    f"超过上限 {_rules_limit:,} tokens。"
                    "长期规则会在**每一次请求**里注入，一旦占满预算，所有会话都会开始报错。"
                    "建议精简为几条关键偏好（如「回答用中文」「引用必须标页码」），"
                    "或把具体规范放到每次提问里说明。"
                ),
            )
        payload["agent_rules"] = body.agent_rules
    # 单次上下文预算：显式值夹到 [12k, 300k]（下限防"设成 1k 直接不可用"，上限对应 1M 窗口的保守用法）
    if body.max_tokens_budget is not None:
        payload["max_tokens_budget"] = max(12_000, min(int(body.max_tokens_budget), 300_000))
    if body.max_tokens_budget_auto is not None:
        payload["max_tokens_budget_auto"] = bool(body.max_tokens_budget_auto)

    if body.providers is not None:
        payload["providers"] = {
            name: prov.model_dump(exclude_none=True) for name, prov in body.providers.items()
        }
    if body.memory_embedder is not None:
        payload["memory_embedder"] = body.memory_embedder
    if body.rag_embedder is not None:
        payload["rag_embedder"] = body.rag_embedder
    if body.use_bge_sparse is not None:
        payload["use_bge_sparse"] = body.use_bge_sparse
    if body.use_jieba is not None:
        payload["use_jieba"] = body.use_jieba
    if body.rag_reranker_fp16 is not None:
        payload["rag_reranker_fp16"] = body.rag_reranker_fp16
    if body.rag_reranker_max_length is not None:
        payload["rag_reranker_max_length"] = body.rag_reranker_max_length
    # ★ 前置校验（2026-06-06）：切 real 之前必须确认重依赖可用，否则用户会"保存成功"，
    #   直到第一次检索才炸出 `ModuleNotFoundError: No module named 'FlagEmbedding'`（500，且难懂）。
    #   桌面首启刻意不装 torch 等（省 4GB）⇒ 这条校验是那个设计能成立的前提。
    if body.rag_embedder == "real":
        from ..services import rag_deps

        deps = rag_deps.probe()
        if not deps.ready:
            reason = rag_deps.describe_target().get("reason")
            tail = f"（{reason}）" if reason else ""
            raise HTTPException(
                status_code=400,
                detail=(
                    "尚未安装真实语义检索所需的运行环境"
                    f"（缺 {'、'.join(m.split('：')[0] for m in deps.missing)}，约 4GB）。"
                    "请先在设置页「语义检索」卡片点『下载模型』，它会先补装依赖再下载权重。"
                    f"若按钮不可用{tail}。"
                ),
            )
    out = update(payload)
    # 切换记忆向量引擎：丢弃共享单例，下次取用时按新引擎重建并重算已有记忆的向量
    if body.memory_embedder is not None:
        reset_memory_manager()
    # T-03：重排档位变更必须连**管线缓存**一起清 —— 否则已缓存的管线仍用旧档位
    # （get_pipeline 命中缓存不校验签名）→ 设置"改了没反应"的静默缺陷。
    # ★ 2026-10-04（O4）：**检索侧的三个配置同样要清**，此前漏掉 ⇒ 改设置后静默不生效：
    #   - `rag_embedder`：换了嵌入模型，但内存里的管线仍用旧 embedder；
    #   - `use_bge_sparse`：稀疏路开关变了，管线仍按旧路数检索；
    #   - `use_jieba`：分词器变了，BM25 结果随之改变。
    # ⚠️ 之所以只能靠"清缓存"而不能靠签名：**`get_pipeline` 命中 `_pipelines` 就直接 return**
    #   （`document_service.py` 内 `if doc_id in self._pipelines`），**根本不校验签名** ——
    #   签名只用于 `load_state` 的磁盘索引判定，**管不到内存缓存这一层**。
    # ⇒ 所以内存缓存必须显式清理；这与 O5（磁盘索引签名纳入权重）**互补、缺一不可**。
    if any(v is not None for v in (
        body.rag_embedder,
        body.use_bge_sparse,
        body.use_jieba,
        body.rag_reranker_fp16,
        body.rag_reranker_max_length,
    )):
        doc_svc = getattr(request.app.state, "doc_svc", None)
        if doc_svc is not None:
            doc_svc.reset_reranker_pipelines()
    return out


@router.post("/settings/test", response_model=SettingsTestOut)
async def test_settings(body: SettingsTestReq | None = None) -> dict:
    """用运行时已保存的 key 对指定厂商做一次非流式调用，验证 key 有效、端点可达。

    返回 {ok, provider, model, reply?, error?}；不抛 HTTP 异常，便于前端直接展示结果。
    """
    provider = (body.provider if body else None) or get_state()["default_model"]
    if provider == "fake" or provider not in PRESET_ENDPOINTS and provider not in MODEL_TO_PROVIDER:
        return {"ok": False, "provider": provider, "error": "fake 为本地演示，无需测试；或未识别的厂商"}
    if provider in MODEL_TO_PROVIDER:
        provider = MODEL_TO_PROVIDER[provider]
    # 2026-09-08 修复 I3：MODEL_TO_PROVIDER 保留了历史/隐藏模型（如 gpt-4o → openai），
    # 但 PRESET_ENDPOINTS 只含**已接入**厂商。缺少这层校验时 `PRESET_ENDPOINTS[provider]`
    # 直接 KeyError → HTTP 500（旧会话遗留 default_model、或前端直接传旧模型 id 时必现）。
    if provider not in PRESET_ENDPOINTS:
        return {"ok": False, "provider": provider,
                "error": f"厂商 {provider} 暂未接入连通性测试（已支持：{'、'.join(PRESET_ENDPOINTS)}）"}
    ep = PRESET_ENDPOINTS[provider]
    pset = get_state()["providers"].get(provider, {}) or {}
    if not pset.get("api_key"):
        return {"ok": False, "provider": provider, "error": f"未配置 {provider} 的 API Key（请先在前端设置页填写）"}
    concrete = pset.get("model") or ep.default_model
    llm = OAICompatibleLLM(
        ModelEndpoint(ep.name, pset.get("base_url") or ep.base_url, ep.api_key_env, concrete),
        api_key=pset["api_key"],
    )
    try:
        resp = await llm.chat(
            [ChatMessage(role=Role.USER, content="用一句话确认你在线，并说出你的模型名。")],
            temperature=0.2,
        )
        return {"ok": True, "provider": provider, "model": concrete, "reply": resp.message.content[:160]}
    except Exception as e:  # noqa: BLE001  # 网络/401/429/5xx 等，统一降级为 ok:False
        return {"ok": False, "provider": provider, "model": concrete,
                "error": f"{type(e).__name__}：{_mask_secrets(str(e)[:300])}"}


@router.post("/settings/test-web-search", response_model=WebSearchTestOut)
async def test_web_search_settings(body: SettingsTestReq | None = None) -> dict:
    """探测『大模型原生联网搜索』是否可用（大模型联网的连通性测试，满足 LLM 模块硬约束）。

    返回 {ok, supported, provider, model, reply?, error?}：
    - supported=false：当前厂商/模型不支持原生搜索（当前仅 Qwen / DashScope 已验证可用），
      提示用户改用 Qwen 或在设置页切换模型；
    - supported=true 且 ok=true：实测 chat_web_search 成功，reply 为模型带引用回答片段；
    - supported=true 但 ok=false：能力声明支持但实测失败（如 key 无效 / 网络异常）。
    不抛 HTTP 异常，便于前端直接展示。
    """
    provider = (body.provider if body else None) or get_state()["default_model"]
    if provider == "fake" or (provider not in PRESET_ENDPOINTS and provider not in MODEL_TO_PROVIDER):
        return {"ok": False, "supported": False, "provider": provider,
                "error": "fake 为本地演示，无需测试；或未识别的厂商"}
    if provider in MODEL_TO_PROVIDER:
        provider = MODEL_TO_PROVIDER[provider]
    # 同 /settings/test（2026-09-08 修复 I3）：MODEL_TO_PROVIDER 的隐藏模型可能映射到
    # PRESET_ENDPOINTS 中不存在的厂商，直接取下标会 KeyError → 500。
    if provider not in PRESET_ENDPOINTS:
        return {"ok": False, "supported": False, "provider": provider,
                "error": f"厂商 {provider} 暂未接入（已支持：{'、'.join(PRESET_ENDPOINTS)}）"}
    ep = PRESET_ENDPOINTS[provider]
    pset = get_state()["providers"].get(provider, {}) or {}
    if not pset.get("api_key"):
        return {"ok": False, "supported": False, "provider": provider,
                "error": f"未配置 {provider} 的 API Key（请先在前端设置页填写）"}
    concrete = pset.get("model") or ep.default_model
    llm = OAICompatibleLLM(
        ModelEndpoint(ep.name, pset.get("base_url") or ep.base_url, ep.api_key_env, concrete),
        api_key=pset["api_key"],
    )
    if not llm.supports_native_web_search():
        return {"ok": False, "supported": False, "provider": provider, "model": concrete,
                "error": f"厂商 {provider} 当前不支持原生联网搜索（仅 qwen / DashScope 已验证可用）；"
                         f"请确认已将『默认模型』切换为 Qwen，或在设置页选择 Qwen 模型。"}
    try:
        resp = await llm.chat_web_search(
            [ChatMessage(role=Role.USER,
                         content="用一句话告诉我今天值得关注的一条 AI 领域新闻，并附上来源。")],
            temperature=0.3,
        )
        return {"ok": True, "supported": True, "provider": provider, "model": concrete,
                "reply": (resp.message.content or "")[:200]}
    except Exception as e:  # noqa: BLE001  # 原生搜索实测失败（如 vision 变体不支持 / 4xx / 网络）
        return {"ok": False, "supported": True, "provider": provider, "model": concrete,
                "error": f"{type(e).__name__}：{_mask_secrets(str(e)[:300])}"}


async def _deepseek_balance(api_key: str) -> dict:
    """DeepSeek /user/balance：key 直查余额（官方辅助端点）。"""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code == 200:
            j = resp.json()
            infos = j.get("balance_infos") or []
            total = sum(float(i.get("total_balance", 0) or 0) for i in infos)
            cur = sum(float(i.get("granted_balance", 0) or 0) for i in infos)
            return {"available": True, "total_balance": round(total, 4),
                    "granted_balance": round(cur, 4), "currency": (infos[0].get("currency") if infos else "CNY")}
        return {"available": False, "reason": f"余额查询失败（HTTP {resp.status_code}）"}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"余额查询异常：{type(e).__name__}"}


@router.get("/settings/balances", response_model=dict[str, dict])
async def get_balances() -> dict:
    """各厂商账户余额（对话页模型选择器展示）。

    - deepseek：/user/balance key 直查（可用）；
    - openai：credit_grants 需组织级授权，未配置则说明不可查；
    - qwen：DashScope key 无法查询账户余额（需阿里云 AK/SK + GetBalance），说明不可查。
    """
    state = get_state()
    out: dict = {}
    for provider in PRESET_ENDPOINTS:
        pset = state["providers"].get(provider, {}) or {}
        key = pset.get("api_key") or ""
        if not key:
            out[provider] = {"available": False, "reason": f"未配置 {provider} 的 API Key"}
            continue
        if provider == "deepseek":
            out[provider] = await _deepseek_balance(key)
        elif provider == "openai":
            # credit_grants 需要组织 Key + 授权；用个人 key 大概率 401，容错说明
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    r = await client.get(
                        "https://api.openai.com/v1/dashboard/billing/credit_grants",
                        headers={"Authorization": f"Bearer {key}"},
                    )
                if r.status_code == 200:
                    j = r.json()
                    out[provider] = {"available": True,
                                     "total_balance": round(float(j.get("total_granted", 0) or 0), 2),
                                     "granted_balance": round(float(j.get("total_available", 0) or 0), 2),
                                     "currency": "USD"}
                else:
                    out[provider] = {"available": False,
                                     "reason": "OpenAI 余额查询需组织级授权（个人 Key 通常不可用）"}
            except Exception:  # noqa: BLE001
                out[provider] = {"available": False, "reason": "OpenAI 余额查询失败"}
        else:  # qwen
            out[provider] = {"available": False,
                             "reason": "Qwen 余额查询需阿里云 AK/SK（当前仅配置了 DashScope Key），可在阿里云控制台查看"}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 真实语义检索模型的**显式安装**（2026-10-05 桌面客户端需求）
#
# 语义：App 装完默认是 fake（零下载）⇒ 用户在设置页**主动**点下载 ⇒ 下载完
#      **也不自动启用**（切成 real 会让既有索引失效，必须"手动启用 + 重建索引"，
#       这个代价要让用户在知情时自己触发）。故这里只管下载，不改任何运行时配置。
# 线程：下载跑在**后台线程**（数 GB / 分钟级），路由立即返回；前端轮询 status。
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/settings/rag/models", response_model=RagModelsOut)
def rag_models() -> dict:
    """语义检索模型清单 + 本机安装状态 + 是否启用 + 磁盘余量。"""
    return get_model_installer().models()


@router.post("/settings/rag/models/install", response_model=RagInstallStatusOut)
def install_rag_models(body: RagInstallReq | None = None) -> dict:
    """触发后台下载（立即返回，前端轮询 `/install/status`）。

    - 400：未知模型名 / 磁盘不足 / 仓库不可达等（消息可直接展示）；
    - 409：已有任务在跑（前端应先禁用按钮或提示先取消）。
    """
    keys = body.models if body else None
    try:
        return get_model_installer().install(keys)
    except AlreadyRunning as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ModelInstallError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/settings/rag/models/install/cancel", response_model=RagInstallStatusOut)
def cancel_rag_models_install() -> dict:
    """取消当前下载（在**下一个文件开始前**生效，已缓存部分会保留以便续传）。"""
    try:
        return get_model_installer().cancel()
    except ModelInstallError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/settings/rag/models/install/status", response_model=RagInstallStatusOut)
def rag_models_install_status() -> dict:
    """轮询安装进度。首次安装的初始状态为 `idle`。"""
    return get_model_installer().task()


# --------------------------------------------------------------------------- #
# 图片增强（Level 2）依赖：**按需安装**（2026-10-09 新增）
#
# 背景：`[figure]` extra（paddleocr + paddlepaddle + transformers + torch + pillow）
# 与 `[rag]` 分离（pyproject 注释：避免拖重核心 RAG 安装）⇒ 需要独立的按需安装通道。
# 此前应用内**没有任何入口** ⇒ 用户打开「图片增强」后 OCR 永远不生效（实测反馈）。
# --------------------------------------------------------------------------- #
@router.get("/settings/figure-enrich/deps", response_model=dict)
def figure_enrich_deps() -> dict:
    """**实时**探测图片增强三能力（`{ocr, table, vlm}`，不做任何缓存）。

    ★ 为什么要独立端点：探测结果原先只随 `/settings` 一次性返回，而前端设置上下文
    **只在 SPA 挂载时拉一次** ⇒ 装完依赖后提示不更新（实测「torch 已装却显示未装」）。
    设置页进入时 / 安装完成后调用本端点即可拿到当前真值。
    """
    return figure_deps.probe()


@router.post("/settings/figure-enrich/install", response_model=FigureInstallStatusOut)
def install_figure_deps() -> dict:
    """触发后台安装（立即返回，前端轮询 `/install/status`）。

    - 409：已有任务在跑（前端应先禁用按钮或提示先取消）；
    - 安装内容：`uv sync --extra figure --frozen --inexact`
      （2026-10-09 实测：**下载 ≈150~250MB · 落盘 ≈559MB · 73.6s / 25 包**；
      若尚未「准备运行环境」，会一并补齐 torch 等重依赖 —— figure 与 rag 共享 torch）。
    """
    try:
        return figure_deps.get_figure_installer().install()
    except figure_deps.AlreadyRunning as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.post("/settings/figure-enrich/install/cancel", response_model=FigureInstallStatusOut)
def cancel_figure_deps_install() -> dict:
    """取消安装（在**当前 uv 命令返回后**生效，已下载部分保留在缓存以便重试）。"""
    try:
        return figure_deps.get_figure_installer().cancel()
    except figure_deps.AlreadyRunning as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/settings/figure-enrich/install/status", response_model=FigureInstallStatusOut)
def figure_deps_install_status() -> dict:
    """轮询图片增强依赖安装进度。"""
    return figure_deps.get_figure_installer().task()


@router.get("/settings/inventory", response_model=InventoryOut)
def inventory() -> dict:
    """本地数据管理：项目可用的工具与技能清单（含权限/门控状态，供设置页展示）。

    只读清单；启停通过 features 开关（工具按 feature 条件注册，见 chat_service.build_agent）。
    2026-09-05 口径对齐技能清单：**全量列出 + enabled 标记**（enabled=false = 功能关停、
    未注册给 Agent），不再把关停工具直接藏掉——用户可在设置页看到完整工具面貌。
    """
    from lightcore.tools.builtin.basic import register_builtin_tools
    from lightcore.tools.spec import ToolRegistry

    from ..skills import default_router

    reg = ToolRegistry()
    register_builtin_tools(reg)
    feat = get_state().get("features", {})
    # 领域工具全量注册（enabled=True 绑定 None stub——handler 仅 execute 时用到，清单场景安全），
    # 再按 feature 映射计算 enabled 标记；code_interpreter 工具实际受 SANDBOX 权限门控。
    from lightcore.tools.builtin.domain import (
        register_memory_tools,
        register_planning_tools,
        register_summary_tools,
    )
    register_memory_tools(reg, memory=None, enabled=True)
    register_summary_tools(reg, summary_svc=None, default_doc_ids=None, enabled=True)
    register_planning_tools(reg, llm=None, rag=None, enabled=True)

    tool_feature = {
        "memory_search": "memory",
        "memory_write": "memory",
        "summarize": "summary",
        "related_work": "summary",
        "plan_executor": "planning",
        "code_interpreter": "code_interpreter",
    }
    tools = [
        {"name": s.name, "description": s.description[:120], "permission": s.permission.value,
         "enabled": bool(feat.get(tool_feature.get(s.name, ""), True))}
        for s in reg.specs()
    ]
    skills = [
        {"name": s.name, "commands": s.commands, "description": s.description[:120],
         "feature": s.feature, "enabled": feat.get("skills", True) and (feat.get(s.feature, True) if s.feature else True),
         "icon": s.icon, "category": s.category}
        for s in default_router()._skills
    ]
    return {"tools": tools, "skills": skills,
            "permissions": ["safe", "network", "sandbox", "admin"],
            "sandbox_enabled": bool(feat.get("code_interpreter", False))}
