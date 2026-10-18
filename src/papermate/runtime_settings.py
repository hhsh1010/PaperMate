"""PaperMate · 运行时 Provider 设置（前端可填的 API key / base_url）。

- 持久化到 data/provider_settings.json（本地文件，已被 .gitignore 覆盖；不等同于「入库」DB，
  符合 docs/10 密钥不入库 NFR：密钥不进数据库，仅留本地文件，与 .env 等价但可由前端写入）。
- 进程内内存缓存 + 落盘；保存时同步注入 os.environ，确保 OAICompatibleLLM._api_key()
  在请求时即时读到最新 key，无需重启后端。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from lightcore.llm.router import MODEL_CONTEXT_WINDOW, PRESET_ENDPOINTS, PROVIDER_MODELS

from .config import get_settings

#: 配置读取函数别名（_default_state 内用于把 env 种子注入 runtime 默认；惰性调用，避免导入期副作用）
_cfg = get_settings

_lock = threading.Lock()
_state: dict | None = None


# 功能开关单一事实来源（docs/统一方案 阶段B/C / F3 修复）：所有后端 features 默认值统一引用此常量，
# 消除 runtime_settings(_default_state/public_view) / chat_service.build_agent / routes_search 的多处拷贝。
# 新增开关只需在此加一键，其余 3 处自动继承；用 dict(...) 拷贝避免被运行时 update 反向改写常量。
# - reproduction   : S8 复现检索（/复现 命令技能，抽取超参 + LLM 三态复现清单；受 SkillRouter 门控）
def _env_flag(name: str) -> bool:
    """读环境变量形式的布尔开关（默认关）。"""
    import os
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


DEFAULT_FEATURES: dict = {
    "memory": True, "summary": True, "multi_query": False,
    "code_interpreter": False, "skills": True,
    "planning": True, "dev_eval": False, "thinking": False,
    # 论文在线检索/导入（阶段 3）：**默认关**，因为它是主动联网能力。
    # 初始种子读环境变量（与 default_embedder 同一范式：只是首次种子，之后以状态文件为准）。
    "paper_search": _env_flag("PAPERMATE_ENABLE_PAPER_SEARCH"),
    "verify_reply": False, "diagram_enrich": False,
    # 失败轮次自评（2026-09-16 首次接线 Reflector）：**只在循环中止路径**（迭代耗尽/连续无进展）
    # 自评一次，若判 NEEDS_IMPROVEMENT 则带建议重生成一次。那一轮本来就要失败，
    # 换一个更好的收尾是划算的；正常收敛的轮次零额外调用。默认关（新增 LLM 调用，遵循开关纪律）。
    "reflect_on_abort": False,
    "deep_read": True, "glossary": True,
    "find_code": True, "research_gap": True,
    "experiment_design": True, "review": True,
    "reproduction": True,
    "revision": True,   # 改稿（/改稿，对照参考文献逐节给修改建议）
    "prompt_enhance": True,  # 提示词增强（输入框把口语需求改写为更明确的任务描述，单轮调用）
    "para_transform": True,  # 段落 AI 改写（改稿工作台：扩写/缩写/润色/去重，单轮调用）
}


def _default_state() -> dict:
    return {
        "default_model": "fake",  # fake | deepseek | openai | qwen（或具体 model id）
        "dedupe_warning": True,  # 重复上传提醒开关（管理员可配，2026-08-20）
        # 功能开关：控制「涉及调用大模型 API」的模块是否启用（docs/统一方案 阶段B/C）
        # - memory        : S3 跨会话记忆（memory 工具 + 轮末 reflect 反思写入 + 系统注记注入）
        # - summary       : S4 一键总结（/summarize /related-work + summarize/related_work 工具）
        # - multi_query   : P1-6 多查询融合检索增强（LLM 改写子查询，fake 时回退单查询）
        # - code_interpreter: S8 沙箱代码执行（强制 Docker 隔离，NFR 默认关闭）
        # - skills        : 命令触发的复合技能（/综述 /相关工作 /复现），受 SkillRouter 门控
        # - planning      : S9 复杂规划（plan_executor 工具 + /规划 命令技能）
        # - dev_eval      : S10 开发评测（/eval 命令技能，读检索基线 + 近期 Trace 出指标卡）
        # - thinking      : V4 思考模式（展示推理链；开启时不强制 tool_choice——官方 thinking 不兼容，
        #                    且带工具轮次需回传 reasoning_content，见 docs/实施记录 2026-08-22）
        # - verify_reply  : P1-2.1 回复逐条核验（L1 引用文本↔页码确定性校验 + L2 LLM 语义核验；
        #                    只标「存疑」不自动改写，L2 超时降级 skipped）
        # - diagram_enrich: P1 结构图识别（多模态模型把矢量结构图还原为 Mermaid + 一句话描述；
        #                    复用 deepseek VLM 通道，无 key 时静默降级为空）
        # - reflect_on_abort: 失败轮次自评（Reflexion 最便宜形态）——**只在循环中止路径**触发：
        #                    自评一次（max_tokens 400 / 超时 15s / 成本超支即跳过），
        #                    NEEDS_IMPROVEMENT 则带建议重生成一次（仍不给 tools，机制上保证收敛）；
        #                    正常收敛的轮次零额外调用。超时或异常一律沿用原兜底草稿，绝不阻断收尾。
        # - deep_read     : P0 单篇精读（/精读 命令技能，对唯一关联文档多轮检索 + LLM 解读）
        # - glossary      : P0 术语表（/术语表 命令技能，从关联文档抽取缩写/术语 + LLM 释义）
        # - find_code     : P1 找代码（/找代码 命令技能，抽取代码线索 + LLM 实现指南）
        # - research_gap  : P1 研究缺口（/研究缺口 命令技能，抽取局限/未来工作 + LLM 缺口分析）
        # - experiment_design: P2 实验设计（/实验设计 命令技能，抽取实验设置 + LLM 方案）
        # - review        : P2 审稿（/审稿 命令技能，抽取贡献/局限 + LLM 结构化审稿意见）
        "features": dict(DEFAULT_FEATURES),
        # 记忆向量引擎（2026-08-29 B 档）："hash"=轻量哈希（默认，零依赖、启动快，但无语义）；
        # "bge-m3"=真实语义向量（需 rag extra 与 ~2.2GB 权重，显著拖慢首次加载）。
        "memory_embedder": "hash",
        # ---- RAG 检索底座（2026-09-01 设置页可切换）----
        # rag_embedder: "fake"=哈希占位（秒级、零依赖、仅验证流程，无真实语义）；
        #               "real"=BGE-M3 真实语义（需 rag extra 与 ~2.2GB 权重，自动联动 Cross-Encoder 重排）。
        # 首次加载时以 config.py 的 env 值（PAPERMATE_DEFAULT_EMBEDDER 等）为种子，
        # 之后由前端设置页持久化覆盖（落盘 provider_settings.json）。
        # 注意：切换后必须「重建索引」才生效（get_pipeline 按文档缓存，不认配置变化）。
        "rag_embedder": (_cfg().default_embedder if _cfg().default_embedder in ("fake", "real") else "fake"),
        "use_bge_sparse": bool(_cfg().use_bge_sparse),
        "use_jieba": bool(_cfg().use_jieba),
        # ---- 重排推理加速（T-03，2026-09-24）----
        # CrossEncoderReranker 的两个加速档位。实测（4 候选 / RTX 4070 Laptop / 线上同源口径）：
        #   baseline(8192/fp32) p50 208.7ms → max_length=384 + fp16 = 48.1ms（**×4.34**），
        #   且 Top-5 完全一致（0/5 换位）、打分 max-abs-diff 1.14e-02。
        # ★ 2026-10-05（质量优先，golden_real 221 case 三臂对照）把默认档位由 384 上调到 **512**：
        #   512 全面优于 384 —— hit@5 +3.27pp / nDCG@5 +5.15pp / MRR@10 +8.36pp，代价仅 +51ms；
        #   而 384 会截断约 35% 候选（实测 chunk 真实 token p90=749 > 384）⇒ 384 属**欠拟合档**。
        #   ⚠️ 同批实测同时**否决**向下调（256：hit@5 −4.92pp / nDCG@5 −7.61pp）。
        #   依据：docs/rerank参数下调-可行性评估-2026-10-05.md
        # 回退（一键回到改动前行为）：rag_reranker_max_length=0（不限长）+ rag_reranker_fp16=False。
        # ⚠️ fp16 只在 CUDA 上启用；CPU 上 fp16 通常更慢 → 由 rerank.py 按设备自动回落 fp32。
        "rag_reranker_fp16": True,
        "rag_reranker_max_length": 512,
        # 图片内容增强（ER-03 Level 2）：OCR / 表格结构还原 / VLM 图注。
        # 默认关闭（避免上传期意外多模态费用）；设置页可开启，开启后对后续上传的图片生效。
        # 依赖缺失（paddleocr / transformers / torch）时对应能力自动降级为空。
        "figure_enrich": {"enabled": False, "methods": ["ocr", "table", "vlm"]},
        # 成本熔断上限（¥/轮，NFR）：单轮 LLM 成本超过此值即熔断中止（阶段D）
        "cost_limit_per_turn": 1.0,
        # 单次请求上下文上限（2026-09-17 语义修正）：本次调用的 **prompt** 超过它即中止。
        # 实测依据：绑多篇文档 + 累积检索结果，单轮 prompt 可达 24883 —— 而它原先当"整轮累计"用，
        # 30000 只够 2-3 轮迭代（max_iterations=12）→ 正常多轮会话必然误伤。
        #
        # ⚠️ **2026-10-03：此值现在是"显式兜底"，默认不生效** —— 见 `max_tokens_budget_auto`。
        "max_tokens_budget": 30000,
        # 是否**按模型窗口自动推断**单次预算（2026-10-03，默认开）。
        # ★ 为什么要自动：固定 30000 是历史遗留（它当年还兼管整轮累计，改了语义没改数值），
        #   而实测模型窗口差 4 倍（qwen3-vl-flash 262k / qwen3.6-flash 1M）⇒ 固定值对谁都不合适：
        #   对大窗口浪费 97%，对将来接入的小窗口（表里有 32k 的 qwen-max）会**超窗**。
        # ★ 置 False ⇒ 回落到上面的显式 `max_tokens_budget`（用户的明确选择优先于自动推断）。
        #   取值规则见 `resolve_budget_for`：clamp(窗口 × 0.25, 12k, 100k)。
        "max_tokens_budget_auto": True,
        # 整轮累计 token 上限（含所有迭代）：2026-09-17 独立出来，防"跑飞"的粗护栏。
        # 正常 3-5 轮 × 单轮 5k~25k ≈ 50k~125k，取 150000（正常用不到，失控时拦住）。
        "max_tokens_per_turn": 150000,
        # 本地长期记忆：用户希望 agent 始终遵循的规则/偏好（自由文本，逐行一条）。
        # 每次构建 Agent 时作为 SYSTEM 注记注入，直接影响所有会话（docs/实施记录 2026-08-22）。
        "agent_rules": "",
        # 学科视角默认（2026-08-29 P1）：general / clinical / law / finance / bioscience。
        # 控制技能层 LLM 解读框架注入；general 为零注入、零回归。
        "default_discipline": "general",
        "providers": {name: {"api_key": "", "base_url": "", "model": ""} for name in PRESET_ENDPOINTS},
    }


def _path() -> Path:
    return get_settings().resolved_data_dir / "provider_settings.json"


def _load() -> dict:
    p = _path()
    if not p.exists():
        return _default_state()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  # 配置损坏/解析失败则回退默认状态
        return _default_state()
    base = _default_state()
    for name, kv in (data.get("providers") or {}).items():
        if name in base["providers"]:
            base["providers"][name] = {
                "api_key": (kv.get("api_key", "") or "").strip(),
                "base_url": kv.get("base_url", "") or "",
                "model": kv.get("model", "") or "",
            }
    if data.get("default_model"):
        base["default_model"] = data["default_model"]
    if isinstance(data.get("dedupe_warning"), bool):
        base["dedupe_warning"] = data["dedupe_warning"]
    # 成本熔断上限（¥/轮）：仅接受正数，防脏数据
    if isinstance(data.get("cost_limit_per_turn"), int | float) and float(data["cost_limit_per_turn"]) > 0:
        base["cost_limit_per_turn"] = float(data["cost_limit_per_turn"])
    # token 上限：正整数且 ≥ 1000（防手滑填 0/负数让熔断失效或常量误伤）
    if isinstance(data.get("max_tokens_budget"), int) and not isinstance(data.get("max_tokens_budget"), bool) \
            and data["max_tokens_budget"] >= 1000:
        base["max_tokens_budget"] = int(data["max_tokens_budget"])
    if isinstance(data.get("max_tokens_per_turn"), int) and not isinstance(data.get("max_tokens_per_turn"), bool) \
            and data["max_tokens_per_turn"] >= 1000:
        base["max_tokens_per_turn"] = int(data["max_tokens_per_turn"])
    # 本地长期记忆规则（自由文本）
    if isinstance(data.get("agent_rules"), str):
        base["agent_rules"] = data["agent_rules"]
    # 学科视角默认：仅接受已知取值，脏数据回退默认
    if data.get("default_discipline") in ("general", "clinical", "law", "finance", "bioscience"):
        base["default_discipline"] = data["default_discipline"]
    # 仅合并已知 feature 键，避免脏数据覆盖
    feat = data.get("features")
    if isinstance(feat, dict):
        for k in base["features"]:
            if isinstance(feat.get(k), bool):
                base["features"][k] = feat[k]
    # 记忆向量引擎：仅接受已知取值，脏数据回退默认
    if data.get("memory_embedder") in ("hash", "bge-m3"):
        base["memory_embedder"] = data["memory_embedder"]
    # RAG 检索底座：rag_embedder 仅接受 fake/real，布尔项仅接受 bool
    if data.get("rag_embedder") in ("fake", "real"):
        base["rag_embedder"] = data["rag_embedder"]
    if isinstance(data.get("use_bge_sparse"), bool):
        base["use_bge_sparse"] = data["use_bge_sparse"]
    if isinstance(data.get("use_jieba"), bool):
        base["use_jieba"] = data["use_jieba"]
    # 重排加速档位（T-03）：fp16 仅接受 bool；max_length 仅接受 0 或 64~8192 的整数
    # （0 = 不限长 = 改动前行为；下界 64 防手滑填 1 把候选截成空串）
    if isinstance(data.get("rag_reranker_fp16"), bool):
        base["rag_reranker_fp16"] = data["rag_reranker_fp16"]
    _ml = data.get("rag_reranker_max_length")
    if isinstance(_ml, int) and not isinstance(_ml, bool) and (_ml == 0 or 64 <= _ml <= 8192):
        base["rag_reranker_max_length"] = _ml
    # 图片内容增强开关（仅接受合法 methods）
    fe = data.get("figure_enrich")
    if isinstance(fe, dict) and isinstance(fe.get("enabled"), bool):
        methods = [m for m in fe.get("methods", []) if m in ("ocr", "table", "vlm")]
        base["figure_enrich"] = {"enabled": fe["enabled"], "methods": methods}
    return base


def get_state() -> dict:
    global _state
    if _state is None:
        _state = _load()
    return _state


# ---- 输入规模的前置上限（A8/C8，2026-09-28）----
#: 单条用户消息可占单次请求预算的比例。
#: **为什么是比例而不是写死的数值**：`max_tokens_budget` 是用户可在设置页调整的活配置，
#: 写死一个数（如 15000）会在用户调高/调低预算后立刻失真 —— 与 `LoopConfig.compact_trigger_tokens`
#: 「派生而非新配一项」是同一条理由（调了一个、另一个没调）。
#:
#: **为什么是 0.5**：用户输入**不可压缩**（历史可压、工具结果可裁，输入不行），
#: 所以它只能占 prompt 的一部分，其余要留给 system prompt + 历史 + 本轮检索注入。
#: 实测（doc-bound 场景）非输入部分常在 10~15k ⇒ 输入占到一半时，本轮基本注定撞上限。
#: 正常提问（数十~数百 tokens）与改稿粘贴整段（数千 tokens）**远远碰不到**。
INPUT_BUDGET_RATIO = 0.5

#: 长期规则（`agent_rules`）可占的比例。**比用户输入严得多**，因为它
#: **每一轮、每个会话都要注入**（`chat_service.build_agent` 拼进 system prompt）
#: ⇒ 它的成本是"每次请求都要付"，而不是"这一次"。
RULES_BUDGET_RATIO = 0.1


def input_token_limit(model: str | None = None) -> int:
    """单条用户消息的 token 上限（由**当前生效的** `max_tokens_budget` 派生）。

    口径与预算体系**同源**：比较时必须用 `lightcore.llm.base._rough_tokens`
    （与 `[上下文构成]`/闸门同一把尺子）—— 换成 `len(text)` 就是又一个"两把尺子"。

    ⚠️ **2026-10-03**：预算改为**按模型自动推断**后，本函数必须接受 `model`
    （否则会拿"显式值"去算输入上限、而实际预算按模型放大 ⇒ **又是两把尺子**）。
    `model=None` ⇒ `resolve_budget_for` 回落到显式值，保持对旧调用方兼容。
    """
    return int(resolve_budget_for(model) * INPUT_BUDGET_RATIO)


#: 整轮累计预算相对**单次**预算的倍数。
#:
#: ### 为什么必须跟着 `max_tokens_budget` 一起放大
#: 实测事故（2026-10-03）：`max_tokens_budget` 由 30k 提到 100k 后，单次 prompt 变大
#: ⇒ 多轮迭代的**累计**很快撞上固定的 `max_tokens_per_turn = 150000`
#: ⇒ 报「本轮累计 token 超限（156085 > 上限 150000 tokens）」。
#: ⚠️ 根因是**两个量不同步**：单次按模型自动了、整轮还是固定值。
#:
#: ### ★★ 为什么是 5 而不是 2（2026-10-04 实测推翻前面的 2）
#: ⚠️ **上一版写 2.0 是错的**，错在**跨配置比较了两个数**：
#: 当时的推理是「实测撞线 156k，单次 100k ⇒ 1.6 倍 ⇒ ×2 留 25% 余量」——
#: 但那次熔断用的是**旧配置**（单次 30k / 整轮 150k），其单次峰值只有 ~24.8k
#: ⇒ 156k 其实是**单次峰值的 6.3 倍**，不是"单次预算的 1.6 倍"。
#: ⇒ **拿旧配置的单次峰值去推新配置的倍数 = 两把尺子**。
#:
#: ### 正确的推算口径（同配置实测，2026-10-04 trace 396）
#: 单次 100k / 8 步实测的真实 prompt（含 `[估算比值]` ratio 1.12 换算）：
#: `19k → 24k → 30k → 32k → 33k → 37k → 39k → 43k`，累计 ≈ **257k**（报出的 214315 是中止时的部分累计）。
#: 按末段增量 +3.2k/步 外推到 `max_iterations = 12` ⇒ 累计 ≈ **463k**、末次 ≈ 56k。
#: ⇒ 需求 = `463k / 100k ≈ 4.6 倍` ⇒ **5.0** 才是同口径的取值（并留约 8% 余量）。
TURN_BUDGET_RATIO = 5.0
#: 累计下限：太小会让**短上下文模型**（30k 单次）连 2 轮都跑不完就被拦。
TURN_BUDGET_FLOOR = 60_000
#: 累计上限：1M 窗口模型也不会无限放大（成本/延迟由 `max_cost_per_turn` 兜底，这里只防跑飞）。
#: ⚠️ 上限必须 ≥ `100k × 5`（= 500k），否则 CAP 会把刚放大的值又夹回去。
TURN_BUDGET_CAP = 600_000


def resolve_turn_budget_for(model: str | None = None) -> int:
    """**整轮累计** token 上限（唯一来源）。

    ★ 必须与 `resolve_budget_for` **同源派生** —— 否则单次放大、整轮不变，
    就会像 2026-10-03 那样出现「单次没撞、整轮先撞」的错配。
    """
    base = resolve_budget_for(model)
    return int(max(TURN_BUDGET_FLOOR, min(base * TURN_BUDGET_RATIO, TURN_BUDGET_CAP)))


def rules_token_limit(model: str | None = None) -> int:
    """长期规则（`agent_rules`）的 token 上限（同源派生，见 `RULES_BUDGET_RATIO`）。"""
    return int(resolve_budget_for(model) * RULES_BUDGET_RATIO)


# ---- 单次上下文预算：**按模型窗口自动推断**（2026-10-03）----
#: 自动推断时，预算占模型窗口的比例。
#: **为什么是 0.25 而不是 0.7**：余量要留给**输出**（completion）与估算误差；
#: 且 prompt 的成本/延迟随规模**线性增长**（实测 `耗时 ≈ 0.015×prompt + 3.16×completion + 727ms`
#: ⇒ prompt 每多 100k 就多约 1.5 秒/次调用）。
#: ⚠️ **窗口只决定"不能超过多少"，不决定"应该用多少"** —— 真正的约束是钱与延迟。
_BUDGET_RATIO_OF_WINDOW = 0.25
#: 自动推断的硬上限：即使模型有 1M 窗口也不再往上加（成本/延迟收益比太差）。
_BUDGET_AUTO_CAP = 100_000
#: 自动推断的下限：小窗口模型也要有可用预算（低于它基本没法带工具跑）。
_BUDGET_AUTO_FLOOR = 12_000


def resolve_budget_for(model: str | None = None) -> int:
    """**单次请求上下文预算的唯一来源**。

    ### 为什么需要它（2026-10-03）
    此前 `max_tokens_budget` 是**固定值 30000**（历史遗留：它当年还兼管"整轮累计"，
    改成只管单次后**数值没跟着调**）。而实测模型窗口差 4 倍
    （`qwen3-vl-flash` 262k vs `qwen3.6-flash` 1M）⇒ 固定值对谁都不合适：
    对大窗口**浪费 97%**，对小窗口**将来会超**（表里还有 32k 的 `qwen-max`）。

    ### 优先级（★ 显式设置优先，自动只做**兜底**）
    1. `max_tokens_budget_auto = False` ⇒ 用显式 `max_tokens_budget`（用户的明确选择）；
    2. 否则按 `MODEL_CONTEXT_WINDOW[model]` 推断：`clamp(窗口 × 0.25, 12k, 100k)`；
    3. ⚠️ **模型未登记时不猜** ⇒ 回落显式值（该表自注"近似值，随模型更新"，
       拿它做硬判据本身有风险 ⇒ **宁可保守**）。

    ⚠️ **不是"越大约好"**：窗口大 ≠ 该用满。1M × 0.7 = 700k ⇒ 每次调用多等 10 秒以上。

    ⚠️ **调用方必须传当前会话的 model** —— 否则退回显式值，
    会与 `build_agent` 里实际生效的预算**不一致**（本项目反复踩的"两把尺子"）。
    """
    st = get_state()
    explicit = int(st.get("max_tokens_budget", 30000))
    if not bool(st.get("max_tokens_budget_auto", True)):
        return explicit
    # ⚠️ 规范化：调用方可能传 `provider:model` 形态（如 "qwen:qwen3.6-flash"），
    # 而 `MODEL_CONTEXT_WINDOW` 的键是**裸模型名** ⇒ 在这里统一取末段，
    # 免得每个调用点各写一遍切分逻辑（那又会变成多处口径）。
    name = (model or "").split(":")[-1].strip()
    window = MODEL_CONTEXT_WINDOW.get(name)
    if not window:
        return explicit
    return max(_BUDGET_AUTO_FLOOR, min(int(window * _BUDGET_RATIO_OF_WINDOW), _BUDGET_AUTO_CAP))


def save() -> None:
    """落盘运行时设置（含 API 密钥）。

    2026-09-06 加固：文件为**明文 JSON 且含密钥**（不入库约束已满足——DB 无密钥字段），
    写入后尽力收紧文件权限（POSIX 0o600；Windows 的 chmod 仅支持只读位，尽力而为不报错），
    降低同机其他用户/进程读取与误同步泄露面。"""
    path = _path()
    path.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass  # Windows 上 chmod 仅支持只读位，尽力而为


def update(payload: dict) -> dict:
    """写入并落盘。

    payload: {"default_model"?: str, "providers"?: {name: {"api_key"?, "base_url"?}}}
    只更新 payload 中出现的字段，未出现的 provider/字段保持不变（避免清空已保存的 key）。
    """
    with _lock:
        state = get_state()
        if payload.get("default_model"):
            state["default_model"] = payload["default_model"]
        if isinstance(payload.get("dedupe_warning"), bool):
            state["dedupe_warning"] = payload["dedupe_warning"]
        # 本地长期记忆规则：仅接受字符串
        if isinstance(payload.get("agent_rules"), str):
            state["agent_rules"] = payload["agent_rules"]
        # 学科视角默认：仅接受已知取值，脏数据忽略
        if payload.get("default_discipline") in ("general", "clinical", "law", "finance", "bioscience"):
            state["default_discipline"] = payload["default_discipline"]
        # 成本熔断上限（¥/轮）：正数才接受
        if isinstance(payload.get("cost_limit_per_turn"), int | float) and float(payload["cost_limit_per_turn"]) > 0:
            state["cost_limit_per_turn"] = float(payload["cost_limit_per_turn"])
        # token 上限：正整数且 ≥ 1000 才接受（单轮上下文 / 整轮累计 各一项）
        if isinstance(payload.get("max_tokens_budget"), int) and not isinstance(payload.get("max_tokens_budget"), bool) \
                and payload["max_tokens_budget"] >= 1000:
            state["max_tokens_budget"] = int(payload["max_tokens_budget"])
        if isinstance(payload.get("max_tokens_per_turn"), int) and not isinstance(payload.get("max_tokens_per_turn"), bool) \
                and payload["max_tokens_per_turn"] >= 1000:
            state["max_tokens_per_turn"] = int(payload["max_tokens_per_turn"])
        # Phase B 功能开关：仅更新出现且为 bool 的已知键
        feat = payload.get("features")
        if isinstance(feat, dict):
            for k in state["features"]:
                if isinstance(feat.get(k), bool):
                    state["features"][k] = feat[k]
        # 记忆向量引擎（hash / bge-m3）：仅接受合法取值，切换后由 memory_hub 重建并按需重算向量
        if payload.get("memory_embedder") in ("hash", "bge-m3"):
            state["memory_embedder"] = payload["memory_embedder"]
        # RAG 检索底座：rag_embedder 仅接受 fake/real，布尔项仅接受 bool
        # （切换后由前端「重建索引」触发 DocumentService.reindex_all 使新配置生效，
        #  此处只持久化，不自动重建，避免在不经意保存时触发重模型加载）
        if payload.get("rag_embedder") in ("fake", "real"):
            state["rag_embedder"] = payload["rag_embedder"]
        if isinstance(payload.get("use_bge_sparse"), bool):
            state["use_bge_sparse"] = payload["use_bge_sparse"]
        if isinstance(payload.get("use_jieba"), bool):
            state["use_jieba"] = payload["use_jieba"]
        # 重排加速档位（T-03）：同一套校验；改动后 DocumentService 会因签名变化重建重排器单例
        if isinstance(payload.get("rag_reranker_fp16"), bool):
            state["rag_reranker_fp16"] = payload["rag_reranker_fp16"]
        _ml = payload.get("rag_reranker_max_length")
        if isinstance(_ml, int) and not isinstance(_ml, bool) and (_ml == 0 or 64 <= _ml <= 8192):
            state["rag_reranker_max_length"] = _ml
        # 图片内容增强开关：仅接受 enabled(bool) + 合法 methods
        fe = payload.get("figure_enrich")
        if isinstance(fe, dict) and isinstance(fe.get("enabled"), bool):
            methods = [m for m in fe.get("methods", []) if m in ("ocr", "table", "vlm")]
            state["figure_enrich"] = {"enabled": fe["enabled"], "methods": methods}
        for name, kv in (payload.get("providers") or {}).items():
            if name not in state["providers"]:
                continue
            if "api_key" in kv and kv["api_key"] is not None:
                # 去除首尾空白/换行：从网页复制 key 常带入尾随换行，会导致 DashScope 判定 key 无效
                state["providers"][name]["api_key"] = kv["api_key"].strip()
            if "base_url" in kv and kv["base_url"] is not None:
                state["providers"][name]["base_url"] = kv["base_url"]
            if "model" in kv and kv["model"] is not None:
                state["providers"][name]["model"] = kv["model"]
        save()
        _apply_env(state)
        return public_view(state)


def _apply_env(state: dict) -> None:
    """把已保存的 key 注入 os.environ，供 OAICompatibleLLM._api_key() 在请求时读取。"""
    for name, kv in state["providers"].items():
        ep = PRESET_ENDPOINTS.get(name)
        if ep and kv.get("api_key"):
            os.environ[ep.api_key_env] = kv["api_key"]


def _enrich_deps() -> dict:
    """探测图片增强（Level 2）依赖是否可用，供前端展示「开启后实际能增强哪些能力」。

    ★ 2026-10-09：实现**收敛到 `services.figure_deps.probe()`（唯一口径）**——
    这里只做委托，避免"探测"与"安装通道"各留一份（同源纪律：不造第二把尺子）。
    """
    from .services.figure_deps import probe as _probe  # 惰性导入：避免模块级循环

    return _probe()


def public_view(state: dict | None = None) -> dict:
    """对外视图：去掉真实 key，仅返回是否已配置 + 脱敏预览 + base_url。"""
    state = state or get_state()
    providers: dict[str, dict] = {}
    for name, kv in state["providers"].items():
        key = kv.get("api_key", "") or ""
        providers[name] = {
            "configured": bool(key),
            "key_masked": (key[:4] + "****" + key[-4:]) if len(key) > 8 else ("****" if key else ""),
            "base_url": kv.get("base_url", "") or "",
            "model": kv.get("model", "") or "",
            "available_models": PROVIDER_MODELS.get(name, []),
        }
    return {
        "default_model": state["default_model"],
        "dedupe_warning": state.get("dedupe_warning", True),
        "features": state.get("features", dict(DEFAULT_FEATURES)),
        "figure_enrich": state.get("figure_enrich", {"enabled": False, "methods": []}),
        "figure_enrich_deps": _enrich_deps(),
        "cost_limit_per_turn": state.get("cost_limit_per_turn", 1.0),
        "max_tokens_budget": state.get("max_tokens_budget", 30000),
        "max_tokens_per_turn": state.get("max_tokens_per_turn", 150000),
        "agent_rules": state.get("agent_rules", ""),
        "memory_embedder": state.get("memory_embedder", "hash"),
        # ★ 2026-10-04（O6 复评）：下面三项的 fallback 改为**与 `_load()` 同源**（走 `_cfg()`）。
        #   起因：审查报告称「设置页把 `use_bge_sparse` 回显成 False 而实际在跑 True」——
        #   **实测该现象不存在**：`_load()` 保证这三个 key 恒存在（默认值直接取 `config.Settings`），
        #   实测删掉 json 里的键后 `get_state()` 仍返回 `use_bge_sparse=True` / `rag_embedder="real"`
        #   ⇒ `state.get(key, <fallback>)` 的 **fallback 永不可达**，回显是正确的。
        # ⇒ 那为何还要改：fallback 值（`False` / `"fake"`）**与真实默认（True / real）不一致**，
        #   属**死默认值** —— 后人读代码会误以为「默认关闭」，而真实默认是开启。
        #   改后 fallback 与默认值**同源**，即使将来 `_load()` 改成不写这些键也不会显示错。
        # ⚠️ 本次**不是修 bug**（对当前行为零影响），是消除误导性默认值。
        "rag_embedder": state.get(
            "rag_embedder",
            _cfg().default_embedder if _cfg().default_embedder in ("fake", "real") else "fake",
        ),
        "use_bge_sparse": state.get("use_bge_sparse", bool(_cfg().use_bge_sparse)),
        "use_jieba": state.get("use_jieba", bool(_cfg().use_jieba)),
        "rag_reranker_fp16": state.get("rag_reranker_fp16", True),
        "rag_reranker_max_length": state.get("rag_reranker_max_length", 512),
        "default_discipline": state.get("default_discipline", "general"),
        "data_dir": str(get_settings().resolved_data_dir),  # 本地存储路径（设置页只读展示）
        "providers": providers,
    }
