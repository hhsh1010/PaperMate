"""PaperMate · 段落 AI 改写（改稿工作台：扩写/缩写/润色/去重）

- POST /workbench/para-transform ：单轮 LLM 元任务（与 /prompt/enhance 同范式）。
  输入目标段落 + 前后文参考；输出改写后的段落文本。

三硬约束落实：
- 后端 API 保留：本路由始终挂载；
- 后端开关：features.para_transform（默认 True），关闭时返回 403；
- 连通性：复用既有 /settings/test（不另建重复测试面）。
降级：fake 模型 / 调用异常 / 提取失败 → applied=False 并返回原文（绝不臆造）。

防幻觉红线：改写只允许「重述/重组/增删修饰与论证表述」，不得引入原文没有的
事实、数据、引用与结论——扩写尤其要克制（宁短勿编）。
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Request

from lightcore.llm.base import ChatMessage, Role

from .. import runtime_settings
from ..schemas import ParaTransformOut, ParaTransformRequest
from ..skills._common import is_real_llm

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workbench"])

#: 提取标记（与 /prompt/enhance 同款 delimiter 范式）
_TAG = re.compile(r"<transformed>(.*?)</transformed>", re.S)

#: 合法动作
_ACTIONS = ("expand", "shorten", "polish", "dedupe")

_ACTION_LABEL = {
    "expand": "扩写",
    "shorten": "缩写",
    "polish": "润色",
    "dedupe": "去重",
}

_COMMON_RULES = """通用规则（最高优先级）：
1. 语言一致性：严格使用与原文相同的语言输出。
2. 防幻觉：不得引入原文没有的事实、数据、引用、实验结果与结论；
   扩写只能展开已有要点的论证与表述，宁短勿编。
3. 你的唯一产出是改写后的段落文本本身——不要解释、不要点评、不要前后缀。
4. 输出格式：用如下标记包裹改写结果，标记外不要有任何内容：
<transformed>
（改写后的段落）
</transformed>"""


#: 各动作的专家指令
ACTION_SYSTEM: dict[str, str] = {
    "expand": (
        "你是一位学术写作专家，任务：把给定段落**扩写**。\n"
        "要求：在忠实原意的前提下展开论证——补充解释性表述、逻辑衔接与细节描述，"
        "使内容更充分；篇幅约为原文的 1.5～2 倍；不得偏题、不得引入新事实。\n"
    ),
    "shorten": (
        "你是一位学术写作专家，任务：把给定段落**缩写**。\n"
        "要求：压缩至原文的约 40%～60%，保留全部核心要点与关键术语；"
        "删除冗余修饰与重复表述；语义不损失、不断章取义。\n"
    ),
    "polish": (
        "你是一位学术写作专家，任务：对给定段落**润色**。\n"
        "要求：修正语病、提升流畅度与学术表达的严谨性；篇幅与原文相当；"
        "保持原意与术语不变，不做结构调整以外的内容改动。\n"
    ),
    "dedupe": (
        "你是一位学术写作专家，任务：对给定段落**去重**。\n"
        "要求：识别并消除段内重复/冗余的表述（同义反复、重复罗列、啰嗦句式），"
        "合并为单一清晰表述；保留全部独立要点；篇幅不增。\n"
    ),
}


def _transform_enabled() -> bool:
    """功能开关（features.para_transform）。接口始终保留，仅按开关启用/停用。"""
    return runtime_settings.get_state().get("features", {}).get("para_transform", True)


def extract_transformed(text: str, fallback: str) -> str:
    """从模型输出提取 <transformed> 内容；标记缺失/为空时回退 fallback（原文）。"""
    m = _TAG.search(text or "")
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    return (fallback or "").strip()


def build_user_prompt(text: str, context_before: str | None, context_after: str | None) -> str:
    """组装 user 消息：目标段落是唯一改写对象，前后文仅供参考。"""
    parts = [f"【目标段落（唯一改写对象）】\n{text}"]
    if context_before and context_before.strip():
        parts.append(f"【上文（仅供参考，不要改写、不要输出）】\n{context_before.strip()[:600]}")
    if context_after and context_after.strip():
        parts.append(f"【下文（仅供参考，不要改写、不要输出）】\n{context_after.strip()[:600]}")
    return "\n\n".join(parts)


@router.post("/workbench/para-transform", response_model=ParaTransformOut)
async def para_transform(body: ParaTransformRequest, request: Request) -> dict:
    if not _transform_enabled():
        raise HTTPException(status_code=403, detail="段落 AI 改写功能当前已禁用（请在设置页开启『段落 AI 改写』）")
    action = (body.action or "").strip()
    if action not in _ACTIONS:
        raise HTTPException(status_code=400, detail=f"action 必须为 {_ACTIONS} 之一，收到：{action!r}")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    chat_svc = request.app.state.chat_svc
    llm = chat_svc.resolve_llm(body.session_id)
    # fake / 未配置真实模型：不臆造，直接返回原文并标记 applied=False
    if not is_real_llm(llm):
        return {"original": text, "transformed": text, "applied": False,
                "reason": "未配置真实模型（fake），无法改写；请在设置页配置并测试模型密钥"}

    system = ACTION_SYSTEM[action] + _COMMON_RULES
    try:
        resp = await llm.chat([
            ChatMessage(role=Role.SYSTEM, content=system),
            ChatMessage(role=Role.USER, content=build_user_prompt(text, body.context_before, body.context_after)),
        ], temperature=0.4)
        raw = (getattr(resp, "message", None) and resp.message.content) or ""
    except Exception as e:  # noqa: BLE001  调用异常：降级原文，绝不 500 阻断
        logger.warning("段落改写调用失败，降级返回原文：%s", e)
        return {"original": text, "transformed": text, "applied": False, "reason": "调用失败，请重试"}

    transformed = extract_transformed(raw, fallback=text)
    applied = bool(transformed) and transformed != text
    return {
        "original": text,
        "transformed": transformed,
        "applied": applied,
        "action": action,
        "action_label": _ACTION_LABEL[action],
        "reason": None if applied else "模型未返回有效改写（结果与原文相同），可重试",
    }
