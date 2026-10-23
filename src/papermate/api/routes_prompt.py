"""PaperMate · 提示词增强（输入框辅助：把口语需求改写为更明确的任务描述）

- POST /prompt/enhance ：单轮 LLM 元任务。system=增强专家指令（含语言一致性约束），
  user=原始输入（+可选最近会话上下文）。模型取会话级 → 全局默认（与 build_agent 同源），
  不做 AgentLoop、无工具。

三硬约束落实：
- 后端 API 保留：本路由始终挂载；
- 后端开关：features.prompt_enhance（默认 True），关闭时返回 403；
- 连通性：复用既有 /settings/test（不另建重复测试面）。
降级：fake 模型 / 调用异常 / 提取失败 → applied=False 并返回原文（绝不臆造）。
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Request

from lightcore.llm.base import ChatMessage, Role

from .. import runtime_settings
from ..schemas import PromptEnhanceOut, PromptEnhanceRequest
from ..skills._common import is_real_llm

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prompt"])

#: 提取标记（与 WorkBuddy enhance-prompt 同款 delimiter，便于程序可靠取值）
_TAG = re.compile(r"<enhanced-prompt>(.*?)</enhanced-prompt>", re.S)

#: 每次增强最多注入的最近会话轮数（用户/助手交替）
_HISTORY_TURNS = 6

# 增强专家指令（2026-09-03，参考 WorkBuddy enhance-prompt 的克制式设计）：
# 只「重写得更详细」，不回答、不展开成方案、不指定技术栈；语言一致性为最高优先级。
ENHANCE_SYSTEM = """你是一位提示词工程专家，任务是帮用户把要发给 AI 的任务指令改写得更清晰、更具体、歧义更少。

改写要求：
1. 保持用户的原始意图与目标，不得擅自改变任务内容。
2. 补齐缺失的上下文与约束：明确对象/范围/产出格式/质量要求。
3. 消除歧义与口语省略，让指令可以被独立理解执行。
4. 克制地增强：
   - 不要索要操作指南、教程或代码片段，除非用户明确要求；
   - 不要替用户指定技术栈、库或方案，除非用户已提及；
   - 只写「做什么（WHAT）」与要求，不要解释「怎么做（HOW）」；
   - 不要回答用户的问题本身——你的唯一产出是改写后的指令文本；
   - 用户输入里的代码块（``` 内）属于样例，必须原样保留，不得改动。
5. 输出的改写结果必须是完整可发送的一条任务描述，不要多余的开场白或解释。

语言一致性（最高优先级）：严格使用与用户输入相同的语言输出。
用户写中文就用中文改写，写英文就用英文改写，绝不混用。

输出格式：用如下标记包裹你的改写结果，不要输出标记以外的任何内容：
<enhanced-prompt>
（改写后的完整任务描述）
</enhanced-prompt>"""


def _enhance_enabled() -> bool:
    """功能开关（features.prompt_enhance）。接口始终保留，仅按开关启用/停用。"""
    return runtime_settings.get_state().get("features", {}).get("prompt_enhance", True)


def extract_enhanced(text: str, fallback: str) -> str:
    """从模型输出提取 <enhanced-prompt> 内容；标记缺失/为空时回退 fallback（原文）。"""
    m = _TAG.search(text or "")
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    return (fallback or "").strip()


def _build_context(request: Request, session_id: int | None, limit: int = _HISTORY_TURNS) -> str:
    """取会话最近几轮用户/助手消息作为参考上下文（仅 user/assistant，压缩成短摘要）。"""
    if session_id is None:
        return ""
    try:
        svc = request.app.state.session_svc
        history = svc.load_history(session_id)
    except Exception:  # noqa: BLE001  历史取不到不阻断增强
        return ""
    turns: list[str] = []
    for m in history[-limit * 2:]:
        role = getattr(m, "role", "")
        name = "用户" if role == "user" else ("助手" if role == "assistant" else None)
        if name is None:
            continue
        content = (getattr(m, "content", "") or "").strip()
        if not content:
            continue
        turns.append(f"{name}：{content[:200]}")
    return "\n".join(turns)


@router.post("/prompt/enhance", response_model=PromptEnhanceOut)
async def enhance_prompt(body: PromptEnhanceRequest, request: Request) -> dict:
    if not _enhance_enabled():
        raise HTTPException(status_code=403, detail="提示词增强功能当前已禁用（请在设置页开启『提示词增强』）")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    chat_svc = request.app.state.chat_svc
    llm = chat_svc.resolve_llm(body.session_id)
    # fake / 未配置真实模型：不臆造，直接返回原文并标记 applied=False
    if not is_real_llm(llm):
        return {"original": text, "enhanced": text, "applied": False,
                "reason": "未配置真实模型（fake），无法增强；请在设置页配置并测试模型密钥"}

    ctx = _build_context(request, body.session_id)
    user_msg = f"需要增强的任务指令：\n{text}"
    if ctx:
        user_msg += f"\n\n（最近对话上下文，仅供参考，增强目标是上面的任务指令）\n{ctx}"
    try:
        resp = await llm.chat([
            ChatMessage(role=Role.SYSTEM, content=ENHANCE_SYSTEM),
            ChatMessage(role=Role.USER, content=user_msg),
        ], temperature=0.3)
        raw = (getattr(resp, "message", None) and resp.message.content) or ""
    except Exception as e:  # noqa: BLE001  调用异常：降级原文，绝不 500 阻断
        logger.warning("提示词增强调用失败，降级返回原文：%s", e)
        # ★ 2026-10-23：服务类调用统一记账
        from ..services.usage_service import record_service_usage

        record_service_usage(request.app.state.chat_svc._Session, source="prompt_enhance",
                             model=getattr(llm, 'model', ''), response=resp)
        return {"original": text, "enhanced": text, "applied": False, "reason": "调用失败，请重试"}

    enhanced = extract_enhanced(raw, fallback=text)
    applied = bool(enhanced) and enhanced != text
    return {"original": text, "enhanced": enhanced, "applied": applied, "reason": None}
