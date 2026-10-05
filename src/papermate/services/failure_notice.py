"""失败轮次的**用户可见文案**（单一来源）：回答气泡文本 + 警示卡 notice。

## 为什么单独成模块（2026-10-02，④）

原先这套文案散在路由里，且**按中文子串猜原因**：

```python
if "余额" in detail or "Insufficient" in detail: ...
elif "token 预算" in detail or "熔断" in detail: ...
elif "达到最大迭代" in detail: ...
```

三个问题：
1. **脆**：文案一改就失配；"熔断"两字出现在无关文本里会**假匹配**；
2. **气泡在撒谎**：熔断中止时持久化的仍是通用兜底
   「可能是模型调用失败或返回内容为空」⇒ 把用户引向**错误方向**
   （查余额、换模型 —— 而这两件事对"上下文超预算"毫无作用）；
3. 同一份"怎么办"的建议在**两个地方**各写一遍（notice 与气泡），必然漂移。

⇒ 本模块按 **`failure_kind`（结构化枚举，`core/failure.py`）** 分流；
**文本匹配只在 kind 缺失时兜底**（老轨迹没有该字段）。
"""

from __future__ import annotations

#: 通用兜底（**逐字保留历史文案**，避免老用户/老测试的观感与断言回归）。
#: 语义边界：它只适用于"原因不明"的失败；**已知原因必须走下面 kind 分流的文案**。
FALLBACK_REPLY = (
    "抱歉，本次没有生成有效回答（可能是模型调用失败或返回内容为空）。"
    "建议：① 检查设置页该厂商的「测试连接」与账户余额；② 换个模型重试；③ 稍后再试。"
)

#: detail（厂商/循环给的 reason）在用户文案里保留的最大长度。
_DETAIL_MAX = 160

#: 分 kind 的「怎么办」——一句话一步自救。
_ADVICE: dict[str, str] = {
    "budget_exceeded": (
        "① 新建会话或精简历史后重试（长会话最容易触发）；"
        "② 减少本会话关联的文档篇数；③ 到设置页上调对应的上下文/成本上限。"
    ),
    "max_iterations": "把问题拆成更小、更具体的子问题后重试（模型步数有限，一次别让它做太多）。",
    "llm_error": "① 到设置页用该厂商的「测试连接」确认连通与余额；② 换一个模型重试。",
    "no_progress": "换个问法或补充线索后重试（模型在原地打转，已自动中止）。",
    "tool_error": "换个问法重试；若反复出现，请确认文档索引可用（文献页可重建索引）。",
}

#: 分 kind 的**一句话定性**——关键是别让用户误判方向。
_STATEMENT: dict[str, str] = {
    "budget_exceeded": "本次回答因**预算熔断**中止（**不是余额或模型故障**）",
    "max_iterations": "模型在允许的步数内没完成任务，已自动中止",
    "llm_error": "模型调用失败，本次未生成回答",
    "no_progress": "模型连续多步没有获得新进展，已自动中止",
    "tool_error": "工具连续失败，已自动中止本轮",
}


def _clip(detail: str) -> str:
    d = (detail or "").strip().replace("\n", " ")
    return d if len(d) <= _DETAIL_MAX else d[:_DETAIL_MAX] + "…"


def fallback_text(kind: str = "", reason: str = "") -> str:
    """**持久化到回答气泡**的兜底文本：按 kind 给出正确归因 + 可操作建议。

    未知 kind ⇒ 返回 `FALLBACK_REPLY`（历史文案，逐字）。
    """
    k = (kind or "").strip()
    if k not in _STATEMENT:
        return FALLBACK_REPLY
    detail = _clip(reason)
    head = _STATEMENT[k] + (f"：{detail}" if detail else "")
    return f"{head}。\n\n建议：{_ADVICE.get(k, '直接重试本问题。')}"


def notice_for(kind: str, reason: str = "", *, fallback_suggestion: str = "") -> dict:
    """警示卡 notice（落 `meta.notice`，刷新后仍可见）。

    ★ 分流用 **kind**；仅当 kind 缺失（老轨迹）才退回"按 reason 文本猜"的老逻辑。
    `notice["kind"]` 仍为 `stream_failed`（前端按它渲染），另有 `reason_kind` 供追踪。
    """
    detail = _clip(reason)
    k = (kind or "").strip()
    # 老逻辑兜底（kind 缺失时才用）：保留是为了**老轨迹/未知来源**仍能给出方向性建议。
    if not k or k not in _STATEMENT:
        legacy = fallback_suggestion
        if detail and not legacy:
            if "余额" in detail or "Insufficient" in detail:
                legacy = ("请前往对应平台充值（DeepSeek: platform.deepseek.com / "
                          "Qwen: bailian.console.aliyun.com），或到设置页更换模型。")
            elif "token 预算" in detail or "熔断" in detail or "超限" in detail:
                legacy = ("本轮消耗超过了上下文/成本上限。可在设置页调高上限，"
                          "或减少本会话关联文档数、新建会话后重试。")
            elif "达到最大迭代" in detail:
                legacy = "模型在允许步数内未能完成任务。请把问题拆成更小、更具体的子问题后重试。"
        message = f"本次回答生成失败，原因：{detail}" if detail else "本次回答生成失败。"
        return _notice(message, legacy or "① 直接重试本问题；② 若反复出现，到设置页排查。", k)

    head = _STATEMENT[k] + (f"（{detail}）" if detail else "")
    return _notice(head + "。", _ADVICE.get(k, "直接重试本问题。"), k)


def _notice(message: str, suggestion: str, reason_kind: str) -> dict:
    n = {
        "kind": "stream_failed",  # 前端依赖此值渲染警示卡（勿改）
        "title": "⚠️ 回答生成中断",
        "message": message,
        "suggestion": suggestion,
    }
    if reason_kind:
        n["reason_kind"] = reason_kind  # 附加：结构化归因，便于追踪与后续分桶
    return n
