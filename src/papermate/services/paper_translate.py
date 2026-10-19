"""论文标题中译（**批量、单次请求、可校验**）

## 为什么这么设计
- **批量**：20 条标题逐条翻译 = 20 次 LLM 调用（贵、慢、更容易撞限流）；塞进**一个** prompt
  ⇒ 1 次调用、1 次计费，且能让模型看到全局上下文、译法更一致。
- **严格 JSON + 条数校验**：模型偶尔会多/少一条、或用 ```json 包起来。
  这里**宽松解析、严格校验**：能救的救（剥代码块），救不了的返回 None（前端保留原文），
  **不猜、不错位**（错位比不翻译严重得多）。
- **不点不花钱**：本模块不自动触发，只由前端「翻译标题」按钮显式调用。

## 三条硬约束（项目对 LLM 功能的约定）
1. 有明确入口（前端按钮）⇒ 用户不点就没有成本；
2. 有上限（一次最多 20 条；`max_tokens` 受限）；
3. 失败**可读**（模型未配置 / 返回非 JSON / 条数不符 ⇒ 都给人能懂的提示）。
"""

from __future__ import annotations

import json
import re

#: 一次最多翻译多少条（与检索单页条数一致）
MAX_TITLES = 20

#: 单次调用的输出上限（20 条中文标题 ≈ 20×30 字，留足余量）
MAX_OUTPUT_TOKENS = 2000

# ★ 命名：不能叫 SYSTEM_PROMPT —— 项目有跨文件约束「全库只能有一处 SYSTEM_PROMPT」
#   （tests/test_trace_and_prompts.py 会扫源码），主提示词在 prompts/system.py。
TRANSLATE_SYSTEM_PROMPT = (
    "你是学术论文标题翻译助手。把英文论文标题翻译成**简洁准确的中文**。"
    "要求：保留模型名、方法名、数据集名与通用缩写（如 RAG、ResNet、LiDAR、Transformer）不译；"
    "不要添加解释、不要扩写；只输出 JSON 数组，格式为 [{\"i\":1,\"zh\":\"…\"}]，i 与输入编号一致。"
)


def build_messages(titles: list[str]) -> list[dict[str, str]]:
    """构造一次批量翻译的 messages。编号从 1 开始，便于回填与校验。"""
    lines = [f"{i}. {t.strip()}" for i, t in enumerate(titles, 1)]
    return [
        {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
        {"role": "user", "content": "请翻译下列论文标题：\n" + "\n".join(lines)},
    ]


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _strip_fence(raw: str) -> str:
    m = _JSON_FENCE.search(raw or "")
    return (m.group(1) if m else raw or "").strip()


def parse_titles_json(raw: str, expected: int) -> list[str | None]:
    """把模型输出解析成**长度恰为 expected** 的列表；无法对应的位置为 None。

    宽松处：允许 ```json 代码块、允许对象/数组两种外层（`{"items":[...]}`）、允许 i 从 1 或 0 开始。
    严格处：**只按 i 回填**，不按顺序猜（顺序错一位会让标题张冠李戴）；条数不符不回填。
    """
    if expected <= 0:
        return []
    text = _strip_fence(raw)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        # 兜底：从文本里抠出第一个 [...] 再试一次（模型偶尔会在前后加话）
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return [None] * expected
        try:
            data = json.loads(text[start:end + 1])
        except (TypeError, ValueError):
            return [None] * expected

    if isinstance(data, dict):
        data = data.get("items") or data.get("titles") or []
    if not isinstance(data, list):
        return [None] * expected

    out: list[str | None] = [None] * expected
    for item in data:
        if not isinstance(item, dict):
            continue
        zh = str(item.get("zh") or item.get("translation") or "").strip()
        if not zh:
            continue
        try:
            idx = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        # 兼容 i 从 1 开始（约定）与从 0 开始（模型偶发）
        pos = idx - 1 if 1 <= idx <= expected else idx
        if 0 <= pos < expected and out[pos] is None:
            out[pos] = zh
    return out


def is_fake_model(model_id: str | None) -> bool:
    """是否是"没有真实模型"的退化状态（fake / 空）。

    ★ 必须显式拦住：`_llm_from_raw` 对**未知/缺失**会**静默退化成 ApiFakeLLM**，
    那种情况下模型会返回占位文本 ⇒ 用户看到一堆假标题，比报错更糟。
    """
    m = (model_id or "").strip().lower()
    return m in ("", "fake", "api-fake", "apifake")


#: 上游错误体里可能带 key（如 "incorrect api key sk-xxx"）⇒ 一律打码
_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")
#: 常见状态码 → 人话（按项目踩过的三类：402 余额 / 429 限流 / 403 免费额度耗尽）
_STATUS_HINT = {
    400: "请求被模型拒绝（400），通常是我们发的参数有问题",
    401: "模型密钥无效或已过期（401）",
    402: "模型账户余额不足（402），请充值或换用其它模型",
    403: "没有权限，或免费额度已用完（403）",
    404: "模型不存在或已被下线（404）",
    429: "请求太频繁被限流（429），请等十几秒再试",
}


def _mask(text: str) -> str:
    return _KEY_RE.sub("sk-***", text or "")


def _upstream_message(resp: object) -> str:
    """从上游响应体里抠出最有用的一句（OpenAI 兼容格式 error.message）。"""
    try:
        body = getattr(resp, "text", "") or ""
    except Exception:  # noqa: BLE001
        body = ""
    if not body:
        return ""
    try:
        data = json.loads(body)
        msg = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                msg = str(err.get("message") or err.get("code") or "")
            msg = msg or str(data.get("message") or data.get("detail") or "")
        if msg:
            return _mask(msg)[:200]
    except (TypeError, ValueError):
        pass
    return _mask(" ".join(body.split()))[:200]


def describe_llm_error(exc: Exception) -> tuple[int, str]:
    """把模型调用异常翻成 (给前端的 HTTP 状态码, 人话)。

    ★ 为什么必须带上游状态码：只说 "HTTPStatusError" 用户**完全无法判断**是密钥错、余额不足还是限流
      —— 项目历史上这三类都踩过（402/429/403）。必须让用户看完就知道下一步做什么。
    ★ 为什么单独返回状态码：401/402/403 是**用户可修复**的（换 key / 充值 / 换模型）⇒ 用 400 语义（客户端问题）；
      429 原样 429（可重试）；上游 5xx / 网络 ⇒ 502（服务端问题）。
    ★ 脱敏：上游错误体常回显 key（`incorrect api key sk-xxx`）⇒ 一律打码再返回。
    """
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    detail = _upstream_message(resp) if resp is not None else ""

    if isinstance(status, int):
        if status in _STATUS_HINT:
            code = 429 if status == 429 else 400
            tail = f"：{detail}" if detail else "。请到设置页检查模型与密钥。"
            return code, _STATUS_HINT[status] + tail
        if status >= 500:
            tail = f"：{detail}" if detail else "。请稍后重试。"
            return 502, f"模型服务故障（{status}）" + tail
        tail = f"：{detail}" if detail else ""
        return 502, f"调用模型失败（HTTP {status}）" + tail

    # 没有响应对象：网络 / 超时 / 其它
    tail = f"：{detail}" if detail else "。请检查网络与设置页的模型配置。"
    return 502, f"调用模型失败（{type(exc).__name__}）" + tail
