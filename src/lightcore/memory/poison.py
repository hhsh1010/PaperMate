"""LightCore · 毒化记忆模式（共享常量，2026-08-29 C 档）

背景：此前同一套「状态观察型 / 拒接型」黑名单存在两份拷贝——
- `LongTermMemory._WRITE_BLOCKLIST`（反思写入侧拦截）
- `ChatService._POISON_PATTERNS`（召回注入侧过滤）

两份内容相近但已开始漂移（召回侧多几条、写入侧少几条），修一处漏一处。
现在收敛为唯一来源，三处使用方统一引用：

- `lightcore.memory.long_term`：写入前拦截
- `papermate.services.chat_service`：召回注记前过滤
- `lightcore.tools.builtin.domain`：记忆工具对外暴露前过滤

这些条目是「某一次拒接 / 守门得出的临时状态观察」，不是用户偏好；
写入后会被 recall 注入 SYSTEM 注记，让 LLM 把「无文档」「静默跳过」当成
硬规则，进而跳过真实 retrieve 调用（见 2026-08-19（三续）事故）。
"""

from __future__ import annotations

#: 命中即视为毒化的「状态观察 / 拒接」表述
POISON_PATTERNS: tuple[str, ...] = (
    "无任何文档", "无任何上传", "无任何可检索", "无上传任何文件", "无上传任何 PDF",
    "未上传任何", "未成功上传", "未上传 PDF", "未上传文档", "未提供任何文件",
    "没有任何上传", "任何上传记录",
    "向量库为空", "未触发任何文档", "静默跳过", "对无效指令", "对重复指令",
    "前提不成立", "无可供分析的文档", "零文档",
    "系统中无任何", "当前无任何文档", "当前未上传任何",
    "所有检索返回空", "所有工具调用",
)

#: 纯拒接开头的表述（避免被注入 SYSTEM 当规则复读）
REFUSAL_PREFIXES: tuple[str, ...] = ("我无法", "无法回答", "无法执行")


def is_poisoned(text: str) -> bool:
    """判断一条记忆文本是否为毒化的状态观察 / 拒接结论。"""
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith(REFUSAL_PREFIXES):
        return True
    return any(p in t for p in POISON_PATTERNS)
