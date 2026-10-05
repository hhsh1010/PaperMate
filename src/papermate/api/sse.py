"""PaperMate · SSE 事件封装（docs/04 §4.6.2 事件协议）

每个事件一行 `event: <name>` + 一行 `data: <json>`，空行分隔。
LoopEvent 自带 type/data/state；通用 sse_event 用于 done/message 等自定义事件。
"""

from __future__ import annotations

import json
from typing import Any

from lightcore.core.loop import LoopEvent


def sse_event(name: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {name}\ndata: {payload}\n\n"


def sse_frame(event: LoopEvent) -> str:
    payload = {"type": event.type, "state": event.state.value, **event.data}
    return sse_event(event.type, payload)
