"""Spike S3 · SSE 流式验证线（docs/13 第 0 周 Spike S3；docs/04 SSE 协议）

目标：确认 FastAPI + Starlette StreamingResponse 能实时把 Agent 循环事件推给前端
（thinking / tool_call / tool_result / message / done / error）。

运行：在仓库根 `uv run scripts/spike_s3_sse.py`，访问 http://127.0.0.1:8011/sse?q=xxx
或 `curl -N http://127.0.0.1:8011/sse?q=LightCore`
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from lightcore.core.loop import AgentLoop, LoopEvent, LoopState
from lightcore.llm.fake import FakeLLM
from lightcore.tools.builtin.basic import register_builtin_tools
from lightcore.tools.spec import ToolExecutor, ToolRegistry

app = FastAPI(title="PaperMate SSE Spike")


def _build_loop() -> AgentLoop:
    registry = ToolRegistry()
    register_builtin_tools(registry)
    return AgentLoop(FakeLLM(), ToolExecutor(registry), system_prompt="你是论文研读助手 PaperMate。")


@app.get("/sse")
async def sse(request: Request, q: str = "LightCore 的核心贡献是什么？"):
    loop = _build_loop()

    async def event_stream():
        yield ": connected\n\n"
        async for ev in loop.run_stream(q):
            # 客户端断开检测
            if await request.is_disconnected():
                break
            yield ev.to_sse()
        yield LoopEvent(type="done", state=LoopState.FINISH, data={"reason": "stream end"}).to_sse()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8011, log_level="info")
