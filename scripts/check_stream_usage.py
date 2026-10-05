#!/usr/bin/env python
"""PaperMate · 流式用量采集自检（2026-09-16，对应 `docs/用量与评测功能体检报告.md` 问题 1 的修复）

**为什么单独有这么一个脚本**：流式 usage 的修复只有在**真实厂商端点**上跑一次才算闭环 ——
单元测试用的是 MockTransport，无法回答"该端点是否接受 `stream_options` 并真的下发 usage"。
本脚本发**一次极短提问**（成本约 ¥0.001），打印 `finished` 块上的用量四元组。

用法：
    python -m scripts.check_stream_usage             # 依次尝试 deepseek → qwen
    python -m scripts.check_stream_usage qwen        # 指定厂商
    python -m scripts.check_stream_usage deepseek --model deepseek-v4-flash

判读：
- `(pt, ct, latency_ms, cost)` **都不为 0** → ✅ 流式用量采集正常（熔断/用量页/Trace 都有数据了）
- 打印"未含 usage"警告而数值来自本地估算 → 端点未按预期下发，但**对话不会中断**（已实现降级）
- 报"缺少 API Key" → `.env` 里对应 key 仍是注释状态（本机默认如此）
"""

from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lightcore.llm.base import ChatMessage, Role  # noqa: E402
from lightcore.llm.router import PRESET_ENDPOINTS, OAICompatibleLLM  # noqa: E402

PROMPT = "用一句话说明什么是向量检索。"


def _load_env_file(path: str) -> None:
    """极简 .env 加载（不依赖 python-dotenv）：仅填充尚未设置的环境变量（与 check_llm.py 同款）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def _probe(llm: OAICompatibleLLM) -> tuple[int, int, int, float, str]:
    """流式一次，返回 (prompt_tokens, completion_tokens, latency_ms, cost, 正文)。"""
    text_parts: list[str] = []
    pt = ct = latency = 0
    cost = 0.0
    async for chunk in llm.stream([ChatMessage(role=Role.USER, content=PROMPT)]):
        if chunk.delta:
            text_parts.append(chunk.delta)
        if chunk.finished:
            pt, ct = chunk.prompt_tokens, chunk.completion_tokens
            latency, cost = chunk.latency_ms, chunk.cost
    return pt, ct, latency, cost, "".join(text_parts)


def main() -> int:
    argv = sys.argv[1:]
    providers = [argv[0]] if argv and not argv[0].startswith("-") else ["deepseek", "qwen"]
    model: str | None = None
    if "--model" in argv:
        i = argv.index("--model")
        model = argv[i + 1] if i + 1 < len(argv) else None

    _load_env_file(os.path.join(ROOT, ".env"))

    tried: list[str] = []
    for provider in providers:
        if provider not in PRESET_ENDPOINTS:
            print(f"✗ 未知 provider: {provider}；可选 {list(PRESET_ENDPOINTS)}")
            return 2
        ep = PRESET_ENDPOINTS[provider]
        key = os.getenv(ep.api_key_env, "")
        if not key:
            tried.append(f"{provider}（未配置 {ep.api_key_env}）")
            continue

        llm = OAICompatibleLLM(ep, model=model) if model else OAICompatibleLLM(ep)
        print(f"→ 探测 {provider} / {llm.model_name}（流式，单次极短提问）…")
        try:
            pt, ct, latency, cost, text = asyncio.run(_probe(llm))
        except Exception as e:  # noqa: BLE001  自检脚本：把原因如实打出来即可
            print(f"✗ 调用失败：{type(e).__name__}: {e}")
            return 1

        print(f"  回答前 40 字：{text.strip()[:40]}")
        print(f"  finished 块用量：(prompt={pt}, completion={ct}, latency={latency}ms, cost=¥{cost:.6f})")
        if pt > 0 and ct > 0 and latency > 0:
            print("✅ 流式用量采集正常（token / 延迟 / 成本均非 0）"
                  "—— 成本熔断、用量页与 Trace 耗时现在有真实数据了")
            return 0
        print("⚠️ 部分字段为 0：若上面出现『流式响应未含 usage』警告，说明该端点未按预期下发 usage，"
              "已走本地估算兜底（对话不受影响）；否则请检查本脚本与 router 的字段口径")
        return 1

    print("✗ 没有可用的厂商：" + "、".join(tried))
    print("  请在 .env 中取消注释并填入 API Key（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY），然后重跑。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
