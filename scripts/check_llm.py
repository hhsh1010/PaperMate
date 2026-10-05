#!/usr/bin/env python
"""PaperMate · LLM 连通性探测（docs/07 §M3.2 接真实模型前的自检）

用法：
    uv run python scripts/check_llm.py                     # 默认 deepseek（厂商默认模型）
    uv run python scripts/check_llm.py openai
    uv run python scripts/check_llm.py qwen
    uv run python scripts/check_llm.py --model qwen3-vl-flash   # 精确探测某模型（2026-08-25 起）

前置：复制 .env.example 为 .env，填入对应 provider 的 API key，
然后（若改了 PAPERMATE_DEFAULT_MODEL）重启后端。本脚本与后端无关，
仅用于「填好 key 后、进前端前」快速验证 key 有效、端点可达。
"""
from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lightcore.llm.base import ChatMessage, Role
from lightcore.llm.router import PRESET_ENDPOINTS, OAICompatibleLLM


def _load_env_file(path: str) -> None:
    """极简 .env 加载（不依赖 python-dotenv）：仅填充尚未设置的环境变量。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


def main() -> int:
    provider: str | None = None
    model: str | None = None
    argv = sys.argv[1:]
    if "--model" in argv:
        i = argv.index("--model")
        model = argv[i + 1] if i + 1 < len(argv) else None
        if model is None:
            print("✗ --model 缺少模型 id，如：check_llm.py --model qwen3-vl-flash")
            return 2
        from lightcore.llm.router import MODEL_TO_PROVIDER

        provider = MODEL_TO_PROVIDER.get(model)
        if provider is None:
            print(f"✗ 未知模型: {model}；请检查 PROVIDER_MODELS")
            return 2
    elif argv:
        provider = argv[0]  # 兼容旧用法：位置参数 = provider

    if provider is None or provider not in PRESET_ENDPOINTS:
        print(f"✗ 未知 provider: {provider}；可选: {list(PRESET_ENDPOINTS)}（或用 --model <模型id>）")
        return 2

    _load_env_file(os.path.join(ROOT, ".env"))

    ep = PRESET_ENDPOINTS[provider]
    llm = OAICompatibleLLM(ep, model=model) if model else OAICompatibleLLM(ep)

    # 1) key 是否配置
    try:
        key = llm._api_key()
    except RuntimeError as e:
        print(f"✗ {e}")
        print(f"  解决：在 .env 中填入 `{ep.api_key_env}=sk-xxx`，再运行本脚本。")
        return 1
    print(f"✓ 已读取 {ep.api_key_env}（长度 {len(key)}）")

    # 2) 发一条测试消息，验证端点可达且能对话
    async def probe() -> str:
        resp = await llm.chat(
            [ChatMessage(role=Role.USER, content="用一句话介绍你自己，并说明你是哪个模型。")],
            temperature=0.3,
        )
        return resp.message.content

    try:
        reply = asyncio.run(probe())
    except Exception as e:  # httpx 网络/超时/限流(429)等
        print(f"✗ 调用 {ep.base_url} 失败：{type(e).__name__}: {e}")
        print("  排查：网络是否可达、key 是否有效、是否触发限流(429)。")
        return 1

    print(f"✓ 端点可用，模型 {llm.model_name} 回复：\n----\n{reply}\n----")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
