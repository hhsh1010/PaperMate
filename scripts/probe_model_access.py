"""模型可用性探针：逐个候选模型发一次最小请求，确认 key 是否真有权限。

### 为什么需要它（2026-09-27 实测触发）
配置里 `default_model = qwen3.6-flash`（`.env` + `provider_settings.json` 一致），
但 `run_eval --judge` 跑出 **HTTP 403「API 无权限访问该模型」**，
15 条抽样**全部**生成失败 ⇒ faithfulness/abstention 都成了「无样本」。

⇒ **「代码里列了该模型」≠「当前 key 有权限」**（同 MEMORY：「配置存在」≠「key 有效」）。
⇒ 本探针把"哪个模型名真能用"变成**实测**，而不是照配置猜。

⚠️ **真实 API 消耗**：每候选一次请求、`max_tokens=5`，合计可忽略。

用法：`.venv/Scripts/python.exe scripts/probe_model_access.py`
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import judge as J  # noqa: E402

from lightcore.llm.base import ChatMessage, Role  # noqa: E402

#: 候选 = 代码库里登记的全部模型名（`PROVIDER_MODELS`）+ 当前配置值
CANDIDATES: list[tuple[str, str]] = [
    ("qwen", "qwen3.7-flash"),
    ("qwen", "qwen3.6-flash"),
    ("qwen", "qwen3-max"),
    ("deepseek", "deepseek-v4-flash-vision-exp"),
    ("deepseek", "deepseek-v4-flash"),
]


async def probe(provider: str, model: str) -> tuple[str, str]:
    """返回 (状态, 说明)。**任何异常都收敛成 FAIL 文本**，不打断探测循环。"""
    try:
        llm = J.build_llm(provider, model=model, timeout=25)
        resp = await llm.chat(
            [ChatMessage(role=Role.USER, content="ping")], max_tokens=5
        )
        return "OK", f"key来源={getattr(llm, 'key_source', '?')} | 回复={(resp.message.content or '')[:24]!r}"
    except Exception as e:  # noqa: BLE001  探针必须跑完全部候选
        return "FAIL", f"{type(e).__name__}: {str(e)[:130]}"


async def main() -> None:
    print(f"{'provider':<9} {'model':<32} {'状态':<5} 说明")
    print("-" * 104)
    for provider, model in CANDIDATES:
        status, note = await probe(provider, model)
        print(f"{provider:<9} {model:<32} {status:<5} {note}")


if __name__ == "__main__":
    asyncio.run(main())
