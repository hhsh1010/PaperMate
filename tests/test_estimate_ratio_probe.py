"""A0（2026-10-02）：`[估算比值]` 打点 + 分布分析脚本。

## 为什么打点必须"每次调用都打"

原有的 `[上下文构成] 实测 vs 估算` 只在偏差 **>1.5 倍**时才输出（省日志），
而 ② 要回答的是"比值的**分布**"（p50 与离散度 ⇒ 该不该上校正系数 k）——
"只在异常时打"给不出分布 ⇒ 必须每次一行，且**带 model**（密度标定与厂商/模型相关，
跨模型平均会得出无意义的数）。

## 本文件在守什么

1. 每次 LLM 调用都有一行 `[估算比值]`，含 `model / step / prompt / est / ratio`；
2. 该行**只是观测**：不得改变任何裁剪行为（`ratio` 不参与闸门判定）；
3. 分析脚本能按 model 分组算 p10/p50/p90，并按报备书 §9.2 的阈值给出**明确结论**
   （p50 ≤ 1.15 ⇒ "② 直接取消"也是合格结论，不许粉饰成"需要校准"）。
"""

from __future__ import annotations

import logging
import re

import pytest

from lightcore.core.loop import AgentLoop, LoopConfig
from lightcore.llm.base import ChatMessage, LLMResponse, Role

LOGGER = "lightcore.core.loop"
RATIO_LINE = re.compile(
    r"\[估算比值\] model=(?P<model>\S+) step=\d+ prompt=(?P<prompt>\d+) est=(?P<est>\d+) ratio=(?P<ratio>[\d.]+)"
)


class _ReportingLLM:
    """报出**固定** prompt_tokens 的 LLM 替身（让比值可预测）。"""

    model_name = "stub-model-x"

    def __init__(self, prompt_tokens: int = 2000) -> None:
        self._pt = prompt_tokens
        self.calls = 0

    async def chat(self, messages, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        self.calls += 1
        return LLMResponse(
            message=ChatMessage(role=Role.ASSISTANT, content=f"回答{self.calls}"),
            prompt_tokens=self._pt, completion_tokens=10, latency_ms=1,
            model=self.model_name,
        )


def _loop(llm) -> AgentLoop:
    loop = AgentLoop(llm, None, LoopConfig())  # type: ignore[arg-type]
    return loop


class TestRatioLogging:
    @pytest.mark.parametrize("prompt_tokens", [800, 2000])
    def test_every_call_logs_ratio(self, caplog, prompt_tokens):
        """★ 每次调用都要有一行，且带 model 与比值（分布才是 ② 的判据）。"""
        caplog.set_level(logging.INFO, logger=LOGGER)
        import asyncio

        loop = _loop(_ReportingLLM(prompt_tokens))
        asyncio.run(loop.run("问题", history=[]))
        lines = [r.getMessage() for r in caplog.records if "[估算比值]" in r.getMessage()]
        assert len(lines) >= 1, "每次 LLM 调用都必须打一行比值（否则给不出分布）"
        m = RATIO_LINE.search(lines[0])
        assert m, f"日志格式变了，请同步脚本与本测试：{lines[0]}"
        assert m.group("model") == "stub-model-x", "必须带 model（比值可能随厂商而变）"
        assert int(m.group("est")) > 0
        assert float(m.group("ratio")) > 0

    def test_ratio_is_observation_only(self):
        """★ 该行只是观测：`ratio` 不得参与闸门判定（源码契约）。"""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "src" / "lightcore" / "core" / "loop.py").read_text(
            encoding="utf-8")
        start = src.index("def _gate_context")
        gate = src[start:src.index("async def _finalize_without_tools", start)]
        assert "[估算比值]" not in gate, "比值只许用于日志，不得进入闸门判定"


class TestAnalysisScript:
    """脚本的解析与判据（用合成日志，不依赖真实运行）。"""

    def _write(self, tmp_path, lines: list[str]):
        p = tmp_path / "app.log"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def _line(self, model: str, ratio: float, step: int = 1) -> str:
        est = 1000
        return (f"2026-10-02 16:00:00,000 INFO lightcore.core.loop INFO:lightcore.core.loop:"
                f"[估算比值] model={model} step={step} prompt={int(est * ratio)} est={est} ratio={ratio:.2f}")

    def test_parses_and_groups_by_model(self, tmp_path):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from probe_estimate_ratio import parse, verdict  # noqa: PLC0415

        p = self._write(tmp_path, [
            self._line("model-a", 1.05), self._line("model-a", 1.10, 2),
            self._line("model-b", 2.00), self._line("model-b", 2.40, 2),
            "无关日志行",
        ])
        data = parse(p)
        assert set(data) == {"model-a", "model-b"}
        assert data["model-a"] == [1.05, 1.10]
        assert len(data["model-b"]) == 2

    @pytest.mark.parametrize("p50,disp,expect", [
        (1.05, 1.2, "取消"),      # 漂移不成立 → 明确说"② 取消"
        (1.60, 1.3, "常数 k"),    # 大但稳定 → 常数 k
        (1.60, 3.0, "不要全局 k"),  # 离散 → 不要全局 k
    ])
    def test_verdict_matches_reported_thresholds(self, p50, disp, expect):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from probe_estimate_ratio import verdict  # noqa: PLC0415

        assert expect in verdict(p50, disp), f"p50={p50} disp={disp} 的结论应含「{expect}」"