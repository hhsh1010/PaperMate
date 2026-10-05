"""T-01 收尾实验（T-01-F）：循环内上下文闸门的端到端效果 —— 峰值 prompt 与溢出次数。

## 为什么要这样设计（先读这段，否则会误读数字）

要证明"闸门降低了峰值、消除了溢出"，两侧必须**逐消息同构**：若让真实模型自由决定
"这一步调不调工具"，两臂的步数、工具结果条数就会不同，差异可能来自模型自身的不确定性，
而不是闸门 —— 那样测出来的不是闸门的效果。

所以本实验把「是否调用工具」**脚本化**（确定性），而**其余全部为真**：
真实 `AgentLoop`（未改动源码，仅"关闸"臂把 `_gate_context` 换成空操作）、
真实工具执行链路（`ToolRegistry`/`ToolExecutor`/`_run_one`/`_format_tool_result`）、
真实 30000 单次请求上限与 70% 触发线、真实 token 计量。

## 计量两档（口径必须写清）

- `--count local`（默认）：`prompt_tokens` 用**已标定**的 `_rough_tokens`（对 API 真值
  相对误差 p50≈5%，见 docs/实施步骤 T-01-A）驱动循环 → **零配额消耗**，可反复跑。
  它决定"闸门在第几步触发""在第几步撞上限"，这两处的余量都远大于 5%（见结论的边界说明）。
- `--count api`：每步真调 API（`max_tokens=1`）取**真实** `prompt_tokens` —— 金标准，
  但消耗配额与峰值成正比。

`--verify-peak`（默认开）：本地模式跑完后，把两臂各自的**峰值调用**原样（同一 messages +
同一 tool_specs）送真实 API 各一次 → 得到**真实峰值**。总消耗 ≈ 两次峰值之和（≈5 万 tokens），
这是"真实数字"与"配额可控"之间的折中；比对结果会打印预测与真实的偏差。

## 产物

`eval/t01_e2e_result.txt`（**追加 + 运行横幅**，可多轮共存互证，不覆盖）。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lightcore.core.loop import AgentLoop, LoopConfig  # noqa: E402
from lightcore.llm.base import ChatMessage, LLMResponse, Role, ToolCall, _rough_tokens  # noqa: E402
from lightcore.tools.spec import (  # noqa: E402
    Permission,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

OUT_FILE = ROOT / "eval" / "t01_e2e_result.txt"
TOOL_NAME = "doc_bulk"

SYSTEM_PROMPT = (
    "你是 PaperMate 的研究助理，负责基于用户上传的文档回答问题。"
    "你的回答必须严格依据工具返回的原文证据，禁止编造页码、标题或结论。"
    "若证据不足，明确说明缺少什么，而不要猜测。回答使用简体中文，条理清晰，"
    "必要时分点陈述，并在引用处标注来源分段。"
)

USER_INPUT = "请把这篇文档的各个分段依次读完，然后总结它的主要论点与实验设置。"

# ---- 确定性正文生成（混合中英 + JSON，贴近真实工具结果形态）----

_ZH = (
    "本文提出一种面向本地单用户场景的检索增强生成方案，把稠密向量检索与稀疏词项检索融合，"
    "并在重排阶段引入交叉编码器以提升首位命中率。实验表明该方案在中文长文档上表现稳定，"
    "推理开销可控。方法部分详细描述了分块策略、向量归一化与候选裁剪之间的取舍。"
)
_EN = (
    "We present a retrieval-augmented generation pipeline for local single-user scenarios. "
    "The system fuses dense and sparse retrieval, then applies a cross-encoder reranker. "
    "Experiments show stable accuracy on long documents with bounded inference cost. "
)
_JSON = (
    '{"id": "chunk-0001", "page": 3, "score": 0.8123, "heading": "Method", "text": "..."}'
)


def _rep(text: str, n: int) -> str:
    if not text or n <= 0:
        return ""
    return (text * (n // len(text) + 1))[:n]


def _build(seq: int, chars: int) -> str:
    third = chars // 3
    return (f"[doc_bulk seq={seq}]\n{_rep(_ZH, third)}\n{_rep(_EN, third)}\n"
            f"{_rep(_JSON, chars - 2 * third)}")


def make_payload(seq: int, target_tokens: int) -> str:
    """按**目标估算 token** 反推字符数（估算器已标定，误差 p50≈5%，够用于定尺寸）。"""
    chars = max(400, int(target_tokens * 2.0))
    text = _build(seq, chars)
    for _ in range(8):
        t = _rough_tokens(text)
        if t <= 0 or abs(t - target_tokens) / target_tokens <= 0.02:
            break
        chars = max(400, int(chars * target_tokens / t))
        text = _build(seq, chars)
    return text


def _msg_tokens(m: ChatMessage) -> int:
    """单条消息进 prompt 的估算量 —— **必须把 tool_calls 的 JSON 也算进去**。"""
    t = _rough_tokens(m.content)
    for tc in (m.tool_calls or []):
        t += _rough_tokens(json.dumps(tc.to_openai(), ensure_ascii=False))
    return t


# ---- 脚本化 LLM：行为确定，计量可切换 ----


class ScriptedCountingLLM:
    def __init__(self, *, inner=None, mode: str = "local", tool_steps: int = 6) -> None:
        self._inner = inner
        self.mode = mode
        self.tool_steps = tool_steps
        self.model_name = getattr(inner, "model_name", "local-counter") if inner else "local-counter"
        self.calls: list[dict] = []

    async def _count(self, messages: list[ChatMessage], specs: list) -> int:
        if self.mode == "api":
            r = await self._inner.chat(messages, specs or None, max_tokens=1)
            return int(r.prompt_tokens or 0)
        total = sum(_msg_tokens(m) for m in messages)
        total += 4 * len(messages)          # 每条消息的框架开销（占位量级）
        if specs:
            total += _rough_tokens(json.dumps(specs, ensure_ascii=False))
        return total

    async def chat(self, messages, tools=None, **kw) -> LLMResponse:  # noqa: ANN001
        snapshot = copy.deepcopy(messages)          # 循环会**原地**改 messages
        spec_snapshot = copy.deepcopy(list(tools)) if tools else []
        pt = await self._count(snapshot, spec_snapshot)
        idx = len(self.calls)
        self.calls.append({"seq": idx, "prompt_tokens": pt, "n_messages": len(snapshot),
                           "messages": snapshot, "tools": spec_snapshot})
        if idx < self.tool_steps:
            msg = ChatMessage(role=Role.ASSISTANT, content="",
                              tool_calls=[ToolCall(id=f"call_{idx}", name=TOOL_NAME,
                                                   arguments={"seq": idx})])
            ct = 25
        else:
            msg = ChatMessage(role=Role.ASSISTANT, content="（实验）已读完全部分段，基于以上证据作答。")
            ct = 20
        return LLMResponse(message=msg, prompt_tokens=pt, completion_tokens=ct,
                           model=self.model_name, latency_ms=0, cost=0.0)

    def peak(self) -> dict:
        return max(self.calls, key=lambda c: c["prompt_tokens"])


# ---- 日志捕获（闸门 / 压缩 / 上下文构成）----


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(record.getMessage())
        except Exception:  # noqa: BLE001
            self.records.append(str(record.msg))


def _say(msg: str) -> None:
    print(msg, flush=True)
    with OUT_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---- 单臂执行 ----


async def run_arm(*, gate_on: bool, args, inner) -> dict:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name=TOOL_NAME,
        description="按序号读取被引用文档的一个分段正文（实验用，确定性内容）。",
        parameters={"type": "object",
                    "properties": {"seq": {"type": "integer", "description": "分段序号，从 0 开始"}},
                    "required": ["seq"]},
        permission=Permission.SAFE,
        handler=lambda seq: ToolResult(ok=True, output=make_payload(int(seq), args.payload_tokens)),
    ))
    llm = ScriptedCountingLLM(inner=inner, mode=args.count, tool_steps=args.tool_steps)
    cfg = LoopConfig(max_iterations=args.max_iterations,
                     max_tokens_budget=args.budget,
                     max_tokens_per_turn=args.turn_budget,
                     temperature=0.0)
    loop = AgentLoop(llm=llm, tools=ToolExecutor(registry), config=cfg,
                     system_prompt=SYSTEM_PROMPT, last_prompt_tokens=0)

    cap = _Capture()
    lg = logging.getLogger("lightcore.core.loop")
    lg.addHandler(cap)
    prev_level = lg.level
    lg.setLevel(logging.INFO)

    events = []
    orig = AgentLoop._gate_context
    if not gate_on:
        AgentLoop._gate_context = lambda self, messages: None      # noqa: ARG005
    t0 = time.perf_counter()
    try:
        async for ev in loop.run_stream(USER_INPUT, [], stream_tokens=False):
            events.append(ev)
    finally:
        AgentLoop._gate_context = orig
        lg.removeHandler(cap)
        lg.setLevel(prev_level)
    elapsed = time.perf_counter() - t0

    aborts = [e for e in events if e.type == "error" and "上下文超限" in str(e.data.get("reason", ""))]
    done = next((e for e in reversed(events) if e.type == "done"), None)
    elide = [r for r in cap.records if "[上下文闸门]" in r and "循环内裁剪" in r]
    saved = 0
    for r in elide:
        try:
            saved += int(r.split("省 ≈")[1].split(" tokens")[0])
        except Exception:  # noqa: BLE001
            pass
    comp = [r for r in cap.records if "上下文构成" in r]

    return {
        "gate_on": gate_on,
        "steps": len(llm.calls),
        "prompts": [c["prompt_tokens"] for c in llm.calls],
        "peak": llm.peak(),
        "peak_est": max(c["prompt_tokens"] for c in llm.calls),
        "aborts": len(aborts),
        "abort_reason": aborts[0].data.get("reason") if aborts else None,
        "elisions": len(elide),
        "elided_tokens": saved,
        "peak_call": llm.peak(),
        "done": done.data if done else None,
        "elapsed_s": elapsed,
        "ctx_logs": comp,
        "est_total": sum(_rough_tokens(m.content) for m in llm.peak()["messages"]),
    }


# ---- 真实 API 复核 ----


async def verify_peak(inner, call: dict) -> int:
    """把记录的峰值调用**原样**送真实 API，取真实 prompt_tokens。"""
    r = await inner.chat(call["messages"], call["tools"] or None, max_tokens=1)
    return int(r.prompt_tokens or 0)


def _build_real_llm(model: str):
    from lightcore.llm.router import PRESET_ENDPOINTS, OAICompatibleLLM
    from papermate import runtime_settings

    st = runtime_settings.get_state()
    pset = (st.get("providers") or {}).get("qwen", {}) or {}
    key = (pset.get("api_key") or "").strip()
    if not key:
        raise SystemExit("✗ runtime_settings 里没有 qwen.api_key（先在设置页填好）")
    return OAICompatibleLLM(PRESET_ENDPOINTS["qwen"], model=model, api_key=key)


async def main() -> None:
    ap = argparse.ArgumentParser(description="T-01 闸门端到端实验（峰值 / 溢出）")
    ap.add_argument("--budget", type=int, default=30000, help="单次请求上限（线上默认 30000）")
    ap.add_argument("--turn-budget", type=int, default=150000)
    ap.add_argument("--max-iterations", type=int, default=12)
    ap.add_argument("--tool-steps", type=int, default=6, help="脚本让模型连续调用工具的步数")
    ap.add_argument("--payload-tokens", type=int, default=6000, help="每次工具结果的目标 token 量")
    ap.add_argument("--count", choices=("local", "api"), default="local")
    ap.add_argument("--model", default="qwen3.6-flash")
    ap.add_argument("--no-verify", action="store_true", help="跳过真实 API 复核（零消耗）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    inner = None if args.count == "local" and args.no_verify else _build_real_llm(args.model)

    _say("")
    _say("=" * 92)
    _say(f"T-01-F 循环内上下文闸门 · 端到端实验（{time.strftime('%Y-%m-%d %H:%M:%S')}）")
    _say(f"  单次请求上限={args.budget}（触发线={int(args.budget * 0.7)}）"
         f"· 轮累计上限={args.turn_budget} · max_iterations={args.max_iterations}"
         f" · 工具步数={args.tool_steps}")
    _say(f"  工具结果目标={args.payload_tokens} tokens/次 · 计量={args.count}"
         f" · 真实复核={'关（零消耗）' if args.no_verify else '开'} · 模型={args.model}")
    _say("=" * 92)

    arms = {}
    for gate_on in (False, True):
        name = "闸门开" if gate_on else "闸门关（改动前行为）"
        _say(f"\n--- {name} ---")
        res = await run_arm(gate_on=gate_on, args=args, inner=inner)
        arms[gate_on] = res
        _say(f"  逐步 prompt（{args.count}）：" + " → ".join(f"{p:,}" for p in res["prompts"]))
        _say(f"  峰值（本地计量）= {res['peak_est']:,} tokens · 溢出中止 = {res['aborts']} 次"
             f" · 闸门裁剪 = {res['elisions']} 次（省 ≈{res['elided_tokens']:,} tokens）"
             f" · 耗时 {res['elapsed_s']:.1f}s")
        if res["abort_reason"]:
            _say(f"  中止原因：{res['abort_reason']}")
        if res["done"]:
            _say(f"  done 事件：steps={res['done'].get('steps')}"
                 f" · 峰值 prompt={res['done'].get('peak_prompt_tokens')}")

    real = {}
    if not args.no_verify:
        _say("\n--- 真实 API 复核（各臂峰值调用原样重放一次）---")
        for gate_on in (False, True):
            call = arms[gate_on]["peak_call"]
            names = [m.role.value for m in call["messages"]]
            head = ",".join(names[:8]) + ("…" if len(names) > 8 else "")
            _say(f"  [{'闸门开' if gate_on else '闸门关'}] 重放 {len(names)} 条消息（{head}）…")
            try:
                real[gate_on] = await verify_peak(inner, call)
                dev = (real[gate_on] / max(1, call["prompt_tokens"]) - 1) * 100
                _say(f"        真实 prompt = {real[gate_on]:,} tokens"
                     f"（本地预测 {call['prompt_tokens']:,}，偏差 {dev:+.1f}%）")
            except Exception as e:  # noqa: BLE001
                _say(f"        复核失败：{type(e).__name__}: {str(e)[:160]}")

    p_off = real.get(False) or arms[False]["peak_est"]
    p_on = real.get(True) or arms[True]["peak_est"]
    tag = "真实 API" if real else "本地估算（未复核）"
    _say(f"\n--- 结论（峰值取 {tag}）---")
    _say(f"  闸门关：峰值 {p_off:,} tokens · 溢出 {arms[False]['aborts']} 次")
    _say(f"  闸门开：峰值 {p_on:,} tokens · 溢出 {arms[True]['aborts']} 次")
    if p_off > 0:
        _say(f"  峰值变化 = {(p_on / p_off - 1) * 100:+.1f}%（目标：≤ -25%）")
    _say(f"  溢出事件 = {arms[False]['aborts']} → {arms[True]['aborts']}（目标：0）")
    if arms[True]["elisions"]:
        _say(f"  闸门裁剪 {arms[True]['elisions']} 次、累计省 ≈{arms[True]['elided_tokens']:,} tokens")
    _say("=" * 92)


if __name__ == "__main__":
    asyncio.run(main())
