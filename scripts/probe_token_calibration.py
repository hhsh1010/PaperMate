"""T-01-A · token 估算标定探针（v2：按**内容类型**拟合）

**要回答的问题**：`_rough_tokens = len/2.5` 的偏差是「常数倍」还是「漂移」？
能不能用一个更好的模型把误差压到 **±10%**？

**v1 的结论与我的设计错误（保留，见下）**
v1 跑出 22 条受控样本，逐条比值**稳定且按内容类型分档**：

| 内容类型 | real/est（小→大） | 收敛值 |
|---|---|---|
| 中文正文 | 1.72 → 1.58 → 1.54 → 1.52 | ~1.50（**低估**） |
| 英文 | 0.78 → 0.56 → 0.50 → 0.48 | ~0.48（**高估一倍**） |
| 代码 | 0.98 → 0.81 → 0.74 → 0.72 | ~0.72 |
| JSON | 1.38 → 1.25 → 1.20 → 1.18 | ~1.18 |
| 混合 | 1.21 → 1.12 → 1.09 | ~1.09 |

→ 两条重要纠正：
1. **不存在"系统性偏低 2 倍"**：中文确实低估，但**英文被高估一倍**、代码也偏高。
   同一把尺子在**中英混合**会话里误差方向相反，因此**任何单一系数都补不平**。
2. **v1 按"角色"拟合是错的**（`w[assistant]=15.6` 明显在拟合噪声）：真实的可解释维度是
   **内容类型**，不是消息角色。v2 改按内容类型拟合。

**v2 方法**
- 每条样本由**显式分段**组成：`segments = [(kind, text), ...]`，kind ∈ {zh, en, code, json}；
- 逐段用 `len(text)` 计字符数 → 两种模型各自预测：
  - **现状模型**：`pred = Σ len(seg) / 2.5`
  - **分型模型**：`pred = Σ len(seg) / density(kind)`，密度由**训练集**拟合；
- 用**留一法（LOO）**评估泛化误差（不是拟合内误差）；
- 判据：**相对误差 p50 ≤ ±10%**。

**产物**：`eval/t01_calibration.txt`（落盘；stdout 同源）

用法：
    .venv/Scripts/python.exe -u scripts/probe_token_calibration.py
    .venv/Scripts/python.exe -u scripts/probe_token_calibration.py --model qwen3.6-flash
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

OUT_FILE = _ROOT / "eval" / "t01_calibration.txt"

#: 内容类型（分段标注用）
KINDS = ("zh", "en", "code", "json")

#: 现状系数（= `_rough_tokens` 的 len/2.5）
CURRENT_DENSITY = 2.5


def _say(msg: str) -> None:
    """同时打印与落盘（T-03 教训：长跑只写 stdout 会丢结果）。"""
    print(msg, flush=True)
    try:
        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:  # noqa: BLE001  落盘失败不中断测量
        pass


# ---------------- 受控语料（确定性，不用随机数） ----------------

_ZH = ("检索增强生成把外部知识注入到生成过程，从而降低幻觉。"
       "本文提出一种按块粒度对齐的重排策略，并在真实语料上验证其有效性。")
_EN = ("Retrieval-augmented generation injects external knowledge into the decoding "
       "process to reduce hallucination. We propose a chunk-aligned reranking strategy. ")
_CODE = ("def rerank(query, candidates, top_k=5, texts=None):\n"
         "    scores = model.predict([(query, t) for t in texts])\n"
         "    order = sorted(range(len(scores)), key=lambda i: -scores[i])\n"
         "    return [(candidates[i], float(scores[i])) for i in order[:top_k]]\n")
_JSON = ('{"chunks":[{"id":"pm-rag-c0","page":1,"score":0.83,"text":"检索结果片段"},'
         '{"id":"pm-rag-c1","page":2,"score":0.71,"text":"按块粒度对齐的重排策略"}]}')

_SRC = {"zh": _ZH, "en": _EN, "code": _CODE, "json": _JSON}


def _rep(s: str, n_chars: int) -> str:
    """把语料重复/截断到约 n_chars 字符（确定性，可复现）。"""
    if not s:
        return ""
    k = max(1, n_chars // len(s) + 1)
    return (s * k)[:n_chars]


def _samples() -> list[tuple[str, list[tuple[str, str]]]]:
    """返回 [(标签, segments)]，segments = [(kind, text), ...]。

    **为什么用显式分段而不是"整条一段"**：真实消息基本都是混合内容
    （中文正文 + 代码 + JSON 工具结果），只有按分段求和才能验证
    "分型密度是否真的可加"，也才能在 LOO 里把训练/验证分开。
    """
    out: list[tuple[str, list[tuple[str, str]]]] = []
    # ① 单类型 × 多长度：给出每型的**渐近密度**（小尺寸含固定开销，会偏高）
    for kind in KINDS:
        for size in (300, 800, 2000, 4000):
            out.append((f"{kind}-{size}", [(kind, _rep(_SRC[kind], size))]))
    # ② 混合（真实场景）：按比例混，验证可加性
    out.append(("mix-zh6code2json2", [("zh", _rep(_ZH, 1200)),
                                      ("code", _rep(_CODE, 400)),
                                      ("json", _rep(_JSON, 400))]))
    out.append(("mix-zh4en4", [("zh", _rep(_ZH, 800)), ("en", _rep(_EN, 800))]))
    out.append(("mix-zh3en3code2json2", [("zh", _rep(_ZH, 600)), ("en", _rep(_EN, 600)),
                                         ("code", _rep(_CODE, 400)),
                                         ("json", _rep(_JSON, 400))]))
    out.append(("mix-en6json4", [("en", _rep(_EN, 1200)), ("json", _rep(_JSON, 800))]))
    out.append(("mix-zh2code8", [("zh", _rep(_ZH, 400)), ("code", _rep(_CODE, 1600))]))
    return out


# ---------------- 调用真实 API ----------------


def _build_llm(model: str):
    from lightcore.llm.router import PRESET_ENDPOINTS, OAICompatibleLLM
    from papermate import runtime_settings

    st = runtime_settings.get_state()
    pset = (st.get("providers") or {}).get("qwen", {}) or {}
    key = (pset.get("api_key") or "").strip()
    if not key:
        raise SystemExit("✗ runtime_settings 里没有 qwen.api_key（先在设置页填好）")
    return OAICompatibleLLM(PRESET_ENDPOINTS["qwen"], model=model, api_key=key)


async def _measure(llm, segments: list[tuple[str, str]]) -> int:
    """把分段拼成一条 user 消息，取厂商返回的**真实** prompt_tokens。

    只用 user 角色（不掺 system/assistant）：固定开销由拟合里的截距吸收，
    这样"每型密度"才是可辨识的。
    """
    from lightcore.llm.base import ChatMessage, Role

    text = "\n".join(t for _, t in segments)
    msgs = [ChatMessage(role=Role.USER, content=text + "\n请回复 OK")]
    resp = await llm.chat(msgs, max_tokens=8)   # 只要 prompt_tokens，输出压到最小
    return int(resp.prompt_tokens or 0)


# ---------------- 拟合与评估 ----------------


def _chars_by_kind(segments: list[tuple[str, str]]) -> dict[str, int]:
    d = dict.fromkeys(KINDS, 0)
    for kind, text in segments:
        d[kind] = d.get(kind, 0) + len(text)
    return d


def _pred_current(chars: dict[str, int]) -> float:
    return sum(chars.values()) / CURRENT_DENSITY


def _pred_kind(chars: dict[str, int], dens: dict[str, float]) -> float:
    return sum(chars.get(k, 0) / dens[k] for k in KINDS if dens.get(k))


def _rel(pred: float, real: float) -> float:
    return abs(pred - real) / max(1.0, real)


def _fit_density(rows: list[dict]) -> dict[str, float]:
    """按类型拟合密度：对每型只用"该型单独出现"的样本来估（避免共线）。

    做法：对每型取该型样本里**最大两个尺寸**的 `chars/real` 的中位数 ——
    大尺寸下固定开销占比最小，最接近渐近密度；取中位数抗单点抖动。
    """
    dens: dict[str, float] = {}
    for kind in KINDS:
        pts = [r["chars"][kind] / r["real"] for r in rows
               if list(r["chars"].values()).count(0) == len(KINDS) - 1 and r["chars"][kind] > 0]
        if len(pts) >= 2:
            pts.sort()
            dens[kind] = statistics.median(pts[-2:])
    return dens


def main() -> int:
    ap = argparse.ArgumentParser(description="T-01 token 估算标定探针 v2（按内容类型）")
    ap.add_argument("--model", default="qwen3.6-flash", help="用于采真值的模型")
    ap.add_argument("--max-calls", type=int, default=40, help="最多真实调用次数（防超预算）")
    ap.add_argument("--sleep", type=float, default=0.3, help="每次调用间隔秒")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("", encoding="utf-8")

    _say("=" * 86)
    _say("T-01-A token 估算标定 v2（真值 = 厂商 prompt_tokens；拟合维度 = **内容类型**）")
    _say(f"  模型：{args.model}（Qwen 免费额度）· 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    _say("=" * 86)

    llm = _build_llm(args.model)
    samples = _samples()[: args.max_calls]
    _say(f"\n① 采集（{len(samples)} 条；①单类型×多长度 给渐近密度，②混合 验证可加性）")
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, (label, segments) in enumerate(samples, 1):
        chars = _chars_by_kind(segments)
        try:
            real = asyncio.run(_measure(llm, segments))
        except Exception as e:  # noqa: BLE001  单条失败不中断整轮
            _say(f"  [{i:02d}] {label:<22} ✗ {type(e).__name__}: {str(e)[:60]}")
            continue
        rows.append({"label": label, "real": real, "chars": chars,
                     "cur": _pred_current(chars)})
        _say(f"  [{i:02d}] {label:<22} real={real:>6} 现状est={_pred_current(chars):>7.0f}"
             f"  real/est={real / max(1, _pred_current(chars)):.2f}")
        if args.sleep:
            time.sleep(args.sleep)
    _say(f"  采集完成：{len(rows)}/{len(samples)} 条，耗时 {time.perf_counter() - t0:.1f}s")
    if len(rows) < 8:
        _say("\n✗ 有效样本不足（<8）。检查 key/额度/网络。")
        return 2

    # ---- ② 现状模型 ----
    cur = [_rel(r["cur"], r["real"]) for r in rows]
    _say("\n② 现状模型（len/2.5，单一系数）")
    _say(f"  相对误差  p50={statistics.median(cur) * 100:.1f}%  "
         f"mean={statistics.fmean(cur) * 100:.1f}%  max={max(cur) * 100:.1f}%")
    _say(f"  ≤±10% 的比例：{sum(1 for e in cur if e <= 0.10)}/{len(cur)}")

    # ---- ③ 按内容类型的渐近密度 ----
    _say("\n③ 各内容类型的渐近密度（字符/token；只用单类型样本、取最大两档中位数）")
    dens = _fit_density(rows)
    for kind in KINDS:
        if kind in dens:
            _say(f"  {kind:<6} 密度 ≈ {dens[kind]:.2f} 字符/token"
                 f"（现状固定 {CURRENT_DENSITY} → 该型估算{'偏小' if dens[kind] < CURRENT_DENSITY else '偏大'}"
                 f" {abs(1 - CURRENT_DENSITY / dens[kind]) * 100:.0f}%）")

    # ---- ④ 分型模型 + LOO ----
    kind_err = [_rel(_pred_kind(r["chars"], dens), r["real"]) for r in rows]
    loo: list[float] = []
    for i in range(len(rows)):
        train = [r for j, r in enumerate(rows) if j != i]
        d_i = _fit_density(train)
        if len(d_i) == len(KINDS):
            loo.append(_rel(_pred_kind(rows[i]["chars"], d_i), rows[i]["real"]))
    _say("\n④ 分型模型（按 kind 选密度）")
    _say(f"  全量拟合误差 p50={statistics.median(kind_err) * 100:.1f}%  "
         f"max={max(kind_err) * 100:.1f}%")
    if loo:
        _say(f"  **留一法(LOO) p50={statistics.median(loo) * 100:.1f}%  "
             f"mean={statistics.fmean(loo) * 100:.1f}%  max={max(loo) * 100:.1f}%**")
        _say(f"  LOO 内 ≤±10% 的比例：{sum(1 for e in loo if e <= 0.10)}/{len(loo)}")

    # ---- ⑤ 混合样本单独看（可加性）----
    _say("\n⑤ 混合样本（验证「分段求和」是否成立 —— 这是真实消息的形态）")
    for r in rows:
        non_zero = sum(1 for v in r["chars"].values() if v > 0)
        if non_zero >= 2:
            e_cur, e_new = _rel(r["cur"], r["real"]), _rel(_pred_kind(r["chars"], dens), r["real"])
            _say(f"  {r['label']:<22} real={r['real']:>6}  现状误差={e_cur * 100:5.1f}%  "
                 f"分型误差={e_new * 100:5.1f}%")

    # ---- ⑥ 判据 ----
    _say("\n⑥ 结论（判据：相对误差 p50 ≤ ±10%）")
    _say(f"  现状（单一系数 2.5）：p50 = {statistics.median(cur) * 100:.1f}% → "
         f"{'达标' if statistics.median(cur) <= 0.10 else '**不达标**'}")
    if loo:
        ok = statistics.median(loo) <= 0.10
        _say(f"  分型（按内容类型）：LOO p50 = {statistics.median(loo) * 100:.1f}% → "
             f"{'**达标 ✓**' if ok else '不达标 ✗'}")
    _say(f"\n结果已落盘：{OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
