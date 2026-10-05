"""T-01 token 估算标定：用**厂商真实 `prompt_tokens`** 守住 `_rough_tokens` 的误差。

**为什么这些"魔法数字"有资格写进测试**：测试里的 `real` 值不是估出来的，
而是 2026-09-25 用 `scripts/probe_token_calibration.py` 对 `qwen3.6-flash`
**真实调用**采集回来的 `prompt_tokens`（原始记录：`eval/t01_calibration.txt`）。
样本文本是确定性生成的（无随机数），所以可在测试里逐字节复现 → 断言才有意义。

**守三件事**
1. **误差达标**：按字符类分档后，相对误差 ≤ ±10%（旧单一系数是 34% 量级）。
2. **结构不能被改回单一系数**：中文与英文的偏差**方向相反**（中文低估、英文高估）→
   任何单一系数都补不平。这条用"两个方向的比值"断言，防止有人"顺手改回 len/2.5"。
3. **不能回归"系统性偏低 2 倍"的旧说法**：旧文档断言过"稳定偏低 2 倍"，
   本次标定证明那只对中文成立 —— 英文是反的。

⚠️ 边界：`real` 值来自 **Qwen** 分词器。换厂商需重跑探针重标，不能直接套用（见 `base.py` 注释）。
"""

from __future__ import annotations

import pytest

from lightcore.llm.base import (
    _TOKEN_DENSITY_ALNUM,
    _TOKEN_DENSITY_CJK,
    _TOKEN_DENSITY_OTHER,
    _rough_tokens,
)

# ---------------- 与探针逐字节一致的确定性语料（勿改：改了 real 值就对不上） ----------------

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


def _rep(s: str, n_chars: int) -> str:
    k = max(1, n_chars // len(s) + 1)
    return (s * k)[:n_chars]


#: (说明, 文本, 厂商真实 prompt_tokens) —— real 来源：eval/t01_calibration.txt（2026-09-25）
FIXTURES = [
    ("中文正文 2000 字符", _rep(_ZH, 2000), 1222),
    ("英文正文 2000 字符", _rep(_EN, 2000), 377),
    ("代码 2000 字符", _rep(_CODE, 2000), 576),
    ("JSON 2000 字符", _rep(_JSON, 2000), 899),
    ("中英混合 4:4", _rep(_ZH, 800) + "\n" + _rep(_EN, 800), 645),
    ("英 JSON 混合 6:4", _rep(_EN, 1200) + "\n" + _rep(_JSON, 800), 585),
]


def _rel_err(est: int, real: int) -> float:
    return abs(est - real) / real


# ---------------- 1. 误差达标 ----------------


@pytest.mark.parametrize(("desc", "text", "real"), FIXTURES, ids=[f[0] for f in FIXTURES])
def test_estimate_within_ten_percent(desc: str, text: str, real: int) -> None:
    """★ 核心：按字符类分档后，相对误差 ≤ ±10%（判据来自卡片 ③）。"""
    est = _rough_tokens(text)
    assert _rel_err(est, real) <= 0.10, f"{desc}: est={est} real={real} 误差 {_rel_err(est, real):.1%}"


def test_old_single_coefficient_would_fail() -> None:
    """反向自检：若退回旧单一系数 `len/2.5`，这些样本**必须**不达标。

    否则说明本组 fixture 没有区分力（"测试通过"可能只是因为样本太宽容）。
    """
    bad = [d for d, t, r in FIXTURES if abs(len(t) / 2.5 - r) / r <= 0.10]
    assert len(bad) <= 1, f"旧口径竟然在多数样本上达标：{bad}（fixture 缺乏区分力）"


# ---------------- 2. 结构：偏差方向相反，单一系数补不平 ----------------


def test_zh_and_en_bias_have_opposite_sign() -> None:
    """★ 中文被低估、英文被高估 —— 这正是"单一系数无解"的证据。

    旧文档曾断言"系统性偏低约 2 倍"；本断言把它钉死为"只对中文成立"。
    若有人改回单一系数，中文与英文不可能同时落在 ±10% 内。
    """
    zh_est = _rough_tokens(_rep(_ZH, 2000))
    en_est = _rough_tokens(_rep(_EN, 2000))
    zh_real = next(r for d, _t, r in FIXTURES if d.startswith("中文"))
    en_real = next(r for d, _t, r in FIXTURES if d.startswith("英文"))
    # 中文：旧口径（len/2.5 = 800）低于真值 → 低估；英文：旧口径（800）高于真值 → 高估
    assert 2000 / 2.5 < zh_real, "中文应当被旧口径低估"
    assert 2000 / 2.5 > en_real, "英文应当被旧口径高估"
    # 新口径两边都要准
    assert _rel_err(zh_est, zh_real) <= 0.10
    assert _rel_err(en_est, en_real) <= 0.10
    # 同样的字符数，真实 token 数差 3 倍以上（这正是不能共用一个系数的原因）
    assert zh_real / en_real > 3.0, f"中英密度差异应当显著（实测 {zh_real}/{en_real}）"


# ---------------- 3. 常数与多模态路径 ----------------


def test_density_constants_are_calibrated_values() -> None:
    """守护常数：改这些值必须有新的标定依据（否则误差会静默漂移）。"""
    assert (_TOKEN_DENSITY_CJK, _TOKEN_DENSITY_ALNUM, _TOKEN_DENSITY_OTHER) == (1.64, 11.87, 1.36)


def test_empty_and_multiblock_paths_unchanged() -> None:
    """旧行为不能退化：空串 0、列表块累加、图片块按 384。"""
    assert _rough_tokens("") == 0
    assert _rough_tokens([]) == 0
    text_blocks = [{"type": "text", "text": "你好"}]
    assert _rough_tokens(text_blocks + [{"type": "image_url", "image_url": {"url": "x"}}]) \
        == _rough_tokens(text_blocks) + 384


def test_estimate_is_monotonic_in_length() -> None:
    """同一内容变长，估算必须单调不减（防止正则/取整引入非单调）。"""
    prev = 0
    for n in (100, 300, 900, 2700, 8100):
        cur = _rough_tokens(_rep(_ZH, n))
        assert cur >= prev
        prev = cur
