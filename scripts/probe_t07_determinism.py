"""T-07 记忆化 · 确定性检查（R6）：`encode` 的输出是否**逐位可复现**？

## 为什么必须测
"记忆化 = 零正确性风险"这个结论**建立在"输出逐位相同"**之上。但：
- fp16 GPU 推理的逐次确定性**没有契约保证**；
- 更关键：若 memo 存的是**批量**编码出的向量，与**单条**重算可能**有微小差异**
  ⇒ 可能改变近邻 **tie**（行为变更）。

## 测什么
1. **solo vs solo**：同一文本、同一实例、单条编码两次 → 是否 bitwise 相等。
2. **solo vs batch**：同一文本，单条编码 vs 混在 32 条批次里编码 → 是否 bitwise 相等、
   最大绝对误差多少。（量化"索引期污染"（R1-B）的风险）

## 跑法（约 30s，单进程单线程，零 API）
    .venv/Scripts/python.exe scripts/probe_t07_determinism.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lightcore.rag.embedder import BgeM3Embedder  # noqa: E402

Q = "分块大小是怎么定的"
FILLER = [f"填充文本第 {i} 条，用于构造批量编码。" for i in range(31)]


def _cmp(a: list[float], b: list[float]) -> tuple[bool, float]:
    bit = len(a) == len(b) and all(x == y for x, y in zip(a, b, strict=False))
    mx = max(abs(x - y) for x, y in zip(a, b, strict=False)) if a and b else float("nan")
    return bit, mx


def main() -> None:
    emb = BgeM3Embedder()
    print(f"device={emb.device} dim={emb.dim}")

    v1 = emb.encode([Q])[0]
    v2 = emb.encode([Q])[0]
    bit, mx = _cmp(v1, v2)
    print(f"\n[1] solo vs solo：bitwise 相等 = {bit}，最大绝对差 = {mx:.3e}")
    print("    ⇒ " + ("**逐位可复现**（记忆化的'零正确性风险'成立）" if bit
                      else "**非逐位相同** ⇒ 记忆化只是'数值等价'，需在文档中如实降级"))

    batch = emb.encode([Q, *FILLER])
    vb = batch[0]
    bit_b, mx_b = _cmp(v1, vb)
    print(f"\n[2] solo vs batch(32)：bitwise 相等 = {bit_b}，最大绝对差 = {mx_b:.3e}")
    print("    ⇒ " + ("批量与单条**完全一致**（R1-B 的污染风险可忽略）" if bit_b
                      else "批量与单条**有微小差异** ⇒ 佐证 R1：memo **不能**混入批量算出的向量"))

    from lightcore.rag.embedder import FakeHashEmbedder  # noqa: E402

    f = FakeHashEmbedder()
    f1, f2 = f.encode([Q])[0], f.encode([Q])[0]
    fb = f.encode([Q, *FILLER])[0]
    print(f"\n[3] FakeHashEmbedder：solo==solo {_cmp(f1, f2)[0]} · solo==batch {_cmp(f1, fb)[0]}")


if __name__ == "__main__":
    main()
