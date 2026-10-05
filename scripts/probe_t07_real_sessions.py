"""T-07 探针 · H2 修订版：用**真实多轮会话提问**测阈值通过率。

## 为什么换数据源（如实说明）
第一版 H2 用"`gold` 有交集"当"同目标"的代理，结果**分辨力太弱**（同目标 p50 0.493 vs 对照 0.445），
**答不了**"真同义改写查询的余弦分布"这个问题 —— 因为"指向同一段证据"**不等于**"同一意图"
（同一段话可被问成"是多少"或"为什么"）。
⇒ 改用**库里真实的多轮提问**：同一会话内的相邻提问，才是卡片所说"同一意图被反复改写提问"的**真实样本**。

## 数据
`data/papermate.db` 的 `messages`（role='user'）—— **只读**。实测：13 个会话 / 245 条提问
（session 7 = 91、14 = 59、15 = 33 …）。

## 三组对照
- **A 相邻对**：同一会话内**相邻**两条用户提问（最可能"同意图改写"）
- **B 同会话对**：同一会话内**任意**两对（更宽；含"换话题"的噪声）
- **C 对照对**：**跨会话**随机对（几乎必不同意图）
⇒ 看 A 与 C 是否可分、以及各阈值下的通过率。

## 跑法
    .venv/Scripts/python.exe scripts/probe_t07_real_sessions.py
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lightcore.rag.embedder import BgeM3Embedder  # noqa: E402

DB = ROOT / "data" / "papermate.db"


def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False)) / (na * nb)


def _load() -> dict[int, list[str]]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select session_id, content from messages where role='user' order by session_id, id"
        ).fetchall()
    finally:
        con.close()
    by: dict[int, list[str]] = {}
    for sid, content in rows:
        t = (content or "").strip()
        if t:
            by.setdefault(sid, []).append(t)
    return by


def main() -> None:
    by = _load()
    total = sum(len(v) for v in by.values())
    print("=" * 78)
    print(f"H2 修订版 · 真实多轮提问：{len(by)} 个会话 / {total} 条用户提问（只读 DB）")
    print("=" * 78)

    # 精确重复率（现有精确缓存能吃下的部分）——作为"语义层还剩多少空间"的对照
    n_dup = sum(len(v) - len(set(v)) for v in by.values())
    print(f"[背景] 同一会话内**完全重复**的提问 = {n_dup} 条（{n_dup / max(1, total):.1%}）")
    print("        ⇒ 这部分**精确缓存已经吃下**，语义层只能吃「字面不同」的剩余部分")

    t0 = time.perf_counter()
    emb = BgeM3Embedder()
    print(f"[load] BgeM3Embedder device={emb.device} 耗时 {time.perf_counter() - t0:.1f}s")

    uniq = sorted({t for v in by.values() for t in v})
    vecs: dict[str, list[float]] = {}
    t1 = time.perf_counter()
    for i in range(0, len(uniq), 32):
        batch = uniq[i:i + 32]
        for t, v in zip(batch, emb.encode(batch), strict=False):
            vecs[t] = v
    print(f"[encode] {len(uniq)} 条去重提问 编码耗时 {time.perf_counter() - t1:.1f}s")

    adj, same, cross = [], [], []
    for _sid, qs in by.items():
        for i in range(len(qs) - 1):
            if qs[i] != qs[i + 1]:
                adj.append(_cos(vecs[qs[i]], vecs[qs[i + 1]]))
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                if qs[i] != qs[j]:          # 完全重复的交给精确缓存，不混入语义口径
                    same.append(_cos(vecs[qs[i]], vecs[qs[j]]))
    sids = list(by)
    rng = random.Random(20260929)
    n_same_literal = 0
    target = max(1, len(adj))
    guard = 0
    while len(cross) < target and guard < target * 200:
        guard += 1
        a, b = rng.sample(sids, 2)
        x, y = rng.choice(by[a]), rng.choice(by[b])
        if x == y:
            # ★ 口径一致：**字面完全相同**的跨会话提问本来就该命中（无害），
            # 与 A 组的处理保持一致，**不计入错命中**。第一版漏了这一步 ⇒ 高估了错命中率。
            n_same_literal += 1
            continue
        cross.append(_cos(vecs[x], vecs[y]))
    print(f"[口径] 跨会话抽样中「字面完全相同」（无害、已剔除）= {n_same_literal} 个")

    def stats(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"[{name}] （无）")
            return
        s = sorted(xs)
        print(f"[{name}] n={len(s)} min={s[0]:.3f} p50={statistics.median(s):.3f} "
              f"p90={s[int(len(s) * 0.9)]:.3f} max={s[-1]:.3f}")

    print()
    stats("A 相邻对（最可能同意图）", adj)
    stats("B 同会话对（宽口径）", same)
    stats("C 对照对（跨会话）", cross)

    print(f"\n{'阈值':<7}{'A 相邻':>9}{'B 同会话':>10}{'C 对照':>9}{'C 是 A 的几倍':>14}")
    for t in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        pa = sum(1 for c in adj if c >= t) / max(1, len(adj))
        pb = sum(1 for c in same if c >= t) / max(1, len(same))
        pc = sum(1 for c in cross if c >= t) / max(1, len(cross))
        ratio = (pc / pa) if pa else float("inf")
        print(f"{t:<7.2f}{pa:>8.1%}{pb:>10.1%}{pc:>9.1%}{ratio:>14.2f}")

    print("\n★ 判读：'C 对照'= 错命中风险；'A 相邻'= 语义层可能吃到的命中率上限。")
    print("  'C 是 A 的几倍' > 1 ⇒ 误命中比有效命中更容易发生（该阈值不可用）。")
    print(json.dumps({"adj_n": len(adj), "same_n": len(same), "cross_n": len(cross),
                      "adj_p50": statistics.median(adj) if adj else None,
                      "cross_p50": statistics.median(cross) if cross else None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
