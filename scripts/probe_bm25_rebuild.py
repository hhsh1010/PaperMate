"""O-11 探针：BM25 索引构建 O(n²) 冗余重建的耗时占比实测（只读，不写任何数据）。

假设：pipeline.add_chunks 按 batch_size=64 分批，每批 add() 触发一次对全部累积文档的
BM25Okapi 全量重建 ⇒ 658 chunks / 11 批 ⇒ 累积重建量 ≈ 6.3×（重建次数比，非耗时比）。
本探针把 add() 的两个组成部分分开计时：
  A) 分词（jieba / 字符级，增量的，本来就不可避免）
  B) BM25Okapi 统计量构建（每批对全部累积文档重复做 —— 冗余所在）
并对比「现行分批重建」vs「惰性单次重建」的总耗时与检索结果逐位一致性。

数据源：data/papermate.db 的 chunks 表真实文本（只读）；不足 658 条时用合成文本补齐。
判据：B 占比显著（>20%）才有改造价值；等价性 = Top-K 索引位次完全一致。
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_bm25 import BM25Okapi  # noqa: E402

from lightcore.rag.bm25 import BM25Retriever, jieba_tokenize  # noqa: E402

DB = Path(__file__).resolve().parents[1] / "data" / "papermate.db"
N_TARGET = 658
BATCH = 64
QUERIES = ["注意力机制 的 作用", "实验 结果 对比", "模型 训练 方法", "图 表 数据"]


def load_texts() -> list[str]:
    texts: list[str] = []
    if DB.exists():
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT content FROM chunks").fetchall()
            texts = [r[0] for r in rows if r[0]]
        finally:
            conn.close()
    n_real = len(texts)
    while len(texts) < N_TARGET:  # 不足则循环复用真实文本补齐（保持长度分布）
        texts = texts + texts
    texts = texts[:N_TARGET]
    print(f"[data] real chunks = {n_real}, total = {len(texts)} (target {N_TARGET})")
    return texts


def probe_current(texts: list[str]) -> tuple[float, float, BM25Retriever]:
    """复刻现行行为：每批 add() → 分词(增量) + BM25Okapi(全量累积重建)，分开计时。"""
    r = BM25Retriever(tokenize=jieba_tokenize)
    tok_t = build_t = 0.0
    for s in range(0, len(texts), BATCH):
        batch = texts[s : s + BATCH]
        t0 = time.perf_counter()
        r._docs.extend(batch)
        r._tokenized.extend(r._tokenize(t) for t in batch)
        tok_t += time.perf_counter() - t0
        t0 = time.perf_counter()
        r._bm25 = BM25Okapi(r._tokenized)  # 现行：对全部累积文档重建
        build_t += time.perf_counter() - t0
    return tok_t, build_t, r


def probe_lazy(texts: list[str]) -> tuple[float, float, BM25Retriever]:
    """惰性：只增量分词，最后重建一次。"""
    r = BM25Retriever(tokenize=jieba_tokenize)
    tok_t = build_t = 0.0
    for s in range(0, len(texts), BATCH):
        batch = texts[s : s + BATCH]
        t0 = time.perf_counter()
        r._docs.extend(batch)
        r._tokenized.extend(r._tokenize(t) for t in batch)
        tok_t += time.perf_counter() - t0
    t0 = time.perf_counter()
    r._bm25 = BM25Okapi(r._tokenized)  # 惰性：仅一次
    build_t += time.perf_counter() - t0
    return tok_t, build_t, r


def dump_queries(r: BM25Retriever) -> list[list[int]]:
    out = []
    for q in QUERIES:
        out.append([i for i, _ in r.search(q, k=10)])
    return out


def main() -> None:
    texts = load_texts()
    print(f"[probe] batch_size={BATCH}, batches={-(-len(texts) // BATCH)}")

    jieba_tokenize("预热 jieba 词典，避免首次加载（~0.5s）污染第一臂计时")  # noqa: B018

    t0 = time.perf_counter()
    cur_tok, cur_build, cur_r = probe_current(texts)
    cur_total = time.perf_counter() - t0
    t0 = time.perf_counter()
    laz_tok, laz_build, laz_r = probe_lazy(texts)
    laz_total = time.perf_counter() - t0

    redundant_build = cur_build - laz_build
    print("\n=== 分批重建（现行） ===")
    print(f"  分词       = {cur_tok * 1000:8.1f} ms ({cur_tok / cur_total:.1%})")
    print(f"  BM25 重建  = {cur_build * 1000:8.1f} ms ({cur_build / cur_total:.1%})  <- 含冗余")
    print(f"  合计       = {cur_total * 1000:8.1f} ms")
    print("=== 惰性单次重建 ===")
    print(f"  分词       = {laz_tok * 1000:8.1f} ms ({laz_tok / laz_total:.1%})")
    print(f"  BM25 重建  = {laz_build * 1000:8.1f} ms ({laz_build / laz_total:.1%})")
    print(f"  合计       = {laz_total * 1000:8.1f} ms")
    print("=== 结论 ===")
    print(f"  冗余重建耗时 ≈ {redundant_build * 1000:.1f} ms，占现行 add 总耗时 "
          f"{redundant_build / cur_total:.1%}")
    print(f"  惰性化总提速 ≈ {cur_total / laz_total:.2f}x")

    cur_top = dump_queries(cur_r)
    laz_top = dump_queries(laz_r)
    ok = cur_top == laz_top
    print(f"\n[equivalence] Top-K 逐位一致 = {ok}")
    for q, a, b in zip(QUERIES, cur_top, laz_top, strict=True):
        print(f"  {q!r}: {'OK' if a == b else f'MISMATCH {a} vs {b}'}")
    print("RESULT=EQUIV" if ok else "RESULT=MISMATCH")


if __name__ == "__main__":
    main()
