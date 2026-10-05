"""T-07 探针 · 结构臂（H3）：**同一次检索里，同一个 query 被编码几次？**

## 为什么要单独一支、且**不加载 BGE-M3**
H3 是**结构**问题（调用次数），与嵌入模型无关 ⇒ 用零依赖的 `FakeHashEmbedder`
（确定性哈希向量）即可，**秒级、零 GPU、零显存**。这样能把 GPU 时间全留给 H1/H2。

## 读码推断（待本脚本证/伪）
`CompositeRAGPipeline.query`（`chat_service.py:218-228`）**逐文档**调 `p.query()`，
而每个 `p.query` → `_hybrid.retrieve` → `embedder.encode([query])`（`pipeline.py:159`），
且各管线**共用同一 embedder 实例**（`document_service._get_embedder` 单例缓存）
⇒ 预期 **N 篇文档 = 同一 query 串被编码 N 次**。

## 跑法
    .venv/Scripts/python.exe scripts/probe_t07_structure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lightcore.rag.embedder import FakeHashEmbedder  # noqa: E402
from lightcore.rag.pipeline import RAGPipeline  # noqa: E402
from papermate.services.chat_service import CompositeRAGPipeline  # noqa: E402

N_DOCS = 3                        # 关联文档数（复现"多文档会话"）
CHUNKS_PER_DOC = 8


class CountingEmbedder(FakeHashEmbedder):
    """统计 `encode` 的**调用次数**与**每次的文本** —— 用来证明"同一串被编 N 次"。"""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim)
        self.calls: list[list[str]] = []

    def encode(self, texts):  # noqa: ANN001, ANN201
        self.calls.append(list(texts))
        return super().encode(texts)


def _build(emb, tag: str) -> RAGPipeline:
    p = RAGPipeline(embedder=emb, reranker=None, use_cross_encoder=False, use_bge_sparse=False)
    items = [(f"{tag}-c{i}", f"{tag} 的第 {i} 段正文内容，用于结构探针与缓存行为验证。", 1)
             for i in range(CHUNKS_PER_DOC)]
    p.add_chunks(items, doc_id=tag)
    return p


def main() -> None:
    emb = CountingEmbedder()
    pipelines = [_build(emb, f"doc{i}") for i in range(N_DOCS)]
    comp = CompositeRAGPipeline(pipelines, doc_ids=list(range(N_DOCS)))

    def run(label: str, q: str) -> None:
        emb.calls.clear()
        res = comp.query(q, top_k=3)
        n = len(emb.calls)
        texts = {t for c in emb.calls for t in c}
        print(f"\n[{label}]")
        print(f"  query              = {q!r}")
        print(f"  encode 调用次数     = {n}   （文档数 N = {N_DOCS}）")
        print(f"  每次编码的文本种数  = {len(texts)}  → {'**同一串被重复编码**' if len(texts) == 1 and n > 1 else '（无重复编码）'}")
        print(f"  编码的文本          = {sorted(texts)}")
        print(f"  返回命中条数        = {len(res)}")

    print("=" * 74)
    print(f"T-07 探针 · 结构臂：N = {N_DOCS} 篇文档，每篇 {CHUNKS_PER_DOC} chunk（FakeHashEmbedder，无 GPU）")
    print("=" * 74)

    q1 = "分块大小是怎么定的"
    run("① 首次查询（必然 miss）", q1)
    run("② 再问**完全相同**的 query（精确缓存应命中 ⇒ 0 次编码）", q1)
    run("③ 语义相同、**字面不同**（精确缓存必 miss ⇒ 语义缓存要探测就得先编码）",
        "分块的大小是如何确定的")

    print("\n" + "=" * 74)
    print("判据：②应 = 0 次（精确缓存生效）；①③ 应 = N 次 —— 若 ①/③ 为 1 次，则读码推断被证伪。")
    print("=" * 74)


if __name__ == "__main__":
    main()
