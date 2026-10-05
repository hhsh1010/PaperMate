"""PaperMate · 轻量检索评测脚手架（2026-09-01，任务 #175）

不依赖联网、不依赖真实大模型：在本地语料上跑 RAGPipeline.query，按标注集（qrels）计算
检索质量指标，用于对比 fake（哈希占位）与 real（BGE-M3）两种 embedding 底座的召回差异。

用法：
  # 1) 自带 demo 语料（2 篇合成文档 + 标注），开箱即跑，验证脚本本身：
  python eval/retrieval_eval.py --demo --embedder fake
  python eval/retrieval_eval.py --demo --embedder real   # 需已下载 BGE-M3 权重

  # 2) 自有语料 + 标注：
  python eval/retrieval_eval.py --corpus data/parsed --qrels eval/my_qrels.jsonl --embedder real --k 5

参数：
  --corpus  语料目录（目录下 *.txt 各算一篇）或单个 .txt 文件
  --qrels   标注集 JSONL，每行 {"q": "查询", "relevant": ["文档 stem", ...]}
            relevant 用「文档文件名去扩展名」匹配（与 --corpus 内文件名一致）
  --embedder fake | real（默认 fake；real 需 rag 依赖与 ~2.2GB 权重）
  --k       取前 k 个结果计算 precision@k（默认 5）
  --out     可选：把评测报告写到的 JSON 路径

指标：
  - precision@k：前 k 个命中结果所属文档中、落在 relevant 集合的比例（文档级）
  - recall@5  ：前 5 个结果覆盖的 relevant 文档比例
  - MRR       ：首个相关文档排名的倒数均值
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _stem_from_chunk_id(cid: str) -> str:
    """chunk id 形如 'doc_stem-c12'，去掉末尾 '-c<数字>' 还原文档 stem。"""
    return re.sub(r"-c\d+$", "", cid)


def _load_corpus(corpus: Path) -> list[tuple[str, str]]:
    """返回 [(stem, text), ...]。目录取 *.txt；单文件取自身。"""
    files: list[Path]
    if corpus.is_dir():
        files = sorted(corpus.glob("*.txt"))
    elif corpus.suffix.lower() == ".txt":
        files = [corpus]
    else:
        raise SystemExit(f"[错误] --corpus 需为目录或 .txt 文件：{corpus}")
    if not files:
        raise SystemExit(f"[错误] 未找到语料 .txt 文件：{corpus}")
    return [(p.stem, p.read_text(encoding="utf-8")) for p in files]


def _load_qrels(qrels: Path) -> list[dict]:
    rows: list[dict] = []
    for line in qrels.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"[错误] qrels 为空：{qrels}")
    return rows


def _demo_corpus() -> list[tuple[str, str]]:
    return [
        ("attention", "自注意力机制（self-attention）通过查询、键、值三组向量计算序列内部各位置的相关性权重，"
                      "使模型能够直接建模任意两个 token 之间的长距离依赖。Transformer 摒弃了循环结构，"
                      "完全基于注意力堆叠，训练可高度并行。多头注意力把表示投影到多个子空间并行计算再拼接。"),
        ("cnn", "卷积神经网络（CNN）使用局部感受野与权值共享的卷积核在图像上滑动提取边缘、纹理等局部特征，"
                 "配合池化层逐步降低分辨率并扩大感受野。LeNet、AlexNet、ResNet 是其代表性结构，"
                 "在视觉识别任务上表现优异，但与 RNN 不同，卷积对长序列全局依赖的建模能力较弱。"),
    ]


def _demo_qrels() -> list[dict]:
    return [
        {"q": "什么是自注意力机制，它如何建模长距离依赖？", "relevant": ["attention"]},
        {"q": "卷积神经网络靠什么提取图像特征？", "relevant": ["cnn"]},
        {"q": "Transformer 为什么能并行训练？", "relevant": ["attention"]},
    ]


def evaluate(corpus: list[tuple[str, str]], qrels: list[dict], embedder_kind: str, k: int) -> dict:
    from lightcore.rag.pipeline import RAGPipeline

    # 延迟导入 BGE-M3：仅在 real 路径引入 sentence-transformers，fake 路径零依赖即可跑（含本沙箱）
    if embedder_kind == "real":
        from lightcore.rag.embedder import BgeM3Embedder

        emb = BgeM3Embedder()
    else:
        from lightcore.rag.embedder import FakeHashEmbedder

        emb = FakeHashEmbedder()
    pipeline = RAGPipeline(embedder=emb)

    print(f"[语料] 载入 {len(corpus)} 篇文档，embedding={embedder_kind}")
    for stem, text in corpus:
        n = pipeline.add_document(text, doc_id=stem)
        print(f"  - {stem}: {n} 块")

    per_query: list[dict] = []
    p_at_k_list: list[float] = []
    r_at_5_list: list[float] = []
    mrr_list: list[float] = []

    for item in qrels:
        q = item["q"]
        relevant = set(item.get("relevant", []))
        results = pipeline.query(q, top_k=max(k, 5))
        pred_stems_at_k = [_stem_from_chunk_id(r.id) for r in results[:k]]
        pred_stems_at_5 = [_stem_from_chunk_id(r.id) for r in results[:5]]

        hit_at_k = sum(1 for s in set(pred_stems_at_k) if s in relevant)
        denom_k = len(set(pred_stems_at_k)) or 1
        p_at_k = hit_at_k / denom_k if relevant else 0.0

        denom_rel = len(relevant) or 1
        r_at_5 = len(set(pred_stems_at_5) & relevant) / denom_rel if relevant else 0.0

        mrr = 0.0
        for rank, r in enumerate(results, start=1):
            if _stem_from_chunk_id(r.id) in relevant:
                mrr = 1.0 / rank
                break

        p_at_k_list.append(p_at_k)
        r_at_5_list.append(r_at_5)
        mrr_list.append(mrr)
        per_query.append({
            "q": q,
            "relevant": sorted(relevant),
            "top": [{"stem": _stem_from_chunk_id(r.id), "score": round(r.score, 4)} for r in results[:k]],
            "precision@k": round(p_at_k, 4),
            "recall@5": round(r_at_5, 4),
            "mrr": round(mrr, 4),
        })
        print(f"[查询] {q[:28]}…  P@{k}={p_at_k:.2f}  R@5={r_at_5:.2f}  MRR={mrr:.2f}")

    report = {
        "embedder": embedder_kind,
        "k": k,
        "num_docs": len(corpus),
        "num_queries": len(qrels),
        "avg_precision@k": round(sum(p_at_k_list) / len(p_at_k_list), 4) if p_at_k_list else 0.0,
        "avg_recall@5": round(sum(r_at_5_list) / len(r_at_5_list), 4) if r_at_5_list else 0.0,
        "avg_mrr": round(sum(mrr_list) / len(mrr_list), 4) if mrr_list else 0.0,
        "per_query": per_query,
    }
    print(f"\n[汇总] embedder={embedder_kind}  "
          f"avgP@{k}={report['avg_precision@k']}  "
          f"avgR@5={report['avg_recall@5']}  avgMRR={report['avg_mrr']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperMate 轻量检索评测")
    parser.add_argument("--corpus", default=None, help="语料目录或 .txt 文件")
    parser.add_argument("--qrels", default=None, help="标注集 JSONL（每行 {q, relevant}）")
    parser.add_argument("--embedder", choices=["fake", "real"], default="fake")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default=None, help="评测报告输出 JSON 路径")
    parser.add_argument("--demo", action="store_true", help="使用内置合成语料 + 标注（开箱即跑）")
    args = parser.parse_args()

    if args.demo:
        corpus = _demo_corpus()
        qrels = _demo_qrels()
    else:
        if not args.corpus or not args.qrels:
            raise SystemExit("[错误] 非 demo 模式需同时提供 --corpus 与 --qrels；或加 --demo 试跑")
        corpus = _load_corpus(Path(args.corpus).resolve())
        qrels = _load_qrels(Path(args.qrels).resolve())

    report = evaluate(corpus, qrels, args.embedder, args.k)

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[写出] 评测报告 → {args.out}")


if __name__ == "__main__":
    main()
