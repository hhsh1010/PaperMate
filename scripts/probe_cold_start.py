"""T-05 探针：冷启动耗时分解（六阶段计时）。

## 为什么必须先测
T-05 的前置分析把「单篇首次 get_pipeline 20~33s」归因于 **`load_state`（读 faiss + 反序列化）**，
并据此提出「mmap 加载」作为首要手段。但静态检查发现：**索引文件只有 100~170 KB/篇**
（`data/faiss/` 15 篇合计 4.4 MB）—— 读这么小的文件 + pickle 绝不可能是 20s。

⇒ 若不先测就动手，会照着**错误的归因**做一堆无效优化（mmap 对 4.4MB 文件收益≈0）。

## 六阶段
| # | 阶段 | 依赖 |
|---|---|---|
| 1 | 读 `.state` + `pickle.loads`（× 15 篇） | 纯 CPU |
| 2 | `faiss.read_index`（× 15 篇） | 纯 CPU |
| 3 | BM25 重建（jieba 分词全部 chunk + `BM25Okapi`） | 纯 CPU |
| 4 | `BgeM3Embedder()` 构造 | GPU / 2.2GB 权重 |
| 5 | `CrossEncoderReranker()` 构造 | GPU / 2.2GB 权重 |
| 6 | jieba 首次初始化（词典加载） | 纯 CPU |

**为什么不直接测端到端 `get_pipeline`**：它需要 DB 里的 doc 与 `_pipeline_signature` 精确匹配，
而 signature 绑定分块内容 + 运行时配置 —— 探针里重建 signature 会引入"测的不是真实路径"的风险。
分阶段测能把**每一步的成本**单独钉死，且不依赖 DB。

用法（需 GPU；约 1~2 分钟）：
    .venv/Scripts/python.exe scripts/probe_cold_start.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAISS_DIR = ROOT / "data" / "faiss"


def _t(label: str, fn):
    """计时执行 `fn()`，返回 (耗时秒, 结果)。"""
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"    {label:<48} {dt * 1000:>9.1f} ms")
    return dt, out


def main() -> None:
    states = sorted(FAISS_DIR.glob("*.state"))
    faiss_files = sorted(FAISS_DIR.glob("*.faiss"))
    if not states:
        raise SystemExit(f"没有找到 .state 文件：{FAISS_DIR}")

    print("=" * 78)
    print("T-05 冷启动耗时分解")
    print("=" * 78)
    print(f"索引目录：{FAISS_DIR}")
    print(f"  .state 文件 {len(states)} 个 · 合计 "
          f"{sum(f.stat().st_size for f in states) / 1024 / 1024:.2f} MB")
    print(f"  .faiss 文件 {len(faiss_files)} 个 · 合计 "
          f"{sum(f.stat().st_size for f in faiss_files) / 1024 / 1024:.2f} MB")

    print("\n--- 阶段 6：jieba 首次初始化（词典加载）---")
    t_jieba, _ = _t("import jieba + 首次分词触发词典加载", lambda: __import__("jieba").lcut("预热"))

    print("\n--- 阶段 1：读 .state + pickle 反序列化（× 15）---")
    def _load_states():
        bundles = []
        for f in states:
            bundles.append(pickle.loads(f.read_bytes()))
        return bundles

    t1, bundles = _t("read_bytes + pickle.loads ×15", _load_states)
    n_chunks = sum(len(b.get("chunks", [])) for b in bundles)
    n_texts = sum(len(b.get("texts", [])) for b in bundles)
    print(f"      → 反序列化出 {n_chunks} 个 chunk / {n_texts} 条 text")

    print("\n--- 阶段 2：faiss.read_index（× 15）---")
    def _read_faiss():
        import faiss

        idxs = [faiss.read_index(str(f)) for f in faiss_files]
        return idxs

    t2, idxs = _t("faiss.read_index ×15", _read_faiss)
    print(f"      → 读出 {len(idxs)} 个索引 · 向量总数 {sum(i.ntotal for i in idxs)}")

    print("\n--- 阶段 3：BM25 重建（jieba 分词全部 chunk + BM25Okapi）---")
    def _build_bm25():
        from lightcore.rag.bm25 import BM25Retriever, jieba_tokenize

        r = BM25Retriever(tokenize=jieba_tokenize)
        for b in bundles:
            r.add(b.get("texts", []))
        return r

    t3, _ = _t("jieba 分词 + BM25Okapi 构建", _build_bm25)
    print(f"      → 处理 {n_texts} 条 text")

    print("\n--- 阶段 4：BgeM3Embedder() 构造（GPU / 2.2GB 权重）---")
    from lightcore.rag.embedder import BgeM3Embedder

    t4, _emb = _t("BgeM3Embedder()", BgeM3Embedder)

    print("\n--- 阶段 5：CrossEncoderReranker() 构造（GPU / 2.2GB 权重）---")
    from papermate.runtime_settings import get_state

    st = get_state()
    print(f"      （档位取自 runtime_settings：fp16={st['rag_reranker_fp16']} "
          f"max_length={st['rag_reranker_max_length']}）")
    from lightcore.rag.rerank import CrossEncoderReranker

    t5, _rr = _t("CrossEncoderReranker()",
                 lambda: CrossEncoderReranker(max_length=st["rag_reranker_max_length"],
                                              fp16=st["rag_reranker_fp16"]))

    # ---- 阶段 7（2026-09-27 补）：首次**使用** vs 第二次使用 ----
    # ⚠️ 原探针只有 1~6，全部是「加载」，**从未跑过一次真实检索**。
    # 这正是 T-05 误判"预热 pipeline 收益有限"的根源：
    # 加载完 ≠ 能用得快 —— 首次检索还要付 embedder/reranker 的 CUDA kernel
    # 初始化 + faiss 首次搜索的开销。实测线上：第一次 retrieve 2148ms vs 后续 641ms。
    # 本阶段直接把"冷"与"热"的差量化出来，让"该不该预热 pipeline"有据可依。
    print("\n--- 阶段 7：首次**前向**冷 vs 热（embedder.encode）---")
    t7_cold = t7_warm = None
    try:
        # 用 encode 而非完整检索：不依赖文档/语料，**且能直接量出 CUDA kernel
        # 初始化的开销**（这正是"首次检索慢 1.5s"的主要成分）。
        t7_cold, _ = _t("第 1 次 encode（冷）", lambda: _emb.encode(["预热"]))
        t7_warm, _ = _t("第 2 次 encode（热）", lambda: _emb.encode(["预热"]))
        if t7_cold is not None and t7_warm is not None:
            print(f"      → 冷 {t7_cold * 1000:.0f}ms / 热 {t7_warm * 1000:.0f}ms "
                  f"（差 {(t7_cold - t7_warm) * 1000:.0f}ms）")
    except Exception as e:  # noqa: BLE001
        print(f"      （阶段 7 跳过：{type(e).__name__}: {str(e)[:120]}）")

    cpu_total = t1 + t2 + t3
    gpu_total = t4 + t5
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"  CPU 部分（阶段 1+2+3）      {cpu_total:>7.2f} s   ← 前置分析归因给这部分")
    print(f"  GPU 部分（阶段 4+5）        {gpu_total:>7.2f} s   ← 前置分析标为「⚠️ 次要项」")
    print(f"  jieba 首次初始化（阶段 6）  {t_jieba:>7.2f} s")
    print("  ────────────────────────────────────")
    print(f"  进程内冷启动总计            {cpu_total + gpu_total + t_jieba:>7.2f} s")
    print(f"  GPU 占比                    {gpu_total / max(cpu_total + gpu_total + t_jieba, 1e-9):.0%}")

    verdict = {}
    if cpu_total < 2.0:
        verdict = {
            "结论": "前置分析归因错误",
            "证据": f"CPU 部分仅 {cpu_total:.2f}s，不可能构成 20~33s",
            "行动": "mmap 收益≈0；方案改为「lifespan 预热模型 + 加载不阻塞事件循环」",
        }
    elif cpu_total > 10.0:
        verdict = {
            "结论": "前置分析归因成立",
            "证据": f"CPU 部分 {cpu_total:.2f}s",
            "行动": "按原方案做 mmap + LRU + 预热",
        }
    else:
        verdict = {
            "结论": "归因部分成立（CPU 部分中等）",
            "证据": f"CPU 部分 {cpu_total:.2f}s",
            "行动": "先看阶段 3（BM25/jieba）能否降，再决定要不要 mmap",
        }
    print("\n" + "=" * 78)
    print("判据结论")
    print("=" * 78)
    for k, v in verdict.items():
        print(f"  {k}：{v}")

    out = ROOT / "eval" / "_t05_cold_start.json"
    out.write_text(json.dumps({
        "n_states": len(states), "n_faiss": len(faiss_files),
        "n_chunks": n_chunks, "n_texts": n_texts,
        "stage1_pickle_s": round(t1, 3), "stage2_faiss_s": round(t2, 3),
        "stage3_bm25_s": round(t3, 3), "stage4_embedder_s": round(t4, 3),
        "stage5_reranker_s": round(t5, 3), "stage6_jieba_s": round(t_jieba, 3),
        # 阶段 7：首次前向的冷开销（**T-05b 的立项依据**；原探针缺这一阶段）
        "stage7_encode_cold_s": None if t7_cold is None else round(t7_cold, 3),
        "stage7_encode_warm_s": None if t7_warm is None else round(t7_warm, 3),
        "stage7_cold_overhead_s": (
            None if (t7_cold is None or t7_warm is None) else round(t7_cold - t7_warm, 3)
        ),
        "cpu_total_s": round(cpu_total, 3), "gpu_total_s": round(gpu_total, 3),
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()