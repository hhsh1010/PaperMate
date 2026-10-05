#!/usr/bin/env python
"""Spike M3-b：真实 BGE-M3 语义检索复测（docs/05 §RAG / docs/13 Spike S2）

目的：对比 FakeHashEmbedder（占位、非语义）与 BgeM3Embedder（真实 BGE-M3，CPU）在
「语义改写查询」上的检索精度，验证真实嵌入能纠正 lexical 陷阱、体现语义排序差异。

运行（需先装 rag extra + 走 HF 镜像下载权重）：
    HF_ENDPOINT=https://hf-mirror.com .venv/Scripts/python.exe scripts/spike_m3_bgem3.py

参数：--mode both|fake|real   默认 both
"""
from __future__ import annotations

import argparse
import sys
import time

from lightcore.rag.embedder import BgeM3Embedder, FakeHashEmbedder
from lightcore.rag.pipeline import RAGPipeline

# 受控语料：8 个主题各异的短段落，每段成 1 个 chunk（id 形如 c1-c0）
CORPUS = [
    ("c1", "Agent 循环采用 ReAct 范式：模型先思考，再决定调用工具，然后观察结果，循环往复直到给出答案。"),
    ("c2", "工具的权限分为安全、网络、沙箱与管理员四级，代码执行必须放在 Docker 沙箱里运行。"),
    ("c3", "长期记忆用向量存储，按相关性、重要性与时效性三者加权打分来召回最相关的历史。"),
    ("c4", "混合检索把向量检索和 BM25 排名用倒数排名融合法合并，融合参数 k 设为六十。"),
    ("c5", "成本方面主用 DeepSeek 模型，本地做嵌入和重排不花钱，单次对话控制在三毛钱以内。"),
    ("c6", "RAG 先用 PDF 解析库抽取文本，再切成三百到五百 token 的语义片段做嵌入。"),
    ("c7", "反思模块让智能体评价自己上一步的结果是否满意，不满意就改写计划再试一次，最多两轮。"),
    ("c8", "全链路贯穿一个 trace_id，每次模型请求和工具调用都被记录，方便排查和评测质量。"),
]

# 语义改写查询：与原文用词尽量不重合，但语义等价；ground-truth = 对应 cN
QUERIES = [
    ("c1", "智能体是怎么一步步推理并选择要执行什么动作的？"),
    ("c2", "运行不可信代码时应该放在哪种隔离环境里？"),
    ("c3", "系统如何衡量一条旧笔记现在还有没有用？"),
    ("c4", "怎样把语义搜索和关键词搜索的结果综合到一起？"),
    ("c5", "怎么把每次问答的花费压到很低？"),
    ("c6", "把论文变成可检索的小段之前要先做什么处理？"),
    ("c7", "如果智能体发现自己答错了，它会怎么自我纠正？"),
    ("c8", "出了问题时怎么回溯是哪一步模型调用导致的？"),
]


def build_pipeline(embedder) -> RAGPipeline:
    pipe = RAGPipeline(embedder=embedder)
    for cid, text in CORPUS:
        n = pipe.add_document(text, doc_id=cid)
        assert n == 1, f"{cid} 应切成 1 块，实际 {n}"
    assert len(pipe._chunks) == len(CORPUS), "chunk 数应与语料数一致"
    return pipe


def dense_top1(pipe: RAGPipeline, embedder, query: str):
    qv = embedder.encode([query])[0]
    res = pipe._vidx.search(qv, k=len(pipe._chunks))
    return pipe._chunks[res[0][0]].id if res else None


def hybrid_top1(pipe: RAGPipeline, query: str):
    out = pipe.query(query, top_k=1)
    return out[0].id if out else None


def bm25_top1(pipe: RAGPipeline, query: str):
    res = pipe._bm25.search(query, k=len(pipe._chunks))
    return pipe._chunks[res[0][0]].id if res else None


def eval_embedder(name: str, embedder) -> None:
    print(f"\n========== 评估嵌入器：{name} (dim={embedder.dim}) ==========")
    pipe = build_pipeline(embedder)
    dense_hit = hybrid_hit = 0
    print(f"{'query→target':<14}{'BM25@1':<10}{'dense@1':<10}{'hybrid@1':<10}正确?")
    for gt, q in QUERIES:
        target = f"{gt}-c0"
        bm = bm25_top1(pipe, q)
        dn = dense_top1(pipe, embedder, q)
        hy = hybrid_top1(pipe, q)
        ok = hy == target
        if dn == target:
            dense_hit += 1
        if ok:
            hybrid_hit += 1
        mark = "✓" if ok else "✗"
        print(f"{q[:12]:<14}{bm:<10}{dn:<10}{hy:<10}{mark}")
    n = len(QUERIES)
    print(f"\n  dense-only  top-1 命中率 : {dense_hit}/{n} = {dense_hit/n:.0%}")
    print(f"  hybrid(RRF) top-1 命中率 : {hybrid_hit}/{n} = {hybrid_hit/n:.0%}")
    return dense_hit, hybrid_hit


def show_lexical_trap(embedder) -> None:
    """显式展示 lexical 陷阱：BM25 被关键词带偏，BGE-M3 仍命中语义正确的块。"""
    pipe = build_pipeline(embedder)
    q = "智能体是怎么一步步推理并选择要执行什么动作的？"  # 目标 c1
    print("\n---------- lexical 陷阱演示 ----------")
    print(f"查询: {q}  (语义目标 = c1 Agent 思考-行动-观察循环)")
    bm = bm25_top1(pipe, q)
    print(f"  BM25 top-1 (纯关键词)  -> {bm}  {'(若≠c1 即被关键词带偏)' if bm != 'c1-c0' else ''}")
    # 打印 BGE-M3 的 dense 全量排名，展示语义差距
    qv = embedder.encode([q])[0]
    ranked = pipe._vidx.search(qv, k=len(pipe._chunks))
    print("  BGE-M3 dense 余弦排名:")
    for idx, score in ranked:
        cid = pipe._chunks[idx].id
        flag = "  <-- 目标" if cid == "c1-c0" else ""
        print(f"    {cid:<10} cos={score:+.4f}{flag}")


def end_to_end(embedder) -> None:
    """真实管线端到端：长文 → 解析 → 分块 → 索引 → 混合检索 → 引用溯源。"""
    PAPER = """\
LightCore 是一种从零自研的轻量级智能体内核，它不依赖任何高层编排框架，所有循环、记忆与工具机制都自己实现，目的是在面试中讲得清每一行代码。
智能体的主循环采用 ReAct 思路：先由模型基于当前上下文进行思考，再决定调用哪一个工具，随后观察工具返回的结果，如此反复迭代，直到形成最终回答或触发中止。
为了防止任意用户代码危害宿主机系统，所有代码执行都被严格限制在 Docker 容器沙箱之中，网络与文件系统都被隔离，而管理员级工具还需要人工二次确认才能放行。
长期记忆模块使用向量数据库保存历史笔记与偏好，并在召回时综合相关性、重要程度与时间新鲜度三项指标进行加权排序，越近、越重要、越相关的笔记越容易被取回。
在检索阶段，系统把稠密向量检索与稀疏 BM25 关键词检索的结果用倒数排名融合法合并，融合常数 k 设定为六十，以平衡语义匹配与字面匹配的两种信号。
为了把成本压到最低，主力生成模型选用 DeepSeek，而文本嵌入与结果重排都在本地 CPU 完成，因此这两步几乎零成本，单次对话总花费被控制在三角钱以内并设有日预算熔断。
文档进入系统后，先用 PDF 解析库按页抽取文本，再按语义切成若干三四百 token 的小段，每一段都会计算嵌入向量并建立索引，同时记录它来自第几页以便引用溯源。
当智能体对上一步的结果不满意时，反思模块会驱动它重新拟定子计划并再尝试一次，整个过程最多允许两轮自我修正，从而避免在同一错误上无限循环。
在可观测性方面，一次会话从开始到结束都携带同一个 trace_id，模型调用与工具执行都被逐一记录，便于事后排查故障与评测检索质量、忠实度等指标。
前端通过服务器推送事件把思考、工具调用、观察结果与最终回答实时推送给用户，并且每条引用都会清晰标出它来自哪一页的哪一段原文。
在评测体系上，项目用 Recall@5 与 MRR@10 衡量检索质量，用忠实度与引用溯源率衡量回答可靠性，要求引用溯源率达到百分之百以防止幻觉。
工具系统采用权限分级，分为安全、网络、沙箱与管理员四级，并为未来接入模型上下文协议也就是 MCP 预留了扩展点。
密钥等敏感配置一律通过环境变量注入，绝不写入代码或数据库，从而满足最小暴露与可审计的安全要求。
规划子系统由 Planner 与 Reflector 两部分组成，前者把用户目标分解成不超过五步的子任务，后者在每轮结束后评判结果是否达标。
数据层基于 SQLAlchemy 与 SQLite 保存会话、消息、文档与记忆，向量索引则使用 FAISS 持久化到磁盘以便重启后直接加载。
部署采用 docker-compose 编排三个容器，分别是反向代理 nginx、后端 api 与前端 web，三者通过内部网络互通并挂载同一份数据卷。
前端用 React 18 配合 Vite 与 TypeScript 构建，提供会话列表、流式回答渲染、引用卡片与用量面板四个核心界面。
测试金字塔要求单元测试覆盖约七成逻辑，集成测试约占两成五，端到端测试只占约半成，全部跑在本地与持续集成环境中。
代码执行工具在本地开发时仅做占位与拒绝，真正运行必须进入 Docker 沙箱，从而把不可信脚本对宿主机的攻击面降到最低。
多轮对话的上下文由短期记忆窗口维护，超过窗口的旧消息会被压缩成摘要，而跨会话的用户偏好则写入长期记忆供后续召回。\
"""
    pipe = RAGPipeline(embedder=embedder)
    added = pipe.add_document(PAPER, doc_id="paper")
    print(f"\n---------- 端到端（真实管线，BGE-M3）----------\n  长文入库分块数 = {added}")
    for q in ["智能体靠什么机制决定下一步调用工具？", "怎么保证运行用户代码不会搞崩电脑？"]:
        print(f"\n  查询: {q}")
        for r in pipe.query(q, top_k=3):
            print(f"    [{r.id} | 页{r.page} | RRF分={r.score:+.4f}] {r.text[:42]}...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "fake", "real"], default="both")
    args = ap.parse_args()

    if args.mode in ("fake", "both"):
        t0 = time.time()
        eval_embedder("FakeHashEmbedder (占位/非语义)", FakeHashEmbedder())
        print(f"  [耗时 {time.time()-t0:.1f}s]")

    if args.mode in ("real", "both"):
        t0 = time.time()
        print("\n[加载 BGE-M3：首次会自动从 HF 镜像下载约 2GB 权重，请耐心等待…]")
        emb = BgeM3Embedder(device="cpu")  # 触发下载/加载
        print(f"[BGE-M3 加载完成 dim={emb.dim}，耗时 {time.time()-t0:.1f}s]")
        eval_embedder("BgeM3Embedder (真实/CPU)", emb)
        show_lexical_trap(emb)
        end_to_end(emb)

    print("\n完成。对比结论见上：BGE-M3 的 dense 命中率应显著高于 FakeHash（接近随机）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
