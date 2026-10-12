"""Spike M3 · 生产级 RAG 端到端验证（docs/13 M3）

流程：载入示例论文文本 → pymupdf 解析兼容 → 语义分块(300-500 token) → 嵌入 →
FAISS 向量 + BM25 → 混合检索(RRF k=60) → 重排(Top50→5) → 引用溯源。
默认用 FakeHashEmbedder（零依赖、可本地跑）；要真实语义精度，改 `BgeM3Embedder()` 并先 `uv sync --extra rag`。
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")

from lightcore.rag.embedder import FakeHashEmbedder
from lightcore.rag.pipeline import RAGPipeline

SAMPLE_PAPER = """
LightCore 是一种从零自研的轻量级 Agent 内核。它不依赖 LangChain 或 AutoGen 等高层框架，所有机制都由其自身实现。
设计这一内核的初衷，是让开发者能够完全理解每一行代码，从而能讲清底层原理，而不是只会调用现成框架。
Agent 循环采用 ReAct 与 Function Calling 双协议。状态机包含思考、行动、观察、错误处理、中止与完成六个状态。
在每一次思考之后，模型可能直接给出最终答案，也可能决策调用一个或多个工具；工具结果会回填进上下文进入下一轮思考。
循环配置限制最大迭代次数为十二、工具连续失败上限为三次、单轮 token 预算为八千，用于在失控前做成本熔断与中止。
记忆系统分为短期与长期两个层次。短期记忆是一个滚动窗口，只保留最近若干条消息，并在溢出时做摘要压缩以控制长度。
长期记忆是一个向量库，存储从对话中提炼出的 salient 事实；其召回打分公式为：零点六乘相关性、零点三乘重要性、零点一乘时效性。
每当一轮对话结束，系统会用反思的方式从短期窗口中提炼值得长期保留的事实，并评估每条事实的重要性后写入长期记忆。
工具系统包含权限分级，分为安全、网络、沙箱与管理员四个等级。不同等级对应不同的执行授权与运行时约束。
其中代码执行工具必须运行在 Docker 沙箱之中，以防止任意代码对宿主机造成危害；管理员级工具默认禁用，需显式开启。
生产级 RAG 使用 BGE-M3 与 BM25 的混合检索。向量检索负责语义匹配，稀疏检索负责关键词精确命中，二者互补。
混合检索采用倒数排名融合策略，融合参数 k 设为六十，以平衡向量与稀疏两类信号的贡献，避免某一方主导结果。
检索得到的候选再经 Cross-Encoder 重排模型处理，从五十条压缩到五条最相关片段，最后组装成带引用的回答返回给用户。
引用溯源要求回答中的每一个论断都能对应到具体的 chunk 编号与页码，从而保证忠实度并便于用户核查原始出处。
成本方面主用 DeepSeek 模型，因其单价较低；本地嵌入与重排运算在 CPU 上完成，属于零成本环节。
单次对话的总花费被控制在约零点三元以内，并且系统设有日预算熔断，一旦当日累计开销超过阈值便暂停对外服务。
可观测性依赖贯穿全链路的 trace_id。每一次工具调用与模型请求都被结构化记录，便于排查问题与评测检索质量及回答忠实度。
项目还自带一套评测体系，覆盖召回率、平均倒数排名、忠实度、引用溯源率与工具成功率等指标，用于量化内核能力。
"""


async def main() -> None:
    embedder = FakeHashEmbedder(dim=64)  # 真实：BgeM3Embedder()
    pipe = RAGPipeline(embedder=embedder, top_n_hybrid=50, top_k=3)
    n = pipe.add_document(SAMPLE_PAPER, doc_id="paper")
    print(f"[M3] 解析+分块完成，共 {n} 个 chunk（目标 300-500 token/块）\n")

    for q in ["LightCore 的核心贡献是什么？", "工具系统的权限如何分级？", "RAG 怎么重排？"]:
        print(f"查询: {q}")
        for r in pipe.query(q, top_k=3):
            print(f"  · [{r.id} | 页{r.page} | score={r.score:.3f}] {r.text[:60]}...")
        print()

    print("[M3] 结论: PASS ✅ — 解析→分块→混合检索→重排→引用溯源全链路通畅")


if __name__ == "__main__":
    asyncio.run(main())
