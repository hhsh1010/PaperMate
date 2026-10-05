"""BGE-M3 真实嵌入冒烟测试（仅在权重已下载时运行，避免 CI 拉取 2GB）。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _bge_m3_cached() -> bool:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = hf_home / "hub"
    if not hub.exists():
        return False
    return any("bge-m3" in p.name for p in hub.iterdir() if p.is_dir())


sentence_transformers = pytest.importorskip("sentence_transformers")
from lightcore.rag.embedder import BgeM3Embedder  # noqa: E402
from lightcore.rag.pipeline import RAGPipeline  # noqa: E402


@pytest.mark.skipif(not _bge_m3_cached(), reason="BGE-M3 权重未下载，跳过（手动跑 scripts/spike_m3_bgem3.py）")
def test_bgem3_semantic_top1():
    emb = BgeM3Embedder(device="cpu")
    pipe = RAGPipeline(embedder=emb)
    corpus = [
        ("c1", "Agent 循环采用 ReAct 范式：模型先思考，再决定调用工具，然后观察结果，循环往复直到给出答案。"),
        ("c2", "工具的权限分为安全、网络、沙箱与管理员四级，代码执行必须放在 Docker 沙箱里运行。"),
        ("c3", "长期记忆用向量存储，按相关性、重要性与时效性三者加权打分来召回最相关的历史。"),
    ]
    for cid, text in corpus:
        assert pipe.add_document(text, doc_id=cid) == 1
    qv = emb.encode(["智能体是怎么一步步推理并选择要执行什么动作的？"])[0]
    top = pipe._vidx.search(qv, k=len(pipe._chunks))[0][0]
    assert pipe._chunks[top].id == "c1-c0"
