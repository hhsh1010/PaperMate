"""检索链路「来源指纹」（供评测报告与运行时一致性校验共用）

**为什么需要**：评测报告原先只绑**用例集**指纹 —— 用例集没变但检索实现/参数改了
（分块策略、`top_n_hybrid`、RRF k…），报告仍被判"新鲜"，于是门禁守的是过去。

**为什么放在内核**：`eval/run_eval.py`（写报告）与 `papermate.skills.dev_eval`（判报告是否过时）
都要算同一个指纹。清单与算法**只此一份** —— 两处各写一份必然漂移
（本项目已有"`SYSTEM_PROMPT` 同名两份内容不同"的前科）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: 参与指纹的检索链路源码（相对仓库根）。
#: ⚠️ 刻意只含**检索实现**：不含 `run_eval.py` 自身（评测脚本的口径已由报告的 `profile` 字段记录），
#: 也不含 `document_service.py`（应用层高频改动，纳入会让"改一行上传逻辑 → 报告作废"，
#: 那种脆弱指纹最后只会被当成噪音忽略）。
#:
#: **2026-09-17 补入 `parser.py` / `chunker.py`**：它们决定"**喂进检索的是什么**"，
#: 是链路里最上游、影响最大的一环 —— 实测因分块碎片化（chunk 平均 22~39 字符）导致
#: 检索片段零散，而**当时这份清单不覆盖它们**，于是"改了分块没重跑评测"根本发现不了。
#: 判据：**凡是能改变「被检索内容」的模块都必须纳入**，否则门禁守不住输入端的回归。
SOURCES: tuple[str, ...] = (
    "src/lightcore/rag/parser.py",
    "src/lightcore/rag/chunker.py",
    "src/lightcore/rag/pipeline.py",
    "src/lightcore/rag/rerank.py",
    "src/lightcore/rag/embedder.py",
    "src/lightcore/rag/bm25.py",
)

_DIGEST_LEN = 12


def repo_root() -> Path:
    """仓库根（`src/lightcore/rag/provenance.py` → parents[3]）。"""
    return Path(__file__).resolve().parents[3]


def digest(root: Path | None = None) -> str:
    """`SOURCES` 的内容指纹（sha256 前 12 位）。缺失文件按 `<missing>` 参与，保证可比对。"""
    base = root or repo_root()
    h = hashlib.sha256()
    for rel in SOURCES:
        h.update(rel.encode("utf-8"))
        p = base / rel
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:_DIGEST_LEN]
