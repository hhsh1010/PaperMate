"""探针：检查各文档的 faiss 磁盘缓存签名是否仍然有效（是否每次访问都重建）。

背景：doc-3/doc-8 的 faiss 停留在 09-07，而 09-07 之后 RAG 配置（embedder/sparse/jieba）
可能已变更 → `_pipeline_signature` 不匹配 → `load_state` 返回 None → **每次访问都全量重编码**
（用户感知为「又在重建索引」）。

用法：.venv/Scripts/python.exe scripts/_probe_sig_check.py
"""

import asyncio
import os
import shutil
import sys
import time

sys.path.insert(0, "src")

PROBE = "data_e2e_probe_verify"
os.makedirs(PROBE, exist_ok=True)
shutil.copytree("data", PROBE, dirs_exist_ok=True)
os.environ["PAPERMATE_DATA_DIR"] = PROBE

from sqlalchemy.orm import sessionmaker  # noqa: E402

from papermate.config import get_settings  # noqa: E402
from papermate.models.db import make_engine  # noqa: E402
from papermate.services.document_service import DocumentService  # noqa: E402


def mtime(doc_id: int) -> str:
    p = os.path.join(PROBE, "faiss", f"doc-{doc_id}.faiss")
    if not os.path.exists(p):
        return "（无缓存文件）"
    return time.strftime("%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))


async def main() -> None:
    s = get_settings()
    eng = make_engine(s.db_url)
    svc = DocumentService(sessionmaker(bind=eng), s)
    rt = svc._rag_rt()
    print(f"当前 RAG 运行时配置: embedder={rt['rag_embedder']} bge_sparse={rt['use_bge_sparse']} jieba={rt['use_jieba']}")

    for doc_id in (3, 8, 9):
        before = mtime(doc_id)
        t0 = time.time()
        p = await svc.get_pipeline(doc_id)
        dt = time.time() - t0
        after = mtime(doc_id)
        verdict = "缓存命中（未重编码）" if before == after else "**发生了重建（faiss 被重写）**"
        print(f"doc {doc_id}: 耗时 {dt:6.1f}s  缓存前 {before} → 后 {after}  ⇒ {verdict}  pipeline={'OK' if p else 'NONE'}")


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n[提示] 探针目录 {PROBE} 保留（副本，可手动删除）")
