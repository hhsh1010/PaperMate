"""PaperMate · 全量重建索引工具（2026-09-01）

切换 embedding 模式 / BGE 稀疏 / 中文分词后，必须用当前配置重建所有文档的检索管线缓存
（get_pipeline 按 doc_id 缓存，不感知配置变化，不重建会沿用旧配置）。

两种用法：
- 设置页（UI）：保存检索底座设置后点「重建索引」按钮（POST /documents/reindex）。
- 本脚本（终端 / 激活清单 env 路径）：切换 PAPERMATE_DEFAULT_EMBEDDER 等环境变量后，
  启后端前或后端已启时都可运行，按当前生效配置重建：

    # 默认读 runtime_settings（被 env 种子初值）：
    python scripts/reindex_documents.py
    # 临时指定 embedding 模式（覆盖本次运行）：
    python scripts/reindex_documents.py --embedder real
    python scripts/reindex_documents.py --embedder fake --no-bge-sparse --no-jieba
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 确保 src 在 sys.path（脚本位于项目根 scripts/，运行期 src 可能因调用方式未自动加入）
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperMate 全量重建检索索引")
    parser.add_argument("--embedder", choices=["fake", "real"], default=None,
                        help="临时覆盖 embedding 模式（fake|real）；缺省则读当前生效配置")
    parser.add_argument("--bge-sparse", dest="bge_sparse", action="store_true", default=None,
                        help="开启 BGE-M3 稀疏向量（与 BM25 并行）")
    parser.add_argument("--no-bge-sparse", dest="bge_sparse", action="store_false",
                        help="关闭 BGE-M3 稀疏向量")
    parser.add_argument("--jieba", dest="jieba", action="store_true", default=None,
                        help="开启 jieba 中文分词")
    parser.add_argument("--no-jieba", dest="jieba", action="store_false",
                        help="关闭 jieba 中文分词（回退字符级）")
    args = parser.parse_args()

    from sqlalchemy.orm import sessionmaker

    from papermate.config import get_settings
    from papermate.models.db import init_db, make_engine
    from papermate.runtime_settings import get_state
    from papermate.services.document_service import DocumentService

    settings = get_settings()
    engine = make_engine(settings.db_url)
    init_db(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    doc_svc = DocumentService(SessionLocal, settings)

    # 允许命令行临时覆盖（仅本次进程生效，不落盘；落盘请用设置页或环境变量）
    if any(v is not None for v in (args.embedder, args.bge_sparse, args.jieba)):
        st = get_state()
        if args.embedder is not None:
            st["rag_embedder"] = args.embedder
        if args.bge_sparse is not None:
            st["use_bge_sparse"] = args.bge_sparse
        if args.jieba is not None:
            st["use_jieba"] = args.jieba
        print(f"[配置] 临时覆盖 → rag_embedder={st['rag_embedder']} "
              f"use_bge_sparse={st['use_bge_sparse']} use_jieba={st['use_jieba']}")

    eff = doc_svc._rag_rt()
    print(f"[生效配置] rag_embedder={eff['rag_embedder']} "
          f"use_bge_sparse={eff['use_bge_sparse']} use_jieba={eff['use_jieba']}")

    n = asyncio.run(doc_svc.reindex_all())
    print(f"[完成] 已重建索引文档数：{n}")


if __name__ == "__main__":
    main()
