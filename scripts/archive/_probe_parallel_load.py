"""探针：多文档管线加载 —— 串行 vs 并发（验证 build_agent 的并发加载收益）。

用法：.venv/Scripts/python.exe scripts/_probe_parallel_load.py
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


async def main() -> None:
    s = get_settings()
    svc = DocumentService(sessionmaker(bind=make_engine(s.db_url)), s)
    docs = [11, 9, 8, 3]

    # 1) 串行（改造前的行为）
    t0 = time.time()
    for d in docs:
        await svc.get_pipeline(d)
    serial = time.time() - t0
    print(f"[串行] 4 篇首次加载合计 {serial:6.1f}s  缓存={sorted(svc._pipelines.keys())}")

    # 2) 清缓存（模拟新进程）/ 保留 embedder 单例（真实重启时单例也需重载，故此处只测加载部分）
    svc._pipelines.clear()
    t0 = time.time()
    res = await asyncio.gather(*(svc.get_pipeline(d) for d in docs), return_exceptions=True)
    parallel = time.time() - t0
    ok = sum(1 for r in res if r is not None and not isinstance(r, BaseException))
    print(f"[并发] 4 篇并发加载合计 {parallel:6.1f}s  成功 {ok}/4")
    print(f"\n收益：{serial - parallel:+.1f}s（{(serial - parallel) / serial:.0%} 减少）")


if __name__ == "__main__":
    asyncio.run(main())
