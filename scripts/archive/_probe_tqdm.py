"""探针：验证检索路径是否还会输出 tqdm 进度条（Batches）。

用法：.venv/Scripts/python.exe scripts/_probe_tqdm.py
"""

import asyncio
import os
import shutil
import sys

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
    p = await svc.get_pipeline(9)
    assert p is not None
    print("=== 开始检索（下方若出现 Batches 行即为进度条噪音）===", flush=True)
    res = p.query("文档中提到的对比模型都有哪些", top_k=5)
    print(f"=== 检索结束：{len(res)} hits ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
