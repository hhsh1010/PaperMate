"""将 data/sample_papers/ 下的学科样本 PDF 批量上传到运行中的 PaperMate 后端。

用法：
  # 默认 http://127.0.0.1:8000，仅上传（不关联会话）
  python scripts/upload_sample_papers.py
  # 指定后端地址，并自动关联到某会话（便于直接测试技能）
  PAPERMATE_BASE_URL=http://127.0.0.1:8000 python scripts/upload_sample_papers.py --session-id 3

说明：
- 后端 API 无鉴权（见 src/papermate/api/app.py），故脚本直连 /documents。
- 上传即触发解析（pymupdf 分块），无需额外步骤；关联会话后可在该会话用 @ 提及或直接运行技能。
- 若需换成真实论文：把真实 PDF 放进 data/sample_papers/（或任意目录）后改 glob 路径即可。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

try:
    import httpx
except ImportError:  # pragma: no cover
    sys.exit("缺少 httpx，请先安装：pip install httpx")

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "..", "data", "sample_papers")
DEFAULT_BASE = os.environ.get("PAPERMATE_BASE_URL", "http://127.0.0.1:8000")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="PaperMate 后端地址")
    ap.add_argument("--session-id", type=int, default=None, help="可选：上传后自动关联到的会话 id")
    ap.add_argument("--glob", default=os.path.join(SAMPLE_DIR, "*.pdf"), help="要上传的 PDF 通配路径")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"未找到 PDF：{args.glob}")
        return 1

    url = f"{args.base_url.rstrip('/')}/documents"
    params = {"session_id": args.session_id} if args.session_id else {}
    ok, fail = 0, 0
    for path in files:
        name = os.path.basename(path)
        with open(path, "rb") as fh:
            data = fh.read()
        try:
            r = httpx.post(url, files={"file": (name, data, "application/pdf")}, params=params, timeout=120)
            r.raise_for_status()
            doc = r.json()
            print(f"OK  {name:28s} -> doc_id={doc.get('id')}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name:28s} -> {e}")
            fail += 1
    print(f"\n上传完成：成功 {ok}，失败 {fail}（后端 {args.base_url}）")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
