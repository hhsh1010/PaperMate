"""探针：改稿工作间连续两轮对话，检查管线缓存是否复用（是否"每轮重建索引"）。

用法：.venv/Scripts/python.exe scripts/_probe_cache_reuse.py
"""

import os
import shutil
import sys
import time

sys.path.insert(0, "src")

PROBE = "data_e2e_probe_verify"  # 复用同一副本目录，避免新增垃圾
os.makedirs(PROBE, exist_ok=True)
shutil.copytree("data", PROBE, dirs_exist_ok=True)
os.environ["PAPERMATE_DATA_DIR"] = PROBE

from fastapi.testclient import TestClient  # noqa: E402

import papermate.config as cfg_mod  # noqa: E402

cfg_mod._settings = None

from papermate.api.app import create_app  # noqa: E402


def _ask(client, sid: int, question: str) -> tuple[float, list[int]]:
    ev = None
    t0 = time.time()
    for _ in ():
        pass
    with client.stream(
        "POST", f"/api/v1/sessions/{sid}/messages/stream", json={"content": question}
    ) as r:
        for line in r.iter_lines():
            line = line.strip()
            if line.startswith("event:"):
                ev = line[6:].strip()
    dt = time.time() - t0
    cached = sorted(client.app.state.chat_svc._docs._pipelines.keys())
    return dt, cached


def main() -> None:
    app = create_app()
    with TestClient(app) as c:
        sid = c.post("/api/v1/sessions", json={"title": "cache-probe"}).json()["id"]
        c.patch(
            f"/api/v1/sessions/{sid}",
            json={"config": {"doc_ids": [11, 9, 8, 3], "draft_doc_id": 9, "mode": "research"}},
        )
        docs_svc = c.app.state.chat_svc._docs
        print(f"会话 {sid}: doc_ids=[11,9,8,3] draft=9")
        print(f"初始管线缓存: {sorted(docs_svc._pipelines.keys())}")

        for i in (1, 2, 3):
            dt, cached = _ask(c, sid, "请概括一下文档中的创新点" if i == 1 else "再说一次要点")
            print(f"[第 {i} 轮] 耗时 {dt:.1f}s  管线缓存={cached}  磁盘缓存文件 mtime 检查见下")
            m = os.path.getmtime(os.path.join(PROBE, "faiss", "doc-9.faiss"))
            print(f"          doc-9.faiss mtime={time.strftime('%H:%M:%S', time.localtime(m))}")

        c.delete(f"/api/v1/sessions/{sid}")
        print("\n判读：第 2/3 轮耗时显著低于第 1 轮 → 缓存复用正常（无重复重建）；")
        print("      若每轮耗时接近且 faiss mtime 持续刷新 → 确实在重复重建。")


if __name__ == "__main__":
    main()
