"""N3-v2 真实 e2e 验收：主文献存在时，改稿工作间对话必须**只**基于主文献作答。

复现用户场景：session 15 等价配置（doc_ids=[11,9,8,3] + draft_doc_id=9 + mode="research"），
问题「请概括一下文档中的创新点」——修复前会答 SCS-YOLO / SS-YOLO / LRM-YOLO 三篇，
修复后应**只**答 LRM-YOLO（doc 9）。

用法：.venv/Scripts/python.exe scripts/_probe_e2e_n3v2.py
"""

import json
import os
import shutil
import sys
import time

sys.path.insert(0, "src")

PROBE = "data_e2e_probe_verify"  # 固定名：重复运行复用，避免产生新目录
os.makedirs(PROBE, exist_ok=True)
shutil.copytree("data", PROBE, dirs_exist_ok=True)
os.environ["PAPERMATE_DATA_DIR"] = PROBE

from fastapi.testclient import TestClient  # noqa: E402

import papermate.config as cfg_mod  # noqa: E402

cfg_mod._settings = None

from papermate.api.app import create_app  # noqa: E402

#: 主文献独有的标识 / 其他篇独有的标识（依据 chunks 实测）
ONLY_DRAFT = ("LRM-YOLO", "LRMN", "LED-Head", "安全帽")
ONLY_OTHERS = ("SCS-YOLO", "SS-YOLO", "PP-LCNet", "LCNWD")


def _ask(client, sid: int, question: str) -> str:
    texts: list[str] = []
    errors: list[str] = []
    ev = None
    t0 = time.time()
    with client.stream(
        "POST", f"/api/v1/sessions/{sid}/messages/stream", json={"content": question}
    ) as r:
        for line in r.iter_lines():
            line = line.strip()
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:") and ev:
                try:
                    d = json.loads(line[5:].strip())
                except Exception:  # noqa: BLE001
                    continue
                if ev == "message" and d.get("text"):
                    texts.append(d["text"])
                elif ev == "error":
                    errors.append(str(d.get("reason")))
    print(f"  [耗时 {time.time() - t0:.1f}s] error={errors} final_len={len(texts[-1]) if texts else 0}")
    return texts[-1] if texts else ""


def main() -> None:
    app = create_app()
    with TestClient(app) as c:
        sid = c.post("/api/v1/sessions", json={"title": "n3v2-probe"}).json()["id"]
        # 等价 session 15：工作间多篇关联 + 主文献 + mode=research（关键：非 revision）
        c.patch(
            f"/api/v1/sessions/{sid}",
            json={"config": {"doc_ids": [11, 9, 8, 3], "draft_doc_id": 9, "mode": "research"}},
        )
        print(f"会话 {sid} 配置: doc_ids=[11,9,8,3] draft_doc_id=9 mode=research")

        print("\n[用例1] 问「文档中的创新点」（有主文献 → 应只答主文献）")
        ans1 = _ask(c, sid, "请概括一下文档中的创新点")
        hit_others = [k for k in ONLY_OTHERS if k in ans1]
        hit_draft = [k for k in ONLY_DRAFT if k in ans1]
        print(f"  命中主文献标识: {hit_draft}")
        print(f"  命中其他篇标识: {hit_others}")
        print(f"  结论: {'✅ 只答主文献（修复生效）' if not hit_others and hit_draft else '❌ 仍混入其他篇' if hit_others else '⚠️ 未命中主文献标识（需人工看内容）'}")
        print(f"  摘要: {ans1[:160]!r}")

        c.delete(f"/api/v1/sessions/{sid}")
    print(f"\n[提示] 探针数据目录 {PROBE} 保留（可由用户手动删除）")


if __name__ == "__main__":
    main()
