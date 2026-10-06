"""T-06 P0：改稿后检索一致性验证（2026-09-30，探针结果 §3 的疑点求证）。

**疑点**：`apply-edits` 只改 docx 源文件；随后触发的 `reindex` 若从**未更新的 chunks 表**
算 signature ⇒ 原样命中磁盘 state 秒回 ⇒ **检索仍是改稿前内容，且无报错**。

**方法**（隔离临时 DATA_DIR，真实 BGE-M3）：
1. python-docx 生成 3 段可控文档（P1 含特征句 A）→ 上传至 ready
2. apply-edits 把 P1 **整段替换**为特征句 B → POST reindex（等效 `_trigger_reindex`）
3. 三重判定：
   ① **chunks 表**（精确）：临时库里 chunk 文本是否含句 B / 仍含句 A
   ② **检索**（语义）：query=句B关键词 → top 结果是否含句 B
   ③ **signature**：reindex 是否秒回（秒回 = state 原样复用 = 未真正重建）

**判定**：①含 B 且不含 A ⇒ 一致 ✓；否则 ⇒ **缺陷证实**。
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

API = "/api/v1"  # 漏掉会 405（探针 v1 踩过）

SENT_A = "The quick brown fox jumps over the lazy dog near the river bank."
SENT_B = "Deep reinforcement learning controls autonomous drone swarms in crowded cities."


def _prepare_env() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="t06c_"))
    os.environ["PAPERMATE_DATA_DIR"] = str(tmp)
    os.environ["PAPERMATE_PREWARM"] = "0"
    # 真实 embedder（2026-10-05：config 默认值已改为 fake ⇒ 必须**显式写 real**，
    # 原 `pop()` 写法依赖"默认值就是 real"，改默认后会静默退化成 fake、测不出真实语义）
    os.environ["PAPERMATE_DEFAULT_EMBEDDER"] = "real"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return tmp


def _make_docx(path: Path) -> None:
    """生成 3 段可控 docx：P0 引言 / P1 特征句A / P2 结论。"""
    from docx import Document

    doc = Document()
    doc.add_paragraph("This is the introduction paragraph of the consistency probe document.")
    doc.add_paragraph(SENT_A)
    doc.add_paragraph("Final remarks and summary of the probe document.")
    doc.save(path)


def _chunks_texts(data_dir: Path) -> list[str]:
    """读临时库 chunks 表的全部文本（精确判定用）。"""
    import sqlite3

    db = data_dir / "papermate.db"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("select content from chunks").fetchall()
    finally:
        con.close()
    return [r[0] or "" for r in rows]


def main() -> int:
    data_dir = _prepare_env()
    from fastapi.testclient import TestClient

    from papermate.api.app import create_app

    docx_path = data_dir / "probe.docx"
    _make_docx(docx_path)
    print(f"[c] 隔离 data_dir = {data_dir}")
    print(f"[c] 特征句A(旧) = {SENT_A}")
    print(f"[c] 特征句B(新) = {SENT_B}")

    app = create_app()
    verdicts: list[tuple[str, bool, str]] = []
    with TestClient(app) as client:
        # 1) 上传至 ready（同步接口，响应即完成）
        with docx_path.open("rb") as f:
            t0 = time.perf_counter()
            r = client.post(
                API + "/documents",
                files={"file": ("probe.docx", f)},
                params={"allow_duplicate": "true"},
            )
        build_s = time.perf_counter() - t0
        r.raise_for_status()
        doc = r.json()
        doc_id = doc["id"]
        print(f"[c] 首建 = {build_s:.1f}s status={doc.get('status')} id={doc_id}")

        # 2) 基线检索：句 A 应可命中
        q = client.post(
            API + "/search",
            json={"query": "quick brown fox jumps lazy dog", "top_k": 3, "doc_ids": [doc_id]},
        ).json()
        hit_a_before = any("quick brown fox" in (c.get("text") or "") for c in q)
        verdicts.append(("基线：改稿前句A可检索", hit_a_before, f"top={len(q)}"))

        # 3) 改稿：整段替换 P1 为句 B
        r = client.post(
            f"{API}/documents/{doc_id}/apply-edits",
            json={"edits": [{"pid": "P1", "op": "replace", "text": SENT_B}]},
        )
        r.raise_for_status()
        applied = r.json()
        print(f"[c] apply-edits applied={applied.get('applied')} rejected={applied.get('rejected')} "
              f"reindex_triggered={applied.get('reindex_triggered')}")

        # 4) reindex（等效后台 _trigger_reindex 的任务体）
        t0 = time.perf_counter()
        client.post(f"{API}/documents/{doc_id}/reindex")
        reindex_s = time.perf_counter() - t0
        print(f"[c] reindex = {reindex_s:.2f}s  （秒回 = state 原样复用的信号）")

        # 5) 三重判定
        texts = _chunks_texts(data_dir)
        has_b = any(SENT_B[:40] in t for t in texts)
        has_a = any("quick brown fox" in t for t in texts)
        verdicts.append(("① chunks 表含新句B", has_b, f"chunks={len(texts)} 条"))
        verdicts.append(("① chunks 表仍含旧句A", has_a, "含旧句 = 未更新"))
        q = client.post(
            API + "/search",
            json={"query": "reinforcement learning drone swarms cities", "top_k": 3,
                  "doc_ids": [doc_id]},
        ).json()
        hit_b = any("reinforcement learning" in (c.get("text") or "") for c in q)
        verdicts.append(("② 检索可命中新句B", hit_b, f"top={len(q)}"))

    print("\n[c] ===== 判定 =====")
    ok = True
    for name, passed, note in verdicts:
        print(f"  {'✅' if passed else '❌'} {name}   ({note})")
    consistent = (
        dict((n, p) for n, p, _ in verdicts)["① chunks 表含新句B"]
        and not dict((n, p) for n, p, _ in verdicts)["① chunks 表仍含旧句A"]
    )
    print("\n[c] ===== 结论 =====")
    if consistent:
        print("  ✅ 一致：改稿后 chunks 已更新，reindex 后检索反映新内容。疑点排除。")
    else:
        print("  ❌ 缺陷证实：改稿后 chunks/reindex 未反映新内容 ⇒ 检索到的是改稿前文本"
              "（`_trigger_reindex` 为假生效）。需修复：apply-edits 后失效索引 state 或同步更新 chunks。")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
