"""T-06 探针（P1）：重建耗时构成实测（2026-09-30，方案见 docs/T06-增量重建-探针方案.md）。

**隔离**：独立临时 `PAPERMATE_DATA_DIR`，**绝不碰真实 `data/`**；真实库只读连接取样本路径。
**无 VLM 臂**：`figure_enrich` 默认关闭 ⇒ 本轮零 API 费用；VLM 占比用
doc10 的历史实测 742s（含 VLM）与本轮无 VLM 耗时之差**反推下界**。

测三个数字（HTTP 走真实路径，TestClient）：
  ① 样本 A（纯文本，eval/corpus/agent-loop.pdf）首建总耗时
  ② 样本 B（图密集，真实 doc10 源文件）首建总耗时
  ③ reindex 单次耗时（弹缓存重建管线；signature 命中应近 0）

用法：`.venv/Scripts/python.exe scripts/probe_t06_rebuild.py`
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: 应用统一挂载前缀（漏掉会 405 —— 2026-09-30 实测踩到）
API = "/api/v1"


def _prepare_env() -> Path:
    """在 import papermate **之前**设置进程级隔离环境。"""
    tmp = Path(tempfile.mkdtemp(prefix="t06_probe_"))
    os.environ["PAPERMATE_DATA_DIR"] = str(tmp)
    os.environ["PAPERMATE_PREWARM"] = "0"
    # 真实 embedder（不设 fake —— fake 测不出真实耗时）
    # 2026-10-05：config 默认值已改为 fake ⇒ 必须**显式写 real**（同上，原 pop 写法已失效）
    os.environ["PAPERMATE_DEFAULT_EMBEDDER"] = "real"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return tmp


def _find_doc10_source() -> Path | None:
    """只读连接真实库，取 doc10（图密集样本）的源文件路径。"""
    db = ROOT / "data" / "papermate.db"
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute(
            "select storage_path from documents where id = 10"
        ).fetchone()
    finally:
        con.close()
    if not row:
        return None
    p = Path(row[0])
    return p if p.exists() else None


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def main() -> int:
    data_dir = _prepare_env()
    # ---- env 就绪后才允许 import papermate ----
    from fastapi.testclient import TestClient

    from papermate.api.app import create_app

    print(f"[probe] 隔离 data_dir = {data_dir}")
    sample_a = ROOT / "eval" / "corpus" / "agent-loop.pdf"
    sample_b = _find_doc10_source()
    print(f"[probe] 样本A(纯文本) = {sample_a} exists={sample_a.exists()}")
    print(f"[probe] 样本B(图密集 doc10) = {sample_b}")

    app = create_app()
    results: dict[str, object] = {"data_dir": str(data_dir)}
    with TestClient(app) as client:
        # 预热：首次真实调用会加载 BGE-M3（~22s），单列不计入样本耗时
        t0 = time.perf_counter()
        client.get(API + "/documents")
        results["embedder_load_s"] = round(time.perf_counter() - t0, 1)
        print(f"[probe] embedder 加载/首次调用 = {results['embedder_load_s']}s")

        # ---- ① 样本 A 首建（同步上传，响应即完成）----
        if sample_a.exists():
            with sample_a.open("rb") as f:
                dt, r = _timed(
                    client.post, API + "/documents",
                    files={"file": ("probe-a.pdf", f)},
                    params={"allow_duplicate": "true"},
                )
            doc_a = r.json()
            results["A_first_build_s"] = round(dt, 1)
            results["A_status"] = doc_a.get("status")
            print(f"[probe] A 首建 = {dt:.1f}s status={doc_a.get('status')} id={doc_a.get('id')}")

            # ---- ③ reindex（弹缓存重建管线；signature 命中应近 0）----
            if doc_a.get("id"):
                dt2, r2 = _timed(client.post, f"{API}/documents/{doc_a['id']}/reindex")
                results["A_reindex_s"] = round(dt2, 1)
                print(f"[probe] A reindex = {dt2:.1f}s status={r2.json().get('status')}")

        # ---- ② 样本 B 首建（图密集；无 VLM 臂）----
        if sample_b:
            with sample_b.open("rb") as f:
                dt3, r3 = _timed(
                    client.post, API + "/documents",
                    files={"file": ("probe-b.pdf", f)},
                    params={"allow_duplicate": "true"},
                )
            doc_b = r3.json()
            results["B_first_build_s"] = round(dt3, 1)
            results["B_status"] = doc_b.get("status")
            print(f"[probe] B 首建(无VLM) = {dt3:.1f}s status={doc_b.get('status')} id={doc_b.get('id')}")
            if doc_b.get("id"):
                dt4, r4 = _timed(client.post, f"{API}/documents/{doc_b['id']}/reindex")
                results["B_reindex_s"] = round(dt4, 1)
                print(f"[probe] B reindex = {dt4:.1f}s status={r4.json().get('status')}")

    # ---- 结论提示 ----
    print("\n[probe] ===== 结果汇总 =====")
    for k, v in results.items():
        print(f"  {k} = {v}")
    print(
        "\n[判读] 历史实测：doc10 含 VLM 整篇重建 742s。"
        "本轮 B_first_build_s 为【无 VLM 臂】，"
        "VLM 占比下界 ≈ (742 - B_first_build_s) / 742。"
    )
    out_log = ROOT / "eval" / "_t06_probe_result.log"
    out_log.write_text(
        "\n".join(f"{k} = {v}" for k, v in results.items()), encoding="utf-8"
    )
    print(f"[probe] 结果已写 {out_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
