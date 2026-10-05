"""PaperMate · 用户侧一键自检（在**你的本机**运行；沙箱无法覆盖 GPU/真实后端/联网路径）。

用法（PowerShell，项目根目录）：

    .venv\\Scripts\\python.exe scripts\\user_side_check.py            # 环境自检（torch/CUDA/设备探测）
    .venv\\Scripts\\python.exe scripts\\user_side_check.py --models    # 额外加载 BGE-M3 并实测编码（首次较慢）
    .venv\\Scripts\\python.exe scripts\\user_side_check.py --pytest    # 额外跑 GPU 专属回归 tests/test_rag_bgem3.py
    .venv\\Scripts\\python.exe scripts\\user_side_check.py --api       # 额外做 API 冒烟（需后端已启动）
    .venv\\Scripts\\python.exe scripts\\user_side_check.py --all       # 以上全部

可加 --base http://127.0.0.1:8000 指定后端地址（默认同上）。
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_RESULTS: list[tuple[str, str, str]] = []  # (状态, 名称, 说明)


def report(ok: bool | None, name: str, detail: str = "") -> None:
    """ok=True/False/None(跳过)。"""
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    _RESULTS.append((tag, name, detail))
    print(f"[{tag}] {name}" + (f" —— {detail}" if detail else ""))


def step_env() -> None:
    print("\n=== 1. Python 环境 ===")
    report(sys.version_info >= (3, 11), "Python ≥ 3.11", sys.version.split()[0])
    venv_ok = Path(sys.prefix).name.lower() in {"venv", ".venv"} or ".venv" in sys.prefix
    report(venv_ok, "运行于项目 .venv", sys.prefix)


def step_torch() -> str | None:
    print("\n=== 2. torch / CUDA ===")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        report(False, "torch 可导入", f"{type(e).__name__}: {e}")
        return None
    ver = torch.__version__
    cuda_tag = getattr(torch.version, "cuda", None)
    # PyPI Windows 默认装 CPU 版：torch.version.cuda 为 None 即 CPU-only（GPU 加速需 cu126 轮子）
    report(cuda_tag is not None, "torch 为 CUDA 构建（非 CPU-only）",
           f"torch {ver}, cuda {cuda_tag or 'None（CPU 版！需 cu126 源重装）'}")
    avail = bool(torch.cuda.is_available())
    if avail:
        props = torch.cuda.get_device_properties(0)
        report(True, "CUDA 可用", f"{props.name}, 显存 {props.total_memory / 1024**3:.1f} GB")
    else:
        report(False, "CUDA 可用", "torch.cuda.is_available()=False（驱动/版本不匹配？）")
    return "cuda" if avail else "cpu"


def step_device() -> str:
    print("\n=== 3. 设备探测（hf_local.auto_device）===")
    import os

    from lightcore.rag.hf_local import auto_device

    dev = auto_device()
    env = os.environ.get("PAPERMATE_DEVICE", "")
    expect = env or dev
    report(dev == expect, f"auto_device() → {dev}" + (f"（env PAPERMATE_DEVICE={env}）" if env else ""),
           "与预期一致" if dev == expect else f"预期 {expect}")
    return dev


def step_models(device: str) -> None:
    print("\n=== 4. BGE-M3 加载与实测编码（首次会较慢）===")
    try:
        from lightcore.rag.embedder import BgeM3Embedder

        t0 = time.perf_counter()
        emb = BgeM3Embedder(device=device)
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        vecs = emb.encode(["PaperMate 用户侧自检：这是一句用于验证嵌入的中文与 English mixed text."])
        enc_ms = (time.perf_counter() - t1) * 1000
        dim = len(vecs[0]) if hasattr(vecs[0], "__len__") else -1
        report(True, f"BGE-M3 加载 + 编码（device={device}）",
               f"加载 {load_s:.1f}s · 编码 {enc_ms:.0f}ms · dim={dim}")
    except Exception as e:  # noqa: BLE001
        report(False, "BGE-M3 加载/编码", f"{type(e).__name__}: {e}")
        print("  提示：① 确认 `uv sync --extra rag`；② 权重未下载时看 HF_ENDPOINT 镜像兜底；"
              "③ 可设 PAPERMATE_BGE_M3_PATH 指向本地快照目录")


def step_pytest() -> None:
    print("\n=== 5. GPU 专属回归（tests/test_rag_bgem3.py）===")
    py = sys.executable
    cmd = [py, "-m", "pytest", "tests/test_rag_bgem3.py", "-q", "-p", "no:cacheprovider"]
    print(f"  $ {' '.join(cmd[1:])}")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
        tail = "\n".join(proc.stdout.strip().splitlines()[-6:])
        print(tail)
        report(proc.returncode == 0, "test_rag_bgem3 全量通过")
    except subprocess.TimeoutExpired:
        report(False, "test_rag_bgem3 超时（>30min）")
    except Exception as e:  # noqa: BLE001
        report(False, "test_rag_bgem3 运行失败", f"{type(e).__name__}: {e}")


def _get(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _send_multipart(url: str, filename: str, content: bytes) -> tuple[int, dict]:
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.write(b"Content-Type: text/plain\r\n\r\n")
    body.write(content)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(url, data=body.getvalue(), method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _delete(url: str) -> None:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=30):
        pass


def step_api(base: str) -> None:
    print(f"\n=== 6. API 冒烟（{base}，需后端已启动）===")
    doc_id: int | None = None
    try:
        _, sessions = _get(f"{base}/api/v1/sessions")
        report(True, "GET /sessions", f"共 {len(sessions)} 个会话")
    except urllib.error.URLError as e:
        report(False, "GET /sessions", f"后端不可达：{e.reason}（请先在终端启动后端）")
        return
    except Exception as e:  # noqa: BLE001
        report(False, "GET /sessions", f"{type(e).__name__}: {e}")
        return

    sample = (
        "DeepSeek-V4 Technical Report\n\n"
        "Abstract. This is a smoke-test document for PaperMate user-side verification. "
        "It mentions retrieval-augmented generation, sparse embedding and jieba tokenization. "
        "Conclusion: the pipeline works end to end.\n" * 3
    ).encode("utf-8")
    try:
        _, doc = _send_multipart(f"{base}/api/v1/documents", f"smoke-{uuid.uuid4().hex[:6]}.txt", sample)
        doc_id = doc.get("id")
        report(doc_id is not None, "POST /documents（上传）", f"doc_id={doc_id}")
    except Exception as e:  # noqa: BLE001
        report(False, "POST /documents（上传）", f"{type(e).__name__}: {e}")
        return

    try:
        deadline = time.time() + 180
        status = "?"
        while time.time() < deadline:
            _, d = _get(f"{base}/api/v1/documents/{doc_id}")
            status = d.get("status", "?")
            if status in {"ready", "failed", "cancelled"}:
                break
            time.sleep(2)
        report(status == "ready", f"上传流水线终态 = {status}",
               "ready 表示 parse→chunk→embed（BGE-M3 GPU）全链路通过" if status == "ready" else "查看后端日志排查")
    except Exception as e:  # noqa: BLE001
        report(False, "上传流水线轮询", f"{type(e).__name__}: {e}")
    finally:
        if doc_id is not None:
            try:
                _delete(f"{base}/api/v1/documents/{doc_id}")
                print(f"[CLEAN] 测试文档 {doc_id} 已删除")
            except Exception:  # noqa: BLE001
                print(f"[WARN] 测试文档 {doc_id} 删除失败，请手动清理")


def main() -> int:
    ap = argparse.ArgumentParser(description="PaperMate 用户侧一键自检")
    ap.add_argument("--models", action="store_true", help="加载 BGE-M3 并实测编码")
    ap.add_argument("--pytest", action="store_true", help="跑 tests/test_rag_bgem3.py")
    ap.add_argument("--api", action="store_true", help="API 冒烟（需后端已启动）")
    ap.add_argument("--all", action="store_true", help="以上全部")
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="后端地址（默认 http://127.0.0.1:8000）")
    args = ap.parse_args()

    step_env()
    device = step_torch()
    step_device()
    if args.models or args.all:
        step_models(device)
    if args.pytest or args.all:
        step_pytest()
    if args.api or args.all:
        step_api(args.base.rstrip("/"))

    fails = [r for r in _RESULTS if r[0] == "FAIL"]
    print("\n" + "=" * 60)
    print(f"结果：PASS {sum(1 for r in _RESULTS if r[0] == 'PASS')} / "
          f"FAIL {len(fails)} / SKIP {sum(1 for r in _RESULTS if r[0] == 'SKIP')}")
    if fails:
        print("失败项：")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
    print("完整手动验证步骤见 docs/用户侧验证清单.md")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
