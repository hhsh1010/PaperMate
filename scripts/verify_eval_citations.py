#!/usr/bin/env python3
"""PaperMate · /eval 溯源率一键验收脚本（端到端）

验证完整链路：
  上传文档 → 等待索引就绪 → 新建会话并关联文档（触发强制检索）
  → 发事实型问题 → 断言 GET /api/v1/eval/summary 的 citation_rate > 0

即「关联文档 → 引用注入 → 溯源率可观测」闭环。这是 dev_eval 指标卡里
「溯源率 0.0」是否为环境假象（fake/未计费）的权威判定手段。

用法（连已运行后端，默认 http://127.0.0.1:8000）：
  uv run python scripts/verify_eval_citations.py

一键隔离验收（脚本自行拉起临时数据目录的后端，不污染真实库）：
  uv run python scripts/verify_eval_citations.py --start

调试（无真实 key 也能跑通流程，仅验证脚本与链路，不校验溯源率）：
  uv run python scripts/verify_eval_citations.py --start --fake

环境前提（连真实模型验收时）：
  DEEPSEEK_API_KEY             DeepSeek 端点（默认模型）所需
  PAPERMATE_DEFAULT_EMBEDDER=real   检索需真实嵌入（BGE-M3），fake 不会命中语义
  HF_ENDPOINT=https://hf-mirror.com  权重镜像（境内拉取 BGE-M3）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "http://127.0.0.1:8000"

# 样例文档：含一句可事实检索的独特陈述（问题必须紧贴该句措辞，确保触发命中）
SAMPLE_MD = """# PaperMate 验收样例文档

PaperMate 是一个本地优先的论文研读助手。

本样例文档用于自动化验收：PaperMate 使用 BGE-M3 作为默认嵌入模型，并配合 BM25 稀疏检索形成混合检索链路。

工作间（Workspace）底层依赖一个文献集（Collection），文档的增删都作用于该集合。
"""

QUESTION = "根据文档原文，PaperMate 的默认嵌入模型是什么？请引用原文回答。"


def log(msg: str) -> None:
    print(f"[verify-eval] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[verify-eval][FAIL] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 底层 HTTP 封装（标准库，无第三方依赖）
# ---------------------------------------------------------------------------
def _http(method: str, url: str, *, json_body=None, raw_body=None, headers=None, timeout: int = 60):
    data = None
    h = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        h["Content-Type"] = "application/json"
    elif raw_body is not None:
        data = raw_body
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in h.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # 非 2xx：把响应体回带，便于定位
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {body[:500]}") from e


def api(method: str, base: str, path: str, **kw):
    return _http(method, f"{base}/api/v1{path}", **kw)


def upload_async(base: str, content: bytes, filename: str) -> int:
    """multipart 上传，复用 /documents/async（立即返回 status=uploaded）。"""
    boundary = "----pmverify" + str(random.randint(0, 1 << 30))
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + content + tail
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    out = _http("POST", f"{base}/api/v1/documents/async", raw_body=body, headers=headers)
    return int(out["id"])


# ---------------------------------------------------------------------------
# 各阶段步骤
# ---------------------------------------------------------------------------
def wait_ready(base: str, doc_id: int, timeout: int = 240) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        doc = api("GET", base, f"/documents/{doc_id}")
        st = doc.get("status")
        if st == "ready":
            return
        if st == "failed":
            raise RuntimeError(f"文档索引失败（status=failed），meta={doc.get('meta')}")
        time.sleep(3)
    raise TimeoutError(f"文档索引超时（{timeout}s 内未到 ready）")


def ask_stream(base: str, sid: int, question: str, timeout: int = 180) -> list:
    """发 SSE 聊天，消费完整流，返回 message 事件里的 citations 列表。"""
    url = f"{base}/api/v1/sessions/{sid}/messages/stream"
    req = urllib.request.Request(
        url, data=json.dumps({"content": question}).encode(), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    citations: list = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "message":
            citations.extend(ev.get("data", {}).get("citations", []) or [])
    return citations


def print_troubleshoot(summary: dict) -> None:
    am = summary.get("agent", {})
    log("排查清单：")
    log("  1) 确认已设置真实模型 key（默认 DeepSeek → DEEPSEEK_API_KEY）")
    log("  2) 确认 PAPERMATE_DEFAULT_EMBEDDER=real 且 BGE-M3 权重已就绪（HF_ENDPOINT 镜像）")
    log("  3) 确认文档 status==ready（索引成功，否则 retrieve 返空）")
    log("  4) 手动验证检索：")
    log('       curl -s -X POST http://<base>/api/v1/search -H "Content-Type: application/json" \\')
    log('         -d \'{"doc_ids":[<doc_id>],"query":"默认嵌入模型"}\'')
    log("  5) 避免用 qwen3-vl-flash 绑文档（工具服从不稳，会自动回落默认模型）")
    log(f"  当前 agent={json.dumps(am, ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# 后端自起（隔离临时数据目录）
# ---------------------------------------------------------------------------
def start_backend(args) -> tuple[subprocess.Popen, str, str]:
    tmp = tempfile.mkdtemp(prefix="pm_verify_")
    port = args.port
    env = dict(os.environ)
    env["PAPERMATE_DATA_DIR"] = tmp
    env["PAPERMATE_PORT"] = str(port)
    if args.fake:
        env["PAPERMATE_DEFAULT_MODEL"] = "fake"
        env["PAPERMATE_DEFAULT_EMBEDDER"] = "fake"
    log(f"启动隔离后端（端口 {port}，数据目录 {tmp}）…")
    proc = subprocess.Popen(
        [sys.executable, "-m", "papermate.api"],
        env=env, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise RuntimeError("后端启动即退出：\n" + out[-3000:])
        try:
            urllib.request.urlopen(f"{base}/api/v1/healthz", timeout=5)
            break
        except Exception:
            time.sleep(1.5)
    else:
        raise TimeoutError("后端 90s 内未就绪（healthz 不可达）")
    log("隔离后端已就绪。")
    return proc, base, tmp


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="PaperMate /eval 溯源率一键验收")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help=f"已运行后端地址（默认 {DEFAULT_BASE}）")
    ap.add_argument("--start", action="store_true", help="自行拉起隔离后端（临时数据目录）")
    ap.add_argument("--port", type=int, default=8011, help="--start 时后端端口（默认 8011）")
    ap.add_argument("--fake", action="store_true", help="调试：用 fake 模型/嵌入（不校验溯源率）")
    ap.add_argument("--no-cleanup", action="store_true", help="--start 时保留后端进程与临时目录")
    ap.add_argument("--doc-timeout", type=int, default=240, help="文档索引等待超时（秒）")
    args = ap.parse_args()

    base = args.base_url
    proc = None
    tmp = None
    try:
        if args.start:
            proc, base, tmp = start_backend(args)

        if not args.fake and not os.environ.get("DEEPSEEK_API_KEY"):
            err("未检测到 DEEPSEEK_API_KEY；连真实模型验收请先 export 该变量，或用 --fake 仅验证流程。")
            return 2

        log("上传样例文档…")
        doc_id = upload_async(base, SAMPLE_MD.encode("utf-8"), "verify_sample.md")
        log(f"文档 id={doc_id}，等待索引就绪…")
        wait_ready(base, doc_id, timeout=args.doc_timeout)

        log("创建会话并关联文档（触发强制检索）…")
        sess = api("POST", base, "/sessions", json_body={"title": "验收-溯源率", "config": {}})
        sid = int(sess["id"])
        api("PATCH", base, f"/sessions/{sid}", json_body={"config": {"doc_ids": [doc_id]}})

        log(f"发事实型问题：{QUESTION}")
        cites = ask_stream(base, sid, QUESTION)
        log(f"本轮回答携带引用 {len(cites)} 条")

        time.sleep(1.5)  # 等 persist_turn 落库，确保 traces 写入
        summary = api("GET", base, "/eval/summary")
        # collect_metrics 返回结构：summary["agent"]["citation_rate"]
        # （不是 agent_metrics，详见 src/papermate/skills/dev_eval.py:collect_metrics）
        cr = float(summary.get("agent", {}).get("citation_rate") or 0)
        log(f"/eval citation_rate = {cr}  (agent={json.dumps(summary.get('agent', {}), ensure_ascii=False)})")

        if cr > 0:
            log("PASS ✅ 溯源率 > 0，引用注入链路正常（先前 0% 确系环境假象，非代码缺陷）。")
            return 0
        err("溯源率 = 0，引用未注入。")
        print_troubleshoot(summary)
        return 1
    except Exception as e:  # 顶层兜底：打印可操作的错误，而非堆栈
        err(f"{type(e).__name__}: {e}")
        return 2
    finally:
        if proc is not None and not args.no_cleanup:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        if tmp is not None and not args.no_cleanup:
            shutil.rmtree(tmp, ignore_errors=True)
        if proc is not None:
            if args.no_cleanup:
                log("隔离后端保留（--no-cleanup）：进程未终止，临时数据目录未删除。")
            else:
                log("隔离后端已停止，临时数据目录已清理。")


if __name__ == "__main__":
    sys.exit(main())
