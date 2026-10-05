"""C1（2026-09-21）：响应契约覆盖率的**防回退**守护。

C1 是"52 个端点补 `response_model`"这类**长跑型**任务：最容易的失败方式不是做错，
而是**做了一半、然后在后续开发里被慢慢抵消**（新增端点又不写契约）。
所以这里锁住**进度下限**，而不是逐个断言某个端点长什么样。

三个不变量：
1. **覆盖率不得低于基线** —— 每批完成后把 `BASELINE_COVERED` 提到新值，回退即红。
2. **已接入的端点不得被摘掉**（明确列名单，删了会红）。
3. **流式/文件响应端点的契约口径与普通端点相反** —— 它们**不应**有 `response_model`
   （加在 `StreamingResponse`/`Response` 上会让 FastAPI 尝试校验流，属于改坏而非改好）。
"""

from __future__ import annotations

import re
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent / "src" / "papermate" / "api"

#: C1 进度基线（2026-09-21 第六批（收口）31 → 74；2026-09-23 doc-scope → **75**）。**每完成一批就上调**。
BASELINE_COVERED = 75

#: 明确已接入契约的端点（防被摘掉）。格式：(HTTP 方法, 路由路径片段)
MUST_HAVE_CONTRACT = [
    ("get", "/collections"),
    ("get", "/collections/{cid}"),
    ("post", "/collections"),
    ("patch", "/collections/{cid}"),
    ("delete", "/collections/{cid}"),
    ("post", "/collections/{cid}/documents"),
    ("delete", "/collections/{cid}/documents/{doc_id}"),
    ("get", "/memories"),
    ("post", "/memories/batch-delete"),
    ("put", "/memories/{mem_id}"),
    ("delete", "/memories/{mem_id}"),
    ("post", "/memories/clear"),
    # 第二批：documents（9 个）
    ("delete", "/documents/{doc_id}"),
    ("post", "/documents/precheck"),
    ("get", "/documents/{doc_id}/progress"),
    ("post", "/documents/{doc_id}/cancel"),
    ("get", "/documents/{doc_id}/text"),
    ("get", "/documents/{doc_id}/images"),
    ("get", "/documents/{doc_id}/tables"),
    ("get", "/documents/{doc_id}/diagrams"),
    ("post", "/documents/reindex/cancel"),
    # 第三批：traces（2）+ settings（4）
    ("get", "/traces"),
    ("get", "/traces/{trace_id}"),
    ("get", "/settings"),
    ("post", "/settings/test"),
    ("post", "/settings/test-web-search"),
    ("get", "/settings/inventory"),
    # 第四批：sessions / eval / usage
    ("get", "/eval/summary"),
    ("get", "/usage/sessions/{session_id}"),
    ("get", "/sessions/{session_id}/network-capability"),
    # 场景可见性（2026-09-23）：会话检索范围
    ("get", "/sessions/{session_id}/doc-scope"),
    ("delete", "/sessions/{session_id}"),
    ("get", "/settings/balances"),
    # 第五批：summary / prompt / transform / settings PUT
    ("post", "/prompt/enhance"),
    ("post", "/workbench/para-transform"),
    ("post", "/summarize"),
    ("post", "/related-work"),
    ("put", "/settings"),
    # 第五批：summary / prompt / transform / settings PUT
    ("post", "/prompt/enhance"),
    ("post", "/workbench/para-transform"),
    ("post", "/summarize"),
    ("post", "/related-work"),
    ("put", "/settings"),
    # 第六批（收口）：documents 剩余 + compare
    ("get", "/documents/{doc_id}/paragraphs"),
    ("get", "/documents/{doc_id}/stream"),
    ("post", "/documents/{doc_id}/apply-edits"),
    ("post", "/documents/{doc_id}/restore-backup"),
    ("post", "/documents/reindex"),
    ("post", "/compare"),
    # 第六批（收口）：documents 剩余 + compare
    ("get", "/documents/{doc_id}/paragraphs"),
    ("get", "/documents/{doc_id}/stream"),
    ("post", "/documents/{doc_id}/apply-edits"),
    ("post", "/documents/{doc_id}/restore-backup"),
    ("post", "/documents/reindex"),
    ("post", "/compare"),
]


def test_literal_paths_registered_before_parametrized():
    """★ 静态路径必须注册在参数化路径**之前**，否则参数化那条会把它吃掉。

    实测案例（C1 第二批发现）：`POST /documents/reindex/cancel` 定义在
    `POST /documents/{doc_id}/cancel` **之后** → 请求 `/documents/reindex/cancel`
    会匹配到 `{doc_id}/cancel`，`doc_id="reindex"` 类型校验失败 → **422**，
    该端点**从来无法访问**。这是既有缺陷，只是从未被测试覆盖。

    这条测试锁住顺序不变量（按文档内出现顺序比较）：
    凡是「字面量段」比「同段数的参数化路径」出现得更晚，就报错。
    """
    import re as _re

    lines = (API_DIR / "routes_documents.py").read_text(encoding="utf-8").split("\n")
    order: list[tuple[str, str, int]] = []
    for i, ln in enumerate(lines):
        m = _re.search(r"@router\.(get|post|put|patch|delete)\([\"']([^\"']+)[\"']", ln)
        if m:
            order.append((m.group(1), m.group(2), i))

    def is_parametrized(p: str) -> bool:
        return "{" in p

    # 冲突条件：**参数化路径在前、同构的字面量路径在后** ——
    # 前者会先匹配上，把后者遮蔽掉。
    # （反向不算冲突：字面量在前是**正确**顺序，本次修复要维持的正是它。）
    problems = []
    for i, (m1, p1, _) in enumerate(order):
        if not is_parametrized(p1):
            continue
        seg1 = p1.split("/")
        for m2, p2, _ in order[i + 1:]:
            if m2 != m1 or is_parametrized(p2):
                continue
            if len(p2.split("/")) == len(seg1) and all(
                a == b or a.startswith("{") for a, b in zip(seg1, p2.split("/"), strict=True)
            ):
                problems.append(f"{m2.upper()} {p2} 被更早注册的 {p1} 遮蔽，应移到它之前")
    assert not problems, "路由注册顺序冲突（静态路径被参数化路径遮蔽）：" + "；".join(problems)

#: 流式/文件响应端点：**不应**有 response_model
STREAMING_ENDPOINTS = [
    # ⚠️ 注意：`/documents/{doc_id}/stream` **不在此列** ——
    # 名字里有 stream，但它返回的是**文档流视图**（JSON 列表，段落/表格按序），
    # 不是 HTTP 流。当初凭名字把它误列进来，第六批加契约时被这条测试挡下，
    # 才发现是清单错了（而不是端点错了）—— 守护测试也会"记错事"。
    ("get", "/documents/{doc_id}/export"),
    ("post", "/sessions/{session_id}/messages/stream"),
    ("post", "/documents/reindex/stream"),
    ("get", "/documents/{doc_id}/pages/{page}/image"),
    ("get", "/documents/{doc_id}/images/{filename}"),
    ("get", "/documents/{doc_id}/download"),
    ("get", "/documents/{doc_id}/file"),
    ("get", "/traces/export"),
    ("get", "/documents/{doc_id}/export"),
]


def _scan() -> list[dict]:
    """扫描所有路由装饰器，返回 [{method, path, has_rm, returns}]。"""
    out: list[dict] = []
    for f in sorted(API_DIR.glob("routes_*.py")):
        lines = f.read_text(encoding="utf-8").split("\n")
        for i, ln in enumerate(lines):
            if not ln.lstrip().startswith("@router."):
                continue
            m = re.search(r"@router\.(get|post|put|patch|delete)", ln)
            if not m:
                continue
            block = ln
            j = i
            while ")" not in block and j + 1 < len(lines) and j - i < 4:
                j += 1
                block += lines[j]
            path = re.search(r'["\']([^"\']+)["\']', block)
            out.append({
                "file": f.name,
                "method": m.group(1),
                "path": path.group(1) if path else "?",
                "has_rm": "response_model" in block,
            })
    return out


def test_coverage_does_not_regress():
    """★ C1 进度只能前进：覆盖数低于基线说明有端点被摘掉契约或新增端点没写契约。"""
    endpoints = _scan()
    covered = [e for e in endpoints if e["has_rm"]]
    assert len(covered) >= BASELINE_COVERED, (
        f"响应契约覆盖数 {len(covered)} < 基线 {BASELINE_COVERED}；"
        f"新增端点必须补 response_model，已完成的不许摘掉"
    )


def test_must_have_contract_endpoints():
    """本批接入的端点必须仍然带着契约（逐条点名，防止被无意删除）。"""
    endpoints = {(e["method"], e["path"]) for e in _scan() if e["has_rm"]}
    missing = [(m, p) for m, p in MUST_HAVE_CONTRACT if (m, p) not in endpoints]
    assert not missing, f"以下端点丢失了 response_model：{missing}"


def test_streaming_endpoints_have_no_response_model():
    """★ 反向口径：流式/文件响应端点**不该**有 response_model。

    给流式端点加 `response_model` 会让 FastAPI 尝试把 `StreamingResponse` 当普通响应校验，
    是**改坏**而不是改好 —— C1 推进时容易"顺手统一"，所以单独锁住。
    """
    endpoints = {(e["method"], e["path"]): e for e in _scan()}
    for method, path in STREAMING_ENDPOINTS:
        e = endpoints.get((method, path))
        if e is None:
            continue  # 端点被重命名/移除时不误报（由别的测试覆盖存在性）
        assert not e["has_rm"], f"流式端点 {method.upper()} {path} 不应设置 response_model"


def test_scan_actually_finds_endpoints():
    """扫描器自身的健全性检查 —— 扫描逻辑坏了会让上面几条**假绿**。"""
    endpoints = _scan()
    assert len(endpoints) >= 70, f"只扫到 {len(endpoints)} 个端点，扫描逻辑可能失效"
