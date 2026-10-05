#!/usr/bin/env python
"""响应契约字段核对（C1 工具，2026-09-21）。

**为什么需要它**：给端点加 `response_model` 后，FastAPI 会按模型**裁剪响应** ——
模型里没声明的字段会被**静默丢掉**（前端拿到 `undefined`，不报错也不告警）。
C1 要补 50+ 个端点，靠人眼逐个比对字段既慢又必然漏。

本工具把这件事自动化：对每个「已接入契约的端点」取**真实响应**，
与模型的声明字段做差集，报出：
- **会被裁掉的字段**（实际有、契约没有）→ 这是必须修的，属静默数据丢失；
- **契约多出的字段**（契约有、实际没有）→ 若该字段有默认值则会静默填默认值，需人工判断是否合理。

用法：
    python scripts/check_response_contracts.py            # 核对全部
    python scripts/check_response_contracts.py --only DocDiagramOut

⚠️ 依赖真实库数据：无数据的端点会被跳过（不代表契约正确，只代表"这次没验证到"）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

#: (模型名, 取真实响应的路径, 从响应的哪一层取字段)
#: - list 端点：模型作用在**列表项**上
#: - dict 端点：模型作用在**响应本身**
#: - 路径里的 `{sid}` 会在运行时替换为**真实存在的首个会话 id**（见 `_resolve_sid`）——
#:   早先这里硬编码 `1`，而真实库首个会话早已不是 1，于是这些端点**被静默跳过**，
#:   一度让人以为「核对过了」。**硬编码的样本 id 是核对工具最隐蔽的失效点。**
CASES: list[tuple[str, str, str]] = [
    # 第一批
    ("CollectionOut", "/api/v1/collections", "items"),
    ("MemoryItemOut", "/api/v1/memories", "items"),
    # 第二批
    ("DocImageOut", "/api/v1/documents/10/images", "self"),
    ("DocTableOut", "/api/v1/documents/10/tables", "self"),
    ("DocDiagramOut", "/api/v1/documents/10/diagrams", "self"),
    ("UploadProgressOut", "/api/v1/documents/10/progress", "self"),
    ("DocTextOut", "/api/v1/documents/10/text", "self"),
    # 第三批
    ("TraceListItemOut", "/api/v1/traces?limit=3", "self"),
    ("SettingsOut", "/api/v1/settings", "self"),
    ("InventoryOut", "/api/v1/settings/inventory", "self"),
    # 第四批
    ("NetworkCapabilityOut", "/api/v1/sessions/{sid}/network-capability", "self"),
    ("EvalSummaryOut", "/api/v1/eval/summary", "self"),
    ("SessionUsageDetailOut", "/api/v1/usage/sessions/{sid}", "self"),
    # 第五批（POST 类：核对工具只发 GET，故仅登记能被 GET 验证的）
    # 2026-09-23 场景可见性
    ("DocScopeOut", "/api/v1/sessions/{sid}/doc-scope", "self"),
]


def _resolve_sid(client) -> int | None:
    """取真实存在的首个会话 id（供 `{sid}` 占位符使用）。

    **为什么不再硬编码 1**：真实库的会话 id 会变（删除后不复用），
    硬编码会让相关端点静默变成「跳过（HTTP 404）」—— 而\"跳过\"看起来和\"通过\"一样安静，
    于是契约核对**悄悄失去覆盖**却没有任何信号。这里取不到时**显式报告跳过原因**。
    """
    try:
        r = client.get("/api/v1/sessions?limit=1")
        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
            return int(r.json()[0]["id"])
    except Exception:  # noqa: BLE001  核对工具的辅助步骤，失败由调用方报出
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="核对响应契约字段（默认全部）")
    ap.add_argument("--only", help="只核对指定模型名")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from papermate import schemas
    from papermate.api.app import create_app

    client = TestClient(create_app())
    problems = 0
    checked = 0

    sid = _resolve_sid(client)
    if sid is None:
        print("  ⚠️ 未取到任何会话 → 依赖 `{sid}` 的端点将全部跳过（本次覆盖不完整）")
    else:
        print(f"  样本会话 id = {sid}（`{{sid}}` 占位符将替换为它）")
    print()

    for model_name, raw_path, level in CASES:
        if args.only and args.only != model_name:
            continue
        if "{sid}" in raw_path:
            if sid is None:
                print(f"  {model_name:<18} 跳过（无可用会话）")
                continue
            path = raw_path.replace("{sid}", str(sid))
        else:
            path = raw_path
        model = getattr(schemas, model_name, None)
        if model is None:
            print(f"  {model_name:<18} ⚠️ schemas 中不存在该模型")
            problems += 1
            continue

        r = client.get(path)
        if r.status_code != 200:
            print(f"  {model_name:<18} 跳过（HTTP {r.status_code}）")
            continue
        data = r.json()
        items = data.get(level) if isinstance(data, dict) and level != "self" else data
        items = items if isinstance(items, list) else [items]
        items = [i for i in items if isinstance(i, dict)]
        if not items:
            print(f"  {model_name:<18} 跳过（无数据）")
            continue

        declared = set(model.model_fields.keys())
        actual: set[str] = set()
        for it in items:
            actual |= set(it.keys())

        trimmed = sorted(actual - declared)
        extra = sorted(declared - actual)
        checked += 1
        marks = []
        if trimmed:
            marks.append(f"⚠️ 会被裁掉: {trimmed}")
            problems += 1
        if extra:
            marks.append(f"契约多出: {extra}")
        print(f"  {model_name:<18} 实际={len(actual):<3} 契约={len(declared):<3} "
              + (" | ".join(marks) if marks else "✓ 无字段丢失"))

    print()
    print(f"  核对 {checked} 个模型 · 问题 {problems} 个")
    if problems:
        print("  ⚠️「会被裁掉」= 实际返回里有、契约没声明 → response_model 会静默丢弃它，必须补上。")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
