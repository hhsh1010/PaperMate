#!/usr/bin/env python
"""静态扫描：端点**所有 return 分支**的字段并集 vs 契约声明字段（2026-09-21）。

## 为什么需要它（来自一次真实漏检）

`POST /workbench/para-transform` 的契约最初只写了 4 个字段，但它的**成功分支**返回 6 个
（多 `action` / `action_label`）。漏掉的两个字段会被 `response_model` **静默裁掉**。

而"取真实响应再写模型"这个方法**这次没兜住它**，因为核对时恰好只走到了**降级路径**
（当时 API 余额不足）—— 降级分支确实只返回 4 个字段。

→ 所以：**契约字段必须取「所有 return 分支的字段并集」，不能取「某次实测到的字段」。**

本工具用**静态分析**（而不是发请求）找出端点函数里每一个 `return {...}` 的顶层 key，
从而覆盖运行期难以触发的分支（成功路径要余额、异常路径要故障注入）。

用法：
    python scripts/check_return_branches.py            # 全部
    python scripts/check_return_branches.py --file src/papermate/api/routes_transform.py

⚠️ 局限（要清楚它测不到什么）：
- 只看 `return {` 这种**字面量字典**；`return some_var` / `return Out(...)` 无法解析（会跳过）；
- 不解析嵌套结构（只比顶层 key），因为本类漏检都发生在顶层。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / "src" / "papermate" / "api"


def _split_functions(lines: list[str]) -> dict[str, tuple[int, int]]:
    """返回 {函数名: (起, 止)} —— 以顶层 `def ` 为界。"""
    # ⚠️ 必须同时匹配 `async def` —— 端点大多是 async，漏掉会让 return 归属到**上一個**
    # 同步函数上，输出变成"函数名与字段对不上"的误导信息（实测踩过）。
    starts = [(i, m.group(1)) for i, ln in enumerate(lines)
              if (m := re.match(r"^(?:async\s+)?def\s+(\w+)", ln))]
    out: dict[str, tuple[int, int]] = {}
    for k, (idx, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        out[name] = (idx, end)
    return out


def _dict_keys(block: list[str], open_idx: int, base_indent: int) -> set[str]:
    """从 `return {` 起，提取与首个 key 同缩进的顶层 key。"""
    keys: set[str] = set()
    for ln in block[open_idx + 1:]:
        stripped = ln.strip()
        if stripped.startswith("}"):
            break
        m = re.match(r'^(\s*)"([^"]+)"\s*:', ln)
        if m and len(m.group(1)) == base_indent + 4:
            keys.add(m.group(2))
    return keys


def scan_file(path: Path) -> dict[str, set[str]]:
    """{函数名: 该函数所有 return 字典的字段并集}。"""
    lines = path.read_text(encoding="utf-8").split("\n")
    result: dict[str, set[str]] = {}
    for name, (start, end) in _split_functions(lines).items():
        block = lines[start:end]
        keys: set[str] = set()
        for i, ln in enumerate(block):
            m = re.match(r"^(\s*)return\s*\{", ln)
            if not m:
                continue
            keys |= _dict_keys(block, i, len(m.group(1)))
        if keys:
            result[name] = keys
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="列出各端点函数 return 字典的字段并集")
    ap.add_argument("--file", help="只扫某个文件（默认扫 routes_*.py）")
    args = ap.parse_args()

    files = [Path(args.file)] if args.file else sorted(API_DIR.glob("routes_*.py"))
    total = 0
    for f in files:
        res = scan_file(f)
        if not res:
            continue
        print(f"=== {f.name} ===")
        for name, keys in sorted(res.items()):
            total += 1
            print(f"  {name:<34} {len(keys)} 字段: {sorted(keys)}")
        print()
    print(f"  共 {total} 个含字面量 return 的函数。")
    print("  ⚠️ 用法：拿这里的字段并集去核对 response_model —— **并集里有、契约里没有的**，")
    print("     就是会被静默裁掉的字段（这正是 para-transform 漏掉 action/action_label 的原因）。")
    print("  ⚠️ 注意：本工具只列事实，不做判定（契约该不该含某字段需结合语义判断）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
