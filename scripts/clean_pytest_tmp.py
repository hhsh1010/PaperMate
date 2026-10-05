#!/usr/bin/env python
"""清理 pytest 残留的 `garbage-*` 临时目录（2026-09-21）。

## 背景（为什么会有这些垃圾）

pytest 把临时目录放在 `%TEMP%/pytest-of-<user>/` 下：
- 当前运行 → `pytest-N/`（保留）；
- 更早的运行 → 改名为 `garbage-*` 并**尝试删除**。

在本开发环境里，这类批量删除会被 **safe-delete 垫片拦截**（`SAFE_DELETE_BULK_CONFIRM_REQUIRED`），
于是 `garbage-*` 只改名、删不掉 → **永久残留**。实测该目录膨胀到 **547 MB / 836 个顶层目录**。

配套改动：`pyproject.toml` 里设了 `tmp_path_retention_count = 0`（不再保留历史运行目录），
从源头减少新增；本脚本负责清理**存量**。

## ⚠️ 删除配额（重要）

垫片按**整个对话轮累计**计删除数（`scope: turn`，每轮上限 50 个）。
所以本脚本**默认每批只删 `--limit`（默认 50）个**，超出的留到下一轮。
想一次清完，最省事的办法是**在资源管理器里手动删**（用户操作不受垫片约束）。

用法：
    python scripts/clean_pytest_tmp.py                 # dry-run：只报告
    python scripts/clean_pytest_tmp.py --apply         # 真删（每批 50 个）
    python scripts/clean_pytest_tmp.py --apply --limit 20
    python scripts/clean_pytest_tmp.py --all           # 只报告全部规模（不删）
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

#: 只清理这个名字前缀的目录 —— 绝不碰当前运行的 `pytest-N/`
PREFIX = "garbage-"


def _size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p, onerror=lambda e: None):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _human(b: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {u}"
        b /= 1024  # type: ignore[assignment]
    return f"{b:.1f} TB"


def main() -> int:
    ap = argparse.ArgumentParser(description="清理 pytest garbage-* 残留（默认 dry-run）")
    ap.add_argument("--apply", action="store_true", help="真正执行删除（默认只报告）")
    ap.add_argument("--limit", type=int, default=50, help="本批最多删多少个（避开每轮 50 的删除配额）")
    ap.add_argument("--all", action="store_true", help="只统计全部规模，不删")
    args = ap.parse_args()

    base = Path(tempfile.gettempdir()) / f"pytest-of-{Path.home().name}"
    if not base.is_dir():
        print(f"  目录不存在：{base}")
        return 0

    targets = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith(PREFIX)])
    total_bytes = sum(_size(d) for d in targets)
    living = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("pytest-")]

    print(f"=== pytest 临时目录清理 · {'APPLY（真删）' if args.apply else 'DRY-RUN'} ===")
    print(f"  根目录: {base}")
    print(f"  garbage-* 残留: {len(targets)} 个 · {_human(total_bytes)}")
    print(f"  当前运行目录（pytest-*，**不动**）: {len(living)} 个")
    print()

    if args.all or not args.apply:
        batch = targets[: args.limit]
        size = sum(_size(d) for d in batch)
        print(f"  本批计划删除: {len(batch)} 个 · {_human(size)}")
        for d in batch[:8]:
            print(f"      {_human(_size(d)):>9}  {d.name}")
        if len(batch) > 8:
            print(f"      ... 另 {len(batch) - 8} 个")
        if not args.apply:
            print()
            print("  这是 DRY-RUN。确认后加 --apply 执行。")
            print("  ⚠️ 每轮删除上限 50 个（垫片按对话轮累计）—— 剩余需下一轮继续，")
            print("     或在资源管理器里手动删除整个 garbage-* 集合（最省事）。")
        return 0

    batch = targets[: args.limit]
    ok = fail = 0
    freed = 0
    for d in batch:
        sz = _size(d)
        try:
            shutil.rmtree(d)
            ok += 1
            freed += sz
        except Exception as e:  # noqa: BLE001  删除失败要如实报，不能静默
            fail += 1
            print(f"  ⚠️ 删除失败: {d.name} → {type(e).__name__}: {e}")
    print(f"  已删 {ok} 个 · 失败 {fail} 个 · 释放约 {_human(freed)}")
    left = len([d for d in base.iterdir() if d.is_dir() and d.name.startswith(PREFIX)])
    print(f"  剩余 garbage-*: {left} 个（下一轮继续，或手动删）")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
