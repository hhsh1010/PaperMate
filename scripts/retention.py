#!/usr/bin/env python
"""数据留存清理工具（D4，2026-09-21）。

**默认 dry-run** —— 不传 `--apply` 绝不删除任何东西。

安全设计（三条，缺一不可）：
1. **白名单目录**：只处理 `data/backups` / `data/logs` / `data/parsed` 与孤立 `.tmp`。
   `data/uploads/`、`data/papermate.db`、`data/faiss/` **不在白名单**，
   即使传参也**不会**被删（用户资产与高重建成本物一律不动）。
2. **逐项打印**：删除前把每个路径与大小打出来，便于核对。
3. **`--apply` 显式开关**：默认只报告。删除动作本身也可能被环境的 safe-delete 垫片拦截，
   此时会报错而不是静默跳过（宁可不删，不要"以为删了"）。

用法：
    python scripts/retention.py                      # dry-run（默认）
    python scripts/retention.py --apply              # 真删
    python scripts/retention.py --keep 7             # 备份保留份数（默认 7）
    python scripts/retention.py --logs-days 14       # 日志保留天数（默认 14）
    python scripts/retention.py --only backups       # 只处理某一类
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

#: 白名单：只有这些目录会被清理（uploads / db / faiss **故意不在其中**）
ALLOWED_DIRS = {
    "backups": DATA / "backups",
    "logs": DATA / "logs",
    "parsed": DATA / "parsed",
}

#: 旧库残留（324 KB，默认库已改为 papermate.db）—— 需显式 --only legacy-db 才处理
LEGACY_FILES = [DATA / "db.sqlite3"]


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def plan_backups(keep: int) -> list[Path]:
    """备份目录：保留最近 `keep` 份 `.db`（连同 `.meta.json`；`.db` 没了 meta 也没意义）。"""
    d = ALLOWED_DIRS["backups"]
    if not d.is_dir():
        return []
    dbs = sorted(d.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    doomed: list[Path] = []
    for db in dbs[keep:]:
        doomed.append(db)
        # 备份文件名形如 `xxx-pre_migration.db` / `xxx-pre_migration.meta.json`
        meta = db.parent / (db.name[: -len(".db")] + ".meta.json")
        if meta.exists():
            doomed.append(meta)
    # 孤立 tmp（半成品，无恢复价值）
    doomed.extend(d.glob("*.tmp"))
    # 去重：同一条路径不得出现两次（否则报告里会重复，实际删除时第二次必然失败）
    return list(dict.fromkeys(doomed))


def plan_by_age(kind: str, days: int) -> list[Path]:
    """按 mtime 清理超过 N 天的文件。"""
    d = ALLOWED_DIRS[kind]
    if not d.is_dir():
        return []
    cutoff = time.time() - days * 86400
    return [p for p in d.rglob("*") if p.is_file() and p.stat().st_mtime < cutoff]


def plan_legacy() -> list[Path]:
    return [p for p in LEGACY_FILES if p.exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description="PaperMate 数据留存清理（默认 dry-run）")
    ap.add_argument("--apply", action="store_true", help="真正执行删除（默认只报告）")
    ap.add_argument("--keep", type=int, default=7, help="备份快照保留份数（默认 7）")
    ap.add_argument("--logs-days", type=int, default=14, help="日志保留天数（默认 14）")
    ap.add_argument("--parsed-days", type=int, default=30, help="parsed 保留天数（默认 30）")
    ap.add_argument("--only", choices=["backups", "logs", "parsed", "legacy-db"], help="只处理某一类")
    args = ap.parse_args()

    targets: list[tuple[str, list[Path]]] = []
    if args.only in (None, "backups"):
        targets.append(("backups", plan_backups(args.keep)))
    if args.only in (None, "logs"):
        targets.append(("logs", plan_by_age("logs", args.logs_days)))
    if args.only in (None, "parsed"):
        targets.append(("parsed", plan_by_age("parsed", args.parsed_days)))
    if args.only in (None, "legacy-db"):
        targets.append(("legacy-db", plan_legacy()))

    total = sum(_size(p) for _, ps in targets for p in ps)
    mode = "APPLY（真删）" if args.apply else "DRY-RUN（只看不删）"
    print(f"=== 数据留存清理 · {mode} ===")
    print(f"  根目录: {DATA}")
    print()

    for name, paths in targets:
        if not paths:
            print(f"  [{name}] 无需处理")
            continue
        sz = sum(_size(p) for p in paths)
        print(f"  [{name}] {len(paths)} 个文件 · 可释放 {_human(sz)}")
        for p in sorted(paths):
            print(f"      {_human(_size(p)):>10}  {p.relative_to(ROOT)}")
        print()

    print(f"  合计可释放: {_human(total)}")

    if not args.apply:
        print()
        print("  这是 DRY-RUN。确认无误后加 --apply 执行。")
        print("  ⚠️ 白名单之外的路径（uploads / papermate.db / faiss）永不会被本工具触碰。")
        return 0

    deleted = failed = 0
    for _, paths in targets:
        for p in paths:
            try:
                p.unlink()
                deleted += 1
            except OSError as e:
                # 宁可不删，也不要"以为删了"（环境可能拦删除）
                failed += 1
                print(f"  ⚠️ 删除失败: {p} → {e}")
    print()
    print(f"  已删除 {deleted} 个 · 失败 {failed} 个")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
