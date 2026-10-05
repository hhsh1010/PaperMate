"""数据库快照备份 / 恢复 CLI（L0-1，2026-09-16）

核心逻辑在 `src/papermate/services/db_backup.py`，本文件只是命令行外壳。

用法：
  python -m scripts.db_backup --create                 # 建快照（保留最近 10 份）
  python -m scripts.db_backup --create --tag before_x  # 带语义标签
  python -m scripts.db_backup --list                   # 列出全部快照（新→旧）
  python -m scripts.db_backup --verify <ref>           # 只校验快照，不改任何东西
  python -m scripts.db_backup --restore <ref>          # 恢复（⚠️ 后端须已停止）

`<ref>` 支持三种写法：快照文件名 / 省略前缀的名字（如 `20260916-160000`）/ 绝对路径。

⚠️ 恢复前会自动给**当前库**留一份 `pre_restore` 快照；
   恢复是原子替换并会清理 `-wal` / `-shm`（残留旧 WAL 会损坏新库，见模块 docstring）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from papermate.services import db_backup as bk  # noqa: E402


def _fmt_size(n: int) -> str:
    return f"{n / 1024 / 1024:.2f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def cmd_list() -> int:
    snaps = bk.list_snapshots()
    if not snaps:
        print("暂无快照。用 --create 建第一份。")
        return 0
    print(f"数据库：{bk.sqlite_path()}")
    print(f"快照目录：{bk.backup_dir()}\n")
    rows = []
    for i, s in enumerate(snaps, 1):
        tag = f"[{s['tag']}] " if s.get("tag") else ""
        rows.append((str(i), tag + s["name"], s["created_at"], _fmt_size(s["size_bytes"]), s.get("row_counts") or {}))
    w_name = max(len(r[1]) for r in rows)
    print(f"{'#':<4}{'文件名':<{w_name + 2}}{'时间':<22}{'大小':<11}行数摘要")
    for idx, name, created, size, counts in rows:
        brief = " ".join(f"{k}={v}" for k, v in counts.items())
        print(f"{idx:<4}{name:<{w_name + 2}}{created:<22}{size:<11}{brief}")
    return 0


def cmd_create(tag: str, keep: int) -> int:
    p = bk.create_snapshot(tag=tag, keep=keep)
    meta = (p.parent / p.name).with_suffix(".meta.json")
    print(f"已创建快照：{p}")
    print(f"元数据：{meta}")
    return 0


def cmd_verify(ref: str) -> int:
    snap = bk._resolve(ref)
    r = bk.verify_snapshot(snap)
    print(f"快照：{snap}")
    print(f"integrity_check = {r['integrity']}")
    print("表行数：")
    for name, n in r["row_counts"].items():
        print(f"  {name:<24}{n}")
    print("结果：" + ("OK（可用）" if r["ok"] else f"不一致 {r['diff']}"))
    return 0 if r["ok"] else 1


def cmd_restore(ref: str, keep: int) -> int:
    print("⚠️ 恢复会替换当前数据库，请确认后端已停止。继续…")
    r = bk.restore_snapshot(ref, keep=keep)
    print(f"已恢复：{r['db']}")
    print(f"来源快照：{r['snapshot']}")
    if r["safety_backup"]:
        print(f"恢复前的当前库已备份为：{r['safety_backup']}")
    print("恢复后行数：")
    for name, n in r["verify"]["row_counts"].items():
        print(f"  {name:<24}{n}")
    print("校验：OK（行数与快照完全一致）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="PaperMate 数据库快照备份 / 恢复（L0-1）")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--create", action="store_true", help="创建快照（默认动作）")
    g.add_argument("--list", action="store_true", help="列出全部快照")
    g.add_argument("--verify", metavar="REF", help="校验快照（不改动任何数据）")
    g.add_argument("--restore", metavar="REF", help="从快照恢复（后端须停止）")
    ap.add_argument("--tag", default="", help="快照标签，如 pre_migration / before_upgrade")
    ap.add_argument("--keep", type=int, default=bk.DEFAULT_KEEP,
                    help=f"保留份数（默认 {bk.DEFAULT_KEEP}，0=不清理）")
    args = ap.parse_args()

    if args.list:
        return cmd_list()
    if args.verify:
        return cmd_verify(args.verify)
    if args.restore:
        return cmd_restore(args.restore, args.keep)
    return cmd_create(args.tag, args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
