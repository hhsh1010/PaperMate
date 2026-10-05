"""PaperMate · 数据库快照备份与恢复（L0-1，2026-09-16）

**为什么需要**：`papermate.db` 是会话 / 消息 / 文档 / 分块 / 文献集 / 长期记忆 / trace
的唯一真源（`docs/项目不足清单-对照五阶段清单复核.md` §3 D3）。
docx 有 `.bak`（保留 5 份）兜住改稿，但**数据库没有任何备份** → 一次误操作即不可恢复。

设计要点（决定了实现方式，改本文件前先读）：

1. **用 sqlite3 在线备份 API（`Connection.backup`），不用文件拷贝**。
   直接 `copy` 在 WAL 模式下会漏掉 `-wal` 中未 checkpoint 的数据，得到"看起来正常但缺数据"的快照；
   在线备份由 SQLite 自己保证页级一致性。
2. **恢复前先给"当前库"也留一份快照**（`pre_restore`）——
   误恢复到旧快照 = 又一次数据丢失，必须先兜一层。
3. **恢复用"写临时文件 + `os.replace`"原子替换**，并**删除 `-wal` / `-shm`**。
   ⚠️ 这是最容易踩的坑：替换主库文件却留着旧 `-wal`，SQLite 下次打开会把**旧 WAL 回放到新库上** → 库损坏。
4. **恢复必须在后端停止时执行**（有活动连接时替换文件不可靠）——
   `restore_snapshot` 会检查 WAL 是否存在并给出警告，但不代为杀进程。
5. 所有失败路径**不修改现有库**：任何异常都只影响临时文件。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: 快照默认保留份数。
#: **D4（2026-09-21）由 10 调为 7**：实测一库 12 MB 时 backups 稳态占用 121 MB
#: （10 份 × 12MB + 元数据），对本地单用户偏大；7 份仍能覆盖"连续几次迁移都出错"的场景，
#: 上限降到约 84 MB。策略见 `docs/数据留存与清理策略.md`。
DEFAULT_KEEP = 7

#: 快照文件名前缀
_PREFIX = "papermate-"


def sqlite_path() -> Path:
    """当前配置下的数据库文件路径。

    ⚠️ 必须读 `Settings.db_url`（**派生**属性），不是 `Settings.database_url` ——
    后者是"可选覆盖项"且默认空串，读它会把路径解析成 `.`，
    表现为 `sqlite3.OperationalError: unable to open database file`
    （2026-09-16 真实库演习时踩到：单测因为显式传 `db_path` 而全部通过）。
    """
    from ..config import get_settings

    url = get_settings().db_url
    if not url.startswith("sqlite:"):
        raise ValueError(f"db_backup 仅支持 sqlite，当前 db_url={url!r}")
    raw = url.split("///", 1)[-1]
    if not raw or raw == ":memory:":
        raise ValueError(f"无法从 db_url 解析文件路径：{url!r}")
    return Path(raw)


def backup_dir(db_path: Path | None = None) -> Path:
    """快照目录：默认与数据库同级的 `backups/`（跟着 data_dir 走，便于整体迁移）。"""
    p = Path(db_path) if db_path else sqlite_path()
    d = p.parent / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _connect_ro(path: Path) -> sqlite3.Connection:
    """只读连接（优先 `mode=ro`；WAL 待恢复时回退普通连接，避免直接失败）。"""
    try:
        return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:  # noqa: PERF203  极少见：WAL 需要恢复
        return sqlite3.connect(path)


def table_row_counts(path: Path) -> dict[str, int]:
    """各业务表的行数（用于快照前后一致性校验）。

    只统计**业务表**（排除 `sqlite_sequence` / `sqlite_stat*` 等内部表），
    因为这些才是"用户数据"的判据。

    ⚠️ 用 `closing()` 显式关闭连接：`with sqlite3.connect(...)` **只提交事务、不关连接**，
    在 Windows 上会留下句柄 → 之后 `prune` 删文件时会报 `WinError 32`（本文件踩过一次）。
    """
    counts: dict[str, int] = {}
    if not Path(path).exists():
        return counts
    with closing(_connect_ro(Path(path))) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (name,) in rows:
            try:
                counts[name] = int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            except sqlite3.Error:  # noqa: PERF203  表损坏时也要给出其余表的结果
                counts[name] = -1
    return counts


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_source(path: Path) -> sqlite3.Connection:
    """打开**源库**用于备份（连接方式直接影响快照是否完整）。

    为什么不是无脑用只读连接：**WAL 模式下只读打开会静默丢数据** ——
    SQLite 要读 `-wal` 需要 `-shm` 的读写权限，权限不足时只读连接退回"只读主库文件"，
    `backup()` 于是只拷到主库内容，**不报错**。
    实测（`tests/test_db_backup.py::test_backup_includes_uncheckpointed_wal_data`）：
    WAL 中已提交但未 checkpoint 的 2 行，只备份出 1 行 —— 得到一个"看着正常、实际缺数据"的快照。

    因此先探测 `journal_mode`：
      - `wal` → 用普通读写连接（SQLite 会正确合并 WAL；关闭时可能做一次 checkpoint，属正常行为）；
      - 其余模式（delete / truncate / persist）数据全在主库文件里 → 只读打开，完全不碰源库。
    """
    mode = ""
    try:
        with closing(_connect_ro(path)) as probe:
            mode = str(probe.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    except sqlite3.Error:  # 探测失败不致命：走下面的降级链
        mode = ""
    if mode == "wal":
        return sqlite3.connect(path)
    return _connect_ro(path)


def _copy_database(src: Path, dst: Path) -> None:
    """用 SQLite 在线备份 API 把 src 一致性复制到 dst。

    打开方式见 `_open_source`；若首选方式失败，退化为普通读写连接再试一次
    （**始终用 backup API，绝不退化成文件拷贝**）。

    ⚠️ **必须显式 `close()`**：`with sqlite3.connect(...)` 只负责提交/回滚事务，
    **不会关闭连接** → 在 Windows 上句柄未释放，随后的 `os.replace` / `unlink` 会报
    `WinError 32 另一个程序正在使用此文件`（本文件踩过一次，见模块 docstring）。
    """
    last: sqlite3.Error | None = None
    for opener in (_open_source, sqlite3.connect):
        conn_src = conn_dst = None
        try:
            conn_src = opener(src)
            dst.unlink(missing_ok=True)  # 上一次尝试可能留下半截目标文件
            conn_dst = sqlite3.connect(dst)
            conn_src.backup(conn_dst)  # 页级一致性快照
            conn_dst.commit()
            return
        except sqlite3.Error as e:  # noqa: PERF203  首选方式失败时降级重试
            last = e
        finally:
            for c in (conn_dst, conn_src):
                if c is not None:
                    try:
                        c.close()
                    except sqlite3.Error:
                        pass
    raise last if last else RuntimeError(f"数据库备份失败：{src}")



def create_snapshot(tag: str = "", db_path: Path | None = None, keep: int = DEFAULT_KEEP) -> Path:
    """创建一致性快照，返回快照路径。

    备份文件：`<data>/backups/papermate-YYYYmmdd-HHMMSS[-tag].db`
    元数据文件：同名 `.meta.json`（含表行数 / sha256 / 源库路径），供恢复前校验。

    `tag` 用于语义标记：`pre_migration`（启动迁移前）/ `pre_restore`（恢复前）。
    """
    src = Path(db_path) if db_path else sqlite_path()
    if not src.exists():
        raise FileNotFoundError(f"数据库不存在：{src}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{tag}" if tag else ""
    out_dir = backup_dir(src)
    final = out_dir / f"{_PREFIX}{stamp}{suffix}.db"

    # 同秒内重复调用：加序号，绝不覆盖既有快照
    n = 1
    while final.exists():
        final = out_dir / f"{_PREFIX}{stamp}{suffix}-{n}.db"
        n += 1

    fd, tmp_name = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _copy_database(src, tmp)  # ← 页级一致性快照（WAL 中未 checkpoint 的数据也会包含）
        os.replace(tmp, final)  # 原子落盘：不会出现"半个快照"
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # 清理失败不能掩盖真正的异常（Windows 上句柄未释放时会发生）
            logger.debug("临时快照清理失败：%s", tmp, exc_info=True)
        raise

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "source": str(src),
        "size_bytes": final.stat().st_size,
        "sha256": _sha256(final),
        "row_counts": table_row_counts(final),
    }
    final.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if keep > 0:
        prune(keep, db_path=src)

    logger.info("数据库快照已创建：%s（%d 字节，tag=%s）", final.name, meta["size_bytes"], tag or "-")
    return final


def list_snapshots(db_path: Path | None = None) -> list[dict]:
    """列出全部快照（新 → 旧），含元数据（缺失时现场补算行数）。"""
    p = Path(db_path) if db_path else sqlite_path()
    out: list[dict] = []
    for f in sorted(backup_dir(p).glob(f"{_PREFIX}*.db"), reverse=True):
        meta_file = f.with_suffix(".meta.json")
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        else:
            meta = {}
        meta.setdefault("created_at", datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"))
        meta.setdefault("size_bytes", f.stat().st_size)
        meta.setdefault("row_counts", table_row_counts(f))
        meta["path"] = str(f)
        meta["name"] = f.name
        out.append(meta)
    out.sort(key=lambda m: m["name"], reverse=True)
    return out


def _resolve(ref: str, db_path: Path | None = None) -> Path:
    """把 `--restore` / `--verify` 的入参解析为快照文件。

    支持四种写法（按顺序尝试）：
      1. 存在的路径（绝对 / 相对 CWD）；
      2. `backups/` 下的精确文件名；
      3. 省略前缀或后缀：`20260916-161932` / `20260916-161932.db` / `papermate-20260916-161932`；
      4. **子串匹配**（如 `--restore pre_migration`）—— 命中唯一才接受，
         命中多个则报错并列出候选，避免"以为恢复的是 A 其实是 B"。
    """
    p = Path(db_path) if db_path else sqlite_path()
    bdir = backup_dir(p)

    cand = Path(ref)
    if cand.exists():
        return cand

    for name in (ref, f"{ref}.db", f"{_PREFIX}{ref}", f"{_PREFIX}{ref}.db"):
        f = bdir / name
        if f.exists():
            return f

    hits = sorted(bdir.glob(f"*{ref}*.db"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        names = ", ".join(h.name for h in hits)
        raise FileNotFoundError(f"快照名不唯一，命中 {len(hits)} 个：{names}（请写完整文件名）")
    raise FileNotFoundError(f"找不到快照：{ref}（可用 --list 查看全部快照）")


def verify_snapshot(snapshot: Path, expect: dict | None = None) -> dict:
    """校验快照可用性：`PRAGMA integrity_check` + 表行数；可选与期望行数比对。

    返回 `{"ok": bool, "integrity": str, "row_counts": {...}, "diff": {...}}`。
    """
    snapshot = Path(snapshot)
    try:
        with closing(sqlite3.connect(snapshot)) as conn:
            integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    except sqlite3.Error as e:  # 非 SQLite 文件 / 文件损坏：把原因如实带回，不抛给调用方判断
        return {
            "ok": False,
            "integrity": f"error: {e}",
            "row_counts": {},
            "diff": {},
        }
    counts = table_row_counts(snapshot)
    diff: dict[str, list[int]] = {}
    if expect:
        for name, want in expect.items():
            got = counts.get(name)
            if got != want:
                diff[name] = [want, -1 if got is None else got]
    return {"ok": integrity == "ok" and not diff, "integrity": integrity, "row_counts": counts, "diff": diff}


def restore_snapshot(ref: str, db_path: Path | None = None, keep: int = DEFAULT_KEEP) -> dict:
    """从快照恢复数据库（**后端须已停止**）。

    流程：校验快照 → **给当前库留 `pre_restore` 快照** → 写临时文件 → 原子替换 → 清 `-wal`/`-shm`
    → 校验恢复结果与快照行数一致。

    返回 `{"restored": bool, "db": str, "safety_backup": str|None, "verify": {...}}`。
    """
    target = Path(db_path) if db_path else sqlite_path()
    snap = _resolve(ref, db_path)

    check = verify_snapshot(snap)
    if check["integrity"] != "ok":
        raise RuntimeError(f"快照完整性校验失败（integrity_check={check['integrity']}），已中止恢复")

    wal = target.with_name(target.name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        logger.warning(
            "检测到非空 WAL（%s）：说明可能有后端仍在运行。恢复将丢弃 WAL 中未落盘的事务，"
            "请确认后端已停止。", wal.name,
        )

    safety: Path | None = None
    if target.exists() and target.stat().st_size > 0:
        safety = create_snapshot(tag="pre_restore", db_path=target, keep=keep)

    tmp = target.with_name(target.name + ".restore-tmp")
    shutil.copyfile(snap, tmp)
    os.replace(tmp, target)  # 原子替换
    # ⚠️ 必须清理：残留的旧 WAL 会在下次打开时被回放到新库上 → 损坏
    for extra in ("-wal", "-shm"):
        side = target.with_name(target.name + extra)
        side.unlink(missing_ok=True)

    expect = table_row_counts(snap)
    result = verify_snapshot(target, expect=expect)
    if not result["ok"]:
        raise RuntimeError(f"恢复后校验不一致：{result['diff']}（安全快照：{safety}）")

    logger.info("数据库已从快照恢复：%s → %s", snap.name, target)
    return {
        "restored": True,
        "db": str(target),
        "snapshot": str(snap),
        "safety_backup": str(safety) if safety else None,
        "verify": result,
    }


def prune(keep: int = DEFAULT_KEEP, db_path: Path | None = None) -> list[str]:
    """只保留最近 `keep` 份快照（连同 `.meta.json`），返回被删除的文件名。"""
    snaps = list_snapshots(db_path)
    removed: list[str] = []
    for meta in snaps[keep:]:
        f = Path(meta["path"])
        f.unlink(missing_ok=True)
        f.with_suffix(".meta.json").unlink(missing_ok=True)
        removed.append(f.name)
    if removed:
        logger.info("已清理 %d 份旧快照：%s", len(removed), ", ".join(removed))
    return removed


def snapshot_if_exists(tag: str = "pre_migration") -> Path | None:
    """尽力而为地建一份快照，**任何失败都只记日志不抛出** —— 供启动迁移前调用。

    启动路径上绝不能因为"备份失败"而阻断服务；但也不能静默：
    失败会以 `WARNING` 落 `app.log`，便于事后发现"迁移前其实没备份成功"。
    """
    try:
        p = sqlite_path()
        if not p.exists() or p.stat().st_size == 0:
            return None
        return create_snapshot(tag=tag, db_path=p)
    except Exception:  # noqa: BLE001  启动路径不允许因备份失败而中断
        logger.warning("启动前快照失败（服务继续启动，请检查 backups 目录权限）", exc_info=True)
        return None
