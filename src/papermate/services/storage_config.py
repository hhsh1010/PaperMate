"""存储位置配置（读写 `app-settings.json` 的"待迁移请求"）。

## 为什么单独一个模块

「改存储位置」这件事的**权威执行者是桌面启动器（Rust）**，不是后端：

* 执行时机必须在**后端启动之前** —— 那一刻 SQLite 未打开、venv 未加载，
  才不存在"拷到一半"导致数据不一致的可能（见 ``lib.rs::apply_pending_migration``）。
* 后端只负责**收下请求 + 立刻回显校验结果**，把请求写进配置文件。

## 配置载体

``%LOCALAPPDATA%\\PaperMate\\app-settings.json``（**固定位置**，由启动器通过
``PAPERMATE_APP_SETTINGS`` 环境变量告知）。★ 不能放在数据目录里 —— 那是鸡生蛋
（要知道位置才能读到位置）。

⚠️ 本模块在**没有**该环境变量时**不可用**（例如开发态直接起 uvicorn）：
此时 API 返回 ``available=false`` 并说明原因，**不会**去别处乱写文件。
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: 与 Rust 侧 ``MIN_FREE_GB``、``dirs`` 布局**必须一致**（两边不同步就会口径不一）。
MIN_FREE_GB = 8.0

#: 与 ``lib.rs`` 的 ``dirs`` 布局一致（data / huggingface / venv / uv-cache）
SUBDIRS = ("data", "huggingface", "venv", "uv-cache")

#: 只对前两个目录量体积（venv / uv-cache 文件数上万，量一次要好几秒，UI 用不上）
SIZE_SUBDIRS = ("data", "huggingface")

_MAX_FILES_FOR_SIZE = 20000


@dataclass
class DriveInfo:
    root: str
    free_gb: float
    total_gb: float
    kind: str  # FIXED / REMOVABLE / REMOTE / UNKNOWN


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def app_settings_path() -> Path | None:
    """配置文件的**固定**位置；拿不到（例如开发态直起 uvicorn）⇒ None。"""
    raw = os.environ.get("PAPERMATE_APP_SETTINGS", "").strip()
    return Path(raw) if raw else None


def load() -> dict:
    """读配置；文件不存在或损坏都返回 ``{}``（**绝不抛异常**）。"""
    p = app_settings_path()
    if p is None or not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(data: dict) -> None:
    """**原子**写配置（先写 ``.tmp`` 再改名）—— 半截文件会让启动器读不出来。"""
    p = app_settings_path()
    if p is None:
        raise RuntimeError("未注入 PAPERMATE_APP_SETTINGS，无法写入存储位置配置")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp.replace(p)
    except OSError:
        # Windows 上目标存在时 replace 偶尔失败 ⇒ 退一步用"删 + 改名"
        try:
            p.unlink()
            tmp.replace(p)
        except OSError:
            if p.is_file():  # 旧的还在 ⇒ 至少不算更坏
                return
            raise


def sub_dirs(root: Path) -> dict[str, str]:
    """root 下四个子目录的绝对路径（键与 ``SUBDIRS`` 对齐）。"""
    return {name: str((root / name).resolve() if (root / name).is_absolute() else root / name) for name in SUBDIRS}


def dir_size_mb(path: Path, max_files: int = _MAX_FILES_FOR_SIZE) -> tuple[float, bool]:
    """目录体积（MB）与"是否因文件数超限而截断"。

    ★ 有 ``max_files`` 上限：venv / uv-cache 上万文件时全量遍历要好几秒，
      而前端只要一个粗略数字 ⇒ 截断比卡住好。截断时返回的是**下界**。
    """
    total = 0
    count = 0
    truncated = False
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                else:
                    total += e.stat(follow_symlinks=False).st_size
                    count += 1
                    if count >= max_files:
                        truncated = True
                        stack.clear()
                        break
            except OSError:
                continue
    return total / 1048576, truncated


def _existing_ancestor(p: Path) -> Path:
    """向上找第一个**已存在**的目录（`shutil.disk_usage` 要求路径存在）。"""
    cur = p
    for _ in range(64):
        if cur.exists():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return cur


def free_gb(p: Path) -> float:
    try:
        return shutil.disk_usage(_existing_ancestor(p)).free / 1073741824
    except OSError:
        return -1.0


def list_drives() -> list[DriveInfo]:
    """列出**本机**磁盘（含可用/总容量）。

    ★ 用 ``GetLogicalDrives`` + ``GetDriveTypeW``（ctypes，**不加依赖**）
      而不是 ``Path("Z:\\\\").exists()`` —— 后者在断开的网络盘上会**卡住数秒**。
    ★ 顺便把**网络盘也列出来**（但标注为 REMOTE）：设置页据此明确劝阻用户。
    """
    if os.name != "nt":
        return []
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        mask = kernel32.GetLogicalDrives()
    except (AttributeError, OSError):
        return []

    kind_names = {2: "REMOVABLE", 3: "FIXED", 4: "REMOTE", 5: "CDROM", 6: "RAMDISK"}
    out: list[DriveInfo] = []
    for i in range(26):
        if not (mask >> i) & 1:
            continue
        root = f"{chr(ord('A') + i)}:\\"
        kind = kind_names.get(int(kernel32.GetDriveTypeW(ctypes.c_wchar_p(root))), "UNKNOWN")
        free = total = 0.0
        try:
            usage = shutil.disk_usage(root)
            free, total = usage.free / 1073741824, usage.total / 1073741824
        except OSError:
            continue
        out.append(DriveInfo(root=root, free_gb=free, total_gb=total, kind=kind))
    return out


def validate_root(target: str, *, install_dir: Path | None = None) -> ValidationResult:
    """**建议性**校验（后端这一侧），返回错误/警告列表。

    ★★ 这里刻意只做**能在后端可靠判断**的几条；权威校验在启动器
      （``lib.rs::validate_data_root``，重启后执行）—— 因为它知道"安装目录"，
      且那时才真正要动文件。
      ⇒ **两边规则不一致是有意的**，但凡两边都查的项，必须口径一致。
    """
    errors: list[str] = []
    warnings: list[str] = []
    raw = (target or "").strip()
    if not raw:
        return ValidationResult(False, ["请填写目标路径"], [])
    if not (raw.startswith("\\") or (len(raw) > 2 and raw[1] == ":")):
        return ValidationResult(False, ["请填写完整路径（例如 D:\\PaperMate）"], [])
    if raw.startswith("\\\\") or raw.startswith("//"):
        errors.append("不能使用网络位置（\\\\服务器\\共享）：SQLite 在网络盘上的锁语义不可靠。")
    p = Path(raw)

    # ★ 盘符必须带尾分隔符才能和 list_drives() 的 `C:\` 对上：
    #   早期版本拿 `Path.drive`（`"C:"`）去比 ⇒ 永远匹配不上 ⇒ 误报"找不到磁盘"。
    root_str = raw[:3] if len(raw) >= 2 and raw[1] == ":" and raw[2] in "\\/" else ""
    if root_str:
        root_str = f"{root_str[:2]}\\"
        match = next((d for d in list_drives() if d.root == root_str), None)
        if match is None:
            errors.append(f"找不到磁盘 {root_str}（未连接？）")
        elif match.kind == "REMOTE":
            errors.append(f"{root_str} 是网络映射盘：SQLite 在其上锁不可靠，请改选本机磁盘。")

    if install_dir is not None:
        try:
            if os.path.normcase(str(p)) == os.path.normcase(str(install_dir)) or str(p).lower().startswith(
                str(install_dir).lower() + os.sep
            ):
                errors.append("不要把数据放在应用目录里 —— 升级时会被覆盖。")
        except OSError:
            pass

    if not errors:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            errors.append(f"无法创建目录：{e}")
        else:
            # ★ 真写真删：Windows 上 os.access(W_OK) 是假信号（本项目 10-06 血泪）
            probe = p / ".papermate-write-probe"
            try:
                probe.write_bytes(b"ok")
            except OSError as e:
                errors.append(f"该位置不可写：{e}")
            finally:
                try:
                    probe.unlink(missing_ok=True)
                except OSError:
                    pass
            fg = free_gb(p)
            if fg >= 0 and fg < MIN_FREE_GB:
                errors.append(
                    f"该磁盘剩余 {fg:.1f} GB，至少需要 {MIN_FREE_GB:.0f} GB"
                    "（模型权重约 4.5GB + 运行环境约 4GB）。"
                )
            if not errors and fg >= MIN_FREE_GB:
                warnings.append(f"位置校验通过（剩余 {fg:.1f} GB）。将使用**新位置**，重启后生效。")
    return ValidationResult(not errors, errors, warnings)


def current_root(default_root: Path) -> str:
    """当前生效的 root（优先配置文件里的 ``data_root``）。"""
    raw = str(load().get("data_root") or "").strip()
    return raw or str(default_root)


def request_change(new_root: str, *, migrate: bool) -> dict:
    """写入**待迁移请求**（``pending_root`` + ``migrate``），保留既有 ``data_root``。

    ★ 只写请求、不执行 —— 执行由启动器在**后端启动之前**做（见模块头注释）。
    ★ 原样保留 ``data_root``：万一迁移失败，启动器还能回退到现在这个位置。
    """
    data = load()
    data.setdefault("schema", 1)
    # ⚠️ `.strip()` 必须作用在**字符串**上（早期版本写成 `Path(new_root).strip()` ⇒ AttributeError）
    data["pending_root"] = str(Path(str(new_root).strip()))
    data["migrate"] = bool(migrate)
    save(data)
    return data