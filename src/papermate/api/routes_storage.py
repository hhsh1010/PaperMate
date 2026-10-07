"""存储位置 API（2026-10-07）：查当前位置 / 申请改位置。

## 分工（★ 别在这里搬数据）

* **本模块只做两件事**：把当前布局与磁盘信息报给设置页；把"改位置"的请求
  **写进配置文件**（``pending_root`` + ``migrate``），并给出**建议性**校验结果。
* **真正的搬迁由桌面启动器在重启时执行**（``lib.rs::apply_pending_migration``），
  且刻意放在**后端启动之前** —— 那一刻 SQLite 未打开、venv 未加载，
  才不存在"拷到一半"导致数据不一致的可能。

## 为什么不复用 ``PUT /settings``

两类配置的**生效时机**根本不同：``/settings`` 写的 ``provider_settings.json``
**立即生效**；而存储位置**必须重启**（进程级 env 在 spawn 那刻就固定了）。
塞进同一个端点会让"保存成功但没生效"变成常态。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..schemas import StorageDrive, StorageOut, StorageSize, StorageUpdateIn, StorageUpdateOut
from ..services import storage_config as sc
from ..services.rag_deps import backend_root

router = APIRouter()

_UNAVAILABLE = (
    "当前不是桌面端启动（进程里没有 PAPERMATE_APP_SETTINGS）。"
    "该能力依赖启动器写入/读取配置文件，开发态直起 uvicorn 时不提供。"
)


def _current_root() -> Path:
    """当前生效的数据根目录 = ``<data_dir>/..``（桌面布局：``<root>/data``）。"""
    return get_settings().resolved_data_dir.parent


@router.get("/settings/storage", response_model=StorageOut)
def get_storage() -> dict:
    """当前存储布局 + 本机磁盘（设置页渲染用）。"""
    root = _current_root()
    cfg_path = sc.app_settings_path()
    if cfg_path is None:
        return {
            "available": False,
            "unavailable_reason": _UNAVAILABLE,
            "data_root": str(root),
            "dirs": sc.sub_dirs(root),
            "drives": [],
            "min_free_gb": sc.MIN_FREE_GB,
            "restart_required": False,
        }

    data = sc.load()
    dirs = sc.sub_dirs(root)
    # ★ 只量 data / huggingface：venv 与 uv-cache 上万文件，量一次好几秒，UI 用不上。
    #   ★ 每个目录**只量一次**（早期版本在推导式里调了两次 ⇒ 白跑一倍时间）。
    sizes: dict[str, StorageSize] = {}
    for name in sc.SIZE_SUBDIRS:
        mb, truncated = sc.dir_size_mb(Path(dirs[name]))
        sizes[name] = StorageSize(mb=round(mb, 1), truncated=truncated)
    return {
        "available": True,
        "config_path": str(cfg_path),
        "data_root": str(data.get("data_root") or root),
        "pending_root": data.get("pending_root"),
        "migrate": bool(data.get("migrate") or False),
        "dirs": dirs,
        "sizes_mb": sizes,
        "drives": [
            StorageDrive(root=d.root, free_gb=round(d.free_gb, 1), total_gb=round(d.total_gb, 1), kind=d.kind)
            for d in sc.list_drives()
        ],
        "min_free_gb": sc.MIN_FREE_GB,
        # 有待生效请求 ⇒ 设置页要显示"重启后生效"横幅
        "restart_required": bool(str(data.get("pending_root") or "").strip()),
    }


@router.put("/settings/storage", response_model=StorageUpdateOut)
def put_storage(body: StorageUpdateIn) -> dict:
    """申请改存储位置。**只写请求，不执行**；重启后由启动器完成搬迁。"""
    if sc.app_settings_path() is None:
        raise HTTPException(status_code=400, detail=_UNAVAILABLE)

    target = (body.data_root or "").strip()
    result = sc.validate_root(target, install_dir=backend_root())
    if not result.ok:
        raise HTTPException(status_code=400, detail="；".join(result.errors))

    try:
        sc.request_change(target, migrate=body.migrate)
    except (OSError, RuntimeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"写入存储位置配置失败：{e}") from e

    warnings = list(result.warnings)
    if body.migrate:
        warnings.append(
            "重启后会先搬迁既有数据（data/ · huggingface/ · uv-cache/），"
            "**原位置的副本会保留不删**；venv 会在新位置重建。"
        )
    else:
        warnings.append("重启后将在新位置以**空数据目录**启动（原位置数据保留不动）。")
    warnings.append("需要**重启应用**才会生效。")
    return {"ok": True, "restart_required": True, "warnings": warnings}