"""「另存为」目标路径的**统一校验**（P-04/P-05，2026-10-09）。

为什么单独成模块：`/documents/{id}/export-to` 与 `/files/write` 两个端点都要校验同一件事，
规则必须**逐字一致**（否则又是一个"两把尺子"）。与 `json_safe.py` / `pagination.py` 同类的单职责小模块。

## 安全边界（如实声明，不夸大也不含糊）
本地单用户应用：`dest` 来自**用户自己的原生「另存为」对话框**（前端 `dialog.save`）。
本校验只做三件事：① 拒绝空值 ② 拒绝相对路径（避免落到进程 CWD 这种不可预期的地方）
③ 要求父目录已存在（不替用户"顺手"建目录）。端点**只写不读**、不枚举目录、不返回目录内容。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

#: 单次写入上限。导出件都是人类可读文本/文档 ⇒ 64MB 足够，同时能挡住失控调用。
MAX_EXPORT_BYTES = 64 * 1024 * 1024


def require_abs_dest(dest: str) -> Path:
    """校验并返回 `Path(dest)`；不合法直接 400（消息可直接展示给用户）。"""
    raw = (dest or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="保存路径不能为空")
    p = Path(raw)
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail=f"保存路径必须是绝对路径：{raw!r}")
    if not p.parent.is_dir():
        raise HTTPException(status_code=400, detail=f"目标目录不存在：{p.parent}")
    return p
