"""本地文件写入 —— 「另存为」的后端落盘口（P-05，2026-10-09）。

## 为什么需要它
桌面端（Tauri / WebView2）里浏览器的 `<a download>` 是**静默落盘到系统下载文件夹**、
界面零反馈（P-04 实测：用户以为无效、连点 12 次，产生 12 个同名副本）。
改成「原生另存为 → 后端写盘 → 明确反馈」后，**前端生成**的字节（CSV / JSONL / Markdown）
需要一个落盘通道 —— 就是这里。

（文档类导出如 docx 走 `/documents/{id}/export-to`：它由后端直接产出字节，不经本接口，
  两者的路径校验共用 `dest_path.require_abs_dest`。）
"""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, HTTPException

from ..schemas import WriteFileIn, WriteFileOut
from .dest_path import MAX_EXPORT_BYTES, require_abs_dest

router = APIRouter()


@router.post("/files/write", response_model=WriteFileOut)
def write_local_file(body: WriteFileIn) -> dict:
    """把 base64 内容写到用户选定的**绝对路径**（只写不读；路径来自原生「另存为」对话框）。

    - 400：路径为空 / 相对路径 / 父目录不存在 / 非合法 base64 / 内容超限；
    - 200：`{path, bytes}` —— 前端据此提示「已保存到 …」。
    """
    dest = require_abs_dest(body.dest)
    try:
        data = base64.b64decode(body.content_b64 or "", validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"content_b64 不是合法 base64：{e}") from e
    if len(data) > MAX_EXPORT_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"内容过大（>{MAX_EXPORT_BYTES // 1048576}MB），拒绝写入",
        )
    dest.write_bytes(data)
    return {"path": str(dest), "bytes": len(data)}
