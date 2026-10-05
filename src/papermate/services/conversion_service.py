"""PaperMate · 文档格式转换服务（PDF ↔ Word 互转，2026-08-23）

互转依赖外部办公引擎（本项目不内置转换内核，如实说明）：
- **docx → pdf**：LibreOffice headless（`soffice --headless --convert-to pdf`）或
  Microsoft Word COM（需 pywin32 + 已安装授权 Word）；
- **pdf → docx**：Word COM 打开 PDF 另存为 docx；或 LibreOffice 的 PDF 导入再导出
  （保真度有限，扫描件无文本层时无法转）。

`detect_engine()` 探测本机可用引擎；`convert()` 按目标格式路由。引擎缺失时抛
ConversionError（提示安装路径），绝不静默失败。默认不启用（features 未接线，
属「按需调用」的服务层能力）。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# 允许的转换组合（源扩展 → 目标扩展）
_ALLOWED = {("docx", "pdf"), ("docx", "txt"), ("pdf", "docx"), ("pdf", "txt")}


class ConversionError(Exception):
    """转换失败（引擎缺失 / 格式不支持 / 执行失败）。消息面向用户可读。"""


def detect_engine() -> dict:
    """探测本机可用的转换引擎，返回能力报告（供设置页/文档说明展示）。"""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    report: dict = {
        "libreoffice": bool(soffice),
        "word_com": False,
        "note": "",
    }
    if soffice:
        report["libreoffice_path"] = soffice
    # Word COM：仅 Windows + pywin32（不强制安装，探测导入）
    if __import__("sys").platform == "win32":
        try:
            import win32com.client  # noqa: F401  # type: ignore

            report["word_com"] = True
        except ImportError:
            report["word_com"] = False
            report["note"] += "pywin32 未安装，Word COM 不可用；"
    if not report["libreoffice"] and not report["word_com"]:
        report["note"] = (
            "未检测到转换引擎：请安装 LibreOffice（免费，支持 headless 批量转换）"
            "或安装 pywin32（pip install pywin32，需本机已安装授权 Word）。"
            f"安装后 docx↔pdf 即可用。当前系统：{__import__('sys').platform}"
        )
    return report


def _libreoffice_convert(src: Path, target_ext: str, out_dir: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise ConversionError("未检测到 LibreOffice（soffice），无法转换")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 注意：LibreOffice 不支持 PDF → DOCX 的有损保真转换（文本可导，布局丢失）
    if src.suffix.lower() == ".pdf" and target_ext == "docx":
        raise ConversionError(
            "LibreOffice 对 PDF→Word 保真度有限；建议改用 Word COM（打开 PDF 另存为 docx）或使用 docx 重排工具"
        )
    cmd = [soffice, "--headless", "--convert-to", target_ext, "--outdir", str(out_dir), str(src)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as e:
        # 2026-09-09 修复 M7：soffice 挂起 / 大文件超时会抛 TimeoutExpired——
        # 未捕获会一路冒泡为 HTTP 500；转成用户可读的转换失败提示。
        raise ConversionError(
            "LibreOffice 转换超时（180s）：文件可能过大或 soffice 挂起，请重试或拆分文档"
        ) from e
    if proc.returncode != 0:
        raise ConversionError(f"LibreOffice 转换失败：{proc.stderr[-300:]}")
    out = out_dir / f"{src.stem}.{target_ext}"
    if not out.exists():
        raise ConversionError("LibreOffice 未产出目标文件（可能格式不支持）")
    return out


def convert(src: str | Path, target_ext: str, out_dir: str | Path | None = None) -> Path:
    """把 src 转换为 target_ext（pdf/docx/txt），返回产出文件路径。

    引擎优先级：LibreOffice headless > Word COM（Word COM 未实现时抛出明确说明）。
    """
    src = Path(src)
    ext = src.suffix.lower().lstrip(".")
    target_ext = target_ext.lower().lstrip(".")
    if (ext, target_ext) not in _ALLOWED:
        raise ConversionError(f"不支持的转换：{ext} → {target_ext}（支持：docx/pdf/txt 之间两两转换）")
    if not src.exists():
        raise ConversionError("源文件不存在")
    out_dir = Path(out_dir) if out_dir else src.parent / "converted"
    report = detect_engine()
    if report["libreoffice"]:
        return _libreoffice_convert(src, target_ext, out_dir)
    if report["word_com"]:
        # Word COM 路径：后续补充（win32com 打开/另存为），当前明确未实现
        raise ConversionError("检测到 Word COM，但该转换路径尚未实现；请先安装 LibreOffice 使用，或等待 COM 实现")
    raise ConversionError(report["note"])
