"""LightCore · 图片/图表提取（ER-03 图表部分）

用 pymupdf 的 page.get_image_info() 提取 PDF 中嵌入的图片（含 bbox 与尺寸），
并基于页内文本启发式猜测图注（caption）。这是「图片型表格/图表/扫描件」
无法被 find_tables() 识别时的兜底能力——至少把图片作为结构化块入库、
可定位、可被检索到图注，而不是静默丢弃。

- extract(path, image_dir=None)：返回 (figures, warnings)
  - figures：ExtractedFigure（page/bbox/xref/caption/导出文件名/kind）
    kind = "figure"（一般图表）| "image_table"（宽幅或图注含 Table/表 → 提示可能为表格图片）
  - warnings：人类可读的提取提示（图片型内容无法直接检索文字 → 建议 OCR 等）
- image_dir 给定则用 doc.extract_image(xref) 导出图片文件（零新依赖）
- 容错：损坏 PDF / 单页异常不阻断，返回已提取部分 + warning
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: bbox 类型（x0, y0, x1, y1）
BBox = tuple[float, float, float, float]

#: 图注行开头模式（Figure 1 / Fig. 2 / 图 3 / Table 1 / 表 2）
_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?\.?\s*\d+|table\s*\d+|图\s*\d+|表\s*\d+)\b", re.IGNORECASE
)
#: 最小图片宽度（pt）：小于此值视为图标/装饰，跳过
_MIN_WIDTH = 80.0
#: 宽幅判定（宽/高 比值）：宽幅图片更可能是表格/宽图表
_WIDE_RATIO = 1.6


@dataclass(slots=True)
class ExtractedFigure:
    page: int | None  # None=无物理页码（docx）；PDF 为 1 起算
    bbox: BBox
    xref: int
    caption: str
    filename: str
    kind: str  # figure | image_table


class FigureExtractor:
    def extract(self, path: str | Path, image_dir: str | Path | None = None) -> tuple[list[ExtractedFigure], list[str]]:
        """提取 PDF 中的嵌入图片。返回 (figures, warnings)。"""
        import pymupdf  # 惰性：仅 PDF 场景才需要

        figures: list[ExtractedFigure] = []
        warnings: list[str] = []
        try:
            doc = pymupdf.open(str(path))
        except Exception:  # noqa: BLE001  打不开（损坏 PDF）→ 无图片
            return figures, ["PDF 无法打开，图片提取跳过"]
        with doc:
            for page_no, page in enumerate(doc, start=1):
                try:
                    texts = [t.strip() for t in page.get_text("text").splitlines() if t.strip()]
                    # xrefs=True：get_image_info 默认不返回 xref，无 xref 则 extract_image 无法导出（实测踩坑）
                    infos = page.get_image_info(xrefs=True)
                except Exception:  # noqa: BLE001  单页异常不影响其他页
                    continue
                page_area = float(page.rect.width) * float(page.rect.height) or 1.0
                # M5 修复（2026-09-10）：先收集本页**入选**图片再逐个处理——
                # ①同页多图原先各自调用 `_guess_caption(texts)`，全部拿到同一条页级图注
                #   （图注错配污染检索）→ 仅当页内恰好一张入选图时才使用页级图注；
                # ②无 xref 的图原先统一命名 `p{n}-x0.{ext}`，同页多张互相覆盖 →
                #   改用页内序号 `p{n}-i{idx}.{ext}`（idx 为入选顺序）。
                picked: list[tuple[dict, tuple, float, float]] = []
                for info in infos:
                    bbox = (round(info["bbox"][0], 1), round(info["bbox"][1], 1), round(info["bbox"][2], 1), round(info["bbox"][3], 1))
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    if width < _MIN_WIDTH or height <= 0:
                        continue  # 图标/装饰小图
                    picked.append((info, bbox, width, height))
                single_figure = len(picked) == 1
                for idx, (info, bbox, width, height) in enumerate(picked):
                    xref = int(info.get("xref") or 0)
                    ext = (info.get("ext") or "png").lower() or "png"
                    fname = f"p{page_no}-x{xref}.{ext}" if xref else f"p{page_no}-i{idx}.{ext}"
                    caption = _guess_caption(texts) if single_figure else ""
                    kind = _classify_kind(caption, width, height)
                    # 整页大图（≥60% 页面面积）→ 疑似扫描件/图片型 PDF：图中表格/文字
                    # 无法直接检索，需 OCR；标记为 image_table 并提示，避免用户误以为丢失
                    cover_ratio = (width * height) / page_area
                    if cover_ratio >= 0.6:
                        kind = "image_table"
                        warnings.append(
                            f"第 {page_no} 页存在疑似整页图片（覆盖页面 {cover_ratio:.0%}，扫描件/图片型 PDF）："
                            "图中表格/文字无法直接检索，需在设置页开启「图片增强(OCR)」后重新上传；"
                            "图片已作为资源保存。"
                        )
                    elif kind == "image_table":
                        warnings.append(
                            f"第 {page_no} 页检测到宽幅图片（可能为截图型表格/图表，bbox={bbox}）："
                            "图片内容无法直接检索文字，需 OCR 或人工查看；图片已作为资源保存。"
                        )
                    if image_dir is not None:
                        _save_image(doc, xref, Path(image_dir), fname)
                    figures.append(
                        ExtractedFigure(page=page_no, bbox=bbox, xref=xref, caption=caption, filename=fname, kind=kind)
                    )
        return figures, warnings


def _guess_caption(text_lines: list[str], start: int = 0, limit: int = 30) -> str:
    """启发式图注：从页内文本里找 'Figure N / 图 N / Table N / 表 N' 开头的行。"""
    for line in text_lines[start: start + limit]:
        if _CAPTION_RE.match(line):
            return line[:120]
    return ""


def _classify_kind(caption: str, width: float, height: float) -> str:
    """figure 与 image_table（截图型表格）的启发式区分。"""
    low = caption.lower()
    if "table" in low or "表" in caption:
        return "image_table"
    if height > 0 and width / height >= _WIDE_RATIO:
        return "image_table"  # 宽幅图片更像表格
    return "figure"


def _save_image(doc, xref: int, image_dir: Path, filename: str) -> None:
    """用 xref 从 PDF 提取原始图片字节并落盘；失败静默（有 figure 记录即可）。"""
    if not xref:
        return
    try:
        info = doc.extract_image(xref)
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / filename).write_bytes(info["image"])
    except Exception:  # noqa: BLE001  单图导出失败不影响整体
        return


__all__ = ["ExtractedFigure", "FigureExtractor"]
