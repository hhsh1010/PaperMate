"""LightCore · Word(.docx) 内嵌图片提取（ER-03 图表部分，docx 版）

python-docx 的 ``Document.inline_shapes`` 提取内联图片字节；产出的 ``ExtractedFigure``
与 PDF 版 ``FigureExtractor`` 同构，故 ``document_service`` 可复用同一套 Level 2 增强
（OCR / 表格结构还原 / VLM 图注）与图块入库逻辑，前端缩略图/灯箱也无需改动。

- extract(path, image_dir=None)：返回 (figures, warnings)
  - figures：``ExtractedFigure``（page=None 表示 docx 无物理页码 / bbox=(0,0,w,h) 单位 EMU / xref=0 / caption / filename / kind）
    kind = "figure"（一般图表）| "image_table"（宽幅 → 可能为截图型表格/宽图表）
  - 图片按 blob 内容哈希命名，同一图多处引用只落一份文件
- 容错：python-docx 缺失 / 解析失败 / 单图异常 → 返回空或跳过，不阻断文档处理
- 局限：仅提取 inline（内联）图片；浮动（wrap）图片未计入（论文图表多为内联，覆盖主场景）
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from docx.oxml.ns import qn

from lightcore.rag.figure_extractor import ExtractedFigure, _classify_kind

#: 图注行开头模式（与 FigureExtractor 一致，便于后续按相邻段落补 caption 时复用）
_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?\.?\s*\d+|table\s*\d+|图\s*\d+|表\s*\d+)\b", re.IGNORECASE
)


class DocxImageExtractor:
    def extract(self, path, image_dir=None) -> tuple[list[ExtractedFigure], list[str]]:
        """提取 .docx 中的内联图片。返回 (figures, warnings)。"""
        try:
            from docx import Document  # 惰性：仅 .docx 场景才需要
        except Exception:  # noqa: BLE001  依赖缺失 → 优雅降级
            return [], ["python-docx 未安装，docx 图片提取跳过"]

        figures: list[ExtractedFigure] = []
        warnings: list[str] = []
        try:
            doc = Document(str(path))
        except Exception as e:  # noqa: BLE001  损坏/非 Word → 无图片，不阻断
            return [], [f"docx 无法解析，图片提取跳过：{type(e).__name__}"]

        out_dir = Path(image_dir) if image_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        for shape in doc.inline_shapes:
            try:
                width = float(getattr(shape.width, "emu", 0) or 0)
                height = float(getattr(shape.height, "emu", 0) or 0)
                blob, ext = _shape_blob_and_ext(shape, doc)
                if not blob:
                    continue
                ext = (ext or "png").lower() or "png"
                digest = hashlib.sha1(blob).hexdigest()[:12]
                fname = f"docx-{digest}.{ext}"
                caption = (getattr(shape, "title", "") or "").strip()
                kind = _classify_kind(caption, width, height)
                if kind == "image_table":
                    warnings.append(
                        "检测到宽幅图片（可能为截图型表格/图表）：图片内容无法直接检索文字，"
                        "需 OCR 或人工查看；图片已作为资源保存。"
                    )
                if out_dir is not None:
                    (out_dir / fname).write_bytes(blob)
                figures.append(
                    ExtractedFigure(
                        page=None,  # docx 无物理页码概念
                        bbox=(0.0, 0.0, width, height),
                        xref=0,
                        caption=caption,
                        filename=fname,
                        kind=kind,
                    )
                )
            except Exception:  # noqa: BLE001  单图异常不影响其余
                continue
        return figures, warnings


def _shape_blob_and_ext(shape, doc) -> tuple[bytes | None, str]:
    """跨 python-docx 版本取内联图片字节与扩展名。

    - 旧版：``shape.image`` / ``shape.picture`` 直接暴露 .blob
    - 新版：InlineShape 不再暴露 image/picture，需从 ``a:blip`` 的 r:embed 关系解析
    """
    pic = getattr(shape, "picture", None) or getattr(shape, "image", None)
    if pic is not None:
        return pic.blob, (getattr(pic, "ext", None) or "png")
    blips = shape._inline.findall(".//" + qn("a:blip"))
    if blips:
        rid = blips[0].get(qn("r:embed"))
        part = doc.part.related_parts.get(rid)
        if part is not None:
            ext = (getattr(part, "ext", None) or "") or ""
            return part.blob, ext
    return None, "png"


__all__ = ["DocxImageExtractor"]
