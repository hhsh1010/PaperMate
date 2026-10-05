"""LightCore · 矢量结构图提取（ER-03 扩展：矢量型结构图）

PDF 里的架构图 / 流程图 / 框图大多是「矢量绘图」——由线条、矩形、曲线与文字标签
构成；``page.get_image_info()`` 只能拿到嵌入位图，拿不到这些矢量路径。本模块用
``page.get_drawings()`` 提取矢量路径，按 bbox 邻近聚合成「结构图块」，抽取簇内
文字标签作为可检索文本，并把该区域栅格化导出为 PNG（供前端预览 / 灯箱 / 保存）。

- extract(path, image_dir=None)：返回 list[ExtractedDiagram]（page/bbox/labels/text/filename）
- 纯函数、惰性导入 pymupdf（与 parser / table_extractor 一致）；单页异常容错不阻断
- 过滤：过小路径（装饰/图标/分隔线）、无标签且路径稀少的形状、整页背景边框
- 去重：与位图 figure 的 bbox 交叠合并由 document_service 处理（见其 _store_chunks 前置过滤）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: bbox 类型（x0, y0, x1, y1）
BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class ExtractedDiagram:
    page: int  # 1 起算页码
    bbox: BBox
    labels: list[str]  # 簇内文字标签（按阅读顺序）
    text: str  # 平铺标签文本（供向量检索）
    filename: str  # 栅格化图片文件名


class DiagramExtractor:
    def __init__(self, gap: float = 15.0, min_w: float = 80.0, min_h: float = 40.0,
                 max_cover: float = 0.95, dpi: int = 144) -> None:
        #: 矢量路径聚类邻近阈值（pt）
        self._gap = gap
        #: 结构图块最小宽/高（pt）：更小视为装饰/图标
        self._min_w = min_w
        self._min_h = min_h
        #: 块面积占页面比例超过该值视为整页背景/边框，跳过
        self._max_cover = max_cover
        #: 栅格化分辨率（72 基准，144=2x 保证清晰）
        self._dpi = dpi

    def extract(self, path: str | Path, image_dir: str | Path | None = None) -> list[ExtractedDiagram]:
        """提取 PDF 中的全部矢量结构图。解析异常返回空列表（容错，不阻断文档处理）。"""
        import pymupdf  # 惰性：仅 PDF 场景才需要

        diagrams: list[ExtractedDiagram] = []
        try:
            doc = pymupdf.open(str(path))
        except Exception:  # noqa: BLE001  打不开（损坏 PDF）→ 无结构图
            return diagrams
        with doc:
            for page_no, page in enumerate(doc, start=1):
                try:
                    diagrams.extend(self._extract_page(page, page_no, image_dir))
                except Exception:  # noqa: BLE001  单页失败不影响其他页
                    continue
        return diagrams

    def _extract_page(self, page, page_no: int, image_dir) -> list[ExtractedDiagram]:
        rects = [d.get("rect") for d in page.get_drawings()]
        if not rects:
            return []
        pw, ph = page.rect.width, page.rect.height
        page_area = float(pw) * float(ph) or 1.0
        out: list[ExtractedDiagram] = []
        for idx, cluster in enumerate(_cluster_rects(rects, self._gap)):
            x0, y0, x1, y1 = _union_bbox(cluster)
            w, h = x1 - x0, y1 - y0
            if w < self._min_w or h < self._min_h:
                continue  # 过小：图标/装饰/分隔线
            if w * h / page_area > self._max_cover:
                continue  # 整页背景/边框装饰
            labels = _labels_in_rect(page, x0, y0, x1, y1)
            if len(cluster) < 3 and not labels:
                continue  # 无标签且路径稀少：形状装饰
            if image_dir is not None:
                fname = self._save_image(page, image_dir, page_no, idx, (x0, y0, x1, y1))
            else:
                fname = f"p{page_no}-d{idx}.png"
            text = " ".join(labels)
            out.append(ExtractedDiagram(
                page=page_no,
                bbox=(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
                labels=labels,
                text=text,
                filename=fname,
            ))
        return out

    def _save_image(self, page, image_dir, page_no: int, idx: int, clip: BBox) -> str:
        """把结构图区域栅格化为 PNG 落盘；失败返回占位文件名（有块记录即可）。"""
        import pymupdf  # 惰性

        fname = f"p{page_no}-d{idx}.png"
        try:
            rect = pymupdf.Rect(clip[0], clip[1], clip[2], clip[3])
            pix = page.get_pixmap(matrix=pymupdf.Matrix(self._dpi / 72, self._dpi / 72), clip=rect)
            dir_path = Path(image_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / fname).write_bytes(pix.tobytes("png"))
        except Exception:  # noqa: BLE001  单块渲染失败不影响整体
            pass
        return fname


def _union_bbox(rects: list[tuple[float, float, float, float]]) -> BBox:
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _near(a: tuple[float, float, float, float], b: tuple[float, float, float, float], gap: float) -> bool:
    """两路径是否属于同一结构图：扩张 gap/2 后相交，或中心距离 < gap。"""
    h = gap / 2.0
    if a[0] - h <= b[2] + h and b[0] - h <= a[2] + h and a[1] - h <= b[3] + h and b[1] - h <= a[3] + h:
        return True
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 < gap


def _cluster_rects(rects: list[tuple[float, float, float, float]], gap: float) -> list[list[tuple[float, float, float, float]]]:
    """按邻近关系把矢量路径聚成簇（union-find，O(n²)，页内路径通常 ≤ 数百）。"""
    n = len(rects)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _near(rects[i], rects[j], gap):
                union(i, j)
    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(rects[i])
    return list(groups.values())


def _labels_in_rect(page, x0: float, y0: float, x1: float, y1: float, margin: float = 8.0) -> list[str]:
    """簇 bbox（扩张 margin）内的文字标签，按行（阅读顺序：上→下、左→右）返回。

    用 dict 级提取取「整行」文本（如 "Node A"），比 words 级（拆成 Node/A）更适合做
    结构图标签与检索文本。
    """
    data = page.get_text("dict")
    lines: list[tuple[float, float, str]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue  # 非文本块（图片等）
        for line in block.get("lines", []):
            bx0, by0, bx1, by1 = line["bbox"]
            if bx0 >= x0 - margin and by0 >= y0 - margin and bx1 <= x1 + margin and by1 <= y1 + margin:
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if text:
                    lines.append((by0, bx0, text))
    lines.sort(key=lambda t: (round(t[0], 1), t[1]))
    return [t[2] for t in lines]


__all__ = ["ExtractedDiagram", "DiagramExtractor"]
