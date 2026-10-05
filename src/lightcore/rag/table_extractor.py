"""LightCore · 表格结构化提取（ER-03 表格优先）

用 pymupdf 内置的 page.find_tables()（1.23+，零新依赖）识别 PDF 中的表格，
转换为 Markdown 表格 + 平铺文本，记录 bbox 坐标（供引用跳转/前端渲染）。

- extract(path)：返回该 PDF 的全部表格（含页码/bbox/markdown/行列数/平铺文本）
- 对无边框文本型表格同样适用（find_tables 默认 auto 策略）
- 纯函数、惰性导入 pymupdf（与 parser.py 一致，基础依赖不强制）

接入：DocumentService 在 PDF 解析阶段调用，表格块作为 chunk_type="table"
的特殊块入库（含 bbox），并把表格文本并入 parsed 文本供向量检索命中。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: bbox 类型（x0, y0, x1, y1）
BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class ExtractedTable:
    page: int
    bbox: BBox
    markdown: str
    rows: int
    cols: int
    text: str  # 平铺文本（供检索）
    header: list[str] = field(default_factory=list)


#: 单页多策略：先用矢量边框（lines），未检出再尝试文本列（text）。
#: 文本列策略能识别无边框、仅靠列对齐的表格（论文/报告极常见），但易把纯段落误检，
#: 故仅当本页 lines 策略未检出时才启用，并配合 2×2 与「非整页」过滤。
_TABLE_STRATEGIES: tuple[dict, ...] = (
    {},  # 默认：horizontal/vertical = "lines"
    {"horizontal_strategy": "text", "vertical_strategy": "text"},
)


def _bbox_overlap_area(a: BBox, b: BBox) -> float:
    """两 bbox 交集面积（用于去重多策略重复检出）。"""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    w, h = ix1 - ix0, iy1 - iy0
    return max(0.0, w) * max(0.0, h)


class TableExtractor:
    def extract(self, path: str | Path) -> list[ExtractedTable]:
        """提取 PDF 中的全部表格。解析异常时返回空列表（容错，不阻断文档处理）。

        支持无边框表格：先以矢量边框策略查找，本页未命中时再用文本列对齐策略兜底，
        合并后按 bbox 去重，并过滤掉过小（<2×2）或几乎占满整页（疑似段落/双栏版式）的误检。
        """
        import pymupdf  # 惰性：仅 PDF 场景才需要

        tables: list[ExtractedTable] = []
        try:
            doc = pymupdf.open(str(path))
        except Exception:  # noqa: BLE001  打不开（损坏 PDF）→ 无表格
            return tables
        with doc:
            for page_no, page in enumerate(doc, start=1):
                try:
                    found = self._extract_page(page, page_no)
                except Exception:  # noqa: BLE001  单页解析失败不影响其他页
                    continue
                tables.extend(found)
        return tables

    def _extract_page(self, page, page_no: int) -> list[ExtractedTable]:
        pw, ph = page.rect.width, page.rect.height
        page_area = float(pw) * float(ph) or 1.0
        out: list[ExtractedTable] = []
        seen: list[BBox] = []
        lines_found = False
        for kw in _TABLE_STRATEGIES:
            try:
                finder = page.find_tables(**kw)
            except Exception:  # noqa: BLE001  该策略不支持/失败 → 试下一个
                continue
            is_text = bool(kw)  # 文本列策略：结果需过更严的网格过滤（防段落误检）
            for t in finder.tables:
                data = _clean_cells(t.extract())
                if not data:
                    continue
                nrows, ncols = len(data), max(len(r) for r in data)
                if nrows < 2 or ncols < 2:  # 至少 2×2，排除单列/单行误检
                    continue
                if is_text and not (3 <= nrows <= 30 and 2 <= ncols <= 8):
                    # text 策略易把纯段落/列表误检成「大表」：真实表格行数通常有界、
                    # 列数很少超过 8（段落会把每行单词拆成十几列）。
                    continue
                bbox = (round(t.bbox[0], 1), round(t.bbox[1], 1), round(t.bbox[2], 1), round(t.bbox[3], 1))
                # 多策略可能重复检出同一表格 → 交集面积占比 > 50% 视为重复
                if any(_bbox_overlap_area(bbox, s) / max(1e-6, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) > 0.5 for s in seen):
                    continue
                # 文本列策略易把整页双栏段落误检为「大表」→ 占比 > 95% 整页则跳过
                if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / page_area > 0.95:
                    continue
                seen.append(bbox)
                if not kw:  # 默认 lines 策略命中
                    lines_found = True
                md = _table_to_markdown(data)
                if not md:
                    continue
                out.append(
                    ExtractedTable(
                        page=page_no,
                        bbox=bbox,
                        markdown=md,
                        rows=nrows,
                        cols=ncols,
                        text=_table_to_text(data),
                        header=list(data[0]) if data else [],
                    )
                )
            # lines 策略已在本页检出表格 → 不再跑 text，避免段落误检放大
            if lines_found:
                break
        return out


def _clean_cells(data: list[list[str | None]]) -> list[list[str]]:
    """单元格去 None/空白，剔除全空行。"""
    rows: list[list[str]] = []
    for row in data:
        cells = [(c or "").strip() for c in row]
        if any(cells):  # 至少一个非空单元格才算有效行
            rows.append(cells)
    return rows


def _table_to_markdown(data: list[list[str]]) -> str:
    """单元格矩阵 → Markdown 表格（首行作表头）。"""
    if not data:
        return ""
    ncols = max(len(r) for r in data)
    if ncols == 0:
        return ""
    header = data[0] + [""] * (ncols - len(data[0]))
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    for row in data[1:]:
        cells = row + [""] * (ncols - len(row))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_to_text(data: list[list[str]]) -> str:
    """单元格矩阵 → 平铺文本（供检索）。"""
    return " ".join(cell for row in data for cell in row if cell)


def _split_md_row(line: str) -> list[str]:
    """切分一行 Markdown 表格单元格（去首尾 |，按 | 拆分并 strip）。"""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_md_separator(line: str) -> bool:
    """是否表头分隔行（仅含 | - : 空白）。"""
    import re

    return bool(re.fullmatch(r"\s*\|[\s:|-]+\|\s*", line))


def md_table_lint(markdown: str) -> list[str]:
    """校验 Markdown 表格结构（P0-2.2）。

    检查项：表头存在且非空、第二行为分隔行、各数据行列数与表头一致。
    返回问题列表；空列表 = 结构合法。供上传管线把异常表格告警并入
    extraction_warnings（提示性，不阻断处理）。
    """
    issues: list[str] = []
    lines = [ln for ln in markdown.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        issues.append("表格缺少表头或数据行")
        return issues
    header = _split_md_row(lines[0])
    if not any(c for c in header):
        issues.append("表头为空")
    if not _is_md_separator(lines[1]):
        issues.append("缺少表头分隔行（|---|）")
    ncols = len(header)
    for idx, ln in enumerate(lines[2:], start=3):
        cells = _split_md_row(ln)
        if len(cells) != ncols:
            issues.append(f"第 {idx} 行列数不一致（{len(cells)} ≠ 表头 {ncols}）")
    return issues
