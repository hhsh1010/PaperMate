"""LightCore · 页码归一化与边界校验（RAG / 引用溯源）

统一的页码语义：
- **有效物理页码**：≥1 的正整数（PDF 第 1 页起算）。
- **无物理页码**：None（docx / md / txt 等非分页文档，以及解析失败回退）。

历史 bug（2026-08-24）：管线由「去页码的纯文本」重建，导致所有检索块 page=0，
LLM 据此引用「第 0 页」。本模块把"0/负数/None/非法 → None"的归一则集中到一处，
任何向 LLM / 前端输出页码的地方都必须经过 normalize_page，杜绝 0 当成真实页码。
"""

from __future__ import annotations


def is_valid_page(page: object) -> bool:
    """page 是否为有效物理页码（≥1 的整数，排除 bool）。"""
    return isinstance(page, int) and not isinstance(page, bool) and page >= 1


def normalize_page(page: object) -> int | None:
    """页码归一化：仅 ≥1 的整数视为有效页码，其余（0/负数/None/浮点/bool/非法）→ None。

    用于 retrieve / SSE citations / pdf_extract / verifier 等所有向外输出页码的环节，
    保证 0 永远不会被当作真实页码喂给 LLM 或展示在引用卡上。
    """
    return page if is_valid_page(page) else None


def validate_requested_page(page: object, available_pages: set[int] | None = None) -> tuple[int | None, str | None]:
    """校验「用户/工具请求的页码」是否合法（边界校验）。

    返回 (normalized_page, error_note)：
    - normalized_page 为 None 且 error_note 非空 → 非法（page<1 或非整数）；
    - normalized_page 有效但不在 available_pages 中 → 仍返回该页码，但附带提示 note
      （调用方可据此告知"文档无第 X 页"而非静默返回空）。
    """
    norm = normalize_page(page)
    if norm is None:
        return None, "页码无效：页码应从 1 开始（正整数）。"
    if available_pages is not None and norm not in available_pages:
        lo = min(available_pages)
        hi = max(available_pages)
        return norm, f"文档无第 {norm} 页（可用页码：{lo}–{hi}）。"
    return norm, None
