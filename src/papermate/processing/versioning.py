"""PaperMate · 文档处理子包 · 差异对比与版本管理（versioning）

- content_fingerprint：归一化内容指纹（去空白/大小写），供近似去重
- content_hash：原文 sha256（精确去重，与 DocumentService 口径一致）
- similarity / diff_texts：difflib 相似度 + unified diff
- DocumentVersioner：版本注册/历史/对比
  版本信息存储在 documents.meta_json["versions"]（零迁移），每条含
  version / content_hash / fingerprint / summary / created_at。

设计约定：versioner 是纯逻辑（仅操作传入的 list[VersionInfo]），
与 DB 解耦；持久化由上层 service 完成。
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections.abc import Iterable

from .models import DiffResult, VersionInfo

_WS_RE = re.compile(r"\s+")


def content_fingerprint(text: str) -> str:
    """归一化指纹：去空白 + 小写后 sha256 前 16 位（近似去重用）。"""
    norm = _WS_RE.sub("", text or "").lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def content_hash(content: bytes | str) -> str:
    """原文 sha256（精确去重用，与 DocumentService._content_hash 一致）。"""
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def similarity(left: str, right: str) -> float:
    """文本相似度 0..1（difflib SequenceMatcher）。"""
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _change_counts(left: str, right: str) -> tuple[int, int]:
    """行级增删统计：返回 (added, removed)。"""
    l_lines = left.splitlines()
    r_lines = right.splitlines()
    added = removed = 0
    for tag, _i1, _i2, _j1, _j2 in difflib.SequenceMatcher(None, l_lines, r_lines).get_opcodes():
        if tag == "delete":
            removed += 1
        elif tag == "insert":
            added += 1
        elif tag == "replace":
            removed += 1
            added += 1
    return added, removed


def diff_texts(
    left: str,
    right: str,
    left_title: str = "old",
    right_title: str = "new",
    max_lines: int = 200,
) -> DiffResult:
    """两段文本的差异对比：相似度 + 增删行统计 + unified diff（截断）。"""
    added, removed = _change_counts(left, right)
    total = max(len(left.splitlines()) + len(right.splitlines()), 1)
    unified = "\n".join(
        difflib.unified_diff(
            left.splitlines(),
            right.splitlines(),
            fromfile=left_title,
            tofile=right_title,
            lineterm="",
        )
    )
    if len(unified.splitlines()) > max_lines:
        unified = "\n".join(unified.splitlines()[:max_lines]) + "\n... (truncated)"
    return DiffResult(
        left_title=left_title,
        right_title=right_title,
        similarity=round(similarity(left, right), 4),
        added_lines=added,
        removed_lines=removed,
        changed_ratio=round((added + removed) / total, 4),
        unified=unified,
        summary=_summarize_change(added, removed, similarity(left, right)),
    )


def _summarize_change(added: int, removed: int, sim: float) -> str:
    if sim >= 0.999:
        return "内容基本相同（仅格式差异）"
    if sim >= 0.85:
        return f"小规模修订：新增 {added} 行、删除 {removed} 行"
    if sim >= 0.5:
        return f"中规模修订：新增 {added} 行、删除 {removed} 行（相似度 {sim:.0%}）"
    return f"大幅改写：新增 {added} 行、删除 {removed} 行（相似度 {sim:.0%}）"


class DocumentVersioner:
    """版本管理：注册新版本、列历史、与上一版本对比。纯逻辑，版本列表由调用方提供。"""

    @staticmethod
    def register(
        versions: list[VersionInfo],
        text: str,
        raw_content: bytes | str,
        prev_text: str | None = None,
    ) -> VersionInfo:
        """注册新版本（自动递增版本号）；与上一版本同指纹时跳过（返回既有末版）。

        prev_text 提供时，变更摘要基于与上一版本的逐行对比生成。
        """
        fingerprint = content_fingerprint(text)
        if versions and versions[-1].fingerprint == fingerprint:
            return versions[-1]
        prev = versions[-1] if versions else None
        if prev is not None and prev_text is not None and prev_text != text:
            added, removed = _change_counts(prev_text, text)
            summary = _summarize_change(added, removed, similarity(prev_text, text))
        elif prev is not None:
            # 2026-09-08 修复 H4：prev_text 缺失、或与当前文本**相同**（说明调用方拿到的是
            # 已被新内容覆盖的文本）时，逐行 diff 无意义。此前会用 prev_text == text 算出
            # similarity=1.0，恒输出「内容基本相同（仅格式差异）」，把真实的内容变更完全掩盖。
            # 指纹已在上方比对过（相同则直接返回末版），故走到这里可确定内容确有变化。
            summary = "内容已更新（与上一版本指纹不同；逐行差异基线不可用）"
        else:
            summary = "初始版本"
        version = VersionInfo(
            version=(prev.version + 1) if prev else 1,
            content_hash=content_hash(raw_content),
            fingerprint=fingerprint,
            summary=summary,
        )
        versions.append(version)
        return version

    @staticmethod
    def list_versions(versions: list[VersionInfo]) -> list[VersionInfo]:
        return sorted(versions, key=lambda v: v.version)

    @staticmethod
    def compare_versions(
        versions: list[VersionInfo],
        left_version: int,
        right_version: int,
        texts: dict[int, str],
    ) -> DiffResult:
        """按版本号对比两个历史版本（texts 为版本号 → 原文）。"""
        if left_version not in texts or right_version not in texts:
            raise KeyError("版本原文缺失")
        return diff_texts(
            texts[left_version],
            texts[right_version],
            left_title=f"v{left_version}",
            right_title=f"v{right_version}",
        )


__all__: Iterable[str] = (
    "content_fingerprint",
    "content_hash",
    "similarity",
    "diff_texts",
    "DocumentVersioner",
)
