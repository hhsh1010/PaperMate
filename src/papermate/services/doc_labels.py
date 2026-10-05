"""文档标签的**单一来源**（2026-10-02，M3）。

## 为什么要有这个模块

「文档一/ 二/ 三…」这套中文序号此前散落**四处**各自实现：

| 位置 | 实现 | 是否与前端一致 |
|---|---|---|
| 前端 `web/src/lib/docLabel.ts` | 按**数组下标** `cn[index]` | 基准 |
| 技能 `skills/_common.mention_order` | `sorted(ids, reverse=True)`（id 降序） | ✅ |
| 技能 `planning_skill._doc_label` | **又抄一份** `cn` 表 | ⚠️ 重复 |
| 服务 `summary_service._doc_label` | **又抄一份** `cn` 表 | ⚠️ 重复 |

★ 真正的坑不是"重复"，而是**两套口径并存**：
- 标签 = **id 降序**（前端 `GET /documents` 是 `order_by(id.desc())`，见
  `document_service.list_documents`）；
- 而 `session.config.doc_ids` 的**关联顺序**是用户勾选的顺序，**与标签顺序无关**。

实测反例（session 18，`doc_ids = [14, 6, 16, 19]`）：

| 标签 | id | `doc_ids[0]`（关联顺序首个） |
|---|---|---|
| 文档一 | **19** | **14** ← 完全不同 |

⇒ `skills/revision.py` / `deep_read.py` 里的 `doc_ids[0]`（"未指定就取第一篇"）
在多文档会话里会取到**用户标签上叫「文档三」**的那篇。

## 本模块的契约

- `doc_label(index)`：下标 → 标签（纯函数，**不依赖顺序口径**）；
- `mention_order(doc_ids)`：id 集合 → **标签顺序**（id 降序，唯一口径）；
- `label_of(doc_id, doc_ids)`：某id 在当前集合里的标签（找不到 ⇒ `None`，**不静默回退**）。

⚠️ **禁止**再写 `doc_ids[0]` 当"第一篇"——要"第一篇"就用 `mention_order(...)[0]`。
"""

from __future__ import annotations

#: 中文数字表（1~10）。超过 10 走 `文档 11` 形式（与前端 `docLabel` 逐字一致）。
_CN = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")


def doc_label(index: int) -> str:
    """下标（0 基）→「文档一/二/…」。

    ⚠️ 传入的必须是 **`mention_order` 里的下标**；直接拿 `doc_ids` 的原始位置
    会与前端标签错位（见模块 docstring 的反例）。
    """
    if 0 <= index < len(_CN):
        return f"文档{_CN[index]}"
    return f"文档 {index + 1}"


def mention_order(doc_ids: list[int]) -> list[int]:
    """「文档N」的**唯一排序口径**：**id 降序**。

    与前端一致：`GET /documents` 是 `order_by(Document.id.desc())`，
    前端 `activeDocs = allDocs.filter(...)` 继承该顺序，再按**下标**调`docLabel`。
    """
    return sorted(doc_ids, reverse=True)


def label_of(doc_id: int, doc_ids: list[int]) -> str | None:
    """某文档在当前集合中的标签；**不在集合内返回 `None`**。

    ⚠️ 刻意**不**回退到"文档一"：静默回退会让**标签指错篇目**（用户以为在讲 A 篇，
    实际讲的是 B 篇）——那是**静默降准**，比报错更难发现。
    """
    ordered = mention_order(list(doc_ids))
    try:
        return doc_label(ordered.index(doc_id))
    except ValueError:
        return None
