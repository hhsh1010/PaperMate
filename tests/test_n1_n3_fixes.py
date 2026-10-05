"""N1 / N3 回归（2026-09-10，用户实测报告）。

- **N1**：引用来源显示几千条 —— 引用在**跨工具调用**层面未去重、无总数上限
  （`clean_citations` 只做单次内去重）。
- **N3**：改稿模式右侧对话「不是该文档的内容」—— 对话检索用了文献集全部文档，
  命中同批其他论文（实测主文献 LRM–YOLO 安全帽检测，回答却是 SCS-YOLO 内容）。
"""

from __future__ import annotations


# ---------------- N1：引用跨调用去重 + 上限 ----------------
def test_dedup_citations_dedups_across_calls_and_caps():
    """N1 回归：同一片段在多次检索中重复出现时只保留一次；总数受上限约束。"""
    from papermate.services.citation_clean import dedup_citations

    keys: set[str] = set()

    first = dedup_citations(
        [{"id": "c1", "text": "甲" * 40}, {"id": "c2", "text": "乙" * 40}], keys, 10
    )
    assert [c["id"] for c in first] == ["c1", "c2"]

    # 第二次检索：c1 重复（应丢弃）、c3 新（应保留）——这是原先膨胀的主因
    second = dedup_citations(
        [{"id": "c1", "text": "甲" * 40}, {"id": "c3", "text": "丙" * 40}], keys, 10
    )
    assert [c["id"] for c in second] == ["c3"]

    # 上限：超出 limit 直接截断（keys 计数为口径）
    keys2: set[str] = set()
    many = [{"id": f"n{i}", "text": "x" * 30 + f"-{i}"} for i in range(20)]
    capped = dedup_citations(many, keys2, 5)
    assert len(capped) == 5
    assert len(keys2) == 5


def test_dedup_citations_falls_back_to_text_key():
    """N1：无 id 的片段退化为文本前缀去重（仍能拦住重复）。"""
    from papermate.services.citation_clean import dedup_citations

    keys: set[str] = set()
    same = {"text": "同一段没有 id 的正文内容，需要靠文本前缀判重。"}
    assert len(dedup_citations([same], keys, 10)) == 1
    assert dedup_citations([same], keys, 10) == []


def test_dedup_citations_still_filters_noise():
    """N1：抽取过程中不得丢掉既有清洗（噪声关键词 / 过短 / 过长）。"""
    from papermate.services.citation_clean import dedup_citations

    keys: set[str] = set()
    out = dedup_citations(
        [
            {"id": "short", "text": "太短"},
            {"id": "doi", "text": "DOI: 10.1234/abcd 版权声明 联系电话 卷号 期号 " + "字" * 20},
            {"id": "ok", "text": "这是一段正常的正文内容，长度合适且不含噪声关键词。"},
        ],
        keys,
        10,
    )
    assert [c["id"] for c in out] == ["ok"]


# ---------------- N3：有主文献即限缩检索到主文献 ----------------
def test_effective_doc_scope_limits_to_draft_whenever_draft_set():
    """N3（v2）回归：**只要存在主文献就限缩**，判据与前端 UI 一致。

    ⚠️ 实测踩坑（v1 修错的原因）：工作间创建时 `mode` 默认 `"research"`
    （`web/src/api/client.ts: mode: body.mode ?? "research"`），而设置主文献
    （`workspace_service.update` 的 `set_mode` / `set_draft` 相互独立）**不会**把 mode 改成
    `revision`；但前端**只看 `draft_doc_id`** 就渲染改稿工作台。
    若后端按 `mode == "revision"` 判据 → 限缩永不生效 → 回复混入文献集里其他论文
    （用户报「回复中还是出现了未添加文档的内容」）。
    """
    from papermate.api.routes_search import _effective_doc_scope

    coll = [11, 9, 8, 3]  # 用户实测的改稿工作间文献集

    # 有主文献 → 限缩（**不论 mode**：revision / research / 显式 workspace 都限缩）
    assert _effective_doc_scope(coll, {"mode": "revision", "draft_doc_id": 9}, []) == [9]
    assert _effective_doc_scope(coll, {"mode": "research", "draft_doc_id": 9}, []) == [9]
    assert _effective_doc_scope(
        coll, {"workspace": True, "mode": "research", "draft_doc_id": 9}, []
    ) == [9]

    # ★ 2026-09-23 语义变更（用户决策）：`@` 的语义由「追加 / 跨篇逃生通道」改为「**聚焦**」。
    #
    # 原断言（保留于此供对照）：`== [11, 9, 8, 3]`
    #   当时 `@` 只用于**抑制主文献限缩**，不作范围限定；理由是"保留跨篇对比的通道"。
    # 变更原因：用户明确要求「**指定了某篇文档时，只检索对应文档内容并回答**」——
    #   即 `@` 是聚焦语义。同时它修复了另一处不一致：agent 检索范围
    #   （`_scope = mention_ids or ...`）本来就按 mention 限缩，而本函数返回全量 →
    #   技能路径的 `@` 指定从未生效。现在两条链同口径：**mention > 主文献 > 全量**。
    assert _effective_doc_scope(coll, {"mode": "revision", "draft_doc_id": 9}, [11]) == [11]

    # 新增守护①：@ 多篇 → 只保留被 @ 的，且保持 id **降序**（与前端「文档N」同口径）
    assert _effective_doc_scope(coll, {}, [8, 11]) == [11, 8]

    # 新增守护②：@ 了**不在本会话范围内**的 id → 不得静默变空，退回默认范围
    # （否则用户 @ 一个已被移出会话的文档，回答会变成"什么都检索不到"且无法解释）
    assert _effective_doc_scope(coll, {"draft_doc_id": 9}, [999]) == [9]

    # 新增守护③：@ 与主文献同时存在 → **@ 优先**（用户显式意图 > 模式默认）
    assert _effective_doc_scope(coll, {"draft_doc_id": 9}, [11]) == [11]

    # 无主文献 → 全部（id 降序，与前端「文档N」同口径）
    assert _effective_doc_scope(coll, {"mode": "revision"}, []) == [11, 9, 8, 3]
    assert _effective_doc_scope(coll, {"mode": "research"}, []) == [11, 9, 8, 3]

    # 主文献已不在集合内（被删除）→ 全部，绝不产生空范围
    assert _effective_doc_scope(coll, {"mode": "revision", "draft_doc_id": 999}, []) == [11, 9, 8, 3]

    # 合并 config.doc_ids 后仍按同一口径排序
    assert _effective_doc_scope(coll, {"doc_ids": [5]}, []) == [11, 9, 8, 5, 3]


def test_max_citations_constant_is_bounded():
    """N1：上限必须是有限正常量（防止回归成无上限）。"""
    from papermate.api.routes_search import _MAX_CITATIONS

    assert 0 < _MAX_CITATIONS <= 200
