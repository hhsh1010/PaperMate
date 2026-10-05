"""M1 + M3（2026-10-02）：手打「@文档N」在普通对话路径失效 + 标签顺序错位。

## 事故（实测，见 docs/诊断-离题回答与熔断预警-2026-10-02.md §11）

用户手打 `@文档二 总结一下论文的实验步骤` ⇒ 模型答
「**未能取到你所指「文档二」的实验步骤**」。

**根因**：`@文档N → doc_id` 的解析**只存在于技能层**（`skills/_common.resolve_mention`，
调用方仅 `deep_read` / `planning_skill`），**普通对话的 `retrieve` 路径完全没接** ——
`routes_search` 只过滤**结构化** `mention_doc_ids` ⇒ 手打文字 ⇒ 范围不限定 ⇒ 全库盲搜。
铁证：事故时段 10 次 `retrieve` 的 args **0 次**带 `doc_ids`。

**M1** = `routes_search` 复用解析函数，结构化字段为空时回退解析正文。
**M3** = 标签↔doc_id 映射收成单一来源（`services/doc_labels`），
并修掉「按 `doc_ids[0]` 取第一篇 = 关联顺序首个」与「标签 = id 降序」两套口径并存。
"""
from __future__ import annotations

import pytest

from papermate.services.doc_labels import doc_label, label_of, mention_order
from papermate.skills._common import resolve_mention, resolve_mention_in_scope


# ============================================================
# 单元层：标签的单一来源（M3）
# ============================================================
class TestDocLabels:
    def test_label_matches_frontend_doclabel_verbatim(self):
        """★ 必须与前端 `web/src/lib/docLabel.ts` **逐字一致**，否则用户看到的
        「文档三」和后端理解的「文档三」不是同一篇（静默降准）。"""
        cn = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        for i, ch in enumerate(cn):
            assert doc_label(i) == f"文档{ch}"
        assert doc_label(10) == "文档 11"   # >10 的形式也必须一致
        assert doc_label(-1) == "文档 0"    # 越界不抛异常

    def test_mention_order_is_id_descending(self):
        """★ 标签顺序 = **id 降序**（与前端 `GET /documents` 的 `order_by(id.desc())` 同源）。"""
        assert mention_order([14, 6, 16, 19]) == [19, 16, 14, 6]

    def test_label_of_maps_real_session18(self, ):
        """★ 事故会话的真实映射（`config.doc_ids = [14, 6, 16, 19]`）。

        ⚠️ **关联顺序的第 1 个是 14，而「文档一」是 19** ⇒
        任何 `doc_ids[0] == 文档一` 的实现都是错的。
        """
        ids = [14, 6, 16, 19]
        assert label_of(19, ids) == "文档一"
        assert label_of(16, ids) == "文档二"
        assert label_of(14, ids) == "文档三"
        assert label_of(6, ids) == "文档四"
        # ★ 反例钉死：关联顺序首个 ≠ 文档一
        assert ids[0] == 14 and label_of(ids[0], ids) == "文档三"

    def test_label_of_returns_none_when_absent(self):
        """★ 不在集合内 ⇒ `None`，**不静默回退「文档一」**（回退会让标签指错篇目）。"""
        assert label_of(999, [14, 6]) is None


# ============================================================
# 单元层：回退解析（M1 的核心）
# ============================================================
class TestResolveMentionInScope:
    def test_handwritten_mention_resolves_to_doc_id(self):
        """★★ 事故复现：手打「@文档二」必须能解析出doc_id。"""
        ids = [14, 6, 16, 19]
        got, rest, invalid = resolve_mention_in_scope("@文档二 总结一下论文的实验步骤", ids)
        assert got == [16], f"「文档二」应解析为 id 16（id 降序第二位），实际 {got}"
        assert rest == "总结一下论文的实验步骤", "必须剥离提及，否则模型会拿「@文档二」当检索词"
        assert invalid is None

    def test_structured_field_priority_is_in_caller(self):
        """★ 优先级「结构化 > 文本回退」由**调用方**保证
        （`routes_search`：`if not mention_ids:` 才回退）⇒ 这里只验证两者一致时不冲突。"""
        ids = [14, 6, 16, 19]
        got, _, _ = resolve_mention_in_scope("@文档三 提问", ids)
        assert got == [14]

    def test_out_of_range_returns_empty_not_guess(self):
        """★★ 越界**不得返回任何 id**（关联 2 篇却写「文档五」）。

        返回空 + 越界标签，让调用方**据实提示**；
        静默猜一篇 ⇒ 用户以为限定生效了，实际在查错的论文。
        """
        got, rest, invalid = resolve_mention_in_scope("@文档五 提问", [14, 6])
        assert got == [], "越界必须返回空（不得猜）"
        assert invalid == "@文档五", "必须把越界标签报回去，供提示用户"
        assert rest == "提问"

    def test_no_mention_returns_empty_and_untouched(self):
        got, rest, invalid = resolve_mention_in_scope("这篇论文的创新点是什么", [14, 6])
        assert got == [] and invalid is None
        assert rest == "这篇论文的创新点是什么", "无提及时正文**不得被改动**"

    def test_arabic_numeral_form_supported(self):
        """`@文档2` 与 `@文档二` 等价（前端弹窗两种都可能产生）。"""
        got, _, _ = resolve_mention_in_scope("@文档2 提问", [14, 6, 16, 19])
        assert got == [16]

    def test_scope_is_valid_set_not_whole_library(self):
        """★★ 标签空间 = **会话有效集**，不是全库。

        否则「文档七」会指到一篇**根本没关联**的论文 ⇒ 检索必然落空。
        """
        # 会话只关联 2篇⇒ 「文档三」越界（哪怕全库里有 17 篇）
        got, _, invalid = resolve_mention_in_scope("@文档三 提问", [16, 19])
        assert got == [] and invalid == "@文档三"

    def test_legacy_resolve_mention_unchanged(self):
        """★ 老函数（技能层）行为**逐字不变** —— M1 是新增路径，不是改老路径。"""
        ids = [14, 6, 16, 19]
        assert resolve_mention("@文档二 提问", ids) == (16, "提问")
        assert resolve_mention("没有提及", ids) == (None, "没有提及")


# ============================================================
# 端到端：M1 是否真的限缩了检索范围
# ============================================================
@pytest.fixture
def client():
    import os
    import tempfile

    from fastapi.testclient import TestClient

    import papermate.config as cfg_mod

    tmp = tempfile.mkdtemp(prefix="papermate_m1_mention_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


DOC_A = "甲论文的实验步骤：数据集为A数据集，训练100轮，学习率0.01，mAP 达到 88.0。"
DOC_B = "乙论文的实验步骤：数据集为B数据集，使用完全不同的小目标检测范式，mAP 仅为 21.5。"


def _setup_docs(client, session_id: int) -> dict:
    """上传两篇文档，返回 {标题片段: doc_id}。"""
    out = {}
    for name, body in (("甲论文", DOC_A), ("乙论文", DOC_B)):
        r = client.post(
            f"/api/v1/documents?session_id={session_id}",
            files={"file": (f"{name}.md", body.encode("utf-8"), "text/markdown")},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ready"
        out[name] = r.json()["id"]
    return out


def test_handwritten_mention_limits_retrieval_scope(client, caplog):
    """★★★ M1 的核心验收：**手打** `@文档N` 也要真正限缩检索范围。

    这是事故的直接对策。判据= 日志里 `[文档限定]` 行的 `doc_ids` 只含被提及的那一篇。
    """
    import logging

    caplog.set_level(logging.INFO, logger="papermate.api.routes_search")
    s = client.post("/api/v1/sessions", json={"title": "m1"}).json()
    ids = _setup_docs(client, s["id"])
    # 前端弹窗会给出的**结构化** id = "文档一" = id 降序首位
    first = mention_order(list(ids.values()))[0]

    r = client.post(
        f"/api/v1/sessions/{s['id']}/messages/stream",
        json={"content": "@文档一 总结一下论文的实验步骤"},
    )
    assert r.status_code == 200, r.text
    lines = [rec.getMessage() for rec in caplog.records if "[文档限定]" in rec.getMessage()]
    assert lines, "手打 @必须被解析并留下可观测日志（此前完全无⇒ 排查不到）"
    assert f"doc_ids=[{first}]" in lines[0], f"范围未限缩到被提及的那篇：{lines[0]}"
    assert "手打文本回退" in lines[0], "必须标明来源是文本回退（区别于前端弹窗）"


def test_structured_mention_still_works_and_logged_as_structured(client, caplog):
    """★ 既有结构化路径**不得回归**（前端弹窗是主路径）。"""
    import logging

    caplog.set_level(logging.INFO, logger="papermate.api.routes_search")
    s = client.post("/api/v1/sessions", json={"title": "m1b"}).json()
    ids = _setup_docs(client, s["id"])
    first = mention_order(list(ids.values()))[0]

    r = client.post(
        f"/api/v1/sessions/{s['id']}/messages/stream",
        json={"content": "总结一下论文的实验步骤", "mention_doc_ids": [first]},
    )
    assert r.status_code == 200
    lines = [rec.getMessage() for rec in caplog.records if "[文档限定]" in rec.getMessage()]
    assert lines, "结构化提及也必须留痕"
    assert "结构化" in lines[0] and "手打文本回退" not in lines[0], lines[0]


def test_no_mention_leaves_scope_untouched(client, caplog):
    """★ **零回归**：没有 @ 提及时**不得**限缩（普通问答仍要搜全部关联文档）。"""
    import logging

    caplog.set_level(logging.INFO, logger="papermate.api.routes_search")
    s = client.post("/api/v1/sessions", json={"title": "m1c"}).json()
    _setup_docs(client, s["id"])
    r = client.post(
        f"/api/v1/sessions/{s['id']}/messages/stream",
        json={"content": "这两篇论文的实验步骤分别是什么"},
    )
    assert r.status_code == 200
    assert not [r_ for r_ in caplog.records if "[文档限定]" in r_.getMessage()], (
        "无提及时不得限缩检索范围"
    )


def test_source_contract_mention_parsed_in_routes():
    """源码契约：M1 的回退必须挂在 `routes_search`（而非只在 skills 层）。

    这条是"回归防护"——若有人把回退删掉、或只留技能层解析，本用例变红。
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py"
    ).read_text(encoding="utf-8")
    assert "resolve_mention_in_scope" in src, "普通对话路径必须接入 @文档N 解析"
    assert "if not mention_ids:" in src, "回退必须以「结构化字段为空」为前提（优先级不可颠倒）"
    #★ 越界标签必须被使用（提示用户），不能解析出来就扔掉
    assert "mention_invalid_label" in src


def test_mention_invalid_label_is_reported(client, caplog):
    """★★ 越界（关联 2 篇却写「文档五」）必须**据实告知**，不得静默全库盲搜。"""
    import logging

    caplog.set_level(logging.INFO, logger="papermate.api.routes_search")
    s = client.post("/api/v1/sessions", json={"title": "m1d"}).json()
    _setup_docs(client, s["id"])
    r = client.post(
        f"/api/v1/sessions/{s['id']}/messages/stream",
        json={"content": "@文档五 总结一下论文的实验步骤"},
    )
    assert r.status_code == 200
    lines = [rec.getMessage() for rec in caplog.records if "[文档限定]" in rec.getMessage()]
    assert lines, "越界也必须留痕（否则用户以为限定生效了）"
    assert "@文档五" in lines[0] and "越界" in lines[0], lines[0]
    # 越界时**不得**限缩到某一篇（宁可多搜，也不能搜错）
    assert mention_order([]) == []


# ============================================================
# M3 收口：标签**只能有一份实现**（防再次分叉）
# ============================================================
class TestLabelSingleSource:
    """★ 三处曾各抄一份 `cn` 表 ⇒ 前端改样式会**静默**失配（技能回复里的
    「文档三」与面板上的「文档三」不是同一篇）。本类钉住"单一来源"。

    实现约定：三处 `_doc_label` 都是**薄委托**，本体在 `services/doc_labels`。
    这里用源码契约断言——项目无 testing-library，且委托关系本身就是契约。
    """

    _ROOT = None

    def _src(self, rel: str) -> str:
        from pathlib import Path

        if self._ROOT is None:
            TestLabelSingleSource._ROOT = Path(__file__).resolve().parents[1]
        return (TestLabelSingleSource._ROOT / rel).read_text(encoding="utf-8")

    def test_no_local_cn_table_left(self):
        """★ 任何模块都**不得**再自带 `cn = ["一", "二", …]` 标签表。"""
        for rel in (
            "src/papermate/skills/planning_skill.py",
            "src/papermate/services/summary_service.py",
            "src/papermate/skills/_common.py",
        ):
            src = self._src(rel)
            assert 'cn = ["一"' not in src, f"{rel} 又抄了一份中文数字表 ⇒ 标签会分叉"

    def test_delegating_modules_use_shared_source(self):
        """两份 `_doc_label` 必须**委托** `doc_labels.doc_label`（而不是各自实现）。

        ⚠️ 两种导入写法都合法：`services/summary_service` 用**同包相对导入**
        （`from .doc_labels import`），`skills/planning_skill` 用**全路径**。
        断言必须**同时接受** —— 否则会把正确写法判成缺陷（假失败），
        或为了让断言通过而把正确写法改成错的（更糟）。
        """
        for rel in (
            "src/papermate/skills/planning_skill.py",
            "src/papermate/services/summary_service.py",
        ):
            src = self._src(rel)
            has_import = (
                "from papermate.services.doc_labels import doc_label" in src
                or "from .doc_labels import doc_label" in src
            )
            assert has_import, f"{rel} 未从单一来源导入 doc_label（相对/全路径皆可）"
            assert "return doc_label(index)" in src, f"{rel} 的 _doc_label 必须委托，不得自实现"

    def test_skills_common_mention_order_is_the_only_order(self):
        """★ 标签顺序的**唯一**口径在 `_common.mention_order`（已转发到 doc_labels）。"""
        from papermate.skills._common import mention_order as mo

        assert mo([14, 6, 16, 19]) == mention_order([14, 6, 16, 19]), (
            "skills._common.mention_order 必须与 doc_labels.mention_order 同源"
        )

    def test_revision_keeps_its_own_ordinal_contract(self):
        """★★ `/改稿` 的「第 N 篇」**刻意保留** `ctx.doc_ids` 顺序（既定契约，**不是缺陷**）。

        2026-10-02 记录：曾尝试统一为 `mention_order`，被 8 条既有测试正确拦住
        —— 那是**行为变更**（会改变用户既有习惯），不是缺陷修复。
        本用例把"**刻意不改**"这个决定钉成契约，避免后人再犯同样的越界。
        """
        src = self._src("src/papermate/skills/revision.py")
        assert "return doc_ids[n - 1]" in src, (
            "/改稿 的序号解析必须保持 ctx.doc_ids 顺序（既定契约，M3 已决定不改）"
        )
        assert "刻意不改" in src, "必须留有该决定的书面说明，避免后人重复尝试并回退"
