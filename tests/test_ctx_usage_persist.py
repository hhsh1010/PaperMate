"""占用率**落库**（P1-a2，2026-10-03）：源码契约 + 边界说明。

## 修的缺陷
P1-a 的占用率只存在于**当轮** SSE 的 `done` 事件里 ⇒ 刷新 / 切会话后历史消息
（`MessageOut`，**无 done 字段**）拿不到 ⇒ 「**当轮看得到、刷新即消失**」
（与 `history.ts` 的 B2-X1 引用卡片同类）。修法：落进既有 `meta_json`（零 DDL）。

## ⚠️ 诚实边界：为什么这里只有**源码契约**、没有行为测试
`_persist_meta` / `_ctx_usage_from_done` 都定义在
`stream_message()` 的**闭包内**（`routes_search.py:610+`，父容器是行 428 的路由函数）
⇒ **无法 import**，也就无法单元测试其行为。
（★ P-03 2026-10-09：原第三个闭包 `_ctx_usage_from_events` 已删除 —— 它的唯一调用方
 是技能路径，而该路径的 `events` 收集器恒为空 ⇒ 接上但永远拿不到数；详见源码处注释。）

⇒ 折中：本文件用**源码契约**守住"接线没断 + 合并不污染 invalid"这两件**结构事实**；
   「回填后前端能否正确显示」由 `web/src/lib/ctxUsageHistory.test.ts` 的**行为测试**覆盖
   （`historyToThread` 是导出纯函数，可直接调用）。
⇒ **若要真正的后端行为测试，需先把这三个函数提取到模块级**（属独立重构，未做）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROUTES = Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py"


@pytest.fixture(scope="module")
def src() -> str:
    return ROUTES.read_text(encoding="utf-8")


class TestPersistWiring:
    """★★ 防"函数写了但没接上" —— 这是本项目反复踩过的失效模式。"""

    def test_both_persist_points_pass_ctx(self, src: str):
        """两处落库点（技能/流式路径）都必须把占用率传进 meta。

        ★ P-03（2026-10-09）更新：技能路径**不再**走 `_ctx_usage_from_events(events)`
        （技能分支 events 恒为空 ⇒ 恒返回 `{}`，等于"接了但没数据"），
        改为直接用 `_UsageRecorder` 捕获的 `_ctx_payload`（与实时 done 同源）。
        """
        assert "ctx=_ctx_usage_from_done(_ctx_payload)" in src, "技能路径没接占用率"
        assert "ctx=_ctx_usage_from_done(done_data)" in src, "流式路径没接占用率"

    def test_skill_done_event_carries_ctx(self, src: str):
        """★ P-03 新增：技能轮的 `done` 事件必须带上 `**_ctx_payload`。

        为什么必须守住：历史回填（meta）与**当轮实时**（done）是两条独立路径，
        只修一条仍会出现"当轮看得到、刷新即消失"（或反过来）。两条都要接。

        反面约束：注入必须**条件化**（`peak_prompt_tokens > 0` 才进 payload）——
        若硬写 `peak_prompt_tokens: 0`，前端 `done?.peak_prompt_tokens ?? ctx?.peak`
        会被 `0` 短路，把"无数据"显示成 0%。
        """
        assert "**_ctx_payload," in src, "技能轮 done 没带上下文占用"
        # ★ 2026-10-23 契约变更：技能轮**不再上报 ctx**。
        #   原因：recorder.peak_prompt_tokens 记的是"技能内部那次 LLM 调用"的 prompt
        #   （实测 456 tokens），不是会话上下文 ⇒ 面板会显示 `456 / 100K = 0.0%`，
        #   用户以为上下文归零 ✗。改为不提供该字段，前端自然回落到上一轮。
        assert "_ctx_payload: dict = {}" in src, (
            "技能轮应保持 _ctx_payload 为空（见 2026-10-23：ctx 在技能路径上语义不成立）"
        )
        assert "recorder.peak_prompt_tokens > 0" not in src, (
            "不应再把技能内部 prompt 当作上下文占用上报"
        )
        # 旧接线必须绝迹：events 收集器在技能分支恒为空 ⇒ 再接它等于接了空
        assert "ctx=_ctx_usage_from_events(events)" not in src, (
            "技能路径又接回了恒空的 events 收集器（P-03 已证伪该接线）"
        )

    def test_zero_cost_path_deliberately_not_wired(self, src: str):
        """直答路径（`_pump_direct`）**刻意不接**：它不走 loop ⇒ 没有峰值这个概念。

        ⚠️ 这里断言的是"**确实没接**"，防止后人"顺手补上"却传了个假值（伪造数据）。
        """
        # `_pump_direct` 的落库传的是**空 events 列表**（用单行锚点：
        # ⚠️ 跨行锚点会因实际换行位置而匹配失败，这是本项目记录过的坑）
        assert "history, [], final_text, model," in src

    def test_helpers_exist(self, src: str):
        # ★ P-03：`_ctx_usage_from_events` 已删除（唯一调用方是恒空的技能 events 路径）
        assert "def _ctx_usage_from_done(" in src, "缺少 _ctx_usage_from_done"
        assert "def _ctx_usage_from_events(" not in src, (
            "_ctx_usage_from_events 被重新引入 —— 它对技能路径恒返回空，勿再使用"
        )


class TestMergeDoesNotPolluteInvalid:
    """★★ 占用率是**中性观测**，绝不能把正常轮次标成 invalid。"""

    def test_ctx_merge_is_a_plain_update(self, src: str):
        """合并语句必须是 `{**(meta or {}), **ctx}` —— 不得夹带 `invalid`。"""
        assert 'meta = {**(meta or {}), **ctx}' in src, (
            "ctx 合并方式变了 ⇒ 复核是否会把正常轮次标成 invalid（污染「连续无效」计数）"
        )

    def test_invalid_only_from_notice(self, src: str):
        """`invalid: True` 只能由 `notice` 触发（原语义不得被改动）。"""
        assert 'meta = {"invalid": True, "notice": notice} if notice else None' in src

    def test_helper_rejects_non_int(self, src: str):
        """三个数都必须过 `isinstance(..., int)` —— 脏 meta 不能抛出（否则整页历史 500）。"""
        body = src.split("def _ctx_usage_from_done(")[1].split("def _ctx_usage_from_events(")[0]
        assert body.count("isinstance(") >= 3, "字段校验不足 ⇒ 脏数据可能让历史接口 500"
        assert "> 0" in body, "须排除 0/负值（无用且会显示成 0%）"
