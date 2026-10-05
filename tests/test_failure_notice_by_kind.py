"""④（2026-10-02）：失败文案按 **`failure_kind`（结构化）** 分流，且**归因必须正确**。

## 缺陷背景（实测）

熔断中止（`budget_exceeded`）时，持久化到**回答气泡**的仍是通用兜底：

> 「抱歉，本次没有生成有效回答（可能是模型调用失败或返回内容为空）。建议：① 检查…账户余额；
> ② 换个模型重试；③ 稍后再试。」

—— 这三条建议对"上下文/成本超预算"**毫无作用**：余额没问题、换模型也不解决超限，
用户被引向**错误方向**（这是本项目"提示必须给对方向"的老账，参见 `_stream_error_notice`
的既有说明：用户曾因 DeepSeek 余额不足连续失败却以为是"模型连接"问题 —— 反向同理）。

同时，原先的 notice 分流靠**中文子串匹配**（`"熔断" in detail`）：文案一改就失配，
且"熔断"两字出现在无关文本里会**假匹配**。

## 本文件在守什么

1. **按枚举分流**（不靠文本）：`budget_exceeded` ⇒ 预算文案；`max_iterations` ⇒ 迭代文案；
2. ★ **正确归因**：预算文案必须**明确排除**"余额/模型故障"这两个误导方向，并给出对上号的建议；
3. **未知 kind 退回历史文案**（逐字），保证老行为与老断言的兼容面；
4. **notice 的 `kind` 仍为 `stream_failed`**（前端依赖），另有 `reason_kind` 供追踪。
"""

from __future__ import annotations

from papermate.services import failure_notice


class TestFallbackTextByKind:
    def test_budget_kind_gives_correct_attribution(self):
        """★★ 核心：预算熔断必须说清"不是余额/模型故障"，并给对上号的建议。"""
        text = failure_notice.fallback_text(
            "budget_exceeded", "单次请求上下文超限（prompt 33052 > 上限 30000 tokens）")
        assert "预算熔断" in text
        assert "不是**余额或模型故障**" in text or "不是余额或模型故障" in text
        # 数字必须带上（用户据此判断该调哪个上限）
        assert "33052" in text and "30000" in text
        # 建议必须指向"缩小上下文"，而不是"充值/换模型"
        assert "新建会话" in text or "精简历史" in text
        assert "充值" not in text

    def test_budget_advice_does_not_mislead(self):
        """★ 反误导：预算文案里不得出现"检查余额/换个模型"这类错误方向。"""
        text = failure_notice.fallback_text("budget_exceeded", "x")
        for wrong in ("账户余额", "换个模型重试"):
            assert wrong not in text, f"预算熔断文案仍在误导用户去「{wrong}」：{text}"

    def test_max_iterations_kind(self):
        text = failure_notice.fallback_text("max_iterations", "达到最大迭代 12")
        assert "步数" in text and "拆成更小" in text
        assert "预算熔断" not in text

    def test_unknown_kind_falls_back_verbatim(self):
        """未知/空 kind ⇒ **逐字**返回历史兜底（老行为与老断言兼容）。"""
        assert failure_notice.fallback_text("", "随便什么") == failure_notice.FALLBACK_REPLY
        assert failure_notice.fallback_text("weird_kind", "x") == failure_notice.FALLBACK_REPLY

    def test_detail_is_clipped(self):
        """厂商响应体可能很长 ⇒ 截断（安全，且不刷屏）。"""
        text = failure_notice.fallback_text("llm_error", "错" * 500)
        assert "…" in text
        assert len(text) < 400


class TestNoticeByKind:
    def test_notice_kind_stays_stream_failed_for_frontend(self):
        """前端按 `kind == "stream_failed"` 渲染警示卡 ⇒ 该值不得改。"""
        n = failure_notice.notice_for("budget_exceeded", "超限")
        assert n["kind"] == "stream_failed"
        assert n["reason_kind"] == "budget_exceeded"
        assert n["title"] and n["message"] and n["suggestion"]

    def test_notice_suggestion_matches_kind(self):
        n = failure_notice.notice_for("budget_exceeded", "单次请求上下文超限（prompt 33052 > 上限 30000）")
        assert "上限" in n["suggestion"] or "新建会话" in n["suggestion"]
        assert "充值" not in n["suggestion"]

    def test_legacy_substring_path_only_when_kind_missing(self):
        """★ kind 缺失（老轨迹）时才退回文本启发式 —— 且**不得**在 kind 存在时被使用。"""
        legacy = failure_notice.notice_for("", "Insufficient balance 余额不足")
        assert "充值" in legacy["suggestion"], "kind 缺失时仍应给出充值建议（老轨迹兜底）"
        assert "reason_kind" not in legacy, "kind 缺失时不应捏造 reason_kind"

        # kind 存在 ⇒ 即使 reason 里带"余额"字样，也必须按 kind 走（防文本假匹配）
        strict = failure_notice.notice_for("budget_exceeded", "余额 熔断 之类的噪声文本")
        assert "充值" not in strict["suggestion"]
        assert strict["reason_kind"] == "budget_exceeded"


class TestRouteWiring:
    def test_route_uses_structured_kind(self):
        """源码契约：路由必须捕获 `failure_kind` 并把它传给文案（而不是只传自由文本）。"""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py").read_text(
            encoding="utf-8")
        assert 'stream_error_kind = str((ev.data or {}).get("failure_kind") or "")' in src, (
            "必须捕获结构化的 failure_kind（否则只能靠中文子串猜原因）"
        )
        assert "failure_notice.fallback_text(stream_error_kind, stream_error_reason)" in src, (
            "气泡文案必须按 kind 分流（否则熔断时仍在说'可能是模型调用失败'）"
        )
        assert "failure_notice.notice_for(kind, reason)" in src, "notice 必须走单一来源"
        # 旧的"按中文子串猜"代码不得留在路由里（已迁到 failure_notice 的兜底分支）
        assert 'if "余额" in detail' not in src, "路由里仍留着按文本猜原因的旧逻辑"


class TestNoticePriorityKeepsLoopRisk:
    """★ 结构化归因优先**不得吃掉**"连续无效"的强打断升级。

    `loop_risk` 是既有行为层招呼（连续多轮无效 ⇒ 提示"卡在同一处，先排查"）。
    ④ 让 kind 归因插到 `_was_empty` 之前后，必须确认阈值以上仍然升级。
    """
    def test_source_contract_keeps_escalation(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py").read_text(
            encoding="utf-8")
        branch = src[src.index("elif _was_empty and stream_error_kind:"):]
        branch = branch[:branch.index("elif _was_empty:")]
        assert "_trailing_invalid >= _LOOP_THRESHOLD" in branch, (
            "结构化归因分支必须保留 loop_risk 升级（否则连续无效的强打断被吃掉）"
        )
        assert "_invalid_notice(\"loop_risk\"" in branch, "达阈值时必须仍走 loop_risk notice"
        assert "_stream_error_notice(stream_error_reason, stream_error_kind)" in branch, (
            "未达阈值时用 kind 自己的归因文案"
        )

    def test_retrieval_error_still_wins(self):
        """检索故障（已确认的现场）仍必须**优先于** kind 归因（更具体、可操作）。"""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "src" / "papermate" / "api" / "routes_search.py").read_text(
            encoding="utf-8")
        i_retrieval = src.index("if _retrieval_error:")
        i_kind = src.index("elif _was_empty and stream_error_kind:")
        assert i_retrieval < i_kind, "检索故障分支必须排在 kind 归因之前"
