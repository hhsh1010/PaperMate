"""D1 / A6（2026-09-21）：运行指标端点 + 阈值告警。

守护的重点是**度量口径**，不只是"接口能返回"：
- 未收尾 ≠ 错误（命名与语义必须一致，否则把主动取消报成故障）
- **小样本不告警**（1 条差评 = 100% 差评率，不能报警）
- 阈值与样本量必须回显（否则用户看不懂为什么报警）
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from papermate.api.routes_metrics import router
from papermate.models.db import Feedback, Message, Trace, init_db, make_engine


def _mk_trace(
    tid: str, *, done: bool = True, latency: int = 1000, cost: float = 0.01, steps: int = 3,
    extra_events: list[dict] | None = None, ttft: int | None = None,
    peak: int | None = None, first: int | None = None,
    gate_hits: int | None = None, compacted: int | None = None,
):
    events = [{"idx": 0, "type": "tool_call", "data": {"name": "retrieve"}}]
    events.append({"idx": 1, "type": "tool_result",
                   "data": {"ok": True, "output": {"results": []}}})
    if done:
        # O-12：`ttft_ms` 只在**已采集**时出现（老轨迹没有这个键 ⇒ 必须计入 missing）
        data: dict = {"steps": steps}
        if ttft is not None:
            data["ttft_ms"] = ttft
        # 2026-10-02 上下文水位：同样「**有才写**」⇒ 老轨迹缺键必须计入 `ctx_peak_missing`。
        # ⚠️ `prompt_tokens`（整轮**累计**）恒存，用它与 `peak`（单次）形成对照 ——
        #    若实现误用 `total` 当水位，`ctx_peak_p50` 会是 20000 而非peak 值。
        data["prompt_tokens"] = 20000
        if peak is not None:
            data["peak_prompt_tokens"] = peak
        if first is not None:
            data["first_prompt_tokens"] = first
        if gate_hits is not None:
            data["_gate_hits"] = gate_hits
        if compacted is not None:
            data["_compacted_dropped"] = compacted
        events.append({"idx": 2, "type": "done", "data": data})
    if extra_events:
        events.extend(extra_events)
    return Trace(
        trace_id=tid, session_id=1, events_json=json.dumps(events, ensure_ascii=False),
        total_steps=steps, total_cost=cost, total_latency_ms=latency,
    )


@pytest.fixture()
def client(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path/'m.db'}")
    init_db(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        from papermate.models.db import ChatSession

        s.add(ChatSession(id=1, title="t"))
        s.add(ChatSession(id=99, title="other"))   # 外键要求先存在（scope 测试用）
        s.add(Message(id=1, session_id=1, role="assistant", content="a", cost=0.02))
        s.commit()

    class _S:
        _Session = Session

        @staticmethod
        def effective_model(session_id: int) -> str:
            """与 `ChatService.effective_model` 同签名的最小 stub。

            ★ 2026-10-05 补：`routes_metrics` 在**指定 session_id** 时会调它
            （按会话所用模型解析上下文预算，见 `routes_metrics.py:183`）——
            此前 stub 只有 `_Session`，带 session_id 的用例直接 `AttributeError`。
            本用例只验证 scope 过滤，不验证模型解析 ⇒ 返回固定值即可。
            """
            return "fake"

    app = FastAPI()
    app.state.chat_svc = _S()  # type: ignore[attr-defined]
    app.include_router(router)
    return TestClient(app), Session


def test_empty_window_is_healthy_and_zero(client):
    """空窗不能除零，也不能报警 —— 没数据 ≠ 系统不健康。"""
    c, _ = client
    m = c.get("/metrics").json()
    assert m["num_traces"] == 0
    assert m["avg_latency_ms"] == 0
    assert m["unfinished_rate"] == 0
    assert m["alerts"] == []
    assert m["healthy"] is True


def test_aggregates_normal_traces(client):
    c, Session = client
    with Session() as s:
        s.add(_mk_trace("t1", latency=1000, cost=0.01))
        s.add(_mk_trace("t2", latency=3000, cost=0.03))
        s.commit()
    m = c.get("/metrics", params={"days": 7}).json()
    assert m["num_traces"] == 2
    assert m["avg_latency_ms"] == pytest.approx(2000, abs=1)
    assert m["max_latency_ms"] == 3000
    assert m["avg_cost"] == pytest.approx(0.02, abs=1e-6)
    assert m["unfinished_rate"] == 0.0
    assert m["healthy"] is True


def test_unfinished_is_not_counted_as_error_but_reported(client):
    """★ 语义：未收尾（含主动取消）单独成指标，不混进"错误率"。"""
    c, Session = client
    with Session() as s:
        s.add(_mk_trace("t1", done=True))
        s.add(_mk_trace("t2", done=False))
        s.commit()
    m = c.get("/metrics").json()
    assert m["unfinished_rate"] == pytest.approx(0.5, abs=1e-6)
    assert "error_rate" not in m          # 不得再出现会误导的 error_rate 字段


def test_tool_error_rate(client):
    c, Session = client
    with Session() as s:
        t = _mk_trace("t1")
        ev = json.loads(t.events_json)
        ev[1]["data"]["ok"] = False       # 工具失败
        t.events_json = json.dumps(ev, ensure_ascii=False)
        s.add(t)
        s.commit()
    m = c.get("/metrics").json()
    assert m["tool_calls"] == 1
    assert m["tool_error_rate"] == 1.0


def test_small_sample_never_alerts(client):
    """★ 只有 1 条差评 → 差评率 100%，但样本 < min_samples，不得报警。"""
    c, Session = client
    with Session() as s:
        s.add(Feedback(session_id=1, message_id=1, rating=-1))
        s.commit()
    m = c.get("/metrics").json()
    assert m["feedback"]["negative"] == 1
    assert m["feedback"]["good_rate"] == 0.0
    assert m["alerts"] == [], "小样本不得产生噪音告警"
    assert m["healthy"] is True


def test_enough_bad_feedback_alerts(client):
    c, Session = client
    with Session() as s:
        for i in range(10):
            s.add(Feedback(session_id=1, message_id=10 + i, rating=-1))
        s.commit()
    m = c.get("/metrics").json()
    codes = [a["code"] for a in m["alerts"]]
    assert "bad_rate" in codes
    assert m["healthy"] is False


def test_alert_carries_threshold_and_sample(client):
    """告警必须自带阈值与样本量 —— 只有 value 的红灯无法判断严重性。

    ★ T-09 口径升级：告警挂 `unfinished_rate_excluding_interrupt`（排除用户主动停止）
    ⇒ 用 **failed** 轨迹（带 error 事件）触发；interrupted 不再触发告警
    （`test_interrupted_is_not_counted_as_failure` 已守）。
    """
    c, Session = client
    with Session() as s:
        for i in range(6):
            s.add(_mk_trace(
                f"fail{i}", done=False, latency=200_000,
                extra_events=[{"idx": 9, "type": "error",
                               "data": {"failure_kind": "llm_error"}}],
            ))
        s.commit()
    m = c.get("/metrics").json()
    by = {a["code"]: a for a in m["alerts"]}
    assert "unfinished_rate_excluding_interrupt" in by
    a = by["unfinished_rate_excluding_interrupt"]
    assert a["threshold"] == pytest.approx(0.20)
    assert a["sample"] == 6
    assert a["message"]
    assert m["thresholds"]["unfinished_rate_excluding_interrupt"] == pytest.approx(0.20)
    assert m["failed_confirmed_rate"] == pytest.approx(1.0)


def test_scope_by_session(client):
    c, Session = client
    with Session() as s:
        s.add(_mk_trace("t1"))
        t2 = _mk_trace("t2")
        t2.session_id = 99
        s.add(t2)
        s.commit()
    assert c.get("/metrics", params={"session_id": 1}).json()["num_traces"] == 1
    assert c.get("/metrics", params={"session_id": 99}).json()["num_traces"] == 1


# ---------------------------------------------------------------- O-12：TTFT 口径
def test_ttft_aggregated_and_missing_counted(client):
    """★ O-12：首字延迟必须**只对已采集样本**算分位，缺口单列（不得用 0 冒充）。

    实测场景：埋点 2026-10-02 才上线 ⇒ 之前的历史轨迹**没有** `ttft_ms` 键。
    若把缺失当 0 参与计算，`p50` 会被稀释成假值（与 `total_latency_ms` 80.1% 空洞同源）。
    """
    c, Session = client
    with Session() as s:
        s.add(_mk_trace("t1", ttft=1000))
        s.add(_mk_trace("t2", ttft=3000))
        s.add(_mk_trace("t3", ttft=None))      # 老轨迹：无该键 ⇒ missing
        s.add(_mk_trace("t4", done=False))     # 未收尾：同样无 done ⇒ missing
        s.commit()
    m = c.get("/metrics", params={"days": 7}).json()
    assert m["ttft_samples"] == 2, "只有 2 条真的采到首字延迟"
    assert m["ttft_missing"] == 2, "老轨迹 + 未收尾必须计入缺口，不能被当成 0"
    assert m["ttft_p50_ms"] == pytest.approx(2000, abs=1)
    assert m["ttft_avg_ms"] == pytest.approx(2000, abs=1)


def test_ttft_empty_window_is_zero_not_crash(client):
    """空窗不得除零、不得抛错（与 latency 同口径）。"""
    c, _ = client
    m = c.get("/metrics").json()
    assert m["ttft_samples"] == 0 and m["ttft_missing"] == 0
    assert m["ttft_avg_ms"] == 0.0 and m["ttft_p50_ms"] == 0.0


def test_extract_ttft_ms_never_fabricates():
    """单元层：`extract_ttft_ms` 对坏形状/缺键一律返回 None，**绝不返回 0 冒充**。"""
    from lightcore.core.trace_metrics import extract_ttft_ms, ttft_percentiles

    assert extract_ttft_ms([{"type": "done", "data": {"ttft_ms": 1234}}]) == 1234
    assert extract_ttft_ms([{"type": "done", "data": {}}]) is None          # 老轨迹
    assert extract_ttft_ms([{"type": "tool_result", "data": {}}]) is None   # 无 done
    assert extract_ttft_ms([{"type": "done", "data": {"ttft_ms": True}}]) is None  # bool 不是耗时
    assert extract_ttft_ms([{"type": "done", "data": {"ttft_ms": "123"}}]) is None  # 字符串不接受
    assert extract_ttft_ms(["not-a-dict"]) is None
    assert extract_ttft_ms(None) is None
    # 分位函数：缺口语义与 latency_percentiles 一致
    p = ttft_percentiles([1000, 3000, None, None])
    assert p["samples"] == 2 and p["missing"] == 2 and p["max"] == 3000
    assert ttft_percentiles([None])["samples"] == 0


# ============================================================
# 上下文水位（2026-10-02）：让"额度够不够"在事发前可见
# ============================================================
class TestContextWaterline:
    def test_peak_p50_uses_single_call_peak_not_round_total(self, client):
        """★★ 口径守卫：水位必须取 `peak_prompt_tokens`（**单次**），
        而**不是** `prompt_tokens`（整轮**累计**）。

        脚手架里 `prompt_tokens` 恒为 20000、`peak` 分别是 10000/16000/24000
        ⇒ 若误用累计值，`ctx_peak_p50` 会是 20000；正确应得 16000。
        这一点在真实库里尤其重要：某条轨迹 `prompt_tokens` 高达 82,057，
        把它当"上下文用量"会把水位读成 2.7 倍 ⇒ 误判"额度不够"。
        """
        c, Session = client
        with Session() as s:
            s.add(_mk_trace("w1", peak=10000, first=9000))
            s.add(_mk_trace("w2", peak=16000, first=15000))
            s.add(_mk_trace("w3", peak=24000, first=22000))
            s.commit()
        m = c.get("/metrics", params={"days": 7}).json()
        assert m["ctx_peak_p50"] == 16000, (
            f"水位取错了口径：得到 {m['ctx_peak_p50']}（16000 才是单次 prompt 的中位；"
            f"20000 是整轮累计）"
        )
        assert m["ctx_peak_max"] == 24000

    def test_budget_and_trigger_come_from_runtime_settings(self, client):
        """★ `ctx_budget` 必须是**当前生效的**配置（用户可在设置页改），
        不是代码默认值 —— 否则面板会报"水位 84%"而线上跑的是另一个额度。"""
        c, _ = client
        m = c.get("/metrics").json()
        assert m["ctx_budget"] == 30000
        assert m["ctx_trigger"] == 21000, "触发线必须是 budget × 0.7（派生，不另设配置）"
        # ⚠️ 本fixture 库里**没有带 peak 的轨迹** ⇒ `peak_p50` 正确地是 `None`（量不到），
        # **不是 0** —— 所以这里只能验额度/触发线，余量须在有样本的库里验（见下一条）。
        assert m["ctx_peak_p50"] is None and m["ctx_headroom_median"] is None

    def test_headroom_is_budget_minus_median_peak(self, client):
        """★ 余量 = `budget − peak_p50`：**中位情况下还剩多少空间可塞**（负数=常在越线）。"""
        c, Session = client
        with Session() as s:
            s.add(_mk_trace("h1", peak=10000))
            s.add(_mk_trace("h2", peak=20000))
            s.commit()
        m = c.get("/metrics", params={"days": 7}).json()
        assert m["ctx_peak_p50"] == 15000
        assert m["ctx_headroom_median"] == 15000, "余量 = 30000 − 15000"

    def test_old_traces_counted_as_missing_not_zero(self, client):
        """★★ 缺口必须可辨识：老轨迹没有 `peak` 字段 ⇒ 计入 `ctx_peak_missing`。

        绝不能让"缺字段"表现为 `peak_p50 = 0"（那会被读成"水位极低、很安全"）。
        真实库实测：250 条轨迹里只有 35 条带 `peak` ⇒ 缺口占 86%。
        """
        c, Session = client
        with Session() as s:
            s.add(_mk_trace("n1"))                # 老轨迹：无 peak
            s.add(_mk_trace("n2", done=False))     # 未收尾：同样无
            s.add(_mk_trace("n3", peak=12000))     # 唯一有样本的
            s.commit()
        m = c.get("/metrics", params={"days": 7}).json()
        assert m["ctx_peak_samples"] == 1
        assert m["ctx_peak_missing"] == 2, "老轨迹 + 未收尾必须计入缺口，不能被当成 0"
        assert m["ctx_peak_p50"] == 12000

    def test_no_peak_sample_reports_null_not_zero(self, client):
        """★ 无任何样本时 `ctx_peak_p50` 必须是 `None`（= 量不到），**不是 0**。

        报 0 会让前端显示"水位 0%"，等于**谎报安全**。
        """
        c, Session = client
        with Session() as s:
            s.add(_mk_trace("z1"))                # 全部无 peak
            s.commit()
        m = c.get("/metrics", params={"days": 7}).json()
        assert m["ctx_peak_samples"] == 0
        assert m["ctx_peak_p50"] is None, "无样本必须报 null（量不到），不得报 0 冒充安全"
        assert m["ctx_headroom_median"] is None

    def test_over_trigger_and_over_budget_counted(self, client):
        """★ 越线计数回答"这常不常发生"：`over_budget > 0` 就是发生过熔断。"""
        c, Session = client
        with Session() as s:
            s.add(_mk_trace("o1", peak=15000))    # 安全
            s.add(_mk_trace("o2", peak=22000))    # 过触发线 21000
            s.add(_mk_trace("o3", peak=33000))    # ★ 直接越熔断线（真实事故值）
            s.commit()
        m = c.get("/metrics", params={"days": 7}).json()
        assert m["ctx_over_trigger"] == 2
        assert m["ctx_over_budget"] == 1, "那次33,052 的熔断必须能被这条指标看见"

    def test_elided_traces_counted_once_per_trace(self, client):
        """★ 「被动过刀」按**轨迹**计一次（与 `failure_kinds` 同口径）。

        证据被裁过的完整性损失**judge 测不出来**（它只测忠实度/拒答），
        所以这个数是"有多少轮证据被动过"的唯一可见信号。
        """
        c, Session = client
        with Session() as s:
            s.add(_mk_trace("e1", peak=25000, gate_hits=2, compacted=1))  # 两种都发生 → 仍计 1
            s.add(_mk_trace("e2", peak=25000, gate_hits=0, compacted=0))  # 未被动刀
            s.commit()
        m = c.get("/metrics", params={"days": 7}).json()
        assert m["ctx_elided_traces"] == 1, "同一轮既裁剪又压缩也只算1 条（分母须与其他分类一致）"


class TestWaterlineExtraction:
    """单元层：`extract_context_waterline` 对坏形状/缺键一律 `None`，不伪造。"""

    def _m(self, events):
        from lightcore.core.trace_metrics import extract_context_waterline
        return extract_context_waterline(events)

    def test_reads_all_waterline_fields(self):
        got = self._m([{"type": "done", "data": {
            "peak_prompt_tokens": 20000, "first_prompt_tokens": 11000,
            "prompt_tokens": 45000, "_gate_hits": 2, "_compacted_dropped": 1,
        }}])
        assert got == {"peak": 20000, "first": 11000, "total": 45000,
                       "gate_hits": 2, "compacted_dropped": 1}

    def test_never_fabricates(self):
        assert self._m([{"type": "done", "data": {}}])["peak"] is None      # 老轨迹
        assert self._m([{"type": "tool_result", "data": {}}])["peak"] is None  # 无 done
        assert self._m("not-a-list")["peak"] is None
        assert self._m(None)["peak"] is None
        # 0 / bool / 字符串 / 负数 一律不认（"没量到" ≠ "量到 0"）
        for bad in (0, True, "20000", -5, None):
            assert self._m([{"type": "done", "data": {"peak_prompt_tokens": bad}}])["peak"] is None, bad

    def test_context_waterline_handles_empty_window(self):
        """空窗不得除零、不得抛错。"""
        from lightcore.core.trace_metrics import context_waterline
        r = context_waterline([], 30000)
        assert r["peak_samples"] == 0 and r["peak_p50"] is None
        assert r["budget"] == 30000 and r["trigger"] == 21000
