"""L0-3 答案层判官回归（`eval/judge.py` + `run_eval --judge`，2026-09-16）

被锁定的**设计取向**（每一条都对应一个真实风险）：

1. **提示词单一来源**：模板从 `judge_prompts/*.md` 的代码块读 —— 不出现"文档一套、代码另一套"
   （本项目 `SYSTEM_PROMPT` 有同名两份不同内容的**前科**）。
2. **不许同源自评**：判官厂商与答案厂商必须不同（同源会系统性偏乐观）→ 显式报错，而非静默接受。
3. **失败 ≠ 通过**：判官输出非 JSON / 缺字段 / 超时 / 调用异常 → `ok=False` + `value=None`；
   聚合时**不计入分母**，但 `errors` 必须可见（否则"分母悄悄缩小"会让指标变好看）。
4. **无密钥必须报错退出**：不产出"检索数字齐全、只是没有判官结论"的报告 —— 那会被误读为通过。
5. **判官异常不拖垮整轮**：单条失败只是那一条不进分母，其余照跑。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest import mock

import httpx
import pytest

_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import judge as judge_mod  # noqa: E402

from lightcore.llm.router import PRESET_ENDPOINTS  # noqa: E402

# ---------------------------------------------------------------- 提示词单一来源


def test_prompt_loaded_from_markdown_template() -> None:
    """模板必须来自 md 的代码块，且含所需占位符（改提示词只改 md）。"""
    faith = judge_mod.load_prompt("faithfulness")
    abst = judge_mod.load_prompt("abstention")

    assert "{context}" in faith and "{answer}" in abst
    assert "{question}" in abst
    # 不应把 md 的标题、说明文字也塞进 prompt
    assert not faith.startswith("#")


def test_missing_prompt_raises(tmp_path: Path) -> None:
    with pytest.raises(judge_mod.JudgeUnavailable):
        judge_mod.load_prompt("nonexistent", template_dir=tmp_path)


def test_render_rejects_template_without_placeholder() -> None:
    """模板缺占位符必须**报错**：否则会静默发出没有资料的 prompt，判分无意义却看不出来。"""
    with pytest.raises(judge_mod.JudgeUnavailable) as ei:
        judge_mod._render("请判断：{answer}", context="资料")

    assert "context" in str(ei.value)


def test_render_keeps_literal_json_braces() -> None:
    """回归：模板含 JSON 示例（字面 `{}`）——曾用 `str.format` 直接 `KeyError: '"faithful"'`。"""
    out = judge_mod._render(
        '输出 {"faithful": true|false}\n资料：{context}', context="X"
    )

    assert out.endswith("资料：X")
    assert '{"faithful": true|false}' in out, "字面花括号必须原样保留"


def test_markdown_without_code_block_raises(tmp_path: Path) -> None:
    (tmp_path / "broken.md").write_text("# 只有标题，没有代码块\n", encoding="utf-8")
    with pytest.raises(judge_mod.JudgeUnavailable):
        judge_mod.load_prompt("broken", template_dir=tmp_path)


# ---------------------------------------------------------------- 跨厂商


def test_judge_provider_is_always_cross_vendor() -> None:
    assert judge_mod.pick_judge_provider("deepseek") == "qwen"
    assert judge_mod.pick_judge_provider("qwen") == "deepseek"


def test_build_llm_unknown_provider_raises() -> None:
    with pytest.raises(judge_mod.JudgeUnavailable):
        judge_mod.build_llm("openai")


def test_build_llm_applies_model_override() -> None:
    llm = judge_mod.build_llm("deepseek", model="deepseek-chat", api_key="k")

    assert llm.model_name == "deepseek-chat"
    assert llm.endpoint.base_url == PRESET_ENDPOINTS["deepseek"].base_url, "换模型不应换端点"


# ---------------------------------------------------------------- 解析容错


@pytest.mark.parametrize("raw", [
    '{"faithful": true, "reason": "ok"}',
    '```json\n{"faithful": true, "reason": "ok"}\n```',
    '```\n{"faithful": true}\n```',
    '好的，我的判断是：{"faithful": true, "reason": "ok"} 以上。',
])
def test_parse_verdict_tolerates_wrappers(raw: str) -> None:
    v = judge_mod.parse_verdict(raw, "faithful")

    assert v.ok is True and v.value is True


@pytest.mark.parametrize("raw", [
    "",
    "模型完全没给 JSON",
    '{"faithful": "yes"}',          # 类型不对
    '{"verdict": true}',            # 字段名不对
    "[1, 2, 3]",                    # 不是对象
])
def test_parse_failure_never_counts_as_pass(raw: str) -> None:
    """核心：解析失败一律 `ok=False` —— **绝不默认 faithful=true**。"""
    v = judge_mod.parse_verdict(raw, "faithful")

    assert v.ok is False
    assert v.value is None, "失败不得落成任何一档结论"
    assert v.error


# ---------------------------------------------------------------- 聚合口径


def test_aggregate_excludes_errors_from_denominator_but_reports_them() -> None:
    vs = [
        judge_mod.JudgeVerdict(ok=True, value=True),
        judge_mod.JudgeVerdict(ok=True, value=False),
        judge_mod.JudgeVerdict(ok=False, value=None, error="超时"),
    ]

    agg = judge_mod.aggregate(vs, "faithfulness")

    assert agg["judged"] == 2 and agg["hits"] == 1
    assert agg["faithfulness"] == 0.5
    assert agg["errors"] == 1, "失败条数必须暴露，否则分母缩小无人察觉"
    assert agg["error_samples"] == ["超时"]


def test_aggregate_all_failed_is_none_not_zero() -> None:
    """全失败 → `None`（无结论），不是 `0.0`（会被读成"全都不忠实"）。"""
    agg = judge_mod.aggregate([judge_mod.JudgeVerdict(ok=False, value=None, error="x")], "faithfulness")

    assert agg["judged"] == 0
    assert agg["faithfulness"] is None


# ---------------------------------------------------------------- 判官调用（stub）


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _reply(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


def _judge_with(handler, **kw) -> judge_mod.Judge:
    llm = judge_mod.build_llm("deepseek", api_key="test-key", client=_client(handler))
    return judge_mod.Judge(llm, **kw)


def test_faithfulness_uses_temperature_zero_and_real_prompt() -> None:
    """判分必须 temperature=0（可复现），且 prompt 里带上上下文与回答。"""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body)
        return _reply('{"faithful": false, "reason": "上下文没有这个数字"}')

    j = _judge_with(handler)
    v = asyncio.run(j.faithfulness(context="资料：分块 300-500 字", answer="分块固定 1000 字"))

    assert v.ok and v.value is False and "没有这个数字" in v.reason
    assert seen["temperature"] == 0.0
    prompt = seen["messages"][-1]["content"]
    assert "资料：分块 300-500 字" in prompt and "分块固定 1000 字" in prompt


def test_abstention_prompt_carries_question() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _reply('{"abstained": true, "reason": "明确说明资料中没有"}')

    j = _judge_with(handler)
    v = asyncio.run(j.abstention(context="资料：分块策略", question="许可证费用是多少", answer="资料中没有相关信息"))

    assert v.ok and v.value is True
    assert "许可证费用是多少" in seen["messages"][-1]["content"]


# ---------------------------------------------- A3（2026-09-21）：改稿保真判官


def test_edit_fidelity_template_carries_all_three_inputs() -> None:
    """★ 改稿判官必须同时拿到 指令 / 原文 / 改后 —— 少任何一个都无法判定。"""
    tpl = judge_mod.load_prompt("edit_fidelity")
    for key in ("instruction", "before", "after"):
        assert key in tpl

    out = judge_mod._render(
        tpl, instruction="润色", before="A 模型达到 77.4% mAP", after="A 模型取得了 74.7% 的 mAP"
    )
    for frag in ("润色", "77.4% mAP", "74.7% 的 mAP"):
        assert frag in out


def test_edit_fidelity_prompt_lists_three_checks() -> None:
    """三项判定（事实保真 / 指令执行 / 未越界）缺一不可 ——
    去掉「指令执行」它就退化成普通忠实度判官（判不出"说完没做"）。"""
    tpl = judge_mod.load_prompt("edit_fidelity")
    for marker in ("事实保真", "指令执行", "未越界"):
        assert marker in tpl


def test_edit_fidelity_catches_changed_number() -> None:
    """★ 数值被改动必须能判否 —— 这是本判官存在的首要理由（改稿最危险的失败模式）。"""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _reply(
            '{"faithful": false, "violations": ["77.4% 被改成 74.7%"], "reason": "数值被改动"}'
        )

    j = _judge_with(handler)
    v = asyncio.run(
        j.edit_fidelity(instruction="润色", before="准确率 77.4%", after="准确率 74.7%")
    )
    assert v.ok and v.value is False
    assert seen["temperature"] == 0.0
    prompt = seen["messages"][-1]["content"]
    assert "润色" in prompt and "77.4%" in prompt and "74.7%" in prompt


def test_judge_http_error_becomes_verdict_error_not_exception() -> None:
    """单条判分失败必须收敛为 `ok=False`，**不得**向上抛（否则一条坏样本毁掉整轮评测）。"""
    j = _judge_with(lambda req: _reply("boo", status=500))
    v = asyncio.run(j.faithfulness(context="c", answer="a"))

    assert v.ok is False and v.value is None
    assert "错误" in v.error or "HTTP" in v.error or v.error


def test_judge_timeout_becomes_verdict_error() -> None:
    class _Slow:
        model_name = "slow"
        endpoint = PRESET_ENDPOINTS["deepseek"]

        async def chat(self, *a, **k):  # noqa: ARG002
            await asyncio.sleep(5)
            raise AssertionError("超时后不应继续")

    j = judge_mod.Judge(_Slow(), timeout_s=0.05)
    v = asyncio.run(j.faithfulness(context="c", answer="a"))

    assert v.ok is False
    assert "超时" in v.error


# ---------------------------------------------------------------- 密钥与厂商校验


def test_missing_api_key_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """无密钥 → `api_key_available()` False，`ensure_available` 抛可操作错误。

    ⚠️ 必须**同时**清掉两处来源（运行时设置 + 环境变量）：密钥解析与线上同源之后，
    只清环境变量是清不干净的 —— 本机 `data/provider_settings.json` 里存着真实密钥，
    测试会真的读到它（"测试隔离要跟着实现一起演进"的实例）。
    """
    from papermate import runtime_settings

    monkeypatch.delenv(PRESET_ENDPOINTS["deepseek"].api_key_env, raising=False)
    with mock.patch.object(runtime_settings, "get_state", lambda: _fake_state(None)):
        llm = judge_mod.build_llm("deepseek")

        assert llm.api_key_available() is False
        with pytest.raises(judge_mod.JudgeUnavailable) as ei:
            judge_mod.ensure_available(llm)
        assert "API Key" in str(ei.value)


def test_available_when_override_key_present() -> None:
    assert judge_mod.build_llm("deepseek", api_key="sk-x").api_key_available() is True


def test_auto_provider_without_any_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--judge` 且两家都没配密钥 → 明确报错（不静默降级成"跳过判官"）。"""
    import run_eval

    from papermate import runtime_settings

    for p in ("deepseek", "qwen"):
        monkeypatch.delenv(PRESET_ENDPOINTS[p].api_key_env, raising=False)
    monkeypatch.delenv("PAPERMATE_DEFAULT_MODEL", raising=False)

    with mock.patch.object(runtime_settings, "get_state", lambda: _fake_state(None)):
        with pytest.raises(judge_mod.JudgeUnavailable) as ei:
            run_eval._auto_answer_provider()
    assert "--judge" in str(ei.value)


def test_same_vendor_judge_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """不许同源自评 —— 显式指定同厂商时必须报错。"""
    import run_eval

    class _Args:
        answer_provider = "deepseek"
        judge_provider = "deepseek"

    with pytest.raises(judge_mod.JudgeUnavailable) as ei:
        run_eval._resolve_providers(_Args())
    assert "同源" in str(ei.value)


def test_auto_resolves_to_cross_vendor() -> None:
    import run_eval

    class _Args:
        answer_provider = "qwen"
        judge_provider = "auto"

    answer, judge_p = run_eval._resolve_providers(_Args())

    assert answer == "qwen" and judge_p == "deepseek"


# ---------------------------------------------------------------- 报告的判官块


def test_judge_block_has_both_numbers() -> None:
    """验收判据之一：报告必须同时出现 `faithfulness` 与 `abstention` 两个比例。"""
    cases = [
        {"id": "q1", "query": "问题一", "answerable": True},
        {"id": "n1", "query": "语料没有的问题", "answerable": False},
    ]
    contexts = {"q1": ["资料一"], "n1": ["资料二"]}

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        if "abstained" in prompt:
            return _reply('{"abstained": true, "reason": "如实说明"}')
        return _reply('{"faithful": true, "reason": "有据"}')

    answer_llm = judge_mod.build_llm("deepseek", api_key="k", client=_client(handler))
    j = judge_mod.Judge(judge_mod.build_llm("qwen", api_key="k", client=_client(handler)))

    class _Args:
        judge_timeout = 5.0

    block = asyncio.run(run_eval_judge_all(cases, contexts, answer_llm, j, _Args()))

    assert block["faithfulness"]["faithfulness"] == 1.0
    assert block["abstention"]["abstention"] == 1.0
    assert block["temperature"] == 0.0
    assert {d["id"] for d in block["details"]} == {"q1", "n1"}
    assert block["details"][0]["kind"] == "faithfulness"
    assert block["details"][1]["kind"] == "abstention"


def run_eval_judge_all(cases, contexts, answer_llm, judge, args) -> dict:
    import run_eval

    return run_eval._judge_all(cases, contexts, answer_llm, judge, args)


# ---------------------------------------------------------------- --judge-dump（事后可标注的唯一保证）


def test_judge_dump_records_exactly_what_was_judged(tmp_path: Path) -> None:
    """★ 回归：dump 必须落盘**判分当时真实用的**上下文与答案。

    动机是实测事故：报告只存 200 字预览且不存上下文 → 事后重新生成答案时，
    40 条里 **22 条与判分时用的答案实质不同**（`temperature=0` 也不确定），
    于是「判官结论」与「答案」配不上，人工 κ 直接失效。
    所以这条断言盯的是**同一性**：dump 里的 context/answer 必须与判分入参逐字一致。
    """
    cases = [{"id": "q1", "kind": "verbatim", "query": "问题一", "answerable": True}]
    contexts = {"q1": ["资料甲", "资料乙"]}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompt = payload["messages"][-1]["content"]
        if "faithful" in prompt and "abstained" not in prompt:
            return _reply('{"faithful": true, "reason": "有据"}')
        return _reply("这是模型生成的答案。")

    answer_llm = judge_mod.build_llm("deepseek", api_key="k", client=_client(handler))
    j = judge_mod.Judge(judge_mod.build_llm("qwen", api_key="k", client=_client(handler)))

    dump = tmp_path / "d.jsonl"
    dump.write_text("上一轮的残留内容\n", encoding="utf-8")   # 必须被清空，不能混在一起

    class _Args:
        judge_timeout = 5.0
        judge_dump = str(dump)

    block = asyncio.run(run_eval_judge_all(cases, contexts, answer_llm, j, _Args()))

    lines = [ln for ln in dump.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1, "残留内容未被清空"
    rec = json.loads(lines[0])
    assert rec["id"] == "q1" and rec["kind"] == "verbatim"
    assert rec["context"] == "资料甲\n\n资料乙", "dump 的上下文必须与判分时逐字一致"
    assert rec["answer"] == block["details"][0]["answer_preview"][:200]
    assert rec["judge_value"] is True and rec["answerable"] is True


def test_judge_dump_absent_by_default(tmp_path: Path) -> None:
    """不带 `--judge-dump` 时**不该**产生任何文件（默认行为不变）。"""
    cases = [{"id": "q1", "query": "问", "answerable": True}]

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return _reply('{"faithful": true, "reason": "x"}') if "faithful" in prompt else _reply("答")

    answer_llm = judge_mod.build_llm("deepseek", api_key="k", client=_client(handler))
    j = judge_mod.Judge(judge_mod.build_llm("qwen", api_key="k", client=_client(handler)))

    class _Args:
        judge_timeout = 5.0
        judge_dump = None

    asyncio.run(run_eval_judge_all(cases, {"q1": ["资料"]}, answer_llm, j, _Args()))
    assert list(tmp_path.iterdir()) == []


def test_generation_failure_lands_in_errors_not_denominator() -> None:
    """答案生成失败 → 该条记 error（不进分母），且**不会**被算成"不忠实"。"""
    cases = [{"id": "q1", "query": "问题", "answerable": True}]

    def handler(request: httpx.Request) -> httpx.Response:
        return _reply("x", status=500)  # 生成与判分都会失败

    answer_llm = judge_mod.build_llm("deepseek", api_key="k", client=_client(handler))
    j = judge_mod.Judge(judge_mod.build_llm("qwen", api_key="k", client=_client(handler)))

    class _Args:
        judge_timeout = 5.0

    block = asyncio.run(run_eval_judge_all(cases, {"q1": ["资料"]}, answer_llm, j, _Args()))

    assert block["faithfulness"]["judged"] == 0
    assert block["faithfulness"]["faithfulness"] is None
    assert block["faithfulness"]["errors"] == 1
    assert "生成失败" in block["details"][0]["error"]


# ---------------------------------------------------------------- 上下文来源


def test_run_case_fills_context_sink(tmp_path: Path) -> None:
    """判官要的"检索原文"由 `run_case` 顺手带出（不重复检索一遍）。"""
    import run_eval

    case = {
        "id": "q1", "query": "分块大小",
        "docs": {"d1": ["语义分块采用三百到五百字。", "另一个话题。"]},
        "gold": ["d1-c0"], "gains": {"d1-c0": 3},
    }
    factory = run_eval.PipelineFactory("fake", "fast")
    sink: dict[str, list[str]] = {}

    run_eval.run_case(case, 5, factory, context_sink=sink)

    assert sink["q1"], "context sink 必须被填充"
    assert any("三百到五百字" in t for t in sink["q1"])


# ---------------------------------------------------------------- 判分抽样（T-02-C）
#
# 为什么需要这些断言：`--judge` 现在是**每例两次真实 LLM 调用**（生成 + 判分），
# 用例集扩到 221 例后全量跑 = 442 次调用。抽样本身很容易写错，而写错的后果比"贵"更糟：
#   ① 抽到空子集 → 指标变成 `null`，看起来像"没评"而不是"没抽到"；
#   ② 抽样不可复现 → 两次运行的数字不可比，无法区分"质量变了"与"抽到的题变了"；
#   ③ 只抽到一类用例 → 另一类指标悄悄失去分母。
# 这三条正是下面锁定的内容。


def _fake_cases() -> list[dict]:
    """构造与真实集同构的小集合：五类，且**负样本在末尾**（真实文件的顺序）。"""
    def mk(prefix: str, n: int, kind: str, answerable: bool = True) -> list[dict]:
        return [{"id": f"{prefix}{i:02d}", "kind": kind, "query": f"q{i}",
                 "answerable": answerable} for i in range(1, n + 1)]

    return (mk("r", 10, "verbatim") + mk("p", 8, "paraphrase") + mk("g", 4, "multi_gold")
            + mk("m", 6, "multi_doc") + mk("n", 5, "negative", answerable=False))


def test_judge_sample_is_deterministic() -> None:
    """同一输入两次必须给出同一组用例 —— 否则判官数字跨版本不可比。"""
    import run_eval

    a, _ = run_eval._plan_judge_sample(_fake_cases(), 12)
    b, _ = run_eval._plan_judge_sample(_fake_cases(), 12)
    assert [c["id"] for c in a] == [c["id"] for c in b]


def test_judge_sample_never_drops_a_whole_kind() -> None:
    """★ 反向断言：名额再紧也必须**每类至少 1 条**。

    负样本在文件末尾；若按顺序截前 N 条，`abstention` 会一条都测不到，
    报告里 `abstention` 变 `null` —— 这会被误读成"该指标未评测"，而不是"没抽到题"。
    """
    import run_eval

    cases = _fake_cases()
    picked, plan = run_eval._plan_judge_sample(cases, len(cases) - 1)
    assert set(by := plan["by_kind"]) == {"verbatim", "paraphrase", "multi_gold", "multi_doc", "negative"}
    assert all(v["sampled"] >= 1 for v in by.values()), by
    assert any(not c.get("answerable", True) for c in picked), "负样本必须进样本，否则 abstention 无分母"


def test_judge_sample_respects_limit_and_covers_each_kind_evenly() -> None:
    import run_eval

    cases = _fake_cases()
    picked, plan = run_eval._plan_judge_sample(cases, 15)
    assert plan["n"] == len(picked) == 15, "抽出的条数必须等于请求值（不是近似）"
    # 层内等距的判据：**必须覆盖到层尾部**（10 取 4 → 下标 0,3,6,9，含最后一条）。
    # ⚠️ 不能用"首条不是第 0 项"当判据 —— 等距抽样**本来就包含首项**，
    #    那样写会误报（我第一版就是这么写错的）。截前缀的特征恰恰是"永远到不了尾部"。
    for kind, last_id in (("verbatim", "r10"), ("paraphrase", "p08"), ("multi_doc", "m06")):
        got = [c["id"] for c in picked if c["kind"] == kind]
        assert got, f"{kind} 未进样本"
        assert got[-1] == last_id, f"{kind} 层内抽样没覆盖到尾部（疑似截前缀）：{got}"


def test_judge_sample_full_mode_is_labelled_not_silent() -> None:
    """limit=0（或超过总数）必须是全量，且**如实标注** —— 不许把"全量"伪装成"抽样后正好全覆盖"。"""
    import run_eval

    cases = _fake_cases()
    for limit in (0, len(cases), len(cases) + 5):
        picked, plan = run_eval._plan_judge_sample(cases, limit)
        assert len(picked) == len(cases)
        assert plan["strategy"] == "full"
        assert plan["n"] == len(cases)


def test_judge_sample_plan_is_auditable() -> None:
    """报告里的抽样口径必须能回答"这版判官数字是在哪些题上算的"。"""
    import run_eval

    picked, plan = run_eval._plan_judge_sample(_fake_cases(), 12)
    assert plan["strategy"] == "stratified_stride"
    assert plan["ids"] == [c["id"] for c in picked], "ids 必须与真实抽出的用例一致"
    assert "不可与全量数字并列比较" in plan["note"]


def test_judge_sample_handles_missing_or_unknown_kind() -> None:
    """缺 kind 的用例不能被静默丢掉（会把它们整类排除在判分之外）。"""
    import run_eval

    cases = [{"id": "x1", "query": "q", "answerable": True}, {"id": "x2", "kind": None, "query": "q"}]
    picked, plan = run_eval._plan_judge_sample(cases, 1)
    assert plan["n"] == 1
    assert set(by := plan["by_kind"]) <= {"unknown"}
    assert sum(v["available"] for v in by.values()) == len(cases), "游离用例必须计入 available"


# ---------------------------------------------- 密钥解析必须与线上同源（T-02-C）
#
# 真实事故（2026-09-25）：用户把 Qwen key 存在**设置页**（= data/provider_settings.json），
# 而本模块原先只读 `os.getenv` → `--judge` 报「缺少 API Key」，明明配了却说没配。
# 应用侧三处都是 `api_key=pset.get("api_key")` 显式传入，环境变量只是回退 —— 评测必须照抄这个优先级。


def _fake_state(key: str | None) -> dict:
    return {"providers": {"deepseek": {"api_key": key}, "qwen": {"api_key": key}}}


def test_resolve_api_key_prefers_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """**设置页存的 key 优先于环境变量** —— 与 `chat_service` 的取值顺序一致。"""
    from papermate import runtime_settings

    monkeypatch.setenv(PRESET_ENDPOINTS["deepseek"].api_key_env, "sk-from-env")
    with mock.patch.object(runtime_settings, "get_state", lambda: _fake_state("sk-from-pset")):
        key, src = judge_mod.resolve_api_key("deepseek")
    assert key == "sk-from-pset"
    assert "运行时设置" in src


def test_resolve_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """运行时设置里没有时，环境变量仍应生效（.env 的配置不能被忽略）。"""
    from papermate import runtime_settings

    monkeypatch.setenv(PRESET_ENDPOINTS["deepseek"].api_key_env, "sk-from-env")
    with mock.patch.object(runtime_settings, "get_state", lambda: _fake_state(None)):
        key, src = judge_mod.resolve_api_key("deepseek")
    assert key == "sk-from-env"
    assert "环境变量" in src


def test_resolve_api_key_absent_is_reported_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    """两处都没有时必须返回 `(None, 未配置)` —— 不能编造、也不能抛异常掩盖。"""
    from papermate import runtime_settings

    monkeypatch.delenv(PRESET_ENDPOINTS["deepseek"].api_key_env, raising=False)
    with mock.patch.object(runtime_settings, "get_state", lambda: _fake_state(None)):
        key, src = judge_mod.resolve_api_key("deepseek")
    assert key is None
    assert src == "未配置"


def test_resolve_api_key_survives_broken_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """读运行时设置抛异常时不能把评测带崩 —— 回退环境变量即可。"""
    from papermate import runtime_settings

    monkeypatch.setenv(PRESET_ENDPOINTS["qwen"].api_key_env, "sk-env-qwen")
    with mock.patch.object(runtime_settings, "get_state", side_effect=RuntimeError("boom")):
        key, _ = judge_mod.resolve_api_key("qwen")
    assert key == "sk-env-qwen"


def test_build_llm_records_key_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """密钥来源要能进报告（否则事后查不出用的是哪一层配置）。"""
    from papermate import runtime_settings

    with mock.patch.object(runtime_settings, "get_state", lambda: _fake_state("sk-x")):
        llm = judge_mod.build_llm("deepseek")
    assert llm.key_source and "运行时设置" in llm.key_source
    assert judge_mod.build_llm("deepseek", api_key="sk-injected").key_source == "显式传入（测试/调用方）"


def test_answer_model_defaults_to_runtime_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ 答案模型必须跟随**线上默认模型**（同源），而不是端点默认值。

    否则答案层评测测的是 `qwen3.7-flash`，而线上跑的是 `qwen3.6-flash`
    —— 数字回答的不是同一个问题。
    """
    import run_eval

    from papermate import runtime_settings

    with mock.patch.object(runtime_settings, "get_state",
                           lambda: {"default_model": "qwen3.6-flash", "providers": {}}):
        assert run_eval._default_model_for("qwen") == "qwen3.6-flash"
        assert run_eval._default_model_for("deepseek") is None, "跨厂商的默认模型不能拿去当答案模型"


def test_answer_model_missing_falls_back_to_endpoint_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_eval

    from papermate import runtime_settings

    monkeypatch.delenv("PAPERMATE_DEFAULT_MODEL", raising=False)
    with mock.patch.object(runtime_settings, "get_state", lambda: {"default_model": "", "providers": {}}):
        assert run_eval._default_model_for("qwen") is None


# ---------------------------------------------- 2026-09-26：abstention 提示词加固（s13）
#
# 背景：κ 校准时人工标注抓到判官在本维度的**唯一错误** —— s13 负样本模型硬答
# "The experiments used 8 GPUs."，判官返回 `true`，理由却写着"8 GPUs 无资料依据"
# （理由与结论自相矛盾）。失败机制：把「内容在资料里查不到依据」当成了「已声明资料不足」。
#
# 加固落在 prompt 里，而 prompt 是**数据**不是代码 —— 不加守护，下一次有人"顺手精简"
# 就会静默失去这次修复，且没人会发现（判分照跑、数字照出）。


def _abstention_prompt() -> str:
    return judge_mod.load_prompt("abstention")


def test_abstention_prompt_guards_against_the_s13_failure_mode() -> None:
    """★ 加固条款必须在模板里：立场 vs 依据、中间字段、反例。"""
    p = _abstention_prompt()
    # ① 显式区分「回答的立场」与「内容的依据」—— 这是 s13 被判错的根因
    assert "立场" in p and "依据" in p
    # ② 中间结论字段：把最容易错的那一步显式化、可审计
    assert "states_no_info" in p
    assert "abstained" in p and "必须" in p
    # ③ 直接写出最容易判错的一类及其正确答案（反例优于抽象规则）
    assert "最容易判错" in p
    assert "绝不能" in p


def test_abstention_prompt_still_has_all_placeholders() -> None:
    """加固后模板必须仍能渲染（缺占位符会被 `_render` 明确报错）。"""
    p = _abstention_prompt()
    for token in ("{context}", "{question}", "{answer}"):
        assert token in p, f"模板缺少占位符 {token}"
    # 三个占位符都要被真实内容替换掉
    rendered = judge_mod._render(p, context="C", question="Q", answer="A")
    assert "{context}" not in rendered and "C" in rendered
    # JSON 示例里的字面 `{}` 不得被误当占位符（_render 只替换精确 token）
    assert "{" in rendered and "}" in rendered


def test_abstention_parses_verdict_with_extra_field() -> None:
    """新增的 `states_no_info` 是**附加**字段：解析仍取 `abstained`，不受影响。"""

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return _reply('{"states_no_info": false, "abstained": false, "reason": "给出了具体数字"}')

    j = _judge_with(handler)
    v = asyncio.run(j.abstention(context="资料：训练配置", question="用了几张卡",
                                 answer="The experiments used 8 GPUs."))
    assert v.ok and v.value is False and "具体数字" in v.reason


def test_abstention_missing_abstained_field_is_error_not_true() -> None:
    """只回 `states_no_info` 而没有 `abstained` → 解析失败（ok=False），**绝不当成通过**。"""

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return _reply('{"states_no_info": true, "reason": "..."}')

    j = _judge_with(handler)
    v = asyncio.run(j.abstention(context="c", question="q", answer="a"))
    assert v.ok is False and v.value is None
    assert "abstained" in (v.error or "")
