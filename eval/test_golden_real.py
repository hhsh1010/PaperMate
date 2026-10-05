"""L0-2 真实文档评测集 · 一致性测试（2026-09-16）

**为什么必须有这组测试**：这套评测集的价值完全建立在"gold 来自真实链路"之上。
如果哪天有人（包括我自己）偷懒手工改 `gold`、或改了语料 PDF 却没重新标注，
指标就会重新变成"自嗨数字"而**没有任何东西会变红**。
本文件把三条硬约束锁住：

1. **gold 可复现**：拿 `eval/corpus/*.pdf` 走生产链路（PDF 解析 + 语义分块），
   每个可答用例的答案片段都必须能定位到**与 `gold` 完全相同**的 chunk id；
2. **分级相关性自洽**：`gold` 的 chunk 必须出现在 `gains` 里且取最高档（3）；
3. **三类覆盖齐备**：单文档 / 多文档 / 负样本，缺任何一类都算退化。

另外锁定 `run_eval` 的负样本语义：**不产出任何指标键**，
从而被 `summarize_results`（只平均"实际存在的键"）自动排除在聚合之外。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from make_golden_real import _norm, build_corpus
from metrics import summarize_results
from run_eval import GOLDEN_REAL, PipelineFactory, load_cases, run_case

_VALID = load_cases(GOLDEN_REAL)

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def test_golden_real_exists_with_three_kinds() -> None:
    """三类覆盖必须齐备 —— 少一类就说明退化（尤其是多文档与负样本最容易被悄悄删掉）。"""
    kinds = {
        "single": [c for c in _VALID if c.get("answerable") and len(c["corpus"]) == 1],
        "multi": [c for c in _VALID if c.get("answerable") and len(c["corpus"]) > 1],
        "negative": [c for c in _VALID if not c.get("answerable")],
    }
    # T-02（2026-09-25）扩容后收紧：下界由 8/2/3/25 提到卡片配额（≥40/≥30/≥30/≥200），
    # 目的是**防止今后悄悄退化** —— 旧下界太松，删掉八成用例也不会红。
    assert len(kinds["single"]) >= 40, f"单文档用例过少：{len(kinds['single'])}"
    assert len(kinds["multi"]) >= 30, f"多文档用例过少：{len(kinds['multi'])}"
    assert len(kinds["negative"]) >= 30, f"负样本过少：{len(kinds['negative'])}"
    assert len(_VALID) >= 200, f"总用例数退化：{len(_VALID)}"


def test_t02_corpus_docs_have_at_least_five_chunks(real_chunks: dict[str, list]) -> None:
    """★ T-02 新增语料**每篇必须 ≥5 块** —— 这是 `precision@k` 口径有效的前提。

    `metrics.precision_at_k` 的分母是 `min(k, 实际返回数)`，而单文档用例的候选数
    **就是该文档的块数**。块数 < k 时分母被压小，precision 会在更短的名次列表上取值，
    **看起来更高却没有意义**（实测：同一份用例集，语料 3 块/篇时 `precision@5`=0.4158、
    分母均值 3.44；补到 5 块/篇后分母均值 4.66、其中 165/183 例正好为 5）。

    注意：既有 3 篇旧语料（pm-rag / agent-loop / mixed-long）**不在此约束内** ——
    它们各只有 1~2 块，属 T-02 之前的历史口径，其用例保留在集内仅供对照。
    """
    from corpus_specs_t02 import CORPUS_T02

    bad = {d: len(real_chunks[d]) for d in CORPUS_T02 if len(real_chunks[d]) < 5}
    assert not bad, (
        f"以下 T-02 语料的块数不足 5，会让 precision@k 的分母被压小：{bad}；"
        "补长语料后重跑 python eval/make_golden_real.py"
    )


def test_all_corpus_docs_have_at_least_five_chunks(real_chunks: dict[str, list]) -> None:
    """★★ 2026-09-26（T-04-B）：约束**扩到全部语料**，旧 3 篇不再豁免。

    上面那条守卫原文写着「旧 3 篇（pm-rag / agent-loop / mixed-long）不在此约束内，
    各只有 1~2 块」。其代价在 T-04 开工时被测出：**18 例可答用例的 precision@5 分母塌缩**
    （如 r01 的候选池只有 1 个 chunk → 分母 = 1，precision 被结构性抬高到 1.0）。

    修法不是改指标，而是**只追加不改现有段落**地补长旧 3 篇（追加在末尾 → 既有分块与
    chunk id 不变，实测：28 个被 gold 引用的 chunk 一个未消失、221 条用例的 gold 零变化）：
    pm-rag 672→3095 字（5 块）、agent-loop 1598→4403（6 块）、mixed-long 745→2989（5 块）。

    扩长后实测：183/183 例的 precision@5 分母 **全部等于 5**（此前均值 4.66）。
    """
    bad = {d: len(v) for d, v in real_chunks.items() if len(v) < 5}
    assert not bad, (
        f"以下语料块数不足 5 → precision@k 的分母被压小（指标失真）：{bad}；"
        "请在 make_golden_real.py 的 CORPUS 里**末尾追加**段落（勿改既有段落，"
        "否则旧 gold 的 chunk id 会失效），再重跑 python eval/make_golden_real.py"
    )


def test_precision_cap_is_reported_and_saturated() -> None:
    """★ T-04-B：`precision@k` 已逼近理论上限 → 该指标**没有分辨力**，必须如实标注。

    上限 = `mean_i(min(|gold_i|, n_i) / n_i)`，`n_i = min(k, 实际返回数_i)`
    （**必须与 `precision_at_k` 的分母同源**，见 `_precision_cap` 的 2026-09-30 回归史）。
    实测各 kind 饱和度 ≈1.00，即"再好的检索也只能到这"。把上限写进报告，是为了让事后
    能区分「质量差」与「指标本来就涨不上去」——否则有人看到 precision=0.24 会误判为崩塌。
    ★ 2026-10-01（O-14）更新：扩充候选池后**饱和度已降到 1 以下**（0.88~0.94），
    这    是 O-14 的**预期成果** —— 饱和度 <1 意味着 precision 终于**能继续上涨**了
    （此前 =1 ⇒ 该指标只能变差）。下界因此放宽到 0.8；上界保持 1.0 —— **>1.0 永远是
    口径错误**（`_precision_cap` 的分母怪病复发），不该因为风向变了就放行。
    """
    import json

    import run_eval

    rep = json.loads((Path(__file__).resolve().parent / "report_real.json").read_text(encoding="utf-8"))
    for kind in ("verbatim", "multi_doc", "multi_gold"):
        blk = rep["metrics_by_kind"][kind]
        cap = blk.get("precision_cap")
        assert cap, f"{kind} 缺少 precision_cap 字段（报告口径不完整）"
        assert cap["cap"] > 0
        assert 0.8 <= cap["saturation"] <= 1.0, (
            f"{kind} 的 precision 饱和度 {cap['saturation']} 偏离预期区间 —— "
            "若 >1.0 说明分母口径又塌缩了；若 <0.8 说明候选池加难过头（gold 本身难召回），"
            "应回看 eval/expand_eval_pool.py 的干扰文档选取规则"
        )
    # 函数本身：空数据不得抛错
    assert run_eval._precision_cap([], 5, []) is None


def test_discriminative_flags_classify_correctly() -> None:
    """O-14（C）：分辨力判定必须"按还能不能涨"分类，而不是按数值好不好看。

    三种情形各自给到v阈值：
    1. 全 1.0 + precision 贴上限（旧语料的真实状态）⇒ 召回/精度类**全部贴顶**，
       只看 ndcg/mrr/map 还有空间；
    2. recall<1 且 precision 未到上限（O-14 扩充后）⇒ 六项**全部有分辨力**；
    3. 空输入不得抛错（开工脚本可能传空 kafka）。
    """
    import run_eval

    # 情形 1：旧语料（候选池≈一篇文档）的真实切片
    sat_metrics = {"recall@5": 1.0, "hit@5": 1.0, "precision@5": 0.24, "ndcg@5": 0.9366,
                   "mrr@10": 0.8445, "map": 0.8248}
    sat_cap = {"cap": 0.24, "observed": 0.24, "saturation": 1.0}
    out = run_eval._discriminative_flags(sat_metrics, sat_cap, 5)
    assert out["saturated_metrics"] == ["hit@5", "precision@5", "recall@5"], out
    assert out["discriminative_metrics"] == ["map", "mrr@10", "ndcg@5"], out

    # 情形 2：O-14 扩充候选池之后（recall 掉了、precision 没到上限）
    room_metrics = {"recall@5": 0.9429, "hit@5": 0.9571, "precision@5": 0.2257, "ndcg@5": 0.8716,
                    "mrr@10": 0.831, "map": 0.8026}
    room_cap = {"cap": 0.24, "observed": 0.2257, "saturation": 0.9405}
    out2 = run_eval._discriminative_flags(room_metrics, room_cap, 5)
    assert out2["saturated_metrics"] == [], out2
    assert set(out2["discriminative_metrics"]) == {
        "recall@5", "hit@5", "precision@5", "ndcg@5", "mrr@10", "map"}, out2
    # headroom 必须等于"离上限还差多少"，不允许写成绝対值以外的口径
    assert out2["headroom"]["recall@5"] == pytest.approx(0.0571, abs=1e-4)
    assert out2["headroom"]["precision@5"] == pytest.approx(0.0143, abs=1e-4)

    # 情形 3：空输入不得抛错
    assert run_eval._discriminative_flags({}, None, 5) == {}


def test_report_exposes_discriminative_flags() -> None:
    """O-14（C）：报告必须自证"哪些指标还能用来判断改动优劣"。

    为什么必须显式写进报告：recall/precision 在本语料的旧版本里是贴顶的
    （见上方两条测试），若报告不说明，后来者会拿贴顶指标去评效果 ⇒ 得到假阴性结论。
    """
    import json

    rep = json.loads((Path(__file__).resolve().parent / "report_real.json").read_text(encoding="utf-8"))
    for kind in ("verbatim", "multi_doc", "multi_gold", "paraphrase"):
        blk = rep["metrics_by_kind"][kind]
        flags = blk.get("discriminative")
        assert flags, f"{kind} 缺少 discriminative 字段（无法自证哪些指标有分辨力）"
        # 有/无分辨力的并集必须覆盖全部被判定指标，且二者互斥
        has, no = set(flags["discriminative_metrics"]), set(flags["saturated_metrics"])
        assert not (has & no), f"{kind} 同一指标同时被判为有/无分辨力：{has & no}"
        assert (has | no) == set(flags["has_headroom"]), f"{kind} 分类未覆盖全部指标"
        # O-14 的目的就是让召回/精度重新有分辨力 —— 若它们又回到贴顶，说明候选池被改回去
        assert "recall@5" in has, (
            f"{kind} 的 recall@5 又贴顶了（saturated={sorted(no)}）—— 候选池可能被改回单一文档，"
            "分辨力回到 0 会让后续所有优化**无从证明**"
        )


def test_precision_cap_denominator_matches_actual_returned() -> None:
    """★ 2026-09-30 回归守卫：上限的分母必须是 `min(k, 实际返回数)`，**不是**写死的 k。

    踩坑现场：单文档语料每篇仅 5~6 块而 k=10 ⇒ 旧公式按 /10 算上限、实测却按 /5 算
    precision ⇒ 饱和度 **1.96 > 1**（上限被突破 = 口径自相矛盾）。这条测试用**最小可复现
    输入**把该口径钉住：旧实现下 `cap` 会是 0.1、饱和度 2.0，直接变红。
    """
    import run_eval

    # 只返回 5 个（< k=10），gold 1 个且已命中 ⇒ precision@10 = 1/5 = 0.2
    rows = [{"id": "a", "precision@10": 0.2, "top_ids": ["g1", "x", "y", "z", "w"]}]
    cases = [{"id": "a", "gold": ["g1"]}]
    cap = run_eval._precision_cap(rows, 10, cases)
    assert cap is not None
    assert cap["cap"] == pytest.approx(0.2), "分母应为 min(k, 返回数)=5，而非 k=10"
    assert cap["saturation"] == pytest.approx(1.0), "旧口径会给 2.0（突破上限）"
    assert cap["k_effective"] == pytest.approx(5.0)
    # 返回满 k 个时，新旧口径一致（不得误伤）
    full = [{"id": "a", "precision@10": 0.1, "top_ids": [f"t{i}" for i in range(10)]}]
    assert run_eval._precision_cap(full, 10, cases)["cap"] == pytest.approx(0.1)


def test_repo_report_uses_documented_k() -> None:
    """★ 2026-09-30 新增：把「仓库自带报告的 `k`」钉住 —— 补上 `trusted` 抓不到的漂移盲区。

    `tests/test_eval_baseline.py` 的 `trusted` 只校验 **embedder / profile / 指纹新鲜度**，
    **不看口径参数**。于是 `report_real.json` 曾被用 `--k 10` 重跑而**没有任何东西报警**：
    `k` 一变，`precision_cap` 的分母语义随之改变（单文档语料只返回 5 个 ⇒ 返回数 5 < k 10），
    直接制造了「饱和度 1.96」的假塌缩。

    文档化生成命令（`docs/L0-优化实施与复测报告.md`、`test_eval_baseline` 的 skip 文案）
    **不含 `--k`** ⇒ 用的是 harness 默认值（`_BASELINE_K`）。这里把它作为契约断言。
    """
    import json

    import run_eval

    rep = json.loads((Path(__file__).resolve().parent / "report_real.json").read_text(encoding="utf-8"))
    assert rep["k"] == run_eval._BASELINE_K, (
        f"仓库自带报告用了非文档化的 k={rep['k']}（期望 {run_eval._BASELINE_K} = 命令行默认值）。"
        "口径参数一变，precision_cap / 各 @k 指标的含义都会跟着变 ⇒ 请按文档化命令重跑，"
        "或在测试与文档里同步声明新的 k。"
    )


def test_hard_case_kinds_present() -> None:
    """难例必须存在，否则指标会饱和在 1.0、失去分辨力（真实教训，见 README「两套集的本质差别」）。

    - `paraphrase`：query 不与答案共享关键词（BM25 帮不上，必须靠语义）；
    - `multi_gold`：一问需两处证据（单 gold 下 recall 恒为 0/1，饱和）。

    ⚠️ **2026-09-17 分块修复后的变化（如实记录）**：修掉"每个 block 立刻 flush"的碎片化后，
    小语料的 `pm-rag.pdf` / `mixed-long.pdf` **各自只剩 1 个 chunk**，于是 multi_gold 用例
    的两处证据落进**同一个块** → gold 退化为 1 条。
    **这不是回归** —— 旧版的"2 条 gold"恰恰是**碎片化**的产物（同一段被切碎成两块）；
    修复后上下文恢复完整，一处命中即拿到全部证据。
    代价是该类型在当前小语料下**失去了"多命中"的分辨力**；要恢复它需要更长的语料
    （多个 chunk）才能把两处证据放在不同块 —— 属评测集设计决策，未擅自改语料。

    ✅ **2026-09-25 T-02 已解决**：新增 8 篇语料并把每篇补到 **≥5 块**，
    multi_gold 改由这些多块文档承担（g03~g24），两处证据均落在**不同块**；
    旧 g01/g02 因原文档只有 1 块、结构上不可能满足而移除
    （原样归档在 `cases/golden_real_25cases.jsonl.bak-20260925`）。
    下面的下界也随之上提到卡片配额（paraphrase ≥40、multi_gold ≥15）。
    """
    by_kind: dict[str, int] = {}
    for c in _VALID:
        by_kind[c.get("kind", "unknown")] = by_kind.get(c.get("kind", "unknown"), 0) + 1

    assert by_kind.get("paraphrase", 0) >= 40, f"缺少改写型难例：{by_kind}"
    assert by_kind.get("multi_gold", 0) >= 15, f"缺少多 gold 用例：{by_kind}"
    multi_gold = [c for c in _VALID if c.get("kind") == "multi_gold"]
    assert all(c["gold"] for c in multi_gold), "multi_gold 用例的 gold 不得为空"
    if any(len(c["gold"]) >= 2 for c in multi_gold):
        assert all(len(c["gold"]) >= 2 for c in multi_gold), \
            "若已有用例达到 2 条 gold，其余也应达到（否则口径不一致）"


def test_golden_real_schema() -> None:
    for c in _VALID:
        assert c["id"] and c["query"], c
        assert c["corpus"] and all((Path(CORPUS_DIR) / f"{d}.pdf").exists() for d in c["corpus"]), c["id"]
        if c.get("answerable", True):
            assert c["gold"], f"{c['id']}: 可答用例必须有 gold"
            assert c["answer_sentence"], f"{c['id']}: 必须记录答案原文片段（gold 的来源）"
            for g in c["gold"]:
                assert g in c["gains"], f"{c['id']}: gold {g} 不在 gains 里"
                assert c["gains"][g] == 3, f"{c['id']}: gold {g} 的等级必须是最高的 3"
        else:
            assert c["gold"] == [], f"{c['id']}: 负样本不应有 gold"
            assert c["gains"] == {}, f"{c['id']}: 负样本不应有 gains"


@pytest.fixture(scope="module")
def real_chunks(tmp_path_factory) -> dict[str, list]:
    """走生产链路把语料解析成分块（模块级复用，避免每个用例重复解析 PDF）。

    直接调用生成脚本里的 `build_corpus()`：它用的就是 `RAGPipeline.add_document`
    （生产同款 `DocumentParser` + `Chunker`），因此"评测集标注"与"线上分块"共用一套实现。

    ★ 传入**临时目录**（2026-10-05 修）：`build_corpus` 每次都会 `write_pdf` 重新生成
    PDF，而 PyMuPDF 会刷新 `/CreationDate` 与 `/ID` ⇒ 写进仓库目录就等于
    **每次跑门禁都改写 11 个已入库文件**（实测大小相同、57 字节不同），
    `git status` 随即变脏 ⇒ 噪声极易被误提交。测试只关心"解析结果"，无需落盘到仓库。
    """
    return build_corpus(tmp_path_factory.mktemp("corpus"))


def test_gold_is_reproducible_from_real_pipeline(real_chunks: dict[str, list]) -> None:
    """核心断言：gold 必须能由真实链路**重新定位**出来（证明它不是手写的）。

    `answer_sentence` 可能是字符串（单片段）或列表（多 gold 用例要求两处证据），
    两种形态都必须逐一重新定位，且**并集与 gold 完全一致**。
    """
    for c in _VALID:
        if not c.get("answerable", True):
            continue
        chunks = real_chunks[c["doc_id"]]
        raw = c["answer_sentence"]
        spans = raw if isinstance(raw, list) else [raw]
        found: list[str] = []
        for span in spans:
            hit = [ch.id for ch in chunks if _norm(span) in _norm(ch.text)]
            assert hit, (
                f"{c['id']}: 答案片段「{span}」在 {c['doc_id']} 的真实分块中定位不到 —— "
                "语料或标注已经漂移，必须重跑 python eval/make_golden_real.py"
            )
            found.extend(h for h in hit if h not in found)
        assert found == c["gold"], f"{c['id']}: gold 与真实链路定位结果不一致（{found} != {c['gold']}）"


def test_negative_cases_produce_no_metric_keys() -> None:
    """负样本不产出指标键 → 自动被聚合排除（否则会以 0 计入，把报告拉成误导性数字）。"""
    neg = next(c for c in _VALID if not c.get("answerable"))
    row = run_case(neg, 5, PipelineFactory("fake", "fast"))

    assert row["answerable"] is False
    metric_keys = [k for k in row if "@" in k or k in ("map", "ndcg@5", "mrr@10")]
    assert metric_keys == [], f"负样本不应产出指标键，却有 {metric_keys}"
    assert row["top_ids"], "负样本仍应记录检索结果（供答案层判官判断是否诱发了错误引用）"


def test_negative_cases_excluded_from_aggregation() -> None:
    """聚合只平均"实际存在的指标键"：负样本进不了分母。"""
    pos = {"id": "p", "recall@5": 1.0, "mrr@10": 1.0}
    neg = {"id": "n", "answerable": False}
    out = summarize_results([pos, neg])

    assert out["recall@5"] == 1.0, "负样本不得把 recall@5 拉低（它根本没有 gold）"
    assert out["mrr@10"] == 1.0


def test_real_case_retrieval_returns_gold_ids_under_fake_embedder() -> None:
    """链路自检：真实文档用例在 fake 嵌入下也能检索出 gold 所在文档（保证语料真的可检索）。

    注意：fake 哈希嵌入只保证"链路通"，**不代表真实质量** —— 质量看 `--embedder real`。
    """
    case = next(c for c in _VALID if c.get("answerable") and len(c["corpus"]) == 1)
    row = run_case(case, 5, PipelineFactory("fake", "fast"))

    assert row["top_ids"], "检索结果不应为空"
    assert row["hit@5"] == 1, f"链路自检失败：{case['id']} 的 gold 未进 Top5（{row['top_ids']}）"


def test_corpus_is_small_on_disk() -> None:
    """语料体积必须保持在 KB 级 —— 防止有人把 9.7MB 的 simhei 字体重新嵌回去。

    （真实教训：用 `insert_font(fontfile=simhei.ttf)` 时单篇 PDF 就是 9.7MB，3 篇近 30MB；
    改用 PyMuPDF 内置 `china-s` 后总大小降到 20KB 级，文本抽取结果完全一致。）
    """
    total = sum(f.stat().st_size for f in CORPUS_DIR.glob("*.pdf"))
    assert total < 500 * 1024, f"语料体积异常：{total / 1024:.0f}KB（应 < 500KB）"
