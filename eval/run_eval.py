"""PaperMate · 评测运行器（A-01）

离线评测：基于 `cases/golden.jsonl` 在 RAG 流水线上跑检索，计算**检索层指标**，输出评测报告。

指标（检索层，见 `lightcore.rag.metrics`）：
  Recall@k / Precision@k / MRR@10 / nDCG@k / MAP / Hit@k + 平均检索延迟(ms)

⚠️ 关于指标可信度（读数字前必看）：
  默认用 **FakeHashEmbedder**（哈希占位向量，零依赖、秒级）——它只保证「链路正确」，
  **不代表真实语义检索质量**（哈希向量仅在 query 与目标块文本高度一致时命中）。
  要拿真实质量必须 `--embedder real`（BGE-M3，首次需加载约 2.2GB 权重）。

⚠️ 报告结构约定（改本文件前务必先看）：
  `avg_recall@5` / `avg_mrr@10` 两个**顶层键是既有契约** ——
  `papermate.services.dev_eval.collect_metrics` 读取它们做「检索基线 + 门禁判定」，
  且被 `tests/test_eval_summary.py` 守护。新增指标请放进 `metrics` 子表，**不要删这两个键**。

忠实度（faithfulness）需跨厂商 LLM + temperature=0 判定，prompt 见 `judge_prompts/faithfulness.md`，
**有意不在线执行**（避免评测依赖外部大模型）。

运行：
  python eval/run_eval.py                     # 默认 fake，秒级出结果
  python eval/run_eval.py --embedder real     # 真实嵌入（需 rag extra + 权重）
  python eval/run_eval.py --k 10 --out eval/report_real.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import judge as judge_mod  # noqa: E402  eval/ 目录在 sys.path（与 metrics 同）
from metrics import (  # noqa: E402
    average_precision,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarize_results,
)

from lightcore.rag.embedder import FakeHashEmbedder  # noqa: E402
from lightcore.rag.pipeline import RAGPipeline  # noqa: E402

CASES_DIR = Path(__file__).resolve().parent / "cases"
GOLDEN = CASES_DIR / "golden.jsonl"
GOLDEN_REAL = CASES_DIR / "golden_real.jsonl"
CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
REPORT = Path(__file__).resolve().parent / "eval_report.json"

#: 基线键固定取 @5 / @10（golden 集按 @5 设计，且既有门禁判定依赖这两个键）
_BASELINE_K = 5
_MRR_K = 10


def _build_embedder(kind: str):
    """按 `--embedder` 构造嵌入器；real 延迟导入（避免 fake 路径引入重依赖）。"""
    if kind == "real":
        from lightcore.rag.embedder import BgeM3Embedder

        return BgeM3Embedder()
    return FakeHashEmbedder()


#: `--sparse` 的取值：`auto` 读运行时设置；`on` / `off` 显式覆盖
SPARSE_CHOICES = ("auto", "on", "off")


def _resolve_sparse(choice: str) -> tuple[bool, str]:
    """把 `--sparse` 解析为 `(是否启用, 取值来源说明)`。

    - **`auto`（默认）**：读 `papermate.runtime_settings` —— 与线上**同源**
      （线上建管线用 `use_bge_sparse=rt["use_bge_sparse"]`，`services/document_service.py:781`）。
      另有 `_pipeline_signature` 也把该开关算进签名（`:840`），切开关会正确失效缓存 → 无"复用旧状态"风险。
    - **`on` / `off`**：显式覆盖，服务于**双臂对照**与"同一配置可复现复跑"。

    **为什么值得单独成一个函数、并且要回传"来源"**：修复前这里是硬编码
    `use_bge_sparse=(self.profile == "prod")`，而 docstring 声称从 `rt` 读 —— **文档与代码不一致**。
    当时线上恰好也是 `True`，所以数字是对的（**纯属巧合**）；但只要设置页一关掉稀疏向量，
    评测就会**继续按"开着"跑、且不报任何异常**，进而得出"停用稀疏不影响质量"这类
    **建立在错误前提上的结论**。回传来源是为了让每份报告自证"这一版开没开、值从哪来"，
    不必再靠读代码推断。

    ⚠️ 读配置失败**不抛异常**（评测不应因设置文件损坏而整体中断），但必须**把来源写清楚** ——
    静默回退正是"两套口径"的温床。
    """
    if choice == "on":
        return True, "--sparse on（显式覆盖）"
    if choice == "off":
        return False, "--sparse off（显式覆盖）"
    try:
        from papermate import runtime_settings

        raw = runtime_settings.get_state().get("use_bge_sparse")
        if isinstance(raw, bool):
            return raw, "runtime_settings.use_bge_sparse（与线上同源）"
        return False, f"运行时设置缺该键（值={raw!r}）→ 回退 RAGPipeline 默认 False"
    except Exception as e:  # noqa: BLE001  配置读取失败不得中断评测，但必须显式记录来源
        return False, f"运行时设置不可读（{type(e).__name__}: {e}）→ 回退默认 False"


#: `--rerank` 的取值：`auto` 读运行时设置；`on` / `off` 显式覆盖
RERANK_CHOICES = ("auto", "on", "off")

#: `on` / `off` 的档位常量（`on` 用实测推荐档：384 + fp16 → ×4.34）
#: `on`/`off` 是**显式覆盖锚点**，用于复现 T-03/T-04 时点的对照（当时线上档位 = 384）。
#: ⚠️ 线上默认档位已于 2026-10-05 上调为 **512** ⇒ `on` **不再等于当前线上口径**；
#: 要"与线上同源"请用默认的 `auto`（读 `runtime_settings`）。
#: **故意不改 `on` 的取值** —— 改了会让历史上以 `--rerank on` 跑出的报告不可复现。
_RERANK_ON = (True, 384)
_RERANK_OFF = (False, 0)


def _resolve_reranker_opts(choice: str) -> tuple[bool, int, str]:
    """把 `--rerank` 解析为 `(fp16, max_length, 来源说明)`（T-03）。

    - **`auto`（默认）**：读 `papermate.runtime_settings` 的 `rag_reranker_fp16` /
      `rag_reranker_max_length` —— 与线上**同源**（线上建管线时由
      `services/document_service.py::_get_reranker` 注入同一个键）。
    - **`on` / `off`**：显式覆盖，服务于**档位对照**与"同一配置可复现复跑"。
      `off` = `(False, 0)` = **改动前行为**（fp32 + 不限长），正是一键回退的那一档。

    **为什么必须补这一步**（2026-09-24 实测踩到）：T-03 只把档位接进了线上路径，
    评测侧 `_build_reranker()` 仍是硬编码 `CrossEncoderReranker()` → **评测在量旧档位**。
    后果是"质量看起来没变"（因为确实没变），但**提速完全测不到** ——
    与 T-00 修的 `use_bge_sparse` 硬编码是**同一类缺陷**：评测口径没跟随线上口径。

    ⚠️ 读配置失败不抛异常（评测不应因设置文件损坏而中断），但必须把来源写清楚 ——
    静默回退正是"两套口径"的温床。
    """
    if choice == "on":
        return (*_RERANK_ON, "--rerank on（显式覆盖：384 + fp16）")
    if choice == "off":
        return (*_RERANK_OFF, "--rerank off（显式覆盖：不限长 + fp32 = 改动前行为）")
    try:
        from papermate import runtime_settings

        st = runtime_settings.get_state()
        fp16 = st.get("rag_reranker_fp16")
        ml = st.get("rag_reranker_max_length")
        if isinstance(fp16, bool) and isinstance(ml, int) and not isinstance(ml, bool):
            return fp16, ml, "runtime_settings.rag_reranker_*（与线上同源）"
        return (*_RERANK_OFF, f"运行时设置键缺失/类型异常（fp16={fp16!r}, ml={ml!r}）→ 回退改动前行为")
    except Exception as e:  # noqa: BLE001  配置读取失败不得中断评测，但必须显式记录来源
        return (*_RERANK_OFF, f"运行时设置不可读（{type(e).__name__}: {e}）→ 回退改动前行为")


def _resolve_cases_path(path: Path | str | None) -> Path:
    """把 `--cases` 入参解析为绝对路径（与 `load_cases` 同一口径，避免两处漂移）。"""
    f = Path(path) if path else GOLDEN
    return f if f.is_absolute() else (Path(__file__).resolve().parent / f)


def _cases_digest(path: Path | str | None) -> str | None:
    """用例集内容指纹（sha256 前 12 位）。

    写进报告供 `dev_eval` 做「报告是否已过时」判定：**用例集变了但没重跑评测**时，
    旧报告的数字不再代表当前评测集，门禁必须能发现（而非默默沿用）。
    """
    try:
        return hashlib.sha256(_resolve_cases_path(path).read_bytes()).hexdigest()[:12]
    except Exception:  # noqa: BLE001 文件缺失/不可读 → 记录为无指纹，由其它判据兜底
        return None


def _pipeline_digest() -> str:
    """检索链路关键源码的内容指纹。

    实现见 `lightcore.rag.provenance`（**单一来源**）：`dev_eval` 判"报告是否过时"用的是同一个
    函数，两处各写一份清单必然漂移。
    """
    from lightcore.rag import provenance

    return provenance.digest()


def load_cases(path: Path | str | None = None) -> list[dict]:
    """读取用例集（JSONL）。

    两种用例形态（可混用）：
      - **内联**（老 `golden.jsonl`）：`docs = {doc_id: [分块文本, ...]}`，直接喂文本；
      - **真实文档**（`golden_real.jsonl`，L0-2）：`corpus = [doc_id, ...]`，
        从 `eval/corpus/{doc_id}.pdf` 走**生产链路**（PDF 解析 + 语义分块）加载。
    """
    f = _resolve_cases_path(path)
    rows: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class PipelineFactory:
    """按指定 profile 构建评测管线，并**全评测共享** embedder / reranker 单例。

    对齐线上需要**三个轴**同时成立（线上建管线见 `services/document_service.py:776-793`）：

    | 轴 | 线上取值 | 本类如何对齐 |
    |---|---|---|
    | reranker | `CrossEncoderReranker` | `profile="prod"` → CrossEncoder；否则 ScoreReranker |
    | tokenize | `jieba_tokenize` | `profile="prod"` → jieba；否则字符级 |
    | `use_bge_sparse` | `rt["use_bge_sparse"]`（**运行时设置**） | `use_bge_sparse=` 参数；由 `main` 按 `--sparse` 决定来源 |

    ⚠️ **第三个轴此前是硬编码**（原代码 `use_bge_sparse=(self.profile == "prod")`），
    而本 docstring 却声称与线上同源 —— **文档与代码不一致**。
    当时线上恰好也是 `True`，数字因此是对的（**纯属巧合**）；但只要有人在设置页关掉稀疏向量，
    评测会**继续按"开着"跑且不报任何异常**，从而得出"停用稀疏不影响质量"这类
    **建立在错误前提上的结论**。T-00 步 1 修的就是它（见 `_resolve_sparse`）。

    另一个关键决定：**模型必须共享单例**。real 模式下 BGE-M3 ≈2.2GB、CrossEncoder ≈2.2GB；
    本评测有 7 组语料组合，各建一份会在 8GB 显卡上直接 OOM。
    生产侧（`document_service._get_embedder/_get_reranker`）同样是单例缓存，理由相同。
    """

    def __init__(self, embedder_kind: str = "fake", profile: str = "fast",
                 use_bge_sparse: bool | None = None, sparse_source: str = "",
                 reranker_fp16: bool | None = None, reranker_max_length: int | None = None,
                 rerank_source: str = "", merge: str = "single") -> None:
        """`use_bge_sparse=None` 表示**沿用旧行为**（`profile == "prod"`）。

        保留 None 分支是为了让既有调用方（`eval/test_golden_real.py`、
        `tests/test_eval_judge.py` 以位置参数构造）语义不变；`main()` 一律显式传解析后的值。

        `merge`（T-04）：多文档用例的合并路径 ——
        - `"single"`（默认 = 改动前行为）：多篇 `add_document` 进**同一个** `RAGPipeline`，
          全局混合检索 + 重排（候选是**一个池子**里全局排序出来的）；
        - `"composite"`：**与线上同源**（`chat_service._build_agent`）—— 每篇独立
          `RAGPipeline` → 各取 top_k → `CompositeRAGPipeline` 合并按分数截断。
          两条路径的候选生成方式不同（合并策略本身会引入损失），**评测哪个就必须跑哪个** ——
          这是 T-00（sparse）/ T-03（rerank 档位）之后第三处「评测与线上不同源」。

        T-03（2026-09-24）新增 `reranker_*` 三项同理：`None` = 沿用**改动前行为**
        （fp32 + 不限长），只有 `main()` 才会显式传入解析后的档位。
        """
        self.embedder_kind = embedder_kind
        self.profile = profile
        self.use_bge_sparse = (
            (profile == "prod") if use_bge_sparse is None else bool(use_bge_sparse)
        )
        #: 该取值的**来源**（写进报告，供事后审计"这一版评测到底开没开稀疏"）
        self.sparse_source = sparse_source or (
            "profile==prod（旧行为，未显式指定）" if use_bge_sparse is None else ""
        )
        # T-03：重排档位（None = 改动前行为：fp32 + 不限长）
        self.reranker_fp16 = bool(reranker_fp16) if reranker_fp16 is not None else False
        self.reranker_max_length = reranker_max_length if reranker_max_length else None
        self.rerank_source = rerank_source or (
            "未指定（改动前行为：fp32 + 不限长）" if reranker_fp16 is None else ""
        )
        self._embedder = _build_embedder(embedder_kind)
        self._reranker = self._build_reranker()
        self._tokenize = self._build_tokenizer()
        self.merge = merge
        # T-04+：检索 top_k（0 = 沿用 `max(k, _MRR_K)`，即改动前行为）
        self.retrieve_top_k = 0
        # T-04+：查询预处理策略（`"sections"` = 与线上同源；`"full"` = 改动前基线）。
        # 由 `main()` 解析后注入；默认 `"sections"` 与 `CompositeRAGPipeline` 的默认值一致。
        self.query_rewrite_mode = "sections"
        self.query_prep: Callable[[str], str] | None = None
        # 两套缓存**分开**：同 key 在两种合并模式下是不同对象，混用会静默返回错路径的结果
        self._pipelines: dict[tuple, RAGPipeline] = {}
        self._composite_cache: dict[tuple, object] = {}

    def _build_reranker(self):
        """real → CrossEncoderReranker（与线上一致）；fake / fast → None（用默认 ScoreReranker）。

        T-03：档位由 `main()` 解析后注入（`reranker_fp16` / `reranker_max_length`）。
        不传时用 `None/False` = **改动前行为**（fp32 + 不限长），保证既有调用方零变化。
        """
        if self.embedder_kind != "real" or self.profile != "prod":
            return None
        try:
            from lightcore.rag.rerank import CrossEncoderReranker

            return CrossEncoderReranker(max_length=self.reranker_max_length,
                                        fp16=self.reranker_fp16)
        except Exception as e:  # noqa: BLE001  权重缺失等 → 降级但必须显式说出来
            print(f"⚠️ CrossEncoderReranker 初始化失败，降级 ScoreReranker：{e}")
            return None

    def _build_tokenizer(self):
        """prod → jieba 按词切分（线上默认）；缺失时回退字符级并提示。"""
        from lightcore.rag.bm25 import _char_tokenize

        if self.profile != "prod":
            return _char_tokenize
        try:
            from lightcore.rag.bm25 import jieba_tokenize

            return jieba_tokenize
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ jieba 不可用，BM25 回退字符级切分：{e}")
            return _char_tokenize

    def for_case(self, case: dict) -> RAGPipeline:
        """取该 case 对应的管线。

        - 内联用例（老 golden）：`docs` 各不相同 → 逐 case 新建（共用会把上一题语料带进来，污染指标）；
        - 真实文档用例：按 `corpus` 组合缓存复用（PDF 解析 + 编码成本高；query 缓存本身按管线隔离）。
        """
        if "corpus" in case:
            key = tuple(case["corpus"])
            if self.merge == "composite":
                # T-04：与线上同源 —— 每篇独立管线（共享 embedder/reranker 单例，显存不翻倍），
                # 由 CompositeRAGPipeline 做「各取 top_k → 合并截断」。
                # 单篇（len==1）也走 Composite：线上单文档会话就是 len==1 的 Composite，行为等价。
                if key not in self._composite_cache:
                    from papermate.services.chat_service import CompositeRAGPipeline

                    pipes: list[RAGPipeline] = []
                    for doc_id in key:
                        pdf = CORPUS_DIR / f"{doc_id}.pdf"
                        if not pdf.exists():
                            raise FileNotFoundError(
                                f"语料缺失：{pdf}（先运行 python eval/make_golden_real.py 生成）"
                            )
                        pipe = RAGPipeline(
                            embedder=self._embedder, reranker=self._reranker,
                            use_bge_sparse=self.use_bge_sparse, tokenize=self._tokenize,
                        )
                        pipe.add_document(pdf, doc_id=doc_id)
                        pipes.append(pipe)
                    # doc_ids 传 chunk 的 doc 归属（与线上 loaded_doc_ids 同语义）
                    # T-04+：注入查询预处理 —— 默认与线上同源（`expand_sections`）；
                    # `--query-rewrite full` 传 `rewrite_query` 以复现改动前基线。
                    self._composite_cache[key] = CompositeRAGPipeline(
                        pipes, doc_ids=list(key), query_prep=self.query_prep
                    )
                return self._composite_cache[key]
            if key not in self._pipelines:
                pipe = RAGPipeline(
                    embedder=self._embedder, reranker=self._reranker,
                    use_bge_sparse=self.use_bge_sparse, tokenize=self._tokenize,
                )
                for doc_id in key:
                    pdf = CORPUS_DIR / f"{doc_id}.pdf"
                    if not pdf.exists():
                        raise FileNotFoundError(
                            f"语料缺失：{pdf}（先运行 python eval/make_golden_real.py 生成）"
                        )
                    pipe.add_document(pdf, doc_id=doc_id)
                self._pipelines[key] = pipe
            return self._pipelines[key]

        pipe = RAGPipeline(
            embedder=self._embedder, reranker=self._reranker,
            use_bge_sparse=self.use_bge_sparse, tokenize=self._tokenize,
        )
        for doc_id, texts in case["docs"].items():
            pipe.add_document(texts, doc_id=doc_id)
        return pipe


def run_case(case: dict, k: int, factory: PipelineFactory,
             context_sink: dict[str, list[str]] | None = None) -> dict:
    """单 case 评测：取管线 → 检索 → 计算各项指标。

    负样本（`answerable: false`，语料里确实没有答案）**不产出任何指标键**：
    没有 gold 时 recall/nDCG 无意义；`summarize_results` 只平均"实际存在的指标键"，
    因此它们会被自动排除在聚合之外（服务于 L0-3 的答案层判官：是否如实拒答）。

    `context_sink`：判官 pass 需要**检索到的原文**（判"答案是否忠于证据"），
    在此顺手带出，避免整轮评测重复检索一遍（25 条 × 150ms + GPU 不是零成本）。
    """
    pipeline = factory.for_case(case)
    t0 = time.perf_counter()
    # T-04+：**检索时的 top_k** 与**指标位次 k** 分开。
    # 默认沿用 `max(k, _MRR_K)`（改动前行为，保证既有数字不变）；
    # 显式给 `--retrieve-top-k` 时用它 —— 用来让"每篇各取 top_k"进入**真截断区**：
    # 评测语料每篇仅 5~6 块，而默认 top_k=10 → 各取 10 = 全取 → 合并策略空转（测不出差异）。
    # 取 3（对齐线上 `pdf_extract` 的 top_k=3）即可让每篇真正被截断。
    top_k = factory.retrieve_top_k or max(k, _MRR_K)
    retrieved = pipeline.query(case["query"], top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000
    if context_sink is not None:
        context_sink[case["id"]] = [r.text for r in retrieved[:k]]

    if not case.get("answerable", True):
        return {
            "id": case["id"],
            "answerable": False,
            "latency_ms": round(latency_ms, 2),
            "gold": [],
            "top_ids": [r.id for r in retrieved[:k]],
            "reference": case.get("reference", ""),
        }

    gold = case["gold"]
    # 分级相关性（可选）：gains = {chunk_id: 3/2/1}，3=直接证据 2=强相关 1=弱相关
    # 未标注（老标注集）时为 None → nDCG 退化为二值版本
    gains = case.get("gains") or None

    row: dict = {
        "id": case["id"],
        "answerable": True,
        f"recall@{k}": round(recall_at_k(retrieved, gold, k), 4),
        f"precision@{k}": round(precision_at_k(retrieved, gold, k), 4),
        f"ndcg@{k}": round(ndcg_at_k(retrieved, gold, k, gains=gains), 4),
        f"hit@{k}": hit_at_k(retrieved, gold, k),
        "map": round(average_precision(retrieved, gold, k), 4),
        "mrr@10": round(mrr_at_k(retrieved, gold, _MRR_K), 4),
        "latency_ms": round(latency_ms, 2),
        "gold": gold,
        "top_ids": [r.id for r in retrieved[:k]],
    }
    # 基线键固定 @5：k≠5 时补出，保证 `recall@5` 在任何 --k 下都存在（既有契约）
    if k != _BASELINE_K:
        row["recall@5"] = round(recall_at_k(retrieved, gold, _BASELINE_K), 4)
        row["precision@5"] = round(precision_at_k(retrieved, gold, _BASELINE_K), 4)
        row["ndcg@5"] = round(ndcg_at_k(retrieved, gold, _BASELINE_K, gains=gains), 4)
        row["hit@5"] = hit_at_k(retrieved, gold, _BASELINE_K)
    return row


# ---------------------------------------------------------------- L0-3 答案层判官


def _preferred_model_name() -> str:
    """当前"用户实际在用的默认模型名"（小写），**按线上口径**：运行时设置 → 环境变量。

    与 `chat_service._resolve_raw_model` 的顺序一致：只读 `.env` 会漏掉设置页里改过的值
    （本项目实测过 `PAPERMATE_DEFAULT_MODEL` 被 pset 完全覆盖的情形）。
    """
    try:
        from papermate import runtime_settings

        name = (runtime_settings.get_state().get("default_model") or "").strip()
        if name:
            return name.lower()
    except Exception:  # noqa: BLE001
        pass
    return (os.getenv("PAPERMATE_DEFAULT_MODEL") or "").strip().lower()


def _auto_answer_provider() -> str:
    """未显式指定时，按「默认模型属于哪家」推断主模型厂商，退化到「哪家配了密钥」。

    密钥/默认模型都按**与线上同源**的方式解析（`judge.resolve_api_key` + `_preferred_model_name`）：
    原先只读 `os.getenv` ⇒ 用户把 key 存在设置页时，这里会直接抛「未配置任何模型密钥」，
    而实际上两家都配好了（2026-09-25 实测踩到）。
    """
    preferred = _preferred_model_name()
    for p in ("deepseek", "qwen"):
        if p in preferred and judge_mod.resolve_api_key(p)[0]:
            return p
    for p in ("deepseek", "qwen"):
        if judge_mod.resolve_api_key(p)[0]:
            return p
    raise judge_mod.JudgeUnavailable(
        "未配置任何模型密钥（运行时设置 data/provider_settings.json 的 providers.*.api_key "
        "与环境变量 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY 均为空）。"
        "`--judge` 需要真实密钥才能产出判分；如需只跑检索，请去掉 `--judge`。"
    )


def _default_model_for(provider: str) -> str | None:
    """若**线上默认模型**属于该厂商，则返回它 —— 让答案层评测测的正是用户实际用的模型。

    不做这一步，答案层会用端点默认值（`qwen3.7-flash`），而线上跑的是 `qwen3.6-flash`
    ⇒ 评测与线上不是同一个模型，"答案质量"数字回答的是另一个问题（同 T-00/T-03 的家族缺陷）。
    """
    from lightcore.llm.router import MODEL_TO_PROVIDER

    name = _preferred_model_name()
    return name if name and MODEL_TO_PROVIDER.get(name) == provider else None


def _resolve_providers(args) -> tuple[str, str]:
    """解析 (答案厂商, 判官厂商)；**默认强制跨厂商**（同源自评会系统性偏乐观）。

    `--allow-same-provider-judge` 是**受限放宽**（2026-09-27 新增）。
    真实触发场景：异源厂商**整体不可用**（qwen 免费额度耗尽）时，强制跨厂商
    ⇒ 两家配对里**必然有一方是 qwen** ⇒ 基线永远补不齐。

    放宽需**同时**满足两条，否则照旧拒绝：
      ① 显式加开关；② 答案与判官**模型名不同**（同厂商同模型 = 真自评，任何情况都不接受）。
    报告会写 `cross_provider: false` 标注 —— 该数字**不得当作跨厂商结论**使用。
    """
    answer = args.answer_provider
    if answer == "auto":
        answer = _auto_answer_provider()
    judge_p = args.judge_provider
    if judge_p == "auto":
        judge_p = judge_mod.pick_judge_provider(answer)
    elif judge_p == answer:
        a_model = getattr(args, "answer_model", None)
        j_model = getattr(args, "judge_model", None)
        if a_model and j_model and a_model == j_model:
            raise judge_mod.JudgeUnavailable(
                f"答案与判官是同一模型（{a_model}）= 真自评，不接受"
                f"（即便加了 --allow-same-provider-judge）。请换一个模型。"
            )
        if not getattr(args, "allow_same_provider_judge", False):
            raise judge_mod.JudgeUnavailable(
                f"判官与主模型同厂商（{judge_p}）—— 同源自评会系统性偏乐观。"
                f"请换一家作判官，或用 `--judge-provider auto` 自动跨厂商。"
                f"（若异源厂商整体不可用，可加 --allow-same-provider-judge 放宽，"
                f"但必须让答案与判官使用**不同模型**；报告会标注同源。）"
            )
    return answer, judge_p


def _stride_indices(n: int, q: int) -> list[int]:
    """在 `range(n)` 上取 q 个**等距**下标（首尾都取到，结果去重保序）。

    用等距而非随机：判官评测要**可复现**（同一个 `--judge-limit` 必须给出同一组用例），
    随机抽样会让两次运行的数字不可比，也就无法区分"质量变了"与"抽到的题变了"。
    """
    if q <= 0:
        return []
    if q >= n:
        return list(range(n))
    idx = sorted({round(i * (n - 1) / (q - 1)) for i in range(q)}) if q > 1 else [0]
    i = 0
    while len(idx) < q and i < n:          # 取整撞车时按顺序补齐，保证恰好 q 个
        if i not in idx:
            idx.append(i)
            idx.sort()
        i += 1
    return idx[:q]


def _plan_judge_sample(cases: list[dict], limit: int) -> tuple[list[dict], dict]:
    """`--judge-limit` 的**确定性分层抽样**：按 `kind` 分层 + 层内等距。

    为什么必须分层：`faithfulness` 的分母只含可答用例、`abstention` 只含负样本 ——
    若按整体顺序截前 N 条，负样本（在文件末尾）会**一条都进不了样本**，
    `abstention` 直接变成 `null`，看起来像"未评测"而不是"没抽到"。
    层内等距则保证同一层内覆盖均匀（文件里的同类用例是按内容分组的，截前缀会只覆盖一个主题）。

    返回 `(抽样后的用例, 抽样口径说明)`；`limit<=0` 或超过总数时为**全量**（且如实标注）。
    """
    total = len(cases)
    if limit <= 0 or limit >= total:
        plan = {
            "strategy": "full",
            "requested": limit,
            "n": total,
            "total_cases": total,
            "note": "未抽样：判分覆盖全部用例",
        }
        return list(cases), plan

    groups: dict[str, list[dict]] = {}
    for c in cases:                                   # 保序：dict 保插入顺序
        # `or "unknown"` 而不是 `.get(kind, "unknown")`：键存在但值为 None 时后者会返回 None，
        # 后续按 kind 排序就会 `None < str` 崩掉（本分支由 tests/test_eval_judge.py 反向断言逼出）。
        groups.setdefault(c.get("kind") or "unknown", []).append(c)

    # 按层大小比例分配名额，**每层至少 1**（保证五类都进样本），再用最大余数法补齐
    quota: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for kind, items in groups.items():
        exact = limit * len(items) / total
        quota[kind] = max(1, min(int(exact), len(items)))
        remainders.append((exact - int(exact), kind))
    while sum(quota.values()) < limit:
        grew = False
        for _, kind in sorted(remainders, key=lambda kv: (-kv[0], kv[1])):
            if sum(quota.values()) >= limit:
                break
            if quota[kind] < len(groups[kind]):
                quota[kind] += 1
                grew = True
        if not grew:
            break
    while sum(quota.values()) > limit:                # 层数多于名额时从最大层回收
        for kind in sorted(quota, key=lambda k: (-quota[k], k)):
            if sum(quota.values()) <= limit:
                break
            if quota[kind] > 1:
                quota[kind] -= 1

    picked: list[dict] = []
    by_kind: dict[str, dict] = {}
    for kind, items in groups.items():
        q = quota[kind]
        chosen = [items[i] for i in _stride_indices(len(items), q)]
        picked.extend(chosen)
        by_kind[kind] = {"sampled": len(chosen), "available": len(items)}

    order = {c["id"]: i for i, c in enumerate(cases)}
    picked.sort(key=lambda c: order.get(c["id"], 0))  # 恢复原始顺序：输出稳定、便于逐条比对

    plan = {
        "strategy": "stratified_stride",
        "requested": limit,
        "n": len(picked),
        "total_cases": total,
        "by_kind": by_kind,
        "ids": [c["id"] for c in picked],
        "note": "**抽样判分**：判官指标只在抽到的子集上计算，不可与全量数字并列比较",
    }
    return picked, plan


async def _judge_all(cases: list[dict], contexts: dict[str, list[str]], answer_llm, judge, args) -> dict:
    """逐条：生成答案 → 判分（串行，避免打爆厂商限流；判官要可复现）。

    ★ `--judge-dump` 的**根因动机**（2026-09-25 实测踩到）：
    报告只存 `answer_preview`（200 字符）且**不存检索上下文** —— 事后想补一张"人工标注表"时
    只能**重新生成答案**；实测重新生成的 40 条里 **22 条与判官当初评的答案实质不同**
    （`qwen3.6-flash` 在 `temperature=0` 下**并不确定**）⇒ 表里「判官结论」与「答案」不是同一条
    → **κ 无法成立**。修法：**谁产生结论，谁把当时的原始输入一并落盘**（边跑边 append，崩了也有部分数据）。
    """
    dump = getattr(args, "judge_dump", None)
    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        Path(dump).write_text("", encoding="utf-8")   # 新一轮从头写，避免与上一轮混在一起
    faith: list = []
    abst: list = []
    details: list[dict] = []
    for case in cases:
        cid = case["id"]
        context = "\n\n".join(contexts.get(cid) or []) or "（检索无结果）"
        answer, gen_err = await judge_mod.generate_answer(
            answer_llm, question=case["query"], context=context, timeout_s=args.judge_timeout
        )
        answerable = bool(case.get("answerable", True))
        if gen_err:
            verdict = judge_mod.JudgeVerdict(ok=False, value=None, error=f"生成失败：{gen_err}")
        elif answerable:
            verdict = await judge.faithfulness(context=context, answer=answer)
        else:
            verdict = await judge.abstention(context=context, question=case["query"], answer=answer)
        (faith if answerable else abst).append(verdict)

        mark = "—" if not verdict.ok else ("✅" if verdict.value else "❌")
        print(f"  {cid:<5} {'faithful ' if answerable else 'abstain  '} {mark} "
              f"{(verdict.error or verdict.reason)[:70]}")
        details.append({
            "id": cid,
            "kind": "faithfulness" if answerable else "abstention",
            "ok": verdict.ok,
            "value": verdict.value,
            "reason": verdict.reason,
            "error": verdict.error,
            "answer_preview": answer[:200],
        })
        if dump:
            # 落盘**当时真实用的**（query, context, answer, 结论）—— 这是"事后能标注"的唯一保证
            with open(dump, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({
                    "id": cid, "kind": case.get("kind", ""), "answerable": answerable,
                    "query": case["query"], "context": context, "answer": answer,
                    "judge_value": verdict.value, "judge_ok": verdict.ok,
                    "judge_reason": verdict.reason, "judge_error": verdict.error,
                    "gen_error": gen_err,
                }, ensure_ascii=False) + "\n")
    same_provider = answer_llm.endpoint.name == judge.provider
    return {
        "enabled": True,
        "answer": {"provider": answer_llm.endpoint.name, "model": answer_llm.model_name},
        "judge": {"provider": judge.provider, "model": judge.model_name},
        "temperature": 0.0,
        # 2026-09-27：同厂商配对是**受限放宽**（--allow-same-provider-judge），必须显式标注，
        # 否则后续读者会把一个"可能偏乐观"的数字当成跨厂商结论使用。
        "cross_provider": not same_provider,
        "warning": (
            "⚠️ 答案与判官为**同厂商**（跨厂商限制已被显式放宽）⇒ 指标可能偏乐观，"
            "不得当作跨厂商结论使用。" if same_provider else None
        ),
        "faithfulness": judge_mod.aggregate(faith, "faithfulness"),
        "abstention": judge_mod.aggregate(abst, "abstention"),
        "details": details,
    }


def _run_judge_pass(cases: list[dict], contexts: dict[str, list[str]], args) -> dict:
    """跑答案层判分并返回报告的 `judge` 块。

    失败语义（与 `dev_eval` 的门禁同一取向）：**没有结论好过假结论** ——
    缺密钥 / 同厂商判官 → 抛 `JudgeUnavailable`，由 `main` 转成非零退出，
    绝不写出"报告看起来正常、只是没有判官数字"的东西。
    """
    answer_p, judge_p = _resolve_providers(args)
    # 答案模型默认取**线上默认模型**（属于该厂商时），而非端点默认值 —— 评测要测用户实际用的那个
    answer_model = args.answer_model or _default_model_for(answer_p)
    answer_llm = judge_mod.build_llm(answer_p, model=answer_model, timeout=args.judge_timeout)
    judge_llm = judge_mod.build_llm(judge_p, model=args.judge_model, timeout=args.judge_timeout)
    judge_mod.ensure_available(answer_llm)   # 先校验密钥：否则跑完检索 + GPU 才发现白跑
    judge_mod.ensure_available(judge_llm)
    judge = judge_mod.Judge(judge_llm, timeout_s=args.judge_timeout)

    _cross = answer_p != judge_p
    _tag = "跨厂商 ✓" if _cross else "⚠️ 同厂商（已显式放宽 ⇒ 指标可能偏乐观，勿当跨厂商结论）"
    print(f"判官：答案由 {answer_llm.model_name}（{answer_p}）生成，"
          f"由 {judge.model_name}（{judge_p}）评分（temperature=0，{_tag}）")
    print(f"      答案模型来源：{'显式 --answer-model' if args.answer_model else '线上默认模型（同源）'}")
    print(f"      密钥来源：答案 {getattr(answer_llm, 'key_source', '?')}｜"
          f"判官 {getattr(judge_llm, 'key_source', '?')}")

    picked, plan = _plan_judge_sample(cases, args.judge_limit)
    if plan["strategy"] != "full":
        detail = " · ".join(f"{k} {v['sampled']}/{v['available']}" for k, v in plan["by_kind"].items())
        print(f"抽样：{plan['n']}/{plan['total_cases']} 例（{detail}）— 判官指标只在子集上计算")
        print(f"      抽样 id：{','.join(plan['ids'])}")

    block = asyncio.run(_judge_all(picked, contexts, answer_llm, judge, args))
    block["sample"] = plan
    # 落盘路径也进报告：事后要重建标注表时，得知道去哪个文件取"判分当时的原始输入"
    block["dump"] = getattr(args, "judge_dump", None)
    # 密钥来源也要进报告：否则事后无法回答"这次的判官到底用的是哪一层配置的 key"
    block["key_source"] = {
        "answer": getattr(answer_llm, "key_source", "?"),
        "judge": getattr(judge_llm, "key_source", "?"),
    }
    return block


def _precision_cap(rows: list[dict], k: int, cases: list[dict]) -> dict | None:
    """`precision@k` 的**理论上限**与饱和度（T-04-B）。

    上限（逐例先算再平均）= ``mean_i( min(|gold_i|, n_i) / n_i )``，其中
    ``n_i = min(k, 实际返回数_i)`` —— **分母必须与 `precision_at_k` 同源**：后者明确
    「分母取 `min(k, 实际返回数)`」（回答"塞进上下文的有多干净"），所以当检索
    **返回数 < k** 时，分子最多只能命中 ``min(|gold|, 返回数)`` 个。

    ⚠️ **2026-09-30 踩坑（本函数的回归史）**：分母曾写死 ``k``
    （``mean(min(|gold|,k))/k``）。单文档语料每篇仅 5~6 块、而 k=10 ⇒ 真实分母是 5 不是 10
    ⇒ 上限被**系统性低估一倍** ⇒ 饱和度算到 **1.96 (>1)**，"理论上限被突破"这一自相矛盾的
    结果反过来被 `test_precision_cap_is_reported_and_saturated` 抓住。
    教训：**上限的分母必须跟着实测的分母走**，不能假定"总是返回 k 个"。

    这是「指标有没有分辨力」的护栏：饱和度接近 1.0 时，precision 的涨跌不再携带信息，
    应改用首中率（MRR/hit）或排序质量（nDCG）来判断。
    """
    gold_of = {c["id"]: len(c.get("gold") or []) for c in cases}
    key = f"precision@{k}"
    caps: list[float] = []
    observed_vals: list[float] = []
    denominators: list[int] = []
    for r in rows:
        p = r.get(key)
        g = gold_of.get(r["id"])
        if p is None or not g:
            continue
        # ★ 与 precision_at_k 同源：分母 = min(k, 实际返回数)；`top_ids` 已按 k 截断
        n = min(k, len(r.get("top_ids") or []))
        if n <= 0:
            continue
        caps.append(min(g, n) / n)
        observed_vals.append(p)
        denominators.append(n)
    if not caps:
        return None
    cap = statistics.fmean(caps)
    observed = statistics.fmean(observed_vals)
    return {
        "k": k,
        # 实测平均分母：一眼看出"是否真的返回了 k 个"（n<k 时写死 k 的口径必然失真）
        "k_effective": round(statistics.fmean(denominators), 2),
        "cap": round(cap, 4),
        "observed": round(observed, 4),
        "saturation": round(observed / cap, 4) if cap else None,
    }


#: 判定"贴顶"的容差：浮点指标达到 1-1e-3 视作已满，饱和度 ≥0.995 视作已贴上限。
_SATURATION_EPS = 1e-3
_SATURATION_CAP_EPS = 0.995


def _discriminative_flags(metrics: dict, cap: dict | None, k: int) -> dict:
    """O-14（2026-10-01）：逐指标标注"是否还有分辨力（headroom）"。

    判据（两条都基于"能不能变好"，而不是"数值好不好"）：
    1. **召回类**（`recall@{k}` / `hit@{k}`）：值已达 1 ⇒ 分子已无处可涨 ⇒ 无分辨力；
    2. **精确率**（`precision@{k}`）：`observed / cap` 已达上限（precision_cap.saturation）
       ⇒ 再好的检索也只能到 cap ⇒ 无分辨力；
    3. **排序质量类**（`ndcg@{k}` / `mrr@{*}` / `map`）：上限恒为 1.0，只要 < 1 就仍有空间。

    为什么必须写在报告里：语料比真实库小，** recall/precision 已结构性饱和**。
    若只看这几个数，"任何等价改动都像没效果"，也让真正的优化**无从证明**。
    `discriminative_metrics` 给出"当前还能用来判优劣的指标"，避免对着贴顶指标做决策。
    """
    flags: dict[str, bool] = {}
    headroom: dict[str, float] = {}

    def _mark(name: str, has_headroom: bool, room: float) -> None:
        flags[name] = has_headroom
        headroom[name] = round(max(0.0, room), 4)

    # ① 召回类：值是否已达上限 1.0
    for key in (f"recall@{k}", f"hit@{k}"):
        v = metrics.get(key)
        if v is None:
            continue
        _mark(key, (1.0 - float(v)) > _SATURATION_EPS, 1.0 - float(v))

    # ② 精确率：是否已达**理论上限**（不是 1.0，因为受 |gold| 限制）
    if cap:
        sat = cap.get("saturation")
        if sat is not None:
            key = f"precision@{k}"
            room = 1.0 - float(sat)
            _mark(key, room > (1.0 - _SATURATION_CAP_EPS), round((cap.get("cap") or 0.0) - (cap.get("observed") or 0.0), 4))

    # ③ 排序质量类：上限恒为 1
    for key, v in metrics.items():
        if key.startswith("ndcg@") or key.startswith("mrr@") or key == "map":
            if isinstance(v, (int, float)) and key not in flags:
                _mark(key, (1.0 - float(v)) > _SATURATION_EPS, 1.0 - float(v))

    if not flags:
        return {}
    return {
        # 逐指标是否"还能上涨"（True = 有分辨力，可用来判断改动优劣）
        "has_headroom": flags,
        # 逐指标的剩余空间（召回/精度类为其缺口的绝对值；precision 为 cap-observed）
        "headroom": headroom,
        # 结论列表：可直接阅读"现在该看哪些指标"
        "discriminative_metrics": sorted(k for k, ok in flags.items() if ok),
        "saturated_metrics": sorted(k for k, ok in flags.items() if not ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperMate 检索评测（A-01）")
    parser.add_argument("--embedder", choices=["fake", "real"], default="fake",
                        help="fake=哈希占位（秒级，仅验证链路）；real=BGE-M3（真实质量）")
    parser.add_argument("--k", type=int, default=5, help="@k 的取值（默认 5）")
    parser.add_argument("--retrieve-top-k", type=int, default=0,
                        help="**检索时**的 top_k（默认 0 = 沿用 max(k,_MRR_K)，即改动前行为）。"
                             "T-04+：评测语料每篇仅 5~6 块而默认 top_k=10 → 「每篇各取 top_k」= 全取"
                             " → 合并策略空转；显式取小值可进入**真截断区**，才测得到合并/配额效应")
    parser.add_argument("--out", default=None, help=f"报告输出路径（默认 {REPORT.name}）")
    parser.add_argument("--cases", default=None,
                        help="用例集路径（默认 cases/golden.jsonl；真实文档集用 cases/golden_real.jsonl）")
    parser.add_argument("--profile", choices=["prod", "fast"], default="prod",
                        help="prod=对齐线上（CrossEncoder 重排 + jieba 分词）；"
                             "fast=ScoreReranker + 字符分词（仅链路自检）")
    # 稀疏向量单独成轴：它由**运行时设置**决定（与线上同源），不再挂在 --profile 上。
    # 修复前它是硬编码 (profile=="prod")，与 docstring 声称的"读 rt"不符 —— 见 _resolve_sparse。
    parser.add_argument("--sparse", choices=list(SPARSE_CHOICES), default="auto",
                        help="稀疏向量（BGE-M3 lexical）：auto=读运行时设置（与线上同源，默认）；"
                             "on/off=显式覆盖（用于双臂对照）")
    parser.add_argument("--rerank", choices=list(RERANK_CHOICES), default="auto",
                        help="重排加速档位（T-03）：auto=读运行时设置（与线上同源，默认）；"
                             "on=384+fp16（实测推荐档）；off=不限长+fp32（= 改动前行为，用于回退对照）")
    parser.add_argument("--merge", choices=["single", "composite"], default="single",
                        help="多文档用例的合并路径（T-04）：single=单管线全局排序（= 改动前评测口径）；"
                             "composite=与线上同源（每篇各取 top_k → Composite 合并截断）。"
                             "测 Composite 的合并策略必须用 composite")
    parser.add_argument("--query-rewrite", choices=["v2", "sections", "legacy"], default="v2",
                        dest="query_rewrite",
                        help="composite 模式的查询预处理（T-04+）："
                             "v2=④ 加固版 rewrite_query（jieba 词边界 + 保留疑问词，= 线上默认）；"
                             "sections=最保守（只做 /命令 剥离 + 章节展开，一个词都不删）；"
                             "legacy=加固前基线（裸子串替换，会破坏 query：'目的是什么'→'目是'）")
    # ↓ L0-3 答案层判官（faithfulness + abstention）
    parser.add_argument("--judge", action="store_true",
                        help="启用答案层判分：真实 LLM 生成答案 + 跨厂商判官打分（需密钥；"
                             "报告将含 faithfulness 与 abstention 两个数字）")
    parser.add_argument("--answer-provider", choices=["auto", "deepseek", "qwen"], default="auto",
                        help="生成答案的厂商（auto=按已配密钥推断）；需与 --judge-provider 不同源")
    parser.add_argument("--judge-provider", choices=["auto", "deepseek", "qwen"], default="auto",
                        help="判官厂商（auto=自动选与答案模型**不同源**的那家）")
    parser.add_argument("--allow-same-provider-judge", action="store_true",
                        help="受限放宽同源限制（2026-09-27）：异源厂商整体不可用"
                             "（如 qwen 额度耗尽）时允许答案与判官同厂商 —— 但**必须模型名不同**，"
                             "且报告会写 `cross_provider: false`，该数字不得当作跨厂商结论使用。")
    parser.add_argument("--answer-model", default=None, help="覆盖答案模型名（默认用厂商默认模型）")
    parser.add_argument("--judge-model", default=None, help="覆盖判官模型名（默认用厂商默认模型）")
    parser.add_argument("--judge-timeout", type=float, default=judge_mod.DEFAULT_TIMEOUT_S,
                        help=f"单次生成/判分超时秒数（默认 {judge_mod.DEFAULT_TIMEOUT_S:g}）")
    parser.add_argument("--judge-limit", type=int, default=0,
                        help="判分抽样条数（0=全量）。生成+判分是**两次真实 LLM 调用/例**，"
                             "用例上百时必须抽样；抽样按 kind 分层 + 层内等距，确定性可复现，"
                             "口径写入报告 judge.sample")
    parser.add_argument("--judge-dump", default=None, metavar="PATH",
                        help="把**判分当时真实使用的** (query, context, answer, 结论) 逐条落盘为 JSONL。"
                             "**要做事后人工标注（κ）就必须带这个** —— 报告只存 200 字预览且不存上下文，"
                             "事后重新生成答案会与判分时用的那条不同（实测 40 条里 22 条不同）")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    sparse_on, sparse_src = _resolve_sparse(args.sparse)
    rk_fp16, rk_ml, rk_src = _resolve_reranker_opts(args.rerank)
    factory = PipelineFactory(args.embedder, args.profile,
                              use_bge_sparse=sparse_on, sparse_source=sparse_src,
                              reranker_fp16=rk_fp16, reranker_max_length=rk_ml,
                              rerank_source=rk_src, merge=args.merge)
    factory.retrieve_top_k = args.retrieve_top_k
    factory.query_rewrite_mode = args.query_rewrite
    if args.query_rewrite == "legacy":
        from lightcore.rag.query_rewrite import rewrite_query_legacy

        factory.query_prep = rewrite_query_legacy   # 加固前基线（仅对照用）
        _prep_note = "（加固前基线 = 裸子串替换，**有已知缺陷**，仅对照用）"
    elif args.query_rewrite == "v2":
        from lightcore.rag.query_rewrite import rewrite_query

        factory.query_prep = rewrite_query          # ④ 加固版
        _prep_note = "（④ 加固版：jieba 词边界 + 保留疑问词）"
    else:
        from lightcore.rag.query_rewrite import expand_sections

        factory.query_prep = expand_sections         # 与线上同源（①-b 止血版）
        _prep_note = "（只做 /命令 剥离 + 章节展开 = 与线上同源）"
    # 把实际口径**打印出来**：这是"评测与线上是否同源"的第一手证据，
    # 也让双臂对照的两轮输出能直接区分（否则两轮日志一模一样，无法确认测的是不同配置）。
    print(f"口径：profile={args.profile} use_bge_sparse={sparse_on}（来源：{sparse_src}）")
    # T-03：重排档位同样打印 —— 否则"改了档位但评测数字没变"时无法区分
    # 「档位没生效」与「档位生效了但不影响这项指标」。
    print(f"      reranker: fp16={rk_fp16} · max_length={rk_ml or '不限'}（来源：{rk_src}）")
    print(f"      多文档合并：{args.merge}"
          + ("（与线上 CompositeRAGPipeline 同源）" if args.merge == "composite"
             else "（单管线全局排序 = 改动前评测口径；**与线上 Composite 不同源**）"))
    print(f"      查询预处理：{args.query_rewrite}{_prep_note}")
    # 判官需要检索原文（判"答案是否忠于证据"）；不启用判官时不收集，避免无谓的字符串拷贝
    contexts: dict[str, list[str]] = {} if args.judge else None  # type: ignore[assignment]
    rows = [run_case(c, args.k, factory, context_sink=contexts) for c in cases]
    metrics = summarize_results(rows)

    # 按用例类型分组统计：全局均值会把"饱和的简单用例"和"有分辨力的难例"混在一起。
    # 实测教训（2026-09-16）：12 条 verbatim 用例 recall@5 全为 1.0，而 multi_gold 只有 0.5，
    # 全局 recall@5=0.975 看不出这个差别 —— 分维度才能看出"指标为什么没分辨力"。
    kind_of = {c["id"]: c.get("kind", "unknown") for c in cases}
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(kind_of.get(r["id"], "unknown"), []).append(r)
    metrics_by_kind: dict[str, dict] = {}
    for kind, sub in sorted(by_kind.items()):
        block: dict = {"n": len(sub), "metrics": summarize_results(sub)}
        # T-04-B：`precision@k` 的**理论上限**必须一并写进报告。
        # 上限 = mean(min(|gold|, k)) / k —— 每条用例的相关块数量决定了这个指标能涨到多少。
        # 实测（2026-09-26）各 kind 达上限率 92%~100%，即该指标**已无分辨力**；
        # 不写这一行，事后无法区分「质量差」与「指标本来就涨不上去」。
        cap = _precision_cap(sub, args.k, cases)
        if cap:
            block["precision_cap"] = cap
        # ★ O-14（2026-10-01）：把"哪些指标还有分辨力"显式写进报告。
        # 动机：recall@5 / precision@5 在本语料上已贴顶（saturation≈1.0、recall=1.0），
        # 它们**只能变差不能变好** ⇒ 拿它们判断"优化有没有用"必然得出假阴性。
        # 报告必须自证这一点，否则后来者看到 0.24 的 precision 会误判为质量崩塌。
        flags = _discriminative_flags(block["metrics"], cap, args.k)
        if flags:
            block["discriminative"] = flags
        metrics_by_kind[kind] = block

    n_pos = sum(1 for r in rows if r.get("answerable", True))
    report = {
        "embedder": args.embedder,
        "profile": args.profile,
        # 口径自证（T-00）：把稀疏向量取值**与来源**写进报告 ——
        # 此前报告里没有它，导致事后无法回答"这版评测的 sparse 到底开没开、跟线上一样吗"。
        "use_bge_sparse": factory.use_bge_sparse,
        "use_bge_sparse_source": factory.sparse_source,
        # T-03 重排加速档位（自证字段）：与 use_bge_sparse 同思路 ——
        # 报告必须能自证"这一版的延迟是在哪个档位下测的"，否则跨版本数字不可比。
        "rag_reranker_fp16": factory.reranker_fp16,
        "rag_reranker_max_length": factory.reranker_max_length or 0,
        "rag_reranker_source": factory.rerank_source,
        # T-04：多文档合并路径 —— 决定 multi_doc/multi_gold 指标测的是哪条代码路径，
        # 两值下的数字**互不可比**（候选生成方式不同），跨版本比较必须同值。
        "merge_strategy": factory.merge,
        # T-04+：检索 top_k（0 = 沿用 max(k,_MRR_K)）。**必须与 merge 一起看** ——
        # 「每篇各取 top_k」是否构成真截断，取决于 top_k 与每篇块数的关系。
        "retrieve_top_k": factory.retrieve_top_k,
        # T-04+ 查询预处理（自证字段）：决定 composite 路径喂给子管线的 query 长什么样。
        # `sections` = 与线上同源（只做 /命令 剥离 + 章节展开）；`full` = 改动前基线（含
        # 疑问词/停用词清洗）。两值下的数字**互不可比** —— 这是第四次"评测与线上不同源"
        # 的同类字段，跨版本比较必须同值。single 模式下该字段不影响结果（不调预处理）。
        "query_rewrite_mode": factory.query_rewrite_mode,
        "k": args.k,
        "cases_file": Path(args.cases).name if args.cases else GOLDEN.name,
        # 用例集指纹 + 生成时间（2026-09-16，体检问题 5）：
        # `dev_eval` 用指纹做「报告是否已过时」判定 —— 用例集改了但没重跑评测时门禁必须能发现，
        # 否则会拿旧数字当基线（比"没有基线"更危险：它让人以为评测在守着质量）。
        "cases_sha256": _cases_digest(args.cases),
        "pipeline_digest": _pipeline_digest(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_cases": len(rows),
        "num_answerable": n_pos,
        "num_unanswerable": len(rows) - n_pos,
        "graded_relevance": any(c.get("gains") for c in cases),  # nDCG 是否用了分级标注
        "metrics": metrics,                       # 新增：完整指标表
        "metrics_by_kind": metrics_by_kind,       # 新增：按用例类型分组（看分辨力）
        # ↓ 既有契约键（勿删）：dev_eval.collect_metrics 的门禁判定读它们，
        #   由 tests/test_eval_summary.py 守护
        "avg_recall@5": metrics.get("recall@5", 0.0),
        "avg_mrr@10": metrics.get("mrr@10", 0.0),
        "detail": rows,
    }

    out = Path(args.out) if args.out else REPORT

    # 答案层判分放在写文件**之前**：若判官不可用（缺密钥 / 同厂商），进程非零退出且不产出报告 ——
    # 一份"检索数字齐全、只是没有判官结论"的报告会被误读为"评测通过"。
    if args.judge:
        try:
            report["judge"] = _run_judge_pass(cases, contexts or {}, args)
        except judge_mod.JudgeUnavailable as e:
            raise SystemExit(f"[FATAL] 答案层判分不可用：{e}") from e

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测完成：{len(rows)} 条样例（embedder={args.embedder}, profile={args.profile}, k={args.k}, "
          f"可答 {n_pos} / 负样本 {len(rows) - n_pos}）")
    for name, value in metrics.items():
        print(f"  {name:<16} = {value}")
    print("按用例类型分组：")
    for kind, block in metrics_by_kind.items():
        r5 = block["metrics"].get("recall@5")
        n5 = block["metrics"].get("ndcg@5")
        shown = f"recall@5={r5}" if r5 is not None else "（不计检索指标）"
        nd = f" ndcg@5={n5}" if n5 is not None else ""
        print(f"  {kind:<12} n={block['n']:<3} {shown}{nd}")
    print(f"报告已写入 {out}")
    jb = report.get("judge")
    if jb:
        for name in ("faithfulness", "abstention"):
            block = jb[name]
            rate = block[name]
            shown = "—（无样本）" if rate is None else f"{rate}"
            print(f"  {name:<14} = {shown}  "
                  f"（{block['hits']}/{block['judged']} 通过，判分失败 {block['errors']} 条不计入分母）")
        print("  ⚠️ 判分失败条数>0 时，比例只反映**成功判分**的样本；请先看 error_samples。")
    if report["graded_relevance"]:
        print("提示：nDCG 已启用**分级相关性**（gains: 3=直接证据 / 2=强相关 / 1=弱相关）")
    if args.embedder == "fake":
        print("⚠️ fake 嵌入：以上数字只证明链路与指标计算正确，不代表真实检索质量；"
              "真实质量请用 --embedder real")
    if args.profile == "fast":
        print("⚠️ fast profile：未启用 CrossEncoder 重排 / jieba / 稀疏向量，与线上口径不同，"
              "数字不可用于对外结论")


if __name__ == "__main__":
    main()
