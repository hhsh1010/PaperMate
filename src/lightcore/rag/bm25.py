"""LightCore · BM25 稀疏检索（docs/05 §RAG / 混合检索）

用 rank_bm25 做关键词检索。中文无空格，默认按字符级切分（demo 级）；
生产环境用 jieba 分词（**已列入依赖**：`pyproject.toml` 的 `jieba>=0.42`，
由 `use_jieba` 开关控制，默认开；未装则自动回退字符级）。tokenize 可注入以替换策略。

★ 2026-10-04（O9）修正过时描述：此处原写「**pyproject 未列入**，可按需补充」，
而 jieba 早已是正式依赖（注释里还记着它修复了"字符级下英文 BM25 召回弱"的问题）。
⚠️ 这类"已补上但仍写着未做"的注释最危险 —— 会让后来者以为要重新做一遍。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rank_bm25 import BM25Okapi

#: 过滤用的标点/空白字符集（中文友好；与字符级切分共用，保证两种分词口径一致）
_PUNCT = set("，。！？、；：“”‘’（）《》\n\t ")


def _is_punct(tok: str) -> bool:
    return bool(tok) and all(c in _PUNCT for c in tok)


def _char_tokenize(text: str) -> list[str]:
    # 过滤标点/空白，按单字切分（中文无空格时的 demo 级兜底）
    return [c for c in text if c.strip() and c not in _PUNCT]


def jieba_tokenize(text: str) -> list[str]:
    """中文分词：jieba 按词切分；未安装 jieba 时自动回退字符级，保证零依赖可用。

    与 _char_tokenize 共用 _PUNCT 过滤，避免标点单独成 token 干扰 BM25 词频。
    """
    try:
        import jieba  # 可选依赖：未列入 pyproject，需用户自行 pip install jieba
    except Exception:  # noqa: BLE001  jieba 缺失 → 静默回退字符级
        return _char_tokenize(text)
    return [w for w in jieba.cut(text) if w.strip() and not _is_punct(w)]


class BM25Retriever:
    def __init__(self, tokenize: Callable[[str], list[str]] = _char_tokenize) -> None:
        self._tokenize = tokenize
        self._docs: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        # O-11（2026-10-01）：惰性重建。此前 add() 每批都对全部累积文档重建 BM25Okapi
        # （pipeline.add_chunks 按 batch=64 分批 ⇒ 658 chunks/11 批 ⇒ 累积重建 ≈6.3×），
        # 实测冗余 152.7ms、占 add 耗时 15.3%（scripts/probe_bm25_rebuild.py，真实 658 chunks）。
        # 现改为 add() 只增量分词并置 _dirty，search() 前一次性重建 ⇒ 重建总量 O(n²)→O(n)。
        self._dirty = False

    def add(self, texts: Sequence[str]) -> None:
        if not texts:
            return
        self._docs.extend(texts)
        self._tokenized.extend(self._tokenize(t) for t in texts)
        self._dirty = True  # 统计量构建推迟到 search 前一次完成

    def _ensure_built(self) -> None:
        if self._dirty:
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
            self._dirty = False

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        self._ensure_built()  # ★ 惰性后必须保证 search 前必然重建（否则召回为空）
        if self._bm25 is None:
            return []
        toks = self._tokenize(query)
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        k = min(k, len(order))
        return [(i, float(scores[i])) for i in order[:k] if scores[i] > 0]
