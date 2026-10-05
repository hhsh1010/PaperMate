"""T-05 冷启动优化：预热 + LRU 上限 + 加载不阻塞事件循环（2026-09-26）。

## 这份测试要守的三件事

1. **预热真的调用了两条重型路径**，且 fake 模式下**直接跳过**（不白拉 22s）；
2. **LRU 淘汰按"最久未用"**，且**不误伤 G1 删除代际**（淘汰 ≠ 删除）；
3. **`_max_pipelines` 的非法值回退** —— 一个手滑的 `0` 不能让保护静默失效。

## 为什么要单独测这些
`_pipelines` 的淘汰与既有三处机制交互（G1 代际 / `invalidate_rag_components` /
`delete`），而"淘汰"这个新行为**最坏情况会变成"用到一半索引消失"**——必须钉死。
"""
from __future__ import annotations

import pytest

import papermate.config as cfg_mod
from papermate.services.document_service import (
    _DEFAULT_MAX_PIPELINES,
    DocumentService,
    _resolve_max_pipelines,
)


class _Query:
    def filter(self, *a):
        return self

    def order_by(self, *a):
        return self

    def all(self):
        return []


class _Session:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def query(self, *a):
        return _Query()


class _SessionMaker:
    def __call__(self):
        return _Session()


def _svc(tmp_path, monkeypatch, **env) -> DocumentService:
    monkeypatch.setenv("PAPERMATE_DATA_DIR", str(tmp_path))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg_mod._settings = None
    import papermate.runtime_settings as rs

    rs._state = None
    from papermate.config import get_settings

    return DocumentService(_SessionMaker(), get_settings())


# ------------------------- 一、_max_pipelines 解析 -------------------------


@pytest.mark.parametrize("raw", ["", "0", "-5", "abc", "  ", "1.5"])
def test_max_pipelines_invalid_falls_back_to_default(monkeypatch, raw):
    """非法值必须回退默认 —— 尤其 `0`：它曾被设想表示"不限制"，
    但那会让保护静默失效（与"设上限防 OOM"的初衷相反）。"""
    monkeypatch.setenv("PAPERMATE_MAX_PIPELINES", raw)
    assert _resolve_max_pipelines() == _DEFAULT_MAX_PIPELINES


def test_max_pipelines_valid_value_honoured(monkeypatch):
    monkeypatch.setenv("PAPERMATE_MAX_PIPELINES", "8")
    assert _resolve_max_pipelines() == 8


# ------------------------- 二、LRU 淘汰语义 -------------------------


def test_lru_evicts_least_recently_used(tmp_path, monkeypatch):
    """★ 核心契约：超上限时淘汰**最久未用**的，而不是任意一个。"""
    svc = _svc(tmp_path, monkeypatch, PAPERMATE_MAX_PIPELINES="2")

    svc._cache_pipeline(1, object(), gen=0)
    svc._cache_pipeline(2, object(), gen=0)
    svc._pipelines.move_to_end(1)          # 模拟"文档 1 刚被访问过"
    svc._cache_pipeline(3, object(), gen=0)  # 超上限 → 应淘汰文档 2

    assert list(svc._pipelines) == [1, 3], f"应淘汰最久未用的 2，实际留下 {list(svc._pipelines)}"


def test_lru_keeps_most_recent_when_at_limit(tmp_path, monkeypatch):
    svc = _svc(tmp_path, monkeypatch, PAPERMATE_MAX_PIPELINES="1")
    svc._cache_pipeline(1, object(), gen=0)
    svc._cache_pipeline(2, object(), gen=0)
    assert list(svc._pipelines) == [2], "上限 1 时应只保留最新写入的"


def test_eviction_does_not_touch_generation(tmp_path, monkeypatch):
    """★ 淘汰 ≠ 删除：淘汰**不得**改动 G1 删除代际，否则会误伤
    「构建期间被删除」的判定（那是另一套机制）。"""
    svc = _svc(tmp_path, monkeypatch, PAPERMATE_MAX_PIPELINES="1")
    svc._pipeline_gen[1] = 7
    svc._cache_pipeline(1, object(), gen=7)
    svc._cache_pipeline(2, object(), gen=0)   # 淘汰文档 1

    assert 1 not in svc._pipelines, "文档 1 应被淘汰"
    assert svc._pipeline_gen.get(1) == 7, "淘汰不得修改代际（那是删除路径的职责）"


def test_generation_guard_still_blocks_stale_write(tmp_path, monkeypatch):
    """G1 既有行为不受影响：代际不符时仍放弃写入。"""
    svc = _svc(tmp_path, monkeypatch, PAPERMATE_MAX_PIPELINES="4")
    svc._cache_pipeline(1, object(), gen=999)   # 代际不符
    assert 1 not in svc._pipelines, "过期构建不得写入缓存"


def test_no_eviction_below_limit(tmp_path, monkeypatch):
    svc = _svc(tmp_path, monkeypatch, PAPERMATE_MAX_PIPELINES="5")
    for d in range(5):
        svc._cache_pipeline(d, object(), gen=0)
    assert len(svc._pipelines) == 5, "未超上限不得淘汰"


# ------------------------- 三、预热 warm_up -------------------------


async def test_warm_up_skips_for_fake_embedder(tmp_path, monkeypatch):
    """fake 模式无需预热（构造是微秒级）—— 这也是绝大多数测试不受影响的原因。"""
    svc = _svc(tmp_path, monkeypatch)
    monkeypatch.setattr(svc, "_rag_rt", lambda: {"rag_embedder": "fake"})
    called: list[str] = []
    monkeypatch.setattr(svc, "_get_embedder", lambda: called.append("emb"))

    out = await svc.warm_up()
    assert out.get("skipped") is True
    assert called == [], "fake 模式不得触发任何加载"


async def test_warm_up_loads_both_for_real_embedder(tmp_path, monkeypatch):
    """real 模式必须**两条都预热**（缺一条就等于把那一半的加载留给用户）。"""
    svc = _svc(tmp_path, monkeypatch)
    monkeypatch.setattr(svc, "_rag_rt", lambda: {"rag_embedder": "real"})
    called: list[str] = []

    async def _emb():
        called.append("embedder")

    async def _rr():
        called.append("reranker")

    monkeypatch.setattr(svc, "_get_embedder", _emb)
    monkeypatch.setattr(svc, "_get_reranker", _rr)

    out = await svc.warm_up()
    assert called == ["embedder", "reranker"]
    assert "total_s" in out and "embedder_s" in out and "reranker_s" in out


# --------------- 三·b、T-05b：预热「首次使用」的冷开销 ---------------
# 背景（2026-09-27 复查）：T-05 只预热了 embedder/reranker **权重**，首个真实请求仍多付 ≈2.1s
#   ① jieba 词典首次加载 ≈0.6s（在首个请求里才发生）
#   ② 首次检索 ≈1.5s：CUDA kernel 初始化 + faiss 首次搜索
# 实测：重启后第一次 retrieve 2148ms（top_k=6）vs 后续 641/670ms（top_k=10）
#   ⇒ top_k 更小却更慢 ⇒ 多出的部分与数据量无关，是"首次使用"开销。
# T-05 当时判"预热 pipeline 收益有限"，依据是探针——而**探针只测加载、从未跑过一次检索**。


class _Pipe:
    def __init__(self):
        self.queries: list[tuple] = []

    def query(self, q, top_k=3):
        self.queries.append((q, top_k))
        return []


async def _real_svc(tmp_path, monkeypatch, **env) -> DocumentService:
    svc = _svc(tmp_path, monkeypatch, **env)
    monkeypatch.setattr(svc, "_rag_rt", lambda: {"rag_embedder": "real", "use_jieba": True})

    async def _emb():
        return None

    async def _rr():
        return None

    monkeypatch.setattr(svc, "_get_embedder", _emb)
    monkeypatch.setattr(svc, "_get_reranker", _rr)
    return svc


async def test_warm_up_warms_tokenizer_when_jieba_on(tmp_path, monkeypatch):
    """jieba 词典必须被预热 —— 否则它会在**首个用户请求**里花 0.6s 加载。"""
    svc = await _real_svc(tmp_path, monkeypatch)
    seen: list[str] = []

    async def _tok():
        return lambda s: seen.append(s)

    monkeypatch.setattr(svc, "_get_tokenizer", _tok)
    monkeypatch.setattr(svc, "_recent_ready_doc_ids", lambda n: [])

    out = await svc.warm_up()
    assert seen == ["预热分词"], f"分词器未被预热调用：{seen}"
    assert out.get("tokenizer_s", -1) >= 0


async def test_warm_up_runs_one_probe_query(tmp_path, monkeypatch):
    """★ 核心：预热必须**真的跑一次检索**（触发 CUDA/faiss 首次使用开销）。

    只 `get_pipeline` 不 `query`，等于把 ~1.5s 的冷开销留在首个请求里。
    """
    svc = await _real_svc(tmp_path, monkeypatch)
    pipe = _Pipe()

    async def _tok():
        return lambda s: []

    monkeypatch.setattr(svc, "_get_tokenizer", _tok)
    monkeypatch.setattr(svc, "_recent_ready_doc_ids", lambda n: [19])

    async def _get_pipe(doc_id, **kw):
        return pipe

    monkeypatch.setattr(svc, "get_pipeline", _get_pipe)

    out = await svc.warm_up()
    assert pipe.queries == [("预热", 1)], f"应只跑一次最小检索（top_k=1），实际 {pipe.queries}"
    assert out.get("prewarmed_docs") == 1


async def test_prewarm_docs_zero_skips_pipeline(tmp_path, monkeypatch):
    """`PAPERMATE_PREWARM_DOCS=0` = 只预热模型，不碰 pipeline（0 是合法值，非解析失败）。"""
    svc = await _real_svc(tmp_path, monkeypatch, PAPERMATE_PREWARM_DOCS="0")
    pipe = _Pipe()

    async def _tok():
        return lambda s: []

    monkeypatch.setattr(svc, "_get_tokenizer", _tok)
    monkeypatch.setattr(svc, "_recent_ready_doc_ids", lambda n: [19])

    async def _get_pipe(doc_id, **kw):
        return pipe

    monkeypatch.setattr(svc, "get_pipeline", _get_pipe)

    out = await svc.warm_up()
    assert pipe.queries == [], "关闭时不该跑检索"
    assert out.get("prewarmed_docs") == 0


async def test_warm_up_survives_pipeline_failure(tmp_path, monkeypatch):
    """预热失败**不得抛出** —— 它只影响首问快慢，不能让服务起不来。"""
    svc = await _real_svc(tmp_path, monkeypatch)

    async def _tok():
        return lambda s: []

    monkeypatch.setattr(svc, "_get_tokenizer", _tok)
    monkeypatch.setattr(svc, "_recent_ready_doc_ids", lambda n: [19])

    async def _boom(doc_id, **kw):
        raise RuntimeError("模拟 pipeline 构建失败")

    monkeypatch.setattr(svc, "get_pipeline", _boom)

    out = await svc.warm_up()   # 不抛即通过
    assert out.get("doc_19_s") == -1.0, f"失败应记为 -1，实际 {out}"


def test_recent_ready_doc_ids_filters_ready(tmp_path, monkeypatch):
    """只取 `ready` 篇 —— 拿 failed/processing 的去预热会构建出空/残缺索引。"""
    svc = _svc(tmp_path, monkeypatch)
    calls: dict = {}

    class _Q:
        def filter(self, *a):
            calls["filter"] = a
            return self

        def order_by(self, *a):
            calls["order_by"] = a
            return self

        def limit(self, n):
            calls["limit"] = n
            return self

        def all(self):
            return [(19,), (13,)]

    class _S:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def query(self, *a):
            calls["query"] = a
            return _Q()

    monkeypatch.setattr(svc, "_Session", lambda: _S())
    assert svc._recent_ready_doc_ids(1) == [19, 13]
    assert calls["limit"] == 1


# ------------------------- 四、预热开关 -------------------------


@pytest.mark.parametrize(("raw", "expected"), [
    (None, True), ("1", True), ("true", True), ("TRUE", True), ("yes", True),
    ("0", False), ("false", False), ("FALSE", False), ("no", False), ("off", False),
])
def test_prewarm_flag_parsing(monkeypatch, raw, expected):
    """开关解析：**默认开**（生产要预热），只有显式关才关。"""
    from papermate.api.app import _PREWARM_ENV_VAR, _prewarm_enabled

    if raw is None:
        monkeypatch.delenv(_PREWARM_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(_PREWARM_ENV_VAR, raw)
    assert _prewarm_enabled() is expected
