"""T-06 P1：图内容哈希缓存（2026-09-30）。

背景（探针 `docs/T06-探针结果.md`）：doc10 类图密集文档改稿重建 742s 里 **VLM 占 75%**
（无 VLM 臂实测 187s）。未变图 0 次调用是改稿提速的主要来源。

守护三件事：
① **同图第二次 enrich 零调用**（缓存命中），结果与首次一致；
② **失效正确**：图片内容变 / methods 变 / vlm_model 变 ⇒ 必须重新调用（键覆盖）；
③ **缓存坏了不能阻断**：cache 抛异常 ⇒ 退回真实增强（缓存是优化，不是依赖）。
另验证 papermate 侧 SQLite 适配器（`_FigureEnrichCache`）的 round-trip 与建表。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import sqlalchemy as sa

from lightcore.rag.figure_content import FigureEnricher


class _FakeCache:
    """内存缓存（记录命中/写入次数，便于断言）。"""

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self.get_calls = 0
        self.put_calls = 0

    def get(self, key: str) -> dict | None:
        self.get_calls += 1
        return self.store.get(key)

    def put(self, key: str, value: dict) -> None:
        self.put_calls += 1
        self.store[key] = dict(value)


class _BrokenCache:
    def get(self, key: str) -> dict | None:
        raise RuntimeError("cache backend down")

    def put(self, key: str, value: dict) -> None:
        raise RuntimeError("cache backend down")


def _make_enricher(cache, *, vlm_model: str = "fake-vlm") -> tuple[FigureEnricher, list[int]]:
    """enabled+vlm 的 enricher；`_vlm` 打桩为计数器（不触真实 API）。"""
    enr = FigureEnricher(
        SimpleNamespace(
            figure_enrich={"enabled": True, "methods": ["vlm"], "vlm_model": vlm_model}
        ),
        cache=cache,
    )
    calls: list[int] = []

    async def _stub_vlm(path, *, kind: str = "figure", **_: object) -> str:
        calls.append(calls[-1] + 1 if calls else 1)
        return f"caption-of-{Path(path).name}-{len(calls)}"

    enr._vlm = _stub_vlm  # type: ignore[method-assign]  测试打桩
    return enr, calls


def _write_image(tmp_path: Path, content: bytes) -> Path:
    p = tmp_path / "img.png"
    p.write_bytes(content)
    return p


# ---- ① 命中 -------------------------------------------------------------------
def test_same_image_second_enrich_is_cache_hit(tmp_path: Path) -> None:
    cache = _FakeCache()
    enr, calls = _make_enricher(cache)
    img = _write_image(tmp_path, b"image-bytes-v1")

    first = _run(enr.enrich(img, kind="figure"))
    second = _run(enr.enrich(img, kind="figure"))

    assert len(calls) == 1, "同图第二次 enrich 不应再调 VLM"
    assert second == first
    assert cache.put_calls == 1 and cache.get_calls == 2


def _run(coro):
    import asyncio

    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# ---- ② 失效 -------------------------------------------------------------------
def test_image_content_change_invalidates(tmp_path: Path) -> None:
    cache = _FakeCache()
    enr, calls = _make_enricher(cache)
    img = _write_image(tmp_path, b"image-bytes-v1")
    _run(enr.enrich(img, kind="figure"))
    img.write_bytes(b"image-bytes-v2")  # 内容变
    _run(enr.enrich(img, kind="figure"))
    assert len(calls) == 2, "图片内容变化必须重新增强"


def test_method_or_model_change_invalidates(tmp_path: Path) -> None:
    cache = _FakeCache()
    img = _write_image(tmp_path, b"same-bytes")

    enr1, calls1 = _make_enricher(cache, vlm_model="fake-vlm")
    _run(enr1.enrich(img, kind="figure"))
    assert len(calls1) == 1

    enr2, calls2 = _make_enricher(cache, vlm_model="another-vlm")
    _run(enr2.enrich(img, kind="figure"))
    assert len(calls2) == 1, "vlm_model 变化必须 miss（不同模型的输出不可复用）"

    enr3, calls3 = _make_enricher(cache, vlm_model="fake-vlm")
    _run(enr3.enrich(img, kind="diagram"))
    assert len(calls3) == 1, "kind 变化必须 miss（提示词不同）"


def test_no_cache_injection_keeps_old_behavior(tmp_path: Path) -> None:
    """不注入 cache ⇒ 行为与改动前完全一致（每次都真实调用）。"""
    enr, calls = _make_enricher(None)
    img = _write_image(tmp_path, b"same-bytes")
    _run(enr.enrich(img, kind="figure"))
    _run(enr.enrich(img, kind="figure"))
    assert len(calls) == 2


# ---- ③ 降级 -------------------------------------------------------------------
def test_broken_cache_falls_back_to_real_enrich(tmp_path: Path) -> None:
    """★ 缓存是优化不是依赖：后端挂了必须退回真实增强，绝不阻断/不改结果语义。"""
    enr, calls = _make_enricher(_BrokenCache())
    img = _write_image(tmp_path, b"image-bytes")
    out = _run(enr.enrich(img, kind="figure"))
    assert out["vlm_caption"].startswith("caption-of-")
    assert len(calls) == 1


# ---- papermate 侧：SQLite 适配器 + 建表 ---------------------------------------
def test_sqlite_cache_adapter_roundtrip(tmp_path: Path) -> None:
    """`_FigureEnrichCache` 适配器与 `figure_enrich_cache` 表的 round-trip（含建表）。"""
    from papermate.models.db import init_db, make_engine
    from papermate.services.document_service import _FigureEnrichCache

    engine = make_engine(f"sqlite:///{tmp_path / 'c.db'}")
    init_db(engine)
    # 表必须存在（init_db 建表）——否则生产启动后首次 put 会 OperationalError
    insp = sa.inspect(engine)
    assert "figure_enrich_cache" in insp.get_table_names()

    adapter = _FigureEnrichCache(sessionmaker_factory(engine))
    assert adapter.get("nope") is None
    adapter.put("k1", {"ocr_text": "o", "table_md": "t", "vlm_caption": "v"})
    got = adapter.get("k1")
    assert got == {"ocr_text": "o", "table_md": "t", "vlm_caption": "v"}
    adapter.put("k1", {"ocr_text": "o2", "table_md": "", "vlm_caption": "v2"})
    assert adapter.get("k1")["ocr_text"] == "o2", "同键覆盖（INSERT OR REPLACE 语义）"


def sessionmaker_factory(engine):
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=engine, expire_on_commit=False)


def test_document_service_wires_the_cache() -> None:
    """守卫：DocumentService 的 enricher 必须注入缓存（漏掉 = 优化静默失效）。"""
    src = (
        Path(__file__).resolve().parents[1] / "src" / "papermate" / "services" / "document_service.py"
    ).read_text(encoding="utf-8")
    assert "cache=_FigureEnrichCache(self._Session)" in src


# ===================================================================== O-07（2026-10-01）
# 模型单例：此前 PaddleOCR / TableTransformer 都写在**逐图函数体**里 ⇒ N 张图 = N 次加载
# （`from_pretrained` / `PaddleOCR(...)` 初始化都是秒级）。用假模块注入 sys.modules，
# **无需真实重依赖**即可对"构造次数"计数。


def test_ocr_engine_loaded_once_per_enricher(monkeypatch):
    """★★ O-07 回归：PaddleOCR 引擎必须**每个 enricher 只构造一次**。"""
    import sys

    calls = {"n": 0}

    class _FakeEngine:
        def __init__(self, **kwargs) -> None:
            calls["n"] += 1
            self.kwargs = kwargs

        def predict(self, path):  # noqa: ANN001  v3 形态：空结果即可
            return []

    fake = SimpleNamespace(__version__="3.2.0", PaddleOCR=_FakeEngine)
    monkeypatch.setitem(sys.modules, "paddleocr", fake)

    e = FigureEnricher()
    assert e._ocr_lines(Path("x.png")) == []
    assert e._ocr_lines(Path("x.png")) == []
    assert calls["n"] == 1, f"OCR 引擎被构造了 {calls['n']} 次（修复前 = 图数）"
    assert e.model_loads == 1


def test_table_models_loaded_once(monkeypatch):
    """★★ O-07 回归：Table Transformer 的 processor/model 必须只 `from_pretrained` 一次。"""
    import sys

    proc = {"n": 0}
    mdl = {"n": 0}

    def _mk(counter):  # noqa: ANN001
        def _load(model_id):  # noqa: ANN001
            counter["n"] += 1
            return ("artifact", model_id)

        return staticmethod(_load)

    fake = SimpleNamespace(
        AutoImageProcessor=SimpleNamespace(from_pretrained=_mk(proc)),
        TableTransformerForObjectDetection=SimpleNamespace(from_pretrained=_mk(mdl)),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)

    e = FigureEnricher()
    p1, m1 = e._get_table_models()
    p2, m2 = e._get_table_models()
    assert p1 is p2 and m1 is m2, "两次调用拿到了不同实例 ⇒ 没有复用"
    assert proc["n"] == 1 and mdl["n"] == 1, f"from_pretrained 各被调了 {proc['n']}/{mdl['n']} 次"
    assert e.model_loads == 1


def test_each_enricher_instance_loads_its_own_models(monkeypatch):
    """作用域 = **实例级**：两个 enricher 各加载一次（共 2 次）。

    刻意**不**做模块级全局单例 —— 不同 enricher 的 `methods`/`vlm_model` 配置可能不同，
    跨实例共享模型会把"配置隔离"变成隐式耦合。作用域与调用粒度匹配：
    document_service **每篇文档建一个** enricher ⇒ 一篇文档 N 张图只加载 1 次。
    """
    import sys

    calls = {"n": 0}

    class _FakeEngine:
        def __init__(self, **kwargs) -> None:
            calls["n"] += 1

        def predict(self, path):  # noqa: ANN001
            return []

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(__version__="3.2.0", PaddleOCR=_FakeEngine))

    a = FigureEnricher()
    b = FigureEnricher()
    a._ocr_lines(Path("x.png"))
    b._ocr_lines(Path("x.png"))
    assert calls["n"] == 2, "实例间必须隔离（各加载一次），既不共享也不漏加载"
