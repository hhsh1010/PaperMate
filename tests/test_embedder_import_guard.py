"""FlagEmbedding 导入失败时的兜底、重试与报错口径（2026-09-18 线上事故回归）。

**事故**：后端在 ``sessions/18`` 检索时报
``RuntimeError: 未安装 FlagEmbedding，请先 `uv sync --extra rag`（见 pyproject[rag]）``，
而底层真因是 ``ImportError: cannot import name 'is_torch_npu_available' from 'transformers'``
—— **包装好了，是它依赖的 `transformers` 符号取不到**。

三条缺陷由此暴露，本文件各钉一条：
1. 旧文案把"没装 / 内部依赖失败 / 包损坏"合成一句 → 真因被掩盖（排查被带偏两轮）；
2. 缺一个**仅用于 NPU 探测**的符号就让整条 RAG 熔断 → 缺兜底；
3. 兜底返回 False 时**直接抛错、连重试都没有** → 而实测该失败属"进程内导入时序问题"，
   **片刻后状态已自愈**（同进程 `hasattr` 已为 True），只差一次重试。
"""

from __future__ import annotations

import importlib
import types

import pytest

from lightcore.rag import embedder

_REAL_CAUSE = "cannot import name 'is_torch_npu_available' from 'transformers'"


def _break_flagembedding(monkeypatch: pytest.MonkeyPatch, *, heal_after: int | None = None, healed=None) -> dict:
    """让 ``importlib.import_module("FlagEmbedding")`` 抛与线上同款的 ImportError。

    ``heal_after=N`` 表示第 N 次之后不再抛错（模拟"状态自愈"）。

    注意：实现走的是 ``importlib.import_module``，它**不经过** ``builtins.__import__``，
    所以必须 patch ``importlib.import_module``（patch builtins 拦不住）。
    """
    calls = {"n": 0}
    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name.startswith("FlagEmbedding"):
            calls["n"] += 1
            if heal_after is None or calls["n"] < heal_after:
                raise ImportError(_REAL_CAUSE)
            return healed
        return real(name, *args, **kwargs)

    monkeypatch.setattr(embedder.importlib, "import_module", fake)
    return calls


# ------------------------------------------------------------------ 兜底注入

def test_symbol_injected_when_missing() -> None:
    """上游缺该符号时必须注入，且注入实现返回 False（本机无 NPU）。"""
    fake = types.ModuleType("fake_transformers")
    fake.__version__ = "0.0-test"

    assert embedder._ensure_transformers_npu_symbol(fake) is True
    assert callable(fake.is_torch_npu_available)
    assert fake.is_torch_npu_available() is False


def test_symbol_not_touched_when_present() -> None:
    """符号已存在 → 不得覆盖（幂等，且不打扰正常环境）。"""
    fake = types.ModuleType("fake_transformers")
    sentinel = lambda check_device=False: False  # noqa: E731
    fake.is_torch_npu_available = sentinel

    assert embedder._ensure_transformers_npu_symbol(fake) is False
    assert fake.is_torch_npu_available is sentinel


def test_symbol_unavailable_module_is_noop() -> None:
    """传入对象不可用时不得抛错（兜底本身不得成为新的故障点）。"""
    assert embedder._ensure_transformers_npu_symbol(object()) is False


# ------------------------------------------------------------------ 重试（本次核心）

def test_retry_succeeds_when_state_recovers(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """**首次失败、状态自愈后重试成功** —— 这正是 2026-09-18 线上那种形态。

    旧实现在"兜底无需注入"（hasattr 已为 True）时**直接抛错**，于是把一次自愈机会浪费掉、
    换来整条 RAG 熔断。新实现无论兜底结果如何都重试一次。
    """
    sentinel = object()
    healed = types.SimpleNamespace(BGEM3FlagModel=sentinel)
    calls = _break_flagembedding(monkeypatch, heal_after=2, healed=healed)
    monkeypatch.setattr(embedder, "_ensure_transformers_npu_symbol", lambda *a, **k: False)

    with caplog.at_level("WARNING", logger="lightcore.rag.embedder"):
        assert embedder._import_bgem3_flag_model() is sentinel

    assert calls["n"] == 2, f"状态自愈后应靠重试拿到结果（共 2 次尝试），实际 {calls['n']}"
    assert "尝试成功" in "\n".join(r.getMessage() for r in caplog.records)


def test_both_attempts_failing_reports_real_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有尝试都失败 → 报错必须**带真因**，并说明已重试次数、兜底是否生效。"""
    calls = _break_flagembedding(monkeypatch)
    monkeypatch.setattr(embedder, "_ensure_transformers_npu_symbol", lambda *a, **k: True)

    with pytest.raises(RuntimeError) as ei:
        embedder._import_bgem3_flag_model()

    total = 1 + len(embedder._IMPORT_BACKOFF_S)
    text = str(ei.value)
    assert calls["n"] == total, f"应有界重试到 {total} 次，实际 {calls['n']}"
    assert _REAL_CAUSE in text, f"报错必须带真因原文：{text}"
    assert "ImportError" in text, f"报错必须带真因类型：{text}"
    assert f"共尝试 {total} 次" in text and "兜底注入=True" in text, f"应说明尝试次数与兜底状态：{text}"
    assert "uv sync --extra rag" in text, f"仍应给出安装提示（真没装时可用）：{text}"


def test_state_snapshot_logged_on_first_failure(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """首次失败必须落**环境快照** —— 这类失败依赖进程内时序，不留证下次仍无从下手。"""
    _break_flagembedding(monkeypatch)
    monkeypatch.setattr(embedder, "_ensure_transformers_npu_symbol", lambda *a, **k: False)

    with caplog.at_level("ERROR", logger="lightcore.rag.embedder"):
        with pytest.raises(RuntimeError):
            embedder._import_bgem3_flag_model()

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "环境快照" in text, f"必须记录快照：{text}"
    for field in ("sys.executable", "线程", "sys.modules 有 transformers"):
        assert field in text, f"快照缺字段 {field}：{text}"


# ------------------------------------------------------------------ 冒烟

def test_flagembedding_importable_in_this_env() -> None:
    """本环境（含 rag extra）应能直接导入 —— 与线上事故形成对照。"""
    pytest.importorskip("FlagEmbedding")
    from FlagEmbedding import BGEM3FlagModel  # noqa: F401
