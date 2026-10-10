"""T-03 重排推理加速：**默认行为不变** + 档位生效 + 校验边界。

设计取舍：**不加载真实模型**。把 ``sentence_transformers.CrossEncoder`` 换成记录参数的假类，
只断言"透传了什么"，因此本文件是**毫秒级、零权重依赖、零 GPU 依赖**的 —— 
这样它才能在 CI/无 rag 权重环境下也守护住契约。

分三层断言：
1. `rerank.py` 的默认值必须**严格等于 2026-09-24 之前的行为**（否则就是静默的行为变更）；
2. 三个新参数要能**透传到位**（max_length / fp16 / batch_size），并含两个易错边界：
   `max_length=0` 视为不限长；**fp16 在非 CUDA 上强制回落 fp32 并告警**（不静默）；
3. `runtime_settings` 的取值与脏数据校验（0 合法、过小非法）；
4. `DocumentService.reset_reranker_pipelines()` 必须同时清**管线缓存**与重排器单例 ——
   否则"设置改了但已缓存管线仍用旧档位"，属静默失效。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

# ★ 2026-10-10：numpy 属 [rag] 侧依赖，不在基础依赖集里 —— 顶层直接 import 会让
# **整个文件在 CI（轻依赖环境）收集期报错**。改成仓库既有惯例 importorskip：
# 缺 numpy 时本文件整体 skip（而非 collection error），CI 才能保持绿。
np = pytest.importorskip("numpy")

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import run_eval  # noqa: E402

from lightcore.rag.rerank import CrossEncoderReranker  # noqa: E402
from papermate import runtime_settings  # noqa: E402
from papermate.services.document_service import DocumentService  # noqa: E402


class _FakeInner:
    """假的 HF 模型：只需支持 `.half()`（fp16 路径会调用）。"""

    def __init__(self) -> None:
        import torch

        self.linear = torch.nn.Linear(2, 1)

    def half(self):
        self.linear = self.linear.half()
        return self


class _FakeCrossEncoder:
    """记录构造参数与 predict 参数，供断言「透传」是否到位。"""

    last_init: dict = {}
    last_predict: dict = {}

    def __init__(self, *args, **kwargs) -> None:
        _FakeCrossEncoder.last_init = dict(kwargs)
        _FakeCrossEncoder.last_init["_positional"] = args
        self.model = _FakeInner()

    def predict(self, pairs, **kwargs):
        _FakeCrossEncoder.last_predict = dict(kwargs)
        return np.full(len(pairs), 0.5, dtype="float32")


@pytest.fixture(autouse=True)
def _fake_ce(monkeypatch):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _FakeCrossEncoder)
    # 不依赖本机是否已有权重快照：强制走"无本地快照"分支（该分支有 try/except 兜底）
    monkeypatch.setattr(CrossEncoderReranker, "_resolve_local", staticmethod(lambda _name: None))
    yield


def _pairs(n: int = 3) -> tuple[list[tuple[int, float]], list[str]]:
    return [(i, 0.5) for i in range(n)], [f"候选文本 {i}" for i in range(n)]


# ---------------- 1. 默认值 = 改动前行为 ----------------


def test_defaults_are_strictly_legacy_behaviour() -> None:
    """★ 核心守护：不传任何新参数时，构造与调用**都必须与 2026-09-24 之前逐字一致**。"""
    rk = CrossEncoderReranker(device="cpu")
    assert rk.max_length is None, "默认必须不限长（= 改动前行为）"
    assert rk.batch_size is None
    assert rk.fp16 is False, "默认不启用半精度"
    assert _FakeCrossEncoder.last_init["max_length"] is None

    cands, texts = _pairs()
    rk.rerank("问题", cands, top_k=2, texts=texts)
    assert "batch_size" not in _FakeCrossEncoder.last_predict, "未显式设置时不得传 batch_size"
    assert _FakeCrossEncoder.last_predict["show_progress_bar"] is False


# ---------------- 2. 参数透传与边界 ----------------


def test_max_length_passthrough_and_zero_means_unlimited() -> None:
    rk = CrossEncoderReranker(device="cpu", max_length=384)
    assert rk.max_length == 384
    assert _FakeCrossEncoder.last_init["max_length"] == 384
    # 0 = 不限长（设置页用 0 表达"回到改动前行为"）
    assert CrossEncoderReranker(device="cpu", max_length=0).max_length is None
    # 负数同样归一到"不限长"，绝不把候选截成空串
    assert CrossEncoderReranker(device="cpu", max_length=-5).max_length is None


def test_fp16_falls_back_to_fp32_off_cuda_and_warns(caplog) -> None:
    """★ fp16 只在 CUDA 生效 —— CPU 上启用反而更慢，必须回落且**显式告警**（不静默）。"""
    with caplog.at_level(logging.WARNING, logger="lightcore.rag.rerank"):
        rk = CrossEncoderReranker(device="cpu", fp16=True)
    assert rk.fp16 is False
    assert any("fp16" in r.message for r in caplog.records), "回落必须留下告警，不能静默"


def test_fp16_enabled_on_cuda() -> None:
    rk = CrossEncoderReranker(device="cuda", fp16=True)
    assert rk.fp16 is True
    import torch

    assert rk._model.model.linear.weight.dtype == torch.float16


def test_batch_size_passed_only_when_explicitly_set() -> None:
    cands, texts = _pairs()
    CrossEncoderReranker(device="cpu", batch_size=16).rerank("问题", cands, top_k=2, texts=texts)
    assert _FakeCrossEncoder.last_predict["batch_size"] == 16


def test_rerank_contract_unchanged() -> None:
    """契约回归：空候选返 []、texts 缺省/长度不符抛 ValueError（M9 / 2026-09-03 修正）。"""
    rk = CrossEncoderReranker(device="cpu")
    assert rk.rerank("问题", [], top_k=2, texts=[]) == []
    with pytest.raises(ValueError):
        rk.rerank("问题", [(0, 1.0)], top_k=1, texts=None)
    with pytest.raises(ValueError):
        rk.rerank("问题", [(0, 1.0)], top_k=1, texts=["a", "b"])


# ---------------- 3. runtime_settings 取值与校验 ----------------


def test_reranker_settings_defaults_and_validation(tmp_path) -> None:
    with mock.patch.object(runtime_settings, "_path", lambda: tmp_path / "provider_settings.json"):
        runtime_settings._state = None
        st = runtime_settings.get_state()
        # 默认档位 = 实测推荐值（512 / 开；2026-10-05 质量优先三臂对照后由 384 上调）
        assert st["rag_reranker_fp16"] is True
        assert st["rag_reranker_max_length"] == 512

        runtime_settings.update({"rag_reranker_max_length": 256, "rag_reranker_fp16": False})
        assert st["rag_reranker_max_length"] == 256
        assert st["rag_reranker_fp16"] is False

        # 0 = 不限长，合法（这是"一键回退"的一半）
        runtime_settings.update({"rag_reranker_max_length": 0})
        assert st["rag_reranker_max_length"] == 0

        # 脏数据：过小（会把候选截成空串）与超上限都必须被拒，保留上次有效值
        runtime_settings.update({"rag_reranker_max_length": 10})
        assert st["rag_reranker_max_length"] == 0
        runtime_settings.update({"rag_reranker_max_length": 99999})
        assert st["rag_reranker_max_length"] == 0
        # bool 不能被当成 int 混进来（True==1 的经典坑）
        runtime_settings.update({"rag_reranker_max_length": True})
        assert st["rag_reranker_max_length"] == 0

        out = runtime_settings.public_view(st)
        assert out["rag_reranker_max_length"] == 0
        assert out["rag_reranker_fp16"] is False


# ---------------- 4. 改设置必须让已缓存管线失效 ----------------


def test_rag_rt_exposes_reranker_knobs(tmp_path) -> None:
    """`_rag_rt()` 必须把两个档位读出来（这是注入链的唯一入口）。"""
    with mock.patch.object(runtime_settings, "_path", lambda: tmp_path / "provider_settings.json"):
        runtime_settings._state = None
        svc = object.__new__(DocumentService)
        svc._settings = SimpleNamespace(default_embedder="fake", use_bge_sparse=False,
                                        use_jieba=False)
        rt = svc._rag_rt()
        assert rt["rag_reranker_fp16"] is True
        assert rt["rag_reranker_max_length"] == 512


def test_reset_reranker_pipelines_clears_both_caches() -> None:
    """★ 只清重排器单例是不够的：管线缓存命中时不校验签名 → 必须一并清，否则设置静默失效。"""
    svc = object.__new__(DocumentService)
    svc._pipelines = {1: object(), 2: object()}
    svc._reranker = object()
    svc._reranker_sig = ("real", True, 384)

    svc.reset_reranker_pipelines()

    assert svc._pipelines == {}
    assert svc._reranker is None
    assert svc._reranker_sig is None


# ---------------- 5. 评测口径必须跟随线上档位（与 T-00 修 use_bge_sparse 同类）----------------


def test_resolve_reranker_opts_auto_follows_runtime_settings(tmp_path) -> None:
    """★ 核心反向断言：设置页关掉档位 → 评测必须跟着回到**改动前行为**。

    若这里写死（就像 T-00 之前的 `use_bge_sparse` 那样），评测会在旧档位下跑出
    "质量没变、也没提速"的数字，而**不报任何异常** —— 结论建立在错误前提上。
    """
    with mock.patch.object(runtime_settings, "_path", lambda: tmp_path / "provider_settings.json"):
        runtime_settings._state = None
        fp16, ml, src = run_eval._resolve_reranker_opts("auto")
        assert (fp16, ml) == (True, 512), "默认应与线上同源（runtime_settings 的默认档）"
        assert "同源" in src

        runtime_settings.update({"rag_reranker_fp16": False, "rag_reranker_max_length": 0})
        fp16, ml, _ = run_eval._resolve_reranker_opts("auto")
        assert (fp16, ml) == (False, 0), "设置关掉后评测必须跟着回到 fp32 + 不限长"


def test_resolve_reranker_opts_explicit_override() -> None:
    assert run_eval._resolve_reranker_opts("on")[:2] == (True, 384)
    assert run_eval._resolve_reranker_opts("off")[:2] == (False, 0)
    assert "显式覆盖" in run_eval._resolve_reranker_opts("on")[2]
    assert "显式覆盖" in run_eval._resolve_reranker_opts("off")[2]


def test_resolve_reranker_opts_unreadable_is_not_silent() -> None:
    """读配置失败必须**说明来源**，不能静默回退（静默回退正是"两套口径"的温床）。"""
    with mock.patch.object(runtime_settings, "get_state", side_effect=RuntimeError("boom")):
        fp16, ml, src = run_eval._resolve_reranker_opts("auto")
    assert (fp16, ml) == (False, 0)
    assert "不可读" in src
    assert "RuntimeError" in src


def test_factory_default_keeps_legacy_reranker_behaviour() -> None:
    """既有调用方（位置参数构造、不传档位）必须拿到**改动前行为**。"""
    f = run_eval.PipelineFactory("fake", "fast")
    assert f.reranker_fp16 is False
    assert f.reranker_max_length is None
    assert f._build_reranker() is None  # fake / fast 本就不建重排器
