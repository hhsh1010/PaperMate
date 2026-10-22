"""GPU 自适应设备探测测试（auto_device + 嵌入器/重排器接线）。

不加载任何真实权重：FlagEmbedding / sentence_transformers 以假模块注入 sys.modules，
只断言构造参数（devices / use_fp16 / device）是否按设备自适应正确传递。
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeFlagModel:
    """捕获构造参数的 BGEM3FlagModel 替身。"""

    last_kwargs: dict | None = None
    last_args: tuple | None = None
    dim = 1024

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        type(self).last_args = args
        type(self).last_kwargs = kwargs


class _FakeCrossEncoder:
    """捕获构造参数的 ST CrossEncoder 替身。"""

    last_kwargs: dict | None = None

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        type(self).last_kwargs = kwargs


@pytest.fixture()
def fake_flagembedding(monkeypatch):
    mod = types.ModuleType("FlagEmbedding")
    mod.BGEM3FlagModel = _FakeFlagModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)
    _FakeFlagModel.last_kwargs = None
    return _FakeFlagModel


@pytest.fixture()
def fake_st(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    _FakeCrossEncoder.last_kwargs = None
    return _FakeCrossEncoder


class TestAutoDevice:
    def test_env_override_wins(self, monkeypatch):
        from lightcore.rag.hf_local import auto_device

        monkeypatch.setenv("PAPERMATE_DEVICE", "cuda:1")
        assert auto_device() == "cuda:1"

    def test_falls_back_to_cpu_without_cuda(self, monkeypatch):
        """沙箱/CI 无 CUDA：探测结果必须回到 cpu（torch 可用但 cuda=False）。"""
        from lightcore.rag.hf_local import auto_device

        monkeypatch.delenv("PAPERMATE_DEVICE", raising=False)
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # 与机器解耦
        assert auto_device() == "cpu"

    def test_matches_torch_availability(self, monkeypatch):
        from lightcore.rag.hf_local import auto_device

        monkeypatch.delenv("PAPERMATE_DEVICE", raising=False)
        import torch

        assert auto_device() == ("cuda" if torch.cuda.is_available() else "cpu")


class TestBgeM3EmbedderDevice:
    def test_explicit_cuda_enables_fp16(self, fake_flagembedding):
        from lightcore.rag.embedder import BgeM3Embedder

        emb = BgeM3Embedder(model_name="/fake/path", device="cuda")
        assert emb.device == "cuda"
        assert fake_flagembedding.last_kwargs["devices"] == "cuda"
        assert fake_flagembedding.last_kwargs["use_fp16"] is True

    def test_cpu_keeps_fp32(self, fake_flagembedding):
        from lightcore.rag.embedder import BgeM3Embedder

        emb = BgeM3Embedder(model_name="/fake/path", device="cpu")
        assert emb.device == "cpu"
        assert fake_flagembedding.last_kwargs["devices"] == "cpu"
        assert fake_flagembedding.last_kwargs["use_fp16"] is False

    def test_auto_detects_no_cuda_env(self, fake_flagembedding, monkeypatch):
        """device 缺省时走 auto_device()：无 CUDA（mock）必须落 cpu + fp32（与旧默认等价）。"""
        import torch

        from lightcore.rag.embedder import BgeM3Embedder

        monkeypatch.delenv("PAPERMATE_DEVICE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        emb = BgeM3Embedder(model_name="/fake/path")
        assert emb.device == "cpu"
        assert fake_flagembedding.last_kwargs["devices"] == "cpu"
        assert fake_flagembedding.last_kwargs["use_fp16"] is False


class TestCrossEncoderRerankerDevice:
    def test_explicit_device_passthrough(self, fake_st):
        from lightcore.rag.rerank import CrossEncoderReranker

        r = CrossEncoderReranker(model_name="/fake/path", device="cuda")
        assert r.device == "cuda"
        assert _FakeCrossEncoder.last_kwargs["device"] == "cuda"

    def test_auto_detects_no_cuda_env(self, fake_st, monkeypatch):
        import torch

        from lightcore.rag.rerank import CrossEncoderReranker

        monkeypatch.delenv("PAPERMATE_DEVICE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        r = CrossEncoderReranker(model_name="/fake/path")
        assert r.device == "cpu"
        assert _FakeCrossEncoder.last_kwargs["device"] == "cpu"


class TestArchSupported:
    """★ 2026-10-22：RTX 5060（Blackwell sm_120）在 torch+cu126 上 is_available()=True
    但**没有 kernel** ⇒ 推理时报 no kernel image ⇒ 对话 500。这里把判据钉死。
    """

    #: 本机 torch 2.13.0+cu126 的真实列表
    CU126 = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]

    def test_supported_arch_returns_true(self):
        from lightcore.rag.hf_local import _arch_supported

        assert _arch_supported((8, 6), self.CU126) is True
        assert _arch_supported((9, 0), self.CU126) is True

    def test_same_major_minor_forward_compatible(self):
        """★ sm_89（RTX 40 系）在只有 sm_86 kernel 的 torch 上**能跑**（真实机器验证过）。
        判据必须是「同 major 且 minor ≤」而不是「完全相等」，否则会把这类机器误判成不支持 ✗。
        """
        from lightcore.rag.hf_local import _arch_supported

        assert _arch_supported((8, 9), self.CU126) is True

    def test_rtx50_sm120_not_supported_by_cu126(self):
        """★ 真实故障：sm_120 不在 cu126 的列表里 ⇒ 必须判 False（从而回退 CPU）。"""
        from lightcore.rag.hf_local import _arch_supported

        assert _arch_supported((12, 0), self.CU126) is False

    def test_too_old_arch_also_false(self):
        from lightcore.rag.hf_local import _arch_supported

        assert _arch_supported((3, 5), self.CU126) is False

    def test_arch_suffix_tolerated(self):
        """sm_90a 这类带后缀的条目要能匹配 sm_90。"""
        from lightcore.rag.hf_local import _arch_supported

        assert _arch_supported((9, 0), ["sm_90a"]) is True

    def test_new_torch_supports_sm120(self):
        """将来换 cu128（含 sm_120）时自动放行，不用再改代码。"""
        from lightcore.rag.hf_local import _arch_supported

        assert _arch_supported((12, 0), [*self.CU126, "sm_120"]) is True
