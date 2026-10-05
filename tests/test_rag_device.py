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
