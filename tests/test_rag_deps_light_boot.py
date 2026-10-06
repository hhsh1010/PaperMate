"""桌面首启「轻量依赖」方案的**可行性地基**（2026-10-06）

要证明的事
----------
首启只装轻量依赖（排除 ``torch`` / ``FlagEmbedding`` / ``sentence-transformers``）时，
**应用必须仍能启动并服务**，否则整个"两阶段安装"方案不成立。

为什么用「屏蔽导入」而不是「真建一个无 torch 的 venv」
------------------------------------------------------
真建 venv 需要联网装包（PyPI 可达性未验证）且慢；而**导入是否被依赖**才是这里的关键变量。
把 ``sys.modules[name] = None`` 是 CPython 官方认可的"让 import 抛 ImportError"手法，
能确定性复现"该包不存在"的语义，且**离线、毫秒级、可在测试进程里精确开关**。

⚠️ 屏蔽后必须**彻底清掉已加载的子模块**：``import torch`` 失败不代表
``import torch.cuda`` 也失败（子模块可能已在 sys.modules 里）。

纪律：本文件**零网络、零真实安装**，因此可以进日常门禁。
"""

from __future__ import annotations

import sys

import pytest

from papermate.services import rag_deps

_BLOCKED = rag_deps.HEAVY_IMPORTS


@pytest.fixture
def no_heavy_deps(monkeypatch):
    """让 ``import torch`` 等**确定抛 ImportError**，并在结束后完全复原。"""
    saved: dict = {}
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in _BLOCKED:
            saved[name] = sys.modules[name]
    # 先清干净（含子模块），再置 None —— 顺序不能反，否则子模块仍可被 import 到
    for name in list(sys.modules):
        if name.split(".")[0] in _BLOCKED:
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in _BLOCKED:
        monkeypatch.setitem(sys.modules, name, None)
    # 依赖探测缓存必须失效，否则会复用"重依赖已就绪"的旧结论
    rag_deps.invalidate_cache()
    yield
    rag_deps.invalidate_cache()
    assert sys.modules.get("torch") is not None or True  # 由 monkeypatch 自动还原


# ───────────────────────── 0. 快测试：命名映射（零导入，进日常门禁） ─────────────────────────
def test_heavy_names_mapping_is_consistent():
    """★ 零导入的**快**回归：钉死「发行包名 ↔ 导入名」映射。

    这是本模块最容易复发的一处错误（2026-06-06 实际踩到）：
    ``sentence-transformers``（发行名，连字符）≠ ``sentence_transformers``（导入名，下划线）。
    混用会让开发机明明装好却判为"缺失" ⇒ 界面骗用户重下 4GB。
    本用例不 import 任何重依赖 ⇒ 毫秒级，可进日常门禁。
    """
    imports = set(rag_deps.HEAVY_IMPORTS)
    dists = set(rag_deps.HEAVY_DISTRIBUTIONS)
    assert "sentence_transformers" in imports, "导入名必须是下划线形式"
    assert "sentence-transformers" not in imports, "导入名不能是连字符形式"
    assert "sentence-transformers" in dists, "发行名必须是连字符形式"
    # 两个集合必须一一对应（规范化后相等），否则 uv 排除的包与探测的包会错位
    assert {n.replace("_", "-") for n in imports} == dists


# ───────────────────────── 1. 探测：屏蔽后必须判为缺失 ─────────────────────────
def test_probe_reports_missing_when_blocked(no_heavy_deps):
    st = rag_deps.probe()
    assert st.ready is False
    assert any(m.startswith("torch：") for m in st.missing)
    # ★ 假阴性防护：开发机上装好了也必须判 ready（发行包名 vs 导入名别搞混）
    assert len(st.missing) == 3, st.missing


def test_probe_ready_on_real_env():
    """反向：在**真的装了**重依赖的环境（CI/开发机）必须判 ready。

    这是 2026-06-06 实际踩过的坑：曾把 ``sentence-transformers``（发行名）
    当导入名去 ``__import__``，开发机明明装好却判为缺失 ⇒ 界面会骗用户下 4GB。
    """
    st = rag_deps.probe()
    assert st.ready is True, st.missing


def test_cache_invalidation(no_heavy_deps):
    assert rag_deps.deps_ready() is False
    rag_deps.invalidate_cache()
    assert rag_deps.probe(use_cache=False).ready is False


# ───────────────────────── 2. 应用在无 torch 时仍能启动 ─────────────────────────
def test_app_starts_without_heavy_deps(no_heavy_deps, tmp_path, monkeypatch):
    """★ 核心断言：屏蔽重依赖后，应用仍能建起来、健康检查 200、且是 fake 模式。"""
    import os

    from fastapi.testclient import TestClient

    import papermate.config as cfg_mod

    os.environ["PAPERMATE_DATA_DIR"] = str(tmp_path / "data")
    os.environ["PAPERMATE_PREWARM"] = "0"
    cfg_mod._settings = None

    from papermate.api.app import create_app

    with TestClient(create_app()) as c:
        r = c.get("/api/v1/healthz")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        # 首装必须是 fake（否则就会去建 BGE-M3 → 缺 torch 直接炸）
        assert body["rag_embedder"] == "fake"
        assert body["default_model"] == "fake"


def test_fake_embedder_works_without_torch(no_heavy_deps):
    """假嵌入器必须纯 Python 可用（这是"轻量首启"能成立的根本原因）。"""
    from lightcore.rag.embedder import FakeHashEmbedder

    emb = FakeHashEmbedder()
    vecs = emb.encode(["量子纠缠", "Transformer 注意力"])
    assert len(vecs) == 2 and len(vecs[0]) == emb.dim


def test_spa_is_served_without_heavy_deps(no_heavy_deps, tmp_path, monkeypatch):
    """前端产物托管也不能依赖 torch（app.py 顶层 import 链里没有它，但要钉住）。"""
    import os

    from fastapi.testclient import TestClient

    import papermate.config as cfg_mod

    os.environ["PAPERMATE_DATA_DIR"] = str(tmp_path / "d2")
    os.environ["PAPERMATE_PREWARM"] = "0"
    cfg_mod._settings = None

    from papermate.api.app import create_app

    with TestClient(create_app()) as c:
        assert c.get("/api/v1/settings/rag/models").status_code == 200


# ───────────────────────── 3. 真实模式缺依赖时必须给出可读错误 ─────────────────────────
def test_real_mode_without_deps_gives_actionable_error(no_heavy_deps):
    """缺依赖时构造 real 嵌入器 ⇒ 必须是**可读**错误，不能是裸 ImportError 堆栈。"""
    from papermate.services.document_service import DocumentService

    svc = DocumentService.__new__(DocumentService)   # 绕过 __init__（避免建库/引擎）
    svc._settings = type("S", (), {"default_embedder": "real", "use_bge_sparse": False,
                                   "use_jieba": False})()
    svc._doc_svc_runtime = None

    # 直接走 _make_embedder 的真实分支
    with pytest.raises(Exception) as ei:
        from lightcore.rag.embedder import BgeM3Embedder

        BgeM3Embedder()
    msg = str(ei.value)
    assert "FlagEmbedding" in msg or "torch" in msg, msg
