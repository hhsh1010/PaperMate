"""图片内容增强（ER-03 Level 2）：figure_content.FigureEnricher

覆盖：
- 关闭 → 全空（不工作）
- OCR / 表格结构依赖缺失 → 优雅降级为空（不抛异常、不阻断）
- VLM：注入假 LLM → 产出图注；无 key（走真实端点）→ 降级为空
- 文档服务开启 figure_enrich 但依赖缺失 → 上传仍成功、figure chunk 仍入库（降级安全）
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lightcore.rag.figure_content import FigureEnricher


def _make_png(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    p = doc.new_page(width=200, height=200)
    p.insert_text((20, 20), "Figure 1: test chart", fontsize=10)
    pix = p.get_pixmap(dpi=80)
    path.write_bytes(pix.tobytes("png"))
    doc.close()


def _fake_llm(text: str = "图表显示精度随模型增大而上升。"):
    class _Msg:
        content = text

    class _Resp:
        message = _Msg()

    class _LLM:
        async def chat(self, messages, **kw):
            return _Resp()

    return _LLM()


def _settings(enabled: bool, methods: list[str]) -> SimpleNamespace:
    return SimpleNamespace(figure_enrich={"enabled": enabled, "methods": methods})


def test_disabled_returns_empty():
    enr = FigureEnricher(_settings(False, ["ocr", "vlm"]))
    out = asyncio.run(enr.enrich("nonexistent.png"))
    assert out == {"ocr_text": "", "table_md": "", "vlm_caption": ""}


def _force_missing(monkeypatch, mod: str) -> None:
    """强制某模块 import 失败（模拟未安装），使降级路径测试在依赖装/未装环境均稳定。"""
    import builtins
    import sys

    monkeypatch.setitem(sys.modules, mod, None)  # 强制重新 import
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == mod or name.startswith(mod + "."):
            raise ImportError(f"{mod} not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_ocr_missing_dep_degrades(tmp_path, monkeypatch):
    _force_missing(monkeypatch, "paddleocr")
    img = tmp_path / "x.png"
    _make_png(img)
    enr = FigureEnricher(_settings(True, ["ocr"]))
    out = asyncio.run(enr.enrich(img))
    assert out["ocr_text"] == ""  # paddleocr 缺失 → 降级


def test_table_missing_dep_degrades(tmp_path, monkeypatch):
    _force_missing(monkeypatch, "transformers")
    img = tmp_path / "x.png"
    _make_png(img)
    enr = FigureEnricher(_settings(True, ["table"]))
    out = asyncio.run(enr.enrich(img))
    assert out["table_md"] == ""  # transformers 缺失 → 降级


def test_vlm_uses_injected_llm(tmp_path):
    img = tmp_path / "x.png"
    _make_png(img)
    enr = FigureEnricher(_settings(True, ["vlm"]), vlm_llm=_fake_llm())
    out = asyncio.run(enr.enrich(img, kind="image_table"))
    assert "上升" in out["vlm_caption"]


def test_vlm_no_key_degrades(tmp_path):
    img = tmp_path / "x.png"
    _make_png(img)
    # 未注入 llm → 走真实 deepseek 端点；无 key 应降级为空（不抛异常、不联网）
    enr = FigureEnricher(_settings(True, ["vlm"]))
    out = asyncio.run(enr.enrich(img))
    assert out["vlm_caption"] == ""


def _capturing_llm(log: list[str]):
    """记录每次收到的 text prompt，返回固定图注（用于断言提示词条件化）。"""

    class _Resp:
        message = SimpleNamespace(content="测试图注")

    class _LLM:
        async def chat(self, messages, **kw):
            content = messages[0].content
            if isinstance(content, list):
                log.append(next((b.get("text", "") for b in content if b.get("type") == "text"), ""))
            else:
                log.append(str(content))
            return _Resp()

    return _LLM()


def test_vlm_prompt_skips_transcribe_when_local_ocr_done(tmp_path):
    """2026-08-25 去重：本地 OCR/表格已产出 → VLM 提示词禁止重复转录/还原行列；
    单独启用 vlm 时保留完整转录+还原要求。"""
    img = tmp_path / "x.png"
    _make_png(img)
    log: list[str] = []
    enr = FigureEnricher(_settings(True, ["vlm"]), vlm_llm=_capturing_llm(log))
    # 非表格图 + 本地已转录 → 禁止重复转录
    asyncio.run(enr._vlm(img, kind="figure", local_text_done=True))
    assert "勿重复转录" in log[-1]
    assert "若含文字请转录" not in log[-1]
    # 非表格图 + 单独 vlm → 保留转录要求
    asyncio.run(enr._vlm(img, kind="figure", local_text_done=False))
    assert "若含文字请转录" in log[-1]
    # 表格图 + 本地已还原 → 禁止重复还原/转录
    asyncio.run(enr._vlm(img, kind="image_table", local_text_done=True, table_done=True))
    assert "不要重复转录文字或还原行列" in log[-1]
    assert "还原其行列结构" not in log[-1]
    # 表格图 + 单独 vlm → 完整还原+转录
    asyncio.run(enr._vlm(img, kind="image_table", local_text_done=False, table_done=False))
    assert "还原其行列结构" in log[-1]


def _make_mixed_pdf(path: str) -> None:
    import pymupdf

    doc = pymupdf.open()
    p1 = doc.new_page(width=300, height=400)
    p1.insert_text((30, 30), "Table 1: Model comparison", fontsize=12)
    pix = p1.get_pixmap(clip=pymupdf.Rect(20, 50, 280, 160), dpi=120)
    png = pix.tobytes("png")
    p2 = doc.new_page(width=300, height=400)
    p2.insert_text((30, 30), "Table 2: image-based table", fontsize=11)
    p2.insert_image(pymupdf.Rect(40, 50, 270, 170), stream=png)
    p3 = doc.new_page(width=300, height=400)
    p3.insert_text((30, 30), "Figure 1: architecture", fontsize=11)
    p3.insert_image(pymupdf.Rect(40, 60, 270, 340), stream=png)
    doc.save(path)
    doc.close()


@pytest.fixture
def client_with_enrich():
    tmp = tempfile.mkdtemp(prefix="papermate_figcontent_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    os.environ["PAPERMATE_FIGURE_ENRICH"] = '{"enabled": true, "methods": ["ocr", "table", "vlm"]}'
    import papermate.config as cfg_mod

    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    os.environ.pop("PAPERMATE_FIGURE_ENRICH", None)
    cfg_mod._settings = None


def test_upload_with_figure_enrich_enabled_degrades(client_with_enrich):
    """开启 figure_enrich 但依赖缺失 → 上传仍成功、figure chunk 仍入库（降级安全）。"""
    import json as _json

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fpath = f.name
    try:
        _make_mixed_pdf(fpath)
        content = open(fpath, "rb").read()
    finally:
        os.unlink(fpath)
    r = client_with_enrich.post("/api/v1/documents", files={"file": ("mixed.pdf", content, "application/pdf")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready"
    doc_id = body["id"]
    with client_with_enrich.app.state.doc_svc._Session() as db:
        from papermate.models.db import Chunk

        fig_chunks = [
            c for c in db.query(Chunk).filter(Chunk.document_id == doc_id).all()
            if _json.loads(c.meta_json or "{}").get("chunk_type") in ("figure", "image_table")
        ]
    # figure chunk 仍入库；meta 含 ocr_text/table_md 字段（降级时为空串）
    assert len(fig_chunks) >= 2
    meta = _json.loads(fig_chunks[0].meta_json)
    assert "ocr_text" in meta and "table_md" in meta
    assert meta["ocr_text"] == "" and meta["table_md"] == ""


# --------------------------------------------------------------------------- #
# P-07（2026-10-09）：PaddleX 缓存根 / 模型源收敛
#
# 为什么必须有这组测试：这两个 env 决定 ①OCR 模型落在哪（外部残留能否污染本应用）
# ②首次下载走哪个源（国内可用性）。**设错了不会有任何运行时报错**——表现只是
# "OCR 静默不工作"或"下载极慢"，正是本项目最警惕的静默失效（P-07 端到端实测踩到过）。
# --------------------------------------------------------------------------- #
class TestPaddlexEnvDefaults:
    def _mod(self):
        from lightcore.rag import figure_content

        return figure_content

    def test_cache_home_follows_hf_root(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
        self._mod()._apply_paddlex_env_defaults()
        assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(
            tmp_path / "huggingface" / "paddlex-cache"
        ), (
            "缓存根必须与 HF **同根**（<HF_HOME>/paddlex-cache）——否则用户全局 ~/.paddlex 里的"
            "损坏残留会再次污染本应用（P-07 实测：半残模型目录 ⇒ 永久 'No valid PaddlePaddle model found'）"
        )

    def test_never_overrides_explicit_user_settings(self, monkeypatch):
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", r"D:\custom\pdx")
        monkeypatch.setenv("PADDLE_PDX_MODEL_SOURCE", "huggingface")
        self._mod()._apply_paddlex_env_defaults()
        assert os.environ["PADDLE_PDX_CACHE_HOME"] == r"D:\custom\pdx", "不得覆盖用户显式设置"
        assert os.environ["PADDLE_PDX_MODEL_SOURCE"] == "huggingface", "不得覆盖用户显式设置"

    def test_model_source_default_is_modelscope(self, monkeypatch):
        monkeypatch.delenv("PADDLE_PDX_MODEL_SOURCE", raising=False)
        self._mod()._apply_paddlex_env_defaults()
        assert os.environ["PADDLE_PDX_MODEL_SOURCE"] == "ModelScope", (
            "默认必须是 ModelScope：paddlex 自身默认 huggingface，国内首次 OCR 会慢/失败"
            "（2026-10-09 实测 ModelScope ≈41MB/s）"
        )

    def test_module_level_call_exists(self):
        """防"函数写了但没接上"：必须在**模块导入期**调用（paddlex 在导入期读这两个变量）。"""
        src = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "lightcore"
            / "rag"
            / "figure_content.py"
        ).read_text(encoding="utf-8")
        assert "\n_apply_paddlex_env_defaults()\n" in src, "模块级未调用 ⇒ 两个 env 不会生效"
