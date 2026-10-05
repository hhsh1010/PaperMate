"""检索配置变更的**静默失效**修复（2026-10-04 O4/O5/O6）。

## 三个缺陷的共性
都属于「**改了设置 / 换了权重，但系统仍在用旧的**」——**没有任何报错**，
用户只能靠"感觉没生效"发现问题。这比崩溃更难排查。

| 编号 | 缺陷 | 影响层 |
|---|---|---|
| O4 | 改 `rag_embedder` / `use_bge_sparse` / `use_jieba` **不清** `_pipelines` | 内存缓存 |
| O5 | 索引签名不含 `PAPERMATE_BGE_M3_PATH` | 磁盘索引（`load_state`） |
| O6 | 设置页回显的 fallback 与真实默认不一致（**死默认值**） | 对外视图 |

## ★ O4 与 O5 为什么**互补、缺一不可**
- `get_pipeline` 命中 `_pipelines` 就**直接 return，根本不校验签名**
  ⇒ 内存缓存只能靠**显式清理**（O4）；
- 签名只用于 `load_state` 的**磁盘索引**判定 ⇒ 磁盘层靠**签名失效**（O5）。

只修一个 ⇒ 另一半仍然静默用旧配置/旧向量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from papermate import runtime_settings
from papermate.services.document_service import _BGE_M3_PATH_ENV, _embedder_identity

# 与 `config.Settings` 的真实默认值同源（`config.py`：`use_bge_sparse = True`）
_CFG = pytest.importorskip("papermate.config")


class TestO5EmbedderIdentity:
    """★ O5：换权重目录必须让「嵌入器身份」改变。"""

    _RT = {"rag_embedder": "real", "use_bge_sparse": True, "use_jieba": True}

    def test_env_change_changes_identity(self, monkeypatch):
        monkeypatch.delenv(_BGE_M3_PATH_ENV, raising=False)
        before = _embedder_identity(self._RT)
        monkeypatch.setenv(_BGE_M3_PATH_ENV, r"C:\weights\bge-m3-v2")
        after = _embedder_identity(self._RT)
        assert before != after, "★ 换权重目录后身份未变 ⇒ 旧向量会被继续复用（检索静默出错）"

    def test_blank_env_treated_as_unset(self, monkeypatch):
        """空白/未设 env 必须等价 —— 否则重启后身份会莫名变化。"""
        monkeypatch.delenv(_BGE_M3_PATH_ENV, raising=False)
        a = _embedder_identity(self._RT)
        monkeypatch.setenv(_BGE_M3_PATH_ENV, "   ")
        assert _embedder_identity(self._RT) == a

    def test_model_name_still_part_of_identity(self, monkeypatch):
        """换模型名也必须改变身份（不能只盯 env）。"""
        monkeypatch.delenv(_BGE_M3_PATH_ENV, raising=False)
        a = _embedder_identity(self._RT)
        b = _embedder_identity({**self._RT, "rag_embedder": "fake"})
        assert a != b

    def test_pipeline_signature_uses_identity(self, monkeypatch):
        """★★ 磁盘索引签名必须随权重变（这是 O5 的实际落点）。"""
        from papermate.services.document_service import DocumentService

        obj = object.__new__(DocumentService)  # 只调纯函数，不需要完整状态
        items = [("doc-1-c0", "正文", 1)]
        rt = self._RT
        monkeypatch.delenv(_BGE_M3_PATH_ENV, raising=False)
        before = obj._pipeline_signature(items, rt)
        monkeypatch.setenv(_BGE_M3_PATH_ENV, r"C:\weights\bge-m3-v2")
        after = obj._pipeline_signature(items, rt)
        assert before != after, "★ 索引签名未随权重变 ⇒ load_state 会命中旧向量"

    def test_signature_still_tracks_other_settings(self, monkeypatch):
        """★ 加了 env 之后**不能丢掉**原有的三项（分块内容 + sparse + jieba）。"""
        from papermate.services.document_service import DocumentService

        obj = object.__new__(DocumentService)
        items = [("doc-1-c0", "正文", 1)]
        rt = self._RT
        monkeypatch.delenv(_BGE_M3_PATH_ENV, raising=False)
        base = obj._pipeline_signature(items, rt)
        assert base != obj._pipeline_signature(items, {**rt, "use_bge_sparse": False})
        assert base != obj._pipeline_signature(items, {**rt, "use_jieba": False})
        assert base != obj._pipeline_signature([("doc-1-c0", "改了", 1)], rt)


class TestO4PipelineCacheInvalidation:
    """★ O4：改检索设置必须清 `_pipelines`（内存层唯一手段）。"""

    #: 真实代码里参与"是否清管线"判定的**全部字段**（`routes_settings.py`）。
    #: ⚠️ 必须显式列出：真实代码是 `any(v is not None for v in (这五项))`，
    #: **无关字段根本不参与判定** ⇒ 镜像函数也必须只挑这些键，否则测试会误判。
    RESET_KEYS = (
        "rag_embedder",
        "use_bge_sparse",
        "use_jieba",
        "rag_reranker_fp16",
        "rag_reranker_max_length",
    )

    @classmethod
    def _needs_reset(cls, body: dict) -> bool:
        """与 `routes_settings` 里那段判定**同源**的镜像（改一处必须同步改测试）。"""
        return any(body.get(k) is not None for k in cls.RESET_KEYS)

    def test_rag_settings_must_reset(self):
        for key in self.RESET_KEYS:
            sentinel = "x" if "embedder" in key or "length" in key else True
            assert self._needs_reset({key: sentinel}), (
                f"★ 改 `{key}` 未触发管线清理 ⇒ 改设置后静默不生效"
            )

    def test_unrelated_settings_need_no_reset(self):
        """无关设置不该白清缓存（清管线 = 下次重建 ≈ 100ms/篇）。"""
        assert not self._needs_reset({"cost_limit_per_turn": 2.0, "default_model": "qwen3.6-flash"})

    def test_none_means_not_provided(self):
        """`None` = 未提供（PATCH 语义）⇒ 不该触发清理。"""
        assert not self._needs_reset({k: None for k in self.RESET_KEYS})

    def test_source_contract(self):
        """源码契约：清理条件必须包含那五项，且不再是"仅重排两项"。"""
        src = (Path(__file__).resolve().parents[1] /
               "src" / "papermate" / "api" / "routes_settings.py").read_text(encoding="utf-8")
        for key in ("body.rag_embedder", "body.use_bge_sparse", "body.use_jieba"):
            assert key in src, f"★ 清理条件漏了 `{key}`"
        # 旧写法（只看重排两项）必须已消失
        assert (
            "if body.rag_reranker_fp16 is not None or body.rag_reranker_max_length is not None:"
            not in src
        ), "★ 仍是旧的「仅重排两项」判定"


def _run_with_app(cases: list[tuple[str, dict, bool]]) -> None:
    """真正发起请求的那一段（隔离由调用方负责）。"""
    from fastapi.testclient import TestClient

    from papermate.api.app import create_app

    class SpyDocSvc:
        def __init__(self) -> None:
            self.calls = 0

        def reset_reranker_pipelines(self) -> None:
            self.calls += 1

    with TestClient(create_app()) as client:
        spy = SpyDocSvc()
        client.app.state.doc_svc = spy
        for label, payload, expect in cases:
            spy.calls = 0
            resp = client.put("/api/v1/settings", json=payload)
            assert resp.status_code == 200, f"{label}: HTTP {resp.status_code}"
            got = spy.calls > 0
            assert got == expect, (
                f"★ {label}: 清管线={got}，期望 {expect}"
                + ("（改检索设置必须清，否则改了像没改）" if expect
                   else "（无关设置不该白清：清管线 = 重建 ≈100ms/篇）")
            )


class TestO4EndToEnd:
    """★★ O4 的**行为级**验证（2026-10-04 补）：真实打 API，用 spy 记录是否清了管线。

    ## 为什么还要这一类（不只有源码契约）
    上面的 `TestO4PipelineCacheInvalidation` 是**镜像函数 + 源码文本断言**，
    能守住"判定条件写了什么"，但守不住：
    - `SettingsUpdate` 的字段名改了（`body.use_bge_sparse` 拼错会静默变成 `None`）；
    - `request.app.state.doc_svc` 取不到（清了等于没清）。
    ⇒ **只有真实调用 API + 记录副作用，才算"行为真变了"**。
    （同项目既有教条：`?raw` 契约只守结构，守不住逻辑。）
    """

    @staticmethod
    def _run(cases: list[tuple[str, dict, bool]]) -> None:
        """跑一组"改设置是否清管线"的用例。

        ★★ **必须自己做完整隔离**（2026-10-04 踩到）：
        本用例会**真实 `PUT /settings`** ⇒ 会写 `provider_settings.json`。
        初版只用了 `os.environ.setdefault("PAPERMATE_DATA_DIR", mkdtemp())`，
        而前面的用例已经设过该 env ⇒ `setdefault` **不生效** ⇒ 改动落进了**共享**的
        临时目录 ⇒ 污染了后面 `TestO6` 的 `get_state()`（实测 `use_jieba` 被改成 False
        ⇒ 断言 `False == True` 失败）。
        ⚠️ 幸亏 conftest 有模块级兜底（`data_dir=Temp/papermate_pytest_proc_*`），
        否则写进的就是**真实 `data/`** —— 那是"测试改坏用户配置"级别的事故。
        ⇒ 教训：**会写盘的测试必须自己强制隔离 + 用完恢复，不能依赖 setdefault**。
        """
        import os
        import tempfile

        import papermate.config as cfg_mod
        from papermate import runtime_settings as rs_mod

        saved_env = os.environ.get("PAPERMATE_DATA_DIR")
        tmp = tempfile.mkdtemp(prefix="papermate_o4_e2e_")
        os.environ["PAPERMATE_DATA_DIR"] = tmp          # ★ 强制，不用 setdefault
        cfg_mod._settings = None
        saved_state = getattr(rs_mod, "_state", None)
        rs_mod._state = None
        try:
            _run_with_app(cases)
        finally:
            if saved_env is None:
                os.environ.pop("PAPERMATE_DATA_DIR", None)
            else:
                os.environ["PAPERMATE_DATA_DIR"] = saved_env
            cfg_mod._settings = None
            rs_mod._state = saved_state                # ★ 还原，避免污染后续用例

    def test_rag_settings_trigger_reset(self):
        """★ 三项新纳入的检索设置，各改一次都必须触发清管线。"""
        self._run([
            ("use_bge_sparse", {"use_bge_sparse": False}, True),
            ("use_jieba", {"use_jieba": False}, True),
            ("rag_embedder", {"rag_embedder": "fake"}, True),
        ])

    def test_unrelated_setting_does_not_reset(self):
        """无关设置不该清 —— 清管线意味着下次重建（≈100ms/篇）。"""
        self._run([("cost_limit_per_turn", {"cost_limit_per_turn": 2.0}, False)])


class TestO6PublicViewFallback:
    """★ O6：**不是**修 bug，是消除与真实默认不一致的死默认值。"""

    def test_fallback_matches_real_default(self):
        """★ 审查报告曾称"回显显示 False 而实际在跑 True"——**实测不成立**。

        本测试钉住的是**更值得防的事**：fallback 与 `config.Settings` 的真实默认保持一致，
        免得后人读 `False` 误以为"默认关闭"。
        """
        view = runtime_settings.public_view()
        assert view["use_bge_sparse"] == bool(_CFG.get_settings().use_bge_sparse)
        assert view["use_jieba"] == bool(_CFG.get_settings().use_jieba)

    def test_state_keys_always_present(self):
        """★ `_load()` 保证这些 key 恒存在 ⇒ `state.get` 的 fallback **永不可达**。

        这解释了"回显错误"为何不成立：fallback 从来不会被取到。
        """
        st = runtime_settings.get_state()
        for key in ("rag_embedder", "use_bge_sparse", "use_jieba"):
            assert key in st, f"★ `{key}` 不在 state 中 ⇒ 回显会落到 fallback"

    def test_source_has_no_hardcoded_false_fallback(self):
        """源码契约：不得再出现 `state.get("use_bge_sparse", False)` 这种硬编码回落。"""
        src = (Path(__file__).resolve().parents[1] /
               "src" / "papermate" / "runtime_settings.py").read_text(encoding="utf-8")
        assert 'state.get("use_bge_sparse", False)' not in src
        assert 'state.get("use_jieba", False)' not in src
