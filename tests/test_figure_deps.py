"""图片增强依赖安装（2026-10-09）：静态契约 + 状态机行为。

## 为什么必须钉死 `--inexact`（本文件的第一职责）
`uv sync` 默认是**精确模式**：删除"不在目标期望集里的包"。本命令期望集 = `base + figure`，
而 **rag 的重依赖（faiss / FlagEmbedding / sentence-transformers）不在其中** ⇒
少了 `--inexact` 就会把它们**静默剪掉** ⇒ 用户的真实语义检索直接失效。
（2026-10-07 / 10-08 两度踩过同一机制；这里用静态断言防止后人"顺手删掉看起来多余的参数"。）

## 为什么状态机要有行为测试
安装是**后台线程 + 轮询**的结构，最容易的失效模式是"状态永远停在 running"或
"失败被吞掉、界面显示成功"。三个分支（成功 / 失败 / 取消）都必须被钉住。
"""

from __future__ import annotations

import threading
import time

from papermate.services import figure_deps
from papermate.services.figure_deps import AlreadyRunning, FigureDepsError, FigureInstaller


def _wait_until_done(ins: FigureInstaller, timeout: float = 5.0) -> dict:
    """轮询等到非 running（测试里用短超时；失败即断言失败，不静默）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = ins.task()
        if st["status"] != "running":
            return st
        time.sleep(0.02)
    raise AssertionError(f"安装任务未在 {timeout}s 内结束：{ins.task()}")


class TestUvArgsContract:
    """★ 静态契约：uv 参数集合（护栏本体）。"""

    def test_pins_inexact_and_figure_extra(self):
        args = figure_deps._extra_args()
        assert args[0] == "sync"
        assert "--extra" in args and "figure" in args, "必须装 figure extra"
        assert "--frozen" in args, "必须严格按 uv.lock"
        assert "--inexact" in args, (
            "少了 --inexact ⇒ uv 精确模式会把 rag 重依赖（faiss/FlagEmbedding/"
            "sentence-transformers）当'多余包'静默剪掉 ⇒ 真实语义检索失效（见模块 docstring）"
        )

    def test_does_not_install_rag_extra(self):
        """只带 figure：带 rag 会触发 4GB 重装（figure 与 rag 共享 torch，无需重复声明）。"""
        args = figure_deps._extra_args()
        extras = [args[i + 1] for i, a in enumerate(args) if a == "--extra"]
        assert extras == ["figure"], f"extra 集合应为 ['figure']，实际 {extras}"


class TestProbe:
    def test_shape(self):
        d = figure_deps.probe()
        assert set(d) == {"ocr", "table", "vlm"}
        assert d["vlm"] is True, "VLM 走已配置 LLM 通道，探测恒为可用"

    def test_needed_matches_probe(self, monkeypatch):
        monkeypatch.setattr(figure_deps, "probe", lambda: {"ocr": True, "table": True, "vlm": True})
        assert figure_deps.needed() is False
        monkeypatch.setattr(figure_deps, "probe", lambda: {"ocr": False, "table": True, "vlm": True})
        assert figure_deps.needed() is True


class TestSyncSkipsWhenReady:
    def test_no_subprocess_when_ready(self, monkeypatch):
        """已就绪 ⇒ 必须**零成本返回**（不得再跑 uv；这也是幂等性的判据）。"""
        monkeypatch.setattr(figure_deps, "needed", lambda: False)

        def boom(*a, **k):  # noqa: ANN002, ANN003
            raise AssertionError("已就绪时不得再调用 subprocess")

        monkeypatch.setattr(figure_deps.subprocess, "run", boom)
        assert "已就绪" in figure_deps.sync_figure_deps()


class TestInstaller:
    def test_success_path(self, monkeypatch):
        hold = threading.Event()
        monkeypatch.setattr(
            figure_deps, "sync_figure_deps", lambda **kw: (hold.wait(5), "ok")[1]
        )
        monkeypatch.setattr(
            figure_deps, "probe", lambda: {"ocr": True, "table": True, "vlm": True}
        )
        ins = FigureInstaller()
        ins.install()
        assert ins.task()["status"] == "running"
        hold.set()
        st = _wait_until_done(ins)
        assert st["status"] == "success"
        assert st["deps"] == {"ocr": True, "table": True, "vlm": True}, (
            "成功后必须带回三能力快照（前端据此刷新'已装/未装'）"
        )

    def test_error_path_records_message(self, monkeypatch):
        def boom(**kw):  # noqa: ANN003
            raise FigureDepsError("uv 挂了")

        monkeypatch.setattr(figure_deps, "sync_figure_deps", boom)
        monkeypatch.setattr(figure_deps, "probe", lambda: {"ocr": False, "table": False, "vlm": True})
        ins = FigureInstaller()
        ins.install()
        st = _wait_until_done(ins)
        assert st["status"] == "error"
        assert "uv 挂了" in (st["error"] or ""), "失败原因必须可见（不许静默吞掉）"

    def test_unexpected_exception_is_caught(self, monkeypatch):
        """兜底：后台线程**不许静默消失**（本项目对后台线程的既定纪律）。"""
        def boom(**kw):  # noqa: ANN003
            raise ValueError("非预期错误")

        monkeypatch.setattr(figure_deps, "sync_figure_deps", boom)
        monkeypatch.setattr(figure_deps, "probe", lambda: {"ocr": False, "table": False, "vlm": True})
        ins = FigureInstaller()
        ins.install()
        st = _wait_until_done(ins)
        assert st["status"] == "error"
        assert "ValueError" in (st["error"] or "")

    def test_cancel_path(self, monkeypatch):
        hold = threading.Event()
        monkeypatch.setattr(
            figure_deps, "sync_figure_deps", lambda **kw: (hold.wait(5), "ok")[1]
        )
        monkeypatch.setattr(figure_deps, "probe", lambda: {"ocr": False, "table": True, "vlm": True})
        ins = FigureInstaller()
        ins.install()
        ins.cancel()
        hold.set()
        st = _wait_until_done(ins)
        assert st["status"] == "cancelled"

    def test_conflict_when_running(self, monkeypatch):
        hold = threading.Event()
        monkeypatch.setattr(
            figure_deps, "sync_figure_deps", lambda **kw: (hold.wait(5), "ok")[1]
        )
        monkeypatch.setattr(figure_deps, "probe", lambda: {"ocr": True, "table": True, "vlm": True})
        ins = FigureInstaller()
        ins.install()
        try:
            try:
                ins.install()
            except AlreadyRunning as e:
                assert "正在安装" in str(e)
            else:  # pragma: no cover
                raise AssertionError("并发安装未被拒绝（应为 409）")
        finally:
            hold.set()
            _wait_until_done(ins)

    def test_idle_state_shape(self):
        st = FigureInstaller().task()
        assert st["status"] == "idle"
        assert st["elapsed_s"] == 0.0


class TestProbeConvergence:
    def test_runtime_settings_delegates_to_probe(self, monkeypatch):
        """★ 单一口径：`runtime_settings._enrich_deps` 必须委托到 `figure_deps.probe`。"""
        sentinel = {"ocr": True, "table": False, "vlm": True}
        monkeypatch.setattr(figure_deps, "probe", lambda: sentinel)
        from papermate import runtime_settings

        assert runtime_settings._enrich_deps() == sentinel
