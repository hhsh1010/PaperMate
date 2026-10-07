"""venv / uv-cache 路径解析的契约测试（2026-10-07）

## 为什么单独钉这一组

`rag_deps.venv_dir()` / `uv_cache_dir()` 原先**只**从 `data_dir.parent` 推导路径。
一旦「存储位置可配置」上线（`data_dir = <root>/data`），`parent` 就变成 `<root>`，
应用会去找 **`<root>/venv`** —— 若那儿没有 venv，
**「准备运行环境」与「下载模型」会静默失效**（只报一句"未找到应用私有 venv"）。

⇒ 因此本组测试钉死两件事：
  1. **显式通道优先**（`PAPERMATE_VENV_DIR` / `PAPERMATE_UV_CACHE_DIR` → `UV_PROJECT_ENVIRONMENT` / `UV_CACHE_DIR`）；
  2. **未设任何 env 时，回退值必须与今天逐字节相同**（默认安全 —— 不许悄悄改默认行为）。

★ 本文件零网络、毫秒级，进日常门禁。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from papermate.services import rag_deps

#: 与 `lib.rs` 注入的四个 env 同名（改 Rust 侧时这里要同步）
_ENV_VARS = (
    "PAPERMATE_VENV_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "PAPERMATE_UV_CACHE_DIR",
    "UV_CACHE_DIR",
)


@pytest.fixture
def clean_env(monkeypatch):
    """清掉四个 env，避免测试依赖运行环境里残留的值（"隔离必须默认安全"）。"""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def fake_settings(monkeypatch):
    """把 `get_settings()` 换成可控对象（`rag_deps` 里是**函数内 import**，故打在源模块上有效）。"""

    def _install(data_dir: Path):
        class _S:
            resolved_data_dir = data_dir

        import papermate.config as cfg

        monkeypatch.setattr(cfg, "get_settings", lambda: _S())
        return data_dir

    return _install


# ─────────────────────────── venv ───────────────────────────
def test_papermate_venv_dir_wins_over_uv_project_environment(clean_env):
    """★ 核心：App 显式指定的位置必须**最优先**（这是"存储位置可配置"的地基）。"""
    clean_env.setenv("PAPERMATE_VENV_DIR", r"D:\RootX\venv")
    clean_env.setenv("UV_PROJECT_ENVIRONMENT", r"C:\Other\venv")
    assert rag_deps.venv_dir() == Path(r"D:\RootX\venv")


def test_uv_project_environment_still_honored(clean_env):
    """旧通道不能被破坏（桌面端一直用它）。"""
    clean_env.setenv("UV_PROJECT_ENVIRONMENT", r"C:\Legacy\venv")
    assert rag_deps.venv_dir() == Path(r"C:\Legacy\venv")


def test_venv_dir_falls_back_to_data_dir_parent(clean_env, fake_settings, tmp_path):
    """★ 默认安全：未设任何 env 时，回退值必须与今天**完全一致** = `<data_dir 的上级>/venv`。

    ★ 注意 `cand.exists()` 的门槛（历史行为）：目录不存在时返回 None。
      这不是笔误 —— 它让调用方走"解析不出"的降级分支，必须一起钉住。
    """
    root = tmp_path / "root"
    (root / "venv").mkdir(parents=True)
    fake_settings(root / "data")
    assert rag_deps.venv_dir() == root / "venv"


def test_venv_dir_returns_none_when_fallback_missing(clean_env, fake_settings, tmp_path):
    """回退目录不存在 ⇒ None（如实降级，不返回一个假路径）。"""
    fake_settings(tmp_path / "root" / "data")
    assert rag_deps.venv_dir() is None


def test_data_root_layout_still_resolves_via_explicit_env(clean_env, fake_settings, tmp_path):
    """★★ 地雷的回归护栏：数据根目录布局下（`data_dir = <root>/data`），
    **只要** App 注入了 `PAPERMATE_VENV_DIR`，就必须解析到它 ——
    而不是被 `data_dir.parent` 带到 `<root>/venv` 之外的地方。"""
    root = tmp_path / "D_drive" / "PaperMate"
    (root / "venv").mkdir(parents=True)
    fake_settings(root / "data")  # ← 只看 settings 会推出 <root>/venv
    clean_env.setenv("PAPERMATE_VENV_DIR", str(root / "venv"))
    assert rag_deps.venv_dir() == root / "venv"


# ─────────────────────────── uv-cache ───────────────────────────
def test_papermate_uv_cache_dir_wins(clean_env):
    clean_env.setenv("PAPERMATE_UV_CACHE_DIR", r"D:\RootX\uv-cache")
    clean_env.setenv("UV_CACHE_DIR", r"C:\Other\uv-cache")
    assert rag_deps.uv_cache_dir() == Path(r"D:\RootX\uv-cache")


def test_uv_cache_dir_env_now_honored(clean_env):
    """★ 修正"两把尺子"：`uv_cache_dir()` 原先**不读** `UV_CACHE_DIR`，
    而 `sync_heavy_deps()` 实际用的却是它 ⇒ "报出去的路径" ≠ "实际用的路径"。
    现在两者同源。"""
    clean_env.setenv("UV_CACHE_DIR", r"C:\Explicit\uv-cache")
    assert rag_deps.uv_cache_dir() == Path(r"C:\Explicit\uv-cache")


def test_uv_cache_dir_falls_back_to_data_dir_parent(clean_env, fake_settings, tmp_path):
    """默认安全：未设 env 时回退到 `<data_dir 的上级>/uv-cache`（与今天一致，且**不要求目录存在**）。"""
    root = tmp_path / "root"
    fake_settings(root / "data")
    assert rag_deps.uv_cache_dir() == root / "uv-cache"


def test_describe_target_reports_the_same_venv_it_would_use(clean_env, fake_settings, tmp_path):
    """★ 同源性：`describe_target()` **报出去**的 venv 路径，必须等于**真正会被使用**的那个。

    ★ 注意 `uv_path` 是 **uv.exe 的位置**（与 uv-cache 无关），不要混淆 ——
      本测试只断言 venv 这一对，避免"比错东西"（那是"两把尺子"的另一种形态）。
    """
    root = tmp_path / "root"
    (root / "venv").mkdir(parents=True)
    fake_settings(root / "data")
    clean_env.setenv("PAPERMATE_VENV_DIR", str(root / "venv"))

    reported = rag_deps.describe_target()["venv_path"]
    assert reported is not None
    assert Path(reported) == rag_deps.venv_dir()
