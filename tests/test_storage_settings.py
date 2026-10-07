"""存储位置（设置页改位置 + 重启迁移）的测试（2026-10-07）

★ 本文件是**迁移安全**的闸门。最关键的一条是
  :func:`test_request_change_preserves_data_root` —— 迁移失败时启动器要能
  **回退到原位置**，前提是写请求时**没有覆盖** ``data_root``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from papermate.services import storage_config as sc


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """把 ``PAPERMATE_APP_SETTINGS`` 指到临时文件（桌面端由启动器注入的那个变量）。"""
    p = tmp_path / "app-settings.json"
    monkeypatch.setenv("PAPERMATE_APP_SETTINGS", str(p))
    return p


# ───────────────────────── 配置读写 ─────────────────────────
def test_app_settings_path_requires_env(monkeypatch):
    """没有该环境变量 ⇒ 不可用（**绝不去别处乱写文件**）。"""
    monkeypatch.delenv("PAPERMATE_APP_SETTINGS", raising=False)
    assert sc.app_settings_path() is None


def test_load_is_empty_when_missing(cfg):
    assert cfg.parent.exists()
    assert sc.load() == {}


def test_load_survives_corrupt_json(cfg):
    """★ 配置损坏**绝不能抛异常** —— 它由 UI 读，抛异常等于设置页整块报错。"""
    cfg.write_text("{ 这不是 json", encoding="utf-8")
    assert sc.load() == {}


def test_save_round_trips_and_leaves_no_tmp(cfg):
    sc.save({"schema": 1, "data_root": r"D:\X"})
    assert json.loads(cfg.read_text(encoding="utf-8"))["data_root"] == r"D:\X"
    assert not cfg.with_suffix(cfg.suffix + ".tmp").exists(), "临时文件没清理干净"


def test_request_change_preserves_data_root(cfg):
    """★★ 迁移失败要能回退 ⇒ 写请求时**绝不能覆盖** ``data_root``。"""
    sc.save({"schema": 1, "data_root": r"C:\old", "extra": "keep-me"})
    sc.request_change(r"D:\new", migrate=True)
    data = sc.load()
    assert data["data_root"] == r"C:\old", "data_root 被覆盖了 ⇒ 迁移失败后无处回退"
    assert data["pending_root"] == r"D:\new"
    assert data["migrate"] is True
    assert data["extra"] == "keep-me", "未识别的键被抹掉了"


def test_request_change_without_migrate_defaults_false(cfg):
    sc.save({"schema": 1, "data_root": r"C:\old"})
    sc.request_change(r"D:\new", migrate=False)
    assert sc.load()["migrate"] is False


# ───────────────────────── 校验 ─────────────────────────
def test_validate_rejects_empty_and_relative():
    assert not sc.validate_root("").ok
    assert not sc.validate_root(r".\data").ok
    assert not sc.validate_root("data").ok


def test_validate_rejects_unc():
    r = sc.validate_root(r"\\server\share\PaperMate")
    assert not r.ok
    assert any("网络" in e for e in r.errors)


def test_validate_rejects_unknown_drive():
    r = sc.validate_root("Z:\\PaperMate")
    assert not r.ok, f"不存在的盘应被拒：{r.errors}"


def test_validate_rejects_network_drive_if_present():
    """真实网络映射盘（Z: 之类）必须被拒 —— SQLite 在其上锁不可靠。"""
    remote = [d for d in sc.list_drives() if d.kind == "REMOTE"]
    if not remote:
        pytest.skip("本机没有网络映射盘，跳过")
    r = sc.validate_root(remote[0].root + "PaperMate")
    assert not r.ok
    assert any("网络" in e for e in r.errors)


def test_validate_rejects_path_under_a_file(tmp_path):
    """★ 真实存在的失败模式：父路径是个**文件** ⇒ create_dir 必失败。"""
    f = tmp_path / "afile"
    f.write_text("x", encoding="utf-8")
    r = sc.validate_root(str(f / "sub"))
    assert not r.ok, f"文件下面的路径应被拒：{r.errors}"


def test_validate_rejects_insufficient_space(tmp_path, monkeypatch):
    """余量不足必须被拦（monkeypatch 掉 `free_gb` 造出"小盘"）。"""
    monkeypatch.setattr(sc, "free_gb", lambda _p: 1.0)
    r = sc.validate_root(str(tmp_path / "ok"))
    assert not r.ok
    assert any("GB" in e for e in r.errors)


def test_validate_accepts_a_normal_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "free_gb", lambda _p: 100.0)
    r = sc.validate_root(str(tmp_path / "ok"))
    assert r.ok, r.errors
    assert (tmp_path / "ok").is_dir(), "校验通过就该把目录建出来"


def test_validate_leaves_no_probe_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "free_gb", lambda _p: 100.0)
    sc.validate_root(str(tmp_path / "ok"))
    assert not (tmp_path / "ok" / ".papermate-write-probe").exists()


def test_validate_rejects_inside_install_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "free_gb", lambda _p: 100.0)
    install = tmp_path / "install"
    install.mkdir()
    r = sc.validate_root(str(install / "data"), install_dir=install)
    assert not r.ok
    assert any("安装" in e or "应用目录" in e for e in r.errors)


# ───────────────────────── 体积 / 磁盘 ─────────────────────────
def test_dir_size_mb_truncates_with_file_cap(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.bin").write_bytes(b"x" * 1024)
    mb, truncated = sc.dir_size_mb(tmp_path, max_files=2)
    assert truncated is True, "超过文件数上限必须标注截断（否则数字会被当成真值）"
    assert mb > 0


def test_dir_size_mb_not_truncated_when_small(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 2048)
    mb, truncated = sc.dir_size_mb(tmp_path)
    assert truncated is False
    assert 0.001 < mb < 1.0


def test_list_drives_on_windows():
    """本机应至少列出一个磁盘，且带可用容量。"""
    drives = sc.list_drives()
    if os.name != "nt":
        assert drives == []
        return
    assert drives, "Windows 上没列出任何磁盘"
    assert all(d.root.endswith("\\") for d in drives)
    assert all(d.free_gb >= 0 for d in drives)


def test_sub_dirs_layout_matches_rust():
    """★ 与 `lib.rs` 的 SUBDIRS 必须逐字一致（Rust 侧同名常量）。"""
    assert sc.SUBDIRS == ("data", "huggingface", "venv", "uv-cache")


def test_min_free_gb_matches_rust():
    """★ 与 `lib.rs::MIN_FREE_GB` 一致（两边不同步就会口径不一）。"""
    lib = (Path(__file__).resolve().parents[1] / "desktop" / "src-tauri" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "const MIN_FREE_GB: f64 = 8.0;" in lib
    assert sc.MIN_FREE_GB == 8.0