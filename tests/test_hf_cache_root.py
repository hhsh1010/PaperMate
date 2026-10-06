"""HF 缓存根目录解析（2026-10-06 首装仿真实测出的**静默失效**回归）

缺陷
----
`resolve_local_hf_snapshot()` 早期硬编码 `~/.cache/huggingface/hub/...`，**无视** `HF_HOME` /
`HF_HUB_CACHE`；而 `huggingface_hub` 本身是尊重这两个变量的。于是出现
**「下载落在一处、探测去找另一处」**：

- 桌面客户端把 `HF_HOME` 指向 `%LOCALAPPDATA%\\PaperMate\\huggingface`（`desktop/src-tauri/src/lib.rs`），
  新装机器上那里是空的 ⇒ 但探测仍去用户主目录找；
- 开发机主目录**恰好有**旧缓存 ⇒ 首装仿真里设置页误报「权重已就绪」，**测不出来**。

本文件把口径钉死：**探测位置 == 下载位置 == 磁盘检查位置**（三者共用
`default_hub_cache_dir()`）。
"""

from __future__ import annotations

import os

from lightcore.rag.hf_local import default_hub_cache_dir, resolve_local_hf_snapshot

_REPO = "BAAI/bge-m3"
_SAFE = "BAAI--bge-m3"


def _clear(monkeypatch) -> None:
    for k in ("HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(k, raising=False)


def test_hub_cache_env_wins(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "by-hub-cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "by-home"))
    got = os.path.normcase(default_hub_cache_dir())
    assert got == os.path.normcase(str(tmp_path / "by-hub-cache"))


def test_hf_home_appends_hub(monkeypatch, tmp_path):
    """★ 桌面端就是这一档：只设 `HF_HOME` ⇒ 实际缓存根是 `$HF_HOME/hub`。"""
    _clear(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "pm"))
    assert default_hub_cache_dir().replace("\\", "/").endswith("/pm/hub")


def test_default_is_user_cache(monkeypatch):
    _clear(monkeypatch)
    got = default_hub_cache_dir().replace("\\", "/")
    assert got.endswith("/.cache/huggingface/hub"), got


def _make_snapshot(root: str) -> str:
    """在 root 下造一个「完整」的快照（config.json + 权重），返回快照目录。"""
    snap = os.path.join(root, "models--" + _SAFE, "snapshots", "deadbeef")
    os.makedirs(snap, exist_ok=True)
    with open(os.path.join(snap, "config.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    with open(os.path.join(snap, "model.safetensors"), "wb") as f:
        f.write(b"x")
    return snap


def test_snapshot_found_under_hf_home(monkeypatch, tmp_path):
    """HF_HOME 下有 ⇒ 探测得到（这条保证修复后桌面端能正常复用已下载权重）。"""
    _clear(monkeypatch)
    home = tmp_path / "pm"
    snap = _make_snapshot(str(home / "hub"))
    monkeypatch.setenv("HF_HOME", str(home))
    got = resolve_local_hf_snapshot(_REPO)
    assert got is not None and os.path.normcase(got) == os.path.normcase(snap)


def test_snapshot_under_user_cache_ignored_when_hf_home_set(monkeypatch, tmp_path):
    """★★ 核心回归：**主目录**里有完整快照，但 ``HF_HOME`` 指向别处 ⇒ **必须返回 None**。

    ★ 为什么桩掉 ``os.path.expanduser`` 而不是 ``monkeypatch.setenv("USERPROFILE", ...)``：
      CPython 的 ``nt._getuserprofile()`` **首次调用后在进程内缓存**，事后改环境变量不重算
      （实测：设了 USERPROFILE 仍解析到真实用户目录）。若依赖"开发机主目录恰好有缓存"
      才能判红，换台没缓存的机器就会**假通过** —— 门禁宁可无结论。
      桩掉 expanduser 后，判据与本机状态完全无关。
    """
    _clear(monkeypatch)
    fake_home = tmp_path / "userprofile"
    _make_snapshot(str(fake_home / ".cache" / "huggingface" / "hub"))

    real_expanduser = os.path.expanduser

    def _fake_expanduser(path: str) -> str:
        return str(fake_home) + path[1:] if path.startswith("~") else real_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", _fake_expanduser)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "fresh"))   # 全新用户：空的

    # 前提自检：主目录里确实"有"权重（否则下面这条断言就没有区分力）
    assert os.path.isfile(
        os.path.join(str(fake_home), ".cache", "huggingface", "hub",
                     "models--" + _SAFE, "snapshots", "deadbeef", "model.safetensors")
    )
    assert resolve_local_hf_snapshot(_REPO) is None


def test_incomplete_snapshot_rejected(monkeypatch, tmp_path):
    """只有 config.json、没有权重 ⇒ 仍算未安装（口径与推理侧一致）。"""
    _clear(monkeypatch)
    hub = tmp_path / "hub" / ("models--" + _SAFE) / "snapshots" / "abc"
    hub.mkdir(parents=True)
    (hub / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert resolve_local_hf_snapshot(_REPO) is None
