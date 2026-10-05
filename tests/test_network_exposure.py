"""监听地址暴露状态测试（2026-09-15，后端工程化 P1-4）。

背景：本项目**无鉴权**（定位本地单用户）。默认监听 `127.0.0.1` 是安全的，但此前
**只在 `config.Settings.host` 的注释里写了风险**，运行时没有任何提示 ——
若用户显式改成 `PAPERMATE_HOST=0.0.0.0`（例如想让手机访问），
同网段任何人都能读写全部会话/文档/记忆，甚至 `PUT /settings` 覆盖模型密钥。

本文件守护三件事：
1. `Settings.network_exposed` 的判定口径（回环的各种写法都要判对）；
2. 启动告警 `warn_if_exposed` 的行为（回环不打扰、非回环必须提示、已确认则降级）；
3. `GET /healthz` 暴露该状态，便于前端/运维察觉。
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from papermate.api.__main__ import warn_if_exposed
from papermate.config import Settings

# ---------- 1. 判定口径 ----------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "LOCALHOST", " ::1 ", "0:0:0:0:0:0:0:1"],
)
def test_loopback_hosts_are_not_exposed(host: str) -> None:
    assert Settings(host=host).network_exposed is False


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.7", "10.0.0.5", "::"])
def test_non_loopback_hosts_are_exposed(host: str) -> None:
    assert Settings(host=host).network_exposed is True


def test_default_host_is_loopback() -> None:
    """默认值必须安全 —— 这是"本地优先"定位的底线。"""
    assert Settings().host == "127.0.0.1"
    assert Settings().network_exposed is False


# ---------- 2. 启动告警行为 ----------


def test_warn_silent_on_loopback() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exposed = warn_if_exposed(False, "127.0.0.1")
    assert exposed is False
    assert buf.getvalue() == "", "回环监听不应有任何输出"


def test_warn_banner_on_non_loopback() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exposed = warn_if_exposed(True, "0.0.0.0")
    out = buf.getvalue()
    assert exposed is True
    assert "安全告警" in out
    assert "没有鉴权" in out
    # 回归：banner 的分隔线必须是**整行**（曾因 Python 相邻字面量 + `*` 优先级
    # 写成 `"\n" "=" * 78`，导致输出 78 行单独的 `=`）
    assert "=" * 78 in out
    assert "\n=\n" not in out


def test_warn_ack_env_downgrades_to_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """设置确认变量后只打一行提示（用户已知情，不再每次刷屏）。"""
    monkeypatch.setenv("PAPERMATE_ACCEPT_PUBLIC_EXPOSURE", "1")
    buf = io.StringIO()
    with redirect_stdout(buf):
        exposed = warn_if_exposed(True, "0.0.0.0")
    out = buf.getvalue()
    assert exposed is True
    assert "已确认对外暴露" in out
    assert "安全告警" not in out


# ---------- 3. healthz 暴露该状态 ----------


def test_healthz_reports_not_exposed_by_default() -> None:
    import os
    import tempfile

    os.environ["PAPERMATE_DATA_DIR"] = tempfile.mkdtemp(prefix="papermate_expose_test_")
    os.environ.pop("PAPERMATE_HOST", None)
    cfg_mod._settings = None
    from papermate.api.app import create_app

    with TestClient(create_app()) as c:
        body = c.get("/api/v1/healthz").json()
    assert body["network_exposed"] is False


# ---------------------------------------------- E3 强化拦截（2026-09-21）

from papermate.api.__main__ import should_block_exposure  # noqa: E402


def test_exposure_is_blocked_without_explicit_ack() -> None:
    """★ E3：非回环 + 未确认 → 必须阻断。

    原实现"告警但不阻断"：手滑改绑 0.0.0.0 时横幅一闪而过（或根本没看），服务照常对外裸奔。
    """
    assert should_block_exposure(True, None) is True
    assert should_block_exposure(True, "0") is True
    assert should_block_exposure(True, "") is True


def test_exposure_allowed_after_ack() -> None:
    """逃生门：显式确认后放行（用户可能确实需要局域网访问）。"""
    assert should_block_exposure(True, "1") is False


def test_loopback_never_blocks() -> None:
    """默认回环不受影响 —— 强化拦截不能干扰日常使用。"""
    assert should_block_exposure(False, None) is False
    assert should_block_exposure(False, "0") is False
