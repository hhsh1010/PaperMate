"""`tauri.conf.json` 离线契约测试（2026-06-06）

为什么要单独测一个 JSON 配置文件
--------------------------------
它是**唯一只在打包时才会被验证**的产物：解析者是 Rust 的 `tauri-build`（serde），
而在开发/CI 环境里跑不了 cargo ⇒ 配置错误会一路漏到出包才炸。

★ 实际踩到（2026-10-06）：把 `webviewInstallMode` 写成 `{"mode": "downloadBootstrapper"}`，
  而 Tauri 2 的该字段是 **tagged enum**，要求键名是 `type`。serde 报的是
  `missing field 'type'` —— 文案指向"缺字段"，真实原因却是"键名写错"，
  很有误导性。而 `cargo check` 需要 Rust + MSVC 工具链才能跑到这一步。
  ⇒ 本文件用仓库内的官方 schema（`desktop/src-tauri/tauri.schema.json`）做**离线**校验。

本文件**零网络、毫秒级**，因此进日常门禁。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TAURI_DIR = _ROOT / "desktop" / "src-tauri"
_CONFIG_PATH = _TAURI_DIR / "tauri.conf.json"
_SCHEMA_PATH = _TAURI_DIR / "tauri.schema.json"

sys.path.insert(0, str(_ROOT / "scripts"))
from validate_tauri_config import load_schema, validate  # noqa: E402


def _cfg() -> dict:
    return json.load(open(_CONFIG_PATH, encoding="utf-8"))


def _validate(cfg: dict) -> list[str]:
    root = load_schema(None)
    errs: list[str] = []
    validate(cfg, root, root, "$", errs)
    return errs


# ───────────────────────── 1. schema 校验（覆盖整类错误）─────────────────────────
def test_config_matches_official_schema():
    """★ 全量 schema 校验 —— 一次报出所有不符合项（cargo 只报第一个）。"""
    errs = _validate(_cfg())
    assert not errs, "tauri.conf.json 不符合 Tauri 2 schema：\n  " + "\n  ".join(errs)


def test_validator_catches_wrong_enum_tag():
    """★ 反向自检：把 tagged enum 的键名写错，校验器**必须**报红。

    没有这条，"校验通过"就只是"校验器坏了"的假象 ——
    （本文件作者的校验器第一版正因漏实现 `allOf` 而恒返回通过。）
    """
    bad = _cfg()
    bad["bundle"]["windows"]["webviewInstallMode"] = {"mode": "downloadBootstrapper"}
    errs = _validate(bad)
    assert errs, "校验器没能发现错误的 enum 键名（mode 应为 type）"
    joined = " ".join(errs)
    assert "type" in joined and "mode" in joined


def test_validator_catches_unknown_field():
    """未知字段也必须被抓到（schema 里 additionalProperties=false）。"""
    bad = _cfg()
    bad["bundle"]["windows"]["notARealField"] = 1
    errs = _validate(bad)
    assert any("notARealField" in e for e in errs), errs


# ───────────────────────── 2. 打包契约 ─────────────────────────
def test_bundle_resources_include_bundled_runtime_and_backend():
    """两处捆绑资源缺一不可：Portable Python + uv（python/）与后端副本（backend/）。

    缺 `python/` ⇒ 用户机器没 Python 时应用起不来；
    缺 `backend/` ⇒ 没有后端可跑。
    """
    res = _cfg()["bundle"]["resources"]
    keys = set(res) if isinstance(res, dict) else set(res)
    assert any("python" in k for k in keys), f"bundle.resources 未包含 resources/python：{keys}"
    assert any("backend" in k for k in keys), f"bundle.resources 未包含 resources/backend：{keys}"


def test_bundle_icon_lists_all_five_required_files():
    """`bundle.icon` 必须列全 5 个文件（缺一个构建失败），且与 `npx tauri icon` 的产出对应。"""
    icons = _cfg()["bundle"]["icon"]
    need = {"32x32.png", "128x128.png", "128x128@2x.png", "icon.icns", "icon.ico"}
    got = {Path(p).name for p in icons}
    assert need <= got, f"bundle.icon 缺少：{sorted(need - got)}"


def test_webview_install_mode_uses_type_tag():
    """★ 钉死本次的真实缺陷：必须是 `{"type": ...}`，不能是 `{"mode": ...}`。"""
    mode = _cfg()["bundle"]["windows"]["webviewInstallMode"]
    assert "type" in mode, f"webviewInstallMode 必须是 tagged enum，用 `type` 键：{mode}"
    assert "mode" not in mode, f"`mode` 不是合法键名（serde 会报 missing field 'type'）：{mode}"


def test_sign_command_launcher_has_no_extra_dependency():
    """★ 签名命令的启动器必须是**系统自带、零额外安装**的程序。

    实测教训：原配置用了 `pwsh`（PowerShell 7），而本机**根本没装 pwsh**
    ⇒ `tauri build` 会在签名环节直接失败（且报错不像"缺 PowerShell"那么直白）。
    `powershell`（5.1，Windows 自带）才是安全选择。
    """
    sc = _cfg()["bundle"]["windows"]["signCommand"]
    cmd = sc if isinstance(sc, str) else sc.get("cmd", "")
    exe = Path(cmd.split()[0]).name.lower()
    assert exe not in {"pwsh", "pwsh.exe"}, (
        "signCommand 不能用 pwsh —— PowerShell 7 不是 Windows 自带，多数机器没有"
    )
    assert exe in {"powershell", "powershell.exe", "cmd", "cmd.exe", "signtool", "signtool.exe"}, (
        f"signCommand 启动器 {exe!r} 不是系统自带程序，请改用 powershell"
    )


def test_sign_command_replaces_placeholder_correctly():
    """`%1` 必须在 args 里（对象形式）或整个字符串里（字符串形式），否则签不到真文件。"""
    sc = _cfg()["bundle"]["windows"]["signCommand"]
    if isinstance(sc, str):
        assert "%1" in sc, "字符串形式必须包含 %1 占位符"
    else:
        assert "%1" in sc.get("args", []), "对象形式必须在 args 里包含 %1 占位符"
        assert "cmd" in sc and "args" in sc, "对象形式必须同时有 cmd 与 args"


def _resolve_from_config(rel: str) -> Path:
    """按 **Tauri 的规则**解析配置里的相对路径：基准是 `tauri.conf.json` 所在目录。

    ★★ 这里踩过一个大坑（2026-10-06）：初版写的是
        `_ROOT / "desktop" / "src-tauri" / rel.replace("../", "")`
      —— 用 `.replace()` **把 `../` 删掉**，而不是解析它。于是
      `"../frontend-dist"` 被算成 `src-tauri/frontend-dist`（恰好存在）⇒ **测试假绿**，
      而 Tauri 实际解析成 `desktop/frontend-dist`（不存在）⇒ 真构建失败。
      **两把尺子：测试的路径规则 ≠ 被测系统的路径规则。**
      ⇒ 修法：忠实复刻 Tauri 的解析方式 —— `(config_dir / rel).resolve()`。
    """
    return (_CONFIG_PATH.parent / rel).resolve()


def test_frontend_dist_exists_like_tauri_resolves_it():
    """`frontendDist` 指向的目录必须真实存在（Tauri 构建时需要，哪怕只是占位页）。"""
    rel = _cfg()["build"]["frontendDist"]
    dist = _resolve_from_config(rel)
    assert dist.is_dir(), (
        f"frontendDist={rel!r} 解析为 {dist}，该目录不存在。"
        f"（注意：Tauri 以 tauri.conf.json 所在目录为基准解析相对路径）"
    )
    assert (dist / "index.html").is_file() or (dist / "loading.html").is_file(), (
        f"{dist} 下既无 index.html 也无 loading.html，窗口会白屏"
    )


def test_bundle_icon_files_exist_like_tauri_resolves_them():
    """`bundle.icon` 里列的每个文件都必须存在（用与 Tauri 相同的解析规则）。

    ★ 这条本可以提前抓到"`icon.ico` not found"—— 当时是等 cargo 报了才知道。
    """
    missing = [p for p in _cfg()["bundle"]["icon"] if not _resolve_from_config(p).is_file()]
    assert not missing, (
        f"bundle.icon 里这些文件不存在（Tauri 会以 tauri.conf.json 目录为基准解析）：{missing}"
    )


def test_bundle_resource_dirs_exist_like_tauri_resolves_them():
    """`bundle.resources` 的每个源路径都必须存在 —— 否则 Tauri 的 build script 直接失败：
    `resource path '...' doesn't exist`（实测在 `cargo check` 阶段就会报）。"""
    res = _cfg()["bundle"]["resources"]
    srcs = list(res.keys()) if isinstance(res, dict) else list(res)
    missing = [p for p in srcs if not _resolve_from_config(p).exists()]
    assert not missing, (
        f"bundle.resources 里这些路径不存在：{missing}。"
        f"先跑 desktop/scripts/build-resources.ps1 生成它们"
    )


@pytest.mark.parametrize("key", ["beforeDevCommand", "beforeBuildCommand"])
def test_before_commands_do_not_require_frontend_toolchain(key):
    """`before*Command` 不能依赖 npm/vite —— 否则打包会去跑一个本仓库不需要的前端构建。

    本项目的架构是：前端由**后端托管** `web/dist`，桌面壳只做窗口 + 拉进程。
    """
    cmd = _cfg()["build"][key]
    low = cmd.lower()
    for bad in ("npm", "pnpm", "yarn", "vite", "tsc"):
        assert bad not in low, f"{key} 引用了 {bad}，但本项目前端不由 Tauri 构建：{cmd}"
