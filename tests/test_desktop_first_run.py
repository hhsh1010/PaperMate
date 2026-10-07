"""桌面端「首启选择数据位置」的契约测试（2026-10-07）

## 为什么需要这一组

这个功能的几个关键点**编译期完全看不出来**，只在真机上炸：

1. `tauri-plugin-dialog` **只加 Cargo 依赖是不够的** —— 必须 `.plugin(...init())` 注册，
   否则 `app.dialog()` 运行时报错（`cargo check` 全绿）。
2. 后端 `rag_deps` 需要 `PAPERMATE_VENV_DIR` / `PAPERMATE_UV_CACHE_DIR` 才能与 launcher **同源**；
   漏注入 ⇒ 用户改了数据位置后「准备运行环境/下载模型」**静默失效**。
3. **默认行为零变化**：没有 `app-settings.json` 时四条路径必须与本改造前**逐字节相同**。
4. `ask_data_root()` 的**每一条返回路径**都必须写配置文件，否则下次启动还会再问一遍。
5. 选择器是**阻塞式**的，插件文档明确要求不得在主线程调用 ⇒ `setup` 必须把它挪到后台线程。

★ 本文件零网络、毫秒级，进日常门禁。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TAURI_DIR = _ROOT / "desktop" / "src-tauri"
_LIB = (_TAURI_DIR / "src" / "lib.rs").read_text(encoding="utf-8")
_HAS_LIB = bool(_LIB.strip())


def _fn_body(name: str, next_marker: str | None) -> str:
    """抽出 `fn <name>(` 到 `next_marker` 之间的源码；`next_marker=None` 表示"到文件末尾"。

    ★ 边界必须命中，否则**报错** —— 免得"抽到空串"让后面的断言**假通过**。
      （这个守卫真的生效过：`bootstrap_and_spawn` 是文件最后一个函数，
        早先版本硬编码找下一个 `\nfn ` → 直接 ValueError，而不是悄悄放过。）
    """
    start = _LIB.index(f"fn {name}(")
    end = len(_LIB) if next_marker is None else _LIB.index(next_marker, start)
    assert end > start, f"未能在 {name} 之后找到边界 {next_marker!r}"
    body = _LIB[start:end]
    assert len(body) > 80, f"{name} 的抽取结果异常短（{len(body)} 字符），边界写错了？"
    return body


def _startup_page() -> Path:
    cfg = json.loads((_TAURI_DIR / "tauri.conf.json").read_text(encoding="utf-8"))
    dist = (_TAURI_DIR / cfg["build"]["frontendDist"]).resolve()
    return dist / cfg["app"]["windows"][0]["url"]


def test_lib_rs_is_readable():
    assert _HAS_LIB, "lib.rs 读不到 —— 后续所有静态断言都会变成假通过"


# ───────────────────── 1. 插件必须注册（编译期看不出来）─────────────────────
def test_dialog_plugin_is_registered_not_just_a_dependency():
    """★★ 只加 Cargo 依赖不够：必须 `.plugin(tauri_plugin_dialog::init())`。

    漏了这一步 `cargo check` **完全绿**，但 `app.dialog()` 在运行时会失败
    —— 而它恰好发生在**首次启动**，用户看到的就是"点了没反应"。
    """
    assert "app.dialog()" in _LIB, "lib.rs 没用到 app.dialog()？那本测试的意义变了，请同步"
    assert ".plugin(tauri_plugin_dialog::init())" in _LIB, (
        "lib.rs 使用了 app.dialog() 却没有注册 tauri_plugin_dialog —— "
        "编译能过，但首启弹不出选择器"
    )


def test_dialog_dependency_is_declared():
    cargo = (_TAURI_DIR / "Cargo.toml").read_text(encoding="utf-8")
    assert "tauri-plugin-dialog" in cargo, "Cargo.toml 里没有 tauri-plugin-dialog"


# ───────────────── 2. venv / uv-cache 必须显式注入（防静默失效）─────────────────
def test_backend_env_injects_explicit_venv_and_cache_paths():
    """★★ 这两个 env 是"存储位置可配置"的地基。

    后端 `rag_deps.venv_dir()` 会优先读 `PAPERMATE_VENV_DIR`；若不注入，它会退回
    用 `data_dir.parent` 推导 —— 用户换了数据位置后就会指向错误目录，
    「准备运行环境 / 下载模型」**静默失效**（`tests/test_rag_deps_paths.py` 另一端钉死）。
    """
    for key in ("PAPERMATE_VENV_DIR", "PAPERMATE_UV_CACHE_DIR"):
        assert f'.env("{key}"' in _LIB, f"lib.rs 未向后端注入 {key}"


def test_all_four_dirs_derive_from_data_root():
    """四个派生目录必须都从 `data_root` 来（**一个入口**决定全部）。"""
    body = _fn_body("bootstrap_and_spawn", None)  # 它是文件最后一个函数
    for sub in ("data", "huggingface", "venv", "uv-cache"):
        assert re.search(rf'data_root\.join\("{re.escape(sub)}"\)', body), (
            f"未从 data_root 派生 {sub}/ —— 用户换了位置它不会跟着走"
        )


def test_default_root_is_local_data_so_behavior_is_unchanged():
    """★ 默认行为零变化：没有配置时 root == `<local_data>` ⇒ 四条路径与本改造前一致。

    ★ 2026-10-07 结构变了：判定逻辑从 `resolve_data_root` 移进了 `AppSettings::read`
      （"读配置"现在被首启与 pending 迁移两处共用）⇒ 断言跟着结构走，**意图不变**。
    """
    read_body = _fn_body("read", "\nfn ")
    assert "is_file()" in read_body, "AppSettings::read 必须先判断配置文件是否存在"
    assert "None" in read_body, "文件不存在（或损坏）时必须返回 None"

    body = _fn_body("resolve_data_root", "\nfn ")
    assert "None => (local_data.to_path_buf(), true)" in body, (
        "没有配置 ⇒ 用默认位置，并标记为「首次」"
    )
    assert "unwrap_or_else(|| local_data.to_path_buf())" in body, (
        "配置里没有 data_root 时必须回退默认位置"
    )


# ───────────────── 3. 首启：判定 / 写配置 / 线程 ─────────────────
def test_first_run_is_decided_by_app_settings_existence():
    """首次判据 = `app-settings.json` 不存在 ⇒ `(默认位置, first_run=true)`。"""
    read_body = _fn_body("read", "\nfn ")
    assert "APP_SETTINGS" in read_body, "AppSettings::read 必须读 APP_SETTINGS 那个文件"
    body = _fn_body("resolve_data_root", "\nfn ")
    assert "(root, false)" in body and "(local_data.to_path_buf(), true)" in body, (
        "有配置 ⇒ 非首次；无配置 ⇒ 首次"
    )


def test_ask_data_root_writes_settings_on_every_exit_path():
    """★★ 三条返回路径（取消 / 成功 / 重试 3 次后放弃）**都必须**写配置。

    漏掉任何一条 ⇒ 该路径下下次启动**又弹一次**选择器（用户会觉得"设了没用"）。
    """
    body = _fn_body("ask_data_root", "\nfn ")
    n = body.count("write_app_settings(")
    assert n >= 3, f"ask_data_root 只有 {n} 处 write_app_settings，三条返回路径应各有一处"


def test_bootstrap_runs_off_the_main_thread():
    """★★ 阻塞式文件夹选择器**不得在主线程调用**（插件源码明确警告会死锁）。

    顺带解决另一个问题：`uv sync` 首次要跑几分钟，阻塞 `setup` 会让窗口"无响应"。
    """
    assert "thread::spawn(move || bootstrap_and_spawn(&handle))" in _LIB, (
        "bootstrap_and_spawn 必须跑在后台线程（阻塞式对话框 + 长时间 uv sync）"
    )
    assert "tauri-plugin-dialog/src/lib.rs:744" in _LIB or "NOT* be used when running" in _LIB, (
        "请在注释里保留「为什么必须后台线程」的依据（免得后人又挪回主线程）"
    )


# ───────────────── 4. 校验：必须真写真删 ─────────────────
def test_validate_data_root_probes_with_a_real_write():
    """★ Windows 上 `os.access(W_OK)` 是**假信号** ⇒ 必须真写真删（本项目 10-06 血泪）。"""
    body = _fn_body("validate_data_root", "\nfn ")
    assert "fs::write(&probe" in body and "remove_file(&probe)" in body, (
        "校验必须真的写一个探针文件再删掉，不能只做权限位判断"
    )
    for needle, why in (
        ("is_absolute", "必须拒绝相对路径（否则 CWD 不同就分叉）"),
        ("starts_with(r\"\\\\\")", "必须拒绝 UNC"),
        ("drive_type", "必须用盘类型拒绝网络映射盘"),
        ("norm_for_cmp", "安装目录比较必须先归一化（\\?\\ 前缀 / 大小写）"),
        ("MIN_FREE_GB", "必须有磁盘余量闸门"),
    ):
        assert needle in body, f"validate_data_root 缺少校验：{why}"


def test_path_comparison_strips_verbatim_prefix():
    """★ Tauri 的 `resource_dir()` 返回带 `\\\\?\\` 的长路径（本项目 10-07 踩过）

    ⇒ 任何"路径相等/包含"的判断都必须先归一化，否则永远不相等。
    """
    body = _fn_body("norm_for_cmp", "\nfn ")
    assert r'r"\\?\"' in body, "norm_for_cmp 必须剥离 \\\\?\\ 前缀"
    assert "to_lowercase()" in body, "Windows 路径比较必须忽略大小写"


# ───────────────── 5. 等待页：首启面板 + 不依赖 IPC ─────────────────
def test_startup_page_exposes_first_run_hooks():
    html = _startup_page().read_text(encoding="utf-8")
    assert "window.pmFirstRun" in html, "等待页缺少 pmFirstRun（Rust 用它切到首启形态）"
    assert "window.pmRootChosen" in html, "等待页缺少 pmRootChosen（回显最终位置，不静默）"
    # 首启期间不能被"已等待 N 秒"覆盖标题
    assert "if (!firstRun)" in html, "首启面板显示期间不应被 tick() 覆盖标题"


def test_startup_page_needs_no_ipc_and_no_capability():
    """★ 设计要点：用户操作发生在**原生对话框**里 ⇒ 页面**不需要** JS→Rust IPC。

    这条如果被破坏（例如有人加了 `window.__TAURI__.invoke`），就必须同时补
    `app.withGlobalTauri=true` **和** capability —— 否则调用静默失败。
    """
    html = _startup_page().read_text(encoding="utf-8")
    assert "__TAURI__" not in html, (
        "等待页不应依赖 Tauri IPC —— 用户操作在原生对话框里，页面只负责展示"
    )
    cfg = json.loads((_TAURI_DIR / "tauri.conf.json").read_text(encoding="utf-8"))
    assert not cfg.get("app", {}).get("withGlobalTauri"), (
        "本项目没有 capability 文件；若开启 withGlobalTauri 而不配 capability，"
        "任何 JS→Rust 调用都会静默失败"
    )


# ───────────────── 6. 设置页改位置 + 重启迁移（2026-10-07 追加）─────────────────
def test_pending_migration_runs_before_the_backend_is_spawned():
    """★★ 迁移必须在**后端启动之前**完成。

    那时 SQLite 未打开、venv 未加载 ⇒ 不存在"拷到一半"导致数据不一致的可能。
    若顺序颠倒（先 spawn 再迁移），搬迁的就是一个**正在被写**的数据库目录。
    """
    body = _fn_body("bootstrap_and_spawn", None)
    i_migrate = body.index("apply_pending_migration(")
    i_spawn = body.index("papermate.api")
    assert i_migrate < i_spawn, "apply_pending_migration 必须在 spawn 后端之前调用"


def test_migration_moves_data_weights_cache_but_never_venv():
    """★ 搬 `data/` `huggingface/` `uv-cache/`，**不搬** `venv/`。

    venv 内部全是绝对路径（`.pth` 指向源码、`pyvenv.cfg` 指向便携 Python、脚本 shebang）
    ⇒ 复制过去会坏；正确做法是在新位置**重建**（`uv-cache` 跟着搬了，重建会命中硬链接）。
    """
    body = _fn_body("migrate_data_dirs", "\nfn ")
    listed = re.search(r'for name in \[([^\]]+)\]', body)
    assert listed, "未找到要搬迁的目录清单"
    names = set(re.findall(r'"([^"]+)"', listed.group(1)))
    assert names == {"data", "huggingface", "uv-cache"}, f"搬迁清单不符预期：{names}"
    assert "venv" not in names, "venv 含绝对路径，**只能重建不能搬**"


def test_migration_never_deletes_the_source():
    """★★ 迁移**永不删除源目录** —— 「删用户数据」不该由程序悄悄做。"""
    body = _fn_body("migrate_data_dirs", "\nfn ")
    assert "remove_dir_all" not in body, "迁移里出现了删除目录的调用"
    body2 = _fn_body("apply_pending_migration", "\nfn ")
    assert "remove_dir_all" not in body2, "pending 处理里出现了删除目录的调用"


def test_migration_verifies_file_count_after_copy():
    """★ 搬完必须**逐项核对文件数**，不一致即报错（跨盘复制中途失败会静默产生半份数据）。"""
    body = _fn_body("migrate_data_dirs", "\nfn ")
    assert "count_files(" in body
    assert re.search(r"count_files\(&from\)\s*,\s*count_files\(&to\)|\(a, b\) = \(count_files", body), (
        "搬迁后必须比较源与目标的文件数"
    )
    assert "不一致" in body, "校验失败应有明确文案"


def test_pending_request_is_cleared_on_both_success_and_failure():
    """★ 请求处理完（含失败）都要把 `pending_root` 清掉，否则每次启动都会重试。"""
    body = _fn_body("apply_pending_migration", "\nfn ")
    assert body.count("write_root_only(") >= 3, (
        f"成功/校验失败/迁移失败/写配置失败 四条路径都应写回（当前 {body.count('write_root_only(')} 处）"
    )


def test_backend_receives_the_settings_path():
    """★ 后端必须知道配置文件在哪 —— 数据目录可能已搬到别处，它无法从 `data_dir` 反推。"""
    assert '.env("PAPERMATE_APP_SETTINGS"' in _LIB, "未向后端注入 PAPERMATE_APP_SETTINGS"
