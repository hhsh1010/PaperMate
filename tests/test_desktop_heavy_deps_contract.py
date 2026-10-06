"""Rust ↔ Python 跨文件契约校验（2026-06-06）

问题
----
"首启不装重依赖"这个优化跨了两种语言：

- ``desktop/src-tauri/src/lib.rs`` 的 ``HEAVY_DEPS``（首启传给 ``uv sync --no-install-package``）
- ``src/papermate/services/rag_deps.py`` 的 ``HEAVY_DISTRIBUTIONS``（启用真实语义时补装）

**两处必须逐字一致**，否则会出现两种都很难察觉的故障：

1. Rust 多写了某个包而 Python 没补装 ⇒ 用户点「下载」跑完，**仍然 import 不到**；
2. Rust 少写了某个包 ⇒ 首启又变回 4.8GB，优化**静默失效**（功能正常，只是白下 4GB）。

★ 沙箱里**没有 cargo，无法编译验证 Rust**，所以这类"改一边忘了另一边"的风险
  只能靠**静态一致性测试**兜住 —— 它不需要 Rust 工具链，纯文本解析，毫秒级。

本文件因此**不进 slow 集合**，是日常门禁的一部分。
"""

from __future__ import annotations

import re
from pathlib import Path

from papermate.services.rag_deps import HEAVY_DISTRIBUTIONS

_LIB_RS = Path(__file__).resolve().parents[1] / "desktop" / "src-tauri" / "src" / "lib.rs"

#: 首启参数里**不允许**出现的裸字面量（必须统一由 HEAVY_DEPS 常量提供）
HEAVY_DEPS_LITERAL_NAMES = list(HEAVY_DISTRIBUTIONS)

_CONST_RE = re.compile(r"const\s+HEAVY_DEPS\s*:\s*\[&str;\s*\d+\]\s*=\s*\[(?P<body>[^\]]*)\]")


def _rust_heavy_deps() -> list[str]:
    """从 lib.rs 解析 HEAVY_DEPS 常量（纯文本，不依赖 cargo）。"""
    src = open(_LIB_RS, encoding="utf-8").read()
    m = _CONST_RE.search(src)
    assert m, f"未在 {_LIB_RS} 找到 `const HEAVY_DEPS: [&str; N] = [...]`（改了 Rust 侧写法请同步本正则）"
    return re.findall(r'"([^"]+)"', m.group("body"))


def test_rust_and_python_heavy_dep_lists_match():
    """★ 核心契约：两侧列表必须完全相同（顺序也一致，便于人工比对）。"""
    rust = _rust_heavy_deps()
    assert rust == list(HEAVY_DISTRIBUTIONS), (
        f"不一致！\n  Rust(lib.rs HEAVY_DEPS) = {rust}\n  Python(rag_deps.HEAVY_DISTRIBUTIONS)"
        f" = {list(HEAVY_DISTRIBUTIONS)}"
    )


def test_rust_list_uses_distribution_names_not_import_names():
    """★ 防"连字符/下划线"回归：uv 的 --no-install-package 只认**发行名**。

    写错成 ``sentence_transformers`` 不会报错，只会**静默不生效** ⇒ 首启仍装 4GB。
    """
    rust = _rust_heavy_deps()
    assert "sentence-transformers" in rust, f"lib.rs 必须用发行名（连字符）：{rust}"
    assert "sentence_transformers" not in rust, f"lib.rs 混入了导入名（下划线）：{rust}"


def _strip_rust_comments(src: str) -> str:
    """去掉 Rust 行注释与块注释，只留**真实代码**。

    ★ 为什么要剥：首启那段的中文注释里写了示例命令
      ``uv sync --extra rag --frozen --no-install-package torch ...``，
      而 ``--no-install-package`` 字符串在代码里是**另一个位置**（push 时的字面量）。
      不剥注释就会拿"注释里的位置"去比"代码里的位置" ⇒ **测试自己被骗**（2026-06-06 实测）。
    """
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def test_first_launch_actually_passes_no_install_package():
    """确认首启那段代码真的把排除项传给了 uv（而不是定义了常量却没用）。"""
    code = _strip_rust_comments(open(_LIB_RS, encoding="utf-8").read())
    assert '"--no-install-package".to_string()' in code, "首启未传 --no-install-package"
    assert "HEAVY_DEPS.iter()" in code, "首启未遍历 HEAVY_DEPS"
    # 排除项在代码里必须排在 --frozen 之后（uv 参数顺序不敏感，但保证可读性）
    assert code.index("--frozen") < code.index("--no-install-package"), \
        "代码中排除项应在 --frozen 之后追加"
    # ★ 每个包名字面量在**代码**里应当只出现 1 次（即常量定义处）。
    #   若在 uv 参数数组里又写了一遍 "torch"，就会与常量脱钩 —— 改常量时漏改这里，
    #   于是首启排除了、启用时没补装（或反之），且**编译与运行都不报错**。
    for pkg in HEAVY_DEPS_LITERAL_NAMES:
        n = code.count(f'"{pkg}"')
        assert n == 1, f'包名 "{pkg}" 在 lib.rs 代码里出现 {n} 次（应为 1 次 = 仅常量定义处）'


def test_backend_stage2_does_not_exclude_heavy_deps():
    """★ 反向契约：后端补装阶段**绝不能**带排除项，否则永远补不齐。"""
    src = open(
        Path(__file__).resolve().parents[1] / "src" / "papermate" / "services" / "rag_deps.py",
        encoding="utf-8",
    ).read()
    # 只看 `sync_heavy_deps` 的函数体 —— 模块文档里**故意**提到过 --no-install-package
    #（说明首启怎么用），拿全文断言会被文档骗（与上面同一个坑）。
    start = src.index("def sync_heavy_deps(")
    body = src[start: src.index("\ndef ", start + 1)] if "\ndef " in src[start + 1:] else src[start:]
    assert "--no-install-package" not in body, \
        "sync_heavy_deps 是「补装」阶段，不能带 --no-install-package（会自我抵消）"
    assert '"sync"' in body and '"--extra"' in body, "补装阶段应照常 sync --extra rag（不加排除项）"


# --------------------------------------------------------------------------- #
# 2026-10-06 真机首装暴露的第二个 P0：「环境是否需要安装」的判据与安全阀
# --------------------------------------------------------------------------- #
def test_first_launch_uses_inexact_so_it_never_prunes_heavy_deps():
    """★★ 首启 uv sync 必须带 `--inexact`，否则会把用户下好的 4GB 重依赖**剪掉**。

    实测证据（**同一环境、同一命令，仅差这一个 flag**，用 `uv sync --dry-run` 观察；
    实验 venv 里预置了一个假的 `torch-2.13.0.dist-info`）::

        默认（精确模式）：      ` - torch==2.13.0`   ← 会**删除** torch
        加 --inexact：          （无删除行）        ← 保留
        两者"将安装"的集合完全相同（88 vs 88），差异**只有那一行删除**

    ⇒ 首启那段只要**在重依赖装好之后再跑到一次**（指纹变化：App 升级换了 uv.lock、
      或安装目录迁移），精确模式就会连带剪掉 torch/CUDA。用户表现是
      "真实语义检索莫名其妙又变回假检索"，而且**没有任何报错**。
    """
    code = _strip_rust_comments(open(_LIB_RS, encoding="utf-8").read())
    assert '"--inexact".to_string()' in code, (
        "首启 uv sync 缺少 --inexact：它会把 `--no-install-package` 排除、"
        "但**环境里已经装好**的包一起剪掉（实测 exact 模式会输出 ` - torch==2.13.0`）"
    )


def test_first_launch_readiness_is_capability_not_file_existence():
    """★★ 「是否需要安装」的判据必须是**能力探测**，不能是"python.exe 是否存在"。

    真实事故：`uv` 在**创建环境**阶段就把 `Scripts/python.exe` 写好了，安装发生在**其后**。
    首启中途失败（缺 README.md ⇒ hatchling 构建本地项目失败）后，
    venv 里只剩一个 `site-packages/_virtualenv.pth` 的**空壳**，而 python.exe 是在的。
    旧判据 ``if !venv_python.exists()`` 被满足 ⇒ **此后每次启动都跳过安装** ⇒
    拿空环境跑 ``-m papermate.api`` ⇒ ``No module named papermate`` ⇒
    后端永不启动、界面永远停在"正在启动"，且**一句错误提示都没有**。
    """
    code = _strip_rust_comments(open(_LIB_RS, encoding="utf-8").read())

    # 1) 已被证伪的"裸文件存在"判据必须消失
    assert not re.search(r"if\s*!\s*venv_python\.exists\(\)\s*\{", code), (
        "首启仍在用 `if !venv_python.exists()`：uv 建环境时就会写 python.exe，"
        "安装失败后它依然存在 ⇒ 会永远跳过安装"
    )
    # 2) 必须有真探测函数，且真的被调用
    assert "fn probe_backend_imports(" in code, "缺少 probe_backend_imports（能力探测）"
    assert "probe_backend_imports(&venv_python" in code, "定义了探针却没用它"
    # 3) 探针必须是"真 import"，不能退化成查文件
    assert "import fastapi" in code, "探针必须是真 import（验能力），不能退化成查文件存在"
    # 4) 标记机制的读取判据在
    assert "DEPS_SENTINEL" in code and "fs::read_to_string(&sentinel_path)" in code, (
        "缺少 deps-ready 标记的读取判据（0ms 快速通道）"
    )


def test_deps_marker_is_written_only_after_capability_probe():
    """★★ 标记只能在「uv 退出码 0 **且**能力探测通过」之后落盘 —— 否则标记会说谎。

    标记为真是后续启动**跳过安装**的唯一依据，所以它必须**蕴含**"环境真的可用"。
    顺序反了 ⇒ 探测失败也落标记 ⇒ 下次启动跳过安装 ⇒ 永久白屏。
    """
    code = _strip_rust_comments(open(_LIB_RS, encoding="utf-8").read())
    i_probe = code.index("if !probe_backend_imports(&venv_python")
    i_write = code.index("fs::write(&sentinel_path")
    assert i_probe < i_write, (
        "`fs::write(&sentinel_path ...)` 出现在能力探测**之前** ⇒ "
        "探测失败时也会落标记 ⇒ 下次启动跳过安装"
    )


def test_probe_does_not_import_the_api_module():
    """探针不得 import ``papermate.api`` —— 它在导入期就 ``create_app()``。

    ★ 副作用（建库/建目录）会让"探测"本身改变被测对象的状态；
      所以探针只 import 两个**空 `__init__.py`** 的包 + fastapi/sqlalchemy。
    """
    code = _strip_rust_comments(open(_LIB_RS, encoding="utf-8").read())
    m = re.search(r'args\(\["-c",\s*"([^"]+)"\]\)', code)
    assert m, "未找到探针的 -c 参数（写法变了？请同步本测试的正则）"
    probe = m.group(1)
    assert "papermate.api" not in probe, f"探针不得 import papermate.api（有副作用）：{probe}"
    assert "fastapi" in probe, f"探针应校验第三方核心依赖：{probe}"
    assert "papermate" in probe and "lightcore" in probe, f"探针应校验两个本地包：{probe}"
