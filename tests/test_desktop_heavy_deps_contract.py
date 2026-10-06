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
