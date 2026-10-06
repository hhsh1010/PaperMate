"""打包资源脚本 ↔ ``pyproject.toml`` 元数据契约校验（2026-10-06）

问题
----
``desktop/scripts/build-resources.ps1`` 把仓库源码拷成一份**自包含的后端副本**
（``desktop/src-tauri/resources/backend``），首启时 ``uv sync`` 会在这份副本里
**安装项目自身**（``hatchling`` editable 构建）。

而 ``hatchling`` 会读取 ``pyproject.toml`` 里声明的元数据文件 ——
``readme`` / ``license.file`` / ``license-files`` 全都是**相对 pyproject 的路径**。
脚本只要漏拷其中一个，构建就会在**用户机器上**炸。

★★ 真实事故（本次真机首装）
---------------------------
脚本此前只拷 ``pyproject.toml`` / ``uv.lock`` / ``.env.example``，**漏了 ``README.md``**，
于是首装走到最后一步失败::

    Building papermate @ file:///.../resources/backend
    error: Failed to build `papermate @ file:///.../resources/backend`
    cause: Call to `hatchling.build.build_editable` failed (exit code: 1)
    OSError: Readme file does not exist: README.md
    [desktop] uv sync failed: exit code: 1

★ 为什么开发机上永远测不出来：项目根目录**一直**有 ``README.md``，
  只有"拷出来的那份副本"才缺 —— 属于**导出副本专属缺陷**。

★ 为什么必须"剥注释后再断言"
----------------------------
本仓库的注释里会写示例命令与文件名（``README.md`` 恰好就在注释里出现过）。
若直接对全文做子串搜索，**测试会被自己的注释骗过**（本仓库已有前科：
``test_desktop_heavy_deps_contract.py`` 的同一坑）。
所以这里只解析**真实的 ``Copy-Item`` 语句**。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PS1 = _ROOT / "desktop" / "scripts" / "build-resources.ps1"
_PYPROJECT = _ROOT / "pyproject.toml"


def _strip_ps_comments(src: str) -> str:
    """丢掉 PowerShell 行注释。

    ⚠️ 只够用于本脚本 —— 它没有任何字符串里含 ``#``。
    若将来脚本里出现带 ``#`` 的字符串（如 URL 片段），本函数要同步升级成状态机。
    """
    out = []
    for line in src.splitlines():
        idx = line.find("#")
        out.append(line if idx < 0 else line[:idx])
    return "\n".join(out)


def _declared_copies() -> set[str]:
    """解析出脚本**真实会拷贝**的仓库根相对路径集合。

    形如：
      - ``Copy-Item "$root/pyproject.toml" "$backend/"``           → ``pyproject.toml``
      - ``foreach ($meta in @("README.md", "LICENSE")) { ... Copy-Item "$root/$meta" ... }``
        → ``README.md`` / ``LICENSE``（**仅当循环体里确实有 ``$root/$meta`` 的拷贝**）
    """
    code = _strip_ps_comments(_PS1.read_text(encoding="utf-8-sig"))

    names: set[str] = set()
    for m in re.finditer(r'Copy-Item\s+"\$root/([^"$]+)"', code):
        names.add(m.group(1).replace("\\", "/"))

    # 循环形式：只有当"循环变量确实被用于拷贝"时才把列表算作已声明，
    # 否则 @("A","B") 可能只是遍历做别的事 ⇒ 会造成**假通过**。
    for m in re.finditer(r"foreach\s*\(\s*\$meta\s+in\s+@\(([^)]*)\)\s*\)", code):
        items = [s for s in re.findall(r'"([^"]+)"', m.group(1))]
        loop_body_start = m.end()
        loop_body = code[loop_body_start: loop_body_start + 800]
        if 'Copy-Item "$root/$meta"' in loop_body:
            names.update(i.replace("\\", "/") for i in items)
    return names


def _referenced_metadata_files() -> set[str]:
    """``pyproject.toml`` 里**会被构建后端读取**的元数据文件（仓库根相对路径）。

    只收集"文件不存在就会让构建失败"的那几项：
      - ``project.readme``（``hatchling`` 会直接读，缺了必失败 ← 本次事故）
      - ``project.license.file``（PEP 621 表形式）
      - ``project.license-files``（PEP 639 显式 glob）
    ★ ``project.license`` 若为字符串则是 SPDX 表达式，**不指向文件**，忽略。
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})

    refs: set[str] = set()

    readme = project.get("readme")
    if isinstance(readme, str):
        refs.add(readme)
    elif isinstance(readme, dict) and "file" in readme:  # {file=..., content-type=...}
        refs.add(readme["file"])

    lic = project.get("license")
    if isinstance(lic, dict) and "file" in lic:
        refs.add(lic["file"])

    for glob in project.get("license-files", []) or []:
        for hit in _ROOT.glob(glob):
            if hit.is_file():
                refs.add(str(hit.relative_to(_ROOT)).replace("\\", "/"))

    return {r.replace("\\", "/") for r in refs}


def test_every_pyproject_metadata_file_is_copied_into_backend_copy():
    """★ 核心契约：pyproject 引用的元数据文件，打包脚本必须全部拷进后端副本。"""
    referenced = _referenced_metadata_files()
    declared = _declared_copies()
    missing = sorted(referenced - declared)
    assert not missing, (
        f"build-resources.ps1 未拷贝 pyproject.toml 引用的元数据文件：{missing}\n"
        f"  已声明拷贝：{sorted(declared)}\n"
        f"  ⇒ 首启 uv sync 构建项目自身时会失败（hatchling: 文件不存在）\n"
        f"  ⇒ 修复：在 build-resources.ps1 的元数据拷贝处补上这些文件。"
    )


def test_declared_copies_all_exist_on_disk():
    """反向：脚本声明的每个文件都必须真实存在，否则是笔误（如 README.mkd）。"""
    declared = _declared_copies()
    assert declared, "未从 build-resources.ps1 解析出任何拷贝项（脚本写法变了？请同步本测试的正则）"
    absent = sorted(n for n in declared if not (_ROOT / n).exists())
    assert not absent, f"build-resources.ps1 声明的文件在仓库里不存在（疑似改名/笔误）：{absent}"


def test_readme_is_declared_because_pyproject_requires_it():
    """回归护栏：``readme = "README.md"`` 必须在拷贝清单里（本次真实事故的锚点）。"""
    assert "README.md" in _referenced_metadata_files(), (
        "pyproject.toml 不再用 README.md 作 readme？那请同步更新本测试与打包脚本。"
    )
    assert "README.md" in _declared_copies(), (
        "README.md 必须出现在 build-resources.ps1 的拷贝清单里 —— "
        "否则首启构建项目自身时 hatchling 会报 `Readme file does not exist: README.md`。"
    )
