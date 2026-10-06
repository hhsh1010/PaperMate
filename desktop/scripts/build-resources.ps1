# build-resources.ps1 — 准备 Tauri 打包所需的运行时资源
# 作用：构建前端 → 拷贝后端源码(含 web/dist) → 放入 Portable Python + uv → 注入 loading.html
# 运行：在 desktop/ 目录下执行  powershell -ExecutionPolicy Bypass -File scripts/build-resources.ps1
# 注意：resources/ 体积大（后端源码 + Portable Python），已加入 .gitignore，不入库。

$ErrorActionPreference = "Stop"
# ⚠️ 2026-10-06 修复：本脚本位于 `<仓库根>/desktop/scripts/`，所以仓库根是 **../..**。
#    此前误写成 `..`（= desktop/），导致后续 robocopy "$root/src"、Copy-Item "$root/pyproject.toml"
#    全部错一层 ⇒ 首次真正执行时以 "系统找不到指定的文件" 失败。
#    ★ 教训：脚本因为"没有执行条件"（沙箱无 npm/Rust）而从未跑过 ⇒ 路径推导错误不会在静态检查里暴露。
$root = Resolve-Path "$PSScriptRoot/../.."        # 仓库根
$tauri = "$root/desktop/src-tauri"
$backend = "$tauri/resources/backend"
$pyDir = "$tauri/resources/python"

# 前置自检：路径推导错误要在第一步就明确报出，而不是让 robocopy 报一堆"找不到文件"
foreach ($must in @("$root/pyproject.toml", "$root/src", "$root/web")) {
    if (-not (Test-Path $must)) {
        Write-Error @"
路径自检失败：未找到 $must
  推断出的仓库根 = $root
  脚本位置        = $PSScriptRoot
  ★ 若上面两行只差一级目录，说明本脚本的相对层级推导被改错了（本脚本应固定为 ../..）。
"@
    }
}

# 1) 清理旧的 backend 副本
if (Test-Path $backend) { Remove-Item $backend -Recurse -Force }
New-Item -ItemType Directory -Force -Path $backend | Out-Null

# 1.5) ★ 两个资源目录都必须**先存在**，否则 Tauri 的 build script 会直接失败：
#      `resource path \`resources\python\` doesn't exist`
#      这个校验发生在 `cargo check` / `cargo build` 阶段（tauri-build 要嵌入资源清单），
#      **不是**等到打包时 —— 所以哪怕只是 `cargo check` 语法，也必须先跑过本脚本。
#      ⚠️ 2026-10-06 修复：此前只"检查 python.exe 是否存在"却从不创建目录，
#      于是缺 Portable Python 时只 Warn、目录始终不存在 ⇒ cargo 报上面那句，很难联想到根因。
New-Item -ItemType Directory -Force -Path $pyDir | Out-Null

# 2) 拷贝后端源码（排除重型/私密/产物目录）
#    ⚠️ /XD tmp* 是刻意加的：开发过程会留下空的 tmp* 目录（git 不跟踪空目录，
#    所以 `git status` 看不见它），而 robocopy /E **会**把空目录一起拷进安装包。
#    ⚠️ /XD __pycache__ 同理且更严重：开发机上的 .pyc 会被打进安装包 ——
#    体积无谓膨胀，且 .pyc 里的绝对路径（含开发机用户名）会随包分发；
#    更麻烦的是**版本不匹配的陈旧 .pyc 会导致难以诊断的导入异常**。
#    ★ 2026-10-06 实测：工作树上跑时拷进了 16 个 __pycache__；
#      而在此之前用 `git archive` 导出的副本测试却"零垃圾" ——
#      **因为导出副本天生不含未跟踪文件，那个测试根本测不出这个缺陷**。
#      教训：打包脚本必须在**真实工作树**上验证，不能只在干净导出上验证。
robocopy "$root/src"          "$backend/src"          /E /NFL /NDL /NJH /XD tmp* __pycache__ .pytest_cache .ruff_cache
robocopy "$root/web"          "$backend/web"          /E /NFL /NDL /NJH /XD node_modules tmp* __pycache__
Copy-Item "$root/pyproject.toml" "$backend/"
Copy-Item "$root/uv.lock"        "$backend/"
Copy-Item "$root/.env.example"   "$backend/"

# ★★ 凡 `pyproject.toml` 元数据里**引用到的文件**都必须一并拷过来 ——
#    否则首启 `uv sync` 在构建项目自身（editable 安装）时会失败。
#    2026-10-06 实测事故：
#      `Building papermate @ file:///.../resources/backend`
#      → `hatchling.build.build_editable` 失败
#      → `OSError: Readme file does not exist: README.md`
#    根因就是本脚本此前只拷 pyproject.toml / uv.lock / .env.example，**漏了 README.md**。
#    ⚠️ 新增/修改 pyproject 的 readme / license.file 等字段时，**必须同步这里**。
#       `tests/test_build_resources_contract.py` 会静态校验这一致性。
foreach ($meta in @("README.md", "LICENSE")) {
    if (Test-Path "$root/$meta") {
        Copy-Item "$root/$meta" "$backend/"
    }
}

# 3) 构建前端（若 dist 缺失）
if (-not (Test-Path "$backend/web/dist/index.html")) {
    Write-Host "[build-resources] building frontend (npm ci + build)..."
    Push-Location "$root/web"
    npm ci
    npm run build
    Pop-Location
    robocopy "$root/web" "$backend/web" /E /NFL /NDL /NJH /XD node_modules tmp* __pycache__
}

# 4) 注入启动等待页（由后端托管，同源轮询 /healthz 后跳转 /）
Copy-Item "$tauri/frontend-dist/loading.html" "$backend/web/dist/loading.html"

# 5) Python 运行时资源检查
if (-not (Test-Path "$pyDir/python.exe")) {
    Write-Warning "[build-resources] 缺少 $pyDir/python.exe —— 请放入 Portable Python 3.12 (embed-amd64) 解压内容"
}
if (-not (Test-Path "$pyDir/uv.exe")) {
    Write-Warning "[build-resources] 缺少 $pyDir/uv.exe —— 请放入 uv 单文件可执行（https://github.com/astral-sh/uv/releases）"
}

Write-Host "[build-resources] done. backend copy at: $backend"
Write-Host "[build-resources] next: cd desktop && npm install && npm run tauri build"
