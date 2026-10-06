# build-resources.ps1 — 准备 Tauri 打包所需的运行时资源
# 作用：构建前端 → 拷贝后端源码(含 web/dist) → 放入 Portable Python + uv → 注入 loading.html
# 运行：在 desktop/ 目录下执行  powershell -ExecutionPolicy Bypass -File scripts/build-resources.ps1
# 注意：resources/ 体积大（后端源码 + Portable Python），已加入 .gitignore，不入库。

$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."          # 仓库根
$tauri = "$root/desktop/src-tauri"
$backend = "$tauri/resources/backend"
$pyDir = "$tauri/resources/python"

# 1) 清理旧的 backend 副本
if (Test-Path $backend) { Remove-Item $backend -Recurse -Force }
New-Item -ItemType Directory -Force -Path $backend | Out-Null

# 2) 拷贝后端源码（排除重型/私密目录）
#    ⚠️ /XD tmp* 是刻意加的：开发过程会留下空的 tmp* 目录（git 不跟踪空目录，
#    所以 `git status` 看不见它），而 robocopy /E **会**把空目录一起拷进安装包。
robocopy "$root/src"          "$backend/src"          /E /NFL /NDL /NJH /XD tmp*
robocopy "$root/web"          "$backend/web"          /E /NFL /NDL /NJH /XD node_modules tmp*
Copy-Item "$root/pyproject.toml" "$backend/"
Copy-Item "$root/uv.lock"        "$backend/"
Copy-Item "$root/.env.example"   "$backend/"

# 3) 构建前端（若 dist 缺失）
if (-not (Test-Path "$backend/web/dist/index.html")) {
    Write-Host "[build-resources] building frontend (npm ci + build)..."
    Push-Location "$root/web"
    npm ci
    npm run build
    Pop-Location
    robocopy "$root/web" "$backend/web" /E /NFL /NDL /NJH /XD node_modules tmp*
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
