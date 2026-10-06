# PaperMate 本地冒烟启动器（阶段0：不上桌面壳，先验证「前端+后端+UI」全链路）
#
# 用法（在仓库根目录）：
#   powershell -ExecutionPolicy Bypass -File scripts/run-desktop-smoke.ps1
#
# 行为：
#   1) 依赖：若 .venv 不存在则 uv sync（首次需联网装 torch 等）
#   2) 前端：若 web/dist/index.html 不存在则 npm run build
#   3) 起后端 python -m papermate.api（默认 fake 模型，免 4GB 模型下载）
#   4) 轮询 /api/v1/healthz 直到就绪
#   5) 打开浏览器到 http://127.0.0.1:8000/
#
# 说明：本脚本仅用于「本地快速试效果」，不是最终桌面客户端（见 .workbuddy/artifacts/desktop-client-plan-2026-10-05.md）。

$ErrorActionPreference = "Stop"
$Root = (Split-Path -Parent $MyInvocation.MyCommand.Path) | Split-Path -Parent
Set-Location $Root

$VenvPy  = Join-Path $Root ".venv/Scripts/python.exe"
$Port    = 8000
$DataDir = Join-Path $env:LOCALAPPDATA "PaperMate/smoke-data"

# 1) 依赖（缺失才装）
if (-not (Test-Path $VenvPy)) {
    Write-Host ">>> 首次安装依赖 (uv sync --extra rag --extra dev) ..."
    uv sync --extra rag --extra dev
}

# 2) 前端产物（缺失才构建）
if (-not (Test-Path (Join-Path $Root "web/dist/index.html"))) {
    Write-Host ">>> 构建前端 (npm ci && npm run build) ..."
    Push-Location web
    npm ci
    npm run build
    Pop-Location
}

# 3) 起后端（fake 模式，免模型下载；仅本机可达）
$env:PAPERMATE_HOST        = "127.0.0.1"
$env:PAPERMATE_PORT        = "$Port"
$env:PAPERMATE_DEFAULT_MODEL = "fake"
$env:PAPERMATE_DATA_DIR    = $DataDir
Write-Host ">>> 启动后端 http://127.0.0.1:$Port (data=$DataDir) ..."
$proc = Start-Process -FilePath $VenvPy -ArgumentList "-m", "papermate.api" -NoNewWindow -PassThru

# 4) 等 healthz
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/healthz" -TimeoutSec 2
        $ok = $true
        break
    }
    catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $ok) {
    Write-Error "后端未在 30s 内就绪，查看日志/终端输出"
    Stop-Process -Force $proc -ErrorAction SilentlyContinue
    exit 1
}
Write-Host ">>> 后端就绪：healthz = $($r | ConvertTo-Json -Compress)"

# 5) 开浏览器
Write-Host ">>> 打开浏览器 http://127.0.0.1:$Port/ （关闭后端进程即退出）"
Start-Process "http://127.0.0.1:$Port/"

Wait-Process $proc
