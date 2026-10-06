# firstuse-sim.ps1 —— 在**本机现有源码树**上仿真「新用户第一次打开 App」
#
# ★★ 核心原理（为什么不用复制项目）
# 「首次使用体验」由**运行时三件套**决定，与代码在哪份副本里无关：
#   ① PAPERMATE_DATA_DIR 为空   ⇒ provider_settings.json 不存在 ⇒ 走首装种子层（fake / 无 key）
#   ② HF_HOME 为空              ⇒ 没有任何模型权重 ⇒ 设置页必须显示「未安装」
#   ③ 进程里没有厂商 key、CWD 下没有 .env ⇒ 无法调任何真实 LLM
# 把这三样一次性换成全新的临时目录，你**用的还是正在开发的那份源码**，
# 所以：仿真结果 == 真机首装逻辑，而本地开发进度完全不受影响。
#
# ⚠️ 三条硬性纪律（本脚本已实现，改动时勿破坏）
#   1. CWD 必须是干净目录：config.py 的 env_file=".env" 从 **CWD** 读取，
#      若在项目根启动就会读到你真实的 .env（带真 key）⇒ 仿真失效 + 密钥语义失真。
#   2. 必须挑一个**未被占用**的端口：你的开发后端可能正占着 8000。
#   3. 绝不写 $Root/data —— 数据目录一律落在临时目录下。
#
# 用法（PowerShell 5.1，不要用 && 连接）：
#   powershell -ExecutionPolicy Bypass -File scripts\firstuse-sim.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\firstuse-sim.ps1 -Port 8124
# 停止：Ctrl+C（后台 Python 进程随之退出；端口不释放时 taskkill /PID <pid> /F）
#
# 判据（启动后另开一个终端核对）：
#   curl.exe http://127.0.0.1:8123/api/v1/healthz
#     → rag_embedder = "fake"、default_model = "fake"
#   curl.exe http://127.0.0.1:8123/api/v1/settings/rag/models
#     → all_installed = false、enabled = false、models[*].installed = false
#   curl.exe http://127.0.0.1:8123/api/v1/settings
#     → providers.deepseek.configured = false（无 key）
#   ※ 再打开 Web UI 走一遍：填 key → 点「下载模型」→ 切 real → 重建索引。
#   ※ 不必真的把 4.5GB 下完 —— 看到 install 任务 running、file_index 在推进就够了
#     （Ctrl+C 中断后已下载部分保留在 HF_HOME，下次会自动续传，正好把续传也验了）。

param(
    [int]$Port = 8123,
    [string]$Root = (Resolve-Path "$PSScriptRoot/..").Path,
    [string]$SimRoot = "$env:TEMP\papermate-firstuse"
)

$ErrorActionPreference = "Stop"

# ── 0) 前置检查：端口必须空闲（否则会与你的开发后端抢同一个 data_dir / SQLite）──
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Error "端口 $Port 已被占用（PID $(($busy[0].OwningProcess))）。请换一个端口：-Port 8124"
}

$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Error "找不到虚拟环境解释器 $venvPy"
}

# ── 1) 全新运行时目录（每次一个时间戳 ⇒ 天然回到「全新用户」状态）──
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$run   = Join-Path $SimRoot $stamp
$data  = Join-Path $run "data"
$hf    = Join-Path $run "huggingface"
New-Item -ItemType Directory -Force -Path $data, $hf | Out-Null

# ── 2) 清掉「用户还没填」的东西：厂商 key ──
foreach ($k in @("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")) {
    Remove-Item "Env:$k" -ErrorAction SilentlyContinue
}
# 删除而非置空：置空字符串在许多 SDK 里仍算"已配置"，会让「无 key」路径测不出来。

# ── 3) 运行时三件套 + 安全默认 ──
$env:PAPERMATE_DATA_DIR       = $data
$env:HF_HOME                  = $hf
$env:HF_ENDPOINT              = "https://hf-mirror.com"
$env:PAPERMATE_HOST           = "127.0.0.1"
$env:PAPERMATE_PORT           = "$Port"
$env:PAPERMATE_PREWARM        = "0"    # 关预热：首次也别偷偷加载 BGE-M3
Remove-Item Env:PAPERMATE_DEFAULT_EMBEDDER -ErrorAction SilentlyContinue  # 走首装种子层
$env:PYTHONPATH               = (Join-Path $Root "src")
$env:PYTHONIOENCODING         = "utf-8"

# ── 4) 自检：确认「新用户」三条件成立（失败就别往下跑，否则是伪仿真）──
if (Test-Path (Join-Path $data "provider_settings.json")) {
    Write-Error "数据目录里有 provider_settings.json —— 这不是新用户状态：$data"
}
if (Test-Path (Join-Path $data "papermate.db")) {
    Write-Error "数据目录里已有 SQLite 库 —— 不是新用户状态：$data"
}
$weightCount = 0
if (Test-Path $hf) { $weightCount = @(Get-ChildItem $hf -Recurse -File -ErrorAction SilentlyContinue).Count }
if ($weightCount -gt 0) {
    Write-Error "HF_HOME 里已有 $weightCount 个文件 —— 不是新用户状态：$hf"
}

Write-Host ""
Write-Host "=== PaperMate 新用户首装仿真 ===" -ForegroundColor Cyan
Write-Host ("  源码       : {0}   (仍在开发中的那一份, 未做任何复制)" -f $Root) -ForegroundColor Gray
Write-Host ("  数据目录   : {0}" -f $data) -ForegroundColor Gray
Write-Host ("  模型缓存   : {0}   (空 => 复现未安装状态)" -f $hf) -ForegroundColor Gray
Write-Host ("  端口       : {0}" -f $Port) -ForegroundColor Gray
Write-Host "  厂商 key   : 全部清空" -ForegroundColor Gray
Write-Host ""
Write-Host "  预期 /healthz：rag_embedder=fake、default_model=fake" -ForegroundColor Yellow
Write-Host "  预期 /settings/rag/models：all_installed=false、enabled=false" -ForegroundColor Yellow
Write-Host "  Ctrl+C 结束。★ 本脚本不读写 $Root\data，你的开发数据零风险。" -ForegroundColor Yellow
Write-Host ""

Push-Location $run
try {
    & $venvPy -m papermate.api
}
finally {
    Pop-Location
    Write-Host "[firstuse-sim] 后端已退出。仿真目录保留在：$run（可删，也可留着复现）" -ForegroundColor DarkGray
}
