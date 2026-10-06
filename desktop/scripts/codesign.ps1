param(
  [Parameter(Mandatory = $true)][string]$FileToSign
)
# Authenticode 代码签名（绕过 SmartScreen / 浏览器拦截）
# 由 tauri.conf.json 的 bundle.windows.signCommand 在打包时自动调用，%1 即待签名文件。
# 未配置证书时自动跳过（仅开发包会告警），不阻断构建。
$ErrorActionPreference = "Stop"

$pfx = $env:CODESIGN_PFX
if (-not $pfx -or -not (Test-Path $pfx)) {
  Write-Warning "未配置 CODESIGN_PFX，跳过 Authenticode 签名（开发包将触发 SmartScreen 警告）"
  exit 0
}

# 查找 signtool（Windows SDK 自带）
$signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $signtool) { throw "找不到 signtool.exe，请安装 Windows SDK" }

$ts = $env:CODESIGN_TIMESTAMP_SERVER
if (-not $ts) { $ts = "http://timestamp.digicert.com" }

& $signtool.FullName sign /f $pfx /p $env:CODESIGN_PASSWORD /fd sha256 /tr $ts /td sha256 $FileToSign
if ($LASTEXITCODE -ne 0) { throw "签名失败: $FileToSign" }
Write-Host "已签名: $FileToSign"
