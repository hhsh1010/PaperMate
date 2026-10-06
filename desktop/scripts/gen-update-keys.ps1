# 生成 Tauri 自更新 ed25519 密钥对
# 私钥离线保管（用于对每次发布的 latest.json 签名）；公钥填入 tauri.conf.json 的 plugins.updater.pubkey
$ErrorActionPreference = "Stop"

$outDir = Join-Path $PSScriptRoot "..\keys"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$keyFile = Join-Path $outDir "updater.key"

$password = $env:UPDATER_KEY_PASSWORD
if (-not $password) { $password = "changeme" }

# cargo tauri signer 由 @tauri-apps/cli 提供（桌面目录需先 npm install）
cargo tauri signer generate --password $password --out-file $keyFile

Write-Host "私钥已生成: $keyFile （请离线备份，切勿入库）"
Write-Host "把上面输出的【公钥 pubkey】填入 desktop/src-tauri/tauri.conf.json -> plugins.updater.pubkey"
