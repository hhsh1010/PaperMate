// 自更新模块（Tauri 2 plugin-updater）
//
// ★ 启用步骤（默认本文件未被 `mod` 引入，不会参与编译；挂载后生效）：
//   1. Cargo.toml 已加 `tauri-plugin-updater = "2"`
//   2. lib.rs 的 run() 增加 `.plugin(tauri_plugin_updater::Builder::new().build())`
//   3. 在 setup 里 spawn 一个异步任务调用 check_for_update（或直接在前端用插件 API）
//   4. 准备更新服务器：返回经 ed25519 私钥签名的 latest.json（见 deploy/update-server/latest.json）
//
// 安全：更新包必须经私钥签名、pubkey 内置于客户端，中间人无法伪造更新。

use tauri::{AppHandle, Manager};
use tauri_plugin_updater::UpdaterExt;

#[tauri::command]
pub async fn check_for_update(app: AppHandle) {
    let updater = match app.updater() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[updater] 初始化失败: {e}");
            return;
        }
    };
    match updater.check().await {
        Ok(Some(update)) => {
            // 下载并安装；安装完成后需重启应用才能生效
            if let Err(e) = update.download_and_install().await {
                eprintln!("[updater] 安装失败: {e}");
            } else {
                app.restart();
            }
        }
        Ok(None) => { /* 已是最新 */ }
        Err(e) => eprintln!("[updater] 检查失败: {e}"),
    }
}
