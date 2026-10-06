Tauri 构建安装包需要图标文件，本目录当前为空占位。

在打包前请生成图标（用任意 1024x1024 PNG logo 执行）：

    cd desktop/src-tauri
    npx tauri icon path/to/logo.png

该命令会在本目录生成 Tauri 要求的全部图标：
    32x32.png
    128x128.png
    128x128@2x.png
    icon.icns   (macOS，Windows 构建可忽略但需存在占位)
    icon.ico    (Windows NSIS 必需)

生成后 `npm run tauri build` 才不会因缺少图标而失败。
