Tauri 构建安装包需要图标文件。本目录已提供 `logo-source.png`（1024x1024 占位源图）。

## 一、生成全套图标（正式打包前必做）

```powershell
cd desktop
npm install                      # 首次会装 @tauri-apps/cli
npx tauri icon src-tauri/icons/logo-source.png
```

`npx tauri icon`（Tauri 官方 CLI）会一次性产出 `bundle.icon` 里配置的全部文件：

| 文件 | 用途 | 必需性 |
|---|---|---|
| `32x32.png` | 任务栏/资源管理器 小图标 | 必需 |
| `128x128.png` | 通用 PNG 图标 | 必需 |
| `128x128@2x.png` | 高 DPI（256x256） | 必需 |
| `icon.ico` | **Windows NSIS 安装包必需** | 必需 |
| `icon.icns` | macOS | Windows 构建也会生成，占位即可 |

> ⚠️ 以上 5 个文件名与 `tauri.conf.json` 的 `bundle.icon` 数组一一对应，**缺任何一个都会构建失败**。
> ⚠️ 官方 README 里常见的说法是"用 `tauri icon` 生成"——它是唯一可靠的产出方式，
> 手工拼 `.ico` 容易缺尺寸档位导致安装器图标变空白。

## 二、换成正式 logo

`logo-source.png` 是**脚本生成的占位图**（几何风格，无字体依赖），正式发布前应替换：

```powershell
# 用真实 logo（建议 ≥1024x1024 正方形 PNG）覆盖后重跑
copy C:\path\to\real-logo.png desktop\src-tauri\icons\logo-source.png
cd desktop ; npx tauri icon src-tauri/icons/logo-source.png
```

若没有现成 logo，可重新生成占位图后手工替换：
```powershell
.venv\Scripts\python.exe desktop\scripts\gen-icon-placeholder.py
```

## 三、素材提醒

`web/public/hero-illustration.png` 是 **1184x760 的横幅插画**，不是方形，**不适合直接当图标**
（会变形或留白）。若想复用它，必须先中心裁切成正方形再喂给 `tauri icon`。
