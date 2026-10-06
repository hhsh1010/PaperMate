Tauri 构建安装包需要图标文件。本目录已提供 `logo-source.png`（1024x1024 占位源图）
与**已生成的全套图标**（5 个文件，见下）。

## 一、生成全套图标（两种方式任选其一）

### 方式 A · 仓库自带脚本（**无需 npm，最快**）

```powershell
cd <项目根>
.venv\Scripts\python.exe desktop\scripts\gen-icon-placeholder.py
```

用 Pillow 直接产出源图 + Tauri 要求的全部 5 个文件（实测可写多尺寸 ICO 与 ICNS）。
生成的 `icon.ico` 内含 **7 个尺寸档位**（16/24/32/48/64/128/256）。
适合"先把 `cargo check` 跑通"，不依赖任何网络与 node 环境。

### 方式 B · 官方 CLI（有真实 logo 时**推荐**）

```powershell
cd desktop
npm install                      # 首次会装 @tauri-apps/cli
npx tauri icon src-tauri/icons/logo-source.png
```

| 文件 | 用途 | 必需性 |
|---|---|---|
| `32x32.png` | 任务栏/资源管理器 小图标 | 必需 |
| `128x128.png` | 通用 PNG 图标 | 必需 |
| `128x128@2x.png` | 高 DPI（256x256） | 必需 |
| `icon.ico` | **Windows NSIS 安装包 + Windows Resource 必需** | 必需 |
| `icon.icns` | macOS | Windows 构建不需要，但配置里列了就必须存在 |

> ⚠️ 以上 5 个文件名与 `tauri.conf.json` 的 `bundle.icon` 数组一一对应。
> **缺任何一个，`cargo check` 阶段就会失败**（不是等到打包时）：
> ```
> `...icons/icon.ico` not found; required for generating a Windows Resource file
> ```
> 手工拼 `.ico` 容易缺尺寸档位导致安装器图标空白，**优先用上面两种方式之一生成**。

## 二、换成正式 logo

`logo-source.png` 是**脚本生成的占位图**（几何风格，无字体依赖），正式发布前应替换：

```powershell
# 用真实 logo（建议 ≥1024x1024 正方形 PNG）覆盖后重跑
copy C:\path\to\real-logo.png desktop\src-tauri\icons\logo-source.png
cd desktop ; npx tauri icon src-tauri/icons/logo-source.png
```

## 三、素材提醒

`web/public/hero-illustration.png` 是 **1184x760 的横幅插画**，不是方形，**不适合直接当图标**
（会变形或留白）。若想复用它，必须先中心裁切成正方形再喂给生成工具。

