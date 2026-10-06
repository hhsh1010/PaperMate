# PaperMate 桌面端 · 出包逐步操作清单

> 面向「第一次真正把安装包打出来」。全部命令在 **PowerShell 5.1** 下可用（**不要用 `&&`** 连接）。
> 每步都给了**成功判据** —— 判据不满足就别往下走，否则错误会推迟到后面才爆。
>
> ⚠️ 本清单涉及的 `cargo` 相关步骤**未在开发环境验证过**（沙箱无 Rust 工具链）。

---

## 第 0 步 · 前置检查（缺一个都别往下走）

在**项目根**目录开一个 PowerShell 窗口，逐条执行：

```powershell
node -v
npm -v
python --version
rustc --version
cargo --version
```

| 项 | 要求 | 不满足怎么办 |
|---|---|---|
| Node | ≥ 18（Vite 5 要求） | 装 Node 20 LTS |
| Python | 仅用于本项目的开发后端，打包**不依赖**系统 Python | —— |
| Rust | `rustc` / `cargo` 都有 | 见 **第 1 步** |

---

## 第 1 步 · 国内镜像（**必做，否则大概率卡死**）

实测：本机 `registry.npmjs.org` **15 秒超时不可达**；`crates.io` 同理。

```powershell
# npm
npm config set registry https://registry.npmmirror.com
npm config get registry
```
✅ 判据：输出 `https://registry.npmmirror.com/`

### cargo 镜像 —— ★ 必须能加速 **crate 下载**，只换索引没用

**实测教训（2026-06-06，本机）**：索引镜像 ≠ 下载镜像。很多教程只让你换 `registry`，
但索引里的 `config.json` 若仍写着官方 `dl`，**大头（几百 MB 的 .crate 文件）还是走官方源**。

| 源 | `config.json` 里的 `dl` | 9.3MB 样本实测 |
|---|---|---|
| `mirrors.ustc.edu.cn/crates.io-index/` | `https://static.crates.io/crates`（**未重写**） | —— |
| `mirrors.tuna.tsinghua.edu.cn/crates.io-index/` | `https://static.crates.io/crates`（**未重写**） | —— |
| **`rsproxy.cn/index/`** | **`https://rsproxy.cn/api/v1/crates`**（已重写 ✅） | **3.3 MB/s** |
| 官方 `static.crates.io`（不配镜像时的实际下载源） | —— | **29 KB/s**（1.1MB 就 40s 超时） |

⇒ **只配 rsproxy 有效**（差约 100 倍）。用清华/ustc 只换索引，下载依旧几小时。

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.cargo" | Out-Null
@"
[source.crates-io]
replace-with = 'rsproxy-sparse'

[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"

[net]
git-fetch-with-cli = true
"@ | Set-Content -Path "$env:USERPROFILE\.cargo\config.toml" -Encoding ASCII
```

> ⚠️ 必须用 `-Encoding ASCII`：PS 5.1 的 `-Encoding UTF8` 会写入 **BOM**，
> 而 cargo 的 TOML 解析器遇到 BOM 会直接报错。本文件内容全是 ASCII，用 ASCII 编码正好无 BOM。

✅ 判据：
```powershell
Get-Content "$env:USERPROFILE\.cargo\config.toml"
```
配好后**重跑 `cargo check`**（若已在跑，`Ctrl+C` 中断再重跑；已落盘的索引缓存会复用）。

---

## 第 2 步 · 装 Rust 工具链

### ★★ 不要用 `irm https://sh.rustup.rs | iex`（常见错误）

`sh.rustup.rs` 返回的是 **POSIX sh 脚本**（给 Linux/macOS 的 `curl | sh` 用），
在 PowerShell 里 `iex` 执行会报一堆 `(”后面应为表达式` / `标记“||”不是有效语句分隔符` —— 因为它根本不是 PS 语法。

实测两个 URL 的返回差异：

| URL | 实际返回 |
|---|---|
| `https://sh.rustup.rs` | `#!/bin/sh` 文本脚本 |
| `https://win.rustup.rs` | **HTTP 200 · `application/x-msdownload` · 9.9MB · 开头 `MZ`**（真正的 PE 可执行文件） |

### 正确做法

```powershell
# 1) 镜像（rustup-init 装工具链时会读这两个变量，必须同一窗口先设好）
$env:RUSTUP_DIST_SERVER = "https://mirrors.tuna.tsinghua.edu.cn/rustup"
$env:RUSTUP_UPDATE_ROOT = "https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"

# 2) 下载安装器
cd $env:USERPROFILE\Downloads
irm https://win.rustup.rs -OutFile rustup-init.exe

# 3) ★ 先校验下对的是"可执行文件"而不是又一次的脚本
Get-Item rustup-init.exe | Select-Object Name, Length
```
✅ 判据：`Length` 约 **9,900,000+** 字节。
❌ 若只有几十 KB ⇒ 又是脚本，**不要执行**，改用下方备用地址。

```powershell
# 备用下载地址（win.rustup.rs 若被网络中间件改写）
irm https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe -OutFile rustup-init.exe
```

```powershell
# 4) 静默安装（默认 stable + x86_64-pc-windows-msvc + 完整 profile）
.\rustup-init.exe -y --default-toolchain stable

# 5) 让**当前**窗口立即可用（不必重开）
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

# 6) 验证
cargo --version
rustc --version
```
✅ 判据：两条都打印版本号

### ★ 关于 `cargo check` 是否需要链接器 —— 更正（2026-06-06 实测踩到）

**早期说法「`cargo check` 不需要链接器」是错的**，至少对**本项目**不成立。

- 对**纯类型检查**确实不链接…**但 `cargo check` 仍会编译并运行依赖的 `build.rs` 构建脚本，
  以及把 proc-macro crate 编译成 `.dll`** —— 这两件事都需要链接器。
- Tauri 的依赖树里全是这类：`proc-macro2` / `quote` / `serde_derive` / `cc` /
  `phf_generator` / `icu_*` …
- 实测症状：`error: linker 'link.exe' not found` + 一大批
  `could not compile xxx (build script) due to 1 previous error`。

⇒ **结论：`cargo check` 与 `tauri build` 都需要 MSVC 链接器**，装 VS Build Tools 无法推迟。

### 装 VC++ 工具链（先做这一步，再跑任何 cargo 命令）

**方式 A · winget（推荐，快）**
```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```
- `--passive`：显示进度但不需交互（`--quiet` 会全程无输出，几 GB 下载期间你会怀疑是不是卡了）
- `--wait`：等待安装完成才返回
- 下载约 **2~4GB**，10~30 分钟

**方式 B · 图形安装器（想看清楚装了什么时用）**
```powershell
cd $env:USERPROFILE\Downloads
irm https://aka.ms/vs/17/release/vs_BuildTools.exe -OutFile vs_BuildTools.exe
Get-Item vs_BuildTools.exe | Select-Object Name, Length   # 应约 1~3MB（只是个引导器）
.\vs_BuildTools.exe
```
在弹出的界面里勾选 **「使用 C++ 的桌面开发」**（Desktop development with C++），然后安装。
⚠️ **不要只装 "MSBuild" 或 "C++ CLI"** —— 必须包含 **MSVC 生成工具 + Windows SDK**，
否则 `link.exe` 仍然找不到。

### ✅ 判据（两条都要过）
```powershell
# 1) vswhere 能查到含 VC++ x64 工具链的实例
& "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
```
✅ 有路径输出（**无输出 = 没装成功**）

```powershell
# 2) 真的能找到 link.exe（Rust 通过 vswhere/注册表定位，不依赖 PATH）
Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\link.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 FullName
```
✅ 打印出 `link.exe` 的完整路径

装完**新开一个 PowerShell 窗口**再跑 `cargo check`。

### ★ 若装好了但 `cargo check` 仍报 `link.exe not found`

极少数情况下 rustc 的工具链探测失效（实例注册信息异常、多版本冲突等）。
此时**直接进开发者环境**绕过探测 —— `vcvars64.bat` 会把 `LIB` / `INCLUDE` / `PATH`
都设好，rustc 见到这些变量就直接用，不再需要 vswhere：

```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo check'
```

跑通后确认是"探测失效"而非"没装"：可在**开发者命令提示符**里长期构建，
或把 `vcvars64.bat` 的内容写进一个 `dev-shell.ps1` 每次构建前调用。

> 本机 2026-10-06 实测：装好 Build Tools 后 `vswhere -requires
> Microsoft.VisualStudio.Component.VC.Tools.x86.x64` **能正常返回安装路径**，
> `cargo check` 不再需要这个兜底。

### 排障用：确认"装了但没注册"还是"压根没装"

VS 实例的注册信息保存在：
```
C:\ProgramData\Microsoft\VisualStudio\Packages\_Instances\<实例ID>\state.json
```
里面有 `installationPath` / `installDate` / `selectedPackages`（含组件 ID）。
**比 `vswhere` 更可靠** —— 它是安装器的原始记录，能区分"文件在但注册丢了"与"没装"。
本机实测该文件记录：`installationPath = ...\2022\BuildTools`、
`installDate = 2026-10-06T13:28:10Z`、含 `Microsoft.VisualStudio.Component.VC.Tools.x86.x64` +
`Windows11SDK.26100`。

### 备选方案：切 GNU 工具链（**不推荐**）
```powershell
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```
GNU 工具链自带 MinGW 的 `ld`，**不需要 VS Build Tools**（只 ~100MB）。
但 **Tauri 在 Windows 上官方支持的是 MSVC 目标**，GNU 可能出现
`webview2-com` / NSIS 打包等环节的兼容问题。**仅在实在装不上 Build Tools 时作为兜底。**

## ★ `cargo check` / `tauri build` 的前置清单（缺一不可）

Tauri 的 build script 在 **`cargo check` 阶段**就会校验下面两类资源，
**不是**等到打包时才检查 —— 少任何一个都会直接失败：

| 前置 | 产出物 | 缺失时的报错 |
|---|---|---|
| 第 3 步 | `resources/python/`（**目录**存在即可） | `resource path 'resources\python' doesn't exist` |
| 第 4 步 | `resources/python/uv.exe` | （打包后运行时才失败） |
| 第 5 步 | `icons/` 下 **5 个图标文件** | `` `...icons/icon.ico` not found; required for generating a Windows Resource file `` |
| 第 6 步 | `resources/backend/`（含 `web/dist`） | `resource path 'resources\backend' doesn't exist` |

> ★ 2026-10-06 实测：连续踩了后两条 —— 都是"以为只有打包才需要、其实 cargo 阶段就查"。
> **判据**：`cargo check` 报"某某不存在"时，先怀疑**前置产物没生成**，而不是怀疑配置写错。

---

## 第 3 步 · 准备 Portable Python 3.12（embed 版）

**为什么必须**：安装包要自带 Python 解释器 —— 目标用户机器上可能根本没装 Python，或版本不对。

1. 下载：<https://www.python.org/downloads/windows/> → 选 **3.12** → 找 **Windows embeddable package (64-bit)**
2. 解压，把**全部内容**放进：
   `desktop\src-tauri\resources\python\`
   （该目录已存在；解压后应能看到 `python.exe`）

### ★ 第 3.1 步 · 改 `python312._pth`（**最容易漏、报错最含糊**）

用记事本打开 `resources\python\python312._pth`，把这一行的 `#` 去掉：

```
python312.zip
.
#import site          ← 改成：import site
```

**为什么必须**：embed 版默认用隔离模式接管 `sys.path`，`#import site` 意味着
**site-packages 机制被禁用** ⇒ `uv` 建 venv / 往里装包会失败，报错还很难懂。

### ✅ 判据（必须跑）

```powershell
cd desktop\src-tauri\resources\python
.\python.exe -c "import sys, site; print(sys.version); print('site ok:', site.getsitepackages() is not None)"
```
✅ 输出里必须有 `site ok: True`
❌ 若报 `ModuleNotFoundError: site` → `.pth` 没改对，回去检查

---

## 第 4 步 · 放 `uv.exe`

1. 下载：<https://github.com/astral-sh/uv/releases> → `uv-x86_64-pc-windows-msvc.zip`
2. 把 `uv.exe`（同目录的 `uvx.exe` 一并）放进**同一个目录**：
   `desktop\src-tauri\resources\python\`

### ✅ 判据

```powershell
cd desktop        # 回到 desktop
.\src-tauri\resources\python\uv.exe --version
```
✅ 打印 `uv 0.x.x`

> ⚠️ 缺这两个文件时 `build-resources.ps1` **只告警不中止** ⇒ 上面两条判据必须自己跑。

---

## 第 5 步 · 装 Tauri CLI + 生成图标

```powershell
cd <项目根>\desktop
npm install
```
✅ 判据：出现 `added N packages`，无 ERR

**★ 两种方式任选其一：**

**方式 A · 仓库自带脚本（Pillow，无需 npm）—— 最快，能立刻解掉 cargo 的图标报错**
```powershell
cd <项目根>
.venv\Scripts\python.exe desktop\scripts\gen-icon-placeholder.py
```
一次产出源图 + **全部 5 个文件**（实测 Pillow 可写多尺寸 ICO 与 ICNS）。
生成的 `icon.ico` 内含 7 个尺寸档位（16/24/32/48/64/128/256），覆盖任务栏与资源管理器各视图。

**方式 B · 官方 CLI（有真实 logo 时推荐）**
```powershell
cd <项目根>\desktop
npm install
npx tauri icon src-tauri/icons/logo-source.png
```
Tauri 自己的 CLI 产出的 `.ico`/`.icns` 质量最好。

### ✅ 判据（5 个文件缺一不可）

```powershell
Get-ChildItem src-tauri\icons\*.png, src-tauri\icons\*.ico, src-tauri\icons\*.icns | Select-Object Name, Length
```
必须看到这 5 个（与 `tauri.conf.json` 的 `bundle.icon` 一一对应）：
```
32x32.png   128x128.png   128x128@2x.png   icon.ico   icon.icns
```

> ⚠️ **缺任何一个，`cargo check` 阶段就会失败**（不是等到打包）——
> `` `...icons/icon.ico` not found; required for generating a Windows Resource file ``
> `logo-source.png` 是占位图，正式发布前用真实 logo 覆盖后重跑（优先方式 B）。

---

## 第 6 步 · 准备后端副本 + 构建前端（★ **必须在 `cargo check` 之前**）

> ### ⚠️ 为什么这一步必须排在 `cargo check` 前面
>
> `tauri.conf.json` 的 `bundle.resources` 声明了 `resources/python` 与 `resources/backend`
> 两个目录，而 **Tauri 的 build script 在 `cargo check` / `cargo build` 阶段就会校验它们是否存在**
> （要嵌入资源清单），**不是**等到打包时。目录不存在时直接失败：
>
> ```
> resource path `resources\python` doesn't exist
> ```
>
> 这两个目录都由本脚本产出 ⇒ **没跑过本脚本就不能跑任何 cargo 命令**。
> （2026-10-06 实测踩到：文档早期把 cargo check 排在本步之前，于是卡在这里。）

```powershell
cd <项目根>\desktop
powershell -ExecutionPolicy Bypass -File scripts\build-resources.ps1
```
✅ 判据：末尾输出
```
[build-resources] done. backend copy at: ...\desktop\src-tauri\resources\backend
```
（末尾关于缺 `python.exe` / `uv.exe` 的**告警**若出现，说明第 3/4 步没做完 ——
 但目录会被创建，`cargo check` 可以先过；真正打包前必须补齐）

### ✅ 产物自检（务必看一眼）

```powershell
Test-Path .\src-tauri\resources\python
Test-Path .\src-tauri\resources\backend\pyproject.toml
Test-Path .\src-tauri\resources\backend\src\papermate\models\db.py
Test-Path .\src-tauri\resources\backend\web\dist\index.html
Test-Path .\src-tauri\resources\backend\web\dist\loading.html
```
五条都应是 `True`。
（第三条尤其重要 —— 历史上 `.gitignore` 的裸模式 `models/` 曾把整个源码包排除在 git 之外，
导致打出来的包启动即 `ModuleNotFoundError`）

---

## 第 7 步 · ★ `cargo check`（验证 `lib.rs` 的改动能编译）

> ### 先跑 7.0：离线校验 `tauri.conf.json`（**不需要 Rust，秒级**）
>
> `tauri.conf.json` 唯一会被校验的时机是 `tauri-build` 用 serde 解析它，而那需要 Rust 工具链，
> 所以没有 Rust 时配置错误会一路漏到出包才炸。本仓库提供了离线校验：
>
> ```powershell
> cd <项目根>
> .venv\Scripts\python.exe scripts\validate_tauri_config.py
> ```
> ✅ 判据：`✅ desktop\src-tauri\tauri.conf.json 通过 Tauri 2 schema 校验`
>
> ⚠️ 别小看这一步。实测踩到：`webviewInstallMode` 写成 `{"mode": "..."}`，
> 而它其实是 **tagged enum**、键名必须是 `type` ⇒ serde 报
> `missing field 'type'`（**文案说"缺字段"，真实原因是"键名写错"**，很有误导性）。
> 同一轮还查出 `signCommand` 用了 `pwsh`，而多数机器没装 PowerShell 7 ⇒
> `tauri build` 会在签名环节失败。这两个错都只需这一条命令就能提前发现。
>
> 官方 schema 已随仓库分发（`desktop/src-tauri/tauri.schema.json`，169KB），**完全离线**。

```powershell
cd desktop\src-tauri
cargo check
```
✅ 判据：末尾出现 `Finished` 且**无 error**

有报错时看简明原因：
```powershell
cargo check --message-format short
```

> **这一步为什么必须你自己跑**：开发沙箱里没有 Rust 工具链，
> `lib.rs` 的改动（首启排除重依赖）**从未被编译验证过**。
> 改动很小（一个常量 + 一个 for 循环 + 两个 env），但仍需你确认。
> 若报 `.iter()` 相关的类型错误，把常量改成 `&[&str]` 即可：
> `const HEAVY_DEPS: &[&str] = &["torch", "FlagEmbedding", "sentence-transformers"];`

确认通过后回一层：
```powershell
cd ..
```

---

## 第 8 步 · 出包（**分两跑，先快后全**）

> ### ★ 重要预期：`cargo check` 之后再 build 仍要重编一遍
>
> `cargo check` **只产出元数据、不做代码生成** ⇒ 它留下的缓存对 `cargo build` **不可复用**。
> 所以第一次 build（哪怕 debug）仍需把所有依赖**真正编译一遍**：
>
> | 命令 | 额外耗时（首次） | 产出 |
> |---|---|---|
> | `--debug --no-bundle` | 最快（无优化、不打包） | 可运行的 exe（debug） |
> | `--debug` | 较快 | debug 安装包 |
> | 默认（release） | **最慢**（开优化 + 需下载 NSIS） | release 安装包（**给用户的那份**） |

### 8A · 先跑「快速验证版」（**推荐第一步**）

```powershell
cd <项目根>\desktop
npm run tauri build -- --debug --no-bundle
```

- `--debug`：用 debug profile，**不开启优化** ⇒ 编译快得多
- `--no-bundle`：**跳过安装包制作** ⇒ 不下载 NSIS、不做签名
- ✅ 判据：`Finished` 且出现 → `target\debug\papermate-desktop.exe`
- 👉 **直接用这个 exe 走第 9 步的首装验收** —— 功能与 release 版一致，迭代快得多

### 8B · 再出正式安装包（验收通过后再做）

```powershell
cd <项目根>\desktop
npm run tauri build
```

耗时最长（开优化重编 + 首次下载 NSIS，**十几分钟到半小时**）。

✅ 判据 — 产物两个：

| 路径 | 说明 |
|---|---|
| `src-tauri\target\release\bundle\nsis\*.exe` | **安装包**（给最终用户） |
| `src-tauri\target\release\papermate-desktop.exe` | **免安装绿色版**（双击即跑） |

### ⚠️ 8B 可能踩到的点

| 现象 | 原因 | 处理 |
|---|---|---|
| 长时间无输出（>10 分钟） | release 编译 Rust 依赖，属正常 | 别中断，看 `target\release` 体积在涨 |
| 签名环节报找不到 `scripts/codesign.ps1` | 该路径相对**运行时的 cwd**（应为 `desktop\`）解析 | 确认是在 `desktop\` 下执行的；或临时把 `tauri.conf.json` 的 `signCommand` 整段删掉再打 |
| 卡在下载 NSIS | 首次打包需从 GitHub 拉 NSIS | 确认能访问 github.com；`--bundles nsis` 已显式指定 |
| 想跳过一切交互提示 | —— | 加 `--ci` |

---

✅ 判据 — 产物两个：

| 路径 | 说明 |
|---|---|
| `src-tauri\target\release\bundle\nsis\*.exe` | **安装包**（给最终用户） |
| `src-tauri\target\release\papermate-desktop.exe` | **免安装绿色版**（快速验证用，跳过 NSIS） |

> 建议：先用绿色版验证首装流程，满意后再出安装包 —— 迭代快得多。

---

## 第 9 步 · 首装验证（**关键：先清干净旧状态**）

安装包把数据写进 `%LOCALAPPDATA%\PaperMate`。在开发机上测首装，**必须先清掉它**（等价于全新机器）：

```powershell
# 1) 若 App 正在运行，先结束
taskkill /IM papermate-desktop.exe /F

# 2) 清空运行时状态
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\PaperMate"
```
✅ 判据：该目录已不存在

### 启动

```powershell
cd <项目根>\desktop
.\src-tauri\target\release\papermate-desktop.exe
```

### 首启日志（后台窗口）应出现两行

```
[desktop] first launch: installing base dependencies via uv (~0.8GB, may take minutes)...
[desktop] note: PyTorch and other heavy deps are DEFERRED; enable them later in Settings > Semantic Retrieval.
```

### ✅ 首装验收判据（逐条核对）

| 检查 | 期望 | 怎么查 |
|---|---|---|
| 窗口不白屏 | 等待后自动跳到界面 | 眼看 |
| 日志无 `uv sync failed` | 首启成功 | 窗口日志 |
| **venv 里没有 torch** | ★ 这是本轮优化的核心 | 见下方命令 |
| 后端健康 | `rag_embedder=fake` | `curl.exe http://127.0.0.1:8000/api/v1/healthz` |
| 设置页·运行环境 | 显示「未安装」 | 界面观察 |
| 设置页·模型 | 两个都「未安装」，按钮为「准备环境并下载（约 8.5GB）」 | 界面观察 |
| API Key 区 | 两家厂商都「未配置」 | 界面观察 |
| **`deps-ready` 标记已写** | 存在，且内容形如 `papermate-deps-v1 <16 位十六进制>` | `Get-Content "$env:LOCALAPPDATA\PaperMate\deps-ready"` |
| 上传 PDF | 正常解析 | 界面操作 |

**★ 验证"重依赖确实没装"（最关键的一条）**：
```powershell
Test-Path "$env:LOCALAPPDATA\PaperMate\venv\Lib\site-packages\torch"
```
✅ 必须是 `False`
（若为 `True` ⇒ `--no-install-package` 没生效，首启又下了 4.8GB，请回到第 6 步确认 Rust 改动已编译）

### 之后：验证补装路径（可选，耗时 20~60 分钟）

点设置页「准备环境并下载（约 8.5GB）」，观察：
1. 进度条是**不确定动画**（正常，uv 不上报文件级进度）
2. 结束后设置页「运行环境」变「已就绪」
3. 再点一次，`Test-Path ...\site-packages\torch` 应变 `True`
4. 把「嵌入模式」切到 `real` 保存 → 到文档列表**重建索引** → 检索生效

> ⚠️ 这一路径**从未真实执行过**（开发环境无 PyPI 出网，测试全程打桩），首次必然需要人工观察。

---

## 第 10 步 · 故障速查

| 现象 | 根因 | 处理 |
|---|---|---|
| `cargo: command not found` | Rust 未装 / 未重开窗口 | 回第 2 步，**新开窗口** |
| `cargo check` 卡住不动 | crates.io 不可达 | 回第 1 步配 cargo 镜像 |
| `npm ci`/`npm install` 卡住 | npmjs 不可达 | 回第 1 步配 npm 镜像 |
| `MissingArrayIndexExpression` / ps1 中文乱码 | `.ps1` 缺 UTF-8 BOM，PS 5.1 按 GBK 解码吞掉引号 | 仓库内 5 个 ps1 已修；自己新写的 ps1 务必存为 **UTF-8 with BOM** |
| `python.exe -c "import site"` 失败 | `python312._pth` 未放开 `import site` | 回第 3.1 步 |
| `uv sync` 报 site-packages 相关错 | 同上 | 回第 3.1 步 |
| `icon.ico not found` | 未跑 `npx tauri icon` | 回第 5 步 |
| **`resource path \`resources\python\` doesn't exist`** | ★ 未先跑 `build-resources.ps1`。Tauri 的 build script 在 `cargo check`/`build` **阶段**就校验 `bundle.resources` 的目录是否存在 | 先跑第 6 步，再 cargo |
| `ModuleNotFoundError: papermate.models` | 打包源缺 `src/papermate/models/` | 检查 `.gitignore` 是否用了裸 `models/`（应锚定为 `/models/`） |
| `resource path not found`（Windows 上找不到 bundled python） | 用 `cargo build`/`tauri dev` 而非 `tauri build` 跑 | 正式出包用 `tauri build`；`resources` 在 target/release 下才完整 |
| 窗口白屏 | 后端未就绪 | 看 `%LOCALAPPDATA%\PaperMate\data\logs\app.log`；`loading.html` 会自动跳转 |
| 启用 real 后检索失败 | 依赖或权重未就绪 | 看设置页「运行环境」与模型安装状态；`HF_HOME` 在 `%LOCALAPPDATA%\PaperMate\huggingface` |
| **`OSError: Readme file does not exist: README.md`** | ★ `pyproject.toml` 声明了 `readme`，但打包脚本**漏拷**进后端副本。首启 `uv sync` 要**安装项目自身**（editable），hatchling 必读该文件 | 重跑第 6 步（脚本已修，会拷 `README.md`/`LICENSE`）。★ 开发机**永远测不出** —— 项目根一直有 README.md，只有**导出副本**才缺 |
| `uv sync failed: exit code: 2` ＋ `拒绝访问 (os error 5)` | 用户**全局** uv 缓存里有永久损坏条目（ACL/属主异常，`uv cache clean` 也删不掉） | 已改 App 私有 `UV_CACHE_DIR=<local_data>\uv-cache`。应急（**无需重编译**）：`$env:UV_CACHE_DIR="$env:LOCALAPPDATA\PaperMate\uv-cache"` 后跑现有 exe |
| **窗口永远停在「正在启动」；后端日志 `No module named papermate`** | ★★★ venv 是**半成品**（`python.exe` 在、`site-packages` 空），而旧判据只看 `python.exe` 是否存在 ⇒ **永远跳过安装、且零报错** | 重编译（已修：能力探测 + `deps-ready` 指纹标记）。应急一键重置：删 `%LOCALAPPDATA%\PaperMate\deps-ready` **和** `venv` 目录后重启 |
| 点过「准备运行环境」后 `torch` 又不见了 | ★★ 首启那段 `uv sync` 重跑了一次，**精确模式把"多余"包剪掉**（只在指纹变化时发生：换 lock / 换安装目录） | 已修：首启带 `--inexact`（绝不删除环境里已有的包）。诊断：`uv sync --dry-run` 若打印 ` - pkg==ver` 就是**会删** |
| **换了构建方式（debug ↔ release）后"改了没生效"** | ★★ venv 固定在 `%LOCALAPPDATA%\PaperMate\venv`，**debug / release 共用同一份**；而 editable 的 `.pth` 仍指向**旧 profile** 的 `resources/backend/src`。uv 见 `papermate==0.1.0` **已满足就不再重装** ⇒ exe **静默 import 旧那份源码** | 删 `%LOCALAPPDATA%\PaperMate\venv`（最省事），或仅删 `_editable_impl_papermate.pth` + `papermate-0.1.0.dist-info`，再启动 |
| **首装跑到一半关了窗口 / 装完了却没写 `deps-ready`** | 写标记的是 App 进程；窗口一关它就没了，uv 虽继续跑完但**没人写标记** ⇒ 下次启动还会重跑一遍 `uv sync` | 重启 App 让它自己补（缓存已热，很快）；或直接删 `deps-ready` 强制重走一遍 |

---

## 附 · 首启「是否需要安装依赖」的判据（2026-10-07 重做）

```
deps-ready 标记（%LOCALAPPDATA%\PaperMate\deps-ready）内容 == 当前指纹
  且 venv\Scripts\python.exe 存在
        └─→ 跳过安装（0ms，不启动任何子进程）
否则
        └─→ uv sync --extra rag --frozen --inexact
              --no-install-package torch/FlagEmbedding/sentence-transformers
            → 退出码 0 **且** `python -c "import fastapi, sqlalchemy, papermate, lightcore"` 成功
            → 才写入 deps-ready 标记
```

- **指纹** = `FNV1a64(uv.lock 字节 ‖ 后端副本绝对路径)` ⇒ **换 lock（升级）** 或 **换安装目录** 都会自动重装。
- ★ **判据是「能力」不是「文件是否存在」** —— `uv` 在**创建环境**阶段就写好 `python.exe`，安装在其后；只看文件存在 ⇒ 首次失败后会**永久跳过安装**（详见故障速查第 3 条）。
- ★ **强制重装**：只删 `%LOCALAPPDATA%\PaperMate\deps-ready` 即可（**venv 不必删**，uv 会自己补全）。
- ★ 探针**只 import 轻量包**，绝不 import `papermate.api` —— 后者在导入期就 `create_app()`，有建库/建目录副作用。
- ★ 为什么不做"每次启动都探测"：**实测**探针要 1670ms（裸解释器启动就 827ms）⇒ 每次白等 ~0.85s，不可接受。
