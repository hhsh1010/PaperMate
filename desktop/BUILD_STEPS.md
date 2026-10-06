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

```powershell
# cargo（仅本次窗口有效；想永久生效请写 %USERPROFILE%\.cargo\config.toml）
$env:CARGO_REGISTRIES_CRATES_IO_PROTOCOL = "sparse"
$env:CARGO_REGISTRIES_CRATES_IO_INDEX = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
```
✅ 判据：无报错（`cargo check` 时才会真正验证）

---

## 第 2 步 · 装 Rust 工具链

```powershell
$env:RUSTUP_DIST_SERVER = "https://mirrors.tuna.tsinghua.edu.cn/rustup"
$env:RUSTUP_UPDATE_ROOT = "https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"
irm https://sh.rustup.rs -useb | iex
```
装完**新开一个窗口**，验证：
```powershell
cargo --version
```
✅ 判据：打印 `cargo 1.x.x`

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

```powershell
npx tauri icon src-tauri/icons/logo-source.png
```
✅ 判据：`src-tauri\icons\` 下出现这 5 个文件（与 `tauri.conf.json` 的 `bundle.icon` 一一对应）：
```
32x32.png   128x128.png   128x128@2x.png   icon.ico   icon.icns
```

> 缺任何一个 ⇒ `tauri build` 会失败。`logo-source.png` 是脚本生成的占位图，
> 正式 logo 到位后覆盖它再重跑本步即可。

---

## 第 6 步 · ★ `cargo check`（验证 `lib.rs` 的改动能编译）

```powershell
cd src-tauri
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

## 第 7 步 · 准备后端副本 + 构建前端

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-resources.ps1
```
✅ 判据：末尾输出
```
[build-resources] done. backend copy at: ...\desktop\src-tauri\resources\backend
```
（末尾关于缺 `python.exe` / `uv.exe` 的两条**告警**若出现，说明第 3/4 步没做完）

### ✅ 产物自检（务必看一眼）

```powershell
Test-Path .\src-tauri\resources\backend\pyproject.toml
Test-Path .\src-tauri\resources\backend\src\papermate\models\db.py
Test-Path .\src-tauri\resources\backend\web\dist\index.html
Test-Path .\src-tauri\resources\backend\web\dist\loading.html
```
四条都应是 `True`。
（第二条尤其重要 —— 历史上 `.gitignore` 的裸模式 `models/` 曾把整个源码包排除在 git 之外，
导致打出来的包启动即 `ModuleNotFoundError`）

---

## 第 8 步 · 出包

```powershell
npm run tauri build
```
耗时较长（首次要编 Rust 依赖 + 下载 NSIS，十几分钟起）。

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
| `ModuleNotFoundError: papermate.models` | 打包源缺 `src/papermate/models/` | 检查 `.gitignore` 是否用了裸 `models/`（应锚定为 `/models/`） |
| `resource path not found`（Windows 上找不到 bundled python） | 用 `cargo build`/`tauri dev` 而非 `tauri build` 跑 | 正式出包用 `tauri build`；`resources` 在 target/release 下才完整 |
| 窗口白屏 | 后端未就绪 | 看 `%LOCALAPPDATA%\PaperMate\data\logs\app.log`；`loading.html` 会自动跳转 |
| 启用 real 后检索失败 | 依赖或权重未就绪 | 看设置页「运行环境」与模型安装状态；`HF_HOME` 在 `%LOCALAPPDATA%\PaperMate\huggingface` |
