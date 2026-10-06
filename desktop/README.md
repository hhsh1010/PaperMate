# PaperMate 桌面客户端 · 构建与打包手册

> 面向「从零把 Windows 安装包打出来并验证首装」。所有命令在 **PowerShell 5.1** 下可用（**不要用 `&&`**）。
> 后端源码托管与前端构建分离：安装包里跑的是 `resources/backend` 这份**独立副本**。

---

## 0. 前置条件

| 工具 | 用途 | 校验命令 |
|---|---|---|
| Node.js ≥ 20 | 前端构建 + Tauri CLI | `node -v` |
| Rust 工具链（stable） | 编译 Tauri 壳 | `cargo --version` |
| WebView2 Runtime | 渲染前端（Win11 自带） | 无需操作 |
| NSIS | Tauri 打包时**自动下载**，无需预装 | — |

```powershell
# Rust 未装时（国内建议先换镜像）
$env:RUSTUP_DIST_SERVER = "https://mirrors.tuna.tsinghua.edu.cn/rustup"
$env:RUSTUP_UPDATE_ROOT = "https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"
irm https://sh.rustup.rs -useb | iex
cargo --version
```

---

## 1. 三样必须放进仓库的资源

放在 `desktop/src-tauri/resources/`（该目录**不入 git**，是本地准备物）。

### 1.1 `resources/python/` —— Portable Python 3.12（embed 版）

1. 下载 `python-3.12.x-embed-amd64.zip`（Windows x64）：
   <https://www.python.org/downloads/windows/> → 选版本 → *Windows embeddable package (64-bit)*
2. 解压，把**全部内容**放进 `desktop/src-tauri/resources/python/`，必须得到 `python.exe`。
3. ★★ **必须改 `python312._pth`**（这是最容易漏的一步）：

   用记事本打开 `resources/python/python312._pth`，把 `#import site` **前面的 `#` 去掉**：

   ```
   python312.zip
   .
   #import site          ← 改成：import site
   ```

   **为什么**：embed 版默认用隔离模式，**禁用 `site` 机制** ⇒ `site-packages` 不可写 ⇒
   `uv sync` / `venv` 装不进任何包，报错通常很含糊（`ModuleNotFoundError: site` 或安装后 import 不到）。

4. 校验（**必须做**）：

   ```powershell
   cd desktop\src-tauri\resources\python
   .\python.exe -c "import sys, site; print(sys.version); print('site ok:', site.getsitepackages() is not None)"
   ```
   看到 `site ok: True` 才算通过。

### 1.2 `resources/python/uv.exe` —— uv 单文件

1. 下载：`https://github.com/astral-sh/uv/releases` → `uv-x86_64-pc-windows-msvc.zip`
2. 把 `uv.exe`（以及同目录的 `uvx.exe`）放进 `resources/python/`
3. 校验：

   ```powershell
   .\resources\python\uv.exe --version
   ```

> `build-resources.ps1` 在缺这两个文件时**只告警不中止** ⇒ 一定要自己跑上面两条校验，
> 否则会等到 `tauri build` 之后才发现。

### 1.3 `icons/` —— 应用图标

**已完成**：本目录已有 `logo-source.png`（1024×1024，占位源图）。生成全套：

```powershell
cd desktop
npm install
npx tauri icon src-tauri/icons/logo-source.png
```

产出 `32x32.png` / `128x128.png` / `128x128@2x.png` / `icon.ico` / `icon.icns`，
与 `tauri.conf.json` 的 `bundle.icon` 一一对应，**缺一个就构建失败**。
详见 `desktop/src-tauri/icons/README.txt`。

---

## 2. 准备后端副本 + 构建

```powershell
cd desktop
powershell -ExecutionPolicy Bypass -File scripts\build-resources.ps1
```

做四件事（脚本内有前置自检，路径错会**第一步就报错**）：

1. `robocopy` 拷贝 `src/` 与 `web/`（排除 `node_modules` 与空 `tmp*` 目录）
2. 拷贝 `pyproject.toml` / `uv.lock` / `.env.example`
3. `web/dist` 不存在时自动 `npm ci && npm run build`
4. 把 `loading.html` 注入 `web/dist/`（启动等待页，同源轮询 `/healthz` 后跳 `/`）

> ⚠️ **国内网络**：`registry.npmjs.org` 实测不可达 ⇒ 先设镜像，否则第 3 步会卡死：
> ```powershell
> npm config set registry https://registry.npmmirror.com
> ```

产物位置：`desktop/src-tauri/resources/backend/`（约 9MB，不含 `.env` 与 `data/`）。

---

## 3. 打包

```powershell
cd desktop
npm run tauri build
```

- 产物：`desktop/src-tauri/target/release/bundle/nsis/*.exe`（安装包）
- 中间产物：`target/release/papermate-desktop.exe`（免安装绿色版，**双击即可跑**，适合先快速验证）

> **代码签名**（可选，去掉 SmartScreen 警告）：`tauri.conf.json` 已配
> `signCommand = scripts/codesign.ps1`。把 `.pfx` 路径与密码放进环境变量
> `CODESIGN_PFX` / `CODESIGN_PASSWORD`，脚本会自动签名；未配置则跳过（不影响出包）。

---

## 4. 首装验证（关键：别在开发机直接测）

安装包会把数据写进 `%LOCALAPPDATA%\PaperMate\`。要在**开发机**上验证首装体验，
先清掉该目录（等价于"全新机器"）：

```powershell
taskkill /IM papermate-desktop.exe /F     # 若正在运行
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\PaperMate"
```

然后跑绿色版 `target\release\papermate-desktop.exe`，首启会：

1. 展开捆绑的 Portable Python，用捆绑的 `uv.exe` 执行 `uv sync --extra rag --frozen`（**联网、数分钟级**）
2. 写 `%LOCALAPPDATA%\PaperMate\data\provider_settings.json`（`default_model=fake`）
3. 写后端 `.env`（`PAPERMATE_DATA_DIR` / `HF_HOME` / `HF_ENDPOINT` 镜像）
4. 拉起后端，打开 `http://127.0.0.1:8000/loading.html`，就绪后跳 `/`

**首装应有的表现**（与"默认 fake"原则一致）：

| 检查 | 期望 |
|---|---|
| `GET /api/v1/healthz` | `rag_embedder=fake`、`default_model=fake` |
| 设置页「语义检索」卡片 | 两个模型均**未安装**，可点「下载模型」 |
| 设置页 API Key 区 | 厂商均**未配置** |
| 上传 PDF | 正常解析（`pymupdf` 在 `rag` extra 里，已随首启安装） |

---

## 5. 不打安装包也能验后端（开发机最快路径）

```powershell
# 用独立数据目录 + 空模型缓存 + 清空 key，完全不碰开发数据
powershell -ExecutionPolicy Bypass -File scripts\firstuse-sim.ps1
# 判据见该脚本头部注释
```

---

## 6. 常见故障速查

| 现象 | 根因 | 处理 |
|---|---|---|
| `MissingArrayIndexExpression` / 脚本中文乱码 | `.ps1` 缺 UTF-8 BOM，PS 5.1 按 GBK 解码吞掉引号 | 用 `utf-8-sig` 重存（仓库内 5 个 ps1 已处理） |
| `uv sync` 装不上包 / `import site` 报错 | Portable Python 的 `python312._pth` 未放开 `import site` | 见 1.1 第 3 步 |
| `npm ci` 卡住无输出 | `registry.npmjs.org` 不可达 | 设 `npmmirror` 镜像 |
| `icon.ico not found` 等图标错误 | 未跑 `npx tauri icon` | 见 1.3 |
| `ModuleNotFoundError: No module named 'papermate.models'` | 打包源缺 `src/papermate/models/` | 检查 `.gitignore` 是否用裸 `models/`（应锚定为 `/models/`） |
| 窗口白屏 | 后端未就绪 | 看 `%LOCALAPPDATA%\PaperMate\data\logs\app.log`；`loading.html` 会自动跳转 |
| 真实语义启用后加载失败 | 模型装在 `HF_HOME`（`%LOCALAPPDATA%\PaperMate\huggingface`） | 确认该目录存在；探测口径见 `lightcore/rag/hf_local.py:default_hub_cache_dir` |
