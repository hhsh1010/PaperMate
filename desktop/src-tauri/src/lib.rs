use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;

use tauri::Manager;
use tauri_plugin_dialog::DialogExt;

// 与主进程绑定的 sidecar 子进程句柄（退出时清理进程树）
static BACKEND: Mutex<Option<Child>> = Mutex::new(None);

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8000;
const APP_DATA: &str = "PaperMate";

/// **App 级引导配置**（存"数据根目录"）。
///
/// ★★ 为什么它必须放在**固定位置**（`<local_data>/app-settings.json`），而不能放进数据目录：
///   数据目录本身是由这个配置决定的 —— 放进去就成了"要知道位置才能读到位置"的鸡生蛋。
const APP_SETTINGS: &str = "app-settings.json";

/// 首启选择数据位置时要求的**最小磁盘余量（GB）**。
///
/// 依据（实测）：权重 ≈4.5GB + 轻量依赖 ≈0.5GB + 装 torch 后 venv ≈4GB + uv-cache ≈1.5GB，
/// 加上临时空间与安全余量，取整 8GB 起步（避免用户选完才发现装不下）。
const MIN_FREE_GB: f64 = 8.0;

/// 等后端**真正开始监听**的最长时间。
///
/// ★ 只在"依赖已装好、后端进程已 spawn"之后才计时，所以不需要迁就首次安装的几十分钟；
///   这个超时针对的是"进程起了但端口一直不通"的异常情况。
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// 首启**不安装**的重依赖（**发行包名**，uv 的 `--no-install-package` 按发行名匹配）。
///
/// ★ 唯一真源在 Python 侧 `papermate/services/rag_deps.py::HEAVY_DISTRIBUTIONS`；
///   两处必须**逐字一致** —— 不一致会导致"首启排除了、启用时没补装"（用户点了没反应）
///   或反之（首启又变成 4.8GB）。`tests/test_desktop_heavy_deps_contract.py` 静态校验这一点。
/// ★ 注意 `sentence-transformers` 是**连字符**（发行名）；其 Python 导入名是下划线形式。
const HEAVY_DEPS: [&str; 3] = ["torch", "FlagEmbedding", "sentence-transformers"];

/// 「依赖已装好」标记文件（与 `venv` / `uv-cache` 并列在 App 数据目录下）。
///
/// 内容 = `deps_fingerprint()` 的结果。**只在「uv sync 退出码 0 且能力探测通过」后才写入**
/// ⇒ 标记为真**蕴含**环境可用（这是后面快速通道敢信任它的唯一理由）。
const DEPS_SENTINEL: &str = "deps-ready";

/// FNV-1a 64 位哈希（纯 std，无需引入 sha2 等 crate）。
///
/// 仅用于本机"变没变"的判定，**不是安全用途**；碰撞概率对本用途可忽略。
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// 当前"应当装出来的环境"的指纹。
///
/// 组成 = `uv.lock` 内容 + **后端副本所在绝对路径**，任一变化都意味着 venv 需要重装：
///   - `uv.lock` 变 ⇒ 依赖集合变了（App 升级到新版本）；
///   - 路径变       ⇒ editable 安装写进 venv 的 `.pth` **指向旧目录**
///                    ⇒ `import papermate` 直接失败（装了新版本却白屏）。
///
/// ★ 版本标签 `v1` 参与比较：将来若要改指纹口径（比如加入 Python 版本），
///   把标签升到 `v2` 即可让所有旧标记**自动失效**，无需额外迁移逻辑。
fn deps_fingerprint(backend_dir: &std::path::Path) -> String {
    let lock = fs::read(backend_dir.join("uv.lock")).unwrap_or_default();
    let path = backend_dir.to_string_lossy().to_lowercase();
    let mut buf = Vec::with_capacity(8 + path.len());
    buf.extend_from_slice(&fnv1a64(&lock).to_le_bytes());
    buf.extend_from_slice(path.as_bytes());
    format!("papermate-deps-v1 {:016x}", fnv1a64(&buf))
}

/// 能力探测：这个 venv **真的**跑得起来后端吗？
///
/// ★★ 2026-10-06 P0（本次真机首装暴露）：判据**绝不能**是"`venv/Scripts/python.exe`
///   是否存在"。`uv` 在**创建环境**阶段就把 `python.exe` 写好了，而**安装包在其后**；
///   首启一旦中途失败（当时是 pyproject 声明的 `README.md` 没被拷进后端副本，
///   `hatchling` 构建本地项目失败），venv 里就只剩一个 `site-packages/_virtualenv.pth`
///   的**空壳**，而 `python.exe` 是在的。于是旧判据 `!venv_python.exists()` 被满足
///   ⇒ **以后每次启动都跳过安装**，直接拿空环境跑 `-m papermate.api`
///   ⇒ `No module named papermate` ⇒ 后端永不启动、界面永久停在"正在启动"，
///   而且**一句错误提示都没有**（用户只会反复重启，越重启越确信"装坏了"）。
///   ⇒ 判据必须是**能力**，不是文件存在性（同源教条："能力存在"≠"能力启用"）。
///
/// ★ 只 import **轻量标记包**，**绝不** import `papermate.api` ——
///   后者在导入期就执行 `create_app()`，会带来建库/建目录等副作用，不适合当探针。
///   实测两个包的 `__init__.py` 都是空文件 ⇒ 零副作用。
fn probe_backend_imports(venv_python: &std::path::Path, cwd: &std::path::Path) -> bool {
    let out = Command::new(venv_python)
        .current_dir(cwd)
        .args(["-c", "import fastapi, sqlalchemy, papermate, lightcore"])
        .output();
    match out {
        Ok(o) if o.status.success() => true,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!(
                "[desktop] venv capability probe FAILED (rc={:?}): {}",
                o.status.code(),
                stderr.trim().lines().last().unwrap_or("").trim()
            );
            false
        }
        Err(e) => {
            eprintln!("[desktop] venv capability probe could not run: {}", e);
            false
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // ★ Tauri 2 的 API 形状（已按 tauri-2.12.1 源码逐一核对，不是照记忆写的）：
    //   · `Builder::run(self, context)` —— **只接受 context，没有回调参数**
    //     （app.rs:2616）
    //   · `App::run<F: FnMut(&AppHandle<R>, RunEvent) + 'static>(self, callback: F)`
    //     —— **处理 RunEvent 的两参回调在这**（app.rs:1384）
    //   · `App::handle(&self) -> &AppHandle<R>`（app.rs:1279）
    //   · setup 闭包收到的 `app` 是 `&mut App`（不是 `&AppHandle`）
    //   ⇒ 所以流程是 build → 拿到 App → 再 run(callback)。
    let built = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        // ★ 原生文件夹选择器（首启让用户选数据位置）——`ask_data_root()` 用 `app.dialog()`。
        //   ⚠️ **必须注册**：只加 Cargo 依赖是不够的；`DialogExt::dialog()` 依赖插件被 init
        //   注册进来的状态。漏了这一步**编译期完全看不出来**，只在运行时炸。
        //   `tests/test_tauri_config.py` 有一条静态契约钉死它。
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // 引导逻辑只需要 `&AppHandle` ⇒ 用 `app.handle()` 取句柄。
            // （直接传 `app` 会报 expected `&AppHandle`, found `&mut App`）
            //
            // ★★ 2026-10-07：**必须放到后台线程**，两个理由：
            //   ① 首启要弹原生**阻塞式**文件夹选择器，而插件文档明确写着
            //      "This is a blocking operation, and should *NOT* be used when running
            //       on the main thread."（`tauri-plugin-dialog/src/lib.rs:744`）—— 主线程调用会死锁；
            //   ② `uv sync` 首次要跑几分钟，阻塞 setup 会把事件循环占住（窗口"无响应"）。
            let handle = app.handle().clone();
            std::thread::spawn(move || bootstrap_and_spawn(&handle));
            Ok(())
        })
        .build(tauri::generate_context!());

    let app = match built {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[desktop] failed to build Tauri app: {}", e);
            return;
        }
    };

    app.run(|_app_handle, event| {
        // 窗口关闭 → 杀掉 Python sidecar，避免 8000 端口残留
        if let tauri::RunEvent::ExitRequested { .. } = event {
            if let Some(mut child) = BACKEND.lock().unwrap().take() {
                let _ = child.kill();
            }
        }
    });
}

/// 把窗口里的等待页切到**失败态**，把原因直接显示给用户。
///
/// ★ 为什么必须做（2026-10-07）：首启失败时（`uv sync` 报错、探针不过、进程起不来），
///   错误此前只打在 stdout/stderr —— 而用户多半是**双击 exe 启动、根本没有控制台**，
///   于是只能看到一个永远转圈的窗口，完全不知道发生了什么，只能反复重启。
fn show_startup_error(app: &tauri::AppHandle, msg: &str) {
    let Some(win) = app.get_webview_window("main") else {
        eprintln!("[desktop] no window to show error in: {}", msg);
        return;
    };
    // ★ 用 serde_json 生成**合法的 JS 字符串字面量**（自动转义引号/反斜杠/换行）。
    //   直接把错误文本拼进 JS 里，一旦文本含引号就会造成语法错误 —— 那时连错误都看不到。
    let lit = serde_json::to_string(msg).unwrap_or_else(|_| "\"(启动失败)\"".to_string());
    let js = format!(
        "(function(){{var h=document.getElementById('hint');\
if(h){{h.style.color='#ff8f8f';h.textContent={lit};}}\
else{{document.body.innerHTML='<pre style=\"color:#ff8f8f;padding:24px;white-space:pre-wrap\">'+{lit}+'</pre>';}}}})();"
    );
    if let Err(e) = win.eval(js) {
        eprintln!("[desktop] could not show error in window: {}", e);
    }
}

/// 轮询等后端开始监听，然后把窗口从**本地等待页**导航到真实 UI。
///
/// ★★ 为什么"就绪判定"由 Rust 做，而不是让等待页自己 `fetch`（2026-10-07）：
///   窗口的 origin 是 `tauri.localhost`，与 `127.0.0.1:8000` **跨源**，而后端**没有 CORS 中间件**
///   ⇒ 页面里的普通 `fetch` 会被 CORS 拦：即使后端已就绪也永远拿不到结果，会无限重试。
///   改成 `mode:"no-cors"` 大概率可行，但那纯粹取决于**浏览器跨源策略**，
///   **无法在无 GUI 环境验证**；一旦被拦，症状是"永远转圈"，比现状更糟。
///   ⇒ 就绪信号改由**知道真相的一侧**（本进程）给出：后端是我们自己 spawn 的，端口通了就是通了。
///   页面的 `no-cors` 探测仅作**兜底**（万一 `eval` 没生效），两者不冲突。
fn navigate_when_backend_ready(app: tauri::AppHandle, port: u16) {
    let addr = std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, port));
    let deadline = std::time::Instant::now() + READY_TIMEOUT;
    loop {
        if std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(400)).is_ok()
        {
            if let Some(win) = app.get_webview_window("main") {
                // ★ 首启是 960×640 紧凑窗；导航到主应用前恢复成常规尺寸并居中
                let _ = win.set_size(tauri::LogicalSize::new(1280u32, 860u32));
                let _ = win.center();
                let _ = win.eval(format!("location.href = 'http://127.0.0.1:{port}/';"));
            }
            return;
        }
        if std::time::Instant::now() >= deadline {
            show_startup_error(
                &app,
                "本地服务启动超时（3 分钟内未就绪）。\n请检查控制台输出与日志，或重启应用。",
            );
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 数据根目录：配置读写 / 校验 / 首启选择（2026-10-07）
//
// 布局：`<root>/{data, huggingface, venv, uv-cache}` —— **一个根目录管四件事**。
//   · data         → PAPERMATE_DATA_DIR（用户数据：DB/索引/上传/备份/日志/设置/记忆）
//   · huggingface  → HF_HOME（模型权重 ≈4.5GB）
//   · venv         → UV_PROJECT_ENVIRONMENT + PAPERMATE_VENV_DIR（装 torch 后 ≈4GB）
//   · uv-cache     → UV_CACHE_DIR + PAPERMATE_UV_CACHE_DIR
// ★ 默认 root = `<local_data>`（= `%LOCALAPPDATA%\PaperMate`）⇒ 四条路径与改造前**逐字节相同**。
// ★ 为什么 venv 也在 root 下：用户选它多半就是为了"别塞满 C 盘"，而 venv 是第二大件。
//   代价：换位置时 venv 必须**重建**（它里面全是绝对路径：.pth 指向源码、pyvenv.cfg
//   指向便携 Python、脚本 shebang）—— 这条已写进设计文档，UI 上也要讲清楚。
// ───────────────────────────────────────────────────────────────────────────

/// 只用两个 kernel32 函数：磁盘余量、盘类型。
///
/// ★ 为什么不引 crate：各十几行 FFI 就够，而本项目刻意保持依赖精简。
/// ⚠️ 形状**已用独立可执行文件真跑过**并与 `df -h` 对照（C: 86.9G / D: 146.3G 一致）——
///   "手写 Rust 无编译器时 API 形状几乎必错"，所以这里不靠记忆、靠实测。
#[cfg(windows)]
mod winfs {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            dir: *const u16,
            free_to_caller: *mut u64,
            total: *mut u64,
            total_free: *mut u64,
        ) -> i32;
        fn GetDriveTypeW(root: *const u16) -> u32;
    }

    fn wide(s: &str) -> Vec<u16> {
        OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
    }

    /// 调用方可用字节数；盘不存在/未就绪时返回 None。
    pub fn free_bytes(dir: &str) -> Option<u64> {
        let p = wide(dir);
        let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
        let ok = unsafe { GetDiskFreeSpaceExW(p.as_ptr(), &mut avail, &mut total, &mut free) };
        if ok == 0 {
            None
        } else {
            Some(avail)
        }
    }

    /// 盘类型：2=可移动 3=固定 4=网络 5=光驱 …。**不产生 I/O**（比 `Path::exists()` 安全：
    /// 后者在断开的网络盘上可能卡住数秒）。
    pub fn drive_type(root: &str) -> u32 {
        let p = wide(root);
        unsafe { GetDriveTypeW(p.as_ptr()) }
    }
}

#[cfg(not(windows))]
mod winfs {
    pub fn free_bytes(_dir: &str) -> Option<u64> {
        None
    }
    pub fn drive_type(_root: &str) -> u32 {
        0
    }
}

/// `app-settings.json` 的结构。带 `schema` 以便将来改语义时能迁移。
#[derive(serde::Deserialize, serde::Serialize)]
struct AppSettings {
    schema: u32,
    #[serde(default)]
    data_root: Option<String>,
    /// ★ 设置页「改存储位置」写入的**待生效目标**（需重启后由启动器执行）。
    ///   为什么不在运行中的后端里直接搬：那时 SQLite 正被打开、venv 正被加载
    ///   ⇒ 拷贝到一半失败 = 数据不一致。改为"重启前、后端启动前"执行，全程无锁。
    #[serde(default)]
    pending_root: Option<String>,
    /// 待生效目标是否**搬迁**既有数据（false = 在新位置以空目录启动）。
    #[serde(default)]
    migrate: Option<bool>,
}

impl AppSettings {
    fn read(local_data: &Path) -> Option<Self> {
        let p = local_data.join(APP_SETTINGS);
        if !p.is_file() {
            return None;
        }
        match fs::read_to_string(&p)
            .ok()
            .and_then(|s| serde_json::from_str::<Self>(&s).ok())
        {
            Some(s) => Some(s),
            None => {
                // ★ 配置坏了**绝不能**导致起不来：回退默认 + 日志。
                eprintln!("[desktop] app-settings.json 无法解析，回退默认数据位置");
                None
            }
        }
    }

    /// 只带 `data_root`（清空待迁移请求）—— 首启选完、以及迁移成功/失败后都走它。
    fn only_root(root: &Path) -> Self {
        AppSettings {
            schema: 1,
            data_root: Some(root.to_string_lossy().to_string()),
            pending_root: None,
            migrate: None,
        }
    }

    fn write(&self, local_data: &Path) -> std::io::Result<()> {
        let body = serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| r#"{"schema":1}"#.to_string());
        let dst = local_data.join(APP_SETTINGS);
        let tmp = local_data.join(format!("{APP_SETTINGS}.tmp"));
        fs::write(&tmp, body)?;
        // Windows 上目标已存在时 rename 会失败 ⇒ 先删再改名。
        if fs::rename(&tmp, &dst).is_err() {
            let _ = fs::remove_file(&dst);
            fs::rename(&tmp, &dst)?;
        }
        Ok(())
    }
}

/// 路径比较用归一化：去掉 Windows 的 `\\?\` verbatim 前缀 + 转小写。
///
/// ★ 必须做：Tauri 的 `resource_dir()` 返回的就是**带 `\\?\` 前缀**的长路径
///   （本项目 10-07 已因此踩过一次），直接比较会"看起来该相等却永远不相等"。
fn norm_for_cmp(p: &Path) -> String {
    let s = p.to_string_lossy().to_lowercase();
    s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
}

/// 解析数据根目录 → `(root, 是否首次运行)`。
///
/// ★ **首次的判据 = `app-settings.json` 是否存在**（该文件只在"位置已确定"之后写，
///   包含"用户取消 → 采用默认"的情况）⇒ 卸载重装后仍是"非首次"（数据还在，正确）。
/// ★ 配置损坏/读不出 ⇒ `AppSettings::read` 返回 None ⇒ 走"首次"分支并回退默认，
///   **绝不 panic**（起不来是最坏结果）。
fn resolve_data_root(local_data: &Path) -> (PathBuf, bool) {
    match AppSettings::read(local_data) {
        None => (local_data.to_path_buf(), true),
        Some(s) => {
            let root = s
                .data_root
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| local_data.to_path_buf());
            (root, false)
        }
    }
}

/// 写「首次启动已确定位置」。
fn write_app_settings(local_data: &Path, root: &Path) -> std::io::Result<()> {
    AppSettings::only_root(root).write(local_data)
}

/// 只更新 `data_root` 并清空待迁移请求。
fn write_root_only(local_data: &Path, root: &Path) -> std::io::Result<()> {
    AppSettings::only_root(root).write(local_data)
}

/// 递归统计文件数（迁移后的**逐项校验**用）。
fn count_files(dir: &Path) -> usize {
    let mut n = 0usize;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                n += 1;
            }
        }
    }
    n
}

fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// 搬走"可搬"的目录：`data/` · `huggingface/` · `uv-cache/`。
///
/// ★ `venv/` **不搬**：它内部全是绝对路径（`.pth` 指向源码、`pyvenv.cfg` 指向便携 Python、
///   脚本 shebang）⇒ 复制过去会坏。正确做法是在新位置**重建**——而因为 `uv-cache` 跟着搬了，
///   `uv sync` 会直接硬链接命中，重建很快。
/// ★ 目标已存在同名目录时**跳过**（绝不覆盖/合并两份数据）。
/// ★★ 每一项搬完都**核对文件数**，不一致即报错回滚（不写配置、继续用旧位置）。
fn migrate_data_dirs(src_root: &Path, dst_root: &Path) -> std::io::Result<()> {
    for name in ["data", "huggingface", "uv-cache"] {
        let from = src_root.join(name);
        if !from.exists() {
            eprintln!("[desktop] migrate: skip {name} (源不存在)");
            continue;
        }
        let to = dst_root.join(name);
        if to.exists() {
            eprintln!("[desktop] migrate: skip {name} (目标已存在：{})", to.display());
            continue;
        }
        copy_dir_all(&from, &to)?;
        let (a, b) = (count_files(&from), count_files(&to));
        if a != b {
            return Err(std::io::Error::other(format!(
                "{name} 搬迁后校验不一致：源 {a} 个文件 / 目标 {b} 个"
            )));
        }
        eprintln!("[desktop] migrate: {name} ok ({a} files) -> {}", to.display());
    }
    Ok(())
}

/// ★ 执行设置页写下的**待迁移请求**，返回最终生效的数据根目录。
///
/// ★★ 为什么必须在**后端启动之前**做：那时 SQLite 未打开、venv 未被加载
///    ⇒ 不存在"拷到一半"导致数据不一致的可能。
/// ★ 任一环节失败 ⇒ **继续用旧位置**（用户的数据就在那儿），并把请求清掉避免反复重试。
fn apply_pending_migration(local_data: &Path, current: &Path, install_dir: &Path) -> PathBuf {
    let Some(s) = AppSettings::read(local_data) else {
        return current.to_path_buf();
    };
    let Some(pending) = s
        .pending_root
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
    else {
        return current.to_path_buf();
    };
    let migrate = s.migrate.unwrap_or(false);

    if norm_for_cmp(&pending) == norm_for_cmp(current) {
        let _ = write_root_only(local_data, current);
        return current.to_path_buf();
    }
    eprintln!(
        "[desktop] applying pending data-root change: {} -> {} (migrate={migrate})",
        current.display(),
        pending.display()
    );
    if let Err(msg) = validate_data_root(&pending, install_dir) {
        eprintln!("[desktop] pending root rejected: {msg}");
        let _ = write_root_only(local_data, current);
        return current.to_path_buf();
    }
    // ⚠️ 不用 `if cond && let Err(e) = ...`（let-chains 仅 edition 2024 稳定，本项目是 2021）
    if migrate {
        if let Err(e) = migrate_data_dirs(current, &pending) {
            eprintln!("[desktop] migration failed: {e} -> keep using {}", current.display());
            let _ = write_root_only(local_data, current);
            return current.to_path_buf();
        }
    }
    match write_root_only(local_data, &pending) {
        Ok(()) => pending,
        Err(e) => {
            eprintln!("[desktop] cannot persist new data root: {e}");
            current.to_path_buf()
        }
    }
}

/// 校验用户选的位置能不能当数据根目录。`Err` 里的文案**直接展示给用户**。
///
/// 每一步都对应一类真实故障：相对路径（CWD 分叉）、网络盘（SQLite 锁不可靠）、
/// 安装目录内（升级被覆盖）、不可写（Windows 上 `os.access` 是**假信号** ⇒ 真写真删）、
/// 余量不足（权重 4.5GB + venv 4GB）。
fn validate_data_root(root: &Path, install_dir: &Path) -> Result<(), String> {
    if !root.is_absolute() {
        return Err("请选择完整的绝对路径（例如 D:\\PaperMate）。".to_string());
    }
    let raw = root.to_string_lossy().to_string();
    if raw.starts_with(r"\\") || raw.starts_with("//") {
        return Err(
            "不能使用网络位置（\\\\服务器\\共享）：SQLite 在网络盘上的锁语义不可靠。\n请改选本机磁盘。"
                .to_string(),
        );
    }
    // 映射出来的网络盘（看着像 Z:\）—— 用盘类型判，不产生 I/O。
    if let Some(letter) = raw.chars().next() {
        if winfs::drive_type(&format!("{letter}:\\")) == 4 {
            return Err(
                "该盘是**网络映射盘**（REMOTE）：SQLite 在其上锁不可靠。\n请改选本机磁盘。".to_string(),
            );
        }
    }
    if norm_for_cmp(root).starts_with(&norm_for_cmp(install_dir)) {
        return Err("不要把数据放在应用安装目录里 —— 升级时会被覆盖。".to_string());
    }
    fs::create_dir_all(root).map_err(|e| format!("无法创建目录：{}\n{}", root.display(), e))?;
    // ★ 真写真删：Windows 上 `os.access(W_OK)` 是假信号（本项目 10-06 血泪）。
    let probe = root.join(".papermate-write-probe");
    fs::write(&probe, b"ok").map_err(|e| format!("该位置不可写：{}\n{}", root.display(), e))?;
    let _ = fs::remove_file(&probe);

    match winfs::free_bytes(&raw) {
        Some(bytes) => {
            let gb = bytes as f64 / 1024.0 / 1024.0 / 1024.0;
            if gb < MIN_FREE_GB {
                return Err(format!(
                    "该磁盘剩余 {gb:.1} GB，至少需要 {MIN_FREE_GB:.0} GB\n（模型权重约 4.5GB + 运行环境约 4GB）。请换一个位置。"
                ));
            }
        }
        None => {
            // 取不到就**不拦**（例如权限受限），但要留痕 —— 宁可放过，也不要卡死首启。
            eprintln!("[desktop] 读不到 {} 的磁盘余量，跳过余量检查", root.display());
        }
    }
    Ok(())
}

/// 列出本机可用磁盘（盘符 + 可用 GB），供首启面板显示。
fn list_drives() -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for c in b'C'..=b'Z' {
        let root = format!("{}:\\", c as char);
        let t = winfs::drive_type(&root);
        if t != 2 && t != 3 {
            continue; // 只列 可移动(2) / 固定(3)
        }
        if let Some(b) = winfs::free_bytes(&root) {
            out.push((root, b as f64 / 1024.0 / 1024.0 / 1024.0));
        }
    }
    out
}

/// 生成合法的 JS 字符串字面量 —— 路径里全是反斜杠，不转义必炸（本项目已踩过同款）。
fn js_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

/// 往窗口注入一段 JS。失败只记日志 —— 这是**展示**通道，不能因为它中断启动。
fn eval_js(app: &tauri::AppHandle, js: String) {
    if let Some(win) = app.get_webview_window("main") {
        if let Err(e) = win.eval(js) {
            eprintln!("[desktop] eval failed: {}", e);
        }
    }
}

/// 让等待页切到「首次启动」形态。
///
/// ★ 用户的操作发生在**原生文件夹选择器**里 ⇒ **不需要 JS→Rust 的 IPC**，
///   也就不需要 capability（本仓库至今没有任何 capability 文件）。
fn show_first_run_panel(app: &tauri::AppHandle, default_root: &Path, has_existing: bool) {
    let drives = list_drives()
        .into_iter()
        .map(|(p, gb)| format!("{p}（可用 {gb:.0} GB）"))
        .collect::<Vec<_>>()
        .join("    ");
    let js = format!(
        "window.pmFirstRun && window.pmFirstRun({{defaultRoot: {}, drives: {}, minFreeGb: {}, hasExisting: {}}});",
        js_str(&default_root.to_string_lossy()),
        js_str(&drives),
        MIN_FREE_GB,
        has_existing
    );
    eval_js(app, js);
    // ★ 首启面板内容紧凑（≈卡片 680px），把窗口缩到最小可尺寸，避免大片留白。
    //   960×640 == tauri.conf.json 的 minWidth/minHeight ⇒ set_size 不会被下限钳制产生漂移。
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.set_size(tauri::LogicalSize::new(960u32, 640u32));
        let _ = win.center();
    }
}

/// 默认位置上**是否已经存在真实数据**。
///
/// ★ 为什么必须查：判据只有 `app-settings.json` 存不存在 ⇒ **老用户升级后首次启动
///   也会弹一次选择框**。此时若他误选了别的位置，看到的就是"数据全没了"，
///   而其实旧数据原地未动。⇒ 必须提前告诉他"继续用默认即可，别换"。
fn default_root_has_data(default_root: &Path) -> bool {
    default_root
        .join("data")
        .join("papermate.db")
        .is_file()
}

/// 回显最终采用的根目录（含"用户取消 → 用默认"：**不静默**）。
fn show_root_choice(app: &tauri::AppHandle, root: &Path) {
    eval_js(
        app,
        format!(
            "window.pmRootChosen && window.pmRootChosen({});",
            js_str(&root.to_string_lossy())
        ),
    );
}

/// 首启：弹**原生**文件夹选择器定数据根目录，返回最终采用的根目录。
///
/// ★ 无论成功/取消/校验失败，**都会写 `app-settings.json`** —— 否则下次启动还会再问一遍。
/// ★ 最多重试 3 次：不让用户被卡死在首启（"首启必须能起来"优先级最高）。
fn ask_data_root(app: &tauri::AppHandle, local_data: &Path, install_dir: &Path) -> PathBuf {
    let has_existing = default_root_has_data(local_data);
    show_first_run_panel(app, local_data, has_existing);
    if has_existing {
        eprintln!(
            "[desktop] first run: default location already holds data -> hinting the user to keep it"
        );
    }
    for attempt in 1..=3 {
        let picked = app
            .dialog()
            .file()
            .set_title("PaperMate 首次启动 · 请选择数据存放位置")
            .set_directory(local_data)
            .blocking_pick_folder();
        let Some(fp) = picked else {
            // 取消 ⇒ 用默认位置，但**显式回显**（不静默）
            let _ = write_app_settings(local_data, local_data);
            show_root_choice(app, local_data);
            return local_data.to_path_buf();
        };
        match fp.into_path() {
            Ok(p) => match validate_data_root(&p, install_dir) {
                Ok(()) => {
                    if let Err(e) = write_app_settings(local_data, &p) {
                        eprintln!("[desktop] cannot write app-settings.json: {e}");
                    }
                    show_root_choice(app, &p);
                    return p;
                }
                Err(msg) => {
                    eprintln!("[desktop] data root rejected (attempt {attempt}): {msg}");
                    show_startup_error(
                        app,
                        &format!("{msg}\n\n请重新选择（第 {attempt}/3 次）。"),
                    );
                }
            },
            Err(e) => {
                eprintln!("[desktop] picked path unusable: {e}");
                show_startup_error(app, "所选位置无法解析，请重新选择。");
            }
        }
    }
    // 3 次都没通过 ⇒ 用默认位置继续（首启不能被卡死）
    let _ = write_app_settings(local_data, local_data);
    show_root_choice(app, local_data);
    local_data.to_path_buf()
}

/// 派生目录解析：**外部已显式设置则尊重它**，否则用 `<root>/<子目录>`。
///
/// ★ 保留这个逃生口的理由与 `HF_ENDPOINT` / `UV_CACHE_DIR` 的处理一致：
///   排查或仿真（如 `firstuse-sim.ps1` 的隔离）时可以临时把某一项指到别处，
///   而默认值仍然由"数据根目录"这一个入口决定（保持确定性）。
fn dir_or_env(env_key: &str, fallback: PathBuf) -> PathBuf {
    std::env::var(env_key)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or(fallback)
}

fn bootstrap_and_spawn(app: &tauri::AppHandle) {
    // ⚠️ Tauri 2.12 起 `resource_dir()` / `local_data_dir()` 返回 **Result<PathBuf, Error>**
    //    而不是早期的 `Option<PathBuf>`（源码 path/desktop.rs:230 / :86）。
    //    按 `Some/None` 匹配会报 expected `Result<_, Error>`, found `Option<_>` ⇒ 必须匹配 Ok/Err。
    let res_dir = match app.path().resource_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[desktop] resource_dir unavailable: {}", e);
            return;
        }
    };
    let backend_dir = res_dir.join("resources/backend");
    let python_exe = res_dir.join("resources/python/python.exe");
    let uv_exe = res_dir.join("resources/python/uv.exe");

    let local_data = match app.path().local_data_dir() {
        Ok(p) => p.join(APP_DATA),
        Err(e) => {
            eprintln!("[desktop] local_data_dir unavailable: {}", e);
            return;
        }
    };
    // ── 数据根目录：首启让用户**选**，之后从 `app-settings.json` 读 ──────────────
    //   ★ 必须在这里、即在**创建任何目录之前**：一旦先建了目录，用户的选择就失去意义。
    let (mut data_root, first_run) = resolve_data_root(&local_data);
    if first_run {
        eprintln!("[desktop] first run: asking the user where to store data ...");
        data_root = ask_data_root(app, &local_data, &res_dir);
        eprintln!("[desktop] data root = {}", data_root.display());
    } else {
        // ★ 设置页「改存储位置」在这里执行：刻意放在**后端启动之前** ——
        //   此刻 SQLite 未打开、venv 未加载 ⇒ 不存在"拷到一半"导致数据不一致的可能。
        data_root = apply_pending_migration(&local_data, &data_root, &res_dir);
    }

    // 四个派生目录。★ 默认 root == local_data ⇒ 与本改造之前**逐字节相同**。
    //   · 每一项都支持"外部显式设置则尊重它"（逃生口，见 dir_or_env）
    //   · ⚠️ 后端 `rag_deps` 需要 `PAPERMATE_VENV_DIR` / `PAPERMATE_UV_CACHE_DIR` 才能
    //     与这里**同源**（否则它会退回用 `data_dir.parent` 推导，见本文件 spawn 处的注入）
    let data_dir = dir_or_env("PAPERMATE_DATA_DIR", data_root.join("data"));
    let hf_home = dir_or_env("HF_HOME", data_root.join("huggingface"));
    let venv_dir = dir_or_env("UV_PROJECT_ENVIRONMENT", data_root.join("venv"));
    // ★ App 专属 uv 分发缓存（与 venv / huggingface 并列）。
    //   2026-10-06 实测事故：共用用户全局 uv 缓存时，首启因一个**损坏的缓存条目**失败：
    //     Failed to write to the distribution cache / failed to open file
    //     ...\uv\cache\sdists-v9\...\jieba-0.42.1-py3-none-any.whl: 拒绝访问 (os error 5)
    //   该 .whl 连 `uv cache clean` 都删不掉、`icacls` 读不到 ACL（属主/ACL 异常），
    //   而同目录其它文件可正常删除 ⇒ 全局缓存里有**永久损坏的条目**。
    //   指向 App 私有目录后，启动不再依赖用户机器上的历史残留。
    let uv_cache_dir = dir_or_env("UV_CACHE_DIR", data_root.join("uv-cache"));
    let venv_python = venv_dir.join("Scripts").join("python.exe");

    if let Err(e) = fs::create_dir_all(&data_dir) {
        eprintln!("[desktop] cannot create data_dir {:?}: {}", data_dir, e);
        show_startup_error(app, &format!("无法创建数据目录：{}\n{}", data_dir.display(), e));
        return;
    }

    // ★ 配置优先级（阶段0 实测更正）：有效默认模型由 data_dir/provider_settings.json 决定，
    //   高于 PAPERMATE_DEFAULT_MODEL 进程 env。所以此处写文件，而非只靠进程 env。
    let pset = data_dir.join("provider_settings.json");
    if !pset.exists() {
        let dm = std::env::var("PAPERMATE_DESKTOP_DEFAULT_MODEL").unwrap_or_else(|_| "fake".to_string());
        let json = serde_json::json!({ "default_model": dm, "providers": {} }).to_string();
        let _ = fs::write(&pset, json);
    }

    // 后端 .env（pydantic-settings 的 env_file=".env" 从 CWD 读取；
    // 捆绑的 backend 副本故意不含项目真实 .env，只写桌面专用覆盖项）。
    //
    // ★ 2026-10-07：改为**每次启动都重写**。原先只在"文件不存在时写"，
    //   于是用户改了数据位置之后，这个文件会一直写着**旧路径** —— 功能上不会错
    //   （进程 env 的优先级高于 .env），但排查时会被它误导（本项目已多次被"陈旧陈述"坑过）。
    //   ⚠️ 安装目录可能只读 ⇒ 写失败只记日志：**权威来源是进程 env，不是这个文件**。
    let env_file = backend_dir.join(".env");
    let content = format!(
        "PAPERMATE_HOST={host}\nPAPERMATE_PORT={port}\nPAPERMATE_DATA_DIR={dd}\nHF_HOME={hf}\nHF_ENDPOINT=https://hf-mirror.com\n",
        host = HOST,
        port = PORT,
        dd = data_dir.display(),
        hf = hf_home.display()
    );
    if let Err(e) = fs::write(&env_file, content) {
        eprintln!(
            "[desktop] cannot write {:?}: {} (env vars still take precedence)",
            env_file, e
        );
    }

    // 首次启动：用捆绑 Portable Python + uv 引导安装**轻量**依赖（联网一次）。
    // venv 放在 LocalAppData 而非安装目录（安装目录可能只读）。
    //
    // ★★ 2026-10-06：首启**刻意不装**重依赖（torch / FlagEmbedding /
    // sentence-transformers，合计约 4.0GB —— 实测 torch 单独就 3997MB，
    // 其中 CUDA 库约 3.5GB）。原因：首装默认是假嵌入（无真实语义），
    // 让用户为了跑"假检索"先下 4.8GB 与"装完即用"的产品原则冲突。
    // ⇒ 真正的真实语义检索由用户在设置页点"下载模型"触发，
    //    后端 `papermate/services/rag_deps.py::sync_heavy_deps()` 会再跑一次
    //    **不带**这些排除项的 `uv sync` 把它们补回来（同一份 uv.lock，幂等）。
    //
    // ⚠️ 排除项必须是**发行包名**（连字符形式），uv 的 --no-install-package
    //    按发行名匹配；其 Python 导入名是下划线形式（sentence_transformers）。
    // ★ HEAVY_DEPS 必须与 Python 侧 `rag_deps.HEAVY_DISTRIBUTIONS` **逐字一致** ——
    //   不一致会导致"首启排除了但启用时没补装"（或反之），且极难察觉。
    //   `tests/test_desktop_heavy_deps_contract.py` 会静态校验这一致性。
    // ★★ 2026-10-06 P0：**"是否需要安装"的判据** = 「venv 解释器在」∧「deps-ready 标记
    //    与当前指纹一致」。为什么不能只看解释器是否存在，见 probe_backend_imports 的说明。
    //    指纹同时覆盖三种真实故障：① 上次装到一半失败；② uv.lock 变了（App 升级）；
    //    ③ 安装目录迁移（editable 的 .pth 指向旧路径）。
    //    命中时**零成本**（不 spawn 子进程）—— 这是安全的，因为标记只在探测通过后才写。
    let sentinel_path = local_data.join(DEPS_SENTINEL);
    let expected_fp = deps_fingerprint(&backend_dir);
    let env_ready = venv_python.exists()
        && fs::read_to_string(&sentinel_path)
            .map(|s| s.trim() == expected_fp)
            .unwrap_or(false);

    if !env_ready {
        eprintln!(
            "[desktop] first launch: installing base dependencies via uv (~0.8GB, may take minutes)..."
        );
        eprintln!(
            "[desktop] note: PyTorch and other heavy deps are DEFERRED; enable them later in Settings > Semantic Retrieval."
        );
        let mut uv_args: Vec<String> = vec![
            "sync".to_string(),
            "--extra".to_string(),
            "rag".to_string(),
            "--frozen".to_string(),
            // ★★ `--inexact` = **绝不删除环境里已有的"多余"包**，这是本段的安全阀。
            //    为什么必须有：本段用 `--no-install-package` 排除了 torch 等重依赖，
            //    而 `uv sync` 默认是**精确模式**（会剪掉不在期望集合里的包）。
            //    一旦用户点过「准备运行环境」补装好 4.0GB 的 torch/CUDA，
            //    此后本段**只要再跑到一次**（例如 App 升级换了 uv.lock 指纹），
            //    精确模式就会把刚下好的 4GB **剪掉** ⇒ 真实语义检索静默失效，
            //    而用户只看到"又变回假检索了"，极难定位。
            //    `uv sync --help` 原文（已核对包内 uv 0.12.23）：
            //      `--inexact  Do not remove extraneous packages present in the environment`
            "--inexact".to_string(),
            // ★ 强制使用捆绑的 Portable Python，避免 uv 联网另下一份 Python
            "--python".to_string(),
            python_exe.to_string_lossy().to_string(),
        ];
        for pkg in HEAVY_DEPS.iter() {
            uv_args.push("--no-install-package".to_string());
            uv_args.push(pkg.to_string());
        }
        match Command::new(&uv_exe)
            .current_dir(&backend_dir)
            .env("UV_PROJECT_ENVIRONMENT", &venv_dir)
            .env("HF_ENDPOINT", "https://hf-mirror.com")
            // ★ 单个 wheel（faiss / pymupdf）较大，uv 默认 30s 超时容易在慢网下失败；
            //   首启失败 = 应用起不来 ⇒ 这里放宽（与后端 sync_heavy_deps 保持一致）。
            .env("UV_HTTP_TIMEOUT", "600")
            .env("UV_HTTP_LOW_SPEED_LIMIT", "10")
            // ★ 避开用户全局 uv 缓存（可能含损坏条目，见上面 uv_cache_dir 的说明）
            .env("UV_CACHE_DIR", &uv_cache_dir)
            .args(uv_args)
            .status()
        {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("[desktop] uv sync failed: {}", s);
                eprintln!(
                    "[desktop] (uv's own output above has the cause; the app will retry on next launch)"
                );
                show_startup_error(
                    app,
                    &format!(
                        "依赖安装失败（uv sync 退出码 {:?}）。\n\n\
                         下次启动会自动重试，无需手动清理。\n\
                         若反复失败，请把控制台输出或 {} 一并反馈。",
                        s.code(),
                        data_dir.join("logs").display()
                    ),
                );
                return;
            }
            Err(e) => {
                eprintln!("[desktop] cannot run uv.exe: {}", e);
                show_startup_error(app, &format!("无法运行捆绑的 uv.exe：{e}"));
                return;
            }
        }

        // ★ 装完**必须验证能力**再落标记：`uv` 退出码 0 **不等于** `import` 得到
        //   （editable 的 .pth 可能指向已失效路径、二进制 wheel 可能与本机不兼容）。
        //   标记一旦写入，后续启动就会**跳过安装** ⇒ 必须保证"标记为真 ⟹ 环境真的可用"。
        if !probe_backend_imports(&venv_python, &backend_dir) {
            eprintln!(
                "[desktop] uv sync reported success but backend imports still fail; NOT writing the deps marker so the next launch will retry."
            );
            show_startup_error(
                app,
                "本地运行环境安装不完整（依赖装上了但后端仍无法导入）。\n\n\
                 下次启动会自动重试。\n\
                 若反复失败，可删除下面的目录后重启应用：\n\
                 %LOCALAPPDATA%\\PaperMate\\venv",
            );
            return;
        }
        // 落标记失败**不影响本次运行**（环境已可用）：最坏后果只是下次启动多跑一次
        // 幂等的 `uv sync`。所以这里刻意不 return。
        if let Err(e) = fs::write(&sentinel_path, format!("{expected_fp}\n")) {
            eprintln!(
                "[desktop] could not write deps marker {:?}: {}",
                sentinel_path, e
            );
        }
    }

    // 启动后端 sidecar：一个进程同时托管 /api/v1/* 与 / (SPA, web/dist)
    //
    // ★ 顺序上有个关键点（2026-10-07 从 tauri 源码确认）：**窗口早在本函数之前就已按
    //   `tauri.conf.json` 的 `windows[0].url` 创建好了**（`app.rs::setup` 里先建窗口、
    //   再跑用户的 setup 闭包）。所以窗口此刻显示的是**包内的本地等待页**（`loading.html`），
    //   需要我们在后端就绪后**主动把它导航**到真实 UI —— 见 `navigate_when_backend_ready`。
    match Command::new(&venv_python)
        .current_dir(&backend_dir)
        .args(["-m", "papermate.api"])
        .env("PAPERMATE_HOST", HOST)
        .env("PAPERMATE_PORT", PORT.to_string())
            .env("PAPERMATE_DATA_DIR", &data_dir)
            .env("HF_HOME", &hf_home)
            .env("HF_ENDPOINT", "https://hf-mirror.com")
            // ★★ 2026-10-07：**显式**告诉后端 venv / uv-cache 在哪。
            //   为什么必须给：后端 `rag_deps.venv_dir()/uv_cache_dir()` 原先**只**从
            //   `data_dir.parent` 推导 —— 用户一旦把数据根目录换到别处，它会去找
            //   `<root>/..` 下的 venv，而真实 venv 在 `<root>/venv`
            //   ⇒ 「准备运行环境 / 下载模型」会**静默失效**（只报"未找到应用私有 venv"）。
            //   ★ 这两个变量把后端绑到**与上面同一份真源**（同源原则，`tests/test_rag_deps_paths.py` 钉死）。
            .env("PAPERMATE_VENV_DIR", &venv_dir)
            .env("PAPERMATE_UV_CACHE_DIR", &uv_cache_dir)
            // ★ 让后端能**写到**"位置配置"本身 —— 它在**固定的** `<local_data>/app-settings.json`，
            //   而数据目录可能已经搬到别处 ⇒ 后端无法从 data_dir 反推出来，只能由启动器告知。
            //   设置页的「改存储位置」就是写这个文件的 `pending_root` 字段（重启后由启动器执行）。
            .env("PAPERMATE_APP_SETTINGS", local_data.join(APP_SETTINGS))
            .spawn()
    {
        Ok(child) => {
            *BACKEND.lock().unwrap() = Some(child);
            // 本函数已经跑在**后台线程**上（见 `run()` 的 `setup`），所以这里直接等：
            // 它只轮询 TCP 端口，不占用事件循环。
            navigate_when_backend_ready(app.clone(), PORT);
        }
        Err(e) => {
            eprintln!("[desktop] failed to spawn backend: {}", e);
            show_startup_error(app, &format!("后端进程启动失败：{e}"));
        }
    }
}
