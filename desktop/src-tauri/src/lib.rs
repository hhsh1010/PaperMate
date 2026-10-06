use std::fs;
use std::process::{Child, Command};
use std::sync::Mutex;

use tauri::Manager;

// 与主进程绑定的 sidecar 子进程句柄（退出时清理进程树）
static BACKEND: Mutex<Option<Child>> = Mutex::new(None);

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8000;
const APP_DATA: &str = "PaperMate";

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
        .setup(|app| {
            // 引导逻辑只需要 `&AppHandle` ⇒ 用 `app.handle()` 取句柄。
            // （直接传 `app` 会报 expected `&AppHandle`, found `&mut App`）
            bootstrap_and_spawn(app.handle());
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
    let data_dir = local_data.join("data");
    let hf_home = local_data.join("huggingface");
    let venv_dir = local_data.join("venv");
    // ★ App 专属 uv 分发缓存（与 venv / huggingface 并列在 App 数据目录下）。
    //   2026-10-06 实测事故：共用用户全局 uv 缓存时，首启因一个**损坏的缓存条目**失败：
    //     Failed to write to the distribution cache / failed to open file
    //     ...\uv\cache\sdists-v9\...\jieba-0.42.1-py3-none-any.whl: 拒绝访问 (os error 5)
    //   该 .whl 连 `uv cache clean` 都删不掉、`icacls` 读不到 ACL（属主/ACL 异常），
    //   而同目录其它文件可正常删除 ⇒ 全局缓存里有**永久损坏的条目**。
    //   指向 App 私有目录后，启动不再依赖用户机器上的历史残留。
    //   ⚠️ 后端 `rag_deps.uv_cache_dir()` 用同一约定（<local_data>/uv-cache），两处必须一致。
    //   ★ 若外部已显式设置 UV_CACHE_DIR，则**尊重它**（与 HF_ENDPOINT 的处理一致）——
    //     既保留"排查时可临时指到别处"的逃生口，也让默认值保持确定性。
    let uv_cache_dir = std::env::var("UV_CACHE_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| local_data.join("uv-cache"));
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
    let env_file = backend_dir.join(".env");
    if !env_file.exists() {
        let content = format!(
            "PAPERMATE_HOST={host}\nPAPERMATE_PORT={port}\nPAPERMATE_DATA_DIR={dd}\nHF_HOME={hf}\nHF_ENDPOINT=https://hf-mirror.com\n",
            host = HOST,
            port = PORT,
            dd = data_dir.display(),
            hf = hf_home.display()
        );
        let _ = fs::write(&env_file, content);
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
        .spawn()
    {
        Ok(child) => {
            *BACKEND.lock().unwrap() = Some(child);
            let handle = app.clone();
            std::thread::spawn(move || navigate_when_backend_ready(handle, PORT));
        }
        Err(e) => {
            eprintln!("[desktop] failed to spawn backend: {}", e);
            show_startup_error(app, &format!("后端进程启动失败：{e}"));
        }
    }
}
