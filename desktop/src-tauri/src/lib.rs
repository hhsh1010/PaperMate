use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

use tauri::Manager;

// 与主进程绑定的 sidecar 子进程句柄（退出时清理进程树）
static BACKEND: Mutex<Option<Child>> = Mutex::new(None);

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8000;
const APP_DATA: &str = "PaperMate";

/// 首启**不安装**的重依赖（**发行包名**，uv 的 `--no-install-package` 按发行名匹配）。
///
/// ★ 唯一真源在 Python 侧 `papermate/services/rag_deps.py::HEAVY_DISTRIBUTIONS`；
///   两处必须**逐字一致** —— 不一致会导致"首启排除了、启用时没补装"（用户点了没反应）
///   或反之（首启又变成 4.8GB）。`tests/test_desktop_heavy_deps_contract.py` 静态校验这一点。
/// ★ 注意 `sentence-transformers` 是**连字符**（发行名）；其 Python 导入名是下划线形式。
const HEAVY_DEPS: [&str; 3] = ["torch", "FlagEmbedding", "sentence-transformers"];

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            bootstrap_and_spawn(app);
            Ok(())
        })
        .run(tauri::generate_context!(), |_app, event| {
            // 窗口关闭 → 杀掉 Python sidecar，避免 8000 端口残留
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(mut child) = BACKEND.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}

fn bootstrap_and_spawn(app: &tauri::AppHandle) {
    let res_dir = match app.path().resource_dir() {
        Some(p) => p,
        None => {
            eprintln!("[desktop] resource_dir unavailable");
            return;
        }
    };
    let backend_dir = res_dir.join("resources/backend");
    let python_exe = res_dir.join("resources/python/python.exe");
    let uv_exe = res_dir.join("resources/python/uv.exe");

    let local_data = match app.path().local_data_dir() {
        Some(p) => p.join(APP_DATA),
        None => {
            eprintln!("[desktop] local_data_dir unavailable");
            return;
        }
    };
    let data_dir = local_data.join("data");
    let hf_home = local_data.join("huggingface");
    let venv_dir = local_data.join("venv");
    let venv_python = venv_dir.join("Scripts").join("python.exe");

    if let Err(e) = fs::create_dir_all(&data_dir) {
        eprintln!("[desktop] cannot create data_dir {:?}: {}", data_dir, e);
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
    if !venv_python.exists() {
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
            .args(uv_args)
            .status()
        {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("[desktop] uv sync failed: {}", s);
                return;
            }
            Err(e) => {
                eprintln!("[desktop] cannot run uv.exe: {}", e);
                return;
            }
        }
    }

    // 启动后端 sidecar：一个进程同时托管 /api/v1/* 与 / (SPA, web/dist)
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
        }
        Err(e) => {
            eprintln!("[desktop] failed to spawn backend: {}", e);
        }
    }
}
