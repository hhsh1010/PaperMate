"""O-20 · 真实语义检索模型的**显式安装**（2026-10-05 桌面客户端）

纪律：本文件**零真实网络、零真实下载、零真实 GPU**。
- `hf_hub_download` / `HfApi.list_repo_files` 一律打桩（首次也被 `check_free_space` 拦在前）；
- 磁盘余量用注入的 `usage` / 直接替换 `check_free_space`；
- 因此本文件**不进** conftest 的 `slow`/`gpu`/`network` 清单。

覆盖四件事（每一件都对应一个真实的坑）：
1. **下载清单筛选**：不可下载 `imgs/.DS_Store` / `onnx/` 之类的无关文件
   （hf-mirror 对前者 403 ⇒ `snapshot_download` 会整批 abort）；
2. **成败判定按"推理必需是否齐备"**，而非"有没有文件失败" ⇒ 无关文件失败照样成功；
3. **后台线程 + 状态机**：成功/失败/取消/并发互斥四条路径；
4. **默认 fake**：首次种子必须是 fake（不许静默拉 4GB）。
"""

from __future__ import annotations

import threading
import time

import pytest

import papermate.config as cfg_mod
from papermate.services import model_installer as mi
from papermate.services.model_installer import (
    AlreadyRunning,
    ModelInstaller,
    ModelInstallError,
    RagModelSpec,
    missing_required,
    plan_files,
    probe_installed,
    reset_model_installer,
)

# bge-m3 仓库里的典型干扰文件（实测 `imgs/.DS_Store` 在 hf-mirror 上返回 403）
_REPO_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
    "imgs/.DS_Store",          # ← 403 元凶
    "onnx/model.onnx",         # ← 几百 MB 的无关产物
    "openvino/openvino_model.xml",
]


@pytest.fixture(autouse=True)
def _isolated_installer(monkeypatch):
    """每个用例独立的 installer 单例 + 固定 HF_ENDPOINT（防 `ensure_hf_mirror` 改进程 env）。"""
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    reset_model_installer()
    yield
    reset_model_installer()


def _spec(**kw) -> RagModelSpec:
    base = dict(key="embedder", repo_id="BAAI/bge-m3", label="BGE-M3", approx_gb=2.3,
                allow_patterns=mi.EMBEDDER_SPEC.allow_patterns,
                required_files=mi.EMBEDDER_SPEC.required_files)
    base.update(kw)
    return RagModelSpec(**base)


def _wait_done(inst: ModelInstaller, timeout: float = 8.0) -> dict:
    """轮询直到任务进入终态（遍历 daemon 线程自身的 join 更直接）。"""
    t = inst._thread  # noqa: SLF001  测试需要直接操作线程退出同步
    if t is not None:
        t.join(timeout)
    deadline = time.time() + timeout
    while time.time() < deadline and inst.task()["status"] == "running":
        time.sleep(0.02)
    return inst.task()


# ───────────────────────── 1. 下载清单筛选 ─────────────────────────
def test_plan_files_filters_trash_and_keeps_essentials():
    wanted = plan_files(list(_REPO_FILES), mi.EMBEDDER_SPEC)

    assert "config.json" in wanted
    assert "model.safetensors" in wanted
    assert "1_Pooling/config.json" in wanted
    # ★ 坑点：这两个下去必然踩 403 / 白占几百 MB
    assert "imgs/.DS_Store" not in wanted
    assert "onnx/model.onnx" not in wanted
    assert "README.md" not in wanted


def test_plan_files_dedups_and_stable_order():
    wanted = plan_files(list(_REPO_FILES), mi.EMBEDDER_SPEC)
    assert len(wanted) == len(set(wanted)), "同一文件不应重复下载"


def test_plan_files_keeps_repo_order_and_matches_shards():
    """分片权重必须被识别（`model-0000x-of-0000y.safetensors`），且保持仓库清单顺序。"""
    files = ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
             "model.safetensors.index.json", "tokenizer.json"]
    assert plan_files(files, mi.RERANKER_SPEC) == files


# ───────────────────────── 2. 成败判定 ─────────────────────────
def test_missing_required_includes_weight_and_tokenizer():
    missing = missing_required(mi.EMBEDDER_SPEC, ["config.json"])
    assert "tokenizer.json" in missing
    assert any("权重" in m for m in missing), missing


@pytest.mark.parametrize("got", [
    ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
    ["config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin"],
    ["config.json", "tokenizer.json", "tokenizer_config.json", "model-00001-of-00002.safetensors"],
])
def test_missing_required_accepts_any_weight_form(got):
    assert missing_required(mi.EMBEDDER_SPEC, got) == []


def test_missing_required_when_no_weight():
    missing = missing_required(mi.EMBEDDER_SPEC,
                               ["config.json", "tokenizer.json", "tokenizer_config.json"])
    assert len(missing) == 1 and "权重" in missing[0]


# ───────────────────────── 3. 磁盘余量 ─────────────────────────
def test_check_free_space_ok(monkeypatch, tmp_path):
    class _U:
        free = int(100 * mi.GB)
        total = int(200 * mi.GB)
        used = int(100 * mi.GB)
    free = mi.check_free_space(2.3, dirpath=str(tmp_path), usage=lambda p: _U())
    assert free == 100.0


def test_check_free_space_insufficient(monkeypatch, tmp_path):
    class _U:
        free = int(1.0 * mi.GB)
        total = int(200 * mi.GB)
        used = int(199 * mi.GB)
    with pytest.raises(ModelInstallError) as ei:
        mi.check_free_space(2.3, dirpath=str(tmp_path), usage=lambda p: _U())
    assert "磁盘余量不足" in str(ei.value)


def test_check_free_space_unreadable_path_is_skipped_not_fatal():
    """★★ 2026-10-07 契约更新（原 `test_check_free_space_unreadable_path`）。

    **旧契约**：探测失败 ⇒ 抛 `ModelInstallError`（原注释："路径不存在/无权限 → 显式失败而非继续下载"）。

    **为什么改**（真机首装实测）：该契约会**误杀真实下载** —— Windows 上对
    "刚被 ensure_hf_mirror 创建出来"的 HF 缓存目录调 `shutil.disk_usage`
    （内部是 `GetDiskFreeSpaceExW`）会抛 `FileNotFoundError`；而此时 4GB 重依赖其实
    已装完，下载却在权重阶段被这条检查中断 ⇒ `status=error`，
    UI 表现为"卡在准备运行环境 / 下载失败"。

    **新契约**：**测不出余量 ≠ 余量不足** ⇒ 探测失败只记 warning 并放行（返回 0.0），
    绝不拦下载；只有"成功测得且确实不足"才拦截（`test_check_free_space_insufficient`
    仍在、语义未变）。与展示口径 `free_space_gb`（探测失败返回 None、不挂卡片）同源。
    """
    def _boom(_p):  # 模拟"探测本身失败"（不可测），而不是"测得不足"
        raise OSError("no")

    assert mi.check_free_space(1.0, dirpath="Z:/no-such-drive/x", usage=_boom) == 0.0


# ───────────────────────── 4. 安装状态探测 ─────────────────────────
def test_probe_installed_uses_runtime_resolver():
    """探测必须与线上推理**同源**（都走 `resolve_local_hf_snapshot`）。"""
    got = probe_installed(mi.EMBEDDER_SPEC, resolver=lambda repo: ("/local/snap" if repo == "BAAI/bge-m3" else None))
    assert got == "/local/snap"
    assert probe_installed(mi.RERANKER_SPEC, resolver=lambda repo: None) is None


# ───────────────────────── 5. 后台线程状态机 ─────────────────────────
def _patch_hub(monkeypatch, files=None, download=None):
    """给 `huggingface_hub` 打桩（本模块是函数内惰性 import ⇒ 打挂载点即可）。"""
    import huggingface_hub

    class _Api:
        def list_repo_files(self, repo_id, repo_type="model"):
            return list(files if files is not None else _REPO_FILES)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download or (lambda repo, name, **kw: f"/cache/{name}"))
    monkeypatch.setattr(mi, "check_free_space", lambda gb: 100.0)


def test_install_success(monkeypatch):
    seen: list[str] = []
    def _dl(repo, name, **kw):
        seen.append(name)
        return f"/cache/{name}"
    _patch_hub(monkeypatch, download=_dl)

    inst = ModelInstaller()
    first = inst.install(["embedder"])
    assert first["status"] == "running" and first["models"] == ["embedder"]

    end = _wait_done(inst)
    assert end["status"] == "success", end.get("error")
    assert end["percent"] == 100.0
    # 下载了必需文件、且**没碰** 403 元凶
    assert "config.json" in seen and "model.safetensors" in seen
    assert "imgs/.DS_Store" not in seen
    # ★ 下载后不自动启用：这里只报"请手动启用 + 重建索引"
    assert "重建索引" in end["message"]


def test_install_tolerates_optional_file_failure(monkeypatch):
    """★ 核心坑：`imgs/.DS_Store` 之类无关文件下载失败 ⇒ 整次安装**仍应成功**。"""
    def _dl(repo, name, **kw):
        if name in ("special_tokens_map.json", "vocab.txt"):
            raise RuntimeError("403 Forbidden")
        return f"/cache/{name}"
    _patch_hub(monkeypatch, download=_dl)

    inst = ModelInstaller()
    inst.install(["embedder"])
    end = _wait_done(inst)
    assert end["status"] == "success", end.get("error")


def test_install_fails_when_required_missing(monkeypatch):
    def _dl(repo, name, **kw):
        if name == "config.json":
            raise RuntimeError("network down")
        return f"/cache/{name}"
    _patch_hub(monkeypatch, download=_dl)

    inst = ModelInstaller()
    inst.install(["embedder"])
    end = _wait_done(inst)
    assert end["status"] == "error"
    assert "不完整" in (end["error"] or "")


def test_install_unexpected_exception_recorded(monkeypatch):
    """后台线程的异常绝不能静默消失（否则前端永远显示"下载中"）。"""
    import huggingface_hub

    _patch_hub(monkeypatch)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: (_ for _ in ()).throw(OSError("boom")))

    inst = ModelInstaller()
    inst.install(["embedder"])
    end = _wait_done(inst)
    assert end["status"] == "error"
    assert "无法访问模型仓库" in (end["error"] or "")


def test_install_unknown_model_key():
    with pytest.raises(ModelInstallError) as ei:
        ModelInstaller().install(["bogus"])
    assert "未知模型" in str(ei.value)


def test_install_rejects_concurrent_run(monkeypatch):
    entered = threading.Event()

    def _dl(repo, name, **kw):
        entered.set()
        time.sleep(0.05)
        return f"/cache/{name}"
    _patch_hub(monkeypatch, download=_dl)

    inst = ModelInstaller()
    inst.install(["embedder"])          # 不传 models 时下面这条才会代表"装全部"
    assert entered.wait(2)
    with pytest.raises(AlreadyRunning):
        inst.install()
    _wait_done(inst)


def test_cancel_stops_before_next_file(monkeypatch):
    """取消在**文件间隙**生效 ⇒ 已下载部分留下，下次可续传。"""
    entered = threading.Event()
    release = threading.Event()
    downloaded: list[str] = []

    def _dl(repo, name, **kw):
        downloaded.append(name)
        entered.set()
        assert release.wait(5), "测试未在取消后放行，会死锁"
        return f"/cache/{name}"
    _patch_hub(monkeypatch, download=_dl)

    inst = ModelInstaller()
    inst.install(["embedder"])
    assert entered.wait(2)
    inst.cancel()
    release.set()
    end = _wait_done(inst)
    assert end["status"] == "cancelled"
    assert len(downloaded) < len(plan_files(list(_REPO_FILES), mi.EMBEDDER_SPEC))


def test_cancel_without_running_task():
    with pytest.raises(ModelInstallError):
        ModelInstaller().cancel()


def test_missing_hub_dependency_is_actionable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(mi, "check_free_space", lambda gb: 100.0)

    inst = ModelInstaller()
    inst.install(["embedder"])
    end = _wait_done(inst)
    assert end["status"] == "error"
    assert "uv sync --extra rag" in (end["error"] or "")


# ───────────────────────── 6. 清单 / 首屏数据 ─────────────────────────
def test_free_space_gb_falls_back_to_existing_ancestor(monkeypatch, tmp_path):
    """★★ 首装回归：HF 缓存目录**还不存在**时，不能返回 None。

    修复前 `models()` 用 `os.path.isdir(cache_dir())` 兜底 ⇒ 新用户首屏
    `disk_free_gb=null` ⇒ 用户无从判断 4.5GB 够不够（首屏最关键的决策信息）。
    """
    missing = tmp_path / "not" / "created" / "yet" / "huggingface" / "hub"
    assert not missing.exists()
    monkeypatch.setattr(mi, "cache_dir", lambda: str(missing))

    got = mi.free_space_gb()
    assert got is not None and got > 0, "缓存目录不存在时应回退到已存在的祖先目录"
    assert isinstance(got, float)


def test_free_space_gb_returns_none_on_unprobeable(monkeypatch, tmp_path):
    """探测彻底失败时给 None 而不是抛错 —— 展示属增强信息，不该让整张卡片挂掉。"""
    monkeypatch.setattr(mi, "cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mi.shutil, "disk_usage", lambda p: (_ for _ in ()).throw(OSError("denied")))
    assert mi.free_space_gb() is None


def test_models_reports_install_state_and_hint_partial(monkeypatch, tmp_path):
    """只装了一半（缺 reranker）⇒ `all_installed=False`，提示"去下载"。"""
    monkeypatch.setattr(mi, "probe_installed",
                        lambda spec: "/local/snap" if spec.key == "embedder" else None)
    monkeypatch.setattr(mi, "cache_dir", lambda: str(tmp_path))

    from papermate import runtime_settings as rs

    monkeypatch.setattr(rs, "get_state", lambda: {"rag_embedder": "fake"})

    out = ModelInstaller().models()
    by_key = {m["key"]: m for m in out["models"]}
    assert by_key["embedder"]["installed"] is True
    assert by_key["reranker"]["installed"] is False
    assert out["all_installed"] is False
    assert out["enabled"] is False
    assert "下载模型" in out["hint"]
    assert isinstance(out["disk_free_gb"], float)


def test_models_hint_tells_user_to_rebuild_index(monkeypatch, tmp_path):
    """★ 两个都装好、但还没启用 ⇒ 必须提示**手动启用 + 重建索引**（不重建则不生效）。"""
    monkeypatch.setattr(mi, "probe_installed", lambda spec: "/local/snap")
    monkeypatch.setattr(mi, "cache_dir", lambda: str(tmp_path))
    from papermate import runtime_settings as rs

    monkeypatch.setattr(rs, "get_state", lambda: {"rag_embedder": "fake"})
    out = ModelInstaller().models()
    assert out["all_installed"] is True and out["enabled"] is False
    assert "重建索引" in out["hint"]


def test_models_hint_when_already_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(mi, "probe_installed", lambda spec: None)
    monkeypatch.setattr(mi, "cache_dir", lambda: str(tmp_path))
    from papermate import runtime_settings as rs

    monkeypatch.setattr(rs, "get_state", lambda: {"rag_embedder": "real"})
    out = ModelInstaller().models()
    assert out["enabled"] is True
    assert out["embedder"] == "real"


# ───────────────────────── 7. API 契约 ─────────────────────────
@pytest.fixture
def client(tmp_path, monkeypatch):
    import os

    os.environ["PAPERMATE_DATA_DIR"] = str(tmp_path / "data")
    cfg_mod._settings = None
    from fastapi.testclient import TestClient

    from papermate.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_api_models_and_status_endpoints(client, monkeypatch):
    monkeypatch.setattr(mi, "check_free_space", lambda gb: 100.0)

    r = client.get("/api/v1/settings/rag/models")
    assert r.status_code == 200, r.text
    body = r.json()
    assert {m["key"] for m in body["models"]} == {"embedder", "reranker"}
    assert body["enabled"] is False           # ★ 默认 fake

    st = client.get("/api/v1/settings/rag/models/install/status")
    assert st.status_code == 200
    assert st.json()["status"] == "idle"


def test_api_install_unknown_model_returns_400(client):
    r = client.post("/api/v1/settings/rag/models/install", json={"models": ["nope"]})
    assert r.status_code == 400
    assert "未知模型" in r.json()["detail"]


def test_api_install_conflict_returns_409(client, monkeypatch):
    entered = threading.Event()

    def _dl(repo, name, **kw):
        entered.set()
        time.sleep(0.05)
        return f"/cache/{name}"
    _patch_hub(monkeypatch, download=_dl)
    # ⚠️ 必须打在**路由模块**上：它是 `from ... import get_model_installer` 的直接引用，
    #    打在 `mi` 上不影响它（踩过：打错位置 ⇒ 操作的是另一个单例，断言失真）。
    from papermate.api import routes_settings as rs_api

    monkeypatch.setattr(rs_api, "get_model_installer", lambda: inst)

    inst = ModelInstaller()
    first = client.post("/api/v1/settings/rag/models/install", json={"models": ["embedder"]})
    assert first.status_code == 200
    assert entered.wait(2)

    second = client.post("/api/v1/settings/rag/models/install", json={"models": ["embedder"]})
    assert second.status_code == 409
    inst._thread.join(8)  # noqa: SLF001  收尾，避免线程泄漏到下一用例


def test_api_cancel_without_task_returns_409(client):
    r = client.post("/api/v1/settings/rag/models/install/cancel")
    assert r.status_code == 409


# ───────────────────────── 9. 两阶段安装（deps → models）─────────────────────────
def test_install_runs_deps_phase_before_models(monkeypatch):
    """★ 桌面首装场景：重依赖缺失 ⇒ 必须先装依赖（约 4GB）再下模型。

    这是"首启只装 0.76GB"方案能成立的前提：否则点下载后 BGE-M3 直接 import 失败。
    """
    order: list[str] = []
    monkeypatch.setattr(mi.rag_deps, "probe", lambda **kw: _Deps(False))
    monkeypatch.setattr(mi.rag_deps, "sync_heavy_deps",
                        lambda **kw: order.append("deps") or "ok")
    _patch_hub(monkeypatch, download=lambda repo, name, **kw: order.append(f"dl:{name}") or "")

    inst = ModelInstaller()
    inst.install(["embedder"])
    end = _wait_done(inst)
    assert end["status"] == "success", end.get("error")
    assert order[0] == "deps", f"依赖安装必须是第一步，实际顺序：{order[:3]}"
    assert any(x.startswith("dl:") for x in order)
    assert end["phase"] == "models"


def test_install_skips_deps_when_already_ready(monkeypatch):
    """开发机/服务端依赖已齐 ⇒ **不得**重复跑 uv sync（否则白等几分钟）。"""
    called: list[str] = []
    monkeypatch.setattr(mi.rag_deps, "probe", lambda **kw: _Deps(True))
    monkeypatch.setattr(mi.rag_deps, "sync_heavy_deps", lambda **kw: called.append("deps"))
    _patch_hub(monkeypatch)

    inst = ModelInstaller()
    inst.install(["embedder"])
    end = _wait_done(inst)
    assert end["status"] == "success", end.get("error")
    assert called == [], "依赖已就绪时不应调用 sync_heavy_deps"


def test_install_reports_deps_failure(monkeypatch):
    """依赖安装失败 ⇒ status=error 且带**可读**原因（不是裸 ImportError）。"""
    monkeypatch.setattr(mi.rag_deps, "probe", lambda **kw: _Deps(False))

    def _boom(**kw):
        raise mi.rag_deps.DepsInstallError("未找到捆绑的 uv.exe（请确认安装包完整）")

    monkeypatch.setattr(mi.rag_deps, "sync_heavy_deps", _boom)
    _patch_hub(monkeypatch)

    inst = ModelInstaller()
    inst.install(["embedder"])
    end = _wait_done(inst)
    assert end["status"] == "error"
    assert "uv.exe" in (end["error"] or "")
    assert end["phase"] == "deps"


def test_models_exposes_deps_state(monkeypatch, tmp_path):
    """首屏必须告诉前端"还要装 4GB 依赖"，否则用户点了才知道。"""
    monkeypatch.setattr(mi, "probe_installed", lambda spec: None)
    monkeypatch.setattr(mi, "cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mi.rag_deps, "probe", lambda **kw: _Deps(False, ["torch：未安装"]))
    monkeypatch.setattr(mi.rag_deps, "describe_target", lambda: {"uv_found": False, "reason": "x"})

    from papermate import runtime_settings as rs

    monkeypatch.setattr(rs, "get_state", lambda: {"rag_embedder": "fake"})
    out = ModelInstaller().models()
    assert out["deps_ready"] is False
    assert out["deps_missing"] == ["torch：未安装"]
    assert "4GB" in out["hint"], out["hint"]


class _Deps:
    """`rag_deps.DepsStatus` 的极简替身（只带本测试用到的字段）。"""

    def __init__(self, ready: bool, missing: list[str] | None = None) -> None:
        self.ready = ready
        self.missing = missing or []
        self.unknown = False
        self.torch_cuda = None
        self.torch_version = None
def test_config_default_embedder_is_fake(monkeypatch):
    """★ 桌面 App 装完不能静默拉 ~4GB：首次种子必须是 fake。

    且必须是**可被 env 覆盖**的（否则 `PAPERMATE_DEFAULT_EMBEDDER` 就成了死配置）。
    """
    from papermate.config import Settings

    monkeypatch.delenv("PAPERMATE_DEFAULT_EMBEDDER", raising=False)
    assert Settings(default_model="fake").default_embedder == "fake"

    monkeypatch.setenv("PAPERMATE_DEFAULT_EMBEDDER", "real")
    assert Settings(default_model="fake").default_embedder == "real"


def test_seed_layer_follows_config(monkeypatch):
    """`runtime_settings` 的种子必须与 config **同源**（那儿硬编码 fake 会让 env 失效）。"""
    from papermate import runtime_settings as rs

    monkeypatch.setattr(rs, "_cfg", lambda: _S(embedder="real"))
    assert rs._default_state()["rag_embedder"] == "real"
    monkeypatch.setattr(rs, "_cfg", lambda: _S(embedder="fake"))
    assert rs._default_state()["rag_embedder"] == "fake"


class _S:
    def __init__(self, embedder: str) -> None:
        self.default_embedder = embedder
        self.use_bge_sparse = False
        self.use_jieba = False


# ───────────────────────── 10. 启用 real 的前置校验（API 层）─────────────────────────
def test_api_rejects_real_when_deps_missing(client, monkeypatch):
    """★ 缺依赖时切 real 必须被**当场拒绝**，而不是保存成功、检索时才 500。"""
    from papermate.api import routes_settings as rs_api
    from papermate.services import rag_deps as rd

    monkeypatch.setattr(rd, "probe", lambda **kw: _Deps(False, ["torch：未安装", "FlagEmbedding：未安装"]))
    monkeypatch.setattr(rd, "describe_target", lambda: {"reason": "未找到应用私有 venv"})

    r = client.put("/api/v1/settings", json={"rag_embedder": "real"})
    assert r.status_code == 400, r.text
    d = r.json()["detail"]
    assert "torch" in d and "4GB" in d and "下载模型" in d
    assert rs_api.router is not None  # 仅确保导入路径正确


def test_api_allows_real_when_deps_ready(client, monkeypatch):
    """依赖就绪时切 real 必须放行（否则把正常路径也堵死）。"""
    from papermate.services import rag_deps as rd

    monkeypatch.setattr(rd, "probe", lambda **kw: _Deps(True))
    r = client.put("/api/v1/settings", json={"rag_embedder": "real"})
    assert r.status_code == 200, r.text
    assert r.json()["rag_embedder"] == "real"
    # 复位，避免影响同文件后续用例
    client.put("/api/v1/settings", json={"rag_embedder": "fake"})


def test_api_fake_always_allowed(client, monkeypatch):
    """★ 缺依赖**不得**阻止用户保存其它设置或切回 fake（校验只拦 real 这一条路）。"""
    from papermate.services import rag_deps as rd

    monkeypatch.setattr(rd, "probe", lambda **kw: _Deps(False, ["torch：未安装"]))
    r = client.put("/api/v1/settings", json={"rag_embedder": "fake", "use_jieba": True})
    assert r.status_code == 200, r.text
    assert r.json()["rag_embedder"] == "fake"
