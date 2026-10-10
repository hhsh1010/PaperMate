"""测试基建：把 src/ 加入 sys.path，使 papermate / lightcore 可被导入。

生产环境默认走真实 BGE-M3（default_embedder=real / use_bge_sparse=true），
但测试套件保持 fake embedder，避免每个用例都加载 2.2GB 权重。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 在 papermate.config.Settings 首次实例化前注入测试默认值，确保全量测试走 fake。
os.environ.setdefault("PAPERMATE_DEFAULT_EMBEDDER", "fake")
os.environ.setdefault("PAPERMATE_USE_BGE_SPARSE", "false")
# T-05（2026-09-26）：关闭服务启动后的 RAG 模型后台预热。
# 为什么必须关：预热会真的加载 BGE-M3（实测 22s）。app 级测试（test_api 等）每个都
# `create_app()` → 每个都拉起一个加载线程 → 多线程争 GPU/磁盘，测试又慢又不稳。
# 而 `_rag_rt()` 读的是 runtime_settings（`data/provider_settings.json`，**优先级高于 .env**），
# 实测该文件里 `rag_embedder=real` —— 所以上面两行 setdefault 挡不住它，
# 必须由预热自己带开关（`PAPERMATE_PREWARM`）。
os.environ.setdefault("PAPERMATE_PREWARM", "0")

# ── T-10（2026-09-30）：**进程级**数据目录兜底隔离 ──────────────────────────────
# 为什么必须在**模块级**（早于任何 fixture、任何 app 创建）：
# `PAPERMATE_DATA_DIR` 是**进程级 env**，而各测试文件是在**自己的 fixture 里**设它，
# 那只保证"该 fixture 生效之后"。在此之前发生的（模块级 import 副作用、漏设 env 的用例）
# 会落到默认 `data_dir = data` —— 实测 `db_url = sqlite:///D:/.../data/papermate.db`
# （= **项目真实库**）。
#
# 串行时这被"前一个测试文件残留的 env"意外掩盖（env 不随文件结束还原）；
# ★ **并行时 xdist 各 worker 分到的文件顺序不同** ⇒ 掩盖失效 ⇒ 多进程同时用真实
# `data/` 建 app ⇒ `init_db` 触发 `snapshot_if_exists` ⇒ 争抢 `data/backups/`
# ⇒ Windows `0x80070005`(ACCESS_DENIED) 致 worker 崩溃。
# （2026-09-30 实测取证：并行跑产生 0 字节 `data/backups/tmp*.tmp`，ctime 精确落在
#   并行时刻；另有 stack overflow 崩溃。）
#
# ⚠️ 语义：**只在未设时兜底**（不覆盖测试自己设的值）⇒ 既有测试语义、用例数均不变。
if not os.environ.get("PAPERMATE_DATA_DIR"):
    os.environ["PAPERMATE_DATA_DIR"] = tempfile.mkdtemp(prefix="papermate_pytest_proc_")

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ── ★ 2026-10-10：进程级全局缓存的**每测试自动重置**（串行顺序依赖的根因修复）────
# 实测取证（CI 后端 job 串行跑暴露）：`test_budget_abort_user_text` 调
#   runtime_settings.update({"max_tokens_budget": 1000})
# ⇒ **进程级 `runtime_settings._STATE` 缓存残留"1000 token 预算"**；而后续测试（如
#   `test_citation_model_truncated`）的 fixture 只清 `cfg_mod._settings`、**没清 `_STATE`**
# ⇒ 那些用例的 Agent 循环被预算闸门提前掐断 ⇒ 检索工具根本没被调用（表现为"没有拿到引用"）。
#
# 为什么必须放 conftest（而不是给单个文件补清缓存）：
#   ① 这类缓存是**进程级**的，污染面 = "任何先跑过的、改过设置的用例" ⇒ 逐个打补丁是打地鼠；
#   ② 并行（-n N）时污染者与被污染者通常落在不同 worker ⇒ **侥幸绿**；串行/CI 串行必现
#      ——这正是"并行掩盖顺序依赖"的经典形态（与既有教训同族：串行被残留 env 掩盖）。
#
# ⚠️ 语义：只在**每个测试前后**清缓存（下一次读取即从磁盘重建）⇒ 不改任何用例的断言与行为；
#    用例内部自己设置的运行时设置照常生效（重置发生在用例之间，不在其内部）。
@pytest.fixture(autouse=True)
def _reset_process_global_caches():
    import papermate.config as _cfg
    from papermate import runtime_settings as _rs

    def _clear() -> None:
        # ⚠️ 变量名是 **`_state`（小写）** —— `test_budget_abort_user_text.py` 曾误写成 `_STATE`（大写）
        #    ⇒ 只是新建了个无关属性、真缓存根本没清（静默失效，已一并修正：见该文件 fixture）。
        _rs._state = None
        _cfg._settings = None

    _clear()
    yield
    _clear()


# ── T-10（2026-09-30）：分层标记的**单一来源** ────────────────────────────────
# 为什么放 conftest 而不是逐个测试文件加装饰器：
#   ① 卡片的范围就是 `pyproject.toml` + `tests/conftest.py`（不动测试源码）；
#   ② 清单是**实测得出的**（基线 `pytest --durations=40`），集中一处便于复核/更新；
#   ③ 零散装饰器在重构中容易丢失，这里若漏了会立刻从 `-m slow` 的计数看出来。
#
# ⚠️ 语义：**只加标记，不改变任何用例的行为** —— 用例数、断言、执行内容全不变；
#    且**默认不排除任何标记**（见 pyproject 里"为什么 addopts 不放 `-m`"）。
#
#: 含 `call` 阶段 ≥1s 用例的文件（日常门禁 `-m "not slow"` 会跳过它们）。
_SLOW_FILES = {
    "test_api.py",           # 断连/取消机制（全库最长 10.43s）+ 全流程 1.65s
    "test_rag_bgem3.py",     # 真加载 BGE-M3 权重（3.63s；同时属 gpu）
    "test_async_upload.py",  # 取消中途上传（asyncio.sleep 2s → 2.70s）
    "test_docx_edit.py",     # 持久化编辑锁（2.35s）
    "test_eval_judge.py",    # 判定器超时路径（1.53s + 1.52s）
    # 2026-06-06 新增：本文件反复把 torch 从 sys.modules 里摘掉再恢复 ⇒ 每次都要
    # 冷读 4GB 的 torch 包目录。实测**串行仅 28s，但 -n 3 并行退化到 207s**
    # （多 worker 同时读同一批 .pyd 的磁盘争用，非用例本身慢）。
    # ⇒ 放进 slow 集合，日常门禁 `-m "not slow"` 跳过；命名映射那条快测试已单独拆出
    #   （test_heavy_names_mapping_is_consistent，零导入、毫秒级）继续覆盖日常回归。
    "test_rag_deps_light_boot.py",
}

#: 需要真实 GPU 或真实模型权重的文件。
#: ⚠️ `test_rag_device.py` / `test_rerank_contract.py` 虽然名字像，但都用**假替身**
#:    （monkeypatch / `_FakeCrossEncoder`）⇒ **不占显存，不进此表**。
_GPU_FILES = {"test_rag_bgem3.py"}

#: 需要真实外网/真实厂商 API 的文件。
#: ⚠️ **当前为空**：全库实测**零真实出网** —— `test_phase_e.py::test_balances_deepseek_live`
#:    对 `httpx.AsyncClient.get` 打了 mock，`test_eval_judge.py` 用 `httpx.MockTransport`。
#:    保留该集合是给未来新增加的真实网络用例一个落点。
_NETWORK_FILES: set[str] = set()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """按**文件**自动打 `slow` / `gpu` / `network` 标记（清单见上，来自实测耗时）。"""
    for item in items:
        fname = item.path.name
        if fname in _SLOW_FILES:
            item.add_marker(pytest.mark.slow)
        if fname in _GPU_FILES:
            item.add_marker(pytest.mark.gpu)
        if fname in _NETWORK_FILES:
            item.add_marker(pytest.mark.network)


# ── O-17（2026-10-01）：源码树卫生守护 —— 测试结束后 `src/` 下不得有 `tmp*` 残留 ──
# 背景：09-21~09-23 期间 `src/` 下累积了 25 个空 `tmp*` 目录（tempfile.mkdtemp 默认前缀
# 形态），非 src 代码所建、当前未再见新增 ⇒ 已清理。本守护防"再生成"：若某次测试改动
# 又开始往源码树泄漏临时目录，全量门禁会直接报错，而不是等数周后再人肉发现。
#
# 为什么用 `pytest_sessionfinish` 而不是普通测试：残留类泄漏没有固定"肇事文件"，
# 普通测试按文件顺序执行，可能跑在肇事文件**之前**而漏检；sessionfinish 在**所有**
# 用例与 fixture 之后执行，任何泄漏都必然已被留下。
def _src_tmp_residue() -> list[Path]:
    return [p for p in (Path(__file__).resolve().parents[1] / "src").rglob("tmp*") if p.is_dir()]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    residue = _src_tmp_residue()
    if residue:
        names = ", ".join(str(p.relative_to(session.config.rootpath)) for p in residue[:5])
        raise AssertionError(
            f"src/ 下发现 {len(residue)} 个 tmp* 测试残留目录（{names}...）。"
            "请排查是哪个用例在源码树里创建临时目录（应使用 tmp_path fixture 或显式 dir=）。"
        )
