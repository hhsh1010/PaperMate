"""T-10 配置守护（2026-09-30）：防止"并行 / 分层标记"的配置被改坏而**静默退化**。

每一条断言都对应一次**实测事故**或一个真实风险：

| 若被改坏 | 后果 |
|---|---|
| addopts 里加了 `-m "not slow"` | `pytest` 裸跑**静默跳过** slow 用例 ⇒ 「假通过」（教义明令禁止） |
| `-n 3` 改成 `-n auto` | 24 个 worker × 启动峰值 ~1.75GB ⇒ **OOM**（实测崩过，遗留孤儿进程） |
| 丢掉 `--dist=loadfile` | 同文件用例被拆到不同 worker ⇒ 破坏"文件内隐式顺序"（env / module fixture） |
| 删掉 conftest 的兜底 DATA_DIR | 并行时会有进程落到**项目真实 `data/`** ⇒ 争抢 + 备份残件（实测事故） |
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _pytest_cfg() -> dict:
    with (_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["tool"]["pytest"]["ini_options"]


def test_addopts_never_excludes_by_default() -> None:
    """★ 默认必须跑**全量**：`-m` / `-k` 一旦进了 addopts，就是"默认漏测"。"""
    addopts = _pytest_cfg()["addopts"]
    tokens = addopts.split()
    assert "-m" not in tokens, f"addopts 不得含 -m（默认排除=假通过）：{addopts!r}"
    assert "-k" not in tokens, f"addopts 不得含 -k：{addopts!r}"


def test_parallelism_is_explicit_small_number() -> None:
    """★ 并行度必须是**显式小数字**：`auto`/`logical` 会按核数起进程 ⇒ 实测 OOM。"""
    addopts = _pytest_cfg()["addopts"]
    m = re.search(r"-n\s+(\S+)", addopts)
    assert m, f"addopts 缺少 -n：{addopts!r}"
    assert m.group(1).isdigit(), (
        f"-n 必须是显式数字（auto/logical 会起 ≈24 个 worker，"
        f"启动峰值 ~1.75GB/个 ⇒ OOM）：{m.group(1)!r}"
    )
    n = int(m.group(1))
    assert 1 <= n <= 3, (
        f"-n {n} 超出本机实测安全区间 1~3"
        "（-n 4 实测 37s 内可用内存 7.6GB→612MB 并崩溃；见 pyproject 注释）"
    )


def test_dist_is_loadfile() -> None:
    """★ 必须按文件分发，保住"文件内隐式顺序"的语义。"""
    assert "--dist=loadfile" in _pytest_cfg()["addopts"]


def test_all_markers_declared() -> None:
    """三层标记必须都在 `markers` 里声明（否则 `-m` 会告警/语义不明）。"""
    declared = " ".join(_pytest_cfg().get("markers", []))
    for name in ("slow", "gpu", "network"):
        assert name in declared, f"markers 未声明 {name}"


def test_conftest_keeps_process_level_data_dir_fallback() -> None:
    """★ 兜底隔离不得被删：它是"并行时不碰真实 data/"的唯一防线（实测事故）。"""
    src = (_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "PAPERMATE_DATA_DIR" in src and "mkdtemp" in src
    assert 'if not os.environ.get("PAPERMATE_DATA_DIR")' in src, (
        "兜底必须是 setdefault 语义（不覆盖测试自己设的值）"
    )
    assert "pytest_collection_modifyitems" in src, "分层标记的单一来源不见了"


def test_single_instance_lock_runner_exists() -> None:
    """★ 防"命令被重放 ⇒ 两套 worker 撞内存"的启动器必须在，且**锁必须落在共享位置**。

    2026-09-30：锁原来放在 `tempfile.gettempdir()`，同一 Bash 调用内并发时有效，但**「turn 重放」
    产生的另一个实例锁不住**（怀疑 TEMP 解析到别处 ⇒ 各锁各的文件）。两个实例共享同一仓库，
    故锁路径改用**仓库内固定路径**。此断言防它被改回系统临时目录。

    ⚠️ 顺带记一条已被证伪的判据：**`bringing up nodes...` 打 2 次不等于"有两个实例"**
    —— 单次 `-n 3` 运行就会打 2 次；真判据是**两条汇总行**。
    """
    runner = _ROOT / "scripts" / "run_tests.py"
    assert runner.exists(), "scripts/run_tests.py 缺失（单实例锁是并行安全的前提）"
    src = runner.read_text(encoding="utf-8")
    # ⚠️ 只断言**真正会执行的那行赋值**：早先写成 `"tempfile.gettempdir()" not in src`，
    #    结果匹配到上面 docstring 里的**解释性文字**而误报（2026-09-30 自证）——
    #    与项目教条"断言须落在会被执行的那段"同源。
    lock_line = next((ln for ln in src.splitlines() if ln.strip().startswith("_LOCK_PATH")), "")
    assert lock_line, "找不到 _LOCK_PATH 赋值（锁的实现被改动了？）"
    assert "gettempdir" not in lock_line, (
        "锁路径不得依赖系统临时目录：重放实例的 TEMP 可能不同 ⇒ 各锁各的 ⇒ 锁失效"
    )
    assert "parents[1]" in lock_line, "锁路径应由 __file__ 推导到仓库根（跨上下文共享的固定位置）"
    # ★ 跨视图兜底：文件锁在"重放实例有独立文件视图"时互不可见 ⇒ 必须有进程表检测
    assert "_other_pytest_running" in src and "psutil" in src, (
        "缺少跨上下文兜底（进程表检测）：文件锁在重放实例里可能失效"
    )


# ============================================================
# 模块 docstring 与 LoopConfig 实际默认值**同源**（2026-10-02）
# ============================================================
def test_loop_module_docstring_matches_real_config_defaults() -> None:
    """★ 模块 docstring 里写的每个默认值，必须等于 `LoopConfig()` 的真实值。

    事故（2026-10-02）：docstring 写 `max_tokens_budget=8000`，实现早已是 **30000**
    ⇒ 读代码的人（和写报告的人）会**按错误数字推理**，且没有任何测试会报警。
    本条就是防止这类"文档悄悄过期"的回归网。

    同时钉住两处**最易误读的口径**（缺了它们，下一个读者还会踩）：
    `max_tokens_budget` 量「单次请求」而非「整段对话」、且 ≠ `max_tokens_per_turn`。
    """
    import sys

    sys.path.insert(0, str(_ROOT / "src"))
    from lightcore.core.loop import LoopConfig

    doc = (_ROOT / "src" / "lightcore" / "core" / "loop.py").read_text(encoding="utf-8")
    cfg = LoopConfig()
    head = doc.split("对外产出")[0]  # 只校验 docstring 段，别把正文里的例子算进来

    for field_name in (
        "max_iterations",
        "max_tool_failures",
        "max_steps_no_progress",
        "max_tokens_budget",
        "max_tokens_per_turn",
        "max_cost_per_turn",
    ):
        real = getattr(cfg, field_name)
        # 允许 30000 / 1.0 / 120.0 这类写法；断言"文档里出现了真实值"
        candidates = {str(real), f"{real:g}", str(int(real)) if real == int(real) else str(real)}
        assert any(f"{field_name}={c}" in head for c in candidates), (
            f"模块 docstring 里 {field_name} 的值已过期：真实默认是 {real}。"
            f"请同步 docstring（它会误导所有读代码的人，且此前无任何测试报警）。"
        )

    # ★ 两处口径必须留在 docstring 里 —— 它们是已踩过的坑，删掉等于让坑复活
    assert "单次请求" in head, "docstring 必须写明 budget 量的是「单次请求」而非「整段对话」"
    assert "≠" in head and "max_tokens_per_turn" in head, (
        "docstring 必须区分 budget（单次）与 per_turn（整轮累计）——曾把 8 万误读成单次上下文"
    )
    assert "provider_settings.json" in head, (
        "docstring 必须提示额度可被 runtime_settings 覆盖，否则「只改代码」会被误认为已生效"
    )
