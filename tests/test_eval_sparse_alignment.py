"""评测口径对齐守护（T-00 步 1）。

**守的是什么**：`eval/run_eval.py` 里 `use_bge_sparse` 的取值必须**来自运行时设置**
（与线上同源），而不是"按 profile 硬编码猜一个"。

**修复前的真实缺陷**（2026-09-23 定位）：:

    use_bge_sparse=(self.profile == "prod")   # ← 硬编码
    # 而同一个类的 docstring 声称：
    use_bge_sparse=rt["use_bge_sparse"]       # ← 声称与线上同源

**文档与代码不一致**。当时线上恰好也是 `True`，所以数字是对的（**纯属巧合**）；
但只要设置页一关掉稀疏向量，评测会**继续按"开着"跑且不报任何异常** ——
最终产出"停用稀疏不影响质量"这类**建立在错误前提上的结论**。

这类缺陷的危险性在于**它永远不报错**：评测照跑、指标照出、报告照存。
唯一能挡住它的是"断言取值来源"的测试。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import run_eval  # noqa: E402


def _patch_state(monkeypatch, fn) -> None:
    """把 `papermate.runtime_settings.get_state` 替换为给定实现。

    ⚠️ **不能用 `monkeypatch.setitem(sys.modules, "papermate.runtime_settings", fake)`**：
    被测代码写的是 `from papermate import runtime_settings`，而该语句**优先取包的属性**
    （`papermate.runtime_settings`）。本文件**单独跑**时该子模块尚未被 import，改 `sys.modules` 有效；
    但**全量跑时其它测试早已 import 过** → 包属性已存在 → **patch 不生效**。

    这正是「**环境依赖测试**」的典型来源，本文件第一版即如此：
    **单独跑 8 passed、全量跑 4 failed**。改为直接 patch 目标**属性**
    （点分路径由 pytest 负责解析与 import），与模块缓存状态无关，两种场景行为一致。
    """
    monkeypatch.setattr("papermate.runtime_settings.get_state", fn)


def _bge_sparse_call_kwargs(src: str) -> list[str]:
    """提取源码中所有 `use_bge_sparse=<expr>` **关键字实参**的表达式文本。

    用 AST 而非字符串匹配：注释、docstring 里的示例代码不参与 —— 否则"文档里引用修复前的
    错误写法"会被误判成违规（第一版测试即因此失败）。
    """
    out: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "use_bge_sparse":
                    out.append(ast.unparse(kw.value))
    return out


def test_auto_follows_runtime_settings_true(monkeypatch):
    """auto 跟随运行时设置（True）。"""
    _patch_state(monkeypatch, lambda: {"use_bge_sparse": True})
    val, src = run_eval._resolve_sparse("auto")
    assert val is True
    assert "runtime_settings" in src


def test_auto_follows_runtime_settings_false(monkeypatch):
    """★★ 核心反向断言：运行时设置关掉 → 评测必须跟着关掉。

    这条才是真正拦住那个缺陷的断言 —— 修复前取值是硬编码 `profile == "prod"`，
    在 prod 下**恒为 True**、与配置无关。所以"配置为 False 时能返回 False"
    就是"取值确实来自配置"的证明。
    """
    _patch_state(monkeypatch, lambda: {"use_bge_sparse": False})
    val, src = run_eval._resolve_sparse("auto")
    assert val is False
    assert "runtime_settings" in src


def test_explicit_on_off_override_config(monkeypatch):
    """显式覆盖优先于配置（双臂对照与"同配置可复现复跑"依赖它）。"""
    _patch_state(monkeypatch, lambda: {"use_bge_sparse": False})
    assert run_eval._resolve_sparse("on")[0] is True
    _patch_state(monkeypatch, lambda: {"use_bge_sparse": True})
    assert run_eval._resolve_sparse("off")[0] is False
    # 覆盖时必须标明"来自命令行"，否则报告无法区分"配置本来就是 True"与"本次强制开"
    assert "显式覆盖" in run_eval._resolve_sparse("on")[1]


def test_unreadable_config_does_not_raise_and_says_so(monkeypatch):
    """★ 配置读不到时**不得抛异常**（评测不该因设置文件损坏而整体中断），
    但来源必须写明失败原因 —— 静默回退正是"两套口径"的温床。"""

    def _boom():
        raise OSError("settings 文件不可读")

    _patch_state(monkeypatch, _boom)
    val, src = run_eval._resolve_sparse("auto")
    assert val is False, "读不到配置时回退 RAGPipeline 默认值 False"
    assert "不可读" in src, "必须显式说明回退原因，不能只写'默认值'了事"


def test_missing_key_is_reported(monkeypatch):
    """键缺失同样是"来源不明"，必须写出来（而不是当成正常默认）。"""
    _patch_state(monkeypatch, lambda: {})
    val, src = run_eval._resolve_sparse("auto")
    assert val is False
    assert "缺该键" in src


def test_factory_uses_injected_value_not_profile():
    """★ 行为守卫：`PipelineFactory` 必须听注入值，不得再按 profile 推断。

    本次缺陷的核心就是"prod profile ⇒ 稀疏恒开"。故此处专测**两种 profile 都被注入值覆盖**。
    """
    f_prod = run_eval.PipelineFactory("fake", "prod", use_bge_sparse=False, sparse_source="t")
    assert f_prod.use_bge_sparse is False, "prod profile 下也必须听注入值"
    assert f_prod.sparse_source == "t"

    f_fast = run_eval.PipelineFactory("fake", "fast", use_bge_sparse=True, sparse_source="t")
    assert f_fast.use_bge_sparse is True, "fast profile 下也听注入值"


def test_factory_legacy_none_keeps_old_behavior():
    """向后兼容：`use_bge_sparse` 缺省（None）时沿用旧行为，且**来源要写明是旧行为** ——
    否则"没传值"与"传了 False"在报告里长得一样，审计时分不清。"""
    f = run_eval.PipelineFactory("fake", "prod")
    assert f.use_bge_sparse is True
    assert "旧行为" in f.sparse_source


def test_source_string_guard_against_hardcoding():
    """★ 源码反例守卫：禁止"按 profile 推断稀疏"的写法回归。

    行为断言覆盖不到所有路径（例如将来有人在第三处建管线时又写回硬编码），
    故保留一条结构断言 —— 与本项目 `test_verifier.py` 的做法一致。

    ⚠️ **必须用 `ast` 而不是字符串匹配**：本模块的 docstring 里**引用了修复前的代码**，
    字符串断言会把文档误判成违规（第一版就是这么失败的）。
    `ast` 只看真实的调用实参，注释与文档一概不参与 —— 这也是"结构断言"的通用正解。
    """
    src = (_EVAL_DIR / "run_eval.py").read_text(encoding="utf-8")
    kwargs = _bge_sparse_call_kwargs(src)
    assert kwargs, "未找到任何 `use_bge_sparse=` 实参 —— 扫描器失效（前几条断言会因此假绿）"
    assert all("profile" not in v for v in kwargs), (
        f"不得按 profile 推断稀疏向量取值；实际写法：{kwargs}"
    )
    # ⚠️ **2026-09-26 更正（T-04）**：原断言写死 `count("self.use_bge_sparse") == 2`，
    # T-04 新增 composite 分支（第三处建管线）后变为 3 → 转红。
    # 但**数量本来就不是本质**：本质是「除工厂注入处外，每处建管线都必须用注入值」。
    # 改为按语义断言：将来再新增管线不会误报，而真有人写回硬编码仍会被抓住。
    injected = [v for v in kwargs if v == "self.use_bge_sparse"]
    others = [v for v in kwargs if v != "self.use_bge_sparse"]
    assert injected, (
        "未找到任何使用 `self.use_bge_sparse` 的建管线调用 —— 扫描器失效或有人改回了硬编码"
    )
    assert others == ["sparse_on"], (
        "除 `PipelineFactory(...)` 的外部注入处外，每处建管线都必须使用 `self.use_bge_sparse`；"
        f"实际写法：{kwargs}"
    )
    # 报告必须自证口径：否则事后无法回答"这版评测的 sparse 开没开、值从哪来"
    assert '"use_bge_sparse": factory.use_bge_sparse' in src
    assert '"use_bge_sparse_source": factory.sparse_source' in src
