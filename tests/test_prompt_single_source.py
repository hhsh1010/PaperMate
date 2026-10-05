"""LightCore 提示词单一来源（C7-lightcore，2026-09-23）。

## 这份测试要守四件事

1. **逐字未改** —— 搬运提示词只许挪位置，不许改字。基线哈希锁定，改一个字就红。
2. **单一来源** —— 提示词常量只许定义在 `lightcore/prompts/`，别处不许再定义同名常量。
3. **★ 提示词与解析代码的「合约成对」** —— 这是本项目最容易静默失效的一类：
   提示词要求模型"第一行输出 SATISFIED"，而解析代码找的也是 `SATISFIED`。
   **只改一边 → 功能静默失效**（永远判 satisfied / critique 恒空），且不报错。
   同理 `反思: …` / `改进: …` 一对。测试把"标记确实写在提示词里"钉住。
4. **★ 层方向** —— `lightcore` 是内核层、`papermate` 是业务层。
   提示词两包**并列**而非合并，正是因为把内核提示词搬进上层包会造成**层倒置**。
   这条约束此前只写在注释里，这里升级为可执行断言。
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
LIGHTCORE = SRC / "lightcore"
PROMPTS_PKG = LIGHTCORE / "prompts"

#: 搬入 `lightcore/prompts` 时的**逐字基线**（sha256 全量 + 长度）。
#: 值取自搬运**之前**的实测值（见 docs/实施记录.md 的 C7-lightcore 小节）。
#: 改动提示词内容时**必须同步更新这里并 bump `PROMPT_VERSION`** —— 这正是这张表的用途。
BASELINE = {
    "SUMMARY_PROMPT": (
        62,
        "8ad5120d8960d9330f8985ae26be515be7320e2c31399ea6b45a95d1701ad8d5",
    ),
    "PLANNER_PROMPT": (
        78,
        "2f58afb2471607baf6dcd0aa26de361abb2e5e6c1cfa9ff981bcd03f920eb497",
    ),
    "REFLECT_PROMPT": (
        104,
        "aae7790da105da229315dfcba982660222820ba00fc140ffb5fd7d913ad4cc77",
    ),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


# --------------------------- 1. 逐字未改 ---------------------------


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_prompt_text_is_byte_identical_to_baseline(name):
    """★ 搬运只许挪位置、不许改字 —— 本项目有过"提示词一改行为就变"的先例。"""
    import lightcore.prompts as P

    val = getattr(P, name)
    exp_len, _ = BASELINE[name]
    # 长度先断（读失败时报错更直观），哈希后断（更严格）
    assert len(val) == exp_len, f"{name} 长度变了：{len(val)} != {exp_len}（提示词被改动过？）"
    assert _sha(val) == BASELINE[name][1], (
        f"{name} 内容变了（sha256 不符）。若确为有意改动：请同步更新本测试的 BASELINE "
        f"并 bump lightcore.prompts.PROMPT_VERSION。"
    )


# --------------------------- 2. 单一来源 ---------------------------


def test_prompt_constants_defined_only_in_prompts_package():
    """★ 提示词常量只许定义在 `lightcore/prompts/` —— 别处再定义就是"改一处漏一处"的开始。"""
    offenders: list[str] = []
    for p in _py_files(LIGHTCORE):
        if PROMPTS_PKG in p.parents:
            continue
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = ln.strip()
            if s.startswith(("SUMMARY_PROMPT =", "PLANNER_PROMPT =", "REFLECT_PROMPT =")):
                offenders.append(f"{p.relative_to(SRC)}:{i}")
    assert not offenders, f"提示词常量在 prompts 包之外被重新定义：{offenders}"


def test_prompts_package_exposes_a_version():
    """版本号必须存在且非空 —— 否则"这句提示词是哪一版"无从回答。"""
    from lightcore.prompts import PROMPT_VERSION

    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION.strip()


# --------------------------- 3. ★ 合约成对 ---------------------------


def test_reflect_markers_are_declared_in_the_prompt():
    """★★ 解析用的标记必须**确实写在提示词里**。

    这是「提示词 ↔ 解析代码」的合约闭合点：
    - 提示词说"第一行输出 SATISFIED"，解析代码才去找 `SATISFIED`；
    - 提示词说"写'反思: …'"和"'改进: …'"，解析代码才按 `反思` / `改进` 取行。
    若有人只改提示词（比如把 SATISFIED 换成 OK）而没改解析，
    反思会**静默失效**（永远判 satisfied），本断言会当场报红。
    """
    from lightcore.prompts import (
        REFLECT_CRITIQUE_LABEL,
        REFLECT_IMPROVEMENT_LABEL,
        REFLECT_NEEDS_IMPROVEMENT,
        REFLECT_PROMPT,
        REFLECT_SATISFIED,
    )

    for token in (
        REFLECT_SATISFIED,
        REFLECT_NEEDS_IMPROVEMENT,
        REFLECT_CRITIQUE_LABEL,
        REFLECT_IMPROVEMENT_LABEL,
    ):
        assert token in REFLECT_PROMPT, (
            f"标记 {token!r} 被解析代码使用，但提示词里已不包含它 —— "
            "提示词与解析的合约已断裂（反思会静默失效）。"
        )


def test_reflect_parser_uses_constants_not_literals():
    """★ 反向守卫：解析代码不得再出现字面串（否则常量形同虚设，仍会各改一边）。"""
    src = (LIGHTCORE / "planning" / "reflector.py").read_text(encoding="utf-8")
    for literal in ('startswith("SATISFIED")', 'startswith("反思")', 'startswith("改进")'):
        assert literal not in src, f"解析代码仍在用字面串 {literal}，应改为引用 prompts 常量"


def test_reflect_satisfied_detection_still_works():
    """端到端补一刀：按提示词规定的格式回应，解析必须判对。

    这条**不只断言常量**，而是把"提示词规定的格式"真的喂给解析器跑一遍 ——
    常量对得上、但解析逻辑被改坏的情况也能拦住。
    """
    from lightcore.llm.base import ChatMessage, LLMResponse, Role
    from lightcore.planning.reflector import Reflector
    from lightcore.prompts import REFLECT_NEEDS_IMPROVEMENT, REFLECT_SATISFIED

    class _Scripted:
        """最小脚本化 LLM：只实现 `Reflector` 用到的那一个方法。"""

        def __init__(self, reply: str) -> None:
            self._reply = reply

        async def chat(self, messages, **_kw):
            return LLMResponse(message=ChatMessage(role=Role.ASSISTANT, content=self._reply))

    ok = asyncio.run(Reflector(_Scripted(REFLECT_SATISFIED)).reflect("目标", "结果"))
    assert ok.satisfied is True

    bad = asyncio.run(
        Reflector(
            _Scripted(f"{REFLECT_NEEDS_IMPROVEMENT}\n反思: 缺少引用\n改进: 补引用")
        ).reflect("目标", "结果")
    )
    assert bad.satisfied is False
    assert bad.critique == "缺少引用" and bad.improvement == "补引用"


# --------------------------- 4. ★ 层方向 ---------------------------


def test_lightcore_does_not_import_papermate():
    """★★ 内核层不得依赖业务层 —— 这是"提示词两包并列而非合并"的**根本理由**。

    若将来有人图省事把 lightcore 的提示词搬进 `papermate/prompts/`，就会引入
    `lightcore → papermate` 的依赖（层倒置）。本断言守住这条边界；
    真要合并时，必须先解决层方向问题，而不是绕过测试。
    """
    offenders: list[str] = []
    for p in _py_files(LIGHTCORE):
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = ln.strip()
            if s.startswith(("import papermate", "from papermate")):
                offenders.append(f"{p.relative_to(SRC)}:{i}: {s[:60]}")
    assert not offenders, f"lightcore 出现了对 papermate 的依赖（层倒置）：{offenders}"


def test_guidance_mark_is_not_duplicated_as_literal():
    """收口引导标记必须来自常量 —— 此前测试自己又写了一遍，改词就静默走错分支。"""
    hits: list[str] = []
    for base in (LIGHTCORE, Path(__file__).resolve().parent):
        for p in _py_files(base):
            if p == PROMPTS_PKG / "library.py":
                continue
            for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if "上一版回答的不足" in ln and "重定义" not in ln:
                    hits.append(f"{p.name}:{i}")
    assert not hits, f"引导标记仍以字面串出现（应引用 GUIDANCE_MARK）：{hits}"
