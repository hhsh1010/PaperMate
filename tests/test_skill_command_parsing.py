"""命令解析（`SkillSpec.matches`）—— 行为测试（2026-10-03 修复）。

## 修的缺陷（见 `docs/诊断-命令功能代码审查-2026-10-03.md`）
原先**只认「命令 + 半角空格 / 全角空格」**两种分隔，实测后果：
- `/精读\\n参数`（**换行**）⇒ 不匹配 —— ⚠️ 输入框是**多行**的，
  "命令后换行写参数"是最**自然**的写法，却静默回落普通 Agent（无任何提示）；
- `/精读\\t参数`（Tab）⇒ 不匹配；
- `/DEEP_READ 参数`（大写）⇒ 不匹配。

⇒ 修为：**任意空白**分隔 + `casefold()` 大小写不敏感。

## ⚠️ 本文件一半的用例是「**必须不命中**」
放宽分隔符的反面风险是**误判**：`/精读论文` 语义歧义，
无条件吞掉会把它当成"命令 `/精读` + 参数 `论文`" —— **比不匹配更危险**
（用户没想调技能，却被塞进技能流程）。⇒ 防误判用例与正向用例**同等重要**。
"""
from __future__ import annotations

import pytest

from lightcore.skills.spec import SkillSpec


def _spec(commands: list[str]) -> SkillSpec:
    async def handler(ctx):  # noqa: ANN001, ANN202 - 测试不执行 handler
        raise NotImplementedError

    return SkillSpec(name="t", description="d", handler=handler, commands=commands)


CMD = ["/精读", "/deep_read"]


class TestSeparators:
    """★ P0-1：命令后的分隔符 —— 从"仅空格"放宽到**任意空白**。"""

    @pytest.mark.parametrize(
        "text, expect_args",
        [
            ("/精读", ""),                      # 纯命令
            ("/精读 参数", "参数"),              # 半角空格（原有能力）
            ("/精读\u3000参数", "参数"),         # 全角空格（原有能力）
            ("/精读\t参数", "参数"),             # ★ Tab（新增）
            ("/精读\n参数", "参数"),             # ★ 换行（新增，最高频场景）
            ("/精读\r\n参数", "参数"),           # ★ CRLF（Windows 粘贴）
            ("/精读\n\n参数", "参数"),           # ★ 多个换行
            ("/精读   多空格", "多空格"),         # ★ 多个空格
            ("  /精读 前导空白", "前导空白"),      # 整串 strip 后的前导空白
        ],
    )
    def test_whitespace_separators_all_match(self, text: str, expect_args: str):
        assert _spec(CMD).matches(text) == expect_args

    def test_args_are_stripped(self):
        """参数两侧空白须去掉（否则技能拿到 "\n参数 " 这种脏输入）。"""
        assert _spec(CMD).matches("/精读 \n 参数 \n") == "参数"


class TestCaseInsensitive:
    """★ P1：大小写不敏感（移动端键盘自动首字母大写、用户手打英文别名时常见）。"""

    @pytest.mark.parametrize(
        "text, expect_args",
        [
            ("/deep_read 参数", "参数"),
            ("/DEEP_READ 参数", "参数"),
            ("/Deep_Read 参数", "参数"),
            ("/DEEP_READ", ""),
            ("/Deep_Read", ""),
        ],
    )
    def test_english_alias_any_case(self, text: str, expect_args: str):
        assert _spec(CMD).matches(text) == expect_args

    def test_mixed_case_with_newline(self):
        """两个修复点叠加（大小写 + 换行）也要成立。"""
        assert _spec(CMD).matches("/DEEP_READ\n参数") == "参数"


class TestNoFalsePositive:
    """★★ 防误判 —— 放宽分隔符**不得**把紧贴写法也吞进来。"""

    @pytest.mark.parametrize(
        "text",
        [
            "/精读论文",      # 命令后紧贴中文：语义歧义，必须不命中
            "/deep_readx",   # 英文别名后紧贴字母
            "/精  读 断裂",    # 命令被拆开
            "/精读, 逗号",     # 命令后跟标点（不是空白）
            "/精读-破折号",
            "/精读：冒号",
        ],
    )
    def test_tight_or_punctuated_forms_do_not_match(self, text: str):
        assert _spec(CMD).matches(text) is None, (
            f"{text!r} 被误判为命令 ⇒ 用户没想调技能却被塞进技能流程（比不匹配更危险）"
        )

    def test_unknown_command_does_not_match(self):
        assert _spec(CMD).matches("/未知命令 x") is None

    def test_plain_text_does_not_match(self):
        assert _spec(CMD).matches("普通文本") is None

    @pytest.mark.parametrize("text", ["", "   ", "/", "/ ", "\n", None])
    def test_degenerate_inputs(self, text):
        assert _spec(CMD).matches(text) is None


class TestCasefoldIndexSafety:
    """★★ 索引必须走**原文**长度 —— 先整体 casefold 会因长度变化切错参数。

    `"ß".casefold() == "ss"`（1 字符 → 2 字符）。若实现写成
    `text.casefold().startswith(cmd.casefold())` 再 `text[len(cmd):]`，
    遇到这类字符就**切错位置**。本用例用一条含 `ß` 的命令把这个不变量钉住。
    """

    def test_len_changing_char_does_not_shift_args(self):
        # ⚠️ 命令本身**不含空格**（`"/ß 命令"` 那种写法是错的：命令不该带空格 → 永远匹配不上）
        spec = _spec(["/ß"])
        # `/ß` 的 casefold 是 `/ss`（长度 2 → 3）；正确实现按原文 len("/ß") == 2 切片
        assert spec.matches("/ß 参数") == "参数"
        assert spec.matches("/ß\n参数") == "参数"

    def test_ascii_and_cjk_lengths_unchanged(self):
        """常用命令（ASCII/中文）casefold 等长 —— 这是能安全比较的前提。"""
        for cmd in ["/精读", "/deep_read", "/综述", "/eval"]:
            assert len(cmd.casefold()) == len(cmd)


class TestRouterIntegration:
    """走 `SkillRouter`（真实入口）确认修复对**路由**生效，而不只是 `matches`。"""

    def test_newline_form_routes_to_skill(self):
        from papermate.skills import default_router

        r = default_router()
        s = r.route("/术语表\n反事实推理", {"skills": True})
        assert s is not None, "换行写法仍未命中 ⇒ 修复没接到路由上"
        assert s.name == "glossary"
        assert r.parse_args("/术语表\n反事实推理", s) == "反事实推理"

    def test_uppercase_alias_routes_to_skill(self):
        from papermate.skills import default_router

        r = default_router()
        s = r.route("/GLOSSARY x", {"skills": True})
        assert s is not None and s.name == "glossary"

    def test_tight_form_still_falls_through(self):
        """回归网：紧贴写法**仍不**被路由（防后人"顺手支持"而引入误判）。"""
        from papermate.skills import default_router

        r = default_router()
        assert r.route("/术语表反事实", {"skills": True}) is None
