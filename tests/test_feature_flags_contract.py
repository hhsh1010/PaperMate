"""前后端功能开关清单的**静态契约**（P-06 问题 4，2026-10-10）。

## 为什么必须有
后端 `DEFAULT_FEATURES`（`runtime_settings.py`）与前端 `FEATURES_DEF`（`SettingsPage.tsx`）
靠人工保持一致（项目纪律"新增开关改 7 处"）。漏改一处的表现：
- 后端有、前端没展示 ⇒ 用户**永远看不见**这个开关；
- 前端有、后端不认 ⇒ 拨了开关保存后**静默丢失**（后端 update 直接忽略未知键）。
两种都是"零报错的静默失效"，只能靠静态契约测试兜（本项目既定做法，同
`test_build_resources_contract.py` 一族）。

## 命名映射
前端键 = `feat` + camelCase（如 `featMultiQuery`），后端键 = snake_case（如 `multi_query`）。
测试内置 camel→snake 转换，**不要求**两边字面相同，只要求**语义键集相等**。
"""

from __future__ import annotations

import re
from pathlib import Path

from papermate.runtime_settings import DEFAULT_FEATURES

SETTINGS_TS = Path(__file__).resolve().parents[1] / "web" / "src" / "pages" / "SettingsPage.tsx"


def _camel_to_snake(name: str) -> str:
    """`MultiQuery` → `multi_query`（feat 前缀已在提取时剥掉）。"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _frontend_feature_keys() -> set[str]:
    src = SETTINGS_TS.read_text(encoding="utf-8")
    keys = re.findall(r'\{ key: "feat([A-Za-z]+)",', src)
    return {_camel_to_snake(k) for k in keys}


def test_frontend_covers_every_backend_flag():
    """后端每个开关都必须在前端有展示位（否则用户永远看不见）。"""
    missing = set(DEFAULT_FEATURES) - _frontend_feature_keys()
    assert not missing, f"后端有、前端没展示的开关：{sorted(missing)}"


def test_frontend_has_no_unknown_flag():
    """前端展示的开关都必须被后端认识（否则拨了也白拨，静默丢失）。"""
    unknown = _frontend_feature_keys() - set(DEFAULT_FEATURES)
    assert not unknown, f"前端展示了后端不认的开关：{sorted(unknown)}"


def test_key_sets_are_exactly_equal():
    """双向都空还不够 —— 键集必须**完全相等**（防两处各自多/漏后正好互相抵消的巧合）。"""
    assert _frontend_feature_keys() == set(DEFAULT_FEATURES)


def test_every_group_is_declared():
    """FEATURES_DEF 引用的 group 必须都在 FEATURE_GROUPS 里声明（分组名打错 ⇒ 开关整组消失）。"""
    src = SETTINGS_TS.read_text(encoding="utf-8")
    declared = set(re.findall(r'\{ key: "(\w+)", title:', src))
    used = set(re.findall(r'group: "(\w+)"', src))
    dangling = used - declared
    assert not dangling, f"FEATURES_DEF 用了未声明的分组：{sorted(dangling)}"
