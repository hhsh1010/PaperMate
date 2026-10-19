"""论文标题中译的判据（**离线**，不打真实模型）。

重点覆盖三类真实故障：
1. 模型把 JSON 包在 ```json 里 / 前后加话；
2. 条数对不上 / 顺序错位 —— **绝不能张冠李戴**（错位比不翻译更糟）；
3. 模型压根没配（fake）—— 必须显式拦住，而不是返回一堆占位文本。
"""

from __future__ import annotations

from papermate.services.paper_translate import (
    MAX_TITLES,
    build_messages,
    is_fake_model,
    parse_titles_json,
)


def test_build_messages_numbers_from_one() -> None:
    msgs = build_messages(["Alpha", "Beta"])
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "1. Alpha" in user and "2. Beta" in user      # 编号从 1 开始，便于回填


def test_parse_plain_json() -> None:
    raw = '[{"i":1,"zh":"甲"},{"i":2,"zh":"乙"}]'
    assert parse_titles_json(raw, 2) == ["甲", "乙"]


def test_parse_json_in_code_fence() -> None:
    raw = "好的，结果如下：\n```json\n[{\"i\":1,\"zh\":\"甲\"}]\n```\n"
    assert parse_titles_json(raw, 1) == ["甲"]


def test_parse_dict_wrapper() -> None:
    raw = '{"items":[{"i":1,"zh":"甲"}]}'
    assert parse_titles_json(raw, 1) == ["甲"]


def test_parse_by_index_not_by_order() -> None:
    """★ 模型把顺序打乱时，必须**按 i 回填**，不能按出现顺序 —— 否则标题张冠李戴。"""
    raw = '[{"i":2,"zh":"乙"},{"i":1,"zh":"甲"}]'
    assert parse_titles_json(raw, 2) == ["甲", "乙"]


def test_parse_missing_items_become_none() -> None:
    """少一条 ⇒ 该位置 None（前端保留英文原文），**不要**用相邻的补上。"""
    raw = '[{"i":1,"zh":"甲"}]'
    assert parse_titles_json(raw, 3) == ["甲", None, None]


def test_parse_garbage_returns_all_none() -> None:
    assert parse_titles_json("抱歉，我无法完成", 2) == [None, None]
    assert parse_titles_json("", 2) == [None, None]


def test_parse_zero_expected() -> None:
    assert parse_titles_json("[{\"i\":1,\"zh\":\"甲\"}]", 0) == []


def test_is_fake_model() -> None:
    assert is_fake_model("fake") and is_fake_model("") and is_fake_model(None)
    assert not is_fake_model("deepseek-chat")


def test_max_titles_kept_in_sync_with_page_size() -> None:
    """一页检索 20 条 ⇒ 一次翻译上限也应是 20（两个数字分开写会漂移）。"""
    assert MAX_TITLES == 20
