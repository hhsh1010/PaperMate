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
    describe_llm_error,
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


# ------------------------------------------------------------------ 模型调用失败的人话映射

class _Resp:
    def __init__(self, status: int, text: str = "") -> None:
        self.status_code = status
        self.text = text


class _HTTPErr(Exception):
    """模拟 httpx.HTTPStatusError（有 .response）。"""

    def __init__(self, status: int, text: str = "") -> None:
        super().__init__(f"HTTP {status}")
        self.response = _Resp(status, text)


def test_describe_401_key_invalid_and_masked() -> None:
    """★ 401 ⇒ 密钥无效；且**上游回显的 key 必须打码**（不能把密钥发到前端/日志）。"""
    code, msg = describe_llm_error(
        _HTTPErr(401, '{"error":{"message":"Incorrect API key provided: sk-abcdef123456"}}')
    )
    assert code == 400
    assert "密钥" in msg
    assert "sk-abcdef123456" not in msg
    assert "sk-***" in msg


def test_describe_402_balance() -> None:
    code, msg = describe_llm_error(_HTTPErr(402, '{"error":{"message":"Insufficient Balance"}}'))
    assert code == 400
    assert "余额" in msg and "Insufficient Balance" in msg


def test_describe_403_quota() -> None:
    code, msg = describe_llm_error(_HTTPErr(403, "free quota exhausted"))
    assert code == 400
    assert "额度" in msg


def test_describe_429_rate_limit() -> None:
    """429 是**可重试**的 ⇒ 状态码原样透出 429（前端可提示"稍后再试"）。"""
    code, msg = describe_llm_error(_HTTPErr(429))
    assert code == 429 and "限流" in msg


def test_describe_5xx_is_502() -> None:
    code, msg = describe_llm_error(_HTTPErr(503, "service unavailable"))
    assert code == 502 and "503" in msg


def test_describe_network_error_without_response() -> None:
    code, msg = describe_llm_error(RuntimeError("connection reset"))
    assert code == 502
    assert "RuntimeError" in msg or "网络" in msg


# ------------------------------------------------------------------ 解析器的形状容忍（线上实测扩展）

def test_parse_plain_string_list() -> None:
    """纯字符串数组：**长度恰好等于 expected** ⇒ 按位置对位（顺序此时可信）。"""
    assert parse_titles_json('["甲", "乙"]', 2) == ["甲", "乙"]


def test_parse_plain_string_list_wrong_len_refused() -> None:
    """★ 长度对不上就**全部不认**（猜位置等于张冠李戴）。"""
    assert parse_titles_json('["甲"]', 2) == [None, None]
    assert parse_titles_json('["甲", "乙", "丙"]', 2) == [None, None]


def test_parse_index_key_alias() -> None:
    """模型常写 index / n / id，不只是 i。"""
    assert parse_titles_json('[{"index":1,"zh":"甲"},{"n":2,"zh":"乙"}]', 2) == ["甲", "乙"]


def test_parse_value_key_aliases() -> None:
    """翻译值的键名不统一：translation / title_zh / cn 都要认。"""
    raw = '[{"i":1,"translation":"甲"},{"i":2,"title_zh":"乙"},{"i":3,"cn":"丙"}]'
    assert parse_titles_json(raw, 3) == ["甲", "乙", "丙"]


def test_parse_wrapper_variants() -> None:
    """外层包装常见四种。"""
    for key in ("items", "translations", "results", "titles"):
        raw = f'{{"{key}": [{{"i":1,"zh":"甲"}}]}}'
        assert parse_titles_json(raw, 1) == ["甲"], key
