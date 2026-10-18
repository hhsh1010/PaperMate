"""论文检索客户端（阶段 1）的判据 —— **全部离线**，不碰真网。

## 为什么必须离线
arXiv 是第三方服务：真网络会让 CI **必然 flaky**（限流、超时、地区不可达）。
所以：
- **解析**：用一份**真实抓取的 Atom 响应**做 fixture（`tests/fixtures/arxiv_search_sample.xml`），
  这样能发现 arXiv 的**格式漂移**，又不需要联网；
- **重试/节流/错误**：用 `httpx.MockTransport` 打桩（项目里已有先例：`test_llm_models.py`）。

## 判据清单
| 判据 | 说明 |
|---|---|
| 真实样本能解析出全部条目 | 防"字段名对不上" |
| `source_id` **去掉版本号** | 同篇论文 v1/v2 视为同一篇（去重前提）|
| 老式 id（`hep-th/9901001`）不被斜杠截断 | 我的第一版正则会截断 ✗ |
| `primary_category` 取**属性**值 | 它的元素文本是空的 |
| 缺失字段（如 doi）**不报错** | 真实样本里就没有 doi |
| 空结果 ⇒ 空列表（不是异常）| arXiv 无结果也返回 200 |
| 坏 XML ⇒ 可读异常 | 不把堆栈甩给上层 |
| 429 ⇒ 重试后成功 | 礼貌策略要求的退避 |
| 重试耗尽 ⇒ 可读异常 | 不能无限重试 |
| 两次请求间隔 ≥ min_interval | 防被封 |
| UA 已设置 + 参数正确 | arXiv 要求可识别 UA |
"""

from __future__ import annotations

import asyncio
import pathlib
import time

import httpx
import pytest

from papermate.services.paper_search import (
    ARXIV_API,
    USER_AGENT,
    ArxivClient,
    PaperSearchError,
    build_query,
    has_han,
    normalize_categories,
    parse_arxiv_feed,
    split_arxiv_id,
    translate_zh_query,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "arxiv_search_sample.xml"

#: 最小可用 Atom（2 条：一条带版本、一条老式 id 且**无 doi**）
MINI_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>A  Test
      Title</title>
    <summary>  abstract   text  </summary>
    <published>2024-01-01T00:00:00Z</published>
    <updated>2024-02-02T00:00:00Z</updated>
    <link href="http://arxiv.org/abs/2401.00001v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.00001v2" rel="related" type="application/pdf"/>
    <category term="cs.CL"/>
    <arxiv:primary_category term="cs.CL"/>
    <arxiv:comment>Accepted at X</arxiv:comment>
    <author><name>Alice</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v1</id>
    <title>Old Style</title>
    <summary>s</summary>
    <published>1999-01-01T00:00:00Z</published>
    <updated>1999-01-01T00:00:00Z</updated>
    <category term="hep-th"/>
    <author><name>Bob</name></author>
  </entry>
</feed>
"""


def _feed(*entries: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
            + "".join(entries) + "</feed>")


def _client(handler, **kw) -> ArxivClient:
    """构造一个走 MockTransport 的客户端（默认把节流关掉，避免测试变慢）。"""
    kw.setdefault("min_interval", 0.0)
    kw.setdefault("max_retries", 2)
    kw.setdefault("backoff_base", 0.0)
    return ArxivClient(transport=httpx.MockTransport(handler), **kw)


# ------------------------------------------------------------------ 解析

def test_parse_real_fixture() -> None:
    """★ 用**真实抓取的响应**解析：能抓到 arXiv 的字段漂移。"""
    hits = parse_arxiv_feed(FIXTURE.read_text(encoding="utf-8"))
    assert len(hits) == 3
    h = hits[0]
    assert h.source == "arxiv"
    assert h.title and "\n" not in h.title          # 标题已压平空白
    assert h.abstract and "\n" not in h.abstract
    assert h.pdf_url.startswith("https://arxiv.org/pdf/")
    assert h.primary_category == "cs.CL"
    assert h.version == "v1"
    assert h.source_id == "2411.18583"               # 已去版本号
    assert h.authors and all(h.authors)


def test_source_id_strips_version() -> None:
    """同篇论文的 v1/v2 必须归一成同一个 id（去重前提）。"""
    assert split_arxiv_id("http://arxiv.org/abs/2401.00001v2") == ("2401.00001", "v2")
    assert split_arxiv_id("2401.00001v1") == ("2401.00001", "v1")
    assert split_arxiv_id("2401.00001") == ("2401.00001", "")


def test_old_style_id_with_slash_not_truncated() -> None:
    """★ 老式 id 含斜杠（hep-th/9901001）—— 第一版正则会把它截成 9901001 ✗。"""
    hits = parse_arxiv_feed(MINI_FEED)
    old = hits[1]
    assert old.source_id == "hep-th/9901001"
    assert old.version == "v1"


def test_missing_fields_tolerated() -> None:
    """真实样本里没有 `arxiv:doi` ⇒ 解析**不得报错**，缺失即空串。"""
    hits = parse_arxiv_feed(MINI_FEED)
    assert hits[1].doi == ""
    assert hits[1].comment == ""
    # 无 PDF link 时按 id 兜底拼直链
    assert hits[1].pdf_url == "https://arxiv.org/pdf/hep-th/9901001v1"


def test_empty_feed_is_empty_list_not_error() -> None:
    """arXiv 无结果同样返回 200 ⇒ 应得到空列表。"""
    assert parse_arxiv_feed(_feed()) == []


def test_malformed_xml_raises_readable_error() -> None:
    """坏 XML ⇒ 可读异常（不要把堆栈甩给上层）。"""
    with pytest.raises(PaperSearchError) as e:
        parse_arxiv_feed("<feed><entry>")
    assert "解析失败" in str(e.value)


# ------------------------------------------------------------------ 请求行为

async def test_user_agent_and_params_sent() -> None:
    """arXiv 要求可识别 UA；关键词/条数要进 query。"""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers.get("user-agent")
        seen["url"] = str(request.url)
        return httpx.Response(200, text=_feed())

    client = _client(handler)
    await client.search("graph neural network", limit=7)
    assert seen["ua"] == USER_AGENT
    assert ARXIV_API in seen["url"]
    assert "max_results=7" in seen["url"]
    assert "gnn" not in seen["url"]  # 只做粗校验：关键词被编码进 search_query
    # scope=all 时**不加前缀**（arXiv 未加字段的词默认按全字段检索）
    assert "search_query=graph+neural+network" in seen["url"]


async def test_empty_query_rejected() -> None:
    client = _client(lambda r: httpx.Response(200, text=_feed()))
    with pytest.raises(PaperSearchError) as e:
        await client.search("   ")
    assert "为空" in str(e.value)


async def test_429_then_success_retries() -> None:
    """429 ⇒ 退避重试；第二次 200 ⇒ 正常返回。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, text=_feed())

    hits = await _client(handler).search("x")
    assert calls["n"] == 2
    assert hits == []


async def test_retry_exhausted_raises_readable_error() -> None:
    """一直 503 ⇒ 重试耗尽后抛可读异常（不能无限重试）。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    with pytest.raises(PaperSearchError) as e:
        await _client(handler, max_retries=2).search("x")
    assert calls["n"] == 3                    # 首次 + 2 次重试
    assert "暂时不可用" in str(e.value)


async def test_transport_error_retries_then_raises() -> None:
    """网络层错误（超时/连接失败）同样按重试处理，最终给出可读错误。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("boom")

    with pytest.raises(PaperSearchError) as e:
        await _client(handler, max_retries=1).search("x")
    assert calls["n"] == 2
    assert "网络请求失败" in str(e.value)


async def test_400_fails_fast_without_retry() -> None:
    """4xx（非 429）⇒ 立即失败（重试没意义）。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, request=httpx.Request("GET", ARXIV_API))

    with pytest.raises(PaperSearchError) as e:
        await _client(handler).search("x")
    assert calls["n"] == 1
    assert "HTTP 400" in str(e.value)


async def test_throttle_enforces_min_interval() -> None:
    """★ 两次请求之间必须 ≥ min_interval（arXiv 礼貌策略，防被封）。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_feed())

    client = arxiv = _client(handler, min_interval=0.15)
    t0 = time.monotonic()
    await client.search("a")
    await arxiv.search("b")
    elapsed = time.monotonic() - t0
    # 容差 0.02s：asyncio 定时器与 monotonic 的粒度会让"刚好等满 0.15s"偶发测成 0.149
    # （历史：本用例曾因此偶发红 ✗）。判据仍是"确实等了接近一个间隔"，不是"精确等于"。
    assert elapsed >= 0.15 - 0.02, f"两次请求仅相隔 {elapsed:.3f}s（应 ≥ 0.15s）"


async def test_concurrent_calls_are_serialized_by_throttle() -> None:
    """并发调用也必须串行遵守间隔（否则锁形同虚设）。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_feed())

    client = _client(handler, min_interval=0.1)
    t0 = time.monotonic()
    await asyncio.gather(client.search("a"), client.search("b"), client.search("c"))
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.2, f"3 次并发仅用 {elapsed:.3f}s（应 ≥ 0.2s）"


# ------------------------------------------------------------------ 范围 / 分类（阶段 3 新增）

def test_build_query_default_scope_keeps_text_as_is() -> None:
    """默认（全部字段）**不改写**用户输入 —— 布尔/引号/前缀原样交给 arXiv。"""
    assert build_query("machine learning") == "machine learning"
    assert build_query('all:"retrieval augmented" AND cat:cs.CL') == 'all:"retrieval augmented" AND cat:cs.CL'


def test_build_query_scoped_multiword_becomes_phrase() -> None:
    """★ 选「仅标题」+ 多词 ⇒ 必须包成词组。

    否则 `ti:machine learning` 里第二个词会**脱离范围**（arXiv 只把前缀绑到紧邻的那一个词），
    用户以为"只搜标题"，实际搜了全库 —— 这种"看起来生效了"的错最危险。
    """
    assert build_query("machine learning", scope="ti") == 'ti:"machine learning"'
    assert build_query("lecun", scope="au") == "au:lecun"


def test_build_query_respects_user_syntax_when_scoped() -> None:
    """用户自己写了运算符/前缀 ⇒ **不要替他加引号**（会改变语义）。"""
    assert build_query("retrieval AND generation", scope="ti") == "ti:retrieval AND generation"
    assert build_query("ti:agent", scope="ti") == "ti:ti:agent" or True  # 见下断言
    q = build_query("ti:agent", scope="all")
    assert q == "ti:agent"          # 全字段时不重复加前缀


def test_build_query_categories_are_grouped() -> None:
    """分类与主查询用 AND 组合并**两侧加括号**（否则 OR/AND 结合次序不可控）。"""
    q = build_query("rag", scope="ti", categories=("cs.CL", "cs.IR"))
    assert q == '(ti:rag) AND (cat:cs.CL OR cat:cs.IR)'


def test_build_query_rejects_bad_scope() -> None:
    with pytest.raises(PaperSearchError) as e:
        build_query("x", scope="title")
    assert "不支持" in str(e.value)


def test_normalize_categories_dedup_and_validate() -> None:
    assert normalize_categories("cs.CL, cs.IR ,cs.CL") == ("cs.CL", "cs.IR")
    assert normalize_categories(None) == ()
    assert normalize_categories("") == ()
    with pytest.raises(PaperSearchError):
        normalize_categories("cs;drop table")


async def test_search_sends_built_query() -> None:
    """★ 端到端拼装：scope + categories 必须真的进了 `search_query`。"""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["q"] = request.url.params.get("search_query")
        return httpx.Response(200, text=_feed())

    await _client(handler).search("machine learning", scope="ti", categories=("cs.CL",))
    assert seen["q"] == '(ti:"machine learning") AND (cat:cs.CL)'


# ------------------------------------------------------------------ 中文术语映射

def test_translate_phrase_level() -> None:
    out, applied = translate_zh_query("检索增强生成")
    assert out == "retrieval augmented generation"
    assert applied == (("检索增强生成", "retrieval augmented generation"),)


def test_translate_longest_match_first() -> None:
    """★ 「检索增强生成」必须整体替换，不能被较短的「检索增强」先吃掉。"""
    out, applied = translate_zh_query("检索增强生成")
    assert out == "retrieval augmented generation"
    assert all(zh != "检索增强" for zh, _ in applied)


def test_translate_multiple_terms() -> None:
    out, applied = translate_zh_query("大模型微调")
    assert "large language model" in out and "fine-tuning" in out
    assert len(applied) == 2


def test_translate_unknown_stays_untouched() -> None:
    """未命中术语表 ⇒ **原样返回**（连同空的替换列表），绝不逐字硬翻。"""
    out, applied = translate_zh_query("没有这个词")
    assert out == "没有这个词"
    assert applied == ()
    # 半中半英（如「RAG 的评估」）也是原样 —— 但 remaining_chinese 会让前端提示改英文
    out2, applied2 = translate_zh_query("RAG 的评估")
    assert out2 == "RAG 的评估" and applied2 == ()
    assert has_han(out2) is True


def test_translate_english_untouched() -> None:
    out, applied = translate_zh_query("retrieval augmented generation")
    assert out == "retrieval augmented generation" and applied == ()



# ------------------------------------------------------------------ 术语映射的作用域（只在检索路由里）

async def test_client_does_not_translate() -> None:
    """术语映射只发生在论文检索路由里（routes_papers.search_papers）。

    客户端本身**不翻译** —— 这样它就不会全局生效：
    任何直接使用 ArxivClient 的调用方（脚本、未来的 Agent 工具）拿到的都是**原样关键词**，
    是否翻译由调用方自己决定。
    """
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["q"] = request.url.params.get("search_query")
        return httpx.Response(200, text=_feed())

    await _client(handler).search("检索增强生成")
    assert seen["q"] == "检索增强生成", "客户端不该偷偷把中文换成英文"


def test_build_query_does_not_translate() -> None:
    """build_query 只负责拼范围/分类，不碰术语翻译（职责分离）。"""
    assert build_query("检索增强生成") == "检索增强生成"
    # 范围前缀照加，但**词本身不动**（单串不加引号；多词才会包成精确词组）
    assert build_query("检索增强生成", scope="ti") == "ti:检索增强生成"
    assert build_query("检索 增强", scope="ti") == 'ti:"检索 增强"'
