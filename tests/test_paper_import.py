"""论文导入（阶段 2）的判据 —— **全部离线**。

覆盖两块：
- **服务层**（`PaperImportService`）：SSRF 白名单 / 大小上限（两种）/ 类型校验 / 去重 /
  成功路径与 meta 写法 / 取消 / 重试；
- **路由层**（`routes_papers`）：**默认关闭时 403** / 空关键词 400 / 非法 URL 400 / 未知任务 404。

## 关键设计（测试自己在验证的约束）
1. **前端回传的 url 是不可信输入** ⇒ 非法域名必须在**发请求之前**就拒绝（用"handler 一旦被调用就失败"来证明）；
2. **超限不下载**（`Content-Length` 超限）vs **超限中止**（无长度、流式累计）—— 两种都要拦；
3. **meta 必须在处理完成之后写**（否则被 `_process_phase2` 覆盖）⇒ 断言 ready 之后才有 meta。
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from types import SimpleNamespace

import httpx
import pytest

from papermate.services.paper_import import (
    PaperImportError,
    PaperImportService,
    paper_filename,
    validate_pdf_url,
)

PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
GOOD_URL = "https://arxiv.org/pdf/2401.00001v1"


# ------------------------------------------------------------------ 假文档服务

class FakeDocs:
    """只实现 `PaperImportService` 用到的那 4 个方法（行为可注入）。

    ★ 刻意模拟真实管线的一个副作用：`_process_phase2` 在**处理完成时整体重写** `meta_json`。
    所以这里在"从处理中 → ready"的那一刻**清空 metas** —— 这样"meta 写得太早会被覆盖"
    这条约束才真的被测到（否则测试是软骨头：写错顺序也能过）。
    """

    def __init__(self, *, existing=None, final_status: str = "ready") -> None:
        self.uploads: list[tuple[str, int]] = []
        self.metas: dict[int, dict] = {}
        self._existing = existing
        self._final = final_status
        self._next_id = 1
        self._polled = False          # 第一次轮询返回"处理中"，第二次才给终态

    def check_duplicate(self, filename: str, content: bytes) -> dict:
        return {"duplicate": self._existing is not None,
                "same_name": False, "existing": self._existing}

    async def upload_and_process(self, filename, content, block=False):  # noqa: ANN001, ANN201
        self.uploads.append((filename, len(content)))
        doc_id = self._next_id
        self._next_id += 1
        return SimpleNamespace(id=doc_id, status="uploaded")

    def get_document(self, doc_id: int):  # noqa: ANN201
        if not self._polled:
            self._polled = True
            return SimpleNamespace(id=doc_id, status="uploaded")   # 处理中
        # ★ 模拟 `_process_phase2` 完成时**重写** meta_json
        self.metas.clear()
        return SimpleNamespace(id=doc_id, status=self._final)

    def merge_meta(self, doc_id: int, patch: dict) -> bool:
        self.metas[doc_id] = patch
        return True


def _ok_transport(body: bytes = PDF_BYTES, *, ctype: str = "application/pdf",
                  cl: bool = True) -> httpx.MockTransport:
    headers = {"content-type": ctype}
    if cl:
        headers["content-length"] = str(len(body))
    return httpx.MockTransport(lambda req: httpx.Response(200, headers=headers, content=body))


class _SlowStream(httpx.AsyncByteStream):
    """按块缓慢产出（用于测"取消"）。"""

    def __init__(self, chunks: list[bytes], delay: float) -> None:
        self._chunks, self._delay = chunks, delay

    async def __aiter__(self):  # noqa: ANN204
        for c in self._chunks:
            await asyncio.sleep(self._delay)
            yield c

    async def aclose(self) -> None:
        return None


class _SlowTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):  # noqa: ANN001, ANN201
        chunks = [b"%PDF-1.4\n"] + [b"x" * 4096 for _ in range(20)]
        return httpx.Response(200, headers={"content-type": "application/pdf"},
                              stream=_SlowStream(chunks, 0.05))


async def _settle(svc: PaperImportService, task_id: str, timeout: float = 5.0):  # noqa: ANN202
    """等任务到终态。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        t = svc.get(task_id)
        if t is not None and t.status in ("ready", "failed", "cancelled"):
            return t
        await asyncio.sleep(0.02)
    raise AssertionError(f"任务未在 {timeout}s 内结束：{svc.get(task_id)}")


# ------------------------------------------------------------------ URL 白名单（SSRF）

@pytest.mark.parametrize("bad", [
    "http://127.0.0.1:8000/x.pdf",          # 内网
    "http://169.254.169.254/latest/meta-data",  # 云元数据
    "https://evil.com/pdf/2401.00001.pdf",  # 非白名单域
    "https://arxiv.org/abs/2401.00001",     # 不是 /pdf/ 路径
    "file:///etc/passwd",                   # 协议
    "",                                     # 空
])
def test_ssrf_whitelist_rejects(bad: str) -> None:
    with pytest.raises(PaperImportError):
        validate_pdf_url(bad)


def test_http_upgraded_to_https() -> None:
    assert validate_pdf_url("http://arxiv.org/pdf/2401.00001").startswith("https://")


async def test_invalid_url_never_makes_a_request() -> None:
    """★ 非法域名必须在**发请求之前**拒绝（handler 一旦被调用就让测试失败）。"""
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("不该发起请求")

    svc = PaperImportService(FakeDocs(), transport=httpx.MockTransport(handler))
    with pytest.raises(PaperImportError):
        svc.submit({"pdf_url": "https://evil.com/pdf/a.pdf", "source_id": "a"})
    assert svc._tasks == {}          # 连任务都不该建


# ------------------------------------------------------------------ 下载约束

async def test_content_length_over_limit_rejected_without_download() -> None:
    """★ Content-Length 超上限 ⇒ 直接拒绝，**一个字节都不下载**。"""
    big = PDF_BYTES + b"0" * 4096
    svc = PaperImportService(FakeDocs(), transport=_ok_transport(big), max_bytes=1024)
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "2401.00001"})
    done = await _settle(svc, task.id)
    assert done.status == "failed"
    assert "过大" in done.error
    assert done.downloaded == 0          # 未下载


async def test_streaming_over_limit_aborts() -> None:
    """无 Content-Length 时，流式累计超限也要中止。"""
    big = PDF_BYTES + b"0" * 8192
    svc = PaperImportService(FakeDocs(), transport=_ok_transport(big, cl=False), max_bytes=1024)
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "2401.00001"})
    done = await _settle(svc, task.id)
    assert done.status == "failed"
    assert "过大" in done.error


async def test_wrong_content_type_rejected() -> None:
    svc = PaperImportService(FakeDocs(), transport=_ok_transport(ctype="text/html"))
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "x"})
    done = await _settle(svc, task.id)
    assert done.status == "failed"
    assert "不是 PDF" in done.error


async def test_wrong_magic_bytes_rejected() -> None:
    """Content-Type 撒谎（说 pdf，内容不是）⇒ 用文件头拦下。"""
    svc = PaperImportService(FakeDocs(), transport=_ok_transport(body=b"<html>nope</html>"))
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "x"})
    done = await _settle(svc, task.id)
    assert done.status == "failed"
    assert "%PDF-" in done.error


# ------------------------------------------------------------------ 成功 / 去重 / meta

async def test_happy_path_upload_and_meta_after_ready() -> None:
    docs = FakeDocs()
    svc = PaperImportService(docs, transport=_ok_transport(), poll_interval_s=0.01)
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "2401.00001", "version": "v1",
                       "title": "T"})
    done = await _settle(svc, task.id)
    assert done.status == "ready"
    assert docs.uploads == [("T.pdf", len(PDF_BYTES))]
    # ★ meta 只在处理完成后写（ready 分支），且内容含来源信息
    assert done.doc_id in docs.metas
    paper = docs.metas[done.doc_id]["paper"]
    assert paper["source_id"] == "2401.00001" and paper["version"] == "v1"


async def test_duplicate_does_not_upload_again() -> None:
    """命中去重 ⇒ **不新建文档**（任务仍算成功，并标 duplicate）。"""
    existing = SimpleNamespace(id=42, status="ready", filename="old.pdf")
    docs = FakeDocs(existing=existing)
    svc = PaperImportService(docs, transport=_ok_transport())
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "2401.00001"})
    done = await _settle(svc, task.id)
    assert done.status == "ready"
    assert done.duplicate is True
    assert done.doc_id == 42
    assert docs.uploads == []          # ★ 没有再次入库
    assert docs.metas == {}            # 也不该改别人文档的 meta


async def test_duplicate_of_failed_doc_is_allowed() -> None:
    """旧文档是 failed ⇒ 允许重传（与上传路由同一判据）。"""
    existing = SimpleNamespace(id=7, status="failed", filename="bad.pdf")
    docs = FakeDocs(existing=existing)
    svc = PaperImportService(docs, transport=_ok_transport(), poll_interval_s=0.01)
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "x"})
    done = await _settle(svc, task.id)
    assert done.status == "ready"
    assert done.duplicate is False
    assert len(docs.uploads) == 1      # 真的重传了


async def test_ingest_failure_marks_task_failed() -> None:
    """管线把文档标 failed ⇒ 任务 failed，且**不写 meta**（没成功就不贴来源）。"""
    docs = FakeDocs(final_status="failed")
    svc = PaperImportService(docs, transport=_ok_transport(), poll_interval_s=0.01)
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "x"})
    done = await _settle(svc, task.id)
    assert done.status == "failed"
    assert "解析失败" in done.error
    assert docs.metas == {}


# ------------------------------------------------------------------ 取消 / 重试

async def test_cancel_mid_download() -> None:
    """下载中途取消 ⇒ 状态 cancelled，且**不入库**。"""
    docs = FakeDocs()
    svc = PaperImportService(docs, transport=_SlowTransport())
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "x"})
    await asyncio.sleep(0.12)                  # 让它开始下载
    assert svc.cancel(task.id) is True
    done = await _settle(svc, task.id)
    assert done.status == "cancelled"
    assert docs.uploads == []
    assert svc.cancel(task.id) is False        # 已结束 ⇒ 幂等返回 False


async def test_retry_then_success_on_503() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, headers={"content-type": "application/pdf",
                                            "content-length": str(len(PDF_BYTES))},
                              content=PDF_BYTES)

    docs = FakeDocs()
    svc = PaperImportService(docs, transport=httpx.MockTransport(handler),
                             backoff_base=0.0, poll_interval_s=0.01)
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "x"})
    done = await _settle(svc, task.id)
    assert calls["n"] == 2
    assert done.status == "ready"


async def test_retry_exhausted_fails_readably() -> None:
    svc = PaperImportService(FakeDocs(),
                             transport=httpx.MockTransport(lambda r: httpx.Response(503)),
                             max_retries=1, backoff_base=0.0)
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "x"})
    done = await _settle(svc, task.id)
    assert done.status == "failed"
    assert "重试" in done.error


# ------------------------------------------------------------------ 路由层

@pytest.fixture
def client(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="papermate_paper_")
    monkeypatch.setenv("PAPERMATE_DATA_DIR", tmp)
    monkeypatch.delenv("PAPERMATE_ENABLE_PAPER_SEARCH", raising=False)
    import papermate.config as cfg_mod

    cfg_mod._settings = None
    from fastapi.testclient import TestClient

    from papermate.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_search_disabled_by_default_returns_403(client) -> None:
    """★ 网络功能**默认关**：关时必须明确 403（而不是静默返回空）。"""
    r = client.get("/api/v1/papers/search", params={"q": "rag"})
    assert r.status_code == 403
    assert "PAPERMATE_ENABLE_PAPER_SEARCH" in r.json()["detail"]


def test_search_empty_query_400(client, monkeypatch) -> None:
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    r = client.get("/api/v1/papers/search", params={"q": "   "})
    assert r.status_code == 400


def test_search_ok_with_stubbed_client(client, monkeypatch) -> None:
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")

    class FakeSearch:
        async def search(self, q, *, limit=10, start=0, sort="relevance",
                         scope="all", categories=()):  # noqa: ANN001, ANN201
            from papermate.services.paper_search import PaperHit
            return [PaperHit(source="arxiv", source_id="2401.00001", version="v1",
                             title="T", abstract="A", pdf_url=GOOD_URL)]

    client.app.state.paper_search = FakeSearch()
    r = client.get("/api/v1/papers/search", params={"q": "rag", "limit": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "rag" and len(body["items"]) == 1
    assert body["items"][0]["source_id"] == "2401.00001"


def test_import_rejects_non_arxiv_url(client, monkeypatch) -> None:
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    r = client.post("/api/v1/papers/import",
                    json={"items": [{"pdf_url": "http://127.0.0.1/a.pdf"}]})
    assert r.status_code == 400
    assert "第 1 条不合法" in r.json()["detail"]


def test_import_task_404_for_unknown(client, monkeypatch) -> None:
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    assert client.get("/api/v1/papers/import/nope").status_code == 404
    assert client.post("/api/v1/papers/import/nope/cancel").status_code == 404


def test_import_items_limit(client, monkeypatch) -> None:
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    items = [{"pdf_url": GOOD_URL} for _ in range(21)]
    r = client.post("/api/v1/papers/import", json={"items": items})
    assert r.status_code == 400
    assert "最多导入" in r.json()["detail"]


def test_env_flag_parsing() -> None:
    from papermate.services.paper_import import paper_import_enabled
    old = os.environ.pop("PAPERMATE_ENABLE_PAPER_SEARCH", None)
    try:
        assert paper_import_enabled() is False
        for v in ("1", "true", "YES", "on"):
            os.environ["PAPERMATE_ENABLE_PAPER_SEARCH"] = v
            assert paper_import_enabled() is True
        os.environ["PAPERMATE_ENABLE_PAPER_SEARCH"] = "0"
        assert paper_import_enabled() is False
    finally:
        os.environ.pop("PAPERMATE_ENABLE_PAPER_SEARCH", None)
        if old is not None:
            os.environ["PAPERMATE_ENABLE_PAPER_SEARCH"] = old


def test_search_bad_scope_400(client, monkeypatch) -> None:
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    r = client.get("/api/v1/papers/search", params={"q": "rag", "scope": "title"})
    assert r.status_code == 400
    assert "不支持的检索范围" in r.json()["detail"]


def test_search_bad_category_400(client, monkeypatch) -> None:
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    r = client.get("/api/v1/papers/search", params={"q": "rag", "categories": "cs;drop"})
    assert r.status_code == 400
    assert "分类格式" in r.json()["detail"]


def test_search_passes_scope_and_categories(client, monkeypatch) -> None:
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    got = {}

    class FakeSearch:
        async def search(self, q, *, limit=10, start=0, sort="relevance", scope="all", categories=()):
            got.update(q=q, scope=scope, categories=tuple(categories))
            return []

    client.app.state.paper_search = FakeSearch()
    r = client.get("/api/v1/papers/search",
                   params={"q": "rag", "scope": "ti", "categories": "cs.CL,cs.IR"})
    assert r.status_code == 200
    assert got == {"q": "rag", "scope": "ti", "categories": ("cs.CL", "cs.IR")}


def test_gate_reads_runtime_settings(monkeypatch) -> None:
    """★ 设置页开关（features.paper_search）也能开启 —— 不只有环境变量一条路。

    判定口径是 **OR**：任一为真即开；关闭需两者都关（已在 docstring 写明）。
    """
    from papermate import runtime_settings
    from papermate.services import paper_import as mod

    monkeypatch.delenv("PAPERMATE_ENABLE_PAPER_SEARCH", raising=False)
    st = runtime_settings.get_state()
    old = st["features"].get("paper_search")
    try:
        st["features"]["paper_search"] = True
        assert mod.paper_import_enabled() is True
        st["features"]["paper_search"] = False
        assert mod.paper_import_enabled() is False
        # env 优先级：即使设置为关，env 仍能开（显式运维动作）
        monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
        assert mod.paper_import_enabled() is True
    finally:
        st["features"]["paper_search"] = bool(old)


def test_settings_toggle_end_to_end(client, monkeypatch) -> None:
    """端到端：通过设置接口打开开关 ⇒ 检索接口从 403 变为 200（不需要环境变量）。"""
    monkeypatch.delenv("PAPERMATE_ENABLE_PAPER_SEARCH", raising=False)
    assert client.get("/api/v1/papers/search", params={"q": "rag"}).status_code == 403

    class FakeSearch:
        async def search(self, q, *, limit=10, start=0, sort="relevance", scope="all", categories=()):
            return []

    client.app.state.paper_search = FakeSearch()
    r = client.put("/api/v1/settings", json={"features": {"paper_search": True}})
    assert r.status_code in (200, 201), r.text
    # 设置改的是内存态 + 落盘；同一进程内立即可见
    r2 = client.get("/api/v1/papers/search", params={"q": "rag"})
    assert r2.status_code == 200, r2.text
    # 收尾：关回去，避免影响同进程的其它用例
    client.put("/api/v1/settings", json={"features": {"paper_search": False}})


def test_search_translates_chinese_and_reports_it(client, monkeypatch) -> None:
    """★ 中文查询 ⇒ 后端换成英文再搜，并把替换明细**如实返回**（界面必须明示）。"""
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    got = {}

    class FakeSearch:
        async def search(self, q, *, limit=10, start=0, sort="relevance", scope="all", categories=()):
            got["q"] = q
            return []

    client.app.state.paper_search = FakeSearch()
    r = client.get("/api/v1/papers/search", params={"q": "检索增强生成"})
    assert r.status_code == 200
    body = r.json()
    assert got["q"] == "retrieval augmented generation"       # 发给 arXiv 的是英文
    assert body["query"] == "检索增强生成"                      # 原样回显
    assert body["query_used"] == "retrieval augmented generation"
    assert body["translations"] == [{"zh": "检索增强生成", "en": "retrieval augmented generation"}]
    assert body["remaining_chinese"] is False


def test_search_reports_remaining_chinese(client, monkeypatch) -> None:
    """未命中术语表 ⇒ 原样发出去，但 remaining_chinese=true（前端据此提示改英文）。"""
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    got = {}

    class FakeSearch:
        async def search(self, q, *, limit=10, start=0, sort="relevance", scope="all", categories=()):
            got["q"] = q
            return []

    client.app.state.paper_search = FakeSearch()
    r = client.get("/api/v1/papers/search", params={"q": "RAG 的评估"})
    assert r.status_code == 200
    body = r.json()
    assert got["q"] == "RAG 的评估"
    assert body["remaining_chinese"] is True
    assert body["translations"] == []


def test_list_tasks_restores_after_switch(client, monkeypatch) -> None:
    """★ 切页/刷新后能恢复：`GET /papers/import` 必须列出**仍在跑**的任务。

    这是用户报过的真实问题：导入中切到别的页面，进度"丢了"（其实后端还在跑）。
    """
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    svc = client.app.state.paper_import

    # 用假文档服务 + 永远返回 503 的传输，制造一个会失败的任务（不必真联网）
    class _Docs:
        def check_duplicate(self, filename, content):
            return {"duplicate": False, "same_name": False, "existing": None}

        async def upload_and_process(self, filename, content, block=False):
            raise AssertionError("不该走到入库")

        def get_document(self, doc_id):
            return None

        def merge_meta(self, doc_id, patch):
            return True

    import httpx as _httpx

    from papermate.services.paper_import import PaperImportService

    monkeypatch.setattr(
        client.app.state, "paper_import",
        PaperImportService(_Docs(), transport=_httpx.MockTransport(lambda r: _httpx.Response(503)),
                           max_retries=0, backoff_base=0.0),
    )
    created = client.post("/api/v1/papers/import",
                          json={"items": [{"pdf_url": "https://arxiv.org/pdf/2401.00001v1"}]})
    assert created.status_code == 200
    tid = created.json()["tasks"][0]["id"]

    listed = client.get("/api/v1/papers/import")
    assert listed.status_code == 200
    ids = [t["id"] for t in listed.json()["tasks"]]
    assert tid in ids, "刷新/切页后应能通过列表接口找回该任务"
    assert svc is not None


# ------------------------------------------------------------------ 文件名（用论文原名）

def test_paper_filename_uses_title() -> None:
    """★ 用户反馈：下载下来是 `2302.10473v6.pdf`，看不出是哪篇 ⇒ 改用论文原名。"""
    assert paper_filename("Attention Is All You Need", "2302.10473", "v6") == "Attention Is All You Need.pdf"


def test_paper_filename_strips_illegal_and_whitespace() -> None:
    """Windows 非法字符要清掉；换行/多空格压平。"""
    assert paper_filename('A/B: C? "D" <E> |F|', "id") == "A B C D E F.pdf"
    assert paper_filename("  multi\n  line\t title  ", "id") == "multi line title.pdf"


def test_paper_filename_truncates_by_chars() -> None:
    """长标题按**字符**截断（中文才不会被切坏半个字），且不超上限。"""
    long_zh = "检索增强生成" * 30
    out = paper_filename(long_zh, "id", max_len=20)
    assert out.endswith(".pdf") and len(out) == 24          # 20 字 + .pdf
    assert out[:-4] == long_zh[:20]


def test_paper_filename_falls_back_to_id_then_placeholder() -> None:
    assert paper_filename("", "2302.10473", "v6") == "2302.10473v6.pdf"
    assert paper_filename("", "") == "paper.pdf"
    assert paper_filename("   ", "x") == "x.pdf"


async def test_import_uses_title_as_filename() -> None:
    """端到端：导入后文档名应是**论文标题**，不是 arXiv id。"""
    docs = FakeDocs()
    svc = PaperImportService(docs, transport=_ok_transport(), poll_interval_s=0.01)
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "2401.00001", "version": "v1",
                       "title": "FAIR-RAG: Faithful Adaptive Iterative Refinement"})
    done = await _settle(svc, task.id)
    assert done.status == "ready"
    assert docs.uploads[0][0] == "FAIR-RAG Faithful Adaptive Iterative Refinement.pdf"


async def test_task_exposes_iso_timestamp() -> None:
    """★ 任务出参要带**可解析的 ISO 时间**（前端 timeAgo 需要 ISO，float 不好直接用）。"""
    from datetime import datetime

    svc = PaperImportService(FakeDocs(), transport=_ok_transport())
    task = svc.submit({"pdf_url": GOOD_URL, "source_id": "x"})
    iso = task.as_dict()["created_at_iso"]
    assert isinstance(iso, str) and iso
    parsed = datetime.fromisoformat(iso)
    assert parsed.tzinfo is not None          # 必须带时区（naive 时间会被前端按本地时区误读）


def test_search_deep_start_rejected(client, monkeypatch) -> None:
    """★ 深层分页在**入口**就挡掉（arXiv 位置太深会报错，别让请求打过去才炸）。"""
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")
    r = client.get("/api/v1/papers/search", params={"q": "rag", "start": 480, "limit": 50})
    assert r.status_code == 400
    assert "最多取前 500 条" in r.json()["detail"]


def test_search_normal_start_allowed(client, monkeypatch) -> None:
    """正常翻页（前端上限 100）不该撞到这道保护。"""
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")

    class FakeSearch:
        async def search(self, q, *, limit=10, start=0, sort="relevance", scope="all", categories=()):
            return []

    client.app.state.paper_search = FakeSearch()
    assert client.get("/api/v1/papers/search", params={"q": "rag", "start": 80, "limit": 20}).status_code == 200


def test_search_reports_total_and_has_more(client, monkeypatch) -> None:
    """★ `has_more` 决定前端「加载更多」按钮出不出现 —— 这里钉死。

    （用户反馈"按钮没出现"，根因是**后端还是旧进程**：没有热重载，`total/has_more` 没返回。
     这条用例保证：只要后端是新的，接口一定会如实给出这两个字段。）
    """
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")

    class FakeSearch:
        async def search_page(self, q, *, limit=10, start=0, sort="relevance", scope="all", categories=()):
            return [], 4821          # 总数远大于已返回 ⇒ 还有更多

    client.app.state.paper_search = FakeSearch()
    body = client.get("/api/v1/papers/search", params={"q": "object detection"}).json()
    assert body["total"] == 4821
    assert body["has_more"] is True


def test_search_last_page_has_no_more(client, monkeypatch) -> None:
    """翻到最后一页 ⇒ `has_more=False`（前端该把按钮收起来）。"""
    monkeypatch.setenv("PAPERMATE_ENABLE_PAPER_SEARCH", "1")

    class FakeSearch:
        async def search_page(self, q, *, limit=10, start=0, sort="relevance", scope="all", categories=()):
            return [], 20

    client.app.state.paper_search = FakeSearch()
    body = client.get("/api/v1/papers/search", params={"q": "x", "start": 20, "limit": 20}).json()
    assert body["total"] == 20
    assert body["has_more"] is False
