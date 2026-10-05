"""文档处理模块（功能增强规格说明 §2 + 批量/版本/统计增强）单元测试

覆盖：
- parsers：Text/Markdown 标题结构解析；PDF 页码（A-04，注入伪页面块，不依赖 pymupdf）；工厂路由
- metadata：语言识别、关键词、校验问题（空/二进制/过短/标题兜底/无章节）、致命错误码
- converter：md/json/txt/bibtex 转换
- versioning：指纹、相似度、diff 统计、版本注册/去重/对比
- pipeline：正常处理（md/text）、致命失败（空/二进制）、可重试失败（parse_error）、产物落盘
- queue：批量提交、失败隔离、可重试错误退避重试、致命错误不重试
- stats：聚合统计、重试计数
- API：单文档处理 / 批量队列 / 已存文档处理+持久化 / 结构化 / 导出 / 版本 / 差异对比
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

import papermate.config as cfg_mod
from papermate.processing import converter
from papermate.processing import metadata as meta_mod
from papermate.processing.metadata import is_fatal
from papermate.processing.models import (
    Job,
    JobItem,
    ProcessingResult,
)
from papermate.processing.parsers import (
    MarkdownParser,
    ParserFactory,
    PdfParser,
    TextParser,
)
from papermate.processing.pipeline import DocumentProcessingPipeline
from papermate.processing.queue import JobQueue
from papermate.processing.stats import queue_stats_to_summary, result_summary, summarize
from papermate.processing.versioning import (
    DocumentVersioner,
    content_fingerprint,
    diff_texts,
)

# ---------- 样例文本 ----------

MD_DOC = """# Attention Is All You Need

## Abstract

We propose the Transformer, a model architecture based on self-attention.

## Method

The attention mechanism computes a weighted sum of values.

## Experiments

Our model achieves state-of-the-art results on WMT tasks.
"""

TXT_DOC = """Transformer 是一种基于自注意力机制的序列建模架构。

它摒弃了循环结构，通过并行计算提升训练效率。

该模型在机器翻译与文本生成任务上取得了显著成果。"""

EN_DOC = (
    "Retrieval augmented generation combines parametric memory with non-parametric corpus.\n\n"
    "The retriever fetches relevant passages, then the generator synthesizes an answer.\n\n"
    "This approach reduces hallucination and improves factual accuracy."
)


# ---------- fixtures ----------

@pytest.fixture
def client():
    tmp = tempfile.mkdtemp(prefix="papermate_proc_test_")
    os.environ["PAPERMATE_DATA_DIR"] = tmp
    cfg_mod._settings = None
    from papermate.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _make_result(filename: str, status: str = "ok", code: str | None = None, error: str | None = None) -> ProcessingResult:
    return ProcessingResult(
        filename=filename,
        source_format="markdown",
        status=status,
        title="T",
        metadata=None,
        sections=[],
        num_chunks=0,
        structured={},
        markdown="",
        duration_ms=1.0,
        error=error,
        error_code=code,
    )


# ---------- parsers ----------

class TestParsers:
    def test_text_parser_sections(self):
        parsed = TextParser().parse(TXT_DOC, title="demo.txt")
        assert parsed.source_format == "text"
        assert parsed.title == "demo.txt"
        assert len(parsed.sections) == 3
        assert all(s.page == 0 for s in parsed.sections)

    def test_markdown_parser_headings(self):
        parsed = MarkdownParser().parse(MD_DOC)
        assert parsed.source_format == "markdown"
        headings = [s.heading for s in parsed.sections if s.heading]
        assert headings == ["Attention Is All You Need", "Abstract", "Method", "Experiments"]
        assert parsed.sections[1].text  # Abstract 节有正文
        assert parsed.sections[1].level == 2

    def test_pdf_parser_preserves_page_numbers(self, monkeypatch):
        """A-04：PDF 解析保留真实页码（注入伪页面块，无需 pymupdf）。"""
        parser = PdfParser()
        monkeypatch.setattr(
            parser,
            "_read_pages",
            lambda _path: [(1, ["Attention Is All You Need", "We propose the Transformer model."]), (2, ["Experiments", "Our model achieves SOTA."])],
        )
        parsed = parser.parse("paper.pdf")
        assert parsed.source_format == "pdf"
        assert parsed.num_pages == 2
        assert parsed.sections[0].page == 1  # 标题在第一页
        body_pages = [s.page for s in parsed.sections if not s.heading]
        assert all(p >= 1 for p in body_pages)
        assert parsed.title == "Attention Is All You Need"

    def test_pdf_parser_normalizes_fullwidth_text(self, monkeypatch):
        """回归（2026-09-09）：中文期刊 PDF 的全角排版入库后呈「乱码/断句错乱」。

        _read_pages 层做 NFKC 归一化：全角字母数字标点→半角、全角空格→空格、
        清除零宽字符；中文标点与换行结构不受影响。
        """
        parser = PdfParser()
        monkeypatch.setattr(
            parser,
            "_read_pages",
            lambda _path: [(1, [
                "Ｆｉｇ．９　Ｓｅｇｍｅｎｔａｔｉｏｎ　Ｃｏｍｐａｒｉｓｏｎ",   # 全角字母/标点/空格
                "第５０卷　本文选取Ｍａｓｋ Ｒ－ＣＮＮ、ＹＯＬＯｖ８，共４８轮。",  # 全角数字/连字符
                "零宽\u200b字符与软连字符\xad应被清除",
                "中文标点不受影响：模型，改进了。结果、很好！",
            ])],
        )
        parsed = parser.parse("paper.pdf")
        joined = "\n".join(((s.heading or "") + "\n" + s.text) for s in parsed.sections)
        assert "Fig.9 Segmentation Comparison" in joined          # 全角字母/句点/空格→半角
        assert "第50卷" in joined and "YOLOv8" in joined and "共48轮" in joined
        assert "R-CNN" in joined                                   # 全角连字符→半角
        assert "\u200b" not in joined and "\xad" not in joined     # 零宽/软连字符已清除
        # 中文标点有意保留（不用 NFKC 的原因：它会把 ，：！？（） 也转半角，劣化中文排版）
        assert "模型，改进了。结果、很好！" in joined
        assert any(s.page == 1 for s in parsed.sections)           # 页码归属不变

    def test_pdf_parser_rejoins_broken_words(self, monkeypatch):
        """回归（2026-09-09，U1 残留项 C）：双栏/窄格 PDF 的断词重连。

        用户实测案例：表格窄格把单词切成 ａ/ｎｄ、Ｍｏ/ｄｅｌ/Ｉｍｐｒｏｖｅ/ｍｅｎｔ，
        引用进回复呈「混乱断句」。两类精确启发式：
        ① 全角窄格断行（行尾+下行首均为全角小写）→ 无缝重连；
        ② 半角连字符断词（improve-\\nment）→ dehyphenation。
        正常全角行（行首大写/中文开头）不得被误合并。
        """
        parser = PdfParser()
        monkeypatch.setattr(
            parser,
            "_read_pages",
            lambda _path: [(1, [
                # 用户实测形态：窄格强制断行（无连字符）
                "Ｄａｔａ Ｆｏｒｍｓ ｂｅｆｏｒｅ ａ\nｎｄ ａｆｔｅｒ Ｍｏ\nｄｅｌ Ｉｍｐｒｏｖｅ\nｍｅｎｔ",
                # 英文标准连字符断词
                "We propose an improve-\nment of the detector.",
                # 不应被误合并：行首大写全角 / 中文开头
                "Ｄｉａｇｒａｍ ｆｏｒ Ｖａｒｉｏｕｓ\nＤａｔａ Ｆｏｒｍｓ",
                "实验结论如下\n模型精度提升。",
            ])],
        )
        parsed = parser.parse("paper.pdf")
        joined = "\n".join(((s.heading or "") + "\n" + s.text) for s in parsed.sections)
        assert "before and after Model Improvement" in joined      # 全角断词已重连并归一化
        assert "improvement of the detector" in joined             # 连字符断词已重连
        assert "Various\nData Forms" in joined                     # 行首大写：不误合并
        assert "实验结论如下\n模型精度提升。" in joined              # 中文行：不误合并

    def test_factory_routing(self):
        assert isinstance(ParserFactory.for_file("a.pdf"), PdfParser)
        assert isinstance(ParserFactory.for_file("b.md"), MarkdownParser)
        assert isinstance(ParserFactory.for_file("c.txt"), TextParser)
        assert isinstance(ParserFactory.for_file("d.unknown"), TextParser)


# ---------- metadata ----------

class TestMetadata:
    def test_language_detection(self):
        assert meta_mod.detect_language(TXT_DOC) == "zh"
        assert meta_mod.detect_language(EN_DOC) == "en"

    def test_extract_metadata_zh(self):
        parsed = TextParser().parse(TXT_DOC, title="transformer.txt")
        meta = meta_mod.extract_metadata(parsed, "transformer.txt")
        assert meta.language == "zh"
        assert meta.chars > 0
        assert meta.keywords  # 中文二元组关键词非空
        assert meta.ok  # 无 error 级问题

    def test_empty_document_error(self):
        parsed = TextParser().parse("", title="empty.txt")
        meta = meta_mod.extract_metadata(parsed, "empty.txt")
        assert not meta.ok
        assert any(i.code == "empty_document" and i.severity == "error" for i in meta.issues)
        assert is_fatal("empty_document")

    def test_binary_like_error(self):
        parsed = TextParser().parse("\x00\x01\x02\x03\x04 garbage \xff\xfe" * 10, title="bin.txt")
        meta = meta_mod.extract_metadata(parsed, "bin.txt")
        assert any(i.code == "binary_like" for i in meta.issues)
        assert is_fatal("binary_like")

    def test_too_short_error(self):
        parsed = TextParser().parse("hi", title="short.txt")
        meta = meta_mod.extract_metadata(parsed, "short.txt")
        assert any(i.code == "too_short" for i in meta.issues)
        assert is_fatal("too_short")

    def test_title_fallback_warning(self):
        # 解析器标题为空时用文件名兜底 → warning
        parsed = TextParser().parse("内容较长的正文段落。\n\n第二段正文内容。", title="")
        meta = meta_mod.extract_metadata(parsed, "my_report.txt")
        assert meta.title == "my_report"
        assert any(i.code == "title_fallback" for i in meta.issues)


# ---------- converter ----------

class TestConverter:
    def test_markdown_roundtrip(self):
        parsed = MarkdownParser().parse(MD_DOC)
        md = converter.to_markdown(parsed)
        assert md.startswith("# Attention Is All You Need")
        assert "## Abstract" in md
        reparsed = MarkdownParser().parse(md)
        assert [s.heading for s in reparsed.sections if s.heading] == [
            "Attention Is All You Need",
            "Abstract",
            "Method",
            "Experiments",
        ]

    def test_structured_contains_sections_and_meta(self):
        parsed = MarkdownParser().parse(MD_DOC)
        meta = meta_mod.extract_metadata(parsed, "paper.md")
        data = converter.to_structured(parsed, meta)
        assert data["title"] == "Attention Is All You Need"
        assert len(data["sections"]) == 4
        assert data["metadata"]["language"] == "en"

    def test_plain_text_no_markdown_syntax(self):
        parsed = MarkdownParser().parse(MD_DOC)
        txt = converter.to_plain_text(parsed)
        assert "#" not in txt

    def test_bibtex(self):
        bib = converter.to_bibtex("Attention Is All You Need", filename="attention_2017.pdf")
        assert bib.startswith("@misc{")
        assert "title" in bib and "2017" in bib


# ---------- versioning ----------

class TestVersioning:
    def test_fingerprint_normalizes_whitespace(self):
        assert content_fingerprint("A  B\nC") == content_fingerprint("ab c")

    def test_diff_counts_and_similarity(self):
        d = diff_texts("line1\nline2\nline3", "line1\nline2 X\nline3\nline4", left_title="v1", right_title="v2")
        assert d.similarity < 1.0
        assert d.added_lines >= 1
        assert d.removed_lines >= 1
        assert "v2" in d.unified

    def test_version_register_and_dedup(self):
        versions: list = []
        v1 = DocumentVersioner.register(versions, TXT_DOC, TXT_DOC.encode())
        assert v1.version == 1 and v1.summary == "初始版本"
        v2 = DocumentVersioner.register(versions, TXT_DOC + "\n新增一段内容。", TXT_DOC.encode() + b"\nnew")
        assert v2.version == 2
        # 相同指纹重复注册 → 跳过（返回末版，不新增）
        v3 = DocumentVersioner.register(versions, TXT_DOC + "\n新增一段内容。", TXT_DOC.encode() + b"\nnew")
        assert v3 is v2 and len(versions) == 2

    def test_compare_versions(self):
        versions: list = []
        DocumentVersioner.register(versions, "A\nB", b"A\nB")
        DocumentVersioner.register(versions, "A\nB\nC", b"A\nB\nC")
        d = DocumentVersioner.compare_versions(versions, 1, 2, {1: "A\nB", 2: "A\nB\nC"})
        assert d.added_lines >= 1


# ---------- pipeline ----------

class TestPipeline:
    def test_process_markdown_ok(self, tmp_path):
        pipe = DocumentProcessingPipeline()
        result = pipe.process("paper.md", MD_DOC.encode(), output_dir=tmp_path)
        assert result.status == "ok"
        assert result.source_format == "markdown"
        assert result.title == "Attention Is All You Need"
        assert result.num_chunks >= 1
        assert result.metadata is not None and result.metadata.ok
        assert result.artifacts  # 产物落盘
        assert (tmp_path / "paper.json").exists()
        assert (tmp_path / "paper.md").exists()

    def test_process_text_ok(self):
        result = DocumentProcessingPipeline().process("doc.txt", TXT_DOC.encode())
        assert result.status == "ok"
        assert result.source_format == "text"

    def test_process_empty_is_fatal(self):
        result = DocumentProcessingPipeline().process("empty.txt", b"")
        assert result.status == "failed"
        assert result.error_code == "empty_document"
        assert is_fatal(result.error_code)

    def test_process_binary_is_fatal(self):
        result = DocumentProcessingPipeline().process("bin.bin", b"\x00\x01\x02\x03" * 100)
        assert result.status == "failed"
        assert result.error_code == "binary_like"

    def test_process_short_is_fatal(self):
        result = DocumentProcessingPipeline().process("short.txt", b"hi")
        assert result.status == "failed"
        assert result.error_code == "too_short"

    def test_unknown_ext_falls_back_to_text(self):
        result = DocumentProcessingPipeline().process("note.xyz", EN_DOC.encode())
        assert result.status == "ok"
        assert result.source_format == "text"


# ---------- queue ----------

class TestQueue:
    async def _drain(self, queue: JobQueue):
        await queue.wait_idle(timeout=5)
        await queue.stop()

    async def test_batch_submit_and_done(self):
        queue = JobQueue(processor=DocumentProcessingPipeline().process, max_workers=2)
        await queue.start()
        jobs = queue.submit(
            [JobItem("a.md", MD_DOC.encode()), JobItem("b.txt", TXT_DOC.encode())]
        )
        assert len(jobs) == 2
        assert all(j.status == "pending" for j in jobs)
        await self._drain(queue)
        assert all(queue.get(j.id).status == "done" for j in jobs)
        assert queue.stats()["done"] == 2

    async def test_failure_isolation(self):
        """单任务失败不影响其他任务。"""
        def _processor(filename: str, _content: bytes) -> ProcessingResult:
            if filename == "bad.txt":
                return _make_result(filename, status="failed", code="empty_document", error="空文档")
            return _make_result(filename)

        queue = JobQueue(processor=_processor, max_workers=1)
        await queue.start()
        jobs = queue.submit([JobItem("bad.txt", b""), JobItem("good.txt", b"x" * 100)])
        await self._drain(queue)
        by_name = {j.filename: j for j in jobs}
        assert by_name["bad.txt"].status == "failed"
        assert by_name["good.txt"].status == "done"

    async def test_retryable_error_retries_then_fails(self):
        calls = {"n": 0}

        def _flaky(_filename: str, _content: bytes) -> ProcessingResult:
            calls["n"] += 1
            return _make_result("f.txt", status="failed", code="parse_error", error="临时失败")

        queue = JobQueue(processor=_flaky, max_workers=1, max_retries=2, retry_delay=0.01)
        await queue.start()
        [job] = queue.submit([JobItem("f.txt", b"x" * 200)])
        await self._drain(queue)
        assert job.status == "failed"
        assert job.attempts == 2  # 首次 + 1 次重试后达上限
        assert calls["n"] == 2
        assert queue.stats()["failed"] == 1

    async def test_retry_then_success(self):
        calls = {"n": 0}

        def _flaky_then_ok(_filename: str, _content: bytes) -> ProcessingResult:
            calls["n"] += 1
            if calls["n"] < 2:
                return _make_result("g.txt", status="failed", code="parse_error", error="第一次失败")
            return _make_result("g.txt")

        queue = JobQueue(processor=_flaky_then_ok, max_workers=1, max_retries=3, retry_delay=0.01)
        await queue.start()
        [job] = queue.submit([JobItem("g.txt", b"x" * 200)])
        await self._drain(queue)
        assert job.status == "done"
        assert job.attempts == 2
        assert calls["n"] == 2

    async def test_fatal_error_no_retry(self):
        calls = {"n": 0}

        def _fatal(_filename: str, _content: bytes) -> ProcessingResult:
            calls["n"] += 1
            return _make_result("e.txt", status="failed", code="empty_document", error="空文档")

        queue = JobQueue(processor=_fatal, max_workers=1, max_retries=3, retry_delay=0.01)
        await queue.start()
        [job] = queue.submit([JobItem("e.txt", b"")])
        await self._drain(queue)
        assert job.status == "failed"
        assert job.attempts == 1  # 致命错误不重试
        assert calls["n"] == 1


# ---------- stats ----------

class TestStats:
    def test_summarize_ok_and_failed(self):
        ok = _make_result("a.md")
        ok.metadata = None
        ok.duration_ms = 10.0
        bad = _make_result("b.txt", status="failed", code="empty_document")
        s = summarize([ok, bad])
        assert s.total == 2 and s.ok == 1 and s.failed == 1
        assert s.by_error == {"empty_document": 1}
        assert s.by_format["markdown"] == 2

    def test_result_summary_fields(self):
        pipe = DocumentProcessingPipeline()
        r = pipe.process("a.md", MD_DOC.encode())
        d = result_summary(r)
        assert d["status"] == "ok"
        assert d["chars"] > 0 and d["headings"] >= 1

    def test_queue_stats_retried_count(self):
        jobs = [
            Job(id="1", filename="a", status="done", attempts=2),
            Job(id="2", filename="b", status="failed", attempts=1, error_code="x"),
            Job(id="3", filename="c", status="done", attempts=1),
        ]
        s = queue_stats_to_summary(jobs)
        assert s.retried == 1  # 仅 attempts>1 的 done/failed


# ---------- API ----------

class TestProcessingApi:
    def _process(self, client, name: str, content: bytes):
        return client.post("/api/v1/documents/process", files={"file": (name, content)})

    def test_process_single(self, client):
        r = self._process(client, "paper.md", MD_DOC.encode())
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "ok"
        assert data["title"] == "Attention Is All You Need"
        assert data["metadata"]["headings"] >= 1
        assert data["structured"]["sections"]

    def test_process_empty_returns_client_error(self, client):
        """空文件属客户端错误（400）；批量队列中则记为 failed 任务（容错路径由 batch 测试覆盖）。"""
        r = self._process(client, "empty.txt", b"")
        assert r.status_code == 400

    def test_batch_and_jobs(self, client):
        files = [
            ("files", ("a.md", MD_DOC.encode(), "text/markdown")),
            ("files", ("b.txt", TXT_DOC.encode(), "text/plain")),
        ]
        r = client.post("/api/v1/documents/process/batch", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 2
        job_ids = [j["id"] for j in body["jobs"]]
        # 等队列排空
        for _ in range(200):
            rr = client.get("/api/v1/documents/process/jobs")
            if all(j["status"] in ("done", "failed") for j in rr.json()):
                break
            time.sleep(0.02)
        statuses = {j["id"]: j["status"] for j in rr.json()}
        assert all(statuses[i] in ("done", "failed") for i in job_ids)
        # 任务详情
        detail = client.get(f"/api/v1/documents/process/jobs/{job_ids[0]}")
        assert detail.status_code == 200
        assert detail.json()["filename"] in ("a.md", "b.txt")
        # 统计
        st = client.get("/api/v1/documents/process/stats")
        assert st.status_code == 200
        assert st.json()["total"] >= 2

    def test_process_existing_doc_persist_and_export(self, client):
        # 1) 先上传（走既有上传流程）
        up = client.post("/api/v1/documents", files={"file": ("paper.md", MD_DOC.encode(), "text/markdown")})
        assert up.status_code == 200
        doc_id = up.json()["id"]
        # 2) 处理并持久化
        r = client.post(f"/api/v1/documents/{doc_id}/process")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["processing"]["status"] == "ok"
        assert body["versions"][0]["version"] == 1
        # 3) 结构化视图
        struct = client.get(f"/api/v1/documents/{doc_id}/structured")
        assert struct.status_code == 200
        assert struct.json()["title"] == "Attention Is All You Need"
        # 4) 导出 md
        exp = client.get(f"/api/v1/documents/{doc_id}/export?format=md")
        assert exp.status_code == 200
        assert "Attention Is All You Need" in exp.text
        # 5) 导出 json
        expj = client.get(f"/api/v1/documents/{doc_id}/export?format=json")
        assert expj.status_code == 200
        assert json.loads(expj.text)["title"] == "Attention Is All You Need"

    def test_versions_and_compare(self, client):
        up1 = client.post("/api/v1/documents", files={"file": ("v1.md", MD_DOC.encode(), "text/markdown")})
        doc_a = up1.json()["id"]
        client.post(f"/api/v1/documents/{doc_a}/process")
        # 注册版本
        vr = client.post(f"/api/v1/documents/{doc_a}/versions")
        assert vr.status_code == 200
        assert vr.json()["version"] == 1
        # 列表
        vl = client.get(f"/api/v1/documents/{doc_a}/versions")
        assert vl.status_code == 200
        assert len(vl.json()) >= 1
        # 第二个文档 + 对比
        up2 = client.post("/api/v1/documents", files={"file": ("v2.md", (MD_DOC + "\n## New Section\nadded").encode(), "text/markdown")})
        doc_b = up2.json()["id"]
        client.post(f"/api/v1/documents/{doc_b}/process")
        cmp = client.get(f"/api/v1/documents/compare?left={doc_a}&right={doc_b}")
        assert cmp.status_code == 200, cmp.text
        assert cmp.json()["similarity"] < 1.0
        assert "added" in cmp.json()["summary"] or cmp.json()["added_lines"] >= 1

    def test_compare_route_not_shadowed_by_doc_id(self, client):
        """/documents/compare 字面路径不被 /documents/{doc_id} 吞掉（缺参时返回 422 而非文档不存在）。"""
        r = client.get("/api/v1/documents/compare")
        assert r.status_code == 422  # 缺 left/right 参数 → 参数校验错误（说明命中 compare 路由）

    def test_missing_doc_404(self, client):
        assert client.get("/api/v1/documents/999/structured").status_code == 404
        assert client.get("/api/v1/documents/999/export?format=md").status_code == 404
        assert client.get("/api/v1/documents/999/versions").status_code == 404
        assert client.post("/api/v1/documents/999/process").status_code == 404
