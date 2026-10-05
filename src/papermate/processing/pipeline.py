"""PaperMate · 文档处理子包 · 单文档处理编排（pipeline）

一次完整处理的流水线：解码 → 选解析器 → 解析（结构化章节）→ 元信息识别与校验
→ 致命问题判定 → 格式转换（md/json/txt）→ 分块估算 → 产出 ProcessingResult。

- 纯逻辑、无 DB 依赖；output_dir 可选（给定则把产物落盘并回填 artifacts）。
- 异常分类：空文档/二进制/过短/不支持的编码 → 致命（error_code 见 metadata.FATAL）；
  其余（如 PDF 解析异常）→ 可重试，交由队列层决定。
- 单个文档处理失败不抛异常，而是以 status="failed" 的 ProcessingResult 返回
  （容错原则：处理层不因单文档异常中断批量）。
"""

from __future__ import annotations

import time
from pathlib import Path

from lightcore.rag.chunker import Chunker
from lightcore.rag.parser import ParsedBlock

from .converter import to_markdown, to_plain_text, to_structured
from .metadata import extract_metadata
from .models import ParsedContent, ProcessingResult
from .parsers import ParserFactory, PdfParser


class DocumentProcessingPipeline:
    """单文档处理编排。process() 为同步纯函数（可在线程中执行，不阻塞事件循环）。"""

    def __init__(self) -> None:
        self._chunker = Chunker()

    def process(self, filename: str, content: bytes, output_dir: str | Path | None = None) -> ProcessingResult:
        start = time.monotonic()
        safe_name = Path(filename or "document.txt").name or "document.txt"
        ext = Path(safe_name).suffix.lower()

        # 1) 解码（二进制/乱码检测放在元信息校验，这里先宽松解码）
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")
        # 二进制内容（非文本）：PDF 走解析器；其余直接判 binary_like
        if ext != ".pdf" and _looks_binary(content):
            return self._failed(safe_name, "binary_like", "内容疑似二进制/乱码，非文本文档", start)

        # 2) 解析
        parser = ParserFactory.for_file(safe_name)
        try:
            parsed = self._do_parse(parser, safe_name, text, content)
        except Exception as exc:  # noqa: BLE001  解析异常（如 PDF 损坏）→ 可重试错误
            return self._failed(safe_name, "parse_error", f"解析失败：{exc}", start)

        # 3) 元信息识别与校验
        meta = extract_metadata(parsed, safe_name)
        errors = [i for i in meta.issues if i.severity == "error"]
        warnings = [i.message for i in meta.issues if i.severity == "warning"]

        # 4) 致命问题 → failed（重试无意义）
        if errors:
            code = errors[0].code
            return self._failed(safe_name, code, errors[0].message, start)

        # 5) 格式转换
        markdown = to_markdown(parsed)
        structured = to_structured(parsed, meta)

        # 6) 分块估算（与 RAG 路径同一 Chunker，保证 num_chunks 口径一致）
        num_chunks = len(self._chunker.chunk_blocks(self._to_blocks(parsed), doc_id="doc"))

        # 7) 落盘产物（可选）
        artifacts: dict[str, str] = {}
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(safe_name).stem[:40] or "document"
            artifacts = {
                "json": str(out / f"{stem}.json"),
                "markdown": str(out / f"{stem}.md"),
                "text": str(out / f"{stem}.txt"),
            }
            (out / f"{stem}.json").write_text(_dump_json(structured), encoding="utf-8")
            (out / f"{stem}.md").write_text(markdown, encoding="utf-8")
            (out / f"{stem}.txt").write_text(to_plain_text(parsed), encoding="utf-8")

        return ProcessingResult(
            filename=safe_name,
            source_format=parsed.source_format,
            status="ok",
            title=meta.title,
            metadata=meta,
            sections=parsed.sections,
            num_chunks=num_chunks,
            structured=structured,
            markdown=markdown,
            duration_ms=round((time.monotonic() - start) * 1000, 1),
            warnings=warnings,
            artifacts=artifacts,
        )

    # ---------- 内部 ----------

    def _do_parse(self, parser, safe_name: str, text: str, content: bytes) -> ParsedContent:
        """按解析器类型分发；PDF 需要文件路径（临时文件），其余用文本字符串。

        不传 title（让解析器从内容推导标题，extract_metadata 再做文件名兜底），
        避免文件名覆盖内容标题。
        """
        if isinstance(parser, PdfParser):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                return parser.parse(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        return parser.parse(text)

    def _to_blocks(self, parsed: ParsedContent) -> list[ParsedBlock]:
        """章节 → ParsedBlock（供 Chunker 估算分块数；标题并入正文）。"""
        blocks: list[ParsedBlock] = []
        for s in parsed.sections:
            text = f"{s.heading}\n{s.text}" if s.heading else s.text
            if text.strip():
                blocks.append(ParsedBlock(page=s.page, text=text.strip()))
        return blocks

    @staticmethod
    def _failed(filename: str, code: str, message: str, start: float) -> ProcessingResult:
        return ProcessingResult(
            filename=filename,
            source_format="unknown",
            status="failed",
            title="",
            metadata=None,
            sections=[],
            num_chunks=0,
            structured={},
            markdown="",
            duration_ms=round((time.monotonic() - start) * 1000, 1),
            error=message,
            error_code=code,
        )


def _looks_binary(content: bytes) -> bool:
    """粗略二进制判定：NUL/高位控制字符比例过高（PDF 由解析器处理，不在此列）。"""
    if not content:
        return False
    sample = content[:2000]
    bad = sum(1 for b in sample if b == 0 or (b < 9) or (13 < b < 32))
    return bad / len(sample) > 0.05


def _dump_json(data: dict) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)
