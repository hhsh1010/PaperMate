"""LightCore · 文档解析（docs/05 §RAG / 解析）

PDF 解析用 pymupdf（fits）；同时支持直接传入纯文本。按页提取文本块，保留页码用于引用溯源。
后续可扩展：标题/章节切分、表格、公式（docs/05 解析阶段）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: 「数字 + 句点 + 空白 + 数字」—— PDF 里被排版插了空格的小数点（见 `_merge_split_numbers`）。
#: 前后都用**零宽断言**，只替换中间那一段空白，数字本身不参与替换。
_SPLIT_NUMBER_RE = re.compile(r"(?<=\d)\.\s+(?=\d)")

#: 刻意**保留**的全角标点（中文排版惯例）。
#: 依据实测分布（3 篇论文 63811 字符，全角占 15.1%）：`，` 出现 1234 次为最高频，
#: 但它是**中文标点**，且对 jieba 分词毫无影响（全角/半角在中文里都是分隔符）——
#: 转成半角只会破坏中文观感，收益为零。
#: 注：`。` `、` `“` `”` `《` `》` 等码点在 U+3001/U+3002/U+201C… 区段，**本就不在
#: 全角 ASCII 范围内**，无需在此列出。
_KEEP_FULLWIDTH = frozenset("，：；？！")


def to_halfwidth(text: str) -> str:
    """把**全角 ASCII** 归一化为半角，保留中文标点（2026-09-17）。

    **为什么必须做**：学术期刊 PDF 常整段使用全角 ASCII —— 实测 doc16 的英文摘要
    全是 `ＹＯＬＯｖ３－Ｔｉｎｙ`、`ｍＡＰ`、`１．４％` 这种形态。而 `ＹＯＬＯ` 与 `YOLO`
    在分词器里是**不同的 token** → BM25（jieba）与 BGE-M3 的英文/数字检索**双双失效**：
    用户搜 "YOLO" 匹配不到文档里的 "ＹＯＬＯ"。

    覆盖：全角字母（FF21–FF3A / FF41–FF5A）、全角数字（FF10–FF19）、全角空格（U+3000）、
    以及全角 `．－（）／％＋＝～` 等英文/数学语境符号（`1．4` / `ＹＯＬＯ－Ｔｉｎｙ` 里都有）。

    ⚠️ 刻意**不用 `unicodedata.normalize("NFKC", …)`**：NFKC 会连带改写罗马数字、
    带圈字符、上下标等（`①`→`1`、`㈱`→`(株)`、`㎡`→`m2`），副作用不可控且与本次目的无关。
    这里只做**显式映射**，行为完全可预期。
    """
    if not text:
        return text
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E and ch not in _KEEP_FULLWIDTH:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def merge_split_numbers(text: str) -> str:
    """合并被排版空格劈开的数字：`77. 4` → `77.4`（2026-09-19）。

    **真因（实测，与直觉相反）**：不是 pymupdf 插的空格 —— `get_text("rawdict")` 的字符级
    坐标显示，`5` `.` **` `（真实空格字符，宽 2.29）** `3` 依次排列，**PDF 内容流里本来
    就有这个空格**（源文档排版产物）。所以只能做文本层后处理，换提取模式解决不了。

    **影响面**：真实语料 541 处命中，抽样判读绝大多数是真小数
    （`74. 1%` / `95. 2` / `0. 5∶0. 95` / `3. 3 消融实验` / `arXiv:2010. 04159`）。
    这些数字被劈开后检索片段无法判读，是"片段质量差"的主因（见 `docs/检索质量诊断-分块碎片化.md` §6.7）。

    ⚠️ **已知误伤（接受）**：参考文献里「年份句点 + 序号」会被合并 ——
    `…, 2019. 3 [12] Ross Girshick` → `2019.3`。实测抽样误伤约 **1/28（≈4%）**，
    且后果轻微（不影响检索命中，读者仍可理解）。判据：**合并收益（数百处数值可读）
    远大于代价（个别年份后少一个空格）**。

    只在「句点两侧都是数字」时生效，因此英文句末（`et al. 3`）、中文句末等**不受影响**。
    """
    if not text or "." not in text:
        return text
    return _SPLIT_NUMBER_RE.sub(".", text)


@dataclass
class ParsedBlock:
    # page=None 表示「无物理页码」（docx/md/txt 等非分页文档）；
    # PDF 为 1 起算的有效页码。绝不使用 0 表示真实页码（见 lightcore/rag/page_utils）。
    page: int | None
    text: str

    def __post_init__(self) -> None:
        # 归一化放在这里 = **单一来源**：parse_text / parse_pdf / parse_docx / parse 的
        # 可迭代分支全都经过本构造器，不可能漏掉某条路径（比在每个 parse_* 里各调一次可靠）。
        # ① 全角→半角；② 合并被排版空格劈开的数字（顺序无关，两者互不重叠）。
        self.text = merge_split_numbers(to_halfwidth(self.text))


class DocumentParser:
    def parse_text(self, text: str) -> list[ParsedBlock]:
        # 纯文本：按空行分段，无物理页码 → page=None
        blocks: list[ParsedBlock] = []
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                blocks.append(ParsedBlock(page=None, text=para))
        return blocks

    def parse_pdf(self, path: str | Path) -> list[ParsedBlock]:
        import pymupdf

        blocks: list[ParsedBlock] = []
        with pymupdf.open(str(path)) as doc:
            # 1 起算：第 1 页 → page_no=1，绝不会出现第 0 页
            for page_no, page in enumerate(doc, start=1):
                for block in page.get_text("blocks"):
                    txt = block[4].strip()
                    if txt:
                        blocks.append(ParsedBlock(page=page_no, text=txt))
        return blocks

    def parse_docx(self, path: str | Path) -> list[ParsedBlock]:
        """Word .docx 解析（python-docx 惰性导入）：段落 + 表格文本，page=None（无物理页码）。

        段落文本按 **「接受全部修订」** 语义提取（2026-09-30 修复）：取全部 `w:t`
        （含 `w:ins` 嵌套 run）、忽略 `w:delText`。★ 原先用 `p.text`：python-docx 的
        `runs` 只含段落**直接子级** `w:r`，改稿（track-changes replace）后的新文字包在
        `w:ins` 里 ⇒ 被整个漏掉，索引出碎片（实测改段后 chunk = `'引言。 e icrow s.'`，
        见 `scripts/probe_edit_consistency.py`）。与 `docx_edit._para_accepted_text`
        同口径（= Word 导出"干净版"的语义）。

        老版 .doc（OLE 二进制）python-docx 无法打开 → 抛 ValueError 提示转存 .docx。
        """
        try:
            from docx import Document  # 惰性：仅 .docx 场景才需要
            from docx.oxml.ns import qn
        except ImportError:
            raise ValueError("缺少 python-docx 依赖（uv add python-docx）") from None
        try:
            doc = Document(str(path))
        except Exception as e:  # noqa: BLE001 .doc 老格式 / 损坏文件
            raise ValueError(
                f"无法解析该 Word 文档（{type(e).__name__}）。"
                f"老版 .doc 二进制格式不受支持，请先用 Word/WPS 转存为 .docx 后重新上传。"
            ) from e

        def _accepted_text(el) -> str:
            return "".join(node.text or "" for node in el.iter() if node.tag == qn("w:t"))

        blocks: list[ParsedBlock] = []
        for p in doc.paragraphs:
            t = _accepted_text(p._p).strip()
            if t:
                blocks.append(ParsedBlock(page=None, text=t))
        for tbl in doc.tables:
            for row in tbl.rows:
                cells = [_accepted_text(c._tc).strip() for c in row.cells]
                if any(cells):
                    blocks.append(ParsedBlock(page=None, text=" | ".join(cells)))
        return blocks

    def parse(self, source: str | Path | Iterable[str]) -> list[ParsedBlock]:
        if isinstance(source, str | Path):
            p = Path(source)
            if p.exists() and p.suffix.lower() == ".pdf":
                return self.parse_pdf(p)
            # 当作纯文本路径或字符串
            if p.exists():
                return self.parse_text(p.read_text(encoding="utf-8", errors="ignore"))
            return self.parse_text(str(source))
        # 可迭代文本块：无页码来源 → page=None
        return [ParsedBlock(page=None, text=str(s)) for s in source]
