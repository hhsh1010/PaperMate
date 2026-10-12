"""README 里的 Mermaid 图：语法陷阱的回归判据。

## 为什么需要（2026-10-12 真事故）
架构图在 GitHub 上渲染失败：

```
Unable to render rich display
Parse error on line 27: .../>+ 用例数下限闸门（防"零用例通过"）"] end GUA
Expecting 'SQE', 'DOUBLECIRCLEEND', ... got 'STR'
```

**根因**：Mermaid 的节点标签本身用双引号包裹（`CI["…"]`），而我在标签**内部**又用了
**ASCII 双引号**（`防"零用例通过"`）⇒ 字符串被**提前闭合** ⇒ 解析到 `"]` 时看到多余文本 ⇒ 报错。

★ 这个坑的恶劣之处：**本地看不出来**（.md 在编辑器里渲染正常），只有 GitHub 的 Mermaid
渲染器会炸 —— 而我在沙箱里连 CDN 都拉不到（HTTP 000），无法用真渲染器验证。
⇒ 那就把**已知陷阱**写成静态判据（能在 CI 里跑），而不是靠"记得"：

| 陷阱 | 后果 | 判据 |
|---|---|---|
| 标签内出现 ASCII `"` | 标签提前闭合 ⇒ 解析失败 | 引号包裹的标签内容里不得含 `"` |
| 标签内出现 Markdown 加粗 `**` | 原样渲染成星号（观感差） | 标签内容里不得含 `**` |
| 标签内出现反引号 `` ` `` | 同上 | 标签内容里不得含反引号 |

> 注：**序列图（sequenceDiagram）的正文/Note 是自由文本**，其中的引号合法 ⇒ 本测试只查
> "被引号包裹的节点标签"这类结构化位置。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

#: 引号包裹的标签形态：`X["…"]` / `X[("…")]` / `subgraph X["…"]`
QUOTED_LABEL_LINE = '["'
#: 贪婪匹配整段标签内容（含可能的内层引号）—— ★ 必须用贪婪 `.*`
#: 否则 `[^"]*` 会在**第一个**引号处停下，内层引号永远进不了捕获组 ⇒ 判据结构上失效（假绿）
LABEL_CONTENT = re.compile(r'\["?(.*?)"?\]')


def _mermaid_blocks() -> list[str]:
    text = README.read_text(encoding="utf-8")
    return re.findall(r"```mermaid\n(.*?)```", text, re.S)


def test_readme_has_mermaid_blocks() -> None:
    """README 至少要有架构图（否则本测试形同虚设 —— 判据本身也要能被证伪）。"""
    assert _mermaid_blocks(), "README 里找不到 ```mermaid 块"


def test_quoted_labels_have_no_inner_ascii_quote() -> None:
    """★ 真事故的判据：**引号包裹的节点标签里不得再出现 ASCII 双引号**。

    反例（曾在 GitHub 上崩）：`CI["…（防"零用例通过"）"]` —— 内层引号让标签提前闭合。

    ★★ 判据写法（吃过一次假绿）：**看整行的引号总数**。
    典型标签行只有 2 个引号（开一个、闭一个）；出现内层引号必然 ≥4 个。
    一开始我写成 `\\["([^"]*)"` —— `[^"]*` 在第一个引号处就停了，内层引号**永远进不了捕获组**，
    于是**修复前的坏行也能通过**（假绿）。故这里用计数法，再用贪婪正则做内容复核。
    """
    bad: list[str] = []
    for bi, block in enumerate(_mermaid_blocks(), 1):
        for ln, line in enumerate(block.splitlines(), 1):
            if QUOTED_LABEL_LINE not in line:
                continue
            n_quotes = line.count('"')  # ⚠️ 不写进 f-string：py311 不允许表达式里带转义引号
            if n_quotes != 2:
                bad.append(f"图{bi} L{ln}: 引号数={n_quotes}（应为 2）→ {line.strip()[:90]}")
                continue
            for m in LABEL_CONTENT.finditer(line):
                if '"' in m.group(1):
                    bad.append(f"图{bi} L{ln}: 标签内含 ASCII 引号 → {line.strip()[:90]}")
    assert not bad, (
        "Mermaid 标签里出现了会破坏解析的 ASCII 引号（会让标签提前闭合、整图渲染失败）：\n  "
        + "\n  ".join(bad)
    )


@pytest.mark.parametrize("name", ["README.md"])
def test_no_markdown_emphasis_inside_labels(name: str) -> None:
    """标签内不得用 Markdown 强调/反引号（Mermaid 不做 Markdown 解析，会原样渲染成 `**` / 反引号）。"""
    for bi, block in enumerate(_mermaid_blocks(), 1):
        for ln, line in enumerate(block.splitlines(), 1):
            if QUOTED_LABEL_LINE not in line:
                continue
            for m in LABEL_CONTENT.finditer(line):
                inner = m.group(1)
                for ch in ("**", "`"):
                    assert ch not in inner, f"图{bi} L{ln} 标签内含 {ch!r}：{line.strip()[:80]}"


def test_subgraph_titles_are_quoted() -> None:
    """`subgraph` 标题含空格/中文时必须加引号（否则解析器在空白处断句）。"""
    for bi, block in enumerate(_mermaid_blocks(), 1):
        for ln, line in enumerate(block.splitlines(), 1):
            s = line.strip()
            if s.startswith("subgraph "):
                rest = s[len("subgraph "):]
                assert "[" in rest and rest.count('"') == 2, f"图{bi} L{ln} subgraph 标题未加引号：{s}"
