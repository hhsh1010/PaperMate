"""PaperMate · 回复逐条核验（P1-2.1，方案 docs/可行性分析与改动方案 2.1）

对 agent 生成的每条回复做逐条核验，重点覆盖「引用能否准确命中用户需求指定的页数位置」：

- **L1 确定性校验**（零 LLM、零延迟）：把每条引用（citation）按 (document_id, page) 从
  Chunk 表重建该页实际文本，归一化后检查引用文本是否真的在该页出现。命中 = 引用与原文
  一致且页码正确；未命中 = 标记「存疑」（页码标错 / 引用被改写 / 引用为编造）。
- **L2 LLM 语义校验**（开关化，随 verify_reply 开启；+1 次调用）：让 LLM 对
  (用户问题, 回复, 引用) 做语义核验——是否答非所问、引用是否支撑结论、页码是否明显错误。
  asyncio 8s 超时 / 解析失败 → 降级 skipped；**只标「存疑」不自动改写回复**。

约束：不修改 final_text；不新增对既有事件流的改动；失败一律静默降级，绝不阻断对话。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from lightcore.llm.base import ChatMessage, Role

#: 引用 id 形如 `doc-{document_id}-c{seq}`（Chunker._make_chunk 生成，doc_id=f"doc-{doc_id}"）
_ID_RE = re.compile(r"^doc-(\d+)-c\d+$")
#: L2 超时（秒）：核验不得拖慢回复收尾
_L2_TIMEOUT_S = 8.0
#: 归一化去空白（含换行/全角空格）
_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS_RE.sub("", (text or "").lower())


def _parse_document_id(citation_id: Any) -> int | None:
    """从引用 id 解析 document_id（`doc-3-c12` → 3）。解析失败返回 None（无法定位）。"""
    m = _ID_RE.match(str(citation_id or ""))
    return int(m.group(1)) if m else None


def _page_text(db: Any, document_id: int, page: int) -> str:
    """按 (document_id, page) 聚合该页全部 chunk 文本（Chunk 表），作为该页实际内容。"""
    from ..models.db import Chunk

    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id, Chunk.page == page)
        # K5 修复（2026-09-10）：显式按 seq 排序——原实现无 order_by，跨 chunk 拼接顺序
        # 取决于 DB 物理返回顺序，页码复核可能因拼接顺序变化误报「引用编造」。
        .order_by(Chunk.seq)
        .all()
    )
    return "\n".join(r.content for r in rows)


def _text_in_page(citation_text: str, page_text: str) -> bool:
    """引用文本是否真的在该页出现（归一化子串 + 前缀宽松匹配）。

    LLM 常对检索块文本做截断/加省略号，故全串失败后用引用前 40 字符做前缀匹配
    （长度 ≥10 才算有效前缀），兼顾严格与抗改写。
    """
    cite = _norm(citation_text)
    page = _norm(page_text)
    if not cite:
        return True  # 空引用无信息量，不判错
    if cite in page:
        return True
    prefix = cite[: min(40, len(cite))]
    return len(prefix) >= 10 and prefix in page


def _has_physical_pages(db: Any, document_id: int) -> bool:
    """文档是否含真实物理页码（page>0）。docx/md/txt 等非 PDF 全为 page=None → False。"""
    from ..models.db import Chunk

    n = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id, Chunk.page > 0)
        .count()
    )
    return n > 0


def verify_l1(citations: list[dict], db: Any) -> dict:
    """L1 确定性校验：逐条核验引用文本 ↔ 页码对应关系。

    返回：{"status": "ok"|"issues", "issues": [{severity, scope, page, text_snippet, reason}]}

    页码处理（2026-08-24 修复「第 0 页」问题）：
    - page 语义：None = 无物理页码（docx/md/txt）；≥1 整数 = 有效页码（PDF，1 起算）。
      解析层与管线层均已保证不会出现 page=0（见 lightcore/rag/page_utils）。
    - 引用无有效页码（None，或防御性 0）且文档有物理页码 → suspect「页码缺失/标注错误」；
    - 引用无页码且文档本就无物理页码（docx/md/txt）→ warn「无页码可核验」，不误判 ok。
    """
    issues: list[dict] = []
    for c in citations:
        page = c.get("page")
        text = c.get("text") or ""
        doc_id = _parse_document_id(c.get("id"))
        snippet = (text or "")[:60]
        if doc_id is None or not page:
            # 无法定位文档，或没有页码信息（0/None 统一视为「无有效页码」）
            if doc_id is not None and not page:
                try:
                    if _has_physical_pages(db, doc_id):
                        issues.append({
                            "severity": "suspect",
                            "scope": "citation",
                            "page": None,
                            "text_snippet": snippet,
                            "reason": "引用未标注页码（该文档有页码，应为标注缺失或错误）",
                        })
                        continue
                except Exception:  # noqa: BLE001
                    pass
            issues.append({
                "severity": "warn" if doc_id is not None else "suspect",
                "scope": "citation",
                "page": page,
                "text_snippet": snippet,
                "reason": (
                    "引用缺少有效页码（文档无物理页码或引用未标注），无法核验页码"
                    if doc_id is not None
                    else "引用无法定位到具体文档页（缺少有效 id/页码），需人工核对"
                ),
            })
            continue
        try:
            page_text = _page_text(db, doc_id, int(page))
        except Exception:  # noqa: BLE001  查询异常按无法核验处理
            page_text = ""
        if not page_text:
            issues.append({
                "severity": "suspect",
                "scope": "citation",
                "page": page,
                "text_snippet": snippet,
                "reason": f"文档 #{doc_id} 第 {page} 页无提取文本，无法核验引用（可能页码标错）",
            })
        elif not _text_in_page(text, page_text):
            issues.append({
                "severity": "suspect",
                "scope": "citation",
                "page": page,
                "text_snippet": snippet,
                "reason": f"引用文本未能在第 {page} 页找到（页码可能标错，或引用被改写/编造）",
            })
    return {"status": "ok" if not issues else "issues", "issues": issues}


async def _verify_l2(user_query: str, final_text: str, citations: list[dict], llm: Any) -> dict:
    """L2 LLM 语义校验：答非所问 / 引用支撑性 / 页码合理性。失败或超时降级 skipped。"""
    try:
        cite_block = "\n".join(
            f"- [p.{c.get('page', '?')}] {c.get('text', '')[:120]}" for c in citations
        ) or "（无引用）"
        prompt = (
            "你是回复质量核验器。请对以下 AI 回复做逐项核验：\n"
            "1) 相关性：回复是否答非所问；\n"
            "2) 引用支撑：引用是否支撑回复结论；\n"
            "3) 页码：引用页码是否明显错误。\n"
            "只输出 JSON（不要其他文字）："
            '{"checks":[{"aspect":"相关性|引用支撑|页码","verdict":"ok|warn|suspect","note":"一句话说明"}]}\n'
            f"【用户问题】\n{user_query[:500]}\n"
            f"【AI 回复】\n{final_text[:1500]}\n"
            f"【引用】\n{cite_block}"
        )
        resp = await asyncio.wait_for(
            llm.chat(
                [ChatMessage(role=Role.SYSTEM, content=prompt)],
                temperature=0.1,
                max_tokens=600,
            ),
            timeout=_L2_TIMEOUT_S,
        )
        raw = (resp.message.content or "").strip()
        data = json.loads(raw)
        checks = data.get("checks", []) if isinstance(data, dict) else []
        issues: list[dict] = []
        for ch in checks:
            verdict = (ch.get("verdict") or "ok").lower()
            if verdict in ("warn", "suspect"):
                issues.append({
                    "severity": "suspect" if verdict == "suspect" else "warn",
                    "scope": "reply",
                    "page": None,
                    "text_snippet": None,
                    "reason": f"[{ch.get('aspect')}] {ch.get('note') or '存疑'}",
                })
        return {"status": "ok" if not issues else "issues", "issues": issues}
    except Exception:  # noqa: BLE001  超时/解析失败/无 key → 降级 skipped
        return {"status": "skipped", "issues": []}


async def verify_reply(
    *,
    user_query: str,
    final_text: str,
    citations: list[dict],
    db: Any,
    llm: Any | None = None,
) -> dict:
    """完整核验入口（L1 必跑 + L2 随 llm 可用性尝试）。

    返回（供 SSE `verify` 事件下发，前端只读展示）：
    {
      "level": 1|2,            # 实际执行到的最高级别
      "l2": "ok"|"issues"|"skipped",  # L2 结果（未执行时省略）
      "status": "ok"|"issues"|"skipped",  # skipped 仅当 L1 无数据且 L2 降级
      "issues": [...]
    }
    """
    l1 = verify_l1(citations or [], db)
    out: dict = {"level": 1, "status": l1["status"], "issues": list(l1["issues"])}
    if llm is not None:
        l2 = await _verify_l2(user_query, final_text, citations or [], llm)
        out["level"] = 2
        out["l2"] = l2["status"]
        out["issues"].extend(l2["issues"])
        if l2["status"] == "issues":
            out["status"] = "issues"
    return out


def only_suspects(result: dict) -> dict | None:
    """把核验结果收敛为「只含 suspect 项」的 L1 报告；无 suspect 时返回 `None`（表示不必下发）。

    **用途（2026-09-16，P0）**：让 **L1 确定性核验常开**，但**不打扰用户**。

    背景：L1 零 LLM、零延迟，抓的是最伤研究场景的错（页码标错 / 引用文本不在该页 / 引用无法定位）。
    但它此前被 `features.verify_reply`（为 **省 L2 的 LLM 调用** 而设的开关）**连坐关闭** ——
    最便宜的一道防线跟着最贵的那道一起被关掉了。

    常开之后必须解决"噪音"问题：L1 对 **无物理页码的文档**（docx / md / txt）会产出
    `warn` 级「引用缺少有效页码，无法核验页码」——这是**正常情况**，不是问题。
    若照单全发，每个带引用的回复都会弹核验卡，用户很快会学会忽略它（警示疲劳）。
    因此开关关闭时**只下发 suspect**：

    | severity | 含义 | 开关关闭时 |
    |---|---|---|
    | `suspect` | 确凿可疑：页码标错、引用文本在该页找不到、引用无法定位到文档 | **下发** |
    | `warn` | 提醒：文档本无物理页码，无从核验 | 静音（正常情况） |

    开关**打开**时维持原行为（L1 + L2 结果全量下发，用户显式要求看核验）。
    """
    suspects = [i for i in (result.get("issues") or []) if i.get("severity") == "suspect"]
    if not suspects:
        return None
    return {"level": 1, "status": "issues", "issues": suspects}


#: 触发「零引用」提示的最短回答长度 —— 太短的应答（"好的"/"收到"）不该要求引用
ZERO_CITE_MIN_LEN = 120


def zero_citation_notice(doc_bound: bool, final_text: str, has_cites: bool) -> dict | None:
    """doc-bound 会话下回答**零引用**时，返回一条 warn 级提示；不适用时返回 `None`。

    **为什么需要**（2026-09-23）：L1 校验的对象是「引用」，而**先验知识幻觉的特征恰恰是没有引用**
    —— 文档里没有那个内容，模型自然引不出来。于是 L1 在"最该拦的场景"里静默通过：
    用户看到一段通顺、专业、无引用的回答，无从察觉它来自训练先验而非他的文档。

    这是「**强制检索 ≠ 强制采信**」的落点：协议层能强制模型*调用* retrieve，
    但没有任何机制能强制它*采信*结果；零引用检测是**事后**唯一能兜住的一层。

    两个限定（避免警示疲劳，否则用户会学会无视它）：
    - 只在 `doc_bound` 会话（未绑文档时本就不该要求引用）；
    - 只在回答有实质长度时（`ZERO_CITE_MIN_LEN`）。

    抽成纯函数而非塞在路由闭包里，是为了**可单测** —— 这段逻辑的边界（长短、绑没绑文档）
    正是最容易写错的地方。
    """
    if has_cites or not doc_bound:
        return None
    text = (final_text or "").strip()
    if len(text) <= ZERO_CITE_MIN_LEN:
        return None
    return {
        "status": "issues",
        "issues": [{
            "severity": "warn",
            "scope": "citations",
            "page": None,
            "text_snippet": text[:80],
            "reason": (
                "本轮回答未引用任何文档片段。若其中包含文档外的事实（模型自身知识），"
                "请要求它先检索再回答；确属无需引用的问答可忽略此提示。"
            ),
        }],
    }
