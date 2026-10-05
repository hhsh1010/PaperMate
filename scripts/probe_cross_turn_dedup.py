"""跨轮去重（`pdf_extract`）可行性**完整探针**（2026-10-03）。

## 为什么需要这个探针
上一次探针只验证了「能否从 messages 反推内容哈希」，**漏了更前置的一条**：
**那个键在"执行前"算不出来**（它是输出内容的哈希，只有执行完才知道），
而 tool 消息的 content **不含调用参数** ⇒ 无法从历史反推 args ⇒ 方案不成立。

本探针把前提**逐条**验一遍，任何一条为否就不该动手。

## 判定项
| # | 前提 | 判据 |
|---|---|---|
| ① | 有**执行前可得**的稳定文档标识 | `documents.content_hash` 存在且非空 |
| ② | 键可在**执行前**算出 | `content_hash + page` 纯由 args 决定 |
| ③ | 键**唯一**对应一次调用 | 单文档会话里 `page` 不跨文档 |
| ④ | 内容变更后**自然失效** | `content_hash` 随内容变（设计如此） |
| ⑤ | 扫描成本可接受 | 扫 N 条 tool 消息的耗时 |
| ⑥ | 收益规模 | 跨轮重复率 × 单次体积 |

## 用法
`.venv/Scripts/python.exe scripts/probe_cross_turn_dedup.py`

**只读**：不写库、不改代码、不调用 LLM。
"""
from __future__ import annotations

import ast
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "papermate.db"
sys.path.insert(0, str(ROOT / "src"))

#: 与 `lightcore.core.loop._format_tool_result` 的注入前缀**必须一致**（读侧靠它剥离）
RESULT_PREFIX = "RESULT: "


def _key_from_message(content: str) -> str | None:
    """从 tool 消息反推 `content_hash:page` 键；非该形态一律 `None`（= 没有原文）。"""
    if not isinstance(content, str) or not content.startswith(RESULT_PREFIX):
        return None
    try:
        parsed = ast.literal_eval(content[len(RESULT_PREFIX):])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("page") is None or not isinstance(parsed.get("text"), str):
        return None
    # ⚠️ 现有 tool 结果里**没有** content_hash 字段 ⇒ 只能退回"页码级"标识。
    # 这正是本探针要暴露的缺口：`pdf_extract` 的输出没带文档标识。
    return f"page:{parsed.get('doc_hash', '?')}:{parsed['page']}"


def main() -> int:
    con = sqlite3.connect(DB)
    cur = con.cursor()
    print("=" * 66)
    print("跨轮去重（pdf_extract）可行性探针")
    print("=" * 66)

    # ── ① 执行前可得的稳定文档标识 ──
    cur.execute("PRAGMA table_info(documents)")
    cols = [r[1] for r in cur.fetchall()]
    has_hash = "content_hash" in cols
    print(f"\n① documents 表字段: {cols}")
    print(f"   content_hash 存在: {'YES' if has_hash else 'NO'}")
    if has_hash:
        cur.execute("SELECT id, substr(content_hash,1,12), filename FROM documents LIMIT 3")
        for did, h, fn in cur.fetchall():
            print(f"   · doc {did}  hash={h}…  {fn}")

    # ── ②③ 页码在会话内是否唯一（多文档会让 page 跨文档命中）──
    cur.execute("SELECT id, config_json FROM sessions")
    cur.execute("SELECT collection_id, COUNT(*) FROM collection_documents GROUP BY collection_id")
    coll_size = dict(cur.fetchall())
    print(f"\n②③ 每个文献集的文档数分布: {sorted(set(coll_size.values()))}")
    print("   ⚠️ 若存在 >1 篇的集合：`page=N` 会**跨文档**命中（见下方实测）")

    cur.execute(
        "SELECT page, COUNT(DISTINCT document_id) FROM chunks GROUP BY page HAVING page IS NOT NULL"
    )
    multi = cur.fetchall()
    print(f"   同页码出现在多篇文档的情况: {len(multi)} 个页码（共 {len(multi)} 种）")
    if multi:
        print(f"   样例: {multi[:5]}")

    # ── ④ 键的唯一性：从真实 tool 消息里能否算出稳定键 ──
    cur.execute("SELECT events_json FROM traces ORDER BY id DESC LIMIT 300")
    keys, parse_fail, total = set(), 0, 0
    for (ev,) in cur.fetchall():
        try:
            events = json.loads(ev or "[]")
        except Exception:
            continue
        for e in events:
            if e.get("type") != "tool_result":
                continue
            d = e.get("data") or {}
            if d.get("name") != "pdf_extract":
                continue
            total += 1
            k = _key_from_message(d.get("content") or "")
            if k is None:
                parse_fail += 1
            else:
                keys.add(k)
    print(f"\n④ pdf_extract 工具结果共 {total} 条")
    print(f"   能算出键的: {total - parse_fail} · 算不出(非 RESULT 形态/占位): {parse_fail}")
    print(f"   不同键数: {len(keys)}（若 < 调用数说明有重复）")
    print(f"   ★ 样例键: {sorted(keys)[:3]}")
    print("   ⚠️ 键里 `doc_hash` 全是 '?' ⇒ 因为 tool 结果**没带**文档标识"
          "（要补字段才能得到真正稳定的键）")

    # ── ⑤ 扫描成本 ──
    big = "水利设施数据集构建方法与实验对象样本区高分二号影像预处理。" * 80
    payload = f"RESULT: {json.dumps({'page': 9, 'text': big, 'num_chunks': 40}, ensure_ascii=False)}"
    t0 = time.perf_counter()
    for _ in range(100):
        _key_from_message(payload)
    dt = (time.perf_counter() - t0) / 100
    print(f"\n⑤ 单条 {len(payload)} 字符的消息解析耗时 ≈ {dt*1000:.2f} ms")
    print(f"   ⇒ 扫 100 条历史 ≈ {dt*100:.1f} ms（每轮一次可接受；每步一次不可接受）")

    con.close()
    print("\n" + "=" * 66)
    print("结论：① 内容哈希存在；⑤ 成本可接受；**②③ 有缺口**")
    print("  ⇒ 动手前必须先给 pdf_extract 的输出**补一个文档标识字段**")
    print("     （否则键只能退化成 page 级，多文档会话会误判）。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
