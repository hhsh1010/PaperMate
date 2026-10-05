"""一次性清洗 data/memories.json 中的"状态观察型/拒接型"毒化条目。

背景（docs/实施记录.md 2026-08-19（三续））：
- 早期未修复 retrieve mock 回退时，LLM 反复以"系统中无任何文档""对重复指令静默跳过"
  等措辞拒接问答；该结论通过 reflect() 被作为 salient 事实写入长期记忆。
- 下次会话触发 recall_long() 时这些记忆被注入 SYSTEM 注记，Qwen 据此连续拒调 retrieve 工具。
- 本脚本按与 chat_service._POISON_PATTERNS / long_term._WRITE_BLOCKLIST 同样的模式过滤，
  把保留下来的回写（带审计日志），立即止血。

用法：
    uv run python scripts/clean_poisoned_memories.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 与 src/papermate/services/chat_service.py 和 src/lightcore/memory/long_term.py 同源。
_POISON_PATTERNS = (
    "无任何文档", "无任何上传", "无任何可检索", "无上传任何文件", "无上传任何 PDF",
    "未上传任何", "未成功上传", "未上传 PDF", "未上传文档", "未提供任何文件",
    "向量库为空", "未触发任何文档", "静默跳过", "对无效指令", "对重复指令",
    "前提不成立", "无可供分析的文档", "零文档",
    "系统中无任何", "当前无任何文档", "所有检索返回空",
)


def is_poisoned(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith(("我无法", "无法回答", "无法执行")):
        return True
    return any(p in t for p in _POISON_PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗 memories.json 的毒化条目")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要删除的条目，不写回")
    parser.add_argument("--path", default="data/memories.json", help="记忆文件路径（相对仓库根）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    mem_path = (project_root / args.path).resolve()
    if not mem_path.exists():
        print(f"未找到 {mem_path}，无需清洗。")
        return 0

    raw = mem_path.read_text(encoding="utf-8")
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败：{e}", file=sys.stderr)
        return 1

    if not isinstance(items, list):
        print("memories.json 顶层不是 list，请人工检查。", file=sys.stderr)
        return 1

    keep: list[dict] = []
    drop: list[dict] = []
    for it in items:
        text = it.get("text", "")
        if is_poisoned(text):
            drop.append(it)
        else:
            keep.append(it)

    print(f"原条目数：{len(items)}")
    print(f"将删除：{len(drop)}")
    print(f"将保留：{len(keep)}")
    if drop:
        print("\n--- 将删除的样例（前 10 条）---")
        for it in drop[:10]:
            print(f"  imp={float(it.get('importance', 0)):.2f} text={(it.get('text') or '')[:120]}")

    if args.dry_run:
        print("\n[dry-run] 未写回。")
        return 0

    mem_path.write_text(
        json.dumps(keep, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写回 {mem_path}（备份原文件请人工 cp .bak）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
