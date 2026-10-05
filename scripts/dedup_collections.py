"""文献集关联表去重（一次性运维脚本）。

背景：2026-09-06 起 `_migrate_collection_documents`（models/db.py）改为**告警式**——
启动时检测到重复关联只打 warning、不再自动 DELETE（旧版按 MIN(id) 静默删行，
会无备份地丢失 seq 排序语义）。实际清理改由本脚本人工执行。

行为：
1. 备份整表到 `collection_documents_bak`（幂等：先 DROP 再 CREATE）；
2. 去重保留 **MIN(seq)** 行（保住文献在集内的排序语义；seq 相同再按 id 取小）；
3. 删除重复行 → 重建唯一索引 idx_coldoc；
4. 打印前后行数对比。

用法：
    python -m scripts.dedup_collections            # 检查 + 执行（含备份）
    python -m scripts.dedup_collections --dry-run  # 只报告重复情况，不做任何修改
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import create_engine, text  # noqa: E402

from papermate.config import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="文献集关联表去重（先备份，按 MIN(seq) 保留）")
    ap.add_argument("--dry-run", action="store_true", help="只报告重复情况，不修改数据")
    args = ap.parse_args()

    engine = create_engine(get_settings().database_url or f"sqlite:///{get_settings().resolved_data_dir}/papermate.db")

    with engine.connect() as conn:
        dup = conn.execute(
            text(
                "SELECT collection_id, document_id, COUNT(*) AS c, "
                "GROUP_CONCAT(id) AS ids FROM collection_documents "
                "GROUP BY collection_id, document_id HAVING c > 1"
            )
        ).fetchall()
        total = conn.execute(text("SELECT COUNT(*) FROM collection_documents")).scalar_one()
        print(f"总关联行数：{total}；重复组：{len(dup)}")
        for cid, did, c, ids in dup:
            print(f"  collection={cid} document={did} × {c}（id: {ids}）")

    if not dup:
        print("无需清理。")
        return 0
    if args.dry_run:
        print("dry-run：未做任何修改。")
        return 0

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS collection_documents_bak"))
        conn.execute(text("CREATE TABLE collection_documents_bak AS SELECT * FROM collection_documents"))
        conn.execute(
            text(
                "DELETE FROM collection_documents WHERE id NOT IN ("
                "  SELECT id FROM ("
                "    SELECT id, ROW_NUMBER() OVER ("
                "      PARTITION BY collection_id, document_id ORDER BY seq ASC, id ASC) AS rn"
                "    FROM collection_documents"
                "  ) WHERE rn = 1"
                ")"
            )
        )
        conn.execute(text("DROP INDEX IF EXISTS idx_coldoc"))
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_coldoc "
                "ON collection_documents (collection_id, document_id)"
            )
        )
        after = conn.execute(text("SELECT COUNT(*) FROM collection_documents")).scalar_one()
    print(f"完成：{total} → {after} 行（删除 {total - after} 行重复，备份在 collection_documents_bak）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
