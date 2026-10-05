"""PaperMate · SQLite 写重试公共工具（2026-08-19 事故复盘）

本机存在环境级文件监视进程，会不定期用「拒绝写共享」的句柄短暂锁定
数据库文件（默认库为 data/papermate.db，见 config.py），使 SQLite 报
"attempt to write a readonly database"。
`busy_timeout` 只能重试 SQLITE_BUSY，对 SQLITE_READONLY 无效，因此所有
DB 写操作需在服务层带退避重试；非锁定类错误立即抛出。
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from typing import Any

try:
    from sqlalchemy.exc import OperationalError
except ImportError:  # pragma: no cover  测试环境无 sqlalchemy 时退化
    OperationalError = Exception  # type: ignore[assignment,misc]


def write_with_retry(fn: Callable[[], Any], tries: int = 4, base_delay: float = 0.6) -> Any:
    """执行一次 DB 写操作；遇 readonly/locked 时指数退避重试。

    - fn：应在一个全新 Session 上下文中执行（失败后旧 Session 已回滚关闭，
      重试即新建 Session，避免脏状态复用）。
    - 返回 fn() 的结果（如 doc_id / 组装好的 out）。
    - 非锁定类异常立即抛出，不吞错。
    """
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except OperationalError as e:
            msg = str(e)
            if "readonly" in msg or "locked" in msg:
                last = e
                _time.sleep(base_delay * (i + 1))
                continue
            raise
    raise RuntimeError(
        f"SQLite 写操作重试 {tries} 次仍失败（疑似数据库文件被外部进程持续锁定）"
    ) from last
