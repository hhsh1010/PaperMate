"""PaperMate · API 入口：python -m papermate.api"""

from __future__ import annotations

import logging
import os
import queue as _queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path

import uvicorn

from ..config import get_settings
from . import app

#: 显式确认"我知道在公开暴露"的环境变量（设为 "1" 即只打一行提示）
_ACK_ENV = "PAPERMATE_ACCEPT_PUBLIC_EXPOSURE"


def should_block_exposure(exposed: bool, ack: str | None) -> bool:
    """E3（2026-09-21）：是否应**阻断启动**。

    抽成纯函数是为了可测：`__main__` 里的启动分支会 `raise SystemExit`，
    直接在测试里跑会杀掉测试进程 —— 判定逻辑与副作用必须分开。
    """
    return bool(exposed) and ack != "1"


def warn_if_exposed(exposed: bool, host: str) -> bool:
    """非回环监听时打印醒目告警；返回是否处于暴露状态。

    背景：本项目定位**本地单用户、无鉴权**（见 config.Settings.host 的安全说明），
    一旦绑到 0.0.0.0 / 局域网地址，**同网段任何人都能读写全部会话、文档、长期记忆**，
    甚至通过 `PUT /settings` 覆盖模型密钥。默认值已是回环，此处针对的是**用户显式改绑**的场景 ——
    此前只在代码注释里写了风险，运行时没有任何提示。

    设计取舍：**告警但不阻断启动**（用户可能确实需要局域网访问），
    设 `PAPERMATE_ACCEPT_PUBLIC_EXPOSURE=1` 可确认知情、降为一行提示。
    """
    # 判定口径来自 config.Settings.network_exposed（单一来源）
    if not exposed:
        return False

    if os.environ.get(_ACK_ENV) == "1":
        print(f"[PaperMate] 已确认对外暴露（host={host}）——请自行确保网络边界安全。")
        return True

    # 注意（2026-09-15 踩过的坑）：不要写成相邻字面量 `"\n" "=" * 78` ——
    # Python 会**先合并相邻字面量**得到 "\n="，再执行 `* 78`，输出变成 `=\n=\n=...`。
    # 这里统一用 f-string，避免隐式拼接与运算符优先级打架。
    bar = "=" * 78
    sep = "-" * 74
    banner = (
        f"\n{bar}\n"
        f"  ⚠️  安全告警：服务正在监听非回环地址 {host}\n"
        f"  {sep}\n"
        "  本项目**没有鉴权机制**（定位为本地单用户工具）。绑到该地址后：\n"
        "    · 同网段任何人都能读写全部会话、文档与长期记忆；\n"
        "    · `PUT /api/v1/settings` 可覆盖模型密钥等配置。\n"
        "\n"
        "  若只是本机使用：请设 PAPERMATE_HOST=127.0.0.1（默认值）后重启。\n"
        f"  若确需局域网访问：请在可信网络内使用，并设置 {_ACK_ENV}=1 关闭本告警。\n"
        f"{bar}\n"
    )
    print(banner, flush=True)
    logging.getLogger(__name__).warning("服务监听非回环地址 %s（无鉴权，请评估暴露风险）", host)
    return True


if __name__ == "__main__":
    # 监听地址走配置（默认 127.0.0.1，见 config.Settings.host 的安全说明）：
    # 既让 .env 里的 PAPERMATE_HOST/PORT 真正生效，也默认不外暴露无鉴权服务。
    cfg = get_settings()

    # 2026-09-06：日志落文件（此前全靠 uvicorn 默认 stdout，console 一关告警全丢）。
    # 2026-09-07 加固：队列化双监听——Windows 控制台 QuickEdit 选中/管道破裂会让
    # StreamHandler 写入永久阻塞，而 logging 全局锁会让「任何一个写日志的线程」
    # 连带冻结整个事件循环（全站假死，所有线程 Wait）。现改为：
    # 业务线程只投递到无界队列（永不阻塞）→ 两个监听线程分别写文件与控制台；
    # 控制台卡死只影响自己的队列，app.log 照常落盘，后端照常服务。
    log_dir = Path(cfg.resolved_data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_h = RotatingFileHandler(
        log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_h.setFormatter(_fmt)

    console_h = logging.StreamHandler()
    console_h.setFormatter(_fmt)

    q_file: _queue.SimpleQueue = _queue.SimpleQueue()
    q_console: _queue.SimpleQueue = _queue.SimpleQueue()
    QueueListener(q_file, file_h).start()
    QueueListener(q_console, console_h).start()

    logging.basicConfig(
        level=logging.INFO,
        handlers=[QueueHandler(q_file), QueueHandler(q_console)],
    )

    # uvicorn 日志显式路由到同一对队列（dictConfig 而非 None——实测 None 时 uvicorn
    # 记录的 message 会自带 "INFO:uvicorn.error:" 双重前缀）。propagate=False 防重复。
    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "queue_file": {"()": QueueHandler, "queue": q_file},
            "queue_console": {"()": QueueHandler, "queue": q_console},
        },
        "loggers": {
            "uvicorn": {"handlers": ["queue_file", "queue_console"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"level": "INFO"},
        },
    }

    # E3 强化拦截（2026-09-21，用户决策）：非回环监听从「告警但不阻断」升级为「必须显式确认」。
    #
    # 为什么升级：原实现打印横幅后照常启动 —— 而"手滑改绑 0.0.0.0"恰恰是**静默**发生的
    # （改个环境变量、复制一条命令），用户没看到横幅就以为没事。本项目**无鉴权**，
    # 且 `PUT /settings` 可覆盖模型密钥，这个暴露面值得一道硬闸。
    # 逃生门保留：`PAPERMATE_ACCEPT_PUBLIC_EXPOSURE=1` 即放行。
    # 默认 host=127.0.0.1 时本分支不触发，**不影响日常使用**。
    # 先放横幅（把人话讲清楚），再决定是否阻断 —— 顺序不能反：
    # 直接阻断而不解释，用户只会看到一句"已阻断"却不知道风险是什么。
    warn_if_exposed(cfg.network_exposed, cfg.host)
    if should_block_exposure(cfg.network_exposed, os.environ.get(_ACK_ENV)):
        print(
            "\n  已阻断启动（E3 强化拦截）。\n"
            f"  确认知情后重试：设置 {_ACK_ENV}=1\n"
            "  或改回本机使用：设置 PAPERMATE_HOST=127.0.0.1（默认值）\n"
        )
        raise SystemExit(2)

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_config=uvicorn_log_config)
