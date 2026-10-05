"""PaperMate · 长期记忆共享单例（2026-08-29 A 档修复）

背景（缺陷）：此前 `ChatService.__init__` 自建一个 `MemoryManager`（启动时把
`data/memories.json` 载入进程内存），而 `api/routes_memory.py` 另建一个
`JSONFileMemoryStore` 直接读写同一个文件。两者只在启动时同步一次，导致
**设置页「长期记忆管理」面板的编辑 / 删除不生效**——对话侧仍按内存中的旧内容
召回，必须重启后端才生效。

修复：收敛为进程级共享单例，路由与对话服务操作同一实例，面板改动即时生效。
单例构建参数与 ChatService 保持一致（FakeHashEmbedder + JSONFileMemoryStore），
路径取自 settings.data_dir（回退 PAPERMATE_DATA_DIR / "data"）。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from lightcore.memory.manager import MemoryManager
from lightcore.memory.store import JSONFileMemoryStore
from lightcore.rag.embedder import FakeHashEmbedder

_lock = threading.Lock()
_shared: MemoryManager | None = None


def memory_path(settings: object) -> Path:
    data_dir = getattr(settings, "data_dir", None) or os.environ.get("PAPERMATE_DATA_DIR", "data")
    return Path(data_dir) / "memories.json"


def _build_embedder() -> tuple[object, str]:
    """按运行时开关构建记忆向量引擎；BGE-M3 不可用（未装 rag extra / 权重缺失）时降级为哈希。

    返回 (embedder, 实际生效名称)。
    """
    try:
        from .. import runtime_settings

        name = runtime_settings.get_state().get("memory_embedder", "hash")
    except Exception:  # noqa: BLE001  运行时设置不可用时按默认处理
        name = "hash"
    if name == "bge-m3":
        try:
            from lightcore.rag.embedder import BgeM3Embedder

            return BgeM3Embedder(), "bge-m3"
        except Exception:  # noqa: BLE001  依赖/权重缺失 → 静默降级，绝不阻塞启动
            return FakeHashEmbedder(), "hash"
    return FakeHashEmbedder(), "hash"


def get_memory_manager(settings: object) -> MemoryManager:
    """返回进程级共享的长期记忆管理器（线程安全懒加载）。"""
    global _shared
    if _shared is None:
        with _lock:
            if _shared is None:
                path = memory_path(settings)
                path.parent.mkdir(parents=True, exist_ok=True)
                embedder, _name = _build_embedder()
                mgr = MemoryManager(embedder, store=JSONFileMemoryStore(path))
                # 切换 embedder 后旧向量维度不匹配 → 按当前引擎重算（余弦比较的前提）
                mgr.long.ensure_embeddings()
                _shared = mgr
    return _shared


def set_memory_embedder(name: str) -> None:
    """切换记忆向量引擎并重建单例（"hash" / "bge-m3"）；非法取值抛错。"""
    if name not in ("hash", "bge-m3"):
        raise ValueError(f"未知的记忆向量引擎: {name}")
    from .. import runtime_settings

    state = runtime_settings.get_state()
    if state.get("memory_embedder") != name:
        state["memory_embedder"] = name
        runtime_settings.save()
    reset_memory_manager()  # 下次取用时按新引擎重建并重算向量


def reset_memory_manager() -> None:
    """丢弃当前单例（仅供测试隔离使用）。"""
    global _shared
    with _lock:
        _shared = None
