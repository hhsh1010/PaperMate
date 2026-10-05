"""LightCore · 检索管线调用适配（**唯一入口**）

### 为什么需要这个模块

本项目的检索管线有**两种形态**，**同名不同步**、调用点上长得一模一样：

- **同步**：`RAGPipeline.query` / `CompositeRAGPipeline.query`
  —— BGE-M3 编码 + 混合检索 + CrossEncoder 重排，**GPU 密集**；
- **异步**：`MultiQueryRAGPipeline.query`（`features.multi_query` 开启时外层包一层）。

靠肉眼分辨形态 ⇒ 本项目**已经踩过两次坑**：

1. **N4（2026-09-10）**：同步取值拿到的是**协程对象**，后续迭代抛
   `TypeError: 'coroutine' object is not iterable` ⇒ 工具连续失败 3 次 ⇒ agent 提前中止 ⇒
   用户只看到「抱歉，本次没有生成有效回答」（真实库 09-10 单日 21 次）。
2. **N9（2026-09-11）**：同步管线**直接跑在事件循环**上 ⇒ 阻塞整个循环
   —— 实测一次工具调用 **262.82s**，期间每 0.3s 的断连轮询协程无法运行 ⇒
   「停止」无效、API 无响应、GPU 满载。

⇒ 把这两条判断**收敛为唯一实现**：所有调用点统一走 `call_query`，
无论是"漏判了异步"还是"忘了下放线程池"，都不再可能**只在某一个调用点**发生。

### 使用

```python
from lightcore.rag.invoke import call_query

hits = await call_query(pipeline, query, top_k=5)   # 自动适配两种形态
```

⚠️ **调用方必须是 async**（同步函数无法使用本模块）。
"""

from __future__ import annotations

import asyncio
import inspect


async def call_query(pipeline: object, query: str, *, top_k: int) -> list:
    """调用检索管线的 `query`，自动适配**同步 / 异步**两种形态。

    - 异步管线 → 直接 `await`；
    - 同步管线 → **下放线程池**（`asyncio.to_thread`），避免阻塞事件循环（N9）；
    - 兜底 → 返回值若仍是 awaitable（自定义实现），再 `await` 一次（N4）。

    :raises TypeError: 管线未提供 `query()`。调用方自行决定是否容错。
    """
    fn = getattr(pipeline, "query", None)
    if fn is None:
        raise TypeError(f"{type(pipeline).__name__} 未提供 query()，无法检索")

    if inspect.iscoroutinefunction(fn):
        results = await fn(query, top_k=top_k)
    else:
        # 同步管线（GPU 密集）必须下放线程池：直接在事件循环里调用会阻塞整个循环。
        results = await asyncio.to_thread(fn, query, top_k=top_k)

    if inspect.isawaitable(results):  # 兜底：自定义实现可能返回 awaitable
        results = await results

    return list(results)
