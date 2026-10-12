"""lightcore 的**进程级资源**工具（与 agent 循环无关，故独立成包）。

当前只有一件事：**重模型加载的跨进程互斥**（见 `resource_lock.py`）。
"""
