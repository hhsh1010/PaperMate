"""工具调用对**未知参数**的容错（2026-10-03，真实事故回归）。

## 事故
我给 `pdf_extract` 新增了 `doc_id` 参数。模型看到它之后，**凭上文类推**也給
`retrieve` 传了 `doc_id` ⇒

    TypeError: _make_retrieve.<locals>._retrieve() got an unexpected keyword argument 'doc_id'

⇒ 工具连续失败 3 次 ⇒ **整轮熔断**。也就是说：一个**装饰性参数**让整轮对话不可用。

## 修法
`ToolExecutor` 在调用前**剔除 handler 签名里不存在的参数**，并记 warning。
- ⚠️ 只对**调用**容错，**不掩盖** handler 内部的真实异常；
- ⚠️ 拿不到签名时**原样放行**（不猜）；
- ⚠️ handler 有 `**kwargs` 时**不剔除**。
"""
from __future__ import annotations

import asyncio

from lightcore.tools.spec import (
    Permission,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    _drop_unknown_args,
)


def _spec(handler, name="t") -> ToolSpec:
    return ToolSpec(name=name, description="d", parameters={"type": "object", "properties": {}},
                    permission=Permission.SAFE, handler=handler)


def _reg(*specs: ToolSpec) -> ToolRegistry:
    r = ToolRegistry()
    for sp in specs:
        r.register(sp)
    return r


class TestDropUnknownArgs:
    def test_drops_undeclared(self):
        args = {"a": 1, "zzz": 2}
        def h(a): ...
        assert _drop_unknown_args(h, args) == ["zzz"]
        assert args == {"a": 1}

    def test_kwargs_handler_keeps_all(self):
        """有 `**kwargs` ⇒ 接受任意参数，**不剔除**。"""
        def h(**kw): ...
        args = {"anything": 1}
        assert _drop_unknown_args(h, args) == []
        assert args == {"anything": 1}

    def test_no_args_handler(self):
        def h(): ...
        args = {"x": 1}
        assert _drop_unknown_args(h, args) == ["x"]
        assert args == {}


class TestExecutorTolerance:
    def test_unknown_arg_no_longer_raises(self):
        """★ 事故回归：给不认参数的 handler 传未知参数 ⇒ **不抛 TypeError**。"""
        # 模拟 retrieve：handler 不接受任何参数
        ex = ToolExecutor(_reg(_spec(lambda: ToolResult(ok=True, output={"ok": 1}))))
        r = asyncio.run(ex.execute("t", {"doc_id": 6}))
        assert r.ok, f"未知参数不该让工具失败：{r.error}"

    def test_real_exception_still_propagates(self):
        """★ **不掩盖真实异常**：handler 内部抛错仍要冒泡（否则会静默降准）。"""
        def boom():
            raise ValueError("内部真实故障")
        ex = ToolExecutor(_reg(_spec(boom)))
        r = asyncio.run(ex.execute("t", {}))
        assert not r.ok
        assert "内部真实故障" in (r.error or "")

    def test_known_args_unaffected(self):
        ex = ToolExecutor(_reg(_spec(lambda page: ToolResult(ok=True, output={"page": page}))))
        r = asyncio.run(ex.execute("t", {"page": 3}))
        assert r.ok and r.output["page"] == 3
