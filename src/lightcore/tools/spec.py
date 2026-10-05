"""LightCore · 工具系统（docs/05 §工具系统）

- Permission: 权限分级（safe/network/sandbox/admin）
    safe     : 本地无副作用（计算、检索本地索引）—— 默认可执行
    network  : 需要出网（搜索、抓取）—— 需用户授权
    sandbox  : 代码执行，必须在 Docker 沙箱中跑（NFR: 代码执行 Docker 沙箱）
    admin    : 高危（删库、写系统）—— 默认禁用，需显式开启
- ToolSpec: 一个工具的声明（name/description/parameters JSON Schema/permission/handler）
- ToolRegistry: 注册与查找，支持 JSON Schema 导出给模型
- ToolExecutor: 执行 + 权限校验 + 失败计数（供 Agent 循环熔断）
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Permission(StrEnum):
    SAFE = "safe"
    NETWORK = "network"
    SANDBOX = "sandbox"
    ADMIN = "admin"


# 权限默认开关（安全优先：safe 开，network 需授权，sandbox 默认关[NFR-07 代码执行默认关闭]，admin 关）
DEFAULT_PERMISSION_ENABLED = {
    Permission.SAFE: True,
    Permission.NETWORK: False,
    Permission.SANDBOX: False,  # NFR-07：代码执行默认关闭，需 features.code_interpreter 显式开启
    Permission.ADMIN: False,
}


@dataclass
class ToolResult:
    ok: bool
    output: Any
    error: str | None = None


ToolHandler = Callable[..., Awaitable[ToolResult]] | Callable[..., ToolResult]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    permission: Permission
    handler: ToolHandler
    # 是否来自 MCP 扩展（预留）
    source: str = "builtin"

    def to_llm(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具名冲突: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"未注册工具: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """全部注册的工具规格（供设置页「本地数据管理」展示）。"""
        return list(self._tools.values())

    def specs_for_llm(self) -> list[dict[str, Any]]:
        return [t.to_llm() for t in self._tools.values()]


def _allowed_args(handler: Callable[..., Any]) -> frozenset[str] | None:
    """取 handler 允许的参数名集合；`**kwargs` 或取不到签名时返回 `None`（= 不剔除）。"""
    try:
        params = inspect.signature(handler).parameters
    except (TypeError, ValueError):  # 取不到签名 ⇒ 不猜
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return frozenset(params)


def _drop_unknown_args(handler: Callable[..., Any], args: dict[str, Any]) -> list[str]:
    """剔除 handler 签名里**不存在**的参数，返回被剔除的名字。

    ★ 为什么不直接抛：模型会**凭上文类推**出未声明的参数（真实事故：给 `pdf_extract`
    加了 `doc_id`，模型就也给 `retrieve` 传），而一个装饰性参数不该让整轮工具不可用。
    ⚠️ 取不到签名 / handler 接受 `**kwargs` ⇒ **原样放行**（不猜、不剔除）。
    """
    allowed = _allowed_args(handler)
    if allowed is None:
        return []
    unknown = [k for k in args if k not in allowed]
    for k in unknown:
        args.pop(k, None)
    return unknown


logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, permission_enabled: dict[Permission, bool] | None = None) -> None:
        self.registry = registry
        self.permission_enabled = permission_enabled or dict(DEFAULT_PERMISSION_ENABLED)
        #: tool name → handler 允许的参数名（`None` = 不剔除）。反射有开销，故缓存。
        self._allowed_cache: dict[str, frozenset[str] | None] = {}

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self.registry.get(name)
        if not self.permission_enabled.get(spec.permission, False):
            return ToolResult(ok=False, output=None,
                              error=f"权限被拒: 工具 '{name}' 需要 {spec.permission.value} 权限，当前未开启。")
        try:
            # ★★ 2026-10-03：调用前**剔除 handler 不认识的参数**（防止模型乱传导致 TypeError）。
            # 起因（真实事故）：我给 `pdf_extract` 新增了 `doc_id`，模型以为它是通用参数，
            # 于是也给 `retrieve` 传了 ⇒ `TypeError: unexpected keyword argument 'doc_id'`
            # ⇒ 工具连续失败 3 次 ⇒ **整轮熔断**（一个装饰性参数引发的"整轮不可用"）。
            # ⚠️ 只对**调用**容错、**不掩盖** handler 内部的真实异常（异常仍照常冒泡）。
            args = dict(arguments or {})
            # ⚠️ 缓存按 **tool name** 存于 executor 实例，**不用 `id(handler)`** ——
            #   `id()` 在对象被回收后会**复用**，会让缓存返回**另一个 handler 的签名**
            #   （实测踩到：临时闭包被回收后 id 撞车 ⇒ 合法参数被当成未知参数剔除）。
            if name not in self._allowed_cache:
                self._allowed_cache[name] = _allowed_args(spec.handler)
            allowed = self._allowed_cache[name]
            unknown = [k for k in args if allowed is not None and k not in allowed]
            for k in unknown:
                args.pop(k, None)
            for k in unknown:
                logger.warning(
                    "工具 %s 收到未声明参数 %s，已忽略（handler 不接受）", name, k)
            result = spec.handler(**args)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as e:  # noqa: BLE001 — 工具失败需被循环捕获并计数
            return ToolResult(ok=False, output=None, error=f"{type(e).__name__}: {e}")
