"""列表端点分页（C4，2026-09-21）。

**要解决的问题**（复检 §3 C4）：多数列表端点**没有分页** ——
`/documents`、`/sessions`、`/memories` 都是"有多少返多少"，只有 `/traces` 硬编码 `limit=50`。
数据量一上来（比如积累上千份文档），前端会一次性拿到全部并渲染。

## 关键决策：分页参数**可选**，返回体**保持数组**

有两种做法：
- (a) 返回 `{items, total, limit, offset}` —— 语义完整，但**破坏既有契约**：
  前端所有 `await api.getDocuments()` 都当数组用，改完要同步改前端与测试。
- (b) **返回仍是数组**，总数走 `X-Total-Count` 响应头 —— HTTP 惯例，**前端零改动**。

这里选 **(b)**。理由是"**契约变更的风险 > 分页的收益**"：
现有前端与测试全部按数组消费，为一个当前数据量下几乎用不到的能力去动契约，
是把风险放在收益前面。需要总数的调用方读响应头即可（`total` 语义在后端单一实现）。

⚠️ **默认上限**：`limit` 缺省为 `DEFAULT_LIMIT`（500）。
它比当前任何真实数据量都大（实测文档 22 / 会话 12），所以**默认行为无变化**；
但它把"无限返回"变成了"有界"，避免将来某天一次性返回上万条。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Query, Response

#: 默认页大小。比当前真实数据量大得多（不改变现状），但给"无限返回"加了个边界。
DEFAULT_LIMIT = 500

#: 单页上限（防止 `?limit=999999` 变相绕过）
MAX_LIMIT = 2000

#: 分页参数的标准注解（各路由复用，保证口径一致）。
#:
#: ⚠️ 必须用 `Annotated`，**不要**写成 `LimitParam = Query(...)` 再当默认值用 ——
#: `Query(...)` 返回的是**一个可变的 FieldInfo 实例**，跨端点/跨 app 共享同一个实例时，
#: FastAPI 注册时写入的状态会互相污染，表现为"某些端点**所有**请求都返回 422"。
#: 实测踩过：单独建一个 app 正常，测试里连续建多个 app 就全 422。
#: `Annotated` 只是类型注解，每个端点各自生成参数定义，没有共享可变状态。
LimitParam = Annotated[int, Query(ge=1, le=MAX_LIMIT, description="本页最多返回多少条")]
OffsetParam = Annotated[int, Query(ge=0, description="跳过前多少条")]


def paginate(query, response: Response, *, limit: int, offset: int) -> list:
    """对 SQLAlchemy 查询分页，并把总数写入 `X-Total-Count`。

    ⚠️ `count()` 前先 `order_by(None)`：某些查询带的排序会让 SQLite 的 count 变慢，
    且排序对计数毫无意义（SQLAlchemy 生成的 `SELECT count(*)` 本就不该带 ORDER BY）。
    """
    total = query.order_by(None).count()
    response.headers["X-Total-Count"] = str(total)
    # 暴露给浏览器端 JS 读取（同源下 CORS 若开启，需显式列出）
    response.headers["Access-Control-Expose-Headers"] = "X-Total-Count"
    return query.offset(offset).limit(limit).all()
