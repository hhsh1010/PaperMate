"""PaperMate · 端到端忠实度看板路由（F11：体验优化 P2）

把已落地的 dev_eval 门禁以用户可读的结构化接口暴露：
- 检索基线：eval/eval_report.json（avg_recall@5 / avg_mrr@10）+ 门禁判定；
- 运行期 Agent 指标：最近 10 条 Trace 聚合（工具成功率 / 溯源率 / 平均成本 / 平均步数）。

指标计算复用 `skills/dev_eval.collect_metrics`（与 /eval 命令单一口径，避免漂移）。

门控说明：本接口为**只读聚合**（无 LLM 调用、无成本、无副作用），不属于三硬约束的
「LLM 调用模块」范畴（与文献集路由同理，不设独立开关）。`features.dev_eval` 门控的是
`/eval` 技能（LLM 命令）本身，二者语义不同：若看板跟随该开关（默认 False），
会导致「功能已交付但默认不可见」。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import EvalSummaryOut
from ..skills.dev_eval import collect_metrics

router = APIRouter(tags=["eval"])


@router.get("/eval/summary", response_model=EvalSummaryOut)
def eval_summary(request: Request) -> dict:
    """端到端忠实度看板数据源（检索基线 + 运行期 Agent 指标 + 门禁判定）。"""
    session_svc = request.app.state.session_svc
    # collect_metrics 为同步 DB 读取；def 路由由 FastAPI 放线程池执行，不阻塞事件循环
    return collect_metrics(session_svc.session_factory)
