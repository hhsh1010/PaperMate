"""PaperMate · 答案层判官（L0-3，`docs/优化路线图-按任务等级.md`）

**为什么需要它**：`run_eval.py` 只测**检索**（找没找到证据），而用户感受到的是**答案**。
本模块把评测从"检索指标"升级为"答案质量"，两份互补判分：

| 判分 | 样本 | 期望 | 失败含义 |
|---|---|---|---|
| `faithfulness` | 有答案的用例 | 忠于检索证据 | 答了但编了（幻觉） |
| `abstention` | **语料无答案**的负样本 | 如实说明没有 | 没答案却硬答（更危险） |

两者必须**同时**看：只测 faithfulness，"对什么都答"的策略会显得很好（它从不拒答，自然也不会拒错）；
只测 abstention 则漏掉幻觉。`run_eval --judge` 一次给出两个数字。

设计约束（每条都对应一个真实风险，勿绕过）：

1. **提示词单一来源** —— 模板从 `judge_prompts/{name}.md` 的代码块读取。改提示词只改 md，
   不会出现"文档里一套、代码里另一套"（本项目 `SYSTEM_PROMPT` 有两份同名不同内容的**前科**）。
2. **temperature=0 + 跨厂商** —— 判官与主模型不同源，避免自夸偏差（模板文档的要求，在代码里强制）。
3. **失败 ≠ 通过** —— JSON 解析失败 / 超时 / 调用异常 → 记 `error`，**既不算 faithful 也不算
   unfaithful，且不计入分母**；报告必须暴露 `errors` 计数，否则"分母悄悄缩小"会让指标变好看
   （与本项目"门禁假通过"是同一类病）。
4. **无密钥必须报错退出** —— 绝不静默产出"没有判官结论"的报告（那比不跑更糟：报告看起来正常）。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from lightcore.llm.base import ChatMessage, LLMResponse, Role
from lightcore.llm.router import PRESET_ENDPOINTS, OAICompatibleLLM

PROMPT_DIR = Path(__file__).resolve().parent / "judge_prompts"

#: 判官与答案生成的默认超时（单次调用）。判官超时**不应**拖垮整轮评测 —— 由调用方捕获后记 error。
DEFAULT_TIMEOUT_S = 60.0

#: 答案生成约束：只依据资料。负样本用例靠这段约束"敢说不知道"，否则 abstention 测的就不是模型能力，
#: 而是提示词有多"宽容"。
ANSWER_SYSTEM = (
    "你是论文阅读助手。**只依据【资料】**回答用户问题；资料中没有的信息绝不能补充、推测或"
    "用你自己的知识替代。若资料不足以回答，请明确说明「资料中没有相关信息」，不要给出猜测性结论。"
)


class JudgeUnavailable(RuntimeError):
    """判官不可用（未配置密钥 / 未注册厂商）—— 调用方应**报错退出**，不得静默跳过。"""


@dataclass
class JudgeVerdict:
    """单条判分结果。`ok=False` 时 `value` 必为 None（失败不等于任何一档结论）。"""

    ok: bool
    value: bool | None
    reason: str = ""
    error: str = ""


def load_prompt(name: str, *, template_dir: Path | None = None) -> str:
    """读取 `judge_prompts/{name}.md` 中的裁判 Prompt（**单一来源**，勿在代码里另抄一份）。"""
    path = (template_dir or PROMPT_DIR) / f"{name}.md"
    if not path.exists():
        raise JudgeUnavailable(f"判官提示词缺失：{path}")
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        raise JudgeUnavailable(f"判官提示词 {path} 中没有可用的 ``` 代码块模板")
    return blocks[0].strip()


def _render(template: str, **values: str) -> str:
    """填充判官模板。

    ⚠️ **不能用 `str.format`**：模板里含 JSON 示例（字面 `{` `}`），`format` 会把
    `{"faithful": true|false}` 当成占位符 → `KeyError: '"faithful"'`（本文件踩过一次）。
    改用显式替换，并对"模板缺占位符"**报错**：否则会静默把一份**没有资料**的 prompt 发给判官，
    判分结果毫无意义却看不出来。
    """
    out = template
    for key, val in values.items():
        token = "{" + key + "}"
        if token not in out:
            raise JudgeUnavailable(f"判官模板缺少占位符 {token}（改模板后请同步检查占位符）")
        out = out.replace(token, val)
    return out


def pick_judge_provider(main_provider: str) -> str:
    """跨厂商选择判官厂商：主 deepseek → 判官 qwen，反之亦然。

    同源自评会系统性偏乐观（模型倾向认为自己的输出忠实），所以这里**强制换厂商**。
    """
    return "qwen" if main_provider == "deepseek" else "deepseek"


def resolve_api_key(provider: str) -> tuple[str | None, str]:
    """按**与线上完全相同的优先级**取密钥，并返回其来源（供打印/审计）。

    优先级：运行时设置（`data/provider_settings.json` 的 `providers.<name>.api_key`）→ 环境变量。

    ⚠️ 为什么必须与线上同源（2026-09-25 实测踩到）：应用侧三处都是
    `api_key=pset.get("api_key")` **显式传入**（`chat_service:315`、`summary_service:216/240`、
    `routes_settings:148/191`），环境变量只是回退。本模块原先只读 `os.getenv` ⇒ 用户把 key 存在
    **设置页**（= pset）时，`--judge` 会报「缺少 API Key」，明明配了却说没配
    —— 与 T-00 的 `use_bge_sparse`、T-03 的重排档位是**同一类"两套口径"缺陷**。
    """
    endpoint = PRESET_ENDPOINTS.get(provider)
    try:                       # 读运行时设置失败不阻断：环境变量仍可能可用
        from papermate import runtime_settings

        pset = (runtime_settings.get_state().get("providers") or {}).get(provider) or {}
        key = (pset.get("api_key") or "").strip()
        if key:
            return key, "运行时设置 data/provider_settings.json"
    except Exception:  # noqa: BLE001
        pass
    if endpoint is not None:
        env_key = (os.getenv(endpoint.api_key_env) or "").strip()
        if env_key:
            return env_key, f"环境变量 {endpoint.api_key_env}"
    return None, "未配置"


def build_llm(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    client=None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> OAICompatibleLLM:
    """构造指定厂商的 LLM（密钥按线上优先级解析；`api_key` / `client` 供测试注入）。"""
    endpoint = PRESET_ENDPOINTS.get(provider)
    if endpoint is None:
        raise JudgeUnavailable(
            f"未知厂商 {provider!r}；可选：{'/'.join(sorted(PRESET_ENDPOINTS))}"
        )
    if model and model != endpoint.default_model:
        # 复用同一 base_url/api_key_env，仅换模型名
        endpoint = type(endpoint)(
            endpoint.name, endpoint.base_url, endpoint.api_key_env, model
        )
    key_source = "显式传入（测试/调用方）"
    if api_key is None:
        api_key, key_source = resolve_api_key(provider)
    llm = OAICompatibleLLM(endpoint, api_key=api_key, client=client, timeout=timeout)
    llm.key_source = key_source       #: 报告/日志用：说明这次用的是哪一层的密钥
    return llm


def ensure_available(llm: OAICompatibleLLM) -> None:
    """提前校验密钥 —— 避免跑完全部检索（几十秒 + GPU）才发现没配密钥。"""
    if not llm.api_key_available():
        raise JudgeUnavailable(
            f"判官不可用：缺少 {llm.endpoint.name} 的 API Key"
            f"（运行时设置 data/provider_settings.json 的 providers.{llm.endpoint.name}.api_key "
            f"与环境变量 {llm.endpoint.api_key_env} 均未配置）。"
            f"`--judge` 需要真实密钥才能产出判分；如需只跑检索，请去掉 `--judge`。"
        )


def _extract_json(text: str) -> dict | None:
    """从判官输出中取 JSON 对象：容忍 ```json 包裹与前后废话。"""
    if not text:
        return None
    candidates = [text.strip()]
    candidates += re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            data = json.loads(cand.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_verdict(text: str, key: str) -> JudgeVerdict:
    """把判官文本解析成 `JudgeVerdict`。**解析失败一律 ok=False**（绝不当成通过）。"""
    data = _extract_json(text)
    if data is None:
        return JudgeVerdict(ok=False, value=None, error=f"判官输出不是 JSON：{(text or '')[:120]!r}")
    if key not in data or not isinstance(data[key], bool):
        return JudgeVerdict(
            ok=False, value=None,
            error=f"判官 JSON 缺少布尔字段 {key!r}：{json.dumps(data, ensure_ascii=False)[:120]}",
        )
    return JudgeVerdict(ok=True, value=bool(data[key]), reason=str(data.get("reason", "")))


class Judge:
    """答案层判官（faithfulness / abstention）。单例复用同一 LLM 客户端。"""

    def __init__(
        self,
        llm: OAICompatibleLLM,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        template_dir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._timeout_s = timeout_s
        self._dir = template_dir

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    @property
    def provider(self) -> str:
        return self._llm.endpoint.name

    async def _ask(self, prompt: str, key: str) -> JudgeVerdict:
        """单次判分。任何异常都收敛为 `ok=False`（由调用方记 error），不让一轮判分拖垮整轮评测。"""
        try:
            resp: LLMResponse = await asyncio.wait_for(
                self._llm.chat(
                    [ChatMessage(role=Role.USER, content=prompt)],
                    temperature=0.0,  # 判分必须可复现（模板文档的硬要求）
                    max_tokens=512,
                ),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            return JudgeVerdict(ok=False, value=None, error=f"判官超时（>{self._timeout_s}s）")
        except Exception as e:  # noqa: BLE001 判官异常不得向上炸掉整轮评测
            return JudgeVerdict(ok=False, value=None, error=f"{type(e).__name__}: {e}")
        return parse_verdict(resp.message.content or "", key)

    async def faithfulness(self, *, context: str, answer: str) -> JudgeVerdict:
        """判断回答是否**完全基于**上下文（防幻觉）。"""
        tpl = load_prompt("faithfulness", template_dir=self._dir)
        return await self._ask(_render(tpl, context=context, answer=answer), "faithful")

    async def abstention(self, *, context: str, question: str, answer: str) -> JudgeVerdict:
        """判断"资料里没有答案"时回答是否**如实说明**（防硬编）。"""
        tpl = load_prompt("abstention", template_dir=self._dir)
        return await self._ask(
            _render(tpl, context=context, question=question, answer=answer), "abstained"
        )

    async def edit_fidelity(self, *, instruction: str, before: str, after: str) -> JudgeVerdict:
        """A3（2026-09-21）：判断改写是否**保住了原文事实**且**真的执行了指令**。

        为什么单独一个判官：改稿的失败模式与问答不同 —— 危险的不是"编造"，而是
        **把数字改错**（`77.4%` → `74.7%`）却读起来通顺，人眼抽查极难发现，
        而它会直接写进用户的论文。判定规则见 `judge_prompts/edit_fidelity.md`。
        """
        tpl = load_prompt("edit_fidelity", template_dir=self._dir)
        return await self._ask(
            _render(tpl, instruction=instruction, before=before, after=after), "faithful"
        )


async def generate_answer(
    llm: OAICompatibleLLM,
    *,
    question: str,
    context: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[str, str]:
    """用**主模型**依资料生成答案，返回 `(answer, error)`；error 非空表示生成失败。

    评测的是"检索+生成"整链的产出，所以这里必须走真实模型（temperature=0 保证可复现）。
    """
    user = f"【资料】\n{context}\n\n【问题】\n{question}"
    try:
        resp = await asyncio.wait_for(
            llm.chat(
                [
                    ChatMessage(role=Role.SYSTEM, content=ANSWER_SYSTEM),
                    ChatMessage(role=Role.USER, content=user),
                ],
                temperature=0.0,
                max_tokens=800,
            ),
            timeout=timeout_s,
        )
    except TimeoutError:
        return "", f"生成超时（>{timeout_s}s）"
    except Exception as e:  # noqa: BLE001 生成失败记为 error（该条不进判分分母）
        return "", f"{type(e).__name__}: {e}"
    return (resp.message.content or "").strip(), ""


def aggregate(verdicts: list[JudgeVerdict], key: str) -> dict:
    """聚合判分结果。

    ⚠️ 分母只算 `ok=True` 的样本；失败条数**必须**单独暴露（`errors`）——
    "失败不算分母"与"失败当通过"是两件事，前者如实、后者造假。
    """
    ok = [v for v in verdicts if v.ok]
    hits = sum(1 for v in ok if v.value)
    return {
        "judged": len(ok),
        "hits": hits,
        key: round(hits / len(ok), 4) if ok else None,
        "errors": len(verdicts) - len(ok),
        "error_samples": [v.error for v in verdicts if not v.ok][:5],
    }
