# 忠实度（Faithfulness）评测 Prompt

用于 A-01 评估「生成内容是否忠于检索证据」。建议使用**跨厂商、temperature=0** 的裁判模型（与主模型不同源，避免自夸偏差）。

## 输入
- `context`：实际喂给生成模型的检索段落（含 chunk id 与页码）。
- `answer`：模型的最终回答文本。

## 裁判 Prompt
```
你是一个严谨的评测员。请判断【回答】是否完全基于【上下文】中的事实，没有编造、没有超出上下文的断言。

上下文：
{context}

回答：
{answer}

仅输出一个 JSON：{"faithful": true|false, "reason": "简要说明"}
```

## 判定规则
- `faithful = true` 且仅当回答中每条可被上下文支撑。
- 若回答出现「幻觉」或与上下文矛盾 → `faithful = false`。
- 聚合：faithfulness@N = N 条样本中 faithful=true 的比例。

## 备注
- 主模型若使用 DeepSeek/Qwen，裁判优先选对方厂商或 Claude/Gemini 等第三方，降低同源偏见。
- 本仓库 `run_eval.py` 默认不在线执行忠实度评测（避免强依赖外部大模型），仅在配置真实 API 后由自动化定期运行。
