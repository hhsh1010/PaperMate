/** 输入框上方两条预警横幅的判定逻辑（2026-09-17 从 ChatInput/ChatPage 抽出为纯函数）。
 *
 *  ## 为什么必须抽出来
 *
 * 用户报「联网提示还是会闪烁出现，无法正常显示」。实测（帧级采样）复现：
 * 点击联网开关后 **约 14~50ms 内** `.composer-warn` 从 1 条变 2 条、随后又变回 1 条 ——
 * 多出来的那条文案是 `当前模型不支持原生联网，联网开关不会生效`（**一句错误归因**）。
 *
 * 根因是两个状态**不同源**：
 *   - `webOn` 是**乐观更新**（点击瞬间就变）—— 当轮的真相；
 *   - `webCapability` 是**上一次请求**的产物 —— 属于过去。
 * 点击「开启」的瞬间：`webOn=true`（新）配 `available=false`（旧，由上一次
 * 「开关关闭」的请求算出，因为后端 `available = enabled and supports`）
 * → 判定成立 → 提示出现；等新请求回来 → `available=true` → 提示消失 = 闪一下。
 *
 * 所以修法不是"调文案"，而是**给能力数据打请求指纹**：
 * 这份数据是「以什么输入」请求来的就记下来，只有指纹与当前输入一致时才允许用于预警。
 * 指纹不匹配 = 数据与当前状态可能矛盾 → 一律不显示（宁可晚一拍，也不要闪错误提示）。
 *
 * 抽成纯函数的额外收益：闪烁属于**时序行为**，组件内很难测；判定逻辑独立后
 * 可以像 `reducer` / `sse` 一样用 vitest 直接钉住（要闪必须先过这个函数）。
 */

import type { NetworkCapabilityOut } from "./types";

/** 能力数据的请求指纹：这份数据是「以这组输入」请求回来的。
 *
 * 刻意包含 `model` —— 切换模型会改变 `effective_model`，从而改变能力结论；
 * 旧模型的能力数据用来判断新模型，同样是**错配**（与乐观更新窗口同性质）。
 */
export function capKeyOf(input: {
  sessionId: number | null;
  webOn: boolean;
  docsLen: number;
  model: string;
}): string {
  const { sessionId, webOn, docsLen, model } = input;
  return `${sessionId ?? 0}|${webOn ? 1 : 0}|${docsLen}|${model}`;
}

export type WarnKind = "web" | "risk";

export interface ComposerWarn {
  /** 关闭记忆用的指纹：会话 + 类型 + **文案** 任一变化 → 提示自动重新出现 */
  key: string;
  text: string;
  kind: WarnKind;
}

export interface WarnDecisionInput {
  sessionId: number | null;
  /** 联网开关的**当前**状态（乐观更新后） */
  webOn: boolean;
  /** 当前关联文档数（doc-bound 会影响联网能力） */
  docsLen: number;
  /** 当前所选模型 */
  model: string;
  /** 能力数据（可能为陈旧值，故必须配 `capKey` 一起看） */
  cap: NetworkCapabilityOut | null;
  /** `cap` 的请求指纹（发起请求时算好的，不是响应时）*/
  capKey: string | null;
  /** 用户已点过「关闭」的提示指纹 */
  dismissed: string[];
}

/** 当前应显示的预警横幅（空数组 = 不显示任何横幅）。 */
export function composerWarns(input: WarnDecisionInput): ComposerWarn[] {
  const { sessionId, webOn, docsLen, model, cap, capKey, dismissed } = input;

  // 指纹不符 = 这份数据属于「别的输入」（开关刚被点、正在切会话/换模型）
  // → 它的结论可能与当前状态相反，**绝不能**用来预警。等下一次请求回来即可。
  const want = capKeyOf({ sessionId, webOn, docsLen, model });
  if (!cap || capKey !== want) return [];

  // 第二道闸门：**后端回显的开关状态必须与前端当前一致**（2026-09-17 实测补）。
  //
  // 指纹只管得住"这份数据是拿什么输入请求的"，管不住"请求发出时后端落库了没有"：
  // 点开关是**乐观更新**（立刻改前端 state），而持久化是异步 PATCH ——
  // 拉取能力的请求完全可能**抢在 PATCH 落库之前**到达后端，于是后端仍按旧开关状态回答
  // （`enabled=false`、`reason=""`，因为它认为开关关着）。
  // 此时前端 `webOn=true` 配 `supports_native=false` → 会显示一句**归因错误**的兜底文案
  // （实测复现：文案是"当前模型不具备原生联网能力"，而真实原因只是"请求抢跑了"）。
  //
  // 用后端回显的 `enabled` 做一致性校验即可堵住：不一致 = 这份数据不反映当前开关状态 → 不显示。
  // 配套：ChatPage 在 PATCH **成功后**会 `bump` 一次 epoch 触发重取，保证正确数据最终到位。
  if (cap.enabled !== webOn) return [];

  const out: ComposerWarn[] = [];

  // ① 联网开关已开但当前模型不具备原生联网能力。
  // 判据用 `supports_native`（**纯能力**，与开关状态无关），**不用** `available`
  // —— 后者是 `enabled and supports`，开关关着时也是 false，用它会把"开关关着"
  //    误读成"能力不具备"（这正是旧实现闪现错误归因的来源）。
  if (webOn && cap.supports_native === false) {
    const text = cap.reason || "当前模型不具备原生联网能力，联网开关不会生效";
    const key = `web|${sessionId ?? 0}|${text}`;
    if (!dismissed.includes(key)) out.push({ key, text, kind: "web" });
  }

  // ② 「绑定文档 + 所选模型」组合有跳过检索的风险（策略 = 警示但放行，不拦截发送）。
  if (cap.model_risk) {
    const text = cap.model_risk_reason || "当前模型在绑定文档的会话中可能跳过检索，回复请核对引用。";
    const key = `risk|${sessionId ?? 0}|${text}`;
    if (!dismissed.includes(key)) out.push({ key, text, kind: "risk" });
  }

  return out;
}
