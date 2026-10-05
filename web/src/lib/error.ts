/**
 * 把未知错误收敛为可读消息。
 * - Error 实例：取其 message；
 * - 其余（字符串 / 对象 / undefined）：返回空串，由调用方用 `|| "默认文案"` 兜底，
 *   与改造前的 `e?.message ?? "默认文案"` 语义一致。
 */
export function toErrorMessage(e: unknown): string {
  return e instanceof Error ? e.message : "";
}
