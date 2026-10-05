/** 相对时间展示：与用量页口径一致（3 天内相对、更早显示日期）。
 * 后端时间字段已统一带 UTC 时区标记（iso_utc），此处 new Date() 能正确按绝对时间解析，
 * 不再受浏览器本地时区影响（审查 P2-1）。
 */
export function timeAgo(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const hours = (Date.now() - d.getTime()) / 3_600_000;
  if (hours < 1) return "刚刚";
  if (hours < 24) return `${Math.floor(hours)} 小时前`;
  if (hours < 72) return `${Math.floor(hours / 24)} 天前`;
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
