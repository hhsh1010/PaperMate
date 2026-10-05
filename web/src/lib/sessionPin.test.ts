/** 侧栏**置顶**的前端契约（2026-10-02）。
 *
 * ★ 与重命名并列为SessionRow 的第二个行内操作，因此"两处共用组件"这条
 *   同样适用于它（改一处即工作间 + 对话两段生效）。
 *
 * ## 为什么前端**必须本地重排**（而不只是改 pinned 标志）
 *
 * 列表顺序来自后端 `ORDER BY pinned DESC, updated_at DESC`。
 * 若只改标志不重排，置顶项会**留在原位** ⇒ 用户点了却看不到"钉住了"的效果，
 * 直到下次刷新才跳上去 —— 表现为"**点了没反应**"。
 *⇒ 前端复刻**同一条**规则重排（代价是"两处排序要同步"，已由本测试守护）。
 */
import { describe, it, expect } from "vitest";
import appSrc from "../App.tsx?raw";
import apiSrc from "../api/client.ts?raw";
import typesSrc from "../lib/types.ts?raw";
import iconsSrc from "../lib/icons.tsx?raw";

describe("侧栏 · 置顶", () => {
  it("★ 两段共用组件，且都接了 onTogglePin（改一处两处生效）", () => {
    expect(appSrc.match(/<SessionRow/g) || []).toHaveLength(2);
    expect(appSrc.split("onTogglePin={() => onTogglePin(s.id)}").length - 1).toBe(2);
  });

  it("★ 走独立端点 /pin（不复用 updateSession）", () => {
    // 置顶要**保持 updated_at 不变**，而改标题/改配置**应该**刷新它
    // ⇒ 两条路径的 updated_at 语义相反，无法共用一个合并更新。
    expect(apiSrc).toContain("async pinSession(");
    const m = apiSrc.match(/async pinSession\([\s\S]*?\n  \},/);
    expect(m?.[0]).toContain("/sessions/${id}/pin");
    expect(m?.[0]).toContain("method: \"PATCH\"");
    expect(m?.[0]).toContain("body: JSON.stringify({ pinned })");
  });

  it("★ 本地重排：pinned 降序 ⇒ updated_at 降序（与后端同规则）", () => {
    expect(appSrc).toContain("const onTogglePin = useCallback(");
    // 两级排序键都要在
    expect(appSrc).toContain("if (!!b.pinned !== !!a.pinned) return b.pinned ? 1 : -1");
    expect(appSrc).toContain("localeCompare(a.updated_at");
  });

  it("★ 失败时回滚**整份列表快照**（不只是标志）", () => {
    // 本地重排已经动过顺序 ⇒ 只回滚标志会留下"顺序错、标志对"的脏状态
    expect(appSrc).toContain("const snapshot = sessions;");
    expect(appSrc).toContain("setSessions(snapshot);");
    expect(appSrc).toContain("throw e;");
  });

  it("置顶按钮沿用 .session-del（移动端折叠态自动隐藏/显示）", () => {
    // ★ 与重命名同一条隐式契约：className 含 session-del ⇒
    //   折叠态的 display:none / sidebar-open 的 display:block 自动覆盖它。
    expect(appSrc).toContain("session-del session-pin");
  });

  it("★ 置顶按钮与删除按钮类名不同（配色才能分开）", () => {
    expect(appSrc).toContain("session-del session-rename");
    expect(appSrc).toContain("session-del session-pin");
    // 三个按钮互不串色
    expect(appSrc).toContain("session-del ${confirming ? \"confirming\" : \"\"}");
  });

  it("★ 置顶状态**常驻标记**（不 hover 才现）", () => {
    // 它是"这一条被钉住了"的**状态**，不是操作 ⇒ 隐藏起来置顶就失去意义。
    expect(appSrc).toContain("const pinned = session.pinned === true;");
    expect(appSrc).toContain("{pinned && (");
    expect(appSrc).toContain("ws-pin-mark");
  });

  it("pinned 缺省按 false 处理（老数据/老接口）", () => {
    // ⚠️ 不能写 `session.pinned` 直接当布尔：undefined 会让排序/样式判断出错
    expect(appSrc).toContain("session.pinned === true");
    expect(typesSrc).toContain("pinned?: boolean;");
    // 类型注释写在字段**上方** ⇒ 从字段位置往前取窗口
    const at = typesSrc.indexOf("pinned?: boolean;");
    expect(at).toBeGreaterThan(-1);
    const seg = typesSrc.slice(Math.max(0, at - 500), at + 100);
    expect(seg).toContain("不要在前端另排");
    expect(seg).toContain("后端");
  });

  it("无障碍：置顶按钮有 aria-pressed（是开关，不是动作）", () => {
    expect(appSrc).toContain("aria-pressed={pinned}");
    expect(appSrc).toContain('aria-label={pinned ? "取消置顶" : "置顶"}');
  });

  it("点击置顶按钮不触发行选中（不跳会话）", () => {
    expect(appSrc).toContain("void onTogglePin();");
    // 该 handler 内部必须有 stopPropagation（与其它行内按钮一致）
    const seg = appSrc.slice(appSrc.indexOf("session-pin ${pinned"), appSrc.indexOf("session-pin ${pinned") + 420);
    expect(seg).toContain("e.stopPropagation();");
  });

  it("图标已注册（Icons.pin）", () => {
    expect(iconsSrc).toContain("pin: icon(");
    expect(appSrc).toContain("<Icons.pin size={14} />");
    expect(appSrc).toContain("<Icons.pin size={12} />");
  });

  it("★ 前端不得自己按 pinned 过滤/分组（顺序只来自后端）", () => {
    // workspaceSessions / chatSessions 是 filter（保持原序）✔
    // 若有人改成 .sort()，就会与后端排序打架（两把尺子）
    expect(appSrc).toContain("const workspaceSessions = sessions.filter(isWorkspace);");
    const seg = appSrc.slice(appSrc.indexOf("const workspaceSessions"), appSrc.indexOf("const workspaceSessions") + 300);
    expect(seg).not.toContain(".sort(");
  });
});
