/** 侧栏排序规则的**行为**测试（2026-10-02 置顶）。
 *
 * ## 为什么必须是行为测试，而不是源码文本断言
 *
 * 初版我把排序逻辑内联在 `onTogglePin` 里，用 `?raw` 文本断言守着。
 * 反向自检时我注入"去掉重排"（把 `.sort(...)` 换成直接 return），
 * **12 条文本断言全绿** —— 因为排序体的**字面量仍然存在于文件里**，
 * 断言只是"文件里有这段文本"，而**没有被执行**。
 *
 * ⇒ 结论（已写入 `App.tsx` 的注释与 MEMORY）：
 * **排序/分支这类"逻辑正确性"必须用行为测试**；文本契约只适合守
 * "某段代码存在/某字段被传了"这类**结构**事实。
 *
 * ★ 另一条实测教训：那次注入把文件改成了**语法错误**（残留悬空 `if`），
 * 而 vitest 因为只读 `?raw` 文本**不编译** ⇒ 全绿假象。
 * **`?raw` 契约测试对"语法是否正确"零覆盖**，必须配`tsc --noEmit`。
 */
import { describe, it, expect } from "vitest";
import { sortSessionsForSidebar } from "../App";

type Row = { id: number; title: string; pinned?: boolean; updated_at: string | null };

const r = (id: number, title: string, pinned: boolean, updated_at: string): Row => ({
  id,
  title,
  pinned,
  updated_at,
});

const titles = (list: Row[]) => list.map((x) => x.title);

describe("侧栏排序 · sortSessionsForSidebar（行为级）", () => {
  it("★ 置顶优先，压过活动时间", () => {
    const old = r(1, "很久没动但已置顶", true, "2026-10-01T00:00:00");
    const fresh = r(2, "刚创建未置顶", false, "2026-10-02T10:00:00");
    expect(titles(sortSessionsForSidebar([old, fresh]))).toEqual([
      "很久没动但已置顶",
      "刚创建未置顶",
    ]);
    expect(titles(sortSessionsForSidebar([fresh, old]))).toEqual([
      "很久没动但已置顶",
      "刚创建未置顶",
    ]);
  });

  it("组内按最近活动倒序（置顶项之间也一样）", () => {
    const list = [
      r(1, "A", true, "2026-10-01T00:00:00"),
      r(2, "B", true, "2026-10-02T00:00:00"),
      r(3, "C", false, "2026-10-01T12:00:00"),
      r(4, "D", false, "2026-10-02T06:00:00"),
    ];
    expect(titles(sortSessionsForSidebar(list))).toEqual(["B", "A", "D", "C"]);
  });

  it("★ 不修改入参（纯函数）", () => {
    const list = [r(1, "A", false, "2026-10-01T00:00:00"), r(2, "B", true, "2026-10-01T00:00:00")];
    const before = list.map((x) => x.id);
    sortSessionsForSidebar(list);
    expect(list.map((x) => x.id)).toEqual(before);
  });

  it("pinned 缺省（undefined / 老数据）视为未置顶", () => {
    const list = [
      { id: 1, title: "新接口", updated_at: "2026-10-01T00:00:00" } as Row,
      r(2, "已置顶", true, "2026-10-01T00:00:00"),
    ];
    expect(titles(sortSessionsForSidebar(list))).toEqual(["已置顶", "新接口"]);
  });

  it("updated_at 为 null（老数据）不会抛错，排在有时间的之后", () => {
    const list = [r(1, "无时间", false, null as unknown as string), r(2, "有时间", false, "2026-10-01T00:00:00")];
    expect(() => sortSessionsForSidebar(list)).not.toThrow();
    expect(titles(sortSessionsForSidebar(list))[0]).toBe("有时间");
  });

  it("★ 取消置顶 ⇒ 回到它本来的活动位置（不是被顶到最前）", () => {
    // A 最早创建（活动最旧）→ 置顶 → 取消后应回到末位
    const a = r(1, "A", false, "2026-10-01T00:00:00");
    const b = r(2, "B", false, "2026-10-01T06:00:00");
    const pinnedList = sortSessionsForSidebar([a, b].map((x) => ({ ...x, pinned: x.id === 1 })));
    expect(titles(pinnedList)).toEqual(["A", "B"]);

    const unpinned = sortSessionsForSidebar(pinnedList.map((x) => ({ ...x, pinned: false })));
    expect(titles(unpinned)).toEqual(["B", "A"]);
  });

  it("空列表 / 单项不抛错", () => {
    expect(sortSessionsForSidebar([])).toEqual([]);
    const one = [r(1, "A", true, "2026-10-01T00:00:00")];
    expect(titles(sortSessionsForSidebar(one))).toEqual(["A"]);
  });
});
