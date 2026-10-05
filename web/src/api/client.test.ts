/**
 * 错误消息提取的单元测试（2026-09-15，P0-1 回流守护）。
 *
 * 背景：`extractDetail` 修的是「后端返回对象数组 detail 时，前端 `new Error(detail)`
 * 会得到 `[object Object]`」——用户完全看不懂哪里错了。
 *
 * 这里把五种可能的响应体形态都钉住：**这类"格式兼容"代码最容易在后续重构中被简化掉**，
 * 而一旦简化，用户端又会退回不可读报错。
 */

import { describe, expect, it } from "vitest";

import { extractDetail } from "./client";

describe("extractDetail", () => {
  it("字符串 detail 直接使用（后端统一错误出口的当前形态）", () => {
    expect(extractDetail({ detail: "会话不存在" }, "fallback")).toBe("会话不存在");
  });

  it("对象数组 detail 压平成可读消息（修复前的 [object Object] 场景）", () => {
    const body = {
      detail: [
        { type: "int_parsing", loc: ["path", "doc_id"], msg: "Input should be a valid integer" },
      ],
    };
    const out = extractDetail(body, "fallback");
    expect(out).toContain("Input should be a valid integer");
    expect(out).not.toContain("[object");
  });

  it("多条校验错误合并展示", () => {
    const body = {
      detail: [{ msg: "字段 A 必填" }, { msg: "字段 B 类型错误" }],
    };
    const out = extractDetail(body, "fallback");
    expect(out).toContain("字段 A 必填");
    expect(out).toContain("字段 B 类型错误");
  });

  it("无 detail 但有 code 时回退到「statusText（code）」", () => {
    expect(extractDetail({ code: "internal_error" }, "服务器错误")).toBe("服务器错误（internal_error）");
  });

  it("空体 / null / 非对象 时回退到 statusText（不抛错）", () => {
    expect(extractDetail(null, "Not Found")).toBe("Not Found");
    expect(extractDetail({}, "Not Found")).toBe("Not Found");
    expect(extractDetail(undefined, "Not Found")).toBe("Not Found");
    expect(extractDetail("plain text", "Not Found")).toBe("Not Found");
  });

  it("detail 为空字符串时不覆盖 fallback", () => {
    expect(extractDetail({ detail: "" }, "Bad Request")).toBe("Bad Request");
  });
});
