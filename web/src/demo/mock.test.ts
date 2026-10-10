/**
 * 演示模式 mock 的行为测试（vitest，node 环境）。
 *
 * 为什么值得测：这个 mock 是 GitHub Pages 上那个静态 demo 的**唯一数据来源** ——
 * 它若是"看起来成功"地返回错形状，界面会静默错乱，而**静态站点上没人会看到报错**。
 * 因此这里钉住三件事：①核心端点返回可解析的 JSON；②未覆盖端点**明确 404**（不假装成功）；
 * ③聊天流真的按 SSE 帧（message_delta → message → done）下发 —— 前端用 getReader() 消费，与线上同构。
 */

import { describe, expect, it } from "vitest";

import { DEMO_PARAGRAPHS, demoRoute } from "./mock";

describe("演示模式 mock", () => {
  it("healthz 返回 ok，且明确不是对外暴露状态", async () => {
    const res = await demoRoute("/api/v1/healthz", "GET");
    expect(res?.status).toBe(200);
    const body = await res!.json();
    expect(body.status).toBe("ok");
    expect(body.network_exposed).toBe(false);
    expect(body.default_model).toBe("demo");
  });

  it("会话列表与段落视图返回可渲染的样例数据", async () => {
    const sessions = await (await demoRoute("/api/v1/sessions", "GET"))!.json();
    expect(Array.isArray(sessions)).toBe(true);
    expect(sessions[0].id).toBe(1);

    const paras = await (await demoRoute("/api/v1/documents/1/paragraphs", "GET"))!.json();
    expect(paras.length).toBe(DEMO_PARAGRAPHS.length);
    expect(paras.filter((p: { editable: boolean }) => p.editable).length).toBeGreaterThan(0);
  });

  it("未覆盖的端点必须返回 404 + demo_not_mocked（不许假装成功）", async () => {
    const res = await demoRoute("/api/v1/settings/whatever", "GET");
    expect(res?.status).toBe(404);
    const body = await res!.json();
    expect(body.code).toBe("demo_not_mocked");
    expect(body.detail).toContain("演示模式");
  });

  it("聊天流按 SSE 三种事件顺序下发（message_delta → message → done）", async () => {
    const res = await demoRoute("/api/v1/sessions/1/messages/stream", "POST");
    expect(res).not.toBeNull();
    expect(res!.headers.get("Content-Type")).toContain("text/event-stream");
    const text = await res!.text();
    const iDelta = text.indexOf("event: message_delta");
    const iMessage = text.indexOf("event: message\n");
    const iDone = text.indexOf("event: done");
    expect(iDelta).toBeGreaterThanOrEqual(0);
    expect(iMessage).toBeGreaterThan(iDelta);
    expect(iDone).toBeGreaterThan(iMessage);
    // 引用必须随最终 message 一并下发（否则引用卡片不会出现）
    expect(text).toContain("citations");
    expect(text).toContain("doc-1-c0");
  });
});
