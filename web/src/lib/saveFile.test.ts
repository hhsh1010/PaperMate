import { describe, expect, it } from "vitest";

import { bytesToBase64, isTauri, saveOutcomeMessage } from "./saveFile";

describe("saveFile · bytesToBase64", () => {
  it("encodes 3 bytes to 4 base64 chars", () => {
    expect(bytesToBase64(new Uint8Array([0x41, 0x42, 0x43]))).toBe("QUJD");
  });

  it("pads 5 bytes correctly (non-multiple-of-3)", () => {
    expect(bytesToBase64(new Uint8Array([0x00, 0xff, 0x10, 0x20, 0x30]))).toBe("AP8QIDA=");
  });

  it("handles inputs larger than the chunk size (0x8000)", () => {
    // 65536 个 'A' = 0x41：21845 组 "QUFB" + 尾字节 "QQ=="（不依赖 btoa，避免 jsdom 缺该 API）
    const big = new Uint8Array(0x10000).fill(0x41);
    expect(bytesToBase64(big)).toBe("QUFB".repeat(21845) + "QQ==");
  });

  it("empty input → empty string", () => {
    expect(bytesToBase64(new Uint8Array(0))).toBe("");
  });
});

describe("saveFile · isTauri", () => {
  it("detects via __TAURI_INTERNALS__.invoke", () => {
    const g = globalThis as unknown as { window?: { __TAURI_INTERNALS__?: unknown } };
    const orig = g.window;
    try {
      delete g.window;
      expect(isTauri()).toBe(false);
      g.window = { __TAURI_INTERNALS__: { invoke: () => {} } };
      expect(isTauri()).toBe(true);
    } finally {
      if (orig === undefined) delete g.window;
      else g.window = orig;
    }
  });
});

describe("saveFile · saveOutcomeMessage", () => {
  it("saved → 已保存到 <path>", () => {
    expect(saveOutcomeMessage({ kind: "saved", path: "D:\\导出\\paper.csv" })).toBe(
      "已保存到 D:\\导出\\paper.csv",
    );
  });
  it("cancelled → null（取消不打扰）", () => {
    expect(saveOutcomeMessage({ kind: "cancelled" })).toBeNull();
  });
  it("fallback → 已开始下载", () => {
    expect(saveOutcomeMessage({ kind: "fallback" })).toBe("已开始下载（保存到「下载」文件夹）。");
  });
});
