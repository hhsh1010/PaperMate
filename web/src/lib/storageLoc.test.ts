import { describe, expect, it } from "vitest";
import {
  buildStorageUpdateInput,
  formatSize,
  normalizeRootPath,
  STORAGE_SUBDIR_LABELS,
  storageNeedsRestart,
  storagePendingBanner,
} from "./storageLoc";
import type { StorageOut } from "./types";

const base: StorageOut = {
  available: true,
  unavailable_reason: null,
  config_path: "C:\\Users\\x\\AppData\\Local\\PaperMate\\app-settings.json",
  data_root: "C:\\Users\\x\\AppData\\Local\\PaperMate",
  pending_root: null,
  migrate: false,
  dirs: {},
  sizes_mb: {},
  drives: [],
  min_free_gb: 8,
  restart_required: false,
};

describe("normalizeRootPath", () => {
  it("去掉尾分隔符", () => {
    expect(normalizeRootPath("D:\\PaperMate\\")).toBe("D:\\PaperMate");
    expect(normalizeRootPath("D:\\PaperMate")).toBe("D:\\PaperMate");
  });

  it("★ 盘符根不能被削成没有斜杠", () => {
    // 早期实现会把 `D:\` 变成 `D:` ⇒ 后面所有路径拼接都少一层分隔符
    expect(normalizeRootPath("D:\\")).toBe("D:\\");
    expect(normalizeRootPath("D:")).toBe("D:\\");
  });

  it("保留 UNC 前缀", () => {
    expect(normalizeRootPath("\\\\server\\share\\")).toBe("\\\\server\\share");
  });

  it("去空白", () => {
    expect(normalizeRootPath("  D:\\X  ")).toBe("D:\\X");
  });
});

describe("buildStorageUpdateInput", () => {
  it("接受绝对路径并带上 migrate 标记", () => {
    const r = buildStorageUpdateInput("D:\\PaperMate", true);
    expect(r.ok).toBe(true);
    expect(r.value).toEqual({ data_root: "D:\\PaperMate", migrate: true });
  });

  it("拒绝空输入", () => {
    const r = buildStorageUpdateInput("   ", false);
    expect(r.ok).toBe(false);
    expect(r.error).toContain("请填写");
  });

  it("拒绝相对路径（★ 没有盘符就是相对路径）", () => {
    for (const bad of ["PaperMate", ".\\data", "data\\x"]) {
      const r = buildStorageUpdateInput(bad, false);
      expect(r.ok, `应拒绝：${bad}`).toBe(false);
      expect(r.error).toContain("完整路径");
    }
  });

  it("UNC 不在前端拦（由后端给出更权威的拒绝理由）", () => {
    const r = buildStorageUpdateInput("\\\\server\\share\\P", false);
    expect(r.ok).toBe(true);
  });

  it("返回的是**可直接提交**的载荷（已归一化）", () => {
    const r = buildStorageUpdateInput("  D:\\X\\  ", false);
    expect(r.value?.data_root).toBe("D:\\X");
  });
});

describe("storageNeedsRestart / storagePendingBanner", () => {
  it("没有请求时为 false 且横幅为 null", () => {
    expect(storageNeedsRestart(base)).toBe(false);
    expect(storagePendingBanner(base, "zh")).toBeNull();
    expect(storagePendingBanner(null, "zh")).toBeNull();
  });

  it("空白字符串不算待生效", () => {
    expect(storageNeedsRestart({ ...base, pending_root: "   " })).toBe(false);
  });

  it("有请求时横幅必须写清两件事：旧数据保留 + venv 重建", () => {
    const zh = storagePendingBanner({ ...base, pending_root: "D:\\X", migrate: true }, "zh") ?? "";
    expect(zh).toContain("D:\\X");
    expect(zh).toContain("重启");
    expect(zh).toContain("保留不删");
    expect(zh).toContain("重建");

    const en = storagePendingBanner({ ...base, pending_root: "D:\\X", migrate: true }, "en") ?? "";
    expect(en).toContain("restart");
    expect(en).toContain("kept");
  });

  it("不迁移时的横幅要说清是空目录启动", () => {
    const zh = storagePendingBanner({ ...base, pending_root: "D:\\X", migrate: false }, "zh") ?? "";
    expect(zh).toContain("空数据目录");
    expect(zh).not.toContain("搬迁");
  });
});

describe("formatSize", () => {
  it("MB / GB 自动切换", () => {
    expect(formatSize(512, false, "zh")).toBe("512.0 MB");
    expect(formatSize(2048, false, "zh")).toBe("2.00 GB");
  });

  it("★ 截断时必须标注（否则下界会被当成真值）", () => {
    expect(formatSize(512, true, "zh")).toContain("部分统计");
    expect(formatSize(512, true, "en")).toContain("+");
  });

  it("缺失时给占位符而不是 NaN", () => {
    expect(formatSize(undefined, false, "zh")).toBe("…");
    expect(formatSize(Number.NaN, false, "zh")).toBe("…");
  });
});

describe("STORAGE_SUBDIR_LABELS", () => {
  it("四个子目录都有标签（★ 必须与后端 SUBDIRS 一致）", () => {
    expect(Object.keys(STORAGE_SUBDIR_LABELS).sort()).toEqual(["data", "huggingface", "uv-cache", "venv"]);
  });

  it("venv 的说明必须写明是重建而非搬迁", () => {
    expect(STORAGE_SUBDIR_LABELS.venv.hint).toContain("重建");
  });
});