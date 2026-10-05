#!/usr/bin/env node
/**
 * C8（2026-09-21）：构建包装脚本 —— 固化「非 ASCII 路径」下的 workaround。
 *
 * **为什么需要它**（此前是口头约定，已被真实踩坑）：
 * 本环境会向子进程注入 `NODE_OPTIONS`（含指向**非 ASCII 路径**的 `--require` 文件），
 * 在部分 Node/工具链组合下会让 `tsc` / `vite` 直接失败。此前做法是让使用者手记一条
 * PowerShell 命令（`$env:NODE_OPTIONS = ""`）—— 换个 shell（bash）语法就不同，
 * 且**忘了就构建失败**。这是架构级约束，不该停在口头约定。
 *
 * 做法：在**本进程内**清掉继承来的 `NODE_OPTIONS`，再用 `spawn` 依次执行
 * `tsc --noEmit` → `vite build`。子进程继承的是已清理的环境，与 shell 种类无关。
 *
 * 用法：`npm run build`（package.json 已指向本脚本）
 */

import { spawn } from "node:child_process";

/** 继承来的 NODE_OPTIONS 会指向非 ASCII 路径的 --require 文件 —— 必须清掉 */
const inherited = process.env.NODE_OPTIONS;
if (inherited) {
  delete process.env.NODE_OPTIONS;
  console.log("[build] 已清除继承的 NODE_OPTIONS（非 ASCII 路径 workaround，见 C8）");
}

/** 依次执行命令；任一失败即中止，并保留原始退出码（CI/调用方要能感知失败） */
function run(command, args) {
  return new Promise((resolve, reject) => {
    console.log(`[build] $ ${command} ${args.join(" ")}`);
    const child = spawn(command, args, {
      stdio: "inherit",
      shell: process.platform === "win32", // Windows 下 npm 的 .cmd 需要 shell
      env: process.env,
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} 退出码 ${code}`));
    });
  });
}

try {
  await run("npx", ["tsc", "--noEmit"]);
  await run("npx", ["vite", "build"]);
  console.log("[build] ✓ 完成");
} catch (err) {
  console.error(`[build] ✗ ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}
