"""生成占位应用图标：源图（1024x1024）+ Tauri 打包所需的**全套图标文件**。

为什么需要它
------------
`tauri.conf.json` 的 `bundle.icon` 列了 5 个文件，**缺任何一个 `tauri-build` 都会失败**：

    32x32.png   128x128.png   128x128@2x.png   icon.ico   icon.icns

而 `cargo check` **阶段**就会校验（要生成 Windows Resource），不是等到打包
⇒ 报错形如：
    `...icons/icon.ico` not found; required for generating a Windows Resource file

官方做法是 `npx tauri icon <源图>`（Tauri 自己的 CLI，最可靠），但它需要先
`npm install` 装 @tauri-apps/cli。**本脚本用 Pillow 直接产出同一套文件**，
让"没有 npm 也能先把 cargo 跑通"这条路成立（实测 Pillow 可写多尺寸 ICO 与 ICNS）。

★ 正式发布前请用真实 logo 覆盖源图，并**优先**用官方 CLI 重新生成：
    cd desktop ; npm install ; npx tauri icon src-tauri/icons/logo-source.png

用法（项目根执行）：
    .venv/Scripts/python.exe desktop/scripts/gen-icon-placeholder.py
    .venv/Scripts/python.exe desktop/scripts/gen-icon-placeholder.py --source-only   # 只要源图
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024

#: Tauri 要求的文件名 → 目标边长（与 tauri.conf.json 的 bundle.icon 一一对应）
ICON_SET: dict[str, int] = {
    "32x32.png": 32,
    "128x128.png": 128,
    "128x128@2x.png": 256,
}

#: Windows .ico 内含的尺寸档位。★ 16/32/48 用于资源管理器与任务栏，256 用于大图标视图；
#: 少档位会让某些场景显示成模糊的放大图。
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

#: macOS .icns 的常见基准尺寸
ICNS_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def _rounded_mask(size: int, radius_ratio: float = 0.22) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=int(size * radius_ratio), fill=255
    )
    return m


def build(text_blocks: int = 4) -> Image.Image:
    """画一张 1024x1024 的占位图标（RGBA）。"""
    base = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # ① 底：深色斜向渐变（蓝紫调，贴合项目整体配色）
    #    ★ 先在 64x64 小图上画再 LANCZOS 放大 —— 直接逐像素画 1024² 既慢又容易出现色带/条纹。
    small = 64
    grad = Image.new("RGBA", (small, small))
    gpx = grad.load()
    c0 = (99, 102, 241)    # indigo-500
    c1 = (139, 92, 246)    # violet-500
    for y in range(small):
        for x in range(small):
            t = x / (small - 1) * 0.55 + y / (small - 1) * 0.45
            gpx[x, y] = (
                int(c0[0] + (c1[0] - c0[0]) * t),
                int(c0[1] + (c1[1] - c0[1]) * t),
                int(c0[2] + (c1[2] - c0[2]) * t),
                255,
            )
    grad = grad.resize((SIZE, SIZE), Image.LANCZOS)
    base.paste(grad, (0, 0), _rounded_mask(SIZE))

    d = ImageDraw.Draw(base)

    # ② 文档主体（白色圆角矩形）
    m = int(SIZE * 0.20)
    doc_w = int(SIZE * 0.46)
    doc_h = int(SIZE * 0.56)
    doc_x = m
    doc_y = (SIZE - doc_h) // 2
    d.rounded_rectangle(
        [doc_x, doc_y, doc_x + doc_w, doc_y + doc_h],
        radius=int(SIZE * 0.05), fill=(255, 255, 255, 245),
    )

    # ③ 文本行（灰色，宽度递减，像正文）
    line_y = doc_y + int(doc_h * 0.22)
    line_h = max(6, int(SIZE * 0.022))
    gap = int(SIZE * 0.058)
    for i in range(text_blocks):
        w = int(doc_w * (0.74 if i == 0 else 0.58 - 0.06 * i))
        if w <= 0:
            break
        d.rounded_rectangle(
            [doc_x + int(doc_w * 0.12), line_y,
             doc_x + int(doc_w * 0.12) + w, line_y + line_h],
            radius=line_h // 2, fill=(148, 163, 184, 255),
        )
        line_y += gap

    # ④ 高亮条（青色，点题"AI 增强"）
    bar_x = doc_x + doc_w + int(SIZE * 0.04)
    bar_w = int(SIZE * 0.085)
    d.rounded_rectangle(
        [bar_x, doc_y + int(doc_h * 0.18), bar_x + bar_w, doc_y + int(doc_h * 0.52)],
        radius=int(SIZE * 0.02), fill=(34, 211, 238, 255),
    )

    # ⑤ 右下角小圆点（装饰，避免右侧过空）
    r = int(SIZE * 0.035)
    cx, cy = doc_x + doc_w + int(SIZE * 0.082), doc_y + int(doc_h * 0.66)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 200))

    return base


def emit_icon_set(src: Path, out_dir: Path) -> list[Path]:
    """从源图产出 Tauri 要求的全套图标，返回写出的文件列表。"""
    im = Image.open(src).convert("RGBA")
    written: list[Path] = []

    for name, size in ICON_SET.items():
        p = out_dir / name
        im.resize((size, size), Image.LANCZOS).save(p, format="PNG")
        written.append(p)

    # Windows .ico：多尺寸打包在**一个**文件里
    ico = out_dir / "icon.ico"
    im.save(ico, format="ICO", sizes=ICO_SIZES)
    written.append(ico)

    # macOS .icns：Pillow 支持直接写出（实测可用）；失败不致命 ——
    # Windows 构建的失败点是 icon.ico，icns 只在 macOS 打包时用得到。
    icns = out_dir / "icon.icns"
    try:
        im.resize((1024, 1024), Image.LANCZOS).save(icns, format="ICNS")
        written.append(icns)
    except Exception as e:  # noqa: BLE001
        print(f"[gen-icon] ⚠️ .icns 生成失败（Windows 构建不受影响）：{type(e).__name__}: {e}")

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="生成占位应用图标（源图 + Tauri 全套）")
    ap.add_argument(
        "--out",
        default="desktop/src-tauri/icons/logo-source.png",
        help="源图输出路径（默认 desktop/src-tauri/icons/logo-source.png）",
    )
    ap.add_argument(
        "--source-only",
        action="store_true",
        help="只生成 1024x1024 源图，不产出 Tauri 图标集（留给 `npx tauri icon` 生成）",
    )
    args = ap.parse_args()

    src = Path(args.out)
    src.parent.mkdir(parents=True, exist_ok=True)
    img = build()
    img.save(src, format="PNG")
    print(f"[gen-icon] 源图已写入 {src}（{img.size[0]}x{img.size[1]}）")

    if args.source_only:
        print(f'[gen-icon] 下一步：cd desktop && npm install && npx tauri icon "../{src.as_posix()}"')
        return 0

    for p in emit_icon_set(src, src.parent):
        print(f"[gen-icon] 已写入 {p}（{p.stat().st_size} 字节）")
    print("[gen-icon] ✅ Tauri 所需的 5 个图标文件已就位，`cargo check` 可继续")
    print("[gen-icon] ⚠️ 正式发布前：用真实 logo 覆盖源图后，优先用 `npx tauri icon` 重新生成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
