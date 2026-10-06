"""生成占位应用图标源图（1024x1024 PNG）—— 供 `npx tauri icon` 生成全套图标。

为什么需要它
------------
`npx tauri icon <logo.png>` 是官方做法（Tauri 自己的 CLI 会正确产出 32x32.png /
128x128.png / 128x128@2x.png / icon.ico / icon.icns 全套格式）。
但它**必须有一张 ≥1024x1024 的方形源图**，而本项目目前只有 `web/public/hero-illustration.png`
（1184x760 的**横幅插画**，直接当图标会很难看）。

所以这里生成一张**几何风格**的占位图（不依赖任何字体文件，避免踩"字体不存在"的老坑）：
- 圆角方形底 + 斜向渐变（品牌感）
- 白色"文档 + 文本行"图形，暗示"论文阅读"
- 右上角一条青色高亮条，点题"AI"

★ 正式发布前请用真实 logo 替换：把 `--out` 指到你的 logo，或直接覆盖本文件后重跑
  `npx tauri icon desktop/src-tauri/icons/logo-source.png`。

用法（项目根执行）：
    .venv/Scripts/python.exe desktop/scripts/gen-icon-placeholder.py
    .venv/Scripts/python.exe desktop/scripts/gen-icon-placeholder.py --out desktop/src-tauri/icons/logo-source.png --text "PM"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024


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


def main() -> int:
    ap = argparse.ArgumentParser(description="生成占位应用图标源图（1024x1024 PNG）")
    ap.add_argument(
        "--out",
        default="desktop/src-tauri/icons/logo-source.png",
        help="输出路径（默认 desktop/src-tauri/icons/logo-source.png）",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = build()
    img.save(out, format="PNG")
    print(f"[gen-icon] 已写入 {out}（{img.size[0]}x{img.size[1]}）")
    print("[gen-icon] 下一步：cd desktop && npm install && npx tauri icon \"../%s\"" % out.as_posix())
    print("[gen-icon] 正式发布前请用真实 logo 覆盖该源图后重跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
