"""PaperMate · figure_enrich 能力验证脚本

用途：在装完 [figure] extra 后，一键确认 OCR / 表格结构还原 / VLM 三条能力线
的依赖是否就绪、开关是否已开，并给出启用指引。依赖缺失时优雅报告，不崩溃。

用法：
  uv run python scripts/verify_figure_enrich.py            # 只查依赖 + 开关
  uv run python scripts/verify_figure_enrich.py --full     # 额外对一张合成图真跑 OCR
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from types import SimpleNamespace

# 与 lightcore.rag.figure_content 一致：paddlepaddle 3.3.x CPU 的 oneDNN(PIR) bug，
# 需在 import paddleocr/paddlex 前禁用 paddlex 默认 oneDNN（用户显式设置过时尊重用户）。
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

DEPS: tuple[tuple[str, str], ...] = (
    ("paddleocr", "图片 OCR（中文友好）"),
    ("paddle", "PaddleOCR 3.x 推理引擎（paddlepaddle，import 名 paddle）"),
    ("transformers", "Table Transformer 表格结构还原"),
    ("torch", "表格结构还原推理后端（CPU）"),
    ("PIL", "图片读写"),
)


def _check_deps() -> dict[str, bool]:
    out: dict[str, bool] = {}
    for mod, _desc in DEPS:
        try:
            importlib.import_module(mod)
            out[mod] = True
        except Exception:  # noqa: BLE001
            out[mod] = False
    return out


def _current_switch() -> dict:
    try:
        from papermate.runtime_settings import get_state

        fe = get_state().get("figure_enrich") or {}
    except Exception:  # noqa: BLE001
        fe = {}
    if not fe:
        try:
            from papermate.config import get_settings

            fe = getattr(get_settings(), "figure_enrich", {}) or {}
        except Exception:  # noqa: BLE001
            fe = {}
    return fe


def main() -> int:
    ap = argparse.ArgumentParser(description="figure_enrich 能力验证")
    ap.add_argument("--full", action="store_true", help="对合成图真跑 OCR（可能下载 PaddleOCR 模型，较慢）")
    args = ap.parse_args()

    print("== figure_enrich 能力验证 ==")
    deps = _check_deps()
    for mod, desc in DEPS:
        print(f"  [{'OK' if deps[mod] else '缺'}] {mod:14s} {desc}")

    switch = _current_switch()
    enabled = bool(switch.get("enabled"))
    methods = switch.get("methods") or []
    print(f"\n开关状态：enabled={enabled}  methods={methods}")

    missing = [mod for mod, ok in deps.items() if not ok]
    if missing:
        print("\n✗ 依赖缺失：", ", ".join(missing))
        print("  请在【本地终端】安装：uv sync --extra rag --extra figure")
        print("  （或 uv pip install --python .venv/Scripts/python.exe paddleocr paddlepaddle transformers torch pillow）")
    if not enabled:
        print("\n✗ 开关未开启：上传时图内信息仍不会入库（增强被跳过）")
        print("  开启方式：设置页「图片增强」打开，或环境变量")
        print("  PAPERMATE_FIGURE_ENRICH='{\"enabled\":true,\"methods\":[\"ocr\",\"table\",\"vlm\"]}'")
    if not missing and enabled:
        print("\n✓ 依赖齐全 + 开关已开：上传含图 PDF/docx 后，图内文字/表格将进入检索索引。")
        print("  提示：VLM 图注还需 DEEPSEEK_API_KEY；--full 可对合成图真跑 OCR 验证。")

    if args.full and not missing:
        print("\n--full：对合成图执行 OCR 验证（首次会下载 PaddleOCR 权重）…")
        try:
            from PIL import Image, ImageDraw

            from lightcore.rag.figure_content import FigureEnricher

            img = Image.new("RGB", (320, 96), "white")
            d = ImageDraw.Draw(img)
            d.text((16, 32), "PaperMate OCR 验证 2026", fill="black")
            tmp = "data/_verify_figure.png"
            img.save(tmp)
            en = FigureEnricher(SimpleNamespace(figure_enrich={"enabled": True, "methods": ["ocr"]}))
            r = asyncio.run(en.enrich(tmp, kind="figure", page=1))
            print(f"  OCR 结果：{r['ocr_text'][:80]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  OCR 实跑失败（不影响上传降级）：{e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
