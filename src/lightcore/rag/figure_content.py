"""LightCore · 图片内容增强（ER-03 图表 · Level 2）

Level 1（figure_extractor）只把图片当资源保存、用启发式图注，图片内的文字/表格结构/
图表语义**不可检索**。本模块在 Level 1 之上做"内容理解"，使图片内信息可被检索与问答命中：

- **OCR（PaddleOCR，中文友好）**：截图型表格 / 扫描件里的文字转文本入索引
- **表格结构还原（Table Transformer，微软 DETR）**：把图片内表格还原为 Markdown 网格，
  并用 OCR 的单元格坐标回填文字（无 OCR 时仅保留空网格结构）
- **多模态 VLM 图注（DeepSeek-V4-Flash-Vision，复用 OAICompatibleLLM）**：对图表生成
  内容描述 / 摘要作为图注，覆盖启发式图注的不足

设计原则：
- **全部惰性导入**：paddleocr / transformers / torch / PIL 缺失时对应能力自动跳过，不阻断上传
- **优雅降级**：任一能力调用失败 → 该字段留空，不影响其余流程
- **配置开关**：受 `settings.figure_enrich`（enabled + methods）控制；默认关闭，
  避免上传期意外产生多模态 API 成本（需显式开启）
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("lightcore.rag.figure_content")

#: paddlepaddle 3.3.x CPU 的 oneDNN(PIR) 路径有缺陷
#（ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<...>]，见 onednn_instruction.cc），
# paddlex 默认开启 oneDNN（PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=True）会导致 OCR 抛 NotImplementedError。
# 在首次 import paddleocr/paddlex 前禁用默认 oneDNN 绕过；用户显式设置过该变量时尊重用户。
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")


def _apply_paddlex_env_defaults() -> None:
    """把 PaddleX（OCR / 表格结构）的**缓存根**与**模型源**收敛到本项目口径。

    幂等；**不覆盖**用户显式设置（全走 setdefault / 空值判断）。

    ★★ 为什么必须收敛（2026-10-09 P-07，端到端实测暴露）：
    1. **损坏残留 ⇒ 永久失败（死循环）**：默认缓存根是 `~/.paddlex`（用户全局），
       本机实测存在 2026-08/09 遗留的**半残模型目录**（`PP-OCRv6_medium_{det,rec}` 各仅 1MB：
       README + `inference.json`，**缺 `inference.pdiparams` 权重**）。而 PaddleX 的判据是
       **"目录存在即跳过下载"**（`paddleX/inference/utils/official_models.py:945` 原文
       "Model files already exist…please delete the directory manually"）⇒ 加载器永远报
       `No valid PaddlePaddle model found`，**用户看到的只是"OCR 装了却不工作"**。
       ⇒ 收敛到**专属缓存根**后，这类"外部残留"再也无法污染本应用（与本项目 HF 侧
       `hf_local.default_hub_cache_dir()` 的"缓存根唯一口径"一致；**禁硬编码 `~`**）。
    2. **默认模型源是 huggingface**（`paddlex/utils/flags.py:72`）⇒ 国内首次 OCR 走 HF
       （慢/可能失败）。实测切 **ModelScope ≈41MB/s**；`modelscope` 已是 `paddleocr` 依赖，无需额外安装。
    """
    if not os.environ.get("PADDLE_PDX_CACHE_HOME", "").strip():
        # 与 HF 缓存**同根**：`<default_hub_cache_dir() 的父目录>/paddlex-cache`
        #   桌面端 HF_HOME = `<数据根>/huggingface` ⇒ 实际落在 `<数据根>/huggingface/paddlex-cache`
        #   开发机未设 HF_HOME ⇒ `~/.cache/huggingface/paddlex-cache`
        from .hf_local import default_hub_cache_dir

        os.environ["PADDLE_PDX_CACHE_HOME"] = str(
            Path(default_hub_cache_dir()).parent / "paddlex-cache"
        )
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "ModelScope")


# ★ 必须在**首次 import paddleocr/paddlex 之前**执行（paddlex 在导入期读取这两个变量）
_apply_paddlex_env_defaults()

#: 支持的图片 MIME（用于 VLM data URL）
_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

#: Table Transformer structure-recognition 标签（microsoft/table-transformer-structure-recognition）
# 0 table column · 1 table row · 2 table spanning cell · 3 projected row header
_LABEL_TABLE_COLUMN = 0
_LABEL_TABLE_ROW = 1
_LABEL_SPANNING_CELL = 2

#: O-07（2026-10-01）：模型 ID 从 `_table` 内联抽出来 —— 单例 getter 与标签常量共用一处定义
_TABLE_MODEL_ID = "microsoft/table-transformer-structure-recognition"


class FigureEnricher:
    """对单张提取出的图片做内容增强。返回 {ocr_text, table_md, vlm_caption}。"""

    def __init__(self, settings=None, *, vlm_llm=None, cache=None):
        cfg = getattr(settings, "figure_enrich", None) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.methods = set(m for m in cfg.get("methods", []) if m in ("ocr", "table", "vlm"))
        # VLM 图注模型可配置（2026-08-25）：cfg["vlm_model"] ∈ MULTIMODAL_MODELS
        # （默认 deepseek-v4-flash-vision-exp；可切 qwen3-vl-flash 走百炼直连）
        self.vlm_model = cfg.get("vlm_model") or "deepseek-v4-flash-vision-exp"
        # 测试可注入假 LLM；生产为 None → 按 self.vlm_model 构造
        self._vlm_llm = vlm_llm
        # T-06 P1（2026-09-30）：图内容缓存（可选注入，协议 `get(key)` / `put(key, value)`）。
        # ★ 键由本类生成（图片内容哈希 + methods + vlm_model + kind）⇒ 缓存实现
        #   无需理解键语义，图片或配置任一变化自动 miss。未注入 = 不缓存（行为不变）。
        self._cache = cache
        # O-07（2026-10-01）：**实例级模型单例**（惰性构造，首次使用后常驻本实例）。
        # 此前 PaddleOCR / TableTransformer 都写在**逐图函数体**里 ⇒ N 张图 = N 次模型加载
        # （`from_pretrained` / `PaddleOCR(...)` 初始化都是**秒级**；Embedder / Reranker
        # 早就单例了，唯独这里没有）。★ 与 P1 图缓存**互补**：
        #   缓存管「重复图」（跨重建复用结果），本项管「同一次处理的 N 张图」（复用模型）。
        # 作用域 = **每个 FigureEnricher 实例**（document_service 每篇文档建一个）
        # ⇒ 一篇文档 N 张图只加载 1 次；跨文档重新加载以换取配置隔离（methods/vlm_model 可变）。
        self._ocr_engine: object | None = None
        self._ocr_is_v3: bool | None = None
        self._table_processor: object | None = None
        self._table_model: object | None = None
        #: 模型实际加载次数（观测用：验收「N 图只加载 1 次」；也可用于显存治理）
        self.model_loads = 0

    async def enrich(self, image_path: str | Path, *, kind: str = "figure", page: int | None = None) -> dict:
        """对单张图片做内容增强；缺失/未开启 → 对应字段为空串。

        返回 {"ocr_text", "table_md", "vlm_caption"}。

        **T-06 P1（2026-09-30）内容缓存**：注入 `cache` 后，以
        `sha256(图片字节) + methods + vlm_model + kind` 为键做 lookup/store ——
        未变图片**零次 OCR/VLM 调用**（改稿重建的主要提速来源：VLM 占 doc10 重建 75%）。
        缓存读写任何异常都只记日志并退回真实增强（**缓存坏了绝不能阻断处理**）。
        """
        out: dict[str, str] = {"ocr_text": "", "table_md": "", "vlm_caption": ""}
        if not self.enabled or not self.methods:
            return out
        path = Path(image_path)
        if not path.exists():
            return out

        cache_key: str | None = None
        if self._cache is not None:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                cache_key = (
                    f"{digest}:{self.vlm_model}:{','.join(sorted(self.methods))}:{kind}"
                )
                cached = self._cache.get(cache_key)
                if isinstance(cached, dict):
                    return {**out, **{k: str(cached.get(k) or "") for k in out}}
            except Exception:  # noqa: BLE001  缓存读失败 ⇒ 退回真实增强
                logger.warning("图内容缓存读取失败，忽略缓存", exc_info=True)
                cache_key = None

        # OCR 文本既可直接入索引，也供表格结构还原回填单元格
        need_ocr = "ocr" in self.methods or "table" in self.methods
        ocr_lines = self._ocr_lines(path) if need_ocr else []
        if "ocr" in self.methods:
            out["ocr_text"] = "\n".join(t for t, _ in ocr_lines).strip()
        if "table" in self.methods:
            out["table_md"] = self._table(path, ocr_lines)
        if "vlm" in self.methods:
            # 2026-08-25 去重：本地 OCR/表格已产出的信息，VLM 不再重复转录/还原行列，
            # 只做语义层（图表类型/趋势/结论），省 token 且避免同图同质劳动。
            out["vlm_caption"] = (
                await self._vlm(
                    path,
                    kind=kind,
                    local_text_done=bool(out["ocr_text"]) or bool(out["table_md"]),
                    table_done=bool(out["table_md"]),
                )
            ).strip()

        if cache_key is not None and any(out.values()):
            try:
                self._cache.put(cache_key, dict(out))
            except Exception:  # noqa: BLE001  缓存写失败不影响本次结果
                logger.warning("图内容缓存写入失败", exc_info=True)
        return out

    # ------------------------------------------------------------------ 模型单例（O-07）
    def _get_ocr(self, paddleocr) -> tuple[object, bool]:
        """惰性单例：PaddleOCR 引擎，返回 `(引擎, 是否 3.x)`；首次构造后常驻本实例。

        版本兼容（2.x / 3.x）的细节见 `_ocr_lines` 的 docstring —— 逻辑原样搬来，
        只把"每次调用都构造"改为"首次构造后复用"。
        """
        if self._ocr_engine is None:
            major = int(paddleocr.__version__.split(".")[0])
            if major >= 3:  # paddleocr 3.x
                try:
                    self._ocr_engine = paddleocr.PaddleOCR(lang="ch", use_textline_orientation=True)
                except TypeError:  # 个别子版本参数名差异
                    self._ocr_engine = paddleocr.PaddleOCR(lang="ch")
                self._ocr_is_v3 = True
            else:  # paddleocr 2.x
                self._ocr_engine = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
                self._ocr_is_v3 = False
            self.model_loads += 1
        return self._ocr_engine, self._ocr_is_v3

    def _get_table_models(self) -> tuple[object, object]:
        """惰性单例：Table Transformer 的 `(processor, model)`。

        `transformers` 缺失时抛 ImportError —— 由 `_table` 降级为"跳过表格结构还原"
        （与原实现同口径：**任一能力不可用都不阻断上传**）。
        """
        if self._table_processor is None or self._table_model is None:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection

            self._table_processor = AutoImageProcessor.from_pretrained(_TABLE_MODEL_ID)
            self._table_model = TableTransformerForObjectDetection.from_pretrained(_TABLE_MODEL_ID)
            self.model_loads += 1
        return self._table_processor, self._table_model

    # ------------------------------------------------------------------ OCR
    def _ocr_lines(self, path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
        """PaddleOCR 逐行识别，返回 [(文本, (x0,y0,x1,y1)), ...]；缺失/失败返回空。

        兼容 paddleocr 2.x（use_angle_cls + ocr.ocr）与 3.x（use_textline_orientation + predict，
        结果字段 rec_texts/rec_polys/rec_boxes）：用 __version__ 主版本显式切换。
        注意：不能靠 except TypeError 探测——paddleocr 3.x 对未知参数（如 show_log）抛的不是
        TypeError（2026-08-25 实测 `Unknown argument: show_log` 未被捕获，OCR 静默降级为空）。
        """
        try:
            import paddleocr
        except Exception:  # noqa: BLE001
            logger.info("[figure_content] PaddleOCR 未安装，跳过 OCR")
            return []
        try:
            ocr, is_v3 = self._get_ocr(paddleocr)
            result = ocr.predict(str(path)) if is_v3 else ocr.ocr(str(path), cls=True)
            lines: list[tuple[str, tuple[float, float, float, float]]] = []
            if is_v3:
                for block in result or []:
                    if not isinstance(block, dict):
                        continue
                    texts = block.get("rec_texts") or []
                    polys = block.get("rec_polys") or block.get("rec_boxes") or []
                    for i, text in enumerate(texts):
                        text = (text or "").strip()
                        if not text:
                            continue
                        lines.append((text, self._bbox_from_pts(polys[i] if i < len(polys) else None)))
            else:
                for block in result or []:
                    for item in block or []:
                        if not isinstance(item, list | tuple) or len(item) < 2:
                            continue
                        pts, text_score = item[0], item[1]
                        text = text_score[0] if isinstance(text_score, list | tuple) else str(text_score)
                        text = (text or "").strip()
                        if not text:
                            continue
                        lines.append((text, self._bbox_from_pts(pts)))
            return lines
        except Exception as e:  # noqa: BLE001
            logger.warning("[figure_content] OCR 失败：%s", e)
            return []

    @staticmethod
    def _bbox_from_pts(pts) -> tuple[float, float, float, float]:
        """统一 2.x 多边形 / 3.x rec_polys(4,2) / rec_boxes(4,) 为 (x0,y0,x1,y1)。

        纯 Python 实现（兼容 list 与 ndarray），不依赖 numpy——numpy 元数据损坏/缺失时
        OCR 的 bbox 解析仍可工作（2026-08-25 uv sync 误清 rag extra 后 numpy 损坏踩坑）。
        """
        try:
            if pts is None or len(pts) == 0:
                return (0.0, 0.0, 0.0, 0.0)
            first = pts[0]
            try:
                is_poly = len(first) >= 2  # 多边形每点为 [x, y]
            except TypeError:
                is_poly = False
            if is_poly:
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                return (min(xs), min(ys), max(xs), max(ys))
            if len(pts) >= 4:  # 扁平 box [x1,y1,x2,y2]
                return (float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3]))
            return (0.0, 0.0, 0.0, 0.0)
        except Exception:  # noqa: BLE001
            return (0.0, 0.0, 0.0, 0.0)

    # ---------------------------------------------------------- 表格结构还原
    def _table(self, path: Path, ocr_lines: list[tuple[str, tuple[float, float, float, float]]]) -> str:
        """Table Transformer 检测单元格 → 按 bbox 网格排序 → Markdown；并用 OCR 行坐标回填文字。"""
        try:
            import torch
            from PIL import Image
        except Exception:  # noqa: BLE001
            logger.info("[figure_content] PIL/torch 未安装，跳过表格结构还原")
            return ""
        try:
            # O-07：processor/model 走**实例级单例**（原实现每次调用都 `from_pretrained`）
            processor, model = self._get_table_models()
        except Exception:  # noqa: BLE001
            logger.info("[figure_content] transformers 未安装，跳过表格结构还原")
            return ""
        try:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            target = torch.tensor([height, width])
            det = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[target])[0]
            boxes = det["boxes"].tolist()
            labels = det["labels"].tolist()

            rows = [b for b, lab in zip(boxes, labels, strict=False) if lab == _LABEL_TABLE_ROW]
            cols = [b for b, lab in zip(boxes, labels, strict=False) if lab == _LABEL_TABLE_COLUMN]
            cells = [b for b, lab in zip(boxes, labels, strict=False) if lab == _LABEL_SPANNING_CELL]
            if not rows or not cols:
                return ""
            row_pos = sorted((b[1] + b[3]) / 2 for b in rows)
            col_pos = sorted((b[0] + b[2]) / 2 for b in cols)

            # 单元格网格：key=(row_idx, col_idx)
            grid: dict[tuple[int, int], str] = {}
            for b in cells:
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                r = sum(1 for y in row_pos if y < cy)
                c = sum(1 for x in col_pos if x < cx)
                grid[(r, c)] = ""

            # OCR 行落入哪个单元格：行中心落在 cell 垂直区间、水平中心落在列区间
            for text, (x0, y0, x1, y1) in ocr_lines:
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                r = sum(1 for y in row_pos if y < cy)
                c = sum(1 for x in col_pos if x < cx)
                if (r, c) in grid:
                    grid[(r, c)] = (grid[(r, c)] + " " + text).strip()
                else:
                    grid[(r, c)] = text  # 落在网格外也兜底填入最近格

            nrows = len(row_pos) + 1
            ncols = len(col_pos) + 1
            md: list[str] = []
            for r in range(nrows):
                md.append("| " + " | ".join(grid.get((r, c), " ") or " " for c in range(ncols)) + " |")
                if r == 0:
                    md.append("| " + " | ".join(["---"] * ncols) + " |")
            return "\n".join(md)
        except Exception as e:  # noqa: BLE001
            logger.warning("[figure_content] 表格结构还原失败：%s", e)
            return ""

    # -------------------------------------------------------------- VLM 图注
    async def _vlm(self, path: Path, *, kind: str, local_text_done: bool = False,
                   table_done: bool = False) -> str:
        """用多模态模型生成图表描述/摘要作为图注。失败/缺 key → 空串。

        模型可配置（self.vlm_model，默认 deepseek-v4-flash-vision-exp，可切 qwen3-vl-flash）；
        配置的模型不在 MULTIMODAL_MODELS 或对应 key 未配 → 回退默认 deepseek 多模态模型。

        去重（2026-08-25）：本地 OCR/表格引擎已产出时（local_text_done / table_done），
        提示词禁止 VLM 重复转录文字或还原行列——VLM 只做语义层（图表类型/趋势/结论），
        避免同图同质劳动与重复 token 计费；单独启用 vlm 时仍完整转录+还原。
        """
        try:
            from lightcore.llm.base import ChatMessage, Role
            from lightcore.llm.router import (
                MODEL_TO_PROVIDER,
                MULTIMODAL_MODELS,
                PRESET_ENDPOINTS,
                OAICompatibleLLM,
            )
        except Exception:  # noqa: BLE001
            return ""
        mime = _IMAGE_MIME.get(path.suffix.lower(), "image/png")
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            return ""
        data_url = f"data:{mime};base64,{b64}"
        if kind == "image_table":
            if table_done:
                instr = ("这张图片的表格结构已由本地引擎还原为 Markdown 入库。"
                         "请只补充该表格展示的数据要点、趋势或结论（中文一两句话），"
                         "不要重复转录文字或还原行列。")
            elif local_text_done:
                instr = ("这是一张表格图片，图中文字已由 OCR 转录入库。"
                         "请只描述其行列结构与数据要点，不要重复转录文字。")
            else:
                instr = "这是一张表格图片，请尽量还原其行列结构与关键数据点，用 Markdown 表格呈现；图中文字请一并转录。"
        else:
            if local_text_done:
                instr = ("图中文字已由 OCR 转录入库，请勿重复转录。"
                         "只说明图表类型、坐标轴含义、趋势与核心结论（中文一两句话）。")
            else:
                instr = "请说明这张学术图片展示的图表类型、坐标轴含义、趋势与核心结论；若含文字请转录。"
        prompt = f"请用中文简要描述该学术图片内容：{instr}只输出描述本身，不要解释或补充。"
        try:
            llm = self._vlm_llm
            if llm is None:
                model = self.vlm_model
                provider = MODEL_TO_PROVIDER.get(model)
                ep = PRESET_ENDPOINTS.get(provider) if provider else None
                if ep is None or model not in MULTIMODAL_MODELS:
                    model = "deepseek-v4-flash-vision-exp"  # 回退默认多模态
                    ep = PRESET_ENDPOINTS["deepseek"]
                llm = OAICompatibleLLM(ep, model=model)
            resp = await llm.chat([
                ChatMessage(role=Role.USER, content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ])
            ])
            return (resp.message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("[figure_content] VLM 图注生成失败：%s", e)
            return ""


#: Mermaid 代码块提取正则（VLM 输出格式约定：```mermaid ... ```）
_MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)


def _split_mermaid(raw: str) -> dict:
    """从 VLM 输出中分离 Mermaid 代码块与一句话描述。"""
    m = _MERMAID_RE.search(raw or "")
    mermaid = (m.group(1).strip() if m else "")
    desc = _MERMAID_RE.sub("", raw or "").strip()
    desc = re.sub(r"^【描述】\s*", "", desc).strip()
    return {"mermaid": mermaid, "description": desc}


class DiagramEnricher:
    """对矢量结构图做多模态结构化识别（P1，受 features.diagram_enrich 开关控制）。

    把 DiagramExtractor 栅格化出的结构图 PNG 喂多模态模型（DeepSeek-V4-Flash-Vision，
    复用 OAICompatibleLLM 通道），输出：
    - mermaid：Mermaid 流程图源码（前端可渲染/复制）
    - description：一句话中文描述（并入向量检索文本）
    失败/缺 key → 空串，静默降级，不阻断上传。
    """

    def __init__(self, vlm_llm=None) -> None:
        # 测试可注入假 LLM；生产为 None → 走预置 deepseek 端点（与 FigureEnricher 一致）
        self._vlm_llm = vlm_llm

    async def enrich(self, image_path: str | Path) -> dict:
        """对单张结构图做结构化识别。返回 {"mermaid": "", "description": ""}。"""
        path = Path(image_path)
        if not path.exists():
            return {"mermaid": "", "description": ""}
        raw = (await self._vlm(path)).strip()
        return _split_mermaid(raw)

    async def _vlm(self, path: Path) -> str:
        """用 DeepSeek-V4-Flash-Vision 识别结构图结构；失败/缺 key → 空串。"""
        try:
            from lightcore.llm.base import ChatMessage, Role
            from lightcore.llm.router import PRESET_ENDPOINTS, OAICompatibleLLM
        except Exception:  # noqa: BLE001
            return ""
        mime = _IMAGE_MIME.get(path.suffix.lower(), "image/png")
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            return ""
        data_url = f"data:{mime};base64,{b64}"
        prompt = (
            "请识别这张结构图（架构图/流程图/框图）的逻辑结构。\n"
            "输出要求：\n"
            "1. 先用 Mermaid 流程图表达（graph TD 或 flowchart），放在 ```mermaid 代码块中；\n"
            "2. 代码块之后另起一行，输出一句话中文描述（图的主题与要点）。\n"
            "若图片不是结构图或无法识别，只输出：无法识别。"
        )
        try:
            llm = self._vlm_llm or OAICompatibleLLM(PRESET_ENDPOINTS["deepseek"])
            resp = await llm.chat([
                ChatMessage(role=Role.USER, content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ])
            ])
            return (resp.message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("[diagram_content] VLM 结构识别失败：%s", e)
            return ""


__all__ = ["FigureEnricher", "DiagramEnricher"]
