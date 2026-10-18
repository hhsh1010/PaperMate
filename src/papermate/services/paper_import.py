"""论文导入服务：**安全下载 PDF → 去重 → 交给现有文档管线**（阶段 2）

## 分工（不要重复造）
- 搜索（阶段 1，`paper_search.py`）**只拿元数据**，绝不下载；
- 本模块**只做三件事**：安全下载 → 内容去重 → `upload_and_process(block=False)`；
- 解析 / 分块 / 向量化的**进度与取消**全部复用既有实现
  （`GET /documents/{id}/progress`、`POST /documents/{id}/cancel`）—— 这里不复刻一份进度。

## 安全约束（前端回传的 `pdf_url` **不可信**）
1. **域名白名单**：只允许 `arxiv.org` / `www.arxiv.org`，且路径必须以 `/pdf/` 开头（**防 SSRF**：
   否则前端可以让我们去请求 `http://127.0.0.1:...` 或云元数据地址）；
2. **大小上限**：`Content-Length` 超限直接拒绝（不下载），流式累计超限即中止（默认 60MB）；
3. **类型校验**：`Content-Type` 含 `pdf`，且落盘前校验**文件头 `%PDF-`**；
4. **失败/取消不留半截文件**（全部在内存里累积，只有成功才交给文档管线）。

## 状态机
`queued → downloading → ingesting(带 doc_id) → ready / failed / cancelled`
> `ingesting` 之后的细粒度进度请看文档进度接口，而不是在本任务上另造一套。

## 为什么任务状态只放内存
任务本身是**瞬时**的（分钟级）；落地的东西是**文档**（DB 里有 `status`，重启后仍在）。
故任务表不做持久化 —— 重启后未完成的导入需要重新发起（前端会看到任务不存在）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from .document_service import DocumentService

logger = logging.getLogger(__name__)

#: 允许下载的主机（只读白名单 —— 防 SSRF）
ALLOWED_HOSTS = frozenset({"arxiv.org", "www.arxiv.org"})

#: 只允许的路径前缀（arXiv 的 PDF 直链都是 /pdf/<id>）
ALLOWED_PATH_PREFIX = "/pdf/"

#: 默认大小上限（60MB：绝大多数论文 5~15MB，个别 30MB+）
DEFAULT_MAX_BYTES = 60 * 1024 * 1024

USER_AGENT = "PaperMate/0.1 (academic paper reader; contact: local-user)"

#: 允许的任务状态
STATUS_QUEUED = "queued"
STATUS_DOWNLOADING = "downloading"
STATUS_INGESTING = "ingesting"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


class PaperImportError(RuntimeError):
    """导入失败（消息面向用户可读）。"""


#: Windows 文件名非法字符（含控制字符）
_ILLEGAL_NAME = re.compile(r'[\\/:*?"<>|\r\n\t\x00-\x1f]')


def paper_filename(title: str, source_id: str, version: str = "", *, max_len: int = 80) -> str:
    """用**论文原名**生成文件名（不再用 arXiv id）。

    - 去掉 Windows 文件名非法字符（反斜杠、斜杠、冒号、星号、问号、引号、尖括号、竖线）与控制字符，压掉多余空白；
    - 截断到 ``max_len`` 个**字符**（不是字节 —— 中文标题按字截才不会切坏半个字）；
    - 标题为空则回退到 ``source_id + version``；两者都没有就用 ``paper``。
    - ★ 文件名**只影响展示**：去重靠 `content_hash`，落盘按 ``doc_id`` 分目录 ⇒ 重名无害。
    """
    raw = " ".join((title or "").split())
    # 替换成空格后**必须再压一次**：`A/B: C` → `A B  C`（双空格）看着像文件名出错
    clean = " ".join(_ILLEGAL_NAME.sub(" ", raw).split()).strip(" .")
    if not clean:
        sid = (source_id or "").strip()
        clean = f"{sid}{version}" if sid else "paper"
        clean = " ".join(_ILLEGAL_NAME.sub(" ", clean).split()).strip(" .") or "paper"
    if len(clean) > max_len:
        clean = clean[:max_len].rstrip(" .")
    return f"{clean}.pdf"


def validate_pdf_url(url: str) -> str:
    """校验并规范化下载地址；**不合法就抛 `PaperImportError`**。

    ★ 这是防 SSRF 的唯一入口 —— 前端回传的 url 一律当作敌意输入。
    """
    raw = (url or "").strip()
    if not raw:
        raise PaperImportError("缺少下载地址")
    u = urlparse(raw)
    if u.scheme not in ("http", "https"):
        raise PaperImportError(f"不支持的协议：{u.scheme or '（空）'}")
    if (u.hostname or "").lower() not in ALLOWED_HOSTS:
        raise PaperImportError(f"仅支持从 arXiv 下载（收到主机：{u.hostname or '（空）'}）")
    if not u.path.startswith(ALLOWED_PATH_PREFIX):
        raise PaperImportError(f"仅支持 arXiv 的 PDF 直链（路径须以 {ALLOWED_PATH_PREFIX} 开头）")
    # 统一成 https（避免明文传输论文内容）
    if u.scheme == "http":
        raw = "https://" + raw.split("://", 1)[1]
    return raw


@dataclass
class ImportTask:
    """一次导入的瞬时状态（**不持久化**，见模块 docstring）。"""

    id: str
    url: str
    source: str = "arxiv"
    source_id: str = ""
    version: str = ""
    title: str = ""
    filename: str = ""
    status: str = STATUS_QUEUED
    downloaded: int = 0
    total: int = 0            # 来自 Content-Length；未知为 0
    doc_id: int | None = None
    duplicate: bool = False   # True ⇒ 命中去重，未新建文档
    error: str = ""
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "pdf_url": self.url,
            "source": self.source,
            "source_id": self.source_id,
            "version": self.version,
            "title": self.title,
            "filename": self.filename,
            "downloaded": self.downloaded,
            "total": self.total,
            "doc_id": self.doc_id,
            "duplicate": self.duplicate,
            "error": self.error,
            # ★ 给前端 timeAgo 用的 ISO 串（float 时间戳前端不好直接渲染）
            "created_at_iso": datetime.fromtimestamp(self.created_at, tz=UTC).isoformat(),
        }


class PaperImportService:
    """导入任务的管理器（下载 → 去重 → 交管线 → 轮询至终态）。"""

    def __init__(
        self,
        doc_svc: DocumentService,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout_s: float = 180.0,
        max_retries: int = 2,
        backoff_base: float = 1.0,
        poll_interval_s: float = 0.5,
        ingest_timeout_s: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._docs = doc_svc
        self._max_bytes = max(1, int(max_bytes))
        self._timeout = timeout_s
        self._max_retries = max(0, int(max_retries))
        self._backoff = max(0.0, backoff_base)
        self._poll = max(0.05, poll_interval_s)
        self._ingest_timeout = max(1.0, ingest_timeout_s)
        self._transport = transport            # 测试注入 MockTransport
        self._tasks: dict[str, ImportTask] = {}
        self._handles: dict[str, asyncio.Task] = {}   # 强引用，防被 GC

    # ---------------- 对外 ----------------
    def submit(self, hit: dict) -> ImportTask:
        """登记一次导入并立即返回任务（下载在后台跑）。"""
        url = validate_pdf_url(str(hit.get("pdf_url") or ""))
        sid = str(hit.get("source_id") or "").strip()
        ver = str(hit.get("version") or "").strip()
        # ★ 文件名用**论文原名**（用户反馈：原来显示 arXiv id，看不出是哪篇）
        #   标题为空才回退到 id；uuid 只作最后兜底。
        filename = paper_filename(str(hit.get("title") or ""), sid, ver)
        if filename == "paper.pdf" and not sid:
            filename = f"{uuid.uuid4().hex[:8]}.pdf"
        task = ImportTask(
            id=uuid.uuid4().hex,
            url=url,
            source=str(hit.get("source") or "arxiv"),
            source_id=sid,
            version=ver,
            title=str(hit.get("title") or ""),
            filename=filename,
        )
        self._tasks[task.id] = task
        handle = asyncio.get_running_loop().create_task(self._run(task))
        self._handles[task.id] = handle
        handle.add_done_callback(lambda _t, tid=task.id: self._handles.pop(tid, None))
        return task

    def list_tasks(self, *, limit: int = 50) -> list[ImportTask]:
        """列出**最近的**任务（新的在前）。

        ★ 为什么需要它：任务状态原本只活在前端组件里 ⇒ 用户切到别的页面（组件卸载）
        进度就"丢"了，而后端其实还在跑。有了这个接口，前端重新挂载时拉一次就能恢复。
        """
        items = sorted(self._tasks.values(), key=lambda t: -t.created_at)
        return items[: max(1, int(limit))]

    def get(self, task_id: str) -> ImportTask | None:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """取消任务；已结束的返回 False。"""
        task = self._tasks.get(task_id)
        handle = self._handles.get(task_id)
        if task is None or handle is None or handle.done():
            return False
        handle.cancel()
        return True

    # ---------------- 内部 ----------------
    async def _run(self, task: ImportTask) -> None:
        try:
            data, content_type = await self._download(task)
            if not data.startswith(b"%PDF-"):
                raise PaperImportError("下载到的内容不是 PDF（文件头不是 %PDF-）")

            # 内容级去重：与上传路由同一判据（只拦非 failed/cancelled 的重复）
            dup = self._docs.check_duplicate(task.filename, data)
            existing = dup.get("existing")
            if dup.get("duplicate") and existing is not None and getattr(existing, "status", "") not in (
                "failed", "cancelled"
            ):
                task.status = STATUS_READY
                task.doc_id = int(existing.id)
                task.duplicate = True
                logger.info("[paper_import] 命中去重 doc=%s（task=%s）", existing.id, task.id)
                return

            task.status = STATUS_INGESTING
            doc = await self._docs.upload_and_process(task.filename, data, block=False)
            task.doc_id = int(doc.id)

            # 复用现有处理进度：这里只等终态（进度细节由 /documents/{id}/progress 提供）
            ready = await self._wait_doc(task)
            # ★ 必须**在处理完成之后**写 meta：`_process_phase2` 会整体重写 meta_json，
            #   先写会被覆盖（这是把 patch 放在这里的唯一原因）。
            if ready:
                self._patch_meta(task)
        except asyncio.CancelledError:
            task.status = STATUS_CANCELLED
            task.error = "已取消"
            # 不 re-raise：任务已登记为终态，避免调用方看到"取消"变成未处理异常
        except (PaperImportError, OSError, httpx.HTTPError) as e:
            task.status = STATUS_FAILED
            task.error = str(e) if isinstance(e, PaperImportError) else f"{type(e).__name__}: {e}"
            logger.warning("[paper_import] 任务失败 task=%s：%s", task.id, task.error)
        except Exception as e:  # noqa: BLE001  兜底：任何异常都必须落到 failed，不能悬空
            task.status = STATUS_FAILED
            task.error = f"内部错误：{type(e).__name__}: {e}"
            logger.exception("[paper_import] 任务异常 task=%s", task.id)

    async def _download(self, task: ImportTask) -> tuple[bytes, str]:
        """流式下载 PDF（带大小上限与重试）。返回 (内容, content-type)。"""
        headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf"}
        last = ""
        for attempt in range(self._max_retries + 1):
            task.status = STATUS_DOWNLOADING
            task.downloaded = 0
            try:
                async with httpx.AsyncClient(timeout=self._timeout,
                                             transport=self._transport,
                                             follow_redirects=True) as client:
                    async with client.stream("GET", task.url, headers=headers) as resp:
                        if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                            last = f"HTTP {resp.status_code}"
                            if attempt >= self._max_retries:
                                break
                            await asyncio.sleep(self._backoff * (2**attempt))
                            continue
                        if resp.status_code != 200:
                            raise PaperImportError(
                                f"下载失败：HTTP {resp.status_code}（{resp.reason_phrase}）"
                            )
                        ctype = (resp.headers.get("content-type") or "").lower()
                        if "pdf" not in ctype:
                            raise PaperImportError(f"下载到的不是 PDF（Content-Type={ctype or '空'}）")
                        cl = resp.headers.get("content-length") or ""
                        if cl.isdigit():
                            task.total = int(cl)
                            if task.total > self._max_bytes:      # ★ 超限：**不下载**
                                raise PaperImportError(
                                    f"论文过大（{task.total / 1048576:.1f}MB > "
                                    f"{self._max_bytes / 1048576:.0f}MB 上限）"
                                )
                        buf = bytearray()
                        async for chunk in resp.aiter_bytes(64 * 1024):
                            buf += chunk
                            task.downloaded = len(buf)
                            if len(buf) > self._max_bytes:        # ★ 流式累计超限：中止
                                raise PaperImportError(
                                    f"论文过大（超过 {self._max_bytes / 1048576:.0f}MB 上限），已中止"
                                )
                        return bytes(buf), ctype
            except PaperImportError:
                raise
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(self._backoff * (2**attempt))
        raise PaperImportError(
            f"下载失败（已重试 {self._max_retries} 次：{last}）；请检查网络能否访问 arxiv.org"
        )

    async def _wait_doc(self, task: ImportTask) -> bool:
        """等文档处理到终态。返回 True 表示 ready。

        进度细节走文档进度接口（`get_upload_progress`），这里只看终态。
        """
        assert task.doc_id is not None
        deadline = time.monotonic() + self._ingest_timeout
        while time.monotonic() < deadline:
            doc = self._docs.get_document(task.doc_id)
            status = getattr(doc, "status", "") if doc is not None else ""
            if status == "ready":
                task.status = STATUS_READY
                return True
            if status == "failed":
                task.status = STATUS_FAILED
                task.error = "文档解析失败（可重试）"
                return False
            if status == "cancelled":
                task.status = STATUS_CANCELLED
                return False
            await asyncio.sleep(self._poll)
        task.status = STATUS_FAILED
        task.error = f"处理超时（超过 {self._ingest_timeout:.0f}s）"
        return False

    def _patch_meta(self, task: ImportTask) -> None:
        """把论文元数据写进 `documents.meta_json`（不动表结构）。"""
        if task.doc_id is None:
            return
        try:
            self._docs.merge_meta(task.doc_id, {
                "paper": {
                    "source": task.source,
                    "source_id": task.source_id,
                    "version": task.version,
                    "title": task.title,
                    "url": task.url,
                }
            })
        except Exception as e:  # noqa: BLE001  元数据是附加信息，失败不该让导入失败
            logger.warning("[paper_import] 写 meta 失败 doc=%s：%s", task.doc_id, e)


def paper_import_enabled() -> bool:
    """论文搜索/导入的**全局开关**（默认关 —— 网络功能必须显式开启）。

    判定口径：**运行时设置 OR 环境变量**，任一为真即开启。
    - 运行时设置：`features.paper_search`（设置页可切，状态文件持久化）；
    - 环境变量：`PAPERMATE_ENABLE_PAPER_SEARCH`（运维/CI 用，免开发设置页）。

    ★ 为什么是 OR 而不是"设置优先"：环境变量是**显式的运维动作**（"我就是要开"），
      若被一个默认 false 的设置覆盖，就会出现"设了 env 却不生效"的死配置（本项目踩过）。
      代价是"设置页关不掉 env 打开的功能" —— 已在文档写明：**关闭需两者都关**。
    """
    if os.environ.get("PAPERMATE_ENABLE_PAPER_SEARCH", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        from .. import runtime_settings  # 延迟导入，避免模块级循环依赖
    except Exception:  # noqa: BLE001
        return False
    try:
        feat = runtime_settings.get_state().get("features", {}) or {}
        return bool(feat.get("paper_search", False))
    except Exception:  # noqa: BLE001  读设置失败按"关闭"处理（网络功能的安全默认）
        return False
