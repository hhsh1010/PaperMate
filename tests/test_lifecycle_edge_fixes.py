"""2026-09-10 P2 收口回归：M9（rerank 空 batch）/ G1（管线缓存代际）/ G2（自取消反模式）。

三个缺陷都属于「边界/并发语义」——单测覆盖不到但会在真实使用中显形。
"""

from __future__ import annotations

import os
import threading

from papermate.services.document_service import DocumentService


# ---------------- M9：CrossEncoderReranker 空候选 ----------------
def test_cross_encoder_rerank_empty_candidates_returns_empty():
    """M9 回归：空候选直接返回 []，绝不调用 predict([])。

    空 batch 会让 tokenizer 无法确定 padding 长度而抛错；唯一调用方虽有兜底，
    但直连调用处会炸，且与 ScoreReranker 的空输入行为不一致。
    """
    from lightcore.rag.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)  # 跳过 __init__（不加载 2.2GB 模型）

    class _BoomModel:
        def predict(self, _pairs, show_progress_bar=None):  # noqa: ANN001
            raise AssertionError("空候选不应调用 predict（M9 回归失败）")

    reranker._model = _BoomModel()
    assert reranker.rerank("query", [], top_k=5, texts=[]) == []


# ---------------- G1 / G2：DocumentService 生命周期 ----------------
class _StubSession:
    def __enter__(self):
        return self

    def __exit__(self, *a):  # noqa: ANN002, ANN003
        return False

    def get(self, _model, _id):  # noqa: ANN001
        return None  # 文档行不存在：_delete_row 直接返回


class _StubSessionMaker:
    def __call__(self):
        return _StubSession()


def _make_service(tmp_path):
    from papermate import config as cfg_mod

    os.environ["PAPERMATE_DATA_DIR"] = str(tmp_path)
    cfg_mod._settings = None  # 重置缓存，确保读入临时 data_dir

    import papermate.runtime_settings as rs

    rs._state = None

    from papermate.config import get_settings

    return DocumentService(_StubSessionMaker(), get_settings())


def test_cache_pipeline_rejects_stale_generation(tmp_path):
    """G1 回归：文档在构建期间被删除（代际变化）→ 过期构建不得写入管线缓存。

    否则 `_pipelines` 会积累**孤儿缓存**（常驻内存、无对应文档、永不回收）。
    """
    svc = _make_service(tmp_path)

    svc._cache_pipeline(1, "PIPE-fresh", gen=0)  # 代际一致 → 正常写入
    assert svc._pipelines[1] == "PIPE-fresh"

    svc._pipelines.pop(1, None)
    svc._pipeline_gen[1] = 1  # 模拟删除（_drop_document_data 递增代际）
    svc._cache_pipeline(1, "PIPE-stale", gen=0)  # 构建开始时记录的旧代际 → 拒绝
    assert 1 not in svc._pipelines, "删除后的过期构建不得写缓存（G1）"

    svc._cache_pipeline(1, "PIPE-new", gen=1)  # 新代际（删除后重建）→ 允许
    assert svc._pipelines[1] == "PIPE-new"


def test_drop_document_data_keeps_lock_and_bumps_generation(tmp_path):
    """G1 回归：数据清理**不**移除 `_locks`（否则并发构建会用两把不同的锁）。

    原先 delete 会 pop 锁 → 新协程 `setdefault` 重建新锁 → 与在飞旧构建并发执行
    同一文档的 BGE-M3 编码（昂贵）并产生孤儿管线。
    """
    svc = _make_service(tmp_path)
    svc._locks[5] = threading.Lock()  # 类型无关：只验证不被移除
    svc._pipelines[5] = "PIPE"

    svc._drop_document_data(5)

    assert 5 in svc._locks, "锁必须保留（G1）——移除会导致并发构建用不同锁"
    assert 5 not in svc._pipelines
    assert svc._pipeline_gen.get(5, 0) == 1, "删除必须递增代际（G1）"


def test_cleanup_upload_record_does_not_cancel_current_task(tmp_path):
    """G2 回归：取消/失败回收**不得** pop 或 cancel 上传任务。

    原先 `_cleanup_upload_record` → `delete()` 会取到**调用者自身**的任务并 cancel()
    （自取消反模式）：任务以 cancelled 语义结束，外部 await 该 task 会收到
    CancelledError，掩盖真实失败原因。
    """
    svc = _make_service(tmp_path)

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()
    svc._upload_tasks[7] = task  # type: ignore[assignment]
    svc._upload_cancel[7] = threading.Event()
    svc._upload_progress[7] = {"status": "parsing"}

    svc._cleanup_upload_record(7)

    assert task.cancelled is False, "回收流程不得取消调用者自身任务（G2）"
    assert svc._upload_tasks.get(7) is task, "任务条目由启动方收尾，清理流程不得 pop（G2）"
    assert 7 not in svc._upload_cancel
    assert 7 not in svc._upload_progress


def test_delete_cancels_task_and_drops_data(tmp_path):
    """对照：**外部主动 delete** 仍必须取消在飞任务（G2 修复不得削弱删除语义）。"""
    svc = _make_service(tmp_path)

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()
    svc._upload_tasks[9] = task  # type: ignore[assignment]
    svc._upload_cancel[9] = threading.Event()
    svc._upload_progress[9] = {"status": "parsing"}

    svc.delete(9)

    assert task.cancelled is True, "外部删除必须取消在飞上传任务"
    assert 9 not in svc._upload_tasks
    assert 9 not in svc._upload_cancel
    assert 9 not in svc._upload_progress
