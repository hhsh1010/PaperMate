"""H3 回归（2026-09-10）：批量 job 终态裁剪前归档，避免已完成任务 404 / 结果丢失。

缺陷：批量路径（POST /documents/process/batch）的 job 只存内存 `_jobs`，
`_prune_terminal_jobs(keep=200)` 淘汰后 `get_job` 返回 None → 路由 404，结果不可恢复。
"""

from __future__ import annotations

from papermate.processing.models import Job, ProcessingResult
from papermate.processing.queue import JobQueue, _restore_result  # noqa: PLC2701


def _job(job_id: str, ts: float, with_result: bool = True) -> Job:
    j = Job(id=job_id, filename=f"{job_id}.txt", status="done", attempts=1)
    j.updated_at = ts
    if with_result:
        j.result = ProcessingResult(
            filename=f"{job_id}.txt", source_format="text", status="ok", title="T",
            metadata=None, sections=[], num_chunks=3,
            structured={"k": "v"}, markdown="# T", duration_ms=1.5,
        )
    return j


def test_prune_archives_and_get_recovers(tmp_path):
    """H3 回归：终态 job 被内存裁剪后，get() 仍能从归档恢复（不再 404）。"""
    q = JobQueue(processor=lambda *a: None, archive_dir=tmp_path)

    for i in range(5):
        q._jobs[f"j{i}"] = _job(f"j{i}", float(i))
    q._prune_terminal_jobs(keep=2)  # 淘汰 j0/j1/j2（最旧的终态）

    assert "j0" not in q._jobs, "前置：j0 应已被裁剪出内存"
    got = q.get("j0")
    assert got is not None, "裁剪后的终态任务必须可从归档恢复（H3）"
    assert got.id == "j0" and got.status == "done" and got.attempts == 1
    assert got.result is not None and got.result.num_chunks == 3
    assert (tmp_path / "j0.json").exists()

    # 仍在内存的任务不受影响
    assert q.get("j4") is not None and q.get("j4").id == "j4"


def test_archived_job_serializes_through_api_mapper(tmp_path):
    """H3 回归：归档还原对象必须能通过 API 序列化（与 _to_job_out/_to_result_out 契约一致）。

    这是最容易踩的坑：`structured`/`warnings`/`artifacts` 直接喂给 pydantic 字段，
    若被包成 namespace 会校验失败。
    """
    from papermate.api.routes_processing import _to_job_out

    q = JobQueue(processor=lambda *a: None, archive_dir=tmp_path)
    q._jobs["a1"] = _job("a1", 1.0)
    q._archive_job(q._jobs["a1"])

    restored = q._load_archived("a1")
    out = _to_job_out(restored)

    assert out.id == "a1" and out.result is not None
    assert out.result.num_chunks == 3
    assert out.result.structured == {"k": "v"}, "structured 必须保持 dict（pydantic 字段）"
    assert out.result.markdown == "# T"


def test_restore_result_keeps_types_for_nested_contracts():
    """H3 单元：还原对象的嵌套属性访问（issues/sections）与 dict 字段类型同时成立。"""
    r = _restore_result({
        "filename": "a.txt",
        "source_format": "pdf",
        "status": "ok",
        "title": "T",
        "num_chunks": 2,
        "duration_ms": 1.0,
        "metadata": {
            "title": "T",
            "issues": [{"severity": "warn", "code": "C1", "message": "m"}],
        },
        "sections": [{"order": 1, "level": 0, "heading": None, "page": 1, "text": "x"}],
        "structured": {"a": 1},
        "warnings": ["w"],
        "artifacts": {"md": "p"},
    })

    assert r.metadata.issues[0].severity == "warn"  # _to_result_out 用属性访问
    assert r.sections[0].order == 1
    assert isinstance(r.structured, dict)  # pydantic 字段需 dict
    assert isinstance(r.warnings, list)
    assert isinstance(r.artifacts, dict)


def test_without_archive_dir_keeps_old_behavior(tmp_path):
    """对照：archive_dir=None（测试/嵌入式用法）时不归档，行为与修复前一致。"""
    q = JobQueue(processor=lambda *a: None)  # 不传 archive_dir

    q._jobs["z1"] = _job("z1", 1.0)
    q._archive_job(q._jobs["z1"])
    q._jobs.pop("z1")

    assert q.get("z1") is None, "未配置归档目录时不应凭空恢复任务"
    assert list(tmp_path.glob("*.json")) == []


def test_load_archived_rejects_path_traversal(tmp_path):
    """防御：job_id 含路径分隔符时不得读越界文件（文件名净化）。"""
    q = JobQueue(processor=lambda *a: None, archive_dir=tmp_path)
    (tmp_path.parent / "evil.json").write_text('{"id": "evil"}', encoding="utf-8")

    assert q._load_archived("../evil") is None
    assert q._load_archived("") is None
