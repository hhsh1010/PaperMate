# scripts/ 目录索引

> **2026-10-08 说明**：本目录 57 个脚本 + `archive/`。
> 曾评估"按类别搬迁到子目录"，实测 **83 个文件**引用了 `scripts/probe_*|exp_*|check_*` 等路径
> （docs / tests / .workbuddy 记录），整体搬迁会大面积破坏引用 ⇒ 改为**原地分组 + 本索引**。
> ⚠️ 新增脚本时请归入下面某一类，并在此登记。

## 一、门禁与发布（CI / 出包必用）

| 脚本 | 用途 |
|---|---|
| `run_tests.py` | **带单实例锁**的 pytest 启动器（防沙箱重放双跑；参数原样透传） |
| `validate_tauri_config.py` | 离线校验 `tauri.conf.json`（Tauri 2 schema，免 Rust） |
| `run-desktop-smoke.ps1` | 桌面端冒烟 |
| `firstuse-sim.ps1` | 首装仿真（隔离环境跑首启链路） |

## 二、运维与数据维护

| 脚本 | 用途 |
|---|---|
| `db_backup.py` | SQLite 备份 |
| `reindex_documents.py` | 重建索引（批量） |
| `retention.py` | 数据留存策略执行 |
| `clean_poisoned_memories.py` / `clean_pytest_tmp.py` | 清理（记忆污染 / pytest 残留） |
| `dedup_collections.py` | 集合去重 |
| `eval_retrieval.py` | 检索评测入口 |
| `upload_sample_papers.py` / `gen_sample_papers.py` | 样本论文上传 / 生成 |
| `build_stress_samples.py` | 压力样本构造 |
| `user_side_check.py` | 用户侧自检 |

## 三、验证与契约检查（不含探针）

| 脚本 | 用途 |
|---|---|
| `check_llm.py` / `check_response_contracts.py` / `check_return_branches.py` / `check_stream_usage.py` | 静态/半静态契约检查 |
| `verify_eval_citations.py` / `verify_figure_enrich.py` / `verify_m11.py` | 专项验证 |
| `e2e_stream_verify.mjs` | SSE 端到端验证（Node） |
| `export_judge_annotation.py` | 导出判官标注 |

## 四、开发期探针与实验（**非产品路径**，可随时清理/归档）

`probe_*.py` ×22（冷启动 / 命中率 / 重排四臂 / TTFT 分段 / token 标定 / 阶段耗时 …）、
`exp_*.py` ×2（T-01 对照实验）、`spike_*.py` ×5（早期技术选型试验）、`run_rewrite_arms.py`。
—— 这些是排查/标定工具：保留是为了**证据可复现**，不参与构建与运行。

## 五、archive/

历史脚本（已被替代或一次性用途）：`archive/`（6 个），只读留档，不再维护。
