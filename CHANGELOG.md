# 更新日志（CHANGELOG）

本文件记录仓库结构与发布相关的关键变更。功能级变更请以 Git 提交历史为准。

## [Unreleased]

### 仓库治理
- 初始化 Git 仓库（分支 `main`），完成源码入库（517 文件，工作区干净）。
- 建立 `.gitignore`（覆盖 `.env` / `data/` / `node_modules/` / `web/dist/` / 各类缓存与 IDE/OS 垃圾）与 `.gitattributes`（源码统一 LF、`eval/cases/*.jsonl` 保原始字节指纹、二进制不转换）。
- 文档分层：`docs/` 默认忽略内部过程文档，仅白名单收录「技术书」12 章与少量必需文档；`eval/` 仅入库脚本 + 用例集 + 语料 + 判官提示词。
- 新增 `CONTRIBUTING.md` 与本文档；`README.md` 补全「快速开始（本地运行）」与目录约定。

### 已知边界
- 项目**无鉴权机制**，定位本地单用户；暴露到局域网需自行评估风险（见 `.env.example` 注释）。
- 评测依赖真实 API Key 与 GPU，相关脚本需本地具备条件后运行。
