# 贡献指南（CONTRIBUTING）

感谢你关注 PaperMate / LightCore！本项目定位为**本地单用户**的学术论文研读 Agent，目前以「个人维护 + 接受 Issue/PR」为主。以下是参与开发的最低约定。

## 环境要求
- Python 3.11+（依赖由 `uv` 统一管理，见 `pyproject.toml` / `uv.lock`）
- Node 22+（前端在 `web/`）
- 一份大模型 API Key（DeepSeek / Qwen / OpenAI 均可；不填也能用 `fake` 模式跑通内核与 SSE）

## 本地启动
```bash
uv sync --extra rag --extra dev          # 安装后端依赖（含 rag / dev 附加组）
cp .env.example .env                       # 按需填写 API Key
# 后端（另开终端）
uv run uvicorn api.__main__:app --reload --port 8000
# 前端（另开终端）
cd web && npm install && npm run dev
```
> 克隆后是「空壳」：`data/` 与 `.env` 均被 `.gitignore` 排除，首次启动会自动建目录。

## 三层门禁（提交前必须全绿）
1. **后端 lint + 类型**：`ruff check src tests`
2. **后端测试**：`.venv/Scripts/python.exe scripts/run_tests.py`（必须走带单实例锁的启动器）
3. **前端**：`cd web && npm run typecheck && npm test && npm run build`

任何「能力」类改动（LLM 接入、联网、新功能）须保留三处硬约束：**后端 API + 开关 / 前端入口 + 开关 / 连通性测试**。

## 提交规范
- 分支：`main` 为主干，功能走 `feat/*`、修复走 `fix/*`。
- 提交信息：中文或英文均可，建议「动词 + 影响面」，例如「fix(retrieve): 修复跨轮去重误判」。
- **绝不**提交：`data/`、`*.db`、`.env`、密钥、`node_modules/`、`web/dist/`、个人绝对路径。

## 评测语料注意
`eval/cases/*.jsonl` 用 `sha256(原始字节)` 做指纹，**不要归一化换行**（CRLF↔LF 会让门禁判「用例集已变」）。改用例集须同步重跑评测刷新指纹。

## Issue / PR
- Bug 请附：可复现步骤 + 后端日志 `data/logs/app.log` 关键行 + 期望/实际。
- 大改动先开 Issue 讨论影响面，再提 PR。
