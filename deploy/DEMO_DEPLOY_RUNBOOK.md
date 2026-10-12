# 在线 Demo 部署 Runbook

> 目标：拿到一个**可访问的 URL**，让别人 30 秒内体验。
> 现状核对（2026-10-10）：`deploy/` 里 `Dockerfile`、`docker-compose.yml`、`nginx/`、`deploy.sh`、`rollback.sh`
> **都已写好，但从未在真实环境跑通过**；本开发机**未安装 Docker**。

---

## 方案对比（先选再动手）

| 方案 | 前置 | 耗时 | 成本 | 适合 |
|---|---|---|---|---|
| **A. 本机 + cloudflared 临时隧道**（推荐先做） | 本机跑得起来（已有 `uv`） | ~20 分钟 | **0** | 临时演示、快速验证可行性 |
| **B. 云服务器 + Docker Compose**（推荐正式版） | 一台 Linux 云主机 + 域名(可选) | 半天 | ~¥30~60/月 | 长期可访问的演示地址 |
| **C. 只做录屏 + 截图** | 无 | ~30 分钟 | 0 | 兜底：至少 README 里能看到 |

> **建议顺序：A 先跑通证明可行 → 再上 B**。A 得到的 URL 是临时的（关隧道即失效），但足以验证 nginx/前端/后端链路。

---

## 方案 A · 本机 + cloudflared（最快路径）

### A1. 本机起服务（两种方式，选一）

> ⚠️ **启动命令以 `python -m papermate.api` 为准**（2026-10-10 更正：早先本文档写的
> `papermate.cli serve ...` 是错的 —— `papermate.cli` 是**命令行对话 REPL**，没有 `serve` 子命令）。

```powershell
# 方式 1（绿色，最省事）：直接跑构建产物（数据根沿用你已配置的）
cd D:\harness-space\agent-project
& "desktop\src-tauri\target\release\papermate-desktop.exe"

# 方式 2（后端单独起，便于看日志）—— ★ 推荐给 Demo 用：**独立数据目录**
cd D:\harness-space\agent-project
$env:PAPERMATE_DATA_DIR = "D:\papermate-demo\data"      # 与真实数据隔离（见下方安全说明）
.venv\Scripts\python.exe -m papermate.api               # 默认绑 127.0.0.1:8000

# 健康检查（★ 正确路径是 /api/v1/healthz —— 2026-10-10 实测更正，早先写的 /healthz 是 404）
curl.exe http://127.0.0.1:8000/api/v1/healthz
# 期望：{"status":"ok","db_ok":true,"network_exposed":false,"default_model":...}
```

> 🔒 **为什么 Demo 要用独立数据目录**：本项目**无鉴权**，且 `PUT /settings` 可写模型密钥。
> 独立目录把公网可达的范围限制在"演示数据"里 —— 不会碰到你真实的论文与 provider_settings.json。
> （另有 E3 硬闸：绑非 127.0.0.1 会**直接拒绝启动**，需显式 `PAPERMATE_ACCEPT_PUBLIC_EXPOSURE=1`；
> 走 cloudflared 隧道时服务仍绑 127.0.0.1，**不触发**，无需放行。）

### A2. 开通临时公网隧道

**首选：cloudflared（winget 安装，1 分钟）**

```powershell
# 安装（装完请**新开一个终端**，让 PATH 刷新）
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://127.0.0.1:8000
# 终端会打印 https://<随机名>.trycloudflare.com —— 这就是 Demo 地址
```

> ⚠️ **别用「下载 exe + curl -C - 断点续传」**（2026-10-10 实测踩坑）：
> `releases/latest/download/...` 会经历多次重定向，`-C -` 的 Range 请求与重定向链**互相打架** ——
> 每次请求返回的 content-length 都不一样（实测 55MB → 29.6MB → 26.3MB），最终把文件拼成 35.5MB 的
> **损坏体**（表现为 `Exec format error`）。要么 winget，要么**一次性完整下载**（不要续传）。

**备用：ssh 隧道（零安装，Windows 自带 OpenSSH；国内偶尔不稳）**

```powershell
# 得到 https://xxxx.lhr.life 形式的地址
ssh -o StrictHostKeyChecking=no -R 80:localhost:8000 nokey@localhost.run
```

### A3. 验收清单

- [ ] 手机 4G（非本机网络）打开该 URL，能看到界面
- [ ] 上传一篇 PDF → 问答 → 有引用溯源
- [ ] 改稿工作台 → 应用一处修订 → 导出「修订版 .docx」可下载

### ⚠️ 方案 A 的安全要点（务必先看）

暴露公网 = 任何人都能调用你的 LLM 额度。**上隧道前至少做两件事之一**：

1. **设一个演示专用 key**：用一个余额很小的 API Key（如 ¥5），泄露损失可承受；
2. **加 Basic Auth**（推荐，nginx 层）：在 `deploy/nginx/` 的 server 块里加
   ```nginx
   auth_basic "PaperMate Demo";
   auth_basic_user_file /etc/nginx/.htpasswd;   # htpasswd -c 生成
   ```
   把账号密码**单独**给使用者（不要在公开渠道贴出口令）。
3. 另外建议关闭高成本功能（如把默认模型切到便宜的、把技能总开关关掉）。

---

## 方案 B · 云服务器 + Docker Compose（正式版）

### B0. 前置检查（在服务器上）

```bash
docker --version && docker compose version      # 需 Docker 20.10+ / compose v2
free -m                                          # 建议 ≥4GB（模型下载与检索吃内存）
df -h /                                          # 建议 ≥40GB（模型权重 8.5GB + 数据）
```

### B1. 拉代码 + 配置

```bash
git clone <你的仓库地址> papermate && cd papermate
cp .env.example .env && vi .env       # 填 API key（密钥只放 .env / 环境变量，绝不入库）
```

### B2. 起服务

```bash
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml logs -f --tail=100   # 观察启动
curl -s http://127.0.0.1:8000/healthz                            # 健康检查
```

### B3. 反代 + TLS

```bash
sudo cp -r deploy/nginx/* /etc/nginx/conf.d/    # 或整目录挂载
sudo nginx -t && sudo systemctl reload nginx
# 域名 + HTTPS：certbot --nginx -d demo.example.com
```

### B4. 更新与回滚（资产已就绪）

```bash
bash deploy/update-server/deploy.sh     # 部署/更新
bash deploy/update-server/rollback.sh   # 回滚上一版
```

### ⚠️ 方案 B 的常见坑

| 坑 | 说明 |
|---|---|
| **首次启动极慢** | 容器内要装运行环境 + 下模型（约 8.5GB）。建议**镜像里预置**或用挂载卷复用缓存，否则每次重建都要等 |
| 端口/防火墙 | 云厂商安全组要放行 80/443（**不要**把 8000 直接暴露公网） |
| 数据持久化 | `data/`（SQLite/FAISS/上传件）必须挂 volume，否则容器重建即丢数据 |
| 内存不足 | 检索 + 嵌入模型吃内存，1GB 小机大概率 OOM |
| 密钥泄露 | `.env` 加入 `.gitignore`（已做）；**永远不要**把 key 写进 compose 文件后提交 |

---

## 方案 C · 兜底：录屏 + 截图（没有服务器也能提升可信度）

1. 录一段 **30~60 秒** 的演示（Win+G 或 OBS）：上传 PDF → 提问（带引用）→ 改稿 → 导出修订版；
2. 截 3 张图：研读问答、改稿工作台的修订对比、设置页功能模块；
3. 放在 `docs/assets/`，README 的「演示」段引用（占位已留好）；
4. GIF 建议 <5MB（`ffmpeg -i input.mp4 -vf "fps=10,scale=960:-1" demo.gif`）。

---

## 完成后要改的三处（我帮你改，说一声即可）

1. README「演示」段：填 Demo URL / 截图；
2. 在项目介绍里加一行 `在线体验：https://…`；
3. `deploy/README.md`：把本 runbook 的结论回写（避免以后自己忘）。
