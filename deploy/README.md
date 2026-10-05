# 部署说明（PaperMate）

> 本目录是**最小可交付的部署形态**：一个 Dockerfile + 一份 compose + 本文档。
> 目标是"能交给别人跑起来并运维"，而不是生产级高可用方案。

---

## ⚠️ 先说三条必读

| # | 事项 | 说明 |
|---|---|---|
| 1 | **本项目没有鉴权** | 定位为本地单用户工具。安全边界**完全依赖"谁能访问到这个端口"** —— 所以 compose 默认只把端口映射到 `127.0.0.1`。**切勿直接映射到公网。** |
| 2 | **镜像很大（6~9 GB）** | `--extra rag` 会装 PyTorch（CUDA 版）+ faiss + FlagEmbedding。这是本地推理的固有成本。 |
| 3 | **构建未在本机实测** | 开发环境没有 Docker，本文件由静态分析产出。请按下面「构建后自检」逐项确认。 |

---

## 一、前置条件

| 项 | 要求 |
|---|---|
| Docker | 20.10+（compose 用 v2 语法） |
| 磁盘 | 建议预留 **15 GB**（镜像 + 模型 + 数据） |
| 内存 | 建议 8 GB+（BGE-M3 + reranker 常驻） |
| GPU（可选） | NVIDIA Container Toolkit；不开也能跑，但编码会回落到 CPU（明显更慢） |
| 前端产物 | **必须先构建**：`cd web && npm ci && npm run build`（生成 `web/dist/`） |

> 为什么前端不在镜像里构建：多阶段构建会让构建时间翻倍且镜像层臃肿；
> 本项目前端产物很小，直接拷入更简单可控。

---

## 二、部署（推荐：compose）

```bash
# 1) 构建前端产物（镜像里不跑 npm）
cd web && npm ci && npm run build && cd ..

# 2) 准备环境变量（模型密钥等；该文件不入库）
cp .env.example .env
#    按需填写 DEEPSEEK_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY

# 3) 起服务
docker compose -f deploy/docker-compose.yml up -d --build

# 4) 自检
curl http://127.0.0.1:8000/api/v1/healthz
```

浏览器打开 `http://127.0.0.1:8000` 即可（后端同时托管前端）。

### 裸 docker（不用 compose）

```bash
docker build -f deploy/Dockerfile -t papermate:local .
docker run -d --name papermate \
  -p 127.0.0.1:8000:8000 \
  -e PAPERMATE_HOST=0.0.0.0 \
  -v papermate-data:/data \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  papermate:local
```

---

## 三、构建后自检（**必做**，因为构建未经实测）

```bash
# ① 服务起来了吗
curl -s http://127.0.0.1:8000/api/v1/healthz
#    期望：{"status":"ok","db_ok":true,"network_exposed":false,...}
#    network_exposed 必须是 false —— 说明宿主只暴露回环（compose 默认如此）

# ② 前端托管正常吗
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/
#    期望：200

# ③ 数据落在卷里吗（重建容器不丢）
docker compose -f deploy/docker-compose.yml restart
curl -s http://127.0.0.1:8000/api/v1/sessions | head -c 200

# ④ GPU 生效了吗（若启用了 GPU 直通）
docker exec papermate python -c "import torch; print('cuda:', torch.cuda.is_available())"
#    期望：cuda: True（未启用 GPU 时为 False，不是错误）

# ⑤ 错误响应是 JSON 吗（统一错误出口）
curl -s http://127.0.0.1:8000/api/v1/documents/abc/file
#    期望：{"detail":"请求参数不合法：doc_id — ...","code":"validation_error","trace_id":"..."}
```

---

## 四、常见问题

| 现象 | 原因与处理 |
|---|---|
| `COPY web/dist: not found` | 没先构建前端。执行 `cd web && npm ci && npm run build` |
| 首次请求很慢（分钟级） | 正在下载 BGE-M3 / reranker 权重（约 4 GB）。已挂 HF 缓存卷后，**只慢这一次** |
| 下载卡住 / 超时 | 国内网络：compose 已默认 `HF_ENDPOINT=https://hf-mirror.com`；也可在 `.env` 里改 |
| 启动日志出现**安全告警** | 容器内 `PAPERMATE_HOST=0.0.0.0` 属正常（否则宿主访问不到）。**真正的边界是端口映射**；确认 compose 的 `ports` 仍是 `127.0.0.1:8000:8000` |
| `torch` 装不上 / 版本找不到 | 锁文件里的 torch 是带 CUDA 后缀的平台构建。若构建失败，可先去掉 `--extra rag` 跑通链路（检索会退化为占位实现），再单独解决 torch 来源 |
| Windows 下 HF 缓存挂载为空 | 确认宿主路径存在（默认 `%USERPROFILE%\.cache\huggingface`），必要时在 `.env` 设 `HF_CACHE=D:\path\to\huggingface` |

---

## 五、安全清单（部署前逐项确认）

- [ ] `ports` 映射的是 `127.0.0.1:8000:8000`（**不是** `8000:8000`）
- [ ] `.env` 未提交到版本库（`.gitignore` 已忽略）
- [ ] `curl /api/v1/healthz` 返回的 `network_exposed` 与预期一致
- [ ] 如需局域网访问：明确知道"同网段任何人都能读写全部数据"，并已限制在可信网络
- [ ] 数据卷已持久化（`/data`），且知道如何备份（整个卷拷走即可：SQLite + faiss + 上传文件都在里面）

---

## 六、当前不支持 / 已知边界

| 项 | 说明 |
|---|---|
| 多用户 / 鉴权 | **不支持**。加多租户需要给会话/文档/记忆等全表加 `user_id`，属架构级改动 |
| 水平扩展 | **不支持**。SQLite 单写锁 + 进程内状态（任务队列、流取消令牌）决定了单实例 |
| 反向代理 / HTTPS | 未提供配置。若前置 Nginx，注意 **SSE 需关闭缓冲**（`proxy_buffering off`）且 `proxy_read_timeout` 要够长 |
| 优雅停机 | 部分：`lifespan` 会停后台 worker，但在飞的 SSE 流未处理 |
| 监控告警 | 未提供。可用于接入的现成接口：`GET /api/v1/healthz`、`GET /api/v1/eval/summary` |

---

> 相关文档：`docs/后端工程体系评估.md`（工程化短板与优先级）、`.env.example`（环境变量说明与安全提示）。
