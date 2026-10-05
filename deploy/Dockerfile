# syntax=docker/dockerfile:1
#
# PaperMate 部署镜像（后端 + 已构建前端）
#
# 设计取舍（为什么是这个形态）：
#   1. **前端不在镜像里构建** —— 要求宿主机先 `cd web && npm ci && npm run build`。
#      多阶段构建会让镜像层变得臃肿且构建时间翻倍；本项目前端产物（web/dist）很小，
#      直接拷进来最简单可控。构建前请确认 `web/dist/index.html` 存在。
#   2. **依赖用 uv + uv.lock 安装**，与本地开发同一套清单，保证可复现。
#   3. **数据目录固定为 /data**（SQLite + faiss 索引 + 上传文件），必须挂卷，
#      否则容器重建即丢数据。
#   4. **HF 模型缓存建议挂宿主机目录** —— BGE-M3 约 2.2GB、reranker 约 2.2GB，
#      不挂载则每次重建容器都要重新下载（见 docker-compose.yml）。
#
# ⚠️ 关于镜像体积：`--extra rag` 会装 PyTorch（CUDA 版），镜像通常 **6~9 GB**。
#    这是本地推理的固有成本；若只需跑通功能可用 fake 嵌入（去掉 --extra rag），
#    但检索质量会退化为占位实现。
#
# ⚠️ 关于构建实测：本文件**未在本机实测**（开发环境没有 Docker）。
#    首次构建后请按 deploy/README.md 的「构建后自检」逐项验证。

FROM python:3.12-slim

# uv：与项目 pyproject.toml / uv.lock 配套（比 pip 快且能严格按锁文件安装）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # 数据落在挂载卷
    PAPERMATE_DATA_DIR=/data \
    # 容器内必须监听 0.0.0.0 才能被宿主访问；
    # ⚠️ 真正的安全边界由 **宿主端口映射** 负责（见 docker-compose.yml 默认只映射 127.0.0.1）。
    #    本项目无鉴权，切勿把容器端口直接映射到公网。
    PAPERMATE_HOST=0.0.0.0 \
    PAPERMATE_PORT=8000

# ---- 依赖层（独立成层以利用缓存：只要清单没变就不重装）----
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra rag

# ---- 源码 + 前端产物 ----
COPY src/ ./src/
COPY web/dist/ ./web/dist/
RUN uv sync --frozen --no-dev --extra rag

EXPOSE 8000
VOLUME ["/data"]

# --no-sync：运行时不再做依赖解析（镜像里已装好），避免启动被网络拖住
CMD ["uv", "run", "--no-sync", "python", "-m", "papermate.api"]
