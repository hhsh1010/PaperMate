#!/usr/bin/env bash
# PaperMate 服务端发布脚本（镜像 tag 轮换式回滚）
# 用法：./deploy.sh <新镜像tag>   例：./deploy.sh 1.2.0
set -euo pipefail
cd "$(dirname "$0")"

NEW_TAG="${1:?用法: ./deploy.sh <新镜像tag>}"
REGISTRY="${REGISTRY:-ghcr.io/your-org}"

# 1) 拉取/构建新镜像（CI 构建了就去掉 pull）
docker compose -f docker-compose.yml pull papermate || true

# 2) 用新 tag 启动 papermate，nginx 依赖其 healthy
TAG="$NEW_TAG" docker compose -f docker-compose.yml up -d papermate

# 3) 等待健康检查
echo "等待 papermate 健康..."
for i in $(seq 1 30); do
  status="$(docker inspect -f '{{.State.Health.Status}}' papermate 2>/dev/null || echo unknown)"
  if [ "$status" = "healthy" ]; then echo "papermate 健康 OK"; break; fi
  sleep 5
done

# 4) 重启 nginx 接管流量
docker compose -f docker-compose.yml up -d nginx

# 5) 把刚发布的 tag 记为 previous（供回滚）
docker tag "${REGISTRY}/papermate:${NEW_TAG}" "${REGISTRY}/papermate:previous" || true
echo "发布完成: ${NEW_TAG}。如需回滚执行 ./rollback.sh"
