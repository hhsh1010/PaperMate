#!/usr/bin/env bash
# PaperMate 服务端回滚脚本（切回 previous 镜像 tag）
# 用法：./rollback.sh   （或 PREV_TAG=v1.1.0 ./rollback.sh）
set -euo pipefail
cd "$(dirname "$0")"

PREV="${PREV_TAG:-previous}"
echo "回滚 papermate 至镜像 tag: $PREV"
TAG="$PREV" docker compose -f docker-compose.yml up -d papermate
docker compose -f docker-compose.yml up -d nginx
echo "回滚完成。确认 curl http://127.0.0.1/api/v1/healthz"
