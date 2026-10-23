#!/usr/bin/env bash
# Timestral API 启动脚本
# 用法: ./server.sh [uvicorn 额外参数]
# 环境变量: HOST(默认 0.0.0.0) PORT(默认 8000) TIMESTRAL_MODEL_PATH TIMESTRAL_DEVICE TIMESTRAL_MAX_HORIZON
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -x ".venv/bin/uvicorn" ]; then
    echo "[server] 未找到 .venv，请先执行 ./install.sh" >&2
    exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "[server] http://${HOST}:${PORT}/docs  (模型加载约需几秒，就绪后 /health 可访问)"
exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
