#!/bin/bash
# ============================================================
# 一键启动: API 服务 + OpenCode Agent
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8000

# 启动 API 服务（后台）
echo "启动 Qwen3.6-27B API 服务..."
bash "$SCRIPT_DIR/serve.sh" &
SERVER_PID=$!

# 等待服务就绪
echo "等待服务启动..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "API 服务已就绪!"
        break
    fi
    sleep 2
done

# 启动 OpenCode
echo "启动 OpenCode Agent..."
"$SCRIPT_DIR/opencode" --model "local/qwen3.6-27b"

# 退出时关闭 API 服务
kill $SERVER_PID 2>/dev/null
echo "已关闭"
