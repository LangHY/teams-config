#!/bin/bash
# ============================================================
# 启动 OpenAI 兼容 API 服务
# 默认: 端口8000, 全部层放GPU
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/llama.cpp/build/bin"
PORT="${1:-8000}"
GPU_LAYERS="${2:-99}"
CTX_SIZE="${3:-8192}"

GGUF=$(ls "$SCRIPT_DIR"/*.gguf 2>/dev/null | head -1)
if [ -z "$GGUF" ]; then
    echo "错误: 未找到 .gguf 文件"; exit 1
fi

echo "============================================"
echo "  Qwen3.6-27B Local API Server"
echo "  模型: $(basename "$GGUF")"
echo "  端口: $PORT | GPU层数: $GPU_LAYERS | 上下文: $CTX_SIZE"
echo "  API: http://localhost:$PORT/v1"
echo "============================================"

exec "$BUILD/llama-server" \
    -m "$GGUF" \
    --host 0.0.0.0 \
    --port "$PORT" \
    -ngl "$GPU_LAYERS" \
    -c "$CTX_SIZE" \
    --chat-template chatml
