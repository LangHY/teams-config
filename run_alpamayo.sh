#!/bin/bash
# Alpamayo 容器启动脚本
# 用法:
#   bash run_alpamayo.sh              # 进入容器
#   bash run_alpamayo.sh opencode     # 启动OpenCode
#   bash run_alpamayo.sh "问题"        # 单次对话

DEPLOY_DIR="/home_data/teacher03/lang"
MODEL_DIR="$DEPLOY_DIR/alpamayo1.5-10B"
CUDA_DEBS="$DEPLOY_DIR"

# 检查CUDA库是否已补全
NEED_CUDA_FIX=$(sudo docker run --rm alpamayo:latest bash -c \
    "ldconfig -p | grep -q cusparseLt && echo no || echo yes" 2>/dev/null)

COMMON_ARGS="--gpus all --network=host \
    -v $MODEL_DIR:/workspace/models \
    -v /:/host \
    -w /host"

# 首次补全CUDA库
if [ "$NEED_CUDA_FIX" = "yes" ]; then
    echo "🔧 首次运行：补全CUDA库..."
    sudo docker run --gpus all --network=host \
        -v "$CUDA_DEBS":/mnt \
        -it alpamayo:latest bash -c "
            mkdir -p /tmp/cudalib
            dpkg-deb -x /mnt/libcusparselt0-cuda-12.deb /tmp/cudalib/ 2>/dev/null || true
            dpkg-deb -x /mnt/cuda-cupti-12-8.deb /tmp/cudalib/ 2>/dev/null || true
            cp -a /tmp/cudalib/usr/lib/x86_64-linux-gnu/libcusparseLt*.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
            cp -a /tmp/cudalib/usr/lib/x86_64-linux-gnu/libcupti*.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
            ldconfig
            echo '✅ CUDA库补全完成'
        "
    CONTAINER_ID=$(sudo docker ps -l -q)
    if [ -n "$CONTAINER_ID" ]; then
        sudo docker commit "$CONTAINER_ID" alpamayo:latest
        echo "✅ 镜像已更新"
    fi
fi

# OpenCode配置（直接写入，不用镜像内的脚本）
OPENCODE_CONFIG='
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/config.json << "OCEOF"
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen3.6-27B",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "***"
      },
      "models": {
        "qwen3.6-27b": {
          "name": "Qwen3.6-27B",
          "limit": {
            "context": 204800,
            "output": 8192
          }
        }
      }
    }
  }
}
OCEOF
'

if [ "$1" = "opencode" ]; then
    sudo docker run $COMMON_ARGS \
        -it alpamayo:latest bash -c "
            $OPENCODE_CONFIG
            opencode
        "
elif [ -n "$1" ]; then
    sudo docker run $COMMON_ARGS \
        -it alpamayo:latest bash -c "
            $OPENCODE_CONFIG
            opencode run '$1'
        "
else
    sudo docker run $COMMON_ARGS \
        -it alpamayo:latest bash
fi
