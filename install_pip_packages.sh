#!/bin/bash
# 安装缺失pip包到容器并commit
set -e

DEPLOY_DIR="/home_data/teacher03/lang"

echo "📦 启动容器安装缺失包..."
CONTAINER_ID=$(sudo docker run --gpus all --network=host \
    -v "$DEPLOY_DIR":/mnt \
    -d alpamayo:latest sleep 600)

echo "容器ID: $CONTAINER_ID"

echo "📦 安装av + physical-ai-av..."
sudo docker exec "$CONTAINER_ID" pip install --no-deps \
    /mnt/av-18.0.0-cp311-abi3-manylinux_2_28_x86_64.whl \
    /mnt/physical_ai_av-0.2.0-py3-none-any.whl

echo "🔍 验证..."
sudo docker exec "$CONTAINER_ID" python -c "
import av; print('av:', av.__version__)
import physical_ai_av; print('physical_ai_av: ok')
"

echo "💾 保存镜像..."
sudo docker stop "$CONTAINER_ID"
sudo docker commit "$CONTAINER_ID" alpamayo:latest

echo "✅ 完成"
