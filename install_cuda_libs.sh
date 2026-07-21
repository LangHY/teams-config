#!/bin/bash
# 在服务器上安装CUDA库到Alpamayo Docker镜像
set -e

echo "=== 安装CUDA库到Alpamayo容器 ==="

# 启动容器并安装所有库
CONTAINER_ID=$(sudo docker run --gpus all \
  -v /dev/lang/offline/cudnn9:/mnt \
  -d alpamayo:latest bash -c "
    echo '1. 安装 CUDA 12.8 runtime...' && \
    dpkg -i /mnt/cuda-cudart-12-8.deb && \
    echo '2. 安装 CUPTI...' && \
    dpkg -i /mnt/cuda-cupti-12-8.deb && \
    echo '3. 安装 cuDNN 9...' && \
    dpkg -i /mnt/libcudnn9-cuda-12.deb && \
    dpkg -i /mnt/libcudnn9-headers-cuda-12.deb && \
    dpkg -i /mnt/libcudnn9-dev-cuda-12.deb && \
    echo '4. 安装 cuSPARSELt...' && \
    dpkg -i /mnt/libcusparselt0-cuda-12.deb && \
    ldconfig && \
    echo '=== 验证 ===' && \
    echo 'libcudart:' && ls -la /usr/lib/x86_64-linux-gnu/libcudart* && \
    echo 'libcudnn:' && ls -la /usr/lib/x86_64-linux-gnu/libcudnn* && \
    echo 'libcusparseLt:' && ls -la /usr/lib/x86_64-linux-gnu/libcusparseLt* && \
    echo 'libcupti:' && ls -la /usr/lib/x86_64-linux-gnu/libcupti* && \
    echo '=== 完成 ==='
")

echo "容器ID: $CONTAINER_ID"
echo "等待安装完成..."
sudo docker wait $CONTAINER_ID
sudo docker logs $CONTAINER_ID

# 保存为新镜像
echo "保存镜像..."
sudo docker commit $CONTAINER_ID alpamayo:latest

echo "=== 完成 ==="
echo "验证: sudo docker run --rm --gpus all alpamayo:latest ldconfig -p | grep -E 'cudnn|cusparselt|cudart|cupti'"
