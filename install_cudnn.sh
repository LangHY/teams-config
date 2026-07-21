#!/bin/bash
# 在服务器上安装cuDNN到Alpamayo Docker镜像
set -e

echo "=== 安装cuDNN到Alpamayo容器 ==="

# 启动容器并安装cuDNN
CONTAINER_ID=$(sudo docker run --gpus all \
  -v /dev/lang/offline:/mnt \
  -d alpamayo:latest bash -c "
    dpkg -i /mnt/libcudnn9-cuda-12.deb && \
    dpkg -i /mnt/libcudnn9-headers-cuda-12.deb && \
    dpkg -i /mnt/libcudnn9-dev-cuda-12.deb && \
    ldconfig && \
    echo 'cuDNN安装完成' && \
    ldconfig -p | grep cudnn
")

echo "容器ID: $CONTAINER_ID"
echo "等待安装完成..."
sudo docker wait $CONTAINER_ID
sudo docker logs $CONTAINER_ID

# 保存为新镜像
echo "保存镜像..."
sudo docker commit $CONTAINER_ID alpamayo:latest

echo "=== 完成 ==="
echo "验证: sudo docker run --rm alpamayo:latest ldconfig -p | grep cudnn"
