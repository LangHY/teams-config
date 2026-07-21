#!/bin/bash
# 安装NVIDIA Container Toolkit（离线）
set -e

echo "📦 解压..."
cd /home_data/teacher03/lang
tar xzf nvidia-container-toolkit.tar.gz

echo "📦 安装依赖..."
sudo dpkg -i libnvidia-container1.deb
sudo dpkg -i libnvidia-container-tools.deb

echo "📦 安装toolkit..."
sudo dpkg -i nvidia-container-toolkit-base.deb
sudo dpkg -i nvidia-container-toolkit.deb

echo "⚙️  配置Docker..."
sudo nvidia-ctk runtime configure --runtime=docker

echo "🔄 重启Docker..."
sudo systemctl restart docker

echo "🔍 验证..."
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi

echo "✅ 安装完成！"
