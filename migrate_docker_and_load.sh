#!/bin/bash
# Docker数据迁移 + 镜像加载脚本
set -e

echo "=========================================="
echo "  Docker数据迁移 + Alpamayo镜像加载"
echo "=========================================="
echo ""

NEW_DOCKER_ROOT="/home_data/docker"
TAR_GZ="/dev/lang/offline/alpamayo-full.tar.gz"

# 检查压缩包
if [ ! -f "$TAR_GZ" ]; then
    echo "❌ 找不到 $TAR_GZ"
    exit 1
fi

# 1. 停止Docker
echo "⏹️  停止Docker..."
sudo systemctl stop docker

# 2. 迁移数据
echo "📦 迁移Docker数据到 $NEW_DOCKER_ROOT ..."
sudo mkdir -p "$NEW_DOCKER_ROOT"
if [ -d "/var/lib/docker" ] && [ "$(ls -A /var/lib/docker 2>/dev/null)" ]; then
    sudo rsync -aP /var/lib/docker/ "$NEW_DOCKER_ROOT/"
fi

# 3. 配置新路径
echo "⚙️  配置Docker数据目录..."
sudo mkdir -p /etc/docker
echo "{\"data-root\": \"$NEW_DOCKER_ROOT\"}" | sudo tee /etc/docker/daemon.json

# 4. 启动Docker
echo "▶️  启动Docker..."
sudo systemctl start docker

# 5. 验证
echo "🔍 验证配置..."
DOCKER_ROOT=$(sudo docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $NF}')
if [ "$DOCKER_ROOT" = "$NEW_DOCKER_ROOT" ]; then
    echo "✅ Docker数据目录: $DOCKER_ROOT"
else
    echo "❌ 配置未生效，当前: $DOCKER_ROOT"
    exit 1
fi

# 6. 删除旧数据
echo "🗑️  释放根分区空间..."
sudo rm -rf /var/lib/docker
echo "✅ 旧数据已删除"

# 7. 加载镜像
echo "📦 加载Alpamayo镜像（约6GB，请耐心等待）..."
gunzip -c "$TAR_GZ" | sudo docker load

# 8. 验证镜像
echo "🔍 验证镜像..."
sudo docker images | grep alpamayo

# 9. 创建启动脚本
cat > /home_data/teacher03/lang/run_alpamayo.sh << 'SCRIPT'
#!/bin/bash
# 用法:
#   bash run_alpamayo.sh              # 进入容器
#   bash run_alpamayo.sh opencode     # 启动OpenCode
#   bash run_alpamayo.sh "问题"        # 单次对话

MODEL_DIR="/home_data/teacher03/lang/alpamayo1.5-10B"

if [ "$1" = "opencode" ]; then
    sudo docker run --gpus all --network=host \
        -v "$MODEL_DIR":/workspace/models \
        -it alpamayo:latest bash -c "
            bash /workspace/setup_opencode.sh
            opencode
        "
elif [ -n "$1" ]; then
    sudo docker run --gpus all --network=host \
        -v "$MODEL_DIR":/workspace/models \
        -it alpamayo:latest opencode run "$1"
else
    sudo docker run --gpus all --network=host \
        -v "$MODEL_DIR":/workspace/models \
        -it alpamayo:latest bash
fi
SCRIPT
chmod +x /home_data/teacher03/lang/run_alpamayo.sh

echo ""
echo "=========================================="
echo "  ✅ 全部完成！"
echo "=========================================="
echo ""
echo "  bash /home_data/teacher03/lang/run_alpamayo.sh              # 进入容器"
echo "  bash /home_data/teacher03/lang/run_alpamayo.sh opencode      # 启动OpenCode"
echo "  bash /home_data/teacher03/lang/run_alpamayo.sh '你的问题'     # 单次对话"
echo ""
