#!/bin/bash
# Alpamayo Docker 完整部署脚本
set -e

echo "=========================================="
echo "  Alpamayo 1.5 + OpenCode Docker 部署"
echo "=========================================="
echo ""

DEPLOY_DIR="/home_data/teacher03/lang"
TAR_FILE="$DEPLOY_DIR/alpamayo-full.tar"

cd "$DEPLOY_DIR"

# 如果是gz压缩的，先解压
if [ -f "$DEPLOY_DIR/alpamayo-full.tar.gz" ] && [ ! -f "$TAR_FILE" ]; then
    echo "📦 解压镜像..."
    gunzip -c "$DEPLOY_DIR/alpamayo-full.tar.gz" > "$TAR_FILE"
fi

if [ ! -f "$TAR_FILE" ]; then
    echo "❌ 找不到 $TAR_FILE"
    echo "   请先上传到 $DEPLOY_DIR/"
    exit 1
fi

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装"
    exit 1
fi

# 检查GPU支持（不自动安装，提示手动操作）
echo "🔍 检查GPU支持..."
if ! sudo docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "⚠️  GPU不可用，请确认已安装NVIDIA Container Toolkit:"
    echo "   sudo apt-get install -y nvidia-docker2"
    echo "   sudo systemctl restart docker"
    echo ""
    echo "   继续加载镜像..."
fi

# 加载镜像
echo "📦 加载Docker镜像（约6GB，请耐心等待）..."
sudo docker load < "$TAR_FILE"

# 验证
echo "✅ 验证镜像..."
sudo docker images | grep alpamayo

# 创建启动脚本
echo "📝 创建启动脚本..."
cat > "$DEPLOY_DIR/run_alpamayo.sh" << 'SCRIPT'
#!/bin/bash
# 用法:
#   bash run_alpamayo.sh              # 进入容器
#   bash run_alpamayo.sh opencode     # 启动OpenCode
#   bash run_alpamayo.sh "问题"        # 单次对话

DEPLOY_DIR="/home_data/teacher03/lang"
MODEL_DIR="$DEPLOY_DIR/alpamayo1.5-10B"

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
chmod +x "$DEPLOY_DIR/run_alpamayo.sh"

echo ""
echo "=========================================="
echo "  ✅ 部署完成！"
echo "=========================================="
echo ""
echo "  bash $DEPLOY_DIR/run_alpamayo.sh              # 进入容器"
echo "  bash $DEPLOY_DIR/run_alpamayo.sh opencode      # 启动OpenCode"
echo "  bash $DEPLOY_DIR/run_alpamayo.sh '你的问题'     # 单次对话"
echo ""
