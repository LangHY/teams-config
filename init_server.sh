#!/bin/bash
# 服务器端初始化脚本
# 在服务器上运行：bash init_server.sh
set -e

BASE="/home_data/teacher03/lang"
cd "$BASE"

echo "=========================================="
echo "  服务器初始化"
echo "=========================================="
echo ""

# 1. 解压llama.cpp源码
echo "📦 解压llama.cpp..."
mkdir -p hf_offline
cd hf_offline
tar xzf ../llama-cpp-src.tar.gz
cd ..

# 2. 编译llama.cpp
echo "🔨 编译llama.cpp（需要GCC和cmake）..."
# 检查GCC版本
GCC_VER=$(gcc --version 2>/dev/null | head -1 || echo "未安装")
CMAKE_VER=$(cmake --version 2>/dev/null | head -1 || echo "未安装")
echo "  GCC: $GCC_VER"
echo "  CMAKE: $CMAKE_VER"

cd hf_offline
if [ -d "llama.cpp" ]; then
    cd llama.cpp
    mkdir -p build && cd build
    cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    make -j$(nproc) 2>&1 | tail -5
    # 复制二进制到hf_offline
    cp bin/llama-server "$BASE/hf_offline/llama-server-new"
    cp bin/llama-cli "$BASE/hf_offline/" 2>/dev/null || true
    echo "✅ llama.cpp编译完成"
    cd "$BASE"
else
    echo "❌ llama.cpp源码目录不存在"
fi

# 3. 加载Docker镜像
echo "📦 加载Docker镜像..."
if [ -f "$BASE/alpamayo-full.tar.gz" ]; then
    gunzip -c "$BASE/alpamayo-full.tar.gz" | sudo docker load
    echo "✅ Docker镜像加载完成"
else
    echo "⚠️  找不到alpamayo-full.tar.gz，跳过"
fi

# 4. 设置权限
chmod +x "$BASE/hf_offline/opencode" 2>/dev/null || true
chmod +x "$BASE/hf_offline/llama-server-new" 2>/dev/null || true

# 5. 创建启动脚本
cat > "$BASE/start_model.sh" << 'EOF'
#!/bin/bash
# 启动Qwen模型服务
BASE="/home_data/teacher03/lang"
pkill llama-server 2>/dev/null || true
sleep 1
cd "$BASE/hf_offline"
./llama-server-new \
    -m "$BASE/Qwen3.6-27B-Q4_K_M.gguf" \
    --host 0.0.0.0 --port 8000 \
    -ngl 99 -c 204800 --jinja
EOF
chmod +x "$BASE/start_model.sh"

# 6. 更新run_alpamayo.sh路径
cat > "$BASE/run_alpamayo.sh" << 'SCRIPTEOF'
#!/bin/bash
# Alpamayo 容器启动脚本
# 用法:
#   bash run_alpamayo.sh              # 进入容器
#   bash run_alpamayo.sh opencode     # 启动OpenCode
#   bash run_alpamayo.sh "问题"        # 单次对话

DEPLOY_DIR="/home_data/teacher03/lang"
MODEL_DIR="$DEPLOY_DIR/alpamayo1.5-10B"
CUDA_DEBS="$DEPLOY_DIR/cudnn9"

# 补全CUDA库
NEED_CUDA_FIX=$(sudo docker run --rm alpamayo:latest bash -c \
    "ldconfig -p | grep -q cusparseLt && echo no || echo yes" 2>/dev/null)

COMMON_ARGS="--gpus all --network=host \
    -v $MODEL_DIR:/workspace/models \
    -v /:/host \
    -w /host"

if [ "$NEED_CUDA_FIX" = "yes" ]; then
    echo "🔧 补全CUDA库..."
    sudo docker run --gpus all --network=host \
        -v "$CUDA_DEBS":/mnt \
        -it alpamayo:latest bash -c "
            mkdir -p /tmp/cudalib
            dpkg-deb -x /mnt/libcusparselt0-cuda-12.deb /tmp/cudalib/ 2>/dev/null || true
            dpkg-deb -x /mnt/cuda-cupti-12-8.deb /tmp/cudalib/ 2>/dev/null || true
            cp -a /tmp/cudalib/usr/lib/x86_64-linux-gnu/libcusparseLt*.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
            cp -a /tmp/cudalib/usr/lib/x86_64-linux-gnu/libcupti*.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
            ldconfig
        "
    CID=$(sudo docker ps -l -q)
    [ -n "$CID" ] && sudo docker commit "$CID" alpamayo:latest && echo "✅ 镜像已更新"
fi

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
    sudo docker run $COMMON_ARGS -it alpamayo:latest bash -c "
        $OPENCODE_CONFIG
        opencode
    "
elif [ -n "$1" ]; then
    sudo docker run $COMMON_ARGS -it alpamayo:latest bash -c "
        $OPENCODE_CONFIG
        opencode run '$1'
    "
else
    sudo docker run $COMMON_ARGS -it alpamayo:latest bash
fi
SCRIPTEOF
chmod +x "$BASE/run_alpamayo.sh"

echo ""
echo "=========================================="
echo "  ✅ 初始化完成"
echo "=========================================="
echo ""
echo "  bash $BASE/start_model.sh           # 启动模型"
echo "  bash $BASE/run_alpamayo.sh opencode  # 启动OpenCode"
echo "  bash $BASE/run_alpamayo.sh           # 进入容器"
echo ""
