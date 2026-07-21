#!/bin/bash
# ============================================================
#  Qwen3.6-27B (GGUF) + llama.cpp + OpenCode 离线安装
#  适用于: Ubuntu + CUDA 11.4 + V100
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Step 1/3: 编译 llama.cpp (CUDA)"
echo "============================================"
if [ ! -d "$SCRIPT_DIR/llama.cpp" ]; then
    cd "$SCRIPT_DIR"
    tar xzf llama-cpp-src.tar.gz
    mv llama.cpp-b* llama.cpp
fi

# 需要 cmake
if ! command -v cmake &>/dev/null; then
    echo "安装 cmake..."
    if command -v conda &>/dev/null; then
        conda install -y cmake
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y cmake
    else
        echo "错误: 需要 cmake，请手动安装"; exit 1
    fi
fi

cd "$SCRIPT_DIR/llama.cpp"
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "✅ llama.cpp 编译完成"

echo ""
echo "============================================"
echo "  Step 2/3: 配置 OpenCode"
echo "============================================"
cd "$SCRIPT_DIR"
if [ ! -f opencode ]; then
    tar xzf opencode-linux-x86_64.tar.gz
    chmod +x opencode
fi

mkdir -p ~/.config/opencode
cat > ~/.config/opencode/config.json << 'OCEOF'
{
  "provider": {
    "local": {
      "name": "Local Qwen3.6-27B",
      "type": "openai",
      "apiKey": "not-needed",
      "baseURL": "http://localhost:8000/v1",
      "models": {
        "qwen3.6-27b": {
          "name": "Qwen3.6-27B",
          "contextWindow": 8192,
          "maxTokens": 4096
        }
      }
    }
  },
  "defaultModel": "local/qwen3.6-27b"
}
OCEOF
echo "✅ OpenCode 配置完成"

echo ""
echo "============================================"
echo "  Step 3/3: 验证"
echo "============================================"
GGUF=$(ls "$SCRIPT_DIR"/*.gguf 2>/dev/null | head -1)
if [ -z "$GGUF" ]; then
    echo "❌ 未找到 .gguf 模型文件"; exit 1
fi
echo "✅ 模型: $(basename "$GGUF") ($(ls -lh "$GGUF" | awk '{print $5}'))"

echo ""
echo "============================================"
echo "  ✅ 全部安装完成!"
echo "============================================"
echo ""
echo "启动方式:"
echo "  bash $SCRIPT_DIR/serve.sh     # 启动 API 服务"
echo "  $SCRIPT_DIR/opencode           # 启动 Agent"
echo "  bash $SCRIPT_DIR/run_agent.sh  # 一键启动"
