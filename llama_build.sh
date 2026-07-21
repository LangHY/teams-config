#!/bin/bash
# llama.cpp 编译脚本
set -e

BASE="/home_data/teacher03/lang"
CUDA_HOME="/home_data/teacher03/cuda-11.4"

# 解压GCC 9（如果还没解压）
if [ ! -d "$BASE/gcc9/bin" ]; then
    echo "📦 解压GCC 9..."
    cd "$BASE" && tar xzf gcc9.tar.gz
fi

# 解压cmake（如果还没解压）
if [ ! -d "$BASE/cmake-3.28.3-linux-x86_64" ]; then
    echo "📦 解压cmake..."
    cd "$BASE" && tar xzf cmake-3.28-linux-x86_64.tar.gz
fi

# 找GCC 9实际路径
GCC_BIN=$(find "$BASE/gcc9/bin" -name "*gcc" -type f | head -1)
GXX_BIN=$(find "$BASE/gcc9/bin" -name "*g++" -type f | head -1)
CMAKE_BIN="$BASE/cmake-3.28.3-linux-x86_64/bin/cmake"

echo "=== 环境检查 ==="
echo "GCC: $GCC_BIN"
echo "GXX: $GXX_BIN"
echo "CMAKE: $CMAKE_BIN ($($CMAKE_BIN --version | head -1))"
echo "NVCC: $CUDA_HOME/bin/nvcc ($($CUDA_HOME/bin/nvcc --version | tail -1))"
echo ""

# 解压llama.cpp源码
cd "$BASE"
if [ ! -d "llama.cpp" ]; then
    echo "📦 解压llama.cpp..."
    tar xzf llama-cpp-src.tar.gz
    mv llama.cpp-b9964 llama.cpp
fi

# 设置环境
export PATH="$BASE/gcc9/bin:$CUDA_HOME/bin:$BASE/cmake-3.28.3-linux-x86_64/bin:$PATH"
export LD_LIBRARY_PATH="$BASE/gcc9/lib:$BASE/gcc9/x86_64-conda-linux-gnu/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 编译
echo "🔨 编译llama.cpp..."
cd "$BASE/llama.cpp"
rm -rf build && mkdir build && cd build

$CMAKE_BIN .. \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$GCC_BIN"

make -j$(nproc)

# 复制二进制
mkdir -p "$BASE/hf_offline"
cp bin/llama-server "$BASE/hf_offline/llama-server-new"
chmod +x "$BASE/hf_offline/llama-server-new"

echo ""
echo "✅ 编译完成: $BASE/hf_offline/llama-server-new"
