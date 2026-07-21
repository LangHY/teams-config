#!/bin/bash
# ============================================================
# 编译 llama.cpp (CUDA 版) - 离线环境完整构建脚本
# ============================================================
set -e

# ============================================================
# Step 0: 一次性修复 conda GCC 缺少的所有系统头文件
# ============================================================
echo "=== 修复 conda GCC 系统头文件 ==="

SYSROOT="/home_data/teacher03/lang/gcc9/x86_64-conda-linux-gnu/sysroot/usr/include"
SRC_INCLUDE="/usr/include"
SRC_INCLUDE_ARCH="/usr/include/x86_64-linux-gnu"

# 需要从系统链接到 sysroot 的目录列表
SYMLINK_DIRS=(
    # 内核头文件
    "linux"
    "asm-generic"
    # 架构相关头文件（x86_64）
    "asm"
    # 常见系统头文件目录
    "drm"
    "misc"
    "mtd"
    "rdma"
    "scsi"
    "sound"
    "video"
    "xen"
)

# 从 /usr/include/ 链接的目录
for dir in "${SYMLINK_DIRS[@]}"; do
    # 确定源目录
    if [ -d "$SRC_INCLUDE_ARCH/$dir" ]; then
        SRC="$SRC_INCLUDE_ARCH/$dir"
    elif [ -d "$SRC_INCLUDE/$dir" ]; then
        SRC="$SRC_INCLUDE/$dir"
    else
        continue  # 源目录不存在，跳过
    fi

    # 创建符号链接（如果目标不存在）
    if [ ! -e "$SYSROOT/$dir" ]; then
        echo "  链接: $dir -> $SRC"
        ln -sf "$SRC" "$SYSROOT/$dir"
    fi
done

# ============================================================
# Step 1: 编译 llama.cpp
# ============================================================
echo ""
echo "=== 开始编译 llama.cpp ==="

cd "$(dirname "$0")/llama.cpp"

# 设置 PATH 包含 cmake 3.28.3（需要 cmake >= 3.14）
export PATH="/home_data/teacher03/lang/cmake-3.28.3-linux-x86_64/bin:$PATH"

# 使用 conda GCC 9.5.0 作为 CUDA 主机编译器（支持 C++17）
export CUDAHOSTCXX=/home_data/teacher03/lang/gcc9/bin/x86_64-conda-linux-gnu-g++

# 清理旧的 build 目录，避免 CMake 缓存问题
rm -rf build
mkdir -p build && cd build

# 使用 conda GCC 9.5.0 编译整个项目（包括 CPU 和 CUDA）
export CC=/home_data/teacher03/lang/gcc9/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home_data/teacher03/lang/gcc9/bin/x86_64-conda-linux-gnu-g++

# 禁用 UI 组件（避免潜在的兼容性问题）
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_UI=OFF \
    -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_EXE_LINKER_FLAGS="-L/home_data/teacher03/cuda-11.4/lib64 -lcuda"

# 编译（使用所有可用核心）
make -j$(nproc)

echo ""
echo "=== 编译完成！==="
echo "二进制文件位置: $(pwd)/bin/"
ls -la bin/llama-cli bin/llama-server bin/llama-quantize 2>/dev/null || echo "（部分二进制文件可能不存在，这是正常的）"
