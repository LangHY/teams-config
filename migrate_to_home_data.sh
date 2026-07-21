#!/bin/bash
# 服务器迁移脚本：/dev/lang/offline → /home_data/teacher03/lang
# 在本机运行：bash migrate_to_home_data.sh
set -e

SERVER="teacher03@172.20.150.239"
REMOTE="/home_data/teacher03/lang"

echo "=========================================="
echo "  迁移到 $REMOTE"
echo "=========================================="
echo ""

# 创建远程目录
ssh "$SERVER" "mkdir -p $REMOTE/cudnn9 $REMOTE/alpamayo_deps"



# 6. pip包 + av + physical-ai-av
echo "📦 [6/6] 上传pip包..."
scp ~/hf_offline/alpamayo_deps/av-18.0.0-cp311-abi3-manylinux_2_28_x86_64.whl "$SERVER:$REMOTE/"
scp ~/hf_offline/alpamayo_deps/physical_ai_av-0.2.0-py3-none-any.whl "$SERVER:$REMOTE/"

echo ""
echo "=========================================="
echo "  ✅ 上传完成"
echo "=========================================="
echo ""
echo "服务器上还需要："
echo "  1. 编译llama.cpp:   cd $REMOTE && bash llama_build.sh"
echo "  2. 加载Docker镜像:  cd $REMOTE && gunzip -c alpamayo-full.tar.gz | sudo docker load"
echo "  3. 启动模型:        cd $REMOTE/hf_offline && ./llama-server-new -m Qwen3.6-27B-Q4_K_M.gguf --host 0.0.0.0 --port 8000 -ngl 99 -c 204800 --jinja"
