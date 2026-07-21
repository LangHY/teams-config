# Alpamayo 1.5-10B 服务器部署说明

## 概述

本说明指导在 Ubuntu 18.04 (glibc 2.27, CUDA 11.4, driver 470) 服务器上部署 Alpamayo 1.5-10B 自动驾驶模型。

所有依赖已打包为 Docker 镜像，解决 glibc/CUDA 兼容性问题。

## 前置条件

- NVIDIA GPU (V100 32GB x2)
- Docker 28.4.0+
- NVIDIA Container Toolkit

## 文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `alpamayo-docker.tar` | 3.8GB | Docker 镜像（CUDA 12.4 + Python 3.12 + PyTorch 2.8.0 + 所有依赖） |
| `alpamayo1.5-10B/` | ~22GB | 模型权重（已上传） |

## 部署步骤

### 第一步：验证 NVIDIA Container Toolkit

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

如果报错 `permission denied`，用 sudo。
如果报错 `could not select device driver`，需要安装 NVIDIA Container Toolkit：

```bash
# 添加仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 第二步：加载 Docker 镜像

```bash
cd /dev/lang/offline
sudo docker load < alpamayo-docker.tar
```

验证加载成功：
```bash
sudo docker images | grep alpamayo
# 应显示: alpamayo   latest   ...   3.8GB
```

### 第三步：运行推理测试

```bash
sudo docker run --gpus all \
  -v /dev/lang/offline/alpamayo1.5-10B:/workspace/models \
  -it alpamayo:latest \
  python src/alpamayo1_5/test_inference.py
```

### 第四步：进入容器调试（可选）

```bash
sudo docker run --gpus all \
  -v /dev/lang/offline/alpamayo1.5-10B:/workspace/models \
  -it alpamayo:latest bash
```

容器内环境：
- Python 3.12
- PyTorch 2.8.0+cu128
- CUDA 12.4.1
- transformers 4.57.1
- 仓库代码在 `/workspace/alpamayo1.5-main/`

## 修改推理参数

进入容器后编辑 `src/alpamayo1_5/test_inference.py`：

```python
# 减少采样数（省内存）
num_traj_samples = 1  # 默认16

# 使用 SDPA 替代 flash-attn（无需编译）
model = Alpamayo1_5.from_pretrained(
    "/workspace/models",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
```

## 多 GPU 推理

```python
model = Alpamayo1_5.from_pretrained(
    "/workspace/models",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
```

## 快捷脚本

创建 `/dev/lang/offline/run_alpamayo.sh`：

```bash
#!/bin/bash
sudo docker run --gpus all \
  -v /dev/lang/offline/alpamayo1.5-10B:/workspace/models \
  --workdir /workspace/alpamayo1.5-main \
  -it alpamayo:latest \
  "$@"
```

使用：
```bash
chmod +x /dev/lang/offline/run_alpamayo.sh
# 运行推理
/dev/lang/offline/run_alpamayo.sh python src/alpamayo1_5/test_inference.py
# 进入容器
/dev/lang/offline/run_alpamayo.sh bash
```

## 常见问题

### Q: OOM (显存不足)

```bash
# 减少采样数
num_traj_samples = 1

# 或4bit量化
model = Alpamayo1_5.from_pretrained(..., load_in_4bit=True)
```

### Q: physical-ai-av 缺失

该包需要从 HuggingFace 下载数据集，必须联网。离线环境需提前下载数据集并挂载。

### Q: 容器内无法访问网络

```bash
sudo docker run --gpus all --network=host -it alpamayo:latest bash
```

### Q: Docker 权限问题

每次用 sudo，或将用户加入 docker 组：
```bash
sudo usermod -aG docker teacher03
# 重新登录生效
```

## 镜像技术栈

| 组件 | 版本 |
|------|------|
| Base Image | nvidia/cuda:12.4.1-runtime-ubuntu22.04 |
| Python | 3.12 |
| PyTorch | 2.8.0+cu128 |
| transformers | 4.57.1 |
| CUDA (容器内) | 12.4.1 |
| 驱动兼容 | 容器自带 CUDA，不依赖宿主机驱动版本 |

## 参考

- 仓库: https://github.com/NVlabs/alpamayo1.5
- 模型: https://huggingface.co/nvidia/Alpamayo-1.5-10B
