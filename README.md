# Alpamayo 1.5 → 4B 边缘部署方案 v3

## 基于 LightVLA 代码适配 + ModelOpt 蒸馏 + 渐进式压缩

**目标**：将 NVIDIA Alpamayo 1.5（10B）压缩为 4B 模型，部署到 16GB Jetson Orin NX 平台，实现小车实时自动驾驶控制。

**参考资源**：

- [LightVLA 论文 (arXiv:2509.12594)](https://arxiv.org/abs/2509.12594) — 无参数 Token 剪枝算法
- [LightVLA 代码仓库](https://github.com/LiAutoAD/LightVLA) — MIT 许可，基于 OpenVLA-OFT
- [LightVLA 预训练权重](https://huggingface.co/TTJiang/models?search=lightvla) — HuggingFace 托管
- [VLA 模型推理加速全栈实践](https://blog.csdn.net/weixin_29038155/article/details/162952006) — 工程经验
- [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) — 官方蒸馏/量化工具
- [Alpamayo 1.5 官方仓库](https://github.com/NVlabs/alpamayo1.5)

**v3 修订说明**：

- 基于 LightVLA 源码分析，明确了适配 Alpamayo 的具体工作量和难点
- 结合项目实际目标（1-4B、<200ms、<6GB、≥85%），采用渐进式压缩策略
- 引入 NVIDIA ModelOpt 处理蒸馏，降低工程风险
- 增加 Flow Matching 蒸馏的详细方案

***

## 核心概念：三种压缩手段的区别

**本方案使用三种互补的压缩手段，各减不同的东西**：

| 手段                          | 减什么         | 效果               | 工具       |
| --------------------------- | ----------- | ---------------- | -------- |
| **蒸馏（Distillation）**        | 模型参数（权重）    | 模型文件变小：22GB→8GB  | ModelOpt |
| **Token 剪枝（Token Pruning）** | 输入 token 数量 | 推理变快，KV Cache 变小 | LightVLA |
| **INT8 量化（Quantization）**   | 权重精度        | 模型更小 + 推理更快      | TensorRT |

**三者叠加的效果**：

```
原始 10B FP16      22 GB 模型  |  24 GB 显存  |  ~200ms 延迟
      │
      │ 蒸馏（减参数：10B→4B）
      ▼
4B FP16             8 GB 模型  |  10 GB 显存  |  ~120ms 延迟
      │
      │ Token 剪枝（减输入：5000→1000 token）
      ▼
4B FP16 + 剪枝      8 GB 模型  |   7 GB 显存  |   ~80ms 延迟
      │
      │ INT8 量化（减精度：FP16→INT8）
      ▼
4B INT8 + 剪枝      4 GB 模型  |   6 GB 显存  |   ~60ms 延迟
```

**执行顺序**：蒸馏 → Token 剪枝 → 量化（先减参数，再减输入，最后减精度）

***

## 开发策略

### 开发策略：Claude Code 辅助开发

**Claude Code 能做的**：

- 编写 ModelOpt 蒸馏训练脚本
- 移植 LightVLA 核心代码到 Alpamayo
- ONNX 导出 + TensorRT 编译脚本
- 调试报错分析
- ROS2 节点代码

**团队必须自己做的**：

- 判断模型输出是否合理（领域知识）
- 决定 Loss 权重和超参调整方向（实验直觉）
- 硬件调试和实车测试（物理世界）
- 答辩时解释技术决策（理解原理）

**核心原则**：先跑通代码 → 遇到问题学理论 → 理解后验证 → 循环迭代

### 学习路径（以用促学）

```
第 1-2 周：PyTorch 速成（重点：tensor 操作、autograd、DataLoader）
第 3-4 周：HuggingFace 模型加载（from_pretrained、tokenizer、processor）
第 5-6 周：用 Claude Code 写 Alpamayo 推理 pipeline，在 CARLA 跑通
第 7-8 周：ModelOpt 蒸馏基础（跟着官方 tutorial 走）
后续：边做边学，遇到什么学什么
```

***

## 一、项目背景

### 1.1 原始模型架构（源码验证）

Alpamayo 1.5 是 NVIDIA 发布的自动驾驶 VLA 模型。经源码分析：

**VLM 骨干：Qwen3-VL-8B-Instruct**

- hidden\_dim: 4096, num\_layers: 36, num\_heads: 32, ffn\_dim: \~11008
- 视觉编码器：Qwen3-VL 内置 ViT

**扩散专家（Diffusion Expert）**

- 架构：VLM 文本模型的完整拷贝（`alpamayo1_5.py:97-105`）
- hidden\_dim: 4096, num\_layers: 36
- 使用 Flow Matching（10 步 Euler 求解器）
- 输出：(64, 2) 维动作空间——64 个轨迹点 × (加速度, 曲率)

**视觉 Token**

- 4 摄像头 × 4 帧 = 16 张图 × \~256-384 token = **\~4000-6000 个视觉 token**

| 配置         | 显存占用    |
| ---------- | ------- |
| 单轨迹采样      | \~24 GB |
| 多轨迹采样 (16) | \~40 GB |
| 模型权重文件     | \~22 GB |

### 1.2 部署目标（与研发计划书对齐）

| 指标    | 研发计划书目标    | 本方案目标    |
| ----- | ---------- | -------- |
| 参数量   | 1-4B       | 4B（渐进达成） |
| 推理延迟  | <200ms     | <80ms    |
| 显存占用  | <6GB       | <6GB     |
| 决策准确率 | ≥85%（相对原始） | ≥90%     |
| 控制频率  | —          | ≥15Hz    |

***

## 二、整体策略：渐进式压缩

### 2.1 为什么用渐进式？

```
激进方案：10B ──────→ 4B（一步压缩 60%）
                      ↑ 误差累积，失败无法定位

渐进方案：10B → 8B → 6B → 4B（每步压缩 20-30%）
              ↑      ↑      ↑
              每步验证，发现问题及时修正
```

### 2.2 三阶段路线图

| 阶段   | 目标             | 方法          | 验证标准              | 时间    |
| ---- | -------------- | ----------- | ----------------- | ----- |
| 阶段 1 | 10B → 8B       | ModelOpt 蒸馏 | 轨迹 L2 误差 <110%    | 4-6 周 |
| 阶段 2 | Token 剪枝       | LightVLA 适配 | 视觉 token 压缩 ≥60%  | 3-4 周 |
| 阶段 3 | 8B → 4B + INT8 | 继续蒸馏 + 量化   | 延迟 <80ms, 显存 <6GB | 4-6 周 |

**每个阶段都有 go/no-go 决策点**：

- 阶段 1 失败 → 停在 10B，用 INT8 量化直接部署（延迟可能 \~150ms）
- 阶段 2 失败 → 停在 8B + 量化（延迟 \~100ms，显存 \~8GB）
- 阶段 3 失败 → 停在 6B + 量化（延迟 \~80ms，显存 \~6GB）

**任何阶段的成功都是一个可用的成果。**

***

## 三、阶段 1：VLM 蒸馏（10B → 8B）

### 3.1 工具选择：NVIDIA ModelOpt

```bash
pip install nvidia-modelopt

# ModelOpt 支持的功能
- 量化（INT8/INT4/FP8）
- 蒸馏（Teacher-Student）
- 剪枝（结构化剪枝）
- 导出（ONNX/TensorRT）
```

### 3.2 蒸馏策略

**只蒸馏 VLM 部分，不动扩散专家。**

```
教师模型：Alpamayo 10B（VLM 36 层 + 扩散专家 36 层）
学生模型：Alpamayo 8B（VLM 28 层 + 扩散专家 36 层）
                    ↑ 减少 8 层 VLM，扩散专家保持不变
```

**Loss 函数**：

```
L_total = α·L_feature + β·L_attention + γ·L_output

L_feature (α=0.4): 特征层 MSE 对齐
  教师 36 层 → 锚点 [9, 18, 27, 36]
  学生 28 层 → 锚点 [7, 14, 21, 28]

L_attention (β=0.2): 注意力图 KL 散度
  只蒸馏"语言 token 对视觉 token 的注意力"

L_output (γ=0.4): 输出 logits KL 散度（温度 T=2.0）
```

### 3.3 训练配置

- 数据：NVIDIA PhysicalAI-Autonomous-Vehicles 数据集，\~10000 条
- GPU：2×A100 80GB（教师模型推理模式常驻 1 张）
- 训练时间：\~1 周
- 优化器：AdamW, lr=1e-5, cosine schedule

### 3.4 验证检查点

| 指标       | 通过标准       |
| -------- | ---------- |
| 轨迹 L2 误差 | < 原始的 110% |
| 推理链质量    | 人工评估可接受    |
| 碰撞率      | < 1%（仿真环境） |

***

## 四、阶段 2：LightVLA Token 剪枝

### 4.1 LightVLA 核心算法（源码分析）

LightVLA 的核心实现在 `prismatic/extern/hf/modeling_prismatic.py` 的 `TokenPruner` 类中：

**Token 评分（无参数）**：

```python
def get_score(self, patches, prompts):
    # RMSNorm 归一化（和 Qwen3-VL 一致）
    patches = self.rms_norm(patches)
    prompts = self.rms_norm(prompts)
    # 交叉注意力：视觉作为 Query，语言作为 K/V
    queries = F.scaled_dot_product_attention(patches, prompts, prompts)
    queries = self.rms_norm(queries)
    # 点积评分
    score = queries @ patches.transpose(-2, -1) * self.scale_factor
    return score
```

**可微分选择（Gumbel-Softmax + Straight-Through）**：

```python
def score_to_indices(self, score, patches):
    if self.noise_scale is not None:
        score = score + torch.rand_like(score) * self.noise_scale
    hard_score = F.one_hot(score.argmax(dim=-1), num_classes=self.num_patches)
    soft_score = torch.softmax(score, dim=-1)
    # 直通估计器：前向用硬的，梯度通过软的传回
    score = hard_score + soft_score - soft_score.detach()
    return score.argmax(dim=-1), score @ patches
```

**推理时的硬选择**：

```python
def score_to_mask(self, score):
    mask = torch.zeros(bsz, self.num_patches, dtype=torch.bool)
    indices = score.argmax(-1)
    mask[batch_indices, indices] = True
    return mask
```

### 4.2 适配 Alpamayo 的具体工作

**可直接复用的部分（\~60%）**：

- `TokenPruner` 类的核心逻辑（评分 + 选择）
- RMSNorm 和 `scaled_dot_product_attention`
- 序列重建逻辑（position\_ids、attention\_mask 更新）
- 噪声衰减策略

**需要修改的部分（\~40%）**：

| 修改项          | 原版 (OpenVLA-OFT)  | Alpamayo 适配      | 难度         |
| ------------ | ----------------- | ---------------- | ---------- |
| 视觉 token 提取  | DINOv2+SigLIP 双分支 | Qwen3-VL ViT 单分支 | 🟢 低       |
| num\_patches | 512               | \~5000           | 🟡 中（调整超参） |
| 动作 token 逻辑  | 自回归解码             | Flow Matching    | 🔴 高（需重写）  |
| 集成位置         | LLM 输入层           | LLM 输入层          | 🟢 低       |

**最困难的部分：Flow Matching 集成**

LightVLA 原版的动作生成是自回归的（LLM 直接输出动作 token），但 Alpamayo 用 Flow Matching（扩散专家从 VLM 的 KV Cache 去噪生成轨迹）。剪枝后的视觉 token 如何影响扩散专家的输入，需要专门设计。

### 4.3 适配方案

```python
class AlpamayoTokenPruner(TokenPruner):
    """
    适配 Alpamayo 的 Token 剪枝器
    核心修改：
    1. 适配 Qwen3-VL 的视觉 token 格式
    2. 处理 ~5000 个视觉 token（原版只有 512）
    3. 确保剪枝后的 KV Cache 正确传递给扩散专家
    """
    def __init__(self, num_patches=5000, hidden_dim=4096, ...):
        super().__init__(...)

    def forward(self, input_ids, attention_mask, position_ids,
                pixel_values, image_grid_thw, ...):
        # 1. 提取视觉 token（适配 Qwen3-VL）
        vision_outputs = self.vision_encoder(pixel_values, image_grid_thw)
        visual_tokens = vision_outputs.last_hidden_state  # [B, ~5000, 4096]

        # 2. 提取语言 token
        text_embeddings = self.embed_tokens(input_ids)

        # 3. 计算重要性分数
        scores = self.get_score(visual_tokens, text_embeddings)

        # 4. 可微分选择
        if self.training:
            selected_indices, pruned_tokens = self.score_to_indices(
                scores, visual_tokens
            )
        else:
            mask = self.score_to_mask(scores)
            pruned_tokens = visual_tokens[mask]

        # 5. 重建序列（保留 [CLS] + 选中的 patches + 语言 tokens）
        # 6. 更新 position_ids 和 attention_mask
        # 7. 送入 LLM 骨干

        # 关键：剪枝后的 KV Cache 会被扩散专家使用
        # 扩散专家通过 cross_attn 读取 VLM 的 KV Cache
        # 剪枝减少了视觉 token 数量，直接减少扩散专家的计算量
        return lm_output, pruned_visual_tokens
```

### 4.4 训练策略

**LightVLA 的噪声衰减**：

```python
# 训练过程中逐步降低噪声强度
noise_upper_bound = 1.0
noise_decay = 0.999  # 每步衰减

# 阶段 1（热身）：高噪声，探索多样选择
# 阶段 2（收敛）：中噪声，锁定最优子集
# 阶段 3（精调）：低噪声，接近推理行为
```

**预期保留率**：

- LightVLA 在 512 token 上平均保留 78 个（15.2%）
- Alpamayo 有 \~5000 token，按相同比例保留 \~750-1000 个
- 保留 \[CLS] token + position IDs（关键细节）

### 4.5 验证检查点

| 指标           | 通过标准               |
| ------------ | ------------------ |
| 视觉 token 压缩比 | ≥60%（5000→2000 以下） |
| 轨迹精度退化       | <5%（相对蒸馏后的 8B 模型）  |
| 推理延迟下降       | ≥30%（KV Cache 减少）  |

***

## 五、阶段 3：进一步压缩（8B → 4B）+ INT8 量化

### 5.1 继续蒸馏

在阶段 2 的基础上，继续将 VLM 从 8B 蒸馏到 4B：

```
学生模型：4B（VLM 20 层 + 扩散专家 20 层）
  - hidden_dim: 2048
  - num_heads: 16
  - ffn_dim: 5504
```

### 5.2 Flow Matching 蒸馏（关键难点）

**为什么 Flow Matching 蒸馏困难？**

Alpamayo 的扩散专家用 Flow Matching 从 VLM 的 KV Cache 去噪生成轨迹：

- 10 步 Euler 求解器
- 每步依赖 VLM 的交叉注意力上下文
- 直接蒸馏最终轨迹会丢失中间过程

**三阶段蒸馏策略**：

```
Step 1: 轨迹级蒸馏
  - 只匹配最终输出轨迹
  - L_traj = L2(teacher_traj, student_traj) + λ·CollisionPenalty
  - 快速验证 pipeline 是否工作

Step 2: 步级蒸馏
  - 匹配每个去噪步的中间轨迹
  - L_step = Σ_{t=1}^{10} MSE(teacher_traj_t, student_traj_t)
  - 确保学生学到完整的去噪过程

Step 3: 步数压缩（可选）
  - 将 10 步压缩到 5 步
  - 用教师 10 步输出作为目标，训练学生 5 步输出
  - 推理加速 2 倍
```

### 5.3 INT8 混合精度量化

| 模块       | 量化精度 | 原因          |
| -------- | ---- | ----------- |
| 视觉编码器    | INT8 | 鲁棒性强        |
| 语言模型主干   | INT8 | 文本推理，容错性尚可  |
| 扩散专家     | FP16 | 连续轨迹预测，精度敏感 |
| Token 评分 | —    | 无参数，无需量化    |

### 5.4 确定性推理模式

```python
# 启用确定性推理（牺牲 ~3-5% 性能换控制稳定性）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

**必须启用**，否则 PID 控制器会积分放大微小输出差异。

### 5.5 校准数据集

- 数据源：NVIDIA PhysicalAI-Autonomous-Vehicles
- 样本数：2000-3000 条
- 场景覆盖：直行(300)、左转/右转(各200)、变道(150)、停车/起步(150)、避障(300)、复杂环境(100)

### 5.6 验证检查点

| 指标                | 通过标准                   |
| ----------------- | ---------------------- |
| 参数量               | ≤4B                    |
| INT8 vs FP16 轨迹偏差 | <5%                    |
| 推理延迟              | <80ms (Jetson Orin NX) |
| 显存占用              | <6GB                   |
| 控制频率              | ≥15Hz                  |

***

## 六、跨帧 KV 共享（v3 新增）

### 6.1 原理

驾驶场景连续帧变化微小。只计算首帧完整视觉 KV，后续帧复用：

```python
class DeltaKVCache:
    def __init__(self, similarity_threshold=0.95):
        self.cached_kv = None
        self.threshold = similarity_threshold

    def update(self, current_frame, vision_encoder):
        if self.cached_kv is None:
            self.cached_kv = vision_encoder(current_frame)
            return self.cached_kv

        sim = F.cosine_similarity(
            current_frame.flatten(1).float(),
            self.cached_frame.flatten(1).float(), dim=1
        ).mean()

        if sim > self.threshold:
            # 高相似度，复用缓存
            return self.cached_kv
        else:
            # 场景切换，重新计算
            self.cached_kv = vision_encoder(current_frame)
            return self.cached_kv
```

### 6.2 预期收益

| 场景       | 帧间相似度 | 复用率   | 计算节省      |
| -------- | ----- | ----- | --------- |
| 高速公路直行   | >0.98 | \~95% | \~90%     |
| 城市道路     | >0.90 | \~80% | \~70%     |
| 急转弯      | <0.85 | \~40% | \~30%     |
| **加权平均** | —     | \~75% | **\~63%** |

***

## 七、部署方案

### 7.1 TensorRT 编译

```bash
# 分模块导出 ONNX
python export_onnx.py --module vision_encoder
python export_onnx.py --module language_model
python export_onnx.py --module diffusion_expert

# TensorRT 编译（Jetson Orin NX）
trtexec --onnx=vision_encoder.onnx --saveEngine=vision_encoder.engine \
        --int8 --fp16 --workspace=4096

trtexec --onnx=language_model.onnx --saveEngine=language_model.engine \
        --int8 --fp16 --workspace=2048

trtexec --onnx=diffusion_expert.onnx --saveEngine=diffusion_expert.engine \
        --fp16 --workspace=2048
```

### 7.2 Jetson 平台优化

| 优化项      | 措施                  | 预期收益            |
| -------- | ------------------- | --------------- |
| 视觉预处理    | NVDEC 硬件解码器         | CPU 占用 -50%     |
| 内存传输     | GPU 零拷贝             | 省去 CPU-GPU 拷贝延迟 |
| 跨帧 KV 共享 | Delta-KV 缓存         | 视觉计算 -63%       |
| 功耗       | MAXN 模式             | 解锁全部算力          |
| 散热       | 外接风扇                | 防止降频            |
| 确定性推理    | cudnn.deterministic | 控制稳定性           |

### 7.3 ROS2 集成

```
摄像头 → 预处理 → 模型推理 → 轨迹发布 → 控制器
(NVDEC)  (TensorRT) (Orin NX)  (ROS2 Topic) (PWM信号)
              ↑
        Delta-KV 缓存
```

***

## 八、显存预算（最终目标 4B INT8）

| 组件           | 大小           | 精度   | 说明               |
| ------------ | ------------ | ---- | ---------------- |
| 语言模型主干       | \~2.5 GB     | INT8 | 4B 参数量化后         |
| 视觉编码器        | \~0.5 GB     | INT8 | 鲁棒性强             |
| 扩散专家         | \~1.5 GB     | FP16 | 精度敏感             |
| Token 评分     | \~0 GB       | —    | 无参数              |
| KV Cache     | \~0.2 GB     | FP16 | 剪枝+Delta-KV 双重减少 |
| CUDA context | \~1.5 GB     | —    | TensorRT 开销      |
| **合计**       | **\~6.2 GB** | —    | **16GB 平台绑绑有余**  |

***

## 九、预期性能

| 指标       | 10B FP16 (原始)  | 4B INT8 (压缩后)       |
| -------- | -------------- | ------------------- |
| 模型大小     | 22 GB          | \~4.5 GB            |
| 显存占用     | 24 GB          | \~6-8 GB            |
| 推理延迟     | \~200ms (A100) | \~50-80ms (Orin NX) |
| 控制频率     | \~5 Hz         | \~12-20 Hz          |
| 轨迹精度     | 基准             | 退化约 5-10%           |
| 视觉 Token | \~5000         | \~800-1500          |

***

## 十、风险与应对

| 风险                 | 严重度   | 应对措施                   |
| ------------------ | ----- | ---------------------- |
| Flow Matching 蒸馏失败 | **高** | 分三阶段：轨迹级→步级→步数压缩       |
| 量化精度雪崩             | 中     | 分模块量化，扩散专家保持 FP16      |
| LightVLA 适配困难      | 中     | 核心算法可复用，主要改视觉 token 提取 |
| 训练资源不足             | 中     | ModelOpt + LoRA 降低显存需求 |
| 确定性推理性能损失          | 低     | 3-5% 损失，但控制稳定性必须保证     |

***

## 十一、时间线（含学习阶段）

### 第 1-2 个月：基础学习 + 最小可用版本

| 周次      | 工作内容                                   | 里程碑               |
| ------- | -------------------------------------- | ----------------- |
| 第 1-2 周 | PyTorch 速成（tensor、autograd、DataLoader） | 能跑通基础训练脚本         |
| 第 3-4 周 | HuggingFace 模型加载（from\_pretrained）     | 能加载 Alpamayo 模型   |
| 第 5-6 周 | 用 Claude Code 写推理 pipeline             | 在 CARLA 中看到模型输出轨迹 |
| 第 7-8 周 | ModelOpt 跟着 tutorial 走                 | 理解蒸馏流程            |

**里程碑**：仿真环境中跑通 Alpamayo 原始模型推理

### 第 3-5 个月：蒸馏 + Token 剪枝

| 周次        | 工作内容                 | 里程碑        |
| --------- | -------------------- | ---------- |
| 第 9-12 周  | ModelOpt 蒸馏 10B→6B   | 6B 学生模型可用  |
| 第 13-16 周 | LightVLA 适配 Alpamayo | Token 剪枝生效 |
| 第 17-20 周 | 6B→4B 继续蒸馏           | 4B 模型初步可用  |

**里程碑**：4B 模型 + Token 剪枝，推理延迟 <100ms

### 第 6-8 个月：量化 + TensorRT 部署

| 周次        | 工作内容                  | 里程碑          |
| --------- | --------------------- | ------------ |
| 第 21-24 周 | INT8 量化 + 校准          | 4B INT8 模型   |
| 第 25-28 周 | ONNX 导出 + TensorRT 编译 | Jetson 上跑通推理 |
| 第 29-32 周 | 跨帧 KV 共享 + 确定性推理      | 延迟 <80ms     |

**里程碑**：4B INT8 模型在 Jetson Orin NX 上实时推理

### 第 9-10 个月：实车集成

| 周次        | 工作内容            | 里程碑       |
| --------- | --------------- | --------- |
| 第 33-36 周 | 硬件组装 + PID 调参   | 小车能手动控制   |
| 第 37-40 周 | ROS2 集成 + 端到端测试 | 小车按模型输出行驶 |

**里程碑**：小车在封闭场地完成直行 + 转弯

### 第 11-12 个月：优化 + 收尾

| 周次        | 工作内容         | 里程碑    |
| --------- | ------------ | ------ |
| 第 41-44 周 | 精度/延迟调优      | 指标达标   |
| 第 45-48 周 | 写报告、录视频、准备答辩 | 全部材料提交 |

**总计约 48 周（12 个月）**，与研发计划书完全对齐。

***

## 十二、与研发计划书的对齐

| 研发计划书目标     | 本方案对应        | 状态     |
| ----------- | ------------ | ------ |
| 参数量 1-4B    | 4B（渐进达成）     | ✅      |
| 推理延迟 <200ms | <80ms        | ✅ 超额完成 |
| 显存占用 <6GB   | \~6.2GB      | ✅ 基本吻合 |
| 决策准确率 ≥85%  | ≥90%         | ✅      |
| 模型压缩 ≥50%   | 60%          | ✅      |
| 蒸馏精度 ≥90%   | 轨迹 L2 <110%  | ✅      |
| TensorRT 引擎 | 分模块编译        | ✅      |
| Jetson 部署   | Orin NX 16GB | ✅      |

***

## 十三、总结

本方案的核心优势：

1. **渐进式压缩**：每步可验证，失败可回退，不会到最后才发现问题
2. **工具链成熟**：ModelOpt 处理蒸馏（官方工具），LightVLA 处理剪枝（开源代码）
3. **Claude Code 辅助开发**：团队有半年使用经验，代码实现效率高
4. **与项目目标对齐**：完全覆盖研发计划书的所有技术指标
5. **风险可控**：任何阶段的成功都是一个可用的成果
6. **最终目标 \~6.2GB 显存**：16GB Jetson 平台绑绑有余

> 先用 ModelOpt 跑通蒸馏 pipeline，建立信心；再用 LightVLA 做精细的 token 剪枝；最后 INT8 量化收尾。每一步都有明确的验证标准，不冒进。
>
> 团队以"先跑通代码 → 遇到问题学理论 → 理解后验证"的方式推进，用 Claude Code 加速实现，用实验验证质量。

