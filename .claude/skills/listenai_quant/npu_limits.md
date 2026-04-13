# LISTENAI NPU 算子支持与限制说明

## 概述

LISTENAI AIOT 芯片的 NPU（神经网络处理器）并非支持所有 PyTorch/ONNX 算子，本文档说明支持的算子和常见的限制，帮助您在模型设计阶段规避不兼容的算子。

---

## 支持的量化算子

linger 支持以下量化算子的训练和导出：

### 基础算子

| PyTorch 算子 | linger 算子 | ONNX 导出 | 说明 |
|--------------|-------------|-----------|------|
| nn.Conv2d | QConv2d | Conv2dInt | 卷积（支持分组、可分离） |
| nn.ConvTranspose2d | QConvTranspose2d | ConvTranspose2dInt | 反卷积 |
| nn.Linear | QLinear | LinearInt | 全连接层 |
| nn.BatchNorm2d | QBatchNorm2d | BatchNorm2dInt | 批归一化 |
| nn.LayerNorm | QLayerNorm | LayerNorm | 层归一化 |

### 池化算子

| PyTorch 算子 | linger 算子 | ONNX 导出 |
|--------------|-------------|-----------|
| nn.AvgPool2d | QAvgPool2d | AvgPool2dInt |
| nn.MaxPool2d | QMaxPool2d | MaxPool2dInt |

### 激活函数

| PyTorch 算子 | linger 算子 | ONNX 导出 |
|--------------|-------------|-----------|
| nn.ReLU | Relu | Relu |
| torch.sigmoid | QSigmoid | QSigmoid |
| torch.tanh | QTanh | QTanh |
| torch.softmax | QSoftmax | QSoftmax |

### 张量操作

| PyTorch 算子 | linger 算子 | ONNX 导出 |
|--------------|-------------|-----------|
| torch.add / + | QAdd | iqAdd |
| torch.mul / * | QMul | iqMul |
| torch.cat | QCat | iqCat |
| torch.bmm | QBmm | BmmInt |
| torch.clamp | Clamp | Clip |
| torch.transpose | Transpose | Transpose |
| view / reshape | view/reshape | Reshape |
| squeeze | squeeze | Squeeze |
| unsqueeze | unsqueeze | Unsqueeze |
| flatten | flatten | Flatten |
| slice | slice | Slice |

### 循环网络

| PyTorch 算子 | linger 算子 | ONNX 导出 |
|--------------|-------------|-----------|
| nn.LSTM | QLSTM | LSTMInt |
| nn.GRU | QGRU | GRUInt |

### 其他

| PyTorch 算子 | linger 算子 | ONNX 导出 |
|--------------|-------------|-----------|
| nn.Embedding | QEmbedding | QEmbedding |

---

## NPU 不支持的算子（需要规避）

### 1. 浮点专用算子

以下算子 **不支持量化训练**，只能在浮点模型中使用：

| 算子 | 说明 | 替代方案 |
|------|------|---------|
| nn.Dropout | Dropout 层 | 训练时移除，推理时手动关闭 |
| nn.BatchNorm1d | 1D 批归一化 | 使用 2D 或替换为 LayerNorm |
| nn.InstanceNorm | 实例归一化 | 替换为 BatchNorm 或 LayerNorm |
| nn.GroupNorm | 组归一化 | 替换为 BatchNorm |
| nn.LocalResponseNorm | 局部响应归一化 | 移除或替换 |
| F.interpolate (插值) | 双线性/双三次插值 | 使用 pool + reshape 替代 |
| F.grid_sample | 网格采样 | 移除（常用数据增强） |
| F.affine_grid | 仿射网格 | 移除 |
| torch.nonzero | 非零索引 | 移除 |
| torch.where | 条件选择 | 用 add/mul 替代 |
| torch.gather | 索引 gather | 移除或重写 |
| torch.scatter | 索引 scatter | 移除 |

### 2. 复杂张量操作

| 算子 | 说明 | 替代方案 |
|------|------|---------|
| torch.einsum | 爱因斯坦求和 | 分解为 matmul + broadcast |
| torch.norm | 张量范数 | 分解为 abs + sum + sqrt |
| torch.var / std | 方差/标准差 | 预先计算并存储 |
| torch.cumsum / cumprod | 累积操作 | 循环展开（如果可以） |
| torch.diff | 差分操作 | 重写 |
| torch.sort / topk | 排序 | 移除或预处理 |
| torch.matmul (3D+) | 矩阵乘法 | 只使用 2D 的 @ 或 mm |
| torch.cholesky | Cholesky 分解 | 移除 |
| torch.linalg.* | 线性代数操作 | 移除 |

### 3. 不支持的 ONNX 算子

导出的 ONNX 模型中，以下算子 **无法被 thinker 运行时支持**：

| ONNX 算子 | 说明 |
|-----------|------|
| Loop | 动态循环 |
| Scan | 扫描操作 |
| If (条件) | 动态条件分支 |
| NonMaxSuppression | NMS（目标检测后处理需在 CPU 完成） |
| Resize | 插值操作 |
| ScatterElements | 散射操作 |
| GatherElements | 收集操作 |
| Expand | 扩展操作 |
| Tile | 瓦片操作 |
| ConstantOfShape | 常量形状 |

---

## 模型设计规范

### 允许的网络结构

1. **卷积网络**：Conv2d → BatchNorm → Activation → Pooling
2. **残差连接**：支持 add 操作的残差块
3. **特征金字塔**：多尺度特征融合（使用 cat/add）
4. **SE/SK 注意力**：通道注意力（需简化）
5. **深度可分离卷积**：MobileNet 风格

### 不建议的结构

1. **复杂注意力机制**：如 Self-Attention、Non-Local、CBAM（除非简化）
2. **多分支复杂结构**：Inception 风格的分叉结构
3. **动态图控制流**：if/else 取决于输入数据的分支
4. **过深的网络**：层级过多影响量化精度

### 设计建议

1. **优先使用 2D 操作**：BatchNorm2d、Conv2d、MaxPool2d
2. **激活函数用 ReLU**：最安全的量化友好激活
3. **避免 float 输出**：确保最后一层输出是量化后的
4. **限制权重范围**：使用约束训练将权重限制在合理范围
5. **预训练浮点模型**：先训练浮点模型，再做量化微调

---

## 常见问题与解决方案

### Q1: 量化训练时 loss 不收敛

**原因**：使用了不兼容的算子

**解决**：
- 检查模型是否使用了 Dropout、Interpolate 等不支持的算子
- 确保所有算子都在支持列表内

### Q2: ONNX 导出失败

**原因**：使用了不支持的 ONNX 算子

**解决**：
- 移除 `torch.nonzero`、`torch.where` 等动态操作
- 将 `F.interpolate` 替换为固定尺寸的 Pooling

### Q3: 推理结果不正确

**原因**：量化精度损失过大

**解决**：
- 使用约束训练（constrain）先约束权重范围
- 调整 `clamp_activation_value` 参数
- 增加量化微调（QAT）训练轮数

---

## 配置文件参考

### 量化配置示例 (quant_config.yaml)

```yaml
# 约束配置
clamp_info:
  clamp_activation_value: 8    # 激活值约束范围 [-8, 8]
  clamp_bias_value: null
  clamp_factor_value: 7        # 权重约束因子

# 量化配置
quant_info:
  open_quant: true             # 启用量化
  platform: venusA             # 目标平台
  activate_bits: 8             # 激活值位宽
  weight_bits: 8               # 权重位宽
  is_perchannel: false          # 是否通道级量化
  is_symmetry: true            # 是否对称量化
  a_strategy: RANGE_MEAN       # 激活量化策略
  w_strategy: RANGE_MEAN       # 权重量化策略
```

---

## 相关文档

- [linger 支持的算子列表](../../thirdparty/linger/doc/tutorial/support_quant_ops.md)
- [thinker 算子支持列表](../../thirdparty/thinker/docs/support_quant_ops.md)
- [量化训练快速入门](../../thirdparty/linger/doc/tutorial/quant_quick_start.md)
- [LISTENAI 量化训练工具链](../listenai_quant/README.md)