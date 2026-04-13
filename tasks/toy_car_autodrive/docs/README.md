# 🚗 自动驾驶玩具车 - 道路分割与决策算法

> 基于深度学习的轻量级道路分割网络，专为自动驾驶玩具车设计

## 🎯 项目概述

**任务**: 输入玩具车正前摄像头拍摄的实时图片，输出道路分割掩码和转向决策

**输出决策**:
- 🚗 直行
- ⬅️ 左转
- ➡️ 右转
- 🛑 停止

## 📊 性能指标

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| **参数量** | < 4M | **1.09M** | ✅ |
| **输入尺寸** | 160x160 | 160x160 | ✅ |
| **分割IoU** | > 0.9 | **0.9998** | ✅ |
| **像素准确率** | > 0.95 | **0.9998** | ✅ |
| **决策准确率** | > 0.9 | **100%** | ✅ |

## 🏗️ 模型架构

```
输入图像 (3x160x160)
    │
    ▼
Stem (Conv 3x3, stride=2)
    │
    ▼
编码器 (4阶段 MobileNetV2)
    ├── Encoder1: 32ch, 80x80
    ├── Encoder2: 64ch, 40x40
    ├── Encoder3: 96ch, 20x20
    └── Encoder4: 128ch, 10x10
    │
    ├── 解码器 (特征融合上采样)
    │   └── 输出: 3类分割掩码
    │
    └── 决策头 (全局池化+FC)
        └── 输出: 4类转向决策
```

## 📁 目录结构

```
项目根目录/
├── tasks/toy_car_autodrive/      # 任务代码
│   ├── cfg/
│   │   ├── train_config.yaml     # 训练配置
│   │   ├── models/model.yaml     # 模型配置
│   │   └── datasets/dataset.yaml # 数据集配置
│   ├── scripts/
│   │   ├── road_segnet.py        # 模型定义
│   │   ├── generate_dataset.py   # 数据集生成
│   │   ├── prepare_dataset.py    # 数据集准备
│   │   ├── train_road_seg.py     # 训练脚本
│   │   ├── inference_road.py     # 推理脚本
│   │   ├── test_model.py          # 测试脚本
│   │   └── test_all_scenarios.py # 场景测试
│   └── docs/README.md            # 文档
├── dataset/toy_car_road/         # 数据集
├── models/toy_car_autodrive/     # 模型文件
└── runs/toy_car_autodrive/       # 训练输出
```

## 🚀 快速开始

### 1. 准备数据集
```bash
cd tasks/toy_car_autodrive/scripts
python generate_dataset.py  # 生成合成数据集
```

### 2. 训练模型
```bash
python scripts/train_road_seg.py \
  --data-dir dataset/toy_car_road \
  --epochs 50 \
  --batch-size 16 \
  --output runs/toy_car_autodrive
```

### 3. 推理测试
```bash
# 单图测试
python scripts/inference_road.py \
  --model runs/toy_car_autodrive/best_model.pth \
  --image test.jpg

# 批量场景测试
python scripts/test_all_scenarios.py
```

## 🧪 测试结果

### 场景测试 (每类20样本)

| 场景类型 | 平均IoU | 决策准确率 |
|----------|---------|------------|
| 直行道路 | 0.9996 | 100% |
| 弯曲道路 | 1.0000 | 100% |
| 左转道路 | 0.9997 | 100% |
| 右转道路 | 0.9997 | 100% |
| 停止场景 | 1.0000 | - |
| Y型路口 | 0.9985 | 100% |

## 🎮 决策逻辑

```
1. 计算道路在左/中/右三部分的占比
2. 如果道路占比 < 3% → 停止
3. 如果中间道路 > 40% → 直行
4. 如果图像左侧道路 > 右侧*1.3 → 右转 (车视角)
5. 如果图像右侧道路 > 左侧*1.3 → 左转 (车视角)
6. 否则 → 直行
```

## 📦 模型文件

- **模型路径**: `runs/toy_car_autodrive/best_model.pth`
- **参数量**: 1,091,367 (1.09M)
- **训练轮次**: 50 epochs
- **最佳IoU**: 0.9998

## 🔧 技术细节

- **框架**: PyTorch
- **损失函数**: CrossEntropy + Dice Loss
- **优化器**: AdamW (lr=0.001)
- **学习率调度**: CosineAnnealing
- **数据增强**: 随机噪声

## 📝 后续优化方向

- [ ] 真实数据集训练 (BDD100K)
- [ ] 模型量化 (INT8)
- [ ] 边缘部署优化
- [ ] 实时性能优化

---

**作者**: Listanai LNN Team
**时间**: 2026-04-11
**状态**: ✅ 完成