# 二维码检测与识别算法

> 基于深度学习的轻量级二维码检测与识别系统

## 项目概述

**任务**: 输入摄像头拍摄的图片，检测并识别其中的二维码

**流水线**:
1. **二维码检测** (QRDetector) - 检测图像中的二维码区域
2. **角度预测** - 预测二维码旋转角度 (0°, 90°, 180°, 270°)
3. **内容识别** - 使用OpenCV解码二维码内容

## 性能指标

| 指标 | 目标 | 实际 |
|------|------|------|
| **参数量** | < 4M | 0.36M ✅ |
| **输入尺寸** | 320×320 | 320×320 |
| **检测精度** | 高 | 依赖训练数据 |

## 模型架构

```
输入图像 (3x320x320)
    │
    ▼
┌─────────────────────────────────────────┐
│  QRDetector (0.36M)                     │
│  ├── MobileNetV2 风格编码器              │
│  ├── 特征融合层                          │
│  ├── 概率图检测头 → probability map     │
│  ├── 角度分类头 → angle (0/90/180/270)  │
│  └── 边界框回归头 → bbox offset         │
└─────────────────────────────────────────┘
    │
    ├── 概率图 → 阈值化 → 轮廓提取 → 边界框
    │
    ▼
┌─────────────────────────────────────────┐
│  QRCodeDecoder                          │
│  └── OpenCV QRCodeDetector              │
│  └── 内容解码                           │
└─────────────────────────────────────────┘
    │
    ▼
输出: 二维码位置 + 角度 + 内容
```

## 目录结构

```
项目根目录/
├── tasks/qrcode_detect/           # 任务代码
│   ├── cfg/
│   │   ├── train_config.yaml      # 训练配置
│   │   ├── models/model.yaml      # 模型配置
│   │   └── datasets/dataset.yaml  # 数据集配置
│   ├── scripts/
│   │   ├── qrcode_models.py       # 模型定义
│   │   ├── download_dataset.py    # 数据集下载/生成
│   │   ├── train_qr_detector.py  # 训练脚本
│   │   └── inference_qr.py        # 推理脚本
│   └── docs/
│       └── README.md              # 文档
├── dataset/qrcode_detection/     # 数据集
├── models/qrcode_detect/           # 模型文件
└── runs/qrcode_detect/             # 训练输出
```

## 快速开始

### 1. 准备数据集

```bash
cd tasks/qrcode_detect/scripts
python download_dataset.py
```

### 2. 训练模型

```bash
python scripts/train_qr_detector.py \
  --data-dir dataset/qrcode_detection \
  --epochs 50 \
  --batch-size 8 \
  --output runs/qrcode_detect
```

### 3. 推理测试

#### 单图测试
```bash
python scripts/inference_qr.py \
  --model models/qrcode_detect/best_detector.pth \
  --input test.jpg \
  --output output/qrcode_detect/
```

#### 批量测试
```bash
python scripts/inference_qr.py \
  --model models/qrcode_detect/best_detector.pth \
  --input /path/to/images \
  --output output/qrcode_detect/
```

#### 摄像头实时检测
```bash
python scripts/inference_qr.py \
  --model models/qrcode_detect/best_detector.pth \
  --input camera
```

## 模型文件

训练完成后，模型保存在:
- `models/qrcode_detect/best_detector.pth` - 最佳检测器 (~0.36M)
- `runs/qrcode_detect/checkpoint_epoch_*.pth` - 定期检查点

## 技术细节

### 模型架构

**QRDetector (0.36M)**
- Backbone: MobileNetV2 风格深度可分离卷积
- 编码器: 4阶段特征提取 (16→16→32→64→96 通道)
- 检测头: DBNet 风格概率图
- 角度头: 全局池化 + 全连接 (4类分类)
- 边界框头: 偏移量回归

### 损失函数

- 检测损失: BCE + Dice Loss
- 角度损失: CrossEntropy Loss (0°, 90°, 180°, 270°)

### 优化器

- 类型: AdamW
- 学习率: 0.001
- 权重衰减: 1e-4
- 调度: CosineAnnealing

### 数据增强

- 随机水平翻转
- 随机亮度调整
- 随机对比度调整

## 数据集

数据集自动生成在: `dataset/qrcode_detection`

- 训练集: 4000 张图像
- 验证集: 500 张图像
- 测试集: 500 张图像

## 参数统计

```
QRDetector: 355,097 (0.36M)
Limit: 4,000,000
Status: ✅ PASS
```

## 后续优化方向

- [ ] 增加更多真实场景数据
- [ ] 模型量化 (INT8)
- [ ] 端到端联合训练
- [ ] 实时性能优化

---

**作者**: Listanai LNN Team
**时间**: 2026-04-12
**状态**: ✅ 完成