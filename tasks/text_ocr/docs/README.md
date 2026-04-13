# 📝 文本检测、分割与OCR识别算法

> 基于深度学习的轻量级文本识别流水线，支持中英文识别

## 🎯 项目概述

**任务**: 输入自然场景图像，检测并识别其中的中文或英文字符串

**三阶段流水线**:
1. **文本检测** (Text Detection) - 检测图像中的文本区域
2. **文本分割** (Text Segmentation) - 精细分割文本区域
3. **OCR识别** (OCR Recognition) - 识别文本内容

## 📊 性能指标

| 阶段 | 模型 | 参数量 | 限制 | 状态 |
|------|------|--------|------|------|
| 检测 | TextDetector | ~3.2M | < 4M | ✅ |
| 分割 | TextSegmenter | ~2.8M | < 4M | ✅ |
| 识别 | CRNN | ~3.5M | < 4M | ✅ |
| **总计** | 完整流水线 | **~9.5M** | - | ✅ |

| 指标 | 目标 | 实际 |
|------|------|------|
| **输入尺寸** | ≤ 640×640 | 640×640 |
| **输入格式** | RGB | RGB |
| **支持语言** | 中文+英文 | 中文+英文 |

## 🏗️ 模型架构

```
输入图像 (3x640x640)
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1: 文本检测 (TextDetector)       │
│  ├── MobileNetV2 编码器                 │
│  ├── 特征金字塔 (FPN)                   ���
│  └── DB检测头 → probability/threshold  │
│  参数量: 3.2M                           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 2: 文本分割 (TextSegmenter)     │
│  ├── MobileNetV2 编码器                 │
│  ├── 特征融合模块                       │
│  └── 解码器 → seg_mask + edge_mask     │
│  参数量: 2.8M                           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 3: OCR识别 (CRNN)               │
│  ├── VGG风格CNN特征提取                 │
│  ├── BiGRU序列建模                      │
│  └── CTC解码器                          │
│  参数量: 3.5M                           │
└─────────────────────────────────────────┘
    │
    ▼
输出: 识别文本 + 文本位置 + 置信度
```

## 📁 目录结构

```
项目根目录/
├── tasks/text_ocr/                # 任务代码
│   ├── cfg/
│   │   ├── train_config.yaml      # 训练配置
│   │   ├── models/model.yaml      # 模型配置
│   │   └── datasets/dataset.yaml  # 数据集配置
│   ├── scripts/
│   │   ├── text_models.py         # 模型定义
│   │   ├── download_dataset.py   # 数据集下载
│   │   ├── train_text_detector.py  # 检测器训练
│   │   ├── train_text_segmenter.py # 分割器训练
│   │   ├── train_ocr.py           # OCR训练
│   │   └── inference_text.py       # 推理脚本
│   └── docs/
│       └── README.md               # 文档
├── dataset/text_detection/         # 数据集
├── models/text_ocr/               # 模型文件
└── runs/text_ocr/                 # 训练输出
    ├── detector/                   # 检测器模型
    ├── segmenter/                  # 分割器模型
    └── ocr/                        # OCR模型
```

## 🚀 快速开始

### 1. 准备数据集
```bash
cd tasks/text_ocr/scripts
python download_dataset.py
```

### 2. 训练模型

#### 训练文本检测器
```bash
python scripts/train_text_detector.py \
  --data-dir dataset/text_detection \
  --epochs 50 \
  --batch-size 8 \
  --output-dir runs/text_ocr/detector
```

#### 训练文本分割器
```bash
python scripts/train_text_segmenter.py \
  --data-dir dataset/text_detection \
  --epochs 50 \
  --batch-size 8 \
  --output-dir runs/text_ocr/segmenter
```

#### 训练OCR
```bash
python scripts/train_ocr.py \
  --data-dir dataset/text_detection \
  --epochs 50 \
  --batch-size 16 \
  --output-dir runs/text_ocr/ocr
```

### 3. 推理测试

#### 单图测试
```bash
python scripts/inference_text.py \
  --input test.jpg \
  --detector runs/text_ocr/detector/best_detector.pth \
  --segmenter runs/text_ocr/segmenter/best_segmenter.pth \
  --ocr runs/text_ocr/ocr/best_ocr.pth \
  --output output/text_ocr/
```

#### 批量测试
```bash
python scripts/inference_text.py \
  --input /path/to/images \
  --detector runs/text_ocr/detector/best_detector.pth \
  --segmenter runs/text_ocr/segmenter/best_segmenter.pth \
  --ocr runs/text_ocr/ocr/best_ocr.pth \
  --output output/text_ocr/
```

## 📦 模型文件

训练完成后，模型保存在:
- `runs/text_ocr/detector/best_detector.pth` - 文本检测器 (~3.2M)
- `runs/text_ocr/segmenter/best_segmenter.pth` - 文本分割器 (~2.8M)
- `runs/text_ocr/ocr/best_ocr.pth` - OCR识别器 (~3.5M)
- `runs/text_ocr/ocr/char_map.json` - 字符映射表

## 🔧 技术细节

### 模型架构

**TextDetector (3.2M)**
- Backbone: MobileNetV2 风格深度可分离卷积
- 检测头: DBNet 风格二值化检测
- 输出: 概率图 + 阈值图 + 二值图

**TextSegmenter (2.8M)**
- Backbone: 4阶段编码器
- 特征融合: 多尺度特征融合
- 输出: 分割掩码 + 边缘掩码

**CRNN (3.5M)**
- CNN: VGG风格5层卷积
- RNN: 双层BiGRU
- 解码: CTC Loss

### 损失函数

- 检测器: BCE + Dice Loss
- 分割器: BCE + Edge Loss
- OCR: CTC Loss

### 优化器

- 类型: AdamW
- 学习率: 0.001
- 权重衰减: 1e-4
- 调度: CosineAnnealing

## 📝 后续优化方向

- [ ] 模型量化 (INT8)
- [ ] 知识蒸馏
- [ ] 端到端联合训练
- [ ] 更多语言支持
- [ ] 实时性能优化

---

**作者**: Listanai LNN Team
**时间**: 2026-04-12
**状态**: ✅ 完成