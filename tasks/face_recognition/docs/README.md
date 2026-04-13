# 人脸检测与识别算法

> 基于深度学习的轻量级人脸识别系统，支持人脸检测、特征提取和人脸锁功能

## 项目概述

**任务**: 输入摄像头拍摄的图片，检测并识别其中的所有人脸，实现人脸注册和人脸验证功能

**三阶段流水线**:
1. **人脸检测** (Face Detection) - 检测图像中的人脸位置
2. **人脸对齐** (Face Alignment) - 框选抠图并进行对齐
3. **人脸识别** (Face Recognition) - 提取特征向量并进行匹配

## 性能指标

| 阶段 | 模型 | 参数量 | 限制 | 状态 |
|------|------|--------|------|------|
| 检测 | FaceDetector | ~1.4M | < 4M | ✅ |
| 识别 | FaceEmbedder | ~2.0M | < 4M | ✅ |
| **总计** | 完整流水线 | **~3.4M** | ≤ 4M | ✅ |

| 指标 | 目标 | 实际 |
|------|------|------|
| **输入尺寸** | 检测: 224×224, 识别: 112×112 | ✓ |
| **输入格式** | RGB | RGB |
| **特征维度** | 128维 | 128维 |
| **匹配方式** | 余弦相似度 | ✓ |

## 模型架构

```
输入图像 (3x224x224)
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1: 人脸检测 (FaceDetector)        │
│  ├── MobileNetV2Lite 编码器             │
│  ├── FPN 特征金字塔                     │
│  └── Anchor-free 检测头                │
│  参数量: 1.4M                           │
└─────────────────────────────────────────┘
    │
    ▼ [检测到的人脸区域]
┌─────────────────────────────────────────┐
│  Stage 2: 人脸对齐与抠图               │
│  ├── 边界框扩展 (margin)                │
│  ├── 人脸对齐校正                       │
│  └── 尺寸归一化 (112×112)               │
└─────────────────────────────────────────┘
    │
    ▼ [人脸裁剪]
┌─────────────────────────────────────────┐
│  Stage 3: 特征提取 (FaceEmbedder)      │
│  ├── MobileNetV2Lite 编码器             │
│  ├── 全局平均池化                       │
│  └── ArcFace 特征嵌入                  │
│  参数量: 2.0M                           │
│  输出: 128维特征向量                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 4: 人脸匹配 (Face Matching)      │
│  ├── 余弦相似度计算                    │
│  └── Top-K 检索                        │
└─────────────────────────────────────────┘
    │
    ▼
输出: 识别结果 + 置信度 + 注册ID
```

## 目录结构

```
项目根目录/
├── tasks/face_recognition/         # 任务代码
│   ├── cfg/
│   │   ├── train_config.yaml      # 训练配置
│   │   ├── models/model.yaml      # 模型配置
│   │   └── datasets/dataset.yaml  # 数据集配置
│   ├── scripts/
│   │   ├── face_models.py         # 模型定义
│   │   ├── download_dataset.py   # 数据集下载
│   │   ├── train_face_detector.py    # 检测器训练
│   │   ├── train_face_embedder.py   # 特征提取器训练
│   │   └── inference_face.py     # 推理和注册脚本
│   └── docs/
│       └── README.md              # 文档
├── dataset/face_recognition/       # 数据集
├── models/face_recognition/        # 模型文件
└── runs/face_recognition/         # 训练输出
    ├── detector/                  # 检测器模型
    └── embedder/                  # 特征提取器模型
```

## 快速开始

### 1. 准备数据集

```bash
cd tasks/face_recognition/scripts
python download_dataset.py --data-dir dataset/face_recognition
```

### 2. 训练模型

#### 训练人脸检测器
```bash
python scripts/train_face_detector.py \
  --train-dir dataset/face_recognition/WIDERFace/synthetic/train \
  --val-dir dataset/face_recognition/WIDERFace/synthetic/val \
  --epochs 80 \
  --batch-size 16 \
  --output runs/face_recognition/detector
```

#### 训练人脸特征提取器
```bash
python scripts/train_face_embedder.py \
  --train-dir dataset/face_recognition/LFW/synthetic/train \
  --epochs 80 \
  --batch-size 32 \
  --num-classes 500 \
  --output runs/face_recognition/embedder
```

### 3. 推理测试

#### 人脸检测
```bash
python scripts/inference_face.py \
  --mode detect \
  --input test_image.jpg \
  --detector runs/face_recognition/detector/best_detector.pth \
  --output output/face_recognition/
```

#### 人脸注册
```bash
python scripts/inference_face.py \
  --mode register \
  --input person_photo.jpg \
  --name "张三" \
  --detector runs/face_recognition/detector/best_detector.pth \
  --embedder runs/face_recognition/embedder/best_embedder.pth \
  --database face_database.pkl \
  --output output/face_recognition/
```

#### 人脸验证
```bash
python scripts/inference_face.py \
  --mode verify \
  --input verify_photo.jpg \
  --threshold 0.5 \
  --detector runs/face_recognition/detector/best_detector.pth \
  --embedder runs/face_recognition/embedder/best_embedder.pth \
  --database face_database.pkl \
  --output output/face_recognition/
```

## 使用示例

### Python API

```python
import sys
sys.path.append('tasks/face_recognition/scripts')

from inference_face import FaceRecognitionSystem

# 初始化系统
face_system = FaceRecognitionSystem(
    detector_path='runs/face_recognition/detector/best_detector.pth',
    embedder_path='runs/face_recognition/embedder/best_embedder.pth',
    database_path='face_database.pkl',
    device='cpu'
)

# 注册人脸
face_id, result_img = face_system.register_face(img, "用户名称")

# 验证人脸
result, embedding, result_img = face_system.verify_face(img, threshold=0.5)
if result:
    metadata, similarity = result
    print(f"识别成功: {metadata['name']}, 相似度: {similarity:.4f}")
else:
    print("未识别")
```

## 技术细节

### 模型架构

**FaceDetector (1.4M)**
- Backbone: MobileNetV2Lite 轻量级编码器
- 特征融合: FPN 特征金字塔
- 检测头: Anchor-free 检测方式
- 输出: 人脸边界框 + 置信度

**FaceEmbedder (2.0M)**
- Backbone: MobileNetV2Lite 轻量级编码器
- 特征嵌入: 128维向量
- 损失函数: ArcFace (Angular Margin Loss)

### 核心技术

- **Anchor-free 检测**: 简化检测头，减少参数量
- **ArcFace Loss**: 增加角度间隔，提高识别精度
- **余弦相似度**: 高效的人脸匹配方式
- **NMS 后处理**: 去除重复检测框

## 后续优化方向

- [ ] 模型量化 (INT8/FP16)
- [ ] 知识蒸馏压缩
- [ ] 实时性能优化 (支持视频流)
- [ ] 活体检测防欺骗
- [ ] 遮挡鲁棒性增强

---

**作者**: Listanai LNN Team
**时间**: 2026-04-12
**状态**: ✅ 完成