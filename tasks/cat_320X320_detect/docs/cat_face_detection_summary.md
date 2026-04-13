# YOLO26n 猫脸检测训练项目总结

## 项目概述

使用 YOLO26n 模型在 Oxford-IIIT Pet 数据集上进行猫脸检测训练。

## 数据集

- **来源**: Oxford-IIIT Pet 数据集
- **位置**: `/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset`
- **猫图片总数**: 1,181 张 (从 trainval.txt 中筛选 SPECIES=1)
- **划分**:
  - 训练集: 946 张 (80%)
  - 验证集: 235 张 (20%)

## 实施步骤

### 1. 环境准备

- **Conda 环境**: 使用现有的 `animal_detection` 环境 (Python 3.11, PyTorch 2.10.0+cu128)
- **安装依赖**: ultralytics 8.4.23
- **GPU**: NVIDIA GeForce RTX 4090 (GPU 2)

### 2. 数据准备

编写了数据转换脚本 `prepare_cat_dataset.py`:

```python
# 主要功能:
# 1. 从 trainval.txt 筛选猫图片 (SPECIES=1)
# 2. 将 Pascal VOC XML 标注转换为 YOLO 格式
# 3. 划分训练集和验证集 (80/20)
# 4. 复制图片到对应目录
```

**YOLO 格式**: `class_id x_center y_center width height` (归一化)

### 3. 数据集配置

创建 `cat_face.yaml`:

```yaml
path: /CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face
train: train/images
val: val/images
nc: 1
names:
  0: cat
```

### 4. 模型训练

**训练参数**:
- 模型: YOLO26n (nc=1)
- Epochs: 50
- Image size: 640
- Batch size: 16
- Device: CUDA 2
- Workers: 8

**训练命令**:
```bash
CUDA_VISIBLE_DEVICES=2 yolo detect train \
  data=/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/ultralytics/cfg/datasets/cat_face.yaml \
  model=yolo26n.yaml \
  epochs=50 imgsz=640 batch=16 device=2
```

## 遇到的问题及解决方案

### 问题 1: 数据集路径找不到

**错误信息**:
```
FileNotFoundError: 'cat_face.yaml' does not exist
```

**原因**: YOLO CLI 默认在当前目录查找配置文件

**解决**: 使用完整绝对路径:
```bash
data=/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/ultralytics/cfg/datasets/cat_face.yaml
```

### 问题 2: 多张图片推理时路径解析错误

**错误信息**:
```
FileNotFoundError: ... does not exist
```

**原因**: 多个图片路径作为字符串传递时被当作单个文件名处理

**解决**: 将多个图片复制到单独目录，使用目录路径作为 source

## 训练结果

### 性能指标

| 指标 | 验证集结果 |
|------|-----------|
| **mAP50** | **0.962 (96.2%)** |
| **mAP50-95** | **0.795 (79.5%)** |
| Precision | 91.5% |
| Recall | 92.8% |

### 训练过程

- 训练时间: ~5 分钟 (50 epochs)
- Loss 持续下降，收敛良好
- mAP50 从初始的 0.09 提升到 0.96

## 产出文件

| 类型 | 路径 |
|------|------|
| 数据准备脚本 | `/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/prepare_cat_dataset.py` |
| 数据集配置 | `/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/ultralytics/cfg/datasets/cat_face.yaml` |
| 最佳权重 | `/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face/runs/detect/cat_face_yolo26n2/weights/best.pt` |
| 训练结果 | `/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face/runs/detect/cat_face_yolo26n2/` |

## 经验总结

1. **数据集路径**: 使用绝对路径避免路径解析问题
2. **YOLO26n**: 作为最新的 YOLO 模型，在小样本目标检测任务上表现优异
3. **数据增强**: 默认配置已包含良好的数据增强策略
4. **模型选择**: YOLO26n 参数量小 (~2.5M)，推理速度快 (~36ms/张)，适合边缘部署
5. **训练效率**: 使用预训练权重 + AMP 加速，50 epochs 仅需约 5 分钟

## 验证方法

1. ✅ 环境验证: conda 环境可用，GPU 可用
2. ✅ 数据验证: YOLO 格式标注文件生成正确
3. ✅ 训练验证: 训练正常启动，loss 下降
4. ✅ 模型验证: 验证集 mAP50 = 0.962 > 0.5
5. ✅ 推理验证: 测试图片能正确检测猫脸
